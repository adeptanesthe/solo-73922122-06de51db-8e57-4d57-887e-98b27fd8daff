"""服务端偏好集存储与 DPO 训练任务管理。

并发与废止语义（需求硬性约定）：
  - 偏好集保存在服务端，带**单调递增版本号**与内容指纹；任何一次保存编辑
    （即使内容改回原值）都会让版本号 +1，正在进行的训练任务因此被永久废止，
    废止后不会因改回内容而恢复；
  - 同一时刻最多一个训练任务；训练在后台线程执行，不阻塞两个网页窗口中的
    原解码对照与偏好集编辑；
  - 训练线程只读取原目标模型（仅做深拷贝，不调用其前向、不改其权重/KV/计数）；
  - 训练成功且交叉校验通过才产出可展示结果；失败或被废止时，完整保留**上次
    成功结果**，并用版本号、指纹与输入摘要标明它属于哪一次输入。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .dpo import (DPOResult, PreferencePair, fingerprint, run_dpo,
                  validate_pairs)
from .model import CausalTransformer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# 首次打开页面即可试跑的演示偏好集（均满足 1～32 个可打印 ASCII 字符）
_DEFAULT_RAW = [
    {"prompt": "the capital of france is ",
     "chosen": "paris", "rejected": "london"},
    {"prompt": "one plus one equals ",
     "chosen": "two", "rejected": "banana"},
]


@dataclass
class RunRecord:
    """一次训练运行的终态记录（成功/失败/废止共用）。"""
    job_id: str
    state: str                      # "success" | "failed" | "invalidated"
    version: int
    fingerprint: str
    n_pairs: int
    started_at: str
    finished_at: str
    input_summary: list[dict]
    error: str | None
    result: DPOResult | None        # 仅成功且对照通过时非空


class PreferenceStore:
    """带版本号的偏好集；所有读写在锁内进行，返回不可变样本对象。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._version = 1
        self._pairs: list[PreferencePair] = validate_pairs(_DEFAULT_RAW)
        self._updated_at = _utc_now()

    def snapshot(self) -> tuple[int, list[PreferencePair]]:
        with self._lock:
            # PreferencePair 为 frozen dataclass，直接返回即可
            return self._version, list(self._pairs)

    def version(self) -> int:
        with self._lock:
            return self._version

    def replace(self, raw_pairs: list[dict]) -> tuple[int, list[PreferencePair]]:
        """校验并整体替换偏好集；版本号单调递增（改回原值也递增）。"""
        new_pairs = validate_pairs(raw_pairs)  # 校验失败则不改动任何状态
        with self._lock:
            self._version += 1
            self._pairs = new_pairs
            self._updated_at = _utc_now()
            return self._version, list(self._pairs)

    def view(self) -> dict:
        with self._lock:
            return {
                "version": self._version,
                "updated_at": self._updated_at,
                "fingerprint": fingerprint(self._pairs),
                "pairs": [
                    {"index": i + 1, "prompt": p.prompt,
                     "chosen": p.chosen, "rejected": p.rejected}
                    for i, p in enumerate(self._pairs)
                ],
            }


class JobManager:
    """后台 DPO 任务的状态机与结果保留。"""

    def __init__(self, target: CausalTransformer, store: PreferenceStore) -> None:
        self._target = target
        self._store = store
        self._lock = threading.RLock()
        self._running_job_id: str | None = None
        self._running: dict | None = None
        self.last_success: RunRecord | None = None
        self.last_terminal: RunRecord | None = None

    # ---- 状态查询 ----
    def is_running(self) -> bool:
        with self._lock:
            return self._running_job_id is not None

    def status(self) -> dict:
        with self._lock:
            return {
                "state": ("running" if self._running_job_id is not None
                          else (self.last_terminal.state
                                if self.last_terminal is not None else "idle")),
                "current": dict(self._running) if self._running else None,
                "last_success": (_serialize_record(self.last_success)
                                 if self.last_success else None),
                "last_terminal": (_serialize_record(self.last_terminal)
                                  if self.last_terminal else None),
            }

    # ---- 启动 ----
    def start(self) -> dict:
        with self._lock:
            if self._running_job_id is not None:
                raise RuntimeError("已有训练任务在运行中。")
            version, pairs = self._store.snapshot()
            job_id = uuid.uuid4().hex[:12]
            started_at = _utc_now()
            input_summary = [
                {"index": i + 1, "prompt": p.prompt,
                 "chosen": p.chosen, "rejected": p.rejected}
                for i, p in enumerate(pairs)
            ]
            fp = fingerprint(pairs)
            self._running_job_id = job_id
            self._running = {
                "job_id": job_id,
                "version": version,
                "fingerprint": fp,
                "n_pairs": len(pairs),
                "started_at": started_at,
                "input_summary": input_summary,
            }
            thread = threading.Thread(
                target=self._worker,
                args=(job_id, version, list(pairs), started_at,
                      list(input_summary)),
                name=f"dpo-{job_id}",
                daemon=True,
            )
            thread.start()
            return dict(self._running)

    # ---- 后台工作线程 ----
    def _worker(self, job_id: str, version: int,
                pairs: list[PreferencePair], started_at: str,
                input_summary: list[dict]) -> None:
        # 深拷贝发生在 run_dpo 内部；此处只传不可变样本
        result = run_dpo(self._target, pairs, version)
        finished_at = _utc_now()
        with self._lock:
            # 任务已被其他启动/废止动作顶替（理论上 start 已拒绝并发）
            if self._running_job_id != job_id:
                return
            self._running_job_id = None
            self._running = None

            # 任何编辑（版本号变化）都永久废止在途结果，改回也不恢复
            if self._store.version() != version:
                record = RunRecord(
                    job_id=job_id, state="invalidated", version=version,
                    fingerprint=result.fingerprint, n_pairs=len(pairs),
                    started_at=started_at, finished_at=finished_at,
                    input_summary=input_summary,
                    error="训练期间偏好集被编辑，本次在途结果已永久废止"
                          "（即使内容改回也不恢复）；上次成功结果保留如下。",
                    result=None,
                )
            elif result.status == "success" and result.check_passed:
                record = RunRecord(
                    job_id=job_id, state="success", version=version,
                    fingerprint=result.fingerprint, n_pairs=len(pairs),
                    started_at=started_at, finished_at=finished_at,
                    input_summary=input_summary, error=None, result=result,
                )
                self.last_success = record
            else:
                record = RunRecord(
                    job_id=job_id, state="failed", version=version,
                    fingerprint=result.fingerprint, n_pairs=len(pairs),
                    started_at=started_at, finished_at=finished_at,
                    input_summary=input_summary,
                    error=result.error or "DPO 训练失败。",
                    result=None,
                )
            self.last_terminal = record


def _serialize_record(record: RunRecord) -> dict:
    out = {
        "job_id": record.job_id,
        "state": record.state,
        "version": record.version,
        "fingerprint": record.fingerprint,
        "n_pairs": record.n_pairs,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "input_summary": record.input_summary,
        "error": record.error,
    }
    if record.result is not None:
        r = record.result
        out["result"] = {
            "status": r.status,
            "check_passed": r.check_passed,
            "beta": r.beta,
            "steps": r.steps,
            "lr": r.lr,
            "n_pairs": r.n_pairs,
            "fingerprint": r.fingerprint,
            "input_version": r.input_version,
            "max_abs_param_error": r.max_abs_param_error,
            "max_abs_prescore_error": r.max_abs_prescore_error,
            "step_losses": [
                {"step": s.step, "pairwise_loss": s.pairwise_loss,
                 "fullbatch_loss": s.fullbatch_loss}
                for s in r.step_losses
            ],
            "pair_scores": [
                {
                    "index": s.index, "prompt": s.prompt,
                    "chosen": s.chosen, "rejected": s.rejected,
                    "chosen_pre": s.chosen_pre,
                    "rejected_pre": s.rejected_pre,
                    "chosen_post": s.chosen_post,
                    "rejected_post": s.rejected_post,
                    "chosen_ref": s.chosen_ref,
                    "rejected_ref": s.rejected_ref,
                    "margin_pre": s.chosen_pre - s.rejected_pre,
                    "margin_post": s.chosen_post - s.rejected_post,
                }
                for s in r.pair_scores
            ],
        }
    return out
