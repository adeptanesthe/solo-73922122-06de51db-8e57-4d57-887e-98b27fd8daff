"""FastAPI 后端：推测解码 vs 无缓存贪心基线的对照服务，并提供人工偏好 DPO 微调。"""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import tokenizer
from .decoding import compare
from .model import get_models
from .preferences import JobManager, PreferenceStore
app = FastAPI(title="Speculative Decoding Verifier")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# 候选提示词（纯 ASCII，固定词表全部可编码）；启动时扫描出
# 分别触发"拒绝"和"整块接受"的两条预设。
_CANDIDATE_PROMPTS = [
    "the quick brown fox jumps over the lazy dog while ",
    "hello world, this is a tiny transformer demo of ",
    "speculative decoding uses a small draft model to ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
    "in the beginning there was only silence and then ",
    "print(\"sum =\")  # ",
    "the answer to life the universe and everything is ",
    "one two three four five six seven eight nine ten ",
    "python java rust go c plus plus ruby swift scala ",
    "a stitch in time saves nine, and early birds get ",
    "the rain in spain stays mainly on the plain while ",
    "to be or not to be that is the question whether ",
    "x = 1\ny = 2\nz = x + y\nprint(z)\n# output is ",
    "all work and no play makes jack a dull boy, so ",
    "once upon a time in a small village lived an old ",
    "the capital of france is paris and the capital of ",
    "0 1 1 2 3 5 8 13 21 34 55 89 144 233 next number ",
    "she sells sea shells by the sea shore, the shells ",
    "machine learning models can be compressed with ",
    "alpha beta gamma delta epsilon zeta eta theta iota ",
]

# 固定预设：经实测与测试套件锁定，分别稳定触发"拒绝"与"整块接受"。
# 拒绝预设：草稿与目标在该上下文上分歧，首轮接受后持续被拒，
# 用于验证"被拒草稿不会污染后续生成"；
# 整块接受预设：每轮草稿 token 全部通过，展示目标前向节省的上限。
PRESET_REJECT = "python java rust go c plus plus ruby swift scala "
PRESET_ALL_ACCEPTED = "the quick brown fox jumps over the lazy dog while "


class DecodeRequest(BaseModel):
    prompt: str = Field(default="", max_length=200)
    k: int = Field(default=3, ge=1, le=4)
    max_new: int = Field(default=32, ge=1, le=32)


# 服务端偏好集与 DPO 任务管理器（首次 get_models 后惰性接线；
# 偏好集的字段/组数/可打印 ASCII 校验集中在 app.dpo.validate_pairs）
preference_store = PreferenceStore()
job_manager: JobManager | None = None


def _get_job_manager() -> JobManager:
    global job_manager
    if job_manager is None:
        _, target = get_models()
        job_manager = JobManager(target, preference_store)
    return job_manager


def _token_view(ids: list[int]) -> list[dict]:
    return [{"id": int(t), "text": tokenizer.render_token(int(t))} for t in ids]


def _serialize(cmp_result, prompt_text: str) -> dict:
    base, spec = cmp_result.baseline, cmp_result.speculative
    return {
        "prompt": prompt_text,
        "prompt_ids": base.prompt_ids,
        "match": cmp_result.match,
        "status": "PASS" if cmp_result.match else "FAIL",
        "first_mismatch_index": cmp_result.first_mismatch_index,
        "baseline": {
            "new_tokens": _token_view(base.new_tokens),
            "text": tokenizer.decode(base.new_tokens),
            "target_forward_count": base.target_forward_count,
            "stopped_by_eos": base.stopped_by_eos,
        },
        "speculative": {
            "new_tokens": _token_view(spec.new_tokens),
            "text": tokenizer.decode(spec.new_tokens),
            "target_forward_count": spec.target_forward_count,
            "draft_forward_count": spec.draft_forward_count,
            "stopped_by_eos": spec.stopped_by_eos,
            "rounds": [asdict(r) | {
                "draft_tokens_view": _token_view(r.draft_tokens),
                "target_predictions_view": _token_view(r.target_predictions),
                "target_tokens_view": _token_view(r.target_tokens),
            } for r in spec.rounds],
        },
        "savings": {
            "baseline_forwards": base.target_forward_count,
            "spec_forwards": spec.target_forward_count,
            "saved": base.target_forward_count - spec.target_forward_count,
        },
    }


@app.get("/api/health")
def health() -> dict:
    draft, target = get_models()
    return {
        "ok": True,
        "vocab_size": tokenizer.VOCAB_SIZE,
        "target_layers": target.n_layers,
        "draft_layers": draft.n_layers,
    }


@app.post("/api/decode")
def decode(req: DecodeRequest) -> JSONResponse:
    prompt_ids = tokenizer.encode(req.prompt)
    if not prompt_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "提示词编码后为空，请输入可打印 ASCII 字符。"},
        )
    draft, target = get_models()
    result = compare(draft, target, prompt_ids, req.k, req.max_new)
    return JSONResponse(_serialize(result, req.prompt))


@app.get("/api/presets")
def presets() -> dict:
    """返回固定的两条预设（由测试套件锁定其判定类型，避免请求时全量扫描）。"""
    return {"reject": PRESET_REJECT, "all_accepted": PRESET_ALL_ACCEPTED}


# ---------------------------------------------------------------- 偏好 DPO

@app.get("/api/preferences")
def get_preferences() -> dict:
    """返回服务端偏好集（含版本号、指纹），供两个窗口同步同一输入。"""
    view = preference_store.view()
    view["training_running"] = _get_job_manager().is_running()
    return view


@app.put("/api/preferences")
async def put_preferences(request: Request) -> JSONResponse:
    """整体替换偏好集。任何保存都会递增版本号并废止在途训练结果。"""
    try:
        body = await request.json()
        if not isinstance(body, dict) or "pairs" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "请求体必须是含 pairs 数组的 JSON 对象。"},
            )
        version, _ = preference_store.replace(body["pairs"])
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # JSON 解析失败等畸形请求
        return JSONResponse(
            status_code=400,
            content={"error": f"请求体必须是含 pairs 数组的 JSON：{exc}"},
        )
    return JSONResponse({
        "ok": True,
        "version": version,
        "preferences": get_preferences(),
        "notice": "偏好集已更新；若有训练在途，其结果已永久废止。",
    })


@app.post("/api/dpo/train")
def dpo_train() -> JSONResponse:
    """启动一次后台 DPO 训练（从原目标权重深拷贝出发）；已有任务时返回 409。"""
    manager = _get_job_manager()
    if manager.is_running():
        return JSONResponse(
            status_code=409,
            content={"error": "已有训练任务在运行中，请等待其结束。"},
        )
    job = manager.start()
    return JSONResponse({"ok": True, "job": job})


@app.get("/api/dpo/status")
def dpo_status() -> dict:
    """训练状态机 + 上次成功结果（带输入版本/指纹标注）。"""
    return _get_job_manager().status()


@app.get("/dpo")
def dpo_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "dpo.html"))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
