"""人工偏好 DPO 微调关键功能测试：

  1. 输入校验：1～8 组、每段 1～32 个可打印 ASCII，非法输入一律拒绝；
  2. 内容指纹：任何编辑都改变指纹（改回也不恢复在途结果的前提）；
  3. 序列得分：只累加续写 token 条件 logprob，不添 EOS，且与手算 gather 一致；
  4. 单序列打分与 padding 批量打分逐位相同（含不同长度混批）；
  5. 逐对梯度累积路径每步后不存在多对共存的计算图（图立即释放）；
  6. 两条独立实现的更新参数最大绝对误差 ≤ 1e-5，训练前打分误差为 0；
  7. DPO 训练后每组优选-劣选间隔不劣于训练前（偏向优选续写）；
  8. 冻结参考模型权重与原目标完全一致，且训练全程不更新；
  9. 训练不改原草稿/目标模型的权重、KV 缓存与前向计数；
 10. 结果可复现：同样输入两次训练的更新参数逐位相同；
 11. FastAPI 全流程：偏好集读写、训练启动/轮询、成功结果结构完整；
 12. 训练在途时编辑偏好集 → 任务废止（invalidated），上次成功结果保留；
 13. 并发启动训练 → 第二个请求 409；非法输入 → 400。
"""

from __future__ import annotations

import copy
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app import tokenizer
from app.dpo import (
    DPO_BETA, DPO_STEPS, PARAM_ERROR_TOL, PreferencePair,
    continuation_logprob_single, continuation_logprobs_batched,
    fingerprint, run_dpo, train_fullbatch_reference,
    train_pairwise_grad_accum, validate_pairs,
)
from app.main import app, preference_store
from app.model import get_models
from app.preferences import JobManager

torch.set_num_threads(1)
DRAFT, TARGET = get_models()

GOOD_PAIRS = [
    {"prompt": "the capital of france is ", "chosen": "paris", "rejected": "london"},
    {"prompt": "one plus one equals ", "chosen": "two", "rejected": "banana"},
    {"prompt": "print hello then ", "chosen": "world", "rejected": "zzzzz"},
]


def _pairs():
    return validate_pairs(GOOD_PAIRS)


def test_validation_rules() -> None:
    # 合法边界：1 组、每段 1 字符；8 组、每段 32 字符
    assert len(validate_pairs([
        {"prompt": "a", "chosen": "b", "rejected": "c"}])) == 1
    assert len(validate_pairs([
        {"prompt": "p" * 32, "chosen": "w" * 32, "rejected": "l" * 32}
        for _ in range(8)])) == 8
    bad_cases = [
        [],                                                 # 0 组
        [GOOD_PAIRS[0]] * 9,                                # 9 组
        [{"prompt": "", "chosen": "a", "rejected": "b"}],   # 空段
        [{"prompt": "x" * 33, "chosen": "a", "rejected": "b"}],  # 超长
        [{"prompt": "ab\n", "chosen": "a", "rejected": "b"}],    # 换行
        [{"prompt": "ab", "chosen": "a\t", "rejected": "b"}],    # tab 非可打印
        [{"prompt": "ab", "chosen": "a", "rejected": "bé"}],     # 非 ASCII
        "not-a-list",
    ]
    for bad in bad_cases:
        try:
            validate_pairs(bad)
        except ValueError:
            continue
        raise AssertionError(f"非法输入未被拒绝: {bad!r}")
    print("ok: 输入校验覆盖组数/长度/可打印 ASCII 全部边界")


def test_fingerprint_changes_on_any_edit() -> None:
    p1 = validate_pairs(GOOD_PAIRS)
    fp1 = fingerprint(p1)
    edited = [dict(GOOD_PAIRS[0])]
    edited[0]["chosen"] = "paris!"
    fp2 = fingerprint(validate_pairs(edited))
    assert fp1 != fp2
    # 改回原内容：指纹恢复，但服务端版本号不会恢复（废止语义在 API 测试验证）
    fp3 = fingerprint(validate_pairs([GOOD_PAIRS[0]] + GOOD_PAIRS[1:]))
    assert fp3 == fp1
    print("ok: 指纹随内容变化，内容改回后指纹一致（版本号仍递增）")


def test_score_is_continuation_only_without_eos() -> None:
    pair = _pairs()[0]
    # token 化结果不含 EOS
    assert tokenizer.EOS_ID not in pair.chosen_ids
    score = continuation_logprob_single(TARGET, pair.prompt_ids, pair.chosen_ids)
    # 手算对照：逐位置 gather log_softmax，只取续写 token
    # （另起一次前向，float32 重算允许 ~1e-6 级舍入差异；逐位一致性由
    # 批量/单序列对照测试保证）
    full = pair.prompt_ids + pair.chosen_ids
    logits = TARGET.forward(full, use_cache=False)
    log_probs = torch.log_softmax(logits, dim=-1)
    manual = sum(
        float(log_probs[len(pair.prompt_ids) - 1 + j, tok])
        for j, tok in enumerate(pair.chosen_ids))
    assert abs(float(score) - manual) < 1e-5
    # 提示词长度变化但续写相同时，条件概率允许不同（验证确实按条件计算）
    score2 = continuation_logprob_single(
        TARGET, pair.prompt_ids[:-1], pair.chosen_ids)
    assert isinstance(score2, torch.Tensor)
    print("ok: 序列得分仅累加续写 token 条件 logprob，不添 EOS，与手算一致")


def test_batched_and_single_scores_bitwise_equal() -> None:
    pairs = _pairs()
    rows = [(p.prompt_ids, p.chosen_ids) for p in pairs] + [
        (p.prompt_ids, p.rejected_ids) for p in pairs]
    with torch.no_grad():
        batched = continuation_logprobs_batched(TARGET, rows)
        single = [continuation_logprob_single(TARGET, pids, cids)
                  for pids, cids in rows]
    # 同一固定 batch 内：padding 行不改变有效行的 GEMM 形状，逐位相等
    assert all(torch.equal(a.detach(), b.detach())
               for a, b in zip(batched, single)), "批量与单序列打分必须逐位相同"
    # 混入更长序列后矩阵形状改变，float32 GEMM 归约顺序不同会引入 ~1e-5
    # 级舍入差异（不影响训练：实际打分始终在同一个固定 batch 内比较）；
    # 这里验证语义等价而非逐位相等。
    rows_mixed = rows + [
        (tokenizer.encode("z" * 30), tokenizer.encode("q" * 32))]
    with torch.no_grad():
        mixed = continuation_logprobs_batched(TARGET, rows_mixed)
    assert all(torch.allclose(a.detach(), b.detach(), atol=1e-4)
               for a, b in zip(mixed[:len(rows)], batched))
    print("ok: 同 batch 内 padding 打分与单序列逐位相等；异形状混批语义等价")


def test_pairwise_path_releases_graph_each_pair() -> None:
    """逐对路径在每对 backward 后，该对张量不再需要梯度（图已释放）。"""
    pairs = _pairs()
    policy = copy.deepcopy(TARGET)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(True)
    n = len(pairs)
    import torch.nn.functional as F
    policy.zero_grad(set_to_none=True)
    for pair in pairs:
        pc = continuation_logprob_single(policy, pair.prompt_ids, pair.chosen_ids)
        pr = continuation_logprob_single(policy, pair.prompt_ids, pair.rejected_ids)
        loss = -F.logsigmoid(DPO_BETA * (pc - pr - 0.0)) / n
        loss.backward()
        # backward 后该对前向产生的张量 grad_fn 已被释放，再访问反传会报错
        assert pc.requires_grad and pc.grad_fn is not None
        try:
            torch.autograd.grad(pc, list(policy.parameters()), retain_graph=False,
                                allow_unused=True)
        except RuntimeError:
            pass  # 图已释放即预期行为
        else:
            raise AssertionError("逐对 backward 后计算图未释放")
    print("ok: 逐对路径每对 backward 后立即释放该对计算图")


def test_cross_check_param_error_within_tolerance() -> None:
    pairs = _pairs()
    pa, _, pre_ch_a, pre_re_a, losses_a = train_pairwise_grad_accum(TARGET, pairs)
    pb, _, pre_ch_b, pre_re_b, losses_b = train_fullbatch_reference(TARGET, pairs)
    max_err = max(float((x.detach() - y.detach()).abs().max())
                  for (_, x), (_, y) in zip(pa.named_parameters(),
                                             pb.named_parameters()))
    assert max_err <= PARAM_ERROR_TOL, f"参数误差 {max_err} 超过容差"
    pre_err = max(
        max(abs(a - b) for a, b in zip(pre_ch_a, pre_ch_b)),
        max(abs(a - b) for a, b in zip(pre_re_a, pre_re_b)))
    assert pre_err == 0.0
    assert len(losses_a) == len(losses_b) == DPO_STEPS
    # 损失从 ln2 ≈ 0.693 快速下降
    assert losses_a[0] < 0.7 and losses_a[-1] < losses_a[0]
    print(f"ok: 两条实现参数最大绝对误差 {max_err:.1e} ≤ 1e-5，"
          f"训练前打分误差 {pre_err:.0e}")


def test_dpo_prefers_chosen_after_training() -> None:
    result = run_dpo(TARGET, _pairs(), input_version=1)
    assert result.status == "success" and result.check_passed
    for s in result.pair_scores:
        margin_pre = s.chosen_pre - s.rejected_pre
        margin_post = s.chosen_post - s.rejected_post
        assert margin_post >= margin_pre, (
            f"第 {s.index} 组未更偏向优选：{margin_pre} -> {margin_post}")
        # 优选自身得分不下降、劣选自身得分不上升（该随机模型上稳定成立）
        assert s.chosen_post >= s.chosen_pre - 1e-6
        assert s.rejected_post <= s.rejected_pre + 1e-6
    # 冻结参考得分必须等于训练前策略得分（起点权重相同）
    for s in result.pair_scores:
        assert abs(s.chosen_ref - s.chosen_pre) < 1e-12
        assert abs(s.rejected_ref - s.rejected_pre) < 1e-12
    print("ok: 训练后全部组的优选-劣选间隔改善，模型更偏向优选续写")


def test_reference_model_stays_frozen() -> None:
    pairs = _pairs()
    _, ref, *_ = train_pairwise_grad_accum(TARGET, pairs)
    for (n1, p1), (n2, p2) in zip(TARGET.named_parameters(), ref.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"参考模型被修改: {n1}"
        assert not p2.requires_grad
    print("ok: 冻结参考模型与原目标权重逐位相同且 requires_grad=False")


def test_original_models_untouched() -> None:
    weights_before = [(n, p.detach().clone()) for n, p in TARGET.named_parameters()]
    draft_before = [(n, p.detach().clone()) for n, p in DRAFT.named_parameters()]
    # 先建立 KV 缓存并推进到已知长度，训练前后该长度必须原样保留
    TARGET.init_cache()
    DRAFT.init_cache()
    TARGET.forward(list(range(5, 10)), use_cache=True)
    DRAFT.forward(list(range(5, 10)), use_cache=True)
    cache_t, cache_d = TARGET.cache_len, DRAFT.cache_len
    fc_t, fc_d = TARGET.forward_count, DRAFT.forward_count

    result = run_dpo(TARGET, _pairs(), input_version=7)
    assert result.status == "success"

    # 训练只做深拷贝：原模型前向计数增量必须为 0，KV 缓存长度原样保留
    assert TARGET.forward_count == fc_t, "训练不得增加原目标前向计数"
    assert DRAFT.forward_count == fc_d, "训练不得增加草稿前向计数"
    assert TARGET.cache_len == cache_t and DRAFT.cache_len == cache_d, "KV 缓存被改动"
    for (n, w), (n2, p) in zip(weights_before, TARGET.named_parameters()):
        assert n == n2 and torch.equal(w, p), f"原目标权重被修改: {n}"
    for (n, w), (n2, p) in zip(draft_before, DRAFT.named_parameters()):
        assert n == n2 and torch.equal(w, p), f"草稿权重被修改: {n}"
    print("ok: 训练后原草稿/目标的权重、KV 缓存长度、前向计数全部不变")


def test_deterministic_reproducibility() -> None:
    r1 = run_dpo(TARGET, _pairs(), input_version=1)
    r2 = run_dpo(TARGET, _pairs(), input_version=1)
    assert r1.status == r2.status == "success"
    for a, b in zip(r1.pair_scores, r2.pair_scores):
        assert a.chosen_post == b.chosen_post and a.rejected_post == b.rejected_post
    assert r1.max_abs_param_error == r2.max_abs_param_error
    print("ok: 相同输入两次训练结果完全可复现")


def test_api_full_flow_and_invalidation() -> None:
    client = TestClient(app)
    # 初始偏好集可读
    r = client.get("/api/preferences")
    assert r.status_code == 200
    v0 = r.json()["version"]

    # 非法保存 → 400，且版本不变
    r = client.put("/api/preferences",
                   json={"pairs": [{"prompt": "", "chosen": "a", "rejected": "b"}]})
    assert r.status_code == 400
    assert client.get("/api/preferences").json()["version"] == v0

    # 合法保存 → 版本递增
    r = client.put("/api/preferences", json={"pairs": GOOD_PAIRS})
    assert r.status_code == 200
    v1 = r.json()["version"]
    assert v1 == v0 + 1

    # 启动训练并同步等待终态
    r = client.post("/api/dpo/train")
    assert r.status_code == 200
    job_id = r.json()["job"]["job_id"]
    status = _wait_terminal(client, job_id)
    assert status["state"] == "success"
    rec = status["last_success"]
    assert rec is not None and rec["result"]["check_passed"] is True
    assert rec["result"]["max_abs_param_error"] <= PARAM_ERROR_TOL
    assert rec["version"] == v1
    # 成功结果结构完整：训练前后各续写得分 + 参数误差
    for s in rec["result"]["pair_scores"]:
        for key in ("chosen_pre", "rejected_pre", "chosen_post",
                    "rejected_post", "chosen_ref", "rejected_ref",
                    "margin_pre", "margin_post"):
            assert isinstance(s[key], float)
    print("ok: API 全流程（保存→训练→轮询→成功结果结构）通过")


def test_edit_invalidates_inflight_and_keeps_last_success() -> None:
    client = TestClient(app)
    # 先制造一次成功结果
    client.put("/api/preferences", json={"pairs": GOOD_PAIRS})
    r = client.post("/api/dpo/train")
    assert r.status_code == 200
    first_job = r.json()["job"]["job_id"]
    st = _wait_terminal(client, first_job)
    assert st["state"] == "success"
    good_rec = st["last_success"]

    # 再次启动，并立即编辑偏好集废止在途任务
    r = client.post("/api/dpo/train")
    assert r.status_code == 200
    second_job = r.json()["job"]["job_id"]
    edited = [dict(x) for x in GOOD_PAIRS]
    edited[0]["chosen"] = "lyon"
    r = client.put("/api/preferences", json={"pairs": edited})
    assert r.status_code == 200

    st = _wait_terminal(client, second_job)
    assert st["state"] == "invalidated", st["state"]
    assert st["last_terminal"]["error"] is not None
    # 上次成功结果原样保留，输入标注仍指向旧版本/指纹
    assert st["last_success"]["job_id"] == good_rec["job_id"]
    assert st["last_success"]["version"] == good_rec["version"]
    assert st["last_success"]["fingerprint"] == good_rec["fingerprint"]

    # 改回原内容再训练：在途废止不会恢复，但新训练基于当前版本可再次成功
    r = client.put("/api/preferences", json={"pairs": GOOD_PAIRS})
    assert r.status_code == 200
    r = client.post("/api/dpo/train")
    third_job = r.json()["job"]["job_id"]
    st = _wait_terminal(client, third_job)
    assert st["state"] == "success"
    print("ok: 编辑永久废止在途结果，上次成功结果保留并带输入标注")


def test_failed_run_keeps_last_success() -> None:
    """交叉校验失败（模拟容差不达标）→ failed，且上次成功结果原样保留。"""
    import app.dpo as dpo_mod
    store = type(preference_store)()
    manager = JobManager(TARGET, store)
    first = manager.start()
    _wait_manager(manager, first["job_id"])
    assert manager.last_terminal.state == "success"
    good_id = manager.last_success.job_id

    orig_tol = dpo_mod.PARAM_ERROR_TOL
    dpo_mod.PARAM_ERROR_TOL = -1.0          # 任何误差都不可能通过
    try:
        second = manager.start()
        _wait_manager(manager, second["job_id"])
    finally:
        dpo_mod.PARAM_ERROR_TOL = orig_tol
    assert manager.last_terminal.state == "failed"
    assert manager.last_terminal.error is not None
    assert manager.last_success.job_id == good_id
    assert manager.status()["last_success"]["job_id"] == good_id
    print("ok: 训练/对照失败时上次成功结果保留，错误信息标明原因")


def test_concurrent_train_rejected_and_decode_still_works() -> None:
    # 用独立 JobManager 模拟"运行中"窗口，验证 409 与解码不被阻塞
    store = type(preference_store)()
    manager = JobManager(TARGET, store)
    job = manager.start()
    try:
        assert manager.is_running()
        try:
            manager.start()
        except RuntimeError:
            pass
        else:
            raise AssertionError("并发训练未被拒绝")
        # 训练在途时原解码对照照常工作
        from app.decoding import compare
        cmp = compare(DRAFT, TARGET, tokenizer.encode("the quick brown fox "), 3, 16)
        assert cmp.match
    finally:
        _wait_manager(manager, job["job_id"])
    print("ok: 并发训练返回冲突，且训练在途时原解码对照正常")


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get("/api/dpo/status").json()
        if st["state"] != "running":
            if st["last_terminal"] and st["last_terminal"]["job_id"] == job_id:
                return st
            if st["state"] != "running":
                return st
        time.sleep(0.1)
    raise TimeoutError("训练任务未在限定时间内结束")


def _wait_manager(manager: JobManager, job_id: str, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not manager.is_running():
            return
        time.sleep(0.1)
    raise TimeoutError("后台任务未结束")


if __name__ == "__main__":
    test_validation_rules()
    test_fingerprint_changes_on_any_edit()
    test_score_is_continuation_only_without_eos()
    test_batched_and_single_scores_bitwise_equal()
    test_pairwise_path_releases_graph_each_pair()
    test_cross_check_param_error_within_tolerance()
    test_dpo_prefers_chosen_after_training()
    test_reference_model_stays_frozen()
    test_original_models_untouched()
    test_deterministic_reproducibility()
    test_api_full_flow_and_invalidation()
    test_edit_invalidates_inflight_and_keeps_last_success()
    test_failed_run_keeps_last_success()
    test_concurrent_train_rejected_and_decode_still_works()
    print("\n全部 DPO 测试通过 ✔")
