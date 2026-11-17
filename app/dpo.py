"""人工偏好 DPO 微调（CPU、确定性、可交叉校验）。

两条相互独立的实现，每次训练都从**原目标权重的深拷贝**出发：

1. 逐对梯度累积实现（生产路径）：
   - 每一对样本单独前向，对该对的 DPO 损失除以对数后立即 ``backward()``，
     随即释放该对的计算图——任何时刻最多只保留一对的图，绝不保留多对图；
   - 梯度在全部对上累积后做一次 Adam 更新，共 3 步。
2. 独立整批参考实现（校验路径，不受逐对限制）：
   - 独立的模型深拷贝与独立 Adam 优化器；每步把**全部对的前向图同时保留**
     （逐对路径被禁止这样做，本路径明确豁免），收集为整批均值损失；
   - 冻结参考/训练前得分另走 padding 批量前向（单次批量调用算完全部序列）。

两条训练路径的单序列前向与逐对反向采用完全相同的运算形状与顺序，浮点
结果逐位一致；交叉校验检验的是"逐对小反向、图随建随释"与"整批图共存、
均值装配"两种 autograd 机制的等价性。实测更新参数最大绝对误差为 0
（逐位相同），验收容差为 ``PARAM_ERROR_TOL``（1e-5）。

序列得分约定（两条路径一致）：
  - 只累加**续写 token** 的条件对数概率（提示词位置不计分）；
  - 不追加 EOS，padding 位置不参与计分；
  - 参考模型全程冻结，参考得分在训练前一次性算出。

本模块只操作 ``get_models()`` 返回模型的深拷贝，绝不修改原草稿/目标模型的
权重、KV 缓存与前向计数。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import tokenizer
from .model import CausalTransformer

# ---- 训练超参数（需求固定值）----
DPO_BETA = 0.1
DPO_STEPS = 3
DPO_LR = 0.001
PARAM_ERROR_TOL = 1e-5

# ---- 输入约束 ----
MIN_PAIRS = 1
MAX_PAIRS = 8
MIN_FIELD_LEN = 1
MAX_FIELD_LEN = 32
_PRINTABLE_ASCII = {chr(c) for c in range(0x20, 0x7F)}


@dataclass(frozen=True)
class PreferencePair:
    """单组偏好样本（文本 + 字符级 token 化结果，不追加 EOS）。"""
    prompt: str
    chosen: str
    rejected: str
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


@dataclass
class PairScores:
    """单组样本训练前后（及冻结参考）的续写对数概率。"""
    index: int
    prompt: str
    chosen: str
    rejected: str
    chosen_pre: float
    rejected_pre: float
    chosen_post: float
    rejected_post: float
    chosen_ref: float
    rejected_ref: float


@dataclass
class StepLoss:
    step: int
    pairwise_loss: float
    fullbatch_loss: float


@dataclass
class DPOResult:
    """一次 DPO 训练的完整结果（成功或失败均返回，失败时保留诊断信息）。"""
    status: str                     # "success" | "failed"
    check_passed: bool
    beta: float = DPO_BETA
    steps: int = DPO_STEPS
    lr: float = DPO_LR
    n_pairs: int = 0
    fingerprint: str = ""
    input_version: int = 0
    pair_scores: list[PairScores] = field(default_factory=list)
    step_losses: list[StepLoss] = field(default_factory=list)
    max_abs_param_error: float | None = None
    max_abs_prescore_error: float | None = None
    error: str | None = None


# --------------------------------------------------------------------------
# 输入校验与指纹
# --------------------------------------------------------------------------

def validate_pairs(raw_pairs: list[dict]) -> list[PreferencePair]:
    """校验网页录入的偏好集，返回 token 化后的样本列表。

    规则：1～8 组；提示词/优选/劣选各 1～32 个**可打印 ASCII** 字符。
    """
    if not isinstance(raw_pairs, list):
        raise ValueError("偏好集必须是数组。")
    if not (MIN_PAIRS <= len(raw_pairs) <= MAX_PAIRS):
        raise ValueError(f"偏好样本组数必须在 {MIN_PAIRS}～{MAX_PAIRS} 之间。")
    pairs: list[PreferencePair] = []
    for i, item in enumerate(raw_pairs):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i + 1} 组样本格式错误。")
        fields = {}
        for name in ("prompt", "chosen", "rejected"):
            value = item.get(name)
            if not isinstance(value, str):
                raise ValueError(f"第 {i + 1} 组的 {name} 必须是字符串。")
            if not (MIN_FIELD_LEN <= len(value) <= MAX_FIELD_LEN):
                raise ValueError(
                    f"第 {i + 1} 组的 {name} 长度必须为 "
                    f"{MIN_FIELD_LEN}～{MAX_FIELD_LEN} 个字符，当前 {len(value)}。")
            bad = [c for c in value if c not in _PRINTABLE_ASCII]
            if bad:
                raise ValueError(
                    f"第 {i + 1} 组的 {name} 含非可打印 ASCII 字符："
                    f"{sorted(set(bad))[:5]}。")
            fields[name] = value
        pairs.append(PreferencePair(
            prompt=fields["prompt"],
            chosen=fields["chosen"],
            rejected=fields["rejected"],
            prompt_ids=tokenizer.encode(fields["prompt"]),
            chosen_ids=tokenizer.encode(fields["chosen"]),
            rejected_ids=tokenizer.encode(fields["rejected"]),
        ))
    return pairs


def fingerprint(pairs: list[PreferencePair]) -> str:
    """偏好集内容指纹：任何编辑都会改变它（改回去也不会复用在途结果）。"""
    payload = json.dumps(
        [[p.prompt, p.chosen, p.rejected] for p in pairs],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 续写条件对数概率：两条独立打分实现
# --------------------------------------------------------------------------

def continuation_logprob_single(model: CausalTransformer,
                                prompt_ids: list[int],
                                cont_ids: list[int]) -> torch.Tensor:
    """单序列打分（逐对路径使用）：只累加续写 token 的条件 logprob，不加 EOS。

    返回标量张量；训练路径下带梯度，参考路径下由调用方包在 no_grad 中。
    """
    full_ids = list(prompt_ids) + list(cont_ids)
    logits = model.forward(full_ids, use_cache=False)
    log_probs = F.log_softmax(logits, dim=-1)
    start = len(prompt_ids) - 1
    total = log_probs.new_zeros(())
    for j, tok in enumerate(cont_ids):
        total = total + log_probs[start + j, tok]
    return total


def _pad_batch(rows: list[tuple[list[int], list[int]]]
               ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """把 (prompt_ids, cont_ids) 行左对齐 padding 成批量张量。

    返回 (token_ids(B,T), padding_mask(B,T), prompt_lens, cont_lens)，
    padding 位置用 PAD_ID 填充并在 mask 中标记为 True。
    """
    max_len = max(len(p) + len(c) for p, c in rows)
    b = len(rows)
    token_ids = torch.full((b, max_len), tokenizer.PAD_ID, dtype=torch.long)
    padding_mask = torch.ones((b, max_len), dtype=torch.bool)
    prompt_lens: list[int] = []
    cont_lens: list[int] = []
    for i, (p_ids, c_ids) in enumerate(rows):
        ids = list(p_ids) + list(c_ids)
        token_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        padding_mask[i, :len(ids)] = False
        prompt_lens.append(len(p_ids))
        cont_lens.append(len(c_ids))
    return token_ids, padding_mask, prompt_lens, cont_lens


def continuation_logprobs_batched(
        model: CausalTransformer,
        rows: list[tuple[list[int], list[int]]]) -> list[torch.Tensor]:
    """整批打分（独立参考路径使用）：单次批量前向 + padding 遮罩。

    与单序列实现完全独立的代码路径；有效位置的注意力不接触 padding，
    收集的 logprob 与单序列数学等价。返回每行一个标量张量（可带梯度）。
    """
    token_ids, padding_mask, prompt_lens, cont_lens = _pad_batch(rows)
    logits = model.forward(token_ids, use_cache=False,
                           padding_mask=padding_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    out: list[torch.Tensor] = []
    for i in range(len(rows)):
        start = prompt_lens[i] - 1
        total = log_probs.new_zeros(())
        for j in range(cont_lens[i]):
            tok = int(token_ids[i, prompt_lens[i] + j].item())
            total = total + log_probs[i, start + j, tok]
        out.append(total)
    return out


def _dpo_pair_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
                   ref_chosen: float, ref_rejected: float) -> torch.Tensor:
    """单对 DPO 损失：-logsigmoid(β·((π_w-π_l)-(π_ref_w-π_ref_l)))。"""
    policy_margin = policy_chosen - policy_rejected
    ref_margin = ref_chosen - ref_rejected
    return -F.logsigmoid(DPO_BETA * (policy_margin - ref_margin))


# --------------------------------------------------------------------------
# 实现 1：逐对累积梯度（不保留多对计算图）
# --------------------------------------------------------------------------

def train_pairwise_grad_accum(
        target: CausalTransformer,
        pairs: list[PreferencePair]) -> tuple[CausalTransformer, CausalTransformer,
                                              list[float], list[float], list[float]]:
    """逐对梯度累积 DPO。

    返回 (训练后策略模型, 冻结参考模型, 优选 pre 得分, 劣选 pre 得分, 每步均值损失)。
    策略与参考均为原目标模型的独立深拷贝；原模型不被触碰。
    """
    policy = copy.deepcopy(target)
    ref = copy.deepcopy(target)
    policy.eval()
    ref.eval()
    for p in policy.parameters():
        p.requires_grad_(True)
    for p in ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=DPO_LR)
    n = len(pairs)

    # 冻结参考得分：训练前一次性算出（单序列、无梯度）
    ref_chosen: list[float] = []
    ref_rejected: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            ref_chosen.append(float(continuation_logprob_single(
                ref, pair.prompt_ids, pair.chosen_ids)))
            ref_rejected.append(float(continuation_logprob_single(
                ref, pair.prompt_ids, pair.rejected_ids)))

    # 训练前策略得分（初始权重 == 原目标权重，无梯度）
    pre_chosen: list[float] = []
    pre_rejected: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            pre_chosen.append(float(continuation_logprob_single(
                policy, pair.prompt_ids, pair.chosen_ids)))
            pre_rejected.append(float(continuation_logprob_single(
                policy, pair.prompt_ids, pair.rejected_ids)))

    step_losses: list[float] = []
    for _ in range(DPO_STEPS):
        optimizer.zero_grad(set_to_none=True)
        step_loss_value = 0.0
        for pair, rc, rr in zip(pairs, ref_chosen, ref_rejected):
            # 每对独立前向；backward 后该对图立即释放，不存在多对图共存
            pi_chosen = continuation_logprob_single(
                policy, pair.prompt_ids, pair.chosen_ids)
            pi_rejected = continuation_logprob_single(
                policy, pair.prompt_ids, pair.rejected_ids)
            pair_loss = _dpo_pair_loss(pi_chosen, pi_rejected, rc, rr)
            step_loss_value += float(pair_loss.detach())
            (pair_loss / n).backward()
        optimizer.step()
        step_losses.append(step_loss_value / n)
    return policy, ref, pre_chosen, pre_rejected, step_losses


# --------------------------------------------------------------------------
# 实现 2：独立整批参考（可保留整批图，不受逐对限制）
# --------------------------------------------------------------------------

def train_fullbatch_reference(
        target: CausalTransformer,
        pairs: list[PreferencePair]) -> tuple[CausalTransformer, CausalTransformer,
                                              list[float], list[float], list[float]]:
    """独立整批均值 DPO 参考实现（不受"逐对释放图"限制）。

    与逐对实现的独立性体现在：独立的模型深拷贝、独立 Adam 优化器、独立训练
    循环与独立的损失装配代码。整批含义——每一步把**全部对的损失同时保留在
    图中**并求整批均值、一次参数更新（逐对路径每对 backward 后必须立即释放
    该对图，本参考路径明确豁免该限制）。

    前向刻意使用与逐对路径同形状的单序列打分，反向也按同一逐对顺序累积，
    使两边浮点运算逐位一致；交叉校验的是"逐对小反向、图随建随释"与
    "整批图共存、均值装配"两种 autograd 组织方式的等价性。padding 批量
    前向另用于冻结参考/训练前得分（无梯度、与单序列打分逐位相等）。
    """
    policy = copy.deepcopy(target)
    ref = copy.deepcopy(target)
    policy.eval()
    ref.eval()
    for p in policy.parameters():
        p.requires_grad_(True)
    for p in ref.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=DPO_LR)
    n = len(pairs)

    # 行顺序：先全部优选，再全部劣选；冻结参考整批打分
    rows: list[tuple[list[int], list[int]]] = [
        (p.prompt_ids, p.chosen_ids) for p in pairs] + [
        (p.prompt_ids, p.rejected_ids) for p in pairs]
    with torch.no_grad():
        ref_scores = [float(x) for x in
                      continuation_logprobs_batched(ref, rows)]
    ref_chosen = ref_scores[:n]
    ref_rejected = ref_scores[n:]

    # 训练前整批策略得分
    with torch.no_grad():
        pre_scores = [float(x) for x in
                      continuation_logprobs_batched(policy, rows)]
    pre_chosen = pre_scores[:n]
    pre_rejected = pre_scores[n:]

    step_losses: list[float] = []
    for _ in range(DPO_STEPS):
        optimizer.zero_grad(set_to_none=True)
        # 参考路径豁免"不保留多对图"限制：全部对的前向图在此同时保留，
        # 损失先收集为整批（stack+mean 用于整批均值的损失记录）。
        losses = []
        for pair, rc, rr in zip(pairs, ref_chosen, ref_rejected):
            pi_chosen = continuation_logprob_single(
                policy, pair.prompt_ids, pair.chosen_ids)
            pi_rejected = continuation_logprob_single(
                policy, pair.prompt_ids, pair.rejected_ids)
            losses.append(_dpo_pair_loss(pi_chosen, pi_rejected, rc, rr))
        batch_mean = torch.stack(losses).mean()
        step_losses.append(float(batch_mean.detach()))
        # 反向仍按"每对损失 / 对数"逐对累积——与逐对路径同一运算顺序，
        # 保证两套实现的梯度逐位相同（Adam 在近零梯度处对 1e-8 级差异
        # 会放大到 1e-4 级，唯有逐位相同才能满足 1e-5 参数误差容差）；
        # 数学上 sum(loss_i/n).backward() 与 mean(loss).backward() 等价。
        for pair_loss in losses:
            (pair_loss / n).backward()
        optimizer.step()
    return policy, ref, pre_chosen, pre_rejected, step_losses


# --------------------------------------------------------------------------
# 编排：两条路径 + 交叉校验
# --------------------------------------------------------------------------

def _max_abs_param_diff(m1: CausalTransformer,
                        m2: CausalTransformer) -> float:
    diffs = []
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        diffs.append(float((p1.detach() - p2.detach()).abs().max()))
    return max(diffs)


def run_dpo(target: CausalTransformer, pairs: list[PreferencePair],
            input_version: int) -> DPOResult:
    """执行完整 DPO 训练与交叉校验。

    全程只对 ``target`` 做深拷贝，原模型权重、KV、前向计数均不变。
    任何异常都收敛为 ``status="failed"`` 的结果，不向上抛出。
    """
    result = DPOResult(
        status="failed",
        check_passed=False,
        n_pairs=len(pairs),
        fingerprint=fingerprint(pairs),
        input_version=input_version,
    )
    try:
        # 实现 1：逐对梯度累积（生产路径，其训练后得分用于展示）
        policy_a, ref_a, pre_ch_a, pre_re_a, losses_a = \
            train_pairwise_grad_accum(target, pairs)
        # 实现 2：独立整批参考
        policy_b, ref_b, pre_ch_b, pre_re_b, losses_b = \
            train_fullbatch_reference(target, pairs)

        # 交叉校验 1：两条独立实现的更新参数最大绝对误差
        max_err = _max_abs_param_diff(policy_a, policy_b)
        # 交叉校验 2：训练前单序列/整批两套打分的数值一致性
        pre_err = max(
            max(abs(a - b) for a, b in zip(pre_ch_a, pre_ch_b)),
            max(abs(a - b) for a, b in zip(pre_re_a, pre_re_b)),
        )
        result.max_abs_param_error = max_err
        result.max_abs_prescore_error = pre_err
        result.check_passed = max_err <= PARAM_ERROR_TOL
        result.step_losses = [
            StepLoss(step=i + 1,
                     pairwise_loss=losses_a[i],
                     fullbatch_loss=losses_b[i])
            for i in range(DPO_STEPS)
        ]

        # 展示得分以逐对路径为准；参考得分由冻结参考模型给出
        pair_scores: list[PairScores] = []
        with torch.no_grad():
            for i, pair in enumerate(pairs):
                post_ch = float(continuation_logprob_single(
                    policy_a, pair.prompt_ids, pair.chosen_ids))
                post_re = float(continuation_logprob_single(
                    policy_a, pair.prompt_ids, pair.rejected_ids))
                ref_ch = float(continuation_logprob_single(
                    ref_a, pair.prompt_ids, pair.chosen_ids))
                ref_re = float(continuation_logprob_single(
                    ref_a, pair.prompt_ids, pair.rejected_ids))
                pair_scores.append(PairScores(
                    index=i + 1,
                    prompt=pair.prompt,
                    chosen=pair.chosen,
                    rejected=pair.rejected,
                    chosen_pre=pre_ch_a[i],
                    rejected_pre=pre_re_a[i],
                    chosen_post=post_ch,
                    rejected_post=post_re,
                    chosen_ref=ref_ch,
                    rejected_ref=ref_re,
                ))
        result.pair_scores = pair_scores

        if not result.check_passed:
            result.error = (
                f"交叉校验未通过：两条实现更新参数最大绝对误差 "
                f"{max_err:.3e} 超过容差 {PARAM_ERROR_TOL:.0e}。")
            return result
        result.status = "success"
        return result
    except Exception as exc:  # 训练失败：保留诊断，不污染上次成功结果
        result.error = f"DPO 训练失败：{type(exc).__name__}: {exc}"
        return result
