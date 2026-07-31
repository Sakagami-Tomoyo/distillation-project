"""用于知识蒸馏的 KL 散度损失函数。

所有函数均采用内存高效的计算方式，仅在有效（非填充）token 位置上计算 softmax，
而非在整个词汇表上计算。
"""

import torch


def compute_fkl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "mean",
    temp: float = 4.0,
) -> torch.Tensor:
    """前向 KL 散度：KL(教师 || 学生)。

    KL = sum(教师概率 * (log(教师概率) - log(学生概率)))

    仅在有效（非填充）位置上进行内存高效的计算。
    """
    valid_mask = target.ne(padding_id)
    valid_indices = valid_mask.nonzero(as_tuple=True)

    if valid_indices[0].numel() == 0:
        return student_logits.new_zeros(student_logits.size(0))

    student_logits_valid = student_logits[valid_indices] / temp
    teacher_logits_valid = teacher_logits[valid_indices] / temp

    student_log_probs = torch.log_softmax(student_logits_valid, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits_valid, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits_valid, -1, dtype=torch.float32)

    kl_flat = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1)

    kl = torch.zeros(
        student_logits.size(0), student_logits.size(1),
        dtype=torch.float32, device=student_logits.device,
    )
    kl[valid_indices] = kl_flat

    if reduction == "sum":
        kl = kl.sum(dim=1)
    elif reduction == "mean":
        kl = kl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    return kl * (temp ** 2)


def compute_rkl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "mean",
    temp: float = 1.0,
) -> torch.Tensor:
    """反向 KL 散度：KL(学生 || 教师)。

    KL = sum(学生概率 * (log(学生概率) - log(教师概率)))
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    valid_mask = target.ne(padding_id)
    valid_indices = valid_mask.nonzero(as_tuple=True)

    if valid_indices[0].numel() == 0:
        return logits.new_zeros(logits.size(0))

    logits_flat = logits[valid_indices]
    teacher_logits_flat = teacher_logits[valid_indices]

    probs = torch.softmax(logits_flat, -1, dtype=torch.float32)
    log_probs = torch.log_softmax(logits_flat, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits_flat, -1, dtype=torch.float32)
    kl_flat = (probs * (log_probs - teacher_log_probs)).sum(-1)

    kl = torch.zeros(
        logits.size(0), logits.size(1),
        dtype=torch.float32, device=logits.device,
    )
    kl[valid_indices] = kl_flat

    if reduction == "sum":
        kl = kl.sum(dim=1)
    elif reduction == "mean":
        kl = kl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    return kl


def compute_skewed_fkl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "mean",
    temp: float = 1.0,
    skew_lambda: float = 0.1,
) -> torch.Tensor:
    """偏斜前向 KL：使用混合目标分布。

    mixed_probs = skew_lambda * 教师概率 + (1 - skew_lambda) * 学生概率
    KL = sum(教师概率 * (log(教师概率) - log(mixed_probs)))
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    valid_mask = target.ne(padding_id)
    valid_indices = valid_mask.nonzero(as_tuple=True)

    if valid_indices[0].numel() == 0:
        return logits.new_zeros(logits.size(0))

    logits_flat = logits[valid_indices]
    teacher_logits_flat = teacher_logits[valid_indices]

    probs = torch.softmax(logits_flat, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits_flat, -1, dtype=torch.float32)
    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
    mixed_log_probs = torch.log(mixed_probs)
    teacher_log_probs = torch.log_softmax(teacher_logits_flat, -1, dtype=torch.float32)
    kl_flat = (teacher_probs * (teacher_log_probs - mixed_log_probs)).sum(-1)

    kl = torch.zeros(
        logits.size(0), logits.size(1),
        dtype=torch.float32, device=logits.device,
    )
    kl[valid_indices] = kl_flat

    if reduction == "sum":
        kl = kl.sum(dim=1)
    elif reduction == "mean":
        kl = kl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    return kl


def compute_skewed_rkl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "mean",
    temp: float = 1.0,
    skew_lambda: float = 0.1,
) -> torch.Tensor:
    """偏斜反向 KL：使用混合目标分布。

    mixed_probs = (1 - skew_lambda) * 教师概率 + skew_lambda * 学生概率
    KL = sum(学生概率 * (log(学生概率) - log(mixed_probs)))
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    valid_mask = target.ne(padding_id)
    valid_indices = valid_mask.nonzero(as_tuple=True)

    if valid_indices[0].numel() == 0:
        return logits.new_zeros(logits.size(0))

    logits_flat = logits[valid_indices]
    teacher_logits_flat = teacher_logits[valid_indices]

    probs = torch.softmax(logits_flat, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits_flat, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits_flat, -1, dtype=torch.float32)
    kl_flat = (probs * (log_probs - mixed_log_probs)).sum(-1)

    kl = torch.zeros(
        logits.size(0), logits.size(1),
        dtype=torch.float32, device=logits.device,
    )
    kl[valid_indices] = kl_flat

    if reduction == "sum":
        kl = kl.sum(dim=1)
    elif reduction == "mean":
        kl = kl.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    return kl
