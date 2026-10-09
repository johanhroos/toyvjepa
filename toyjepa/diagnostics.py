"""Per-layer training diagnostics: attention logits, attention entropy, residual stream and gradient norms."""

import math

import torch
import torch.nn as nn

from toyjepa.datasets import Batch
from toyjepa.eval import prepare_inputs

# Query rows per fp32 logit block of one sample; bounds memory for any clip length
QUERY_BLOCK = 1024


def layer_names(student_enc: nn.Module, predictor: nn.Module) -> list[str]:
    return [f"enc_{i:02d}" for i in range(len(student_enc.blocks))] + [
        f"pred_{i:02d}" for i in range(len(predictor.blocks))
    ]


def _running_max(current: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    return value if current is None else torch.maximum(current, value)


class _LayerStats:
    """Per-layer accumulators."""

    def __init__(self, num_registers: int) -> None:
        self.max_logit: torch.Tensor | None = None
        self.entropy_sum: torch.Tensor | None = None  # per head
        self.num_queries = 0
        self.num_registers = num_registers  # appended after the patch tokens
        self.resid_rms_patch: list[torch.Tensor] = []  # per patch token
        self.resid_max_patch: torch.Tensor | None = None
        self.resid_max_register: torch.Tensor | None = None

    @torch.no_grad()
    def add_residual(self, x: torch.Tensor) -> None:
        num_patches = x.shape[1] - self.num_registers
        patches = x[:, :num_patches].float()
        self.resid_rms_patch.append(patches.square().mean(dim=-1).sqrt().flatten())
        self.resid_max_patch = _running_max(self.resid_max_patch, patches.abs().amax())
        if self.num_registers:
            self.resid_max_register = _running_max(self.resid_max_register, x[:, num_patches:].float().abs().amax())

    @torch.no_grad()
    def add_attention(self, q: torch.Tensor, k: torch.Tensor) -> None:
        log_num_keys = math.log(k.shape[-2])
        with torch.autocast(device_type=q.device.type, enabled=False):
            for i in range(q.shape[0]):
                keys = k[i].float().transpose(-1, -2)  # H d N
                for j in range(0, q.shape[2], QUERY_BLOCK):
                    logits = (q[i, :, j : j + QUERY_BLOCK].float() @ keys).mul_(q.shape[-1] ** -0.5)  # H Q N
                    self.max_logit = _running_max(self.max_logit, logits.amax())
                    log_p = logits.log_softmax(dim=-1)
                    entropy = -(log_p.exp() * log_p).sum(dim=-1).div_(log_num_keys)  # H Q, in [0, 1]
                    head_sum = entropy.sum(dim=1)
                    self.entropy_sum = head_sum if self.entropy_sum is None else self.entropy_sum + head_sum
                    self.num_queries += entropy.shape[1]

    def min_head_entropy(self) -> float:
        assert self.entropy_sum is not None, "no attention recorded"
        return (self.entropy_sum / self.num_queries).min().item()

    def median_resid_rms_patch(self) -> float:
        return torch.cat(self.resid_rms_patch).median().item()


@torch.no_grad()
def forward_diagnostics(
    batches: list[Batch],
    *,
    student_enc: nn.Module,
    predictor: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Per-layer attention and residual-stream statistics of the student encoder and predictor.

    Runs the training forward pass (student on the context tokens, then the predictor) on fixed
    batches and returns, keyed `<metric>/<layer>`:
      attn_max_logit:    largest pre-softmax attention logit
      attn_min_entropy:  lowest per-head attention entropy, averaged over queries and normalized
                         by log(#keys), so 1 is uniform attention and 0 is one-hot
      resid_rms_patch:     median per-token RMS of the patch tokens in the residual stream after
                           the block, its typical scale
      resid_max_patch:     largest |x| over those patch tokens
      resid_max_register:  the same over the register tokens (encoder blocks only), which carry
                           attention-sink activations far above the patch tokens'

    The pass runs under torch.compiler.set_stance("force_eager") with temporary hooks, so the hooks
    never enter the compiled blocks.
    """
    blocks = [*student_enc.blocks, *predictor.blocks]
    names = layer_names(student_enc, predictor)
    # the predictor drops the encoder's register tokens, so only encoder blocks see them
    num_registers = 0 if student_enc.register_tokens is None else student_enc.register_tokens.shape[1]
    stats = [_LayerStats(num_registers) for _ in student_enc.blocks] + [_LayerStats(0) for _ in predictor.blocks]
    handles = []
    for blk, layer in zip(blocks, stats, strict=True):

        def attention_hook(module, args, layer=layer):
            q, k, _ = module.project_qkv(*args)
            layer.add_attention(q, k)

        def residual_hook(module, args, output, layer=layer):
            layer.add_residual(output)

        handles.append(blk.attn.register_forward_pre_hook(attention_hook))
        handles.append(blk.register_forward_hook(residual_hook))

    student_enc.eval(), predictor.eval()
    try:
        with torch.compiler.set_stance("force_eager"), torch.autocast(device_type=device.type, dtype=dtype):
            for batch in batches:
                context_ids = batch["context_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                h_student = student_enc(prepare_inputs(batch["inputs"], device), ids=context_ids)
                predictor(
                    h_student,
                    mask_policy=batch["mask_policy"],
                    context_ids=context_ids,
                    target_ids=target_ids,
                    grid_size=tuple(batch["tokenized_shape"]),
                )
    finally:
        for handle in handles:
            handle.remove()
        student_enc.train(), predictor.train()

    results = {}
    for name, layer in zip(names, stats, strict=True):
        results[f"attn_max_logit/{name}"] = layer.max_logit.item()
        results[f"attn_min_entropy/{name}"] = layer.min_head_entropy()
        results[f"resid_rms_patch/{name}"] = layer.median_resid_rms_patch()
        results[f"resid_max_patch/{name}"] = layer.resid_max_patch.item()
        if layer.num_registers:
            results[f"resid_max_register/{name}"] = layer.resid_max_register.item()
    return results


def layer_lines(stats: dict[str, float], fmt: str = "7.4g") -> list[str]:
    """Format `<metric>/<layer>` stats as one line per module and metric, blocks in order.

    Parameters outside the blocks (`enc_other`, `pred_other`) come last, after a bar. A metric
    that a module doesn't report (registers in the predictor) gets no line.
    """
    lines = []
    metrics = list(dict.fromkeys(key.split("/")[0] for key in stats))
    width = max(map(len, metrics), default=0)
    for prefix, module in (("enc", "encoder"), ("pred", "predictor")):
        for metric in metrics:
            blocks = sorted(key for key in stats if key.startswith(f"{metric}/{prefix}_") and key[-1].isdigit())
            other = f"{metric}/{prefix}_other"
            if not blocks and other not in stats:
                continue
            line = f"{module:<9s} {metric:<{width}s} " + " ".join(format(stats[key], fmt) for key in blocks)
            if other in stats:
                line += f"  | other {stats[other]:{fmt}}"
            lines.append(line)
    return lines


def grad_norm_groups(student_enc: nn.Module, predictor: nn.Module, params: list[nn.Parameter]) -> list[str]:
    """Name the gradient-norm group of each parameter: its block (`enc_03`, `pred_01`) or `enc_other`/`pred_other`."""
    group_of = {}
    for prefix, model in (("enc", student_enc), ("pred", predictor)):
        for param in model.parameters():
            group_of[id(param)] = f"{prefix}_other"
        for i, blk in enumerate(model.blocks):
            for param in blk.parameters():
                group_of[id(param)] = f"{prefix}_{i:02d}"
    return [group_of[id(param)] for param in params]


class GradNormTracker:
    """Gradient norms per group and overall, with their maximum since the last reset.

    `update` runs every optimizer step without a host sync, so short bursts between logging
    steps still show up in the window maximum.
    """

    def __init__(self, params: list[nn.Parameter], groups: list[str], device: torch.device) -> None:
        self.params = params
        self.names = ["all", *sorted(set(groups))]
        index = {name: i for i, name in enumerate(self.names[1:])}
        self.group_index = torch.tensor([index[group] for group in groups], device=device)
        self.zero = torch.zeros(1, device=device)
        self.current = torch.zeros(len(self.names), device=device)
        self.window_max = torch.zeros(len(self.names), device=device)

    @torch.no_grad()
    def update(self) -> None:
        grads = [param.grad if param.grad is not None else self.zero for param in self.params]
        squares = torch.stack(torch._foreach_norm(grads)).float().square()
        groups = torch.zeros(len(self.names) - 1, device=squares.device).index_add_(0, self.group_index, squares)
        self.current = torch.cat([groups.sum().unsqueeze(0), groups]).sqrt()
        torch.maximum(self.window_max, self.current, out=self.window_max)

    def total_norm(self) -> float:
        norm = self.current[0].item()
        if not math.isfinite(norm):
            raise RuntimeError(f"The total norm of gradients is non-finite: {norm}")
        return norm

    def pop_window_max(self) -> dict[str, float]:
        values = dict(zip(self.names, self.window_max.tolist(), strict=True))
        self.window_max.zero_()
        return values
