"""Chunked cross-entropy over the token dimension.

WHAT IT REMOVES. HuggingFace's causal-LM loss materializes the whole logit
chain at once: `lm_head(hidden)` in fp16, `.float()` again in fp32, and
log_softmax's fp32 output, all live simultaneously, for 12 B per token per
vocabulary entry rising to 14 B once the backward adds its gradient. At the
tier-A shape that is 437.50 MiB per 1024 tokens against 3.27 for everything
else outside the layer stack --- roughly 99% of the denominator every span is
divided by.

WHY FIXED CHUNK SIZE RATHER THAN A FIXED CHUNK COUNT. A fixed count n leaves
the term at 437.50/n MiB per 1024 tokens --- still proportional to sequence
length, so it re-binds at exactly the frontier the paper measures. A fixed
number of tokens per chunk makes one chunk's logit chain a CONSTANT, moving
the term out of the peak's slope and into its intercept. Same move as the
whole-layer recompute unit: change the functional form, not a constant in it.

WHY THE PER-CHUNK CHECKPOINT IS NOT OPTIONAL. Chunking alone frees nothing.
Autograd keeps each chunk's log_softmax output alive until backward reaches
that chunk, so all chunks are resident at once and the peak is unmoved.
Checkpointing each chunk is what drops them at forward and replays them one
at a time, and it costs exactly one extra head forward pass per step.

LOSSLESS, UP TO SUMMATION ORDER. Every chunk uses the same fp32 upcast and the
same kernel as the unchunked path; the only difference is that the reduction
is a sum of per-chunk sums. `tests/test_chunked_ce.py` checks agreement
against HuggingFace's own loss on the same weights and inputs.
"""

import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _chunk_sum_loss(lm_head, hidden_chunk, labels_chunk):
    """One chunk's head + cross-entropy, as an unnormalized sum.

    Runs inside the checkpoint, so everything it allocates --- the fp16
    logits, the fp32 upcast, log_softmax's output --- is dropped at the end
    of forward and rebuilt one chunk at a time during backward.
    """
    logits = lm_head(hidden_chunk).float()
    return F.cross_entropy(logits, labels_chunk,
                           ignore_index=IGNORE_INDEX, reduction='sum')


def chunked_causal_lm_loss(model, hidden_states, labels, chunk_tokens):
    """Mean cross-entropy, computed `chunk_tokens` tokens at a time.

    Reproduces HuggingFace's ForCausalLMLoss exactly: labels are shifted so
    token t predicts t+1, the final position has no target, and the reduction
    divides by the number of non-ignored targets rather than by the token
    count.
    """
    # Shift: pad on the right with the ignore index, then drop position 0.
    shift_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
    flat_labels = shift_labels.reshape(-1).to(flat_hidden.device)

    n_valid = (flat_labels != IGNORE_INDEX).sum()
    total = None
    for lo in range(0, flat_hidden.size(0), chunk_tokens):
        hi = min(lo + chunk_tokens, flat_hidden.size(0))
        part = checkpoint(_chunk_sum_loss, model.lm_head,
                          flat_hidden[lo:hi], flat_labels[lo:hi],
                          use_reentrant=False)
        total = part if total is None else total + part
    return total / n_valid.clamp(min=1).to(total.dtype)


def _forward(self, input_ids=None, labels=None, **kwargs):
    """Drop-in replacement for LlamaForCausalLM.forward.

    Bound onto the instance BEFORE FSDP wraps it, so it runs inside the root
    FSDP unit's forward and `self.lm_head`'s parameters are gathered. Returns
    the same object the driver reads, with `logits=None` --- materializing
    them is the whole thing this avoids.
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    kwargs.pop('logits_to_keep', None)
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    loss = (None if labels is None else chunked_causal_lm_loss(
        self, hidden_states, labels, self._chunked_ce_tokens))
    return CausalLMOutputWithPast(
        loss=loss, logits=None,
        past_key_values=getattr(outputs, 'past_key_values', None),
        hidden_states=getattr(outputs, 'hidden_states', None),
        attentions=getattr(outputs, 'attentions', None))


def install(model, chunk_tokens: int):
    """Swap the causal-LM forward for the chunked one. Call before FSDP."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if not hasattr(model, 'lm_head'):
        raise TypeError(
            f"chunked CE needs a causal-LM head; {type(model).__name__} has "
            "no .lm_head")
    model._chunked_ce_tokens = int(chunk_tokens)
    model.forward = types.MethodType(_forward, model)
    return model


def peak_logit_bytes(chunk_tokens: int, vocab_size: int) -> int:
    """Bytes in the largest logit chain that is ever simultaneously live.

    12 B per token per vocabulary entry (375.00 MiB per 1024 chunk tokens at
    v=32000). That is the forward-resident constant, not the
    unchunked path's 14 B: every chunk here is checkpointed, so the chain is
    replayed rather than held alongside its gradient. Independent of sequence
    length by construction --- that is the point of the lever.
    """
    return chunk_tokens * vocab_size * 12
