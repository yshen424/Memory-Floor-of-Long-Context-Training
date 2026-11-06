import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100

def _chunk_sum_loss(lm_head, hidden_chunk, labels_chunk):
    logits = lm_head(hidden_chunk).float()
    return F.cross_entropy(logits, labels_chunk,
                           ignore_index=IGNORE_INDEX, reduction='sum')

def chunked_causal_lm_loss(model, hidden_states, labels, chunk_tokens):
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
    return chunk_tokens * vocab_size * 12
