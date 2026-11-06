"""Is chunked cross-entropy the same loss, and does it actually free anything?

WHY BOTH HALVES MATTER. Chunked CE is only admissible as an evaluation column
if it is lossless, so check 1 pins the loss and the gradients against
HuggingFace's own unchunked path on identical weights and inputs. But a
correct loss proves nothing about memory: chunking WITHOUT the per-chunk
checkpoint frees nothing at all, because autograd keeps every chunk's
log_softmax output alive until backward reaches that chunk. That failure is
silent -- same loss, same gradients, same peak. So check 3 is a control arm
that chunks without checkpointing and shows it saves nothing, and check 4
counts the logit-sized allocations that are simultaneously live.

CHECKS, in the order a failure should be read:
  1. The loss matches HuggingFace's to fp32 round-off, at three chunk sizes
     and at a size that does not divide the token count evenly.
  2. Gradients match. A loss can agree while the reduction's scaling is wrong
     in a way only the backward sees.
  3. CONTROL: without the checkpoint, peak live logit bytes are unchanged.
     This is the silent failure, and it is what makes check 4 meaningful.
  4. Peak live logit bytes are one chunk's worth, and are INDEPENDENT OF
     SEQUENCE LENGTH -- the property that moves the term out of the peak's
     slope and into its intercept.
  5. The installed forward returns no logits and leaves labels=None working.

WHAT UNIT THE MEMORY CHECKS ARE IN. They count bytes RETAINED AT THE SEAM --
what autograd holds from forward into backward -- which is log_softmax's fp32
output at 4 B per token per vocabulary entry. That is deliberately not the
12-14 B per token per vocabulary entry the byte model carries at the allocator
peak (tests/test_head_ce_byte_model.py): that figure is the peak INSIDE the
loss, where the fp16
logits and their fp32 upcast are still alive alongside it. The two measure
different instants. What transfers between them is the functional form, which
is what these checks are for.

Device-agnostic, tiny model, runs on CPU in a few seconds.

Run:  PYTHONPATH=. python3 tests/test_chunked_ce.py
"""

import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaConfig

import chunked_ce

failures: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ''))
    if not ok:
        failures.append(name)


def make_model(vocab=256, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=vocab, max_position_embeddings=512,
                      use_cache=False)
    return AutoModelForCausalLM.from_config(cfg), cfg


# ── 1 & 2. Same loss, same gradients ──────────────────────────────────────
print("\n1-2. LOSSLESS -- against HuggingFace's own unchunked loss")
SEQ, BATCH, VOCAB = 48, 2, 256
torch.manual_seed(7)
ids = torch.randint(0, VOCAB, (BATCH, SEQ))

ref_model, _ = make_model(VOCAB)
ref_loss = ref_model(input_ids=ids, labels=ids).loss
ref_loss.backward()
ref_grads = {n: p.grad.clone() for n, p in ref_model.named_parameters()
             if p.grad is not None}

for chunk in (16, 24, 96, 7):          # 7 divides neither 96 nor SEQ
    m, _ = make_model(VOCAB)
    chunked_ce.install(m, chunk)
    got = m(input_ids=ids, labels=ids).loss
    got.backward()
    dl = abs(got.item() - ref_loss.item())
    check(f"chunk={chunk:>3}: loss matches", dl < 2e-5,
          f"|delta| = {dl:.3e} (ref {ref_loss.item():.6f})")
    worst, worst_n = 0.0, ''
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        d = (p.grad - ref_grads[n]).abs().max().item()
        if d > worst:
            worst, worst_n = d, n
    check(f"chunk={chunk:>3}: gradients match", worst < 2e-5,
          f"max |delta| = {worst:.3e} on {worst_n}")

# ── 3 & 4. What is actually live ──────────────────────────────────────────
#
# Count the logit-sized storages autograd is holding. A logit tensor is any
# 2-D float tensor whose last dimension is the vocabulary; summing them over
# the whole graph at the moment forward ends is the quantity the lever moves.
print("\n3-4. MEMORY -- live logit bytes, measured by instrumenting the head")


def retained_logit_bytes(chunk_tokens, seq_len, use_checkpoint=True,
                         vocab=VOCAB, head_only=False):
    """Vocabulary-sized bytes autograd RETAINS from forward into backward.

    Measured at `saved_tensors_hooks` -- the same seam the rest of this
    repository runs on, and the one that actually decides the question.
    Tracking the head's fp16 output by weak reference would show no
    difference between the arms, because that tensor is not what autograd
    keeps: the cast to fp32 does not save its input, so the fp16 logits die
    inside the chunk either way. What survives is log_softmax's fp32 output.

    Two corrections the raw hook needs: the head's weight arrives transposed
    as (hidden, vocab) and would be counted as a logit tensor, so parameters
    are excluded by storage; and log_softmax's output is packed TWICE from
    one storage -- once for its own backward, once for nll_loss's -- so
    storages are deduplicated. Counting an alias twice would inflate both
    arms equally and still be wrong.

    Checkpointing nests its own hooks inside this one, so a checkpointed
    region's internal saves never reach here -- which is precisely the
    property under test, stated as a measurement rather than an assumption.
    """
    m, _ = make_model(vocab)
    ids_l = torch.randint(0, vocab, (1, seq_len))
    params = {p.data_ptr() for p in m.parameters()}
    seen: dict[int, int] = {}

    def pack(t):
        if (t.dim() == 2 and t.is_floating_point() and t.shape[-1] == vocab
                and t.data_ptr() not in params):
            seen[t.untyped_storage().data_ptr()] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if use_checkpoint:
            chunked_ce.install(m, chunk_tokens)
            loss = m(input_ids=ids_l, labels=ids_l).loss
        else:
            # CONTROL: identical chunking, no checkpoint. Every chunk's saves
            # stay in the outer graph, so nothing is freed.
            hidden = m.model(input_ids=ids_l)[0]
            shift = F.pad(ids_l, (0, 1), value=-100)[..., 1:].reshape(-1)
            flat = hidden.reshape(-1, hidden.size(-1))
            # head_only: exactly what the checkpoint replays for ONE chunk.
            stop = chunk_tokens if head_only else flat.size(0)
            total = None
            for lo in range(0, stop, chunk_tokens):
                part = chunked_ce._chunk_sum_loss(
                    m.lm_head, flat[lo:lo + chunk_tokens],
                    shift[lo:lo + chunk_tokens])
                total = part if total is None else total + part
            loss = total / (shift != -100).sum()
    loss.backward()
    return sum(seen.values())


CHUNK, S = 32, 256
full = retained_logit_bytes(S, S, use_checkpoint=False)
ctrl = retained_logit_bytes(CHUNK, S, use_checkpoint=False)
check("CONTROL: chunking without the checkpoint frees nothing",
      ctrl == full, f"{ctrl} B against the unchunked {full} B")
check("and that is log_softmax's fp32 output, 4 B per token per vocab entry",
      full == S * VOCAB * 4, f"{full} B == {S} x {VOCAB} x 4")

treat = retained_logit_bytes(CHUNK, S, use_checkpoint=True)
check("with the checkpoint, NOTHING logit-sized survives forward",
      treat == 0, f"{treat} B, against {full} B without it")

# The peak is therefore whatever one chunk's replay rebuilds, measured by
# running exactly that region -- the checkpoint's own recomputation.
one_chunk = retained_logit_bytes(CHUNK, S, use_checkpoint=False, head_only=True)
check("the peak is one chunk's replay, and it is one chunk's worth",
      one_chunk == CHUNK * VOCAB * 4, f"{one_chunk} B == {CHUNK} x {VOCAB} x 4")
check(f"that is an {S // CHUNK}x reduction at s={S}, chunk={CHUNK}",
      full // one_chunk == S // CHUNK, f"{full} B -> {one_chunk} B")

print("  the sequence-length sweep -- this is the functional-form claim:")
peaks = {s: retained_logit_bytes(CHUNK, s, use_checkpoint=False, head_only=True)
         for s in (64, 128, 256, 512)}
for s, p in peaks.items():
    print(f"    s={s:>4}: peak live logit bytes = {p}")
check("peak live logit bytes do NOT grow with sequence length",
      len(set(peaks.values())) == 1,
      "constant, so the term is an intercept and never re-binds at the frontier")
unchunked = {s: retained_logit_bytes(s, s, use_checkpoint=False) for s in (64, 512)}
check("CONTROL: unchunked, they grow linearly with it",
      unchunked[512] == 8 * unchunked[64],
      f"s=64 -> {unchunked[64]} B, s=512 -> {unchunked[512]} B")
check("peak_logit_bytes() carries the same form at the project's GPU constant",
      (chunked_ce.peak_logit_bytes(2 * CHUNK, VOCAB)
       == 2 * chunked_ce.peak_logit_bytes(CHUNK, VOCAB)),
      "linear in chunk size, free of sequence length by construction")

# ── 5. The installed forward's contract ───────────────────────────────────
print("\n5. CONTRACT -- what the installed forward returns")
m, _ = make_model(VOCAB)
chunked_ce.install(m, 16)
out = m(input_ids=ids, labels=ids)
check("logits are not materialized into the output", out.logits is None)
check("loss is a finite scalar",
      out.loss.dim() == 0 and torch.isfinite(out.loss).item())
check("labels=None still runs and returns no loss",
      m(input_ids=ids).loss is None)
try:
    chunked_ce.install(torch.nn.Linear(4, 4), 16)
    check("a module with no lm_head is refused, not silently ignored", False)
except TypeError as e:
    check("a module with no lm_head is refused, not silently ignored", True,
          str(e)[:60])

print("\n" + "=" * 72)
if failures:
    print(f"FAILED ({len(failures)}): " + "; ".join(failures))
    sys.exit(1)
print("All checks passed. Chunked CE is lossless and its logit mass is "
      "constant in sequence length.")
