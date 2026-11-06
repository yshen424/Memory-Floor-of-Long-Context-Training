"""The byte model, term by term.

Slope terms are bytes per token; divide by 1024 for MiB per x (x = tokens/1024).
Intercept terms are bytes. Every term names what it is."""
MiB = 2 ** 20


def param_counts(shape):
    h, I, d = shape.hidden, shape.intermediate, shape.head_dim
    p_layer = (h * shape.n_q * d          # q_proj
               + 2 * h * shape.n_kv * d   # k_proj, v_proj
               + shape.n_q * d * h        # o_proj
               + 3 * h * I                # gate, up, down
               + 2 * h)                   # two RMSNorm weights
    p_root = 2 * shape.vocab * h + h      # embedding, lm_head, final norm
    return p_layer, p_root, shape.layers * p_layer + p_root


def slope_terms(run):
    """Per-token bytes at the peak operation (the MLP multiply's backward in the
    first replayed layer). Returns (migratable, non_migratable) dicts."""
    sh, s = run.shape, run.act_bytes
    h, I, d = sh.hidden, sh.intermediate, sh.head_dim
    kv_w = h if run.repeat_kv else sh.n_kv * d
    mig = {'boundaries': sh.layers * s * h}
    non = {
        'mlp: gate, up, act saved + 3 gradients': 6 * s * I,
        'norm outputs (2)': 2 * s * h,
        'query': s * sh.n_q * d,
        'key, value saved': 2 * s * kv_w,
        'attention output': s * sh.n_q * d,
        'residual sum': s * h,
        'hidden gradient': s * h,
        'rotary cos, sin': 2 * s * d,
        'log-sum-exp (fp32)': 4 * sh.n_q,
        'int64 ids (2)': 16,
        'rmsnorm rstd (2, fp32)': 8,
    }
    if run.attn == 'flash':
        # 633912 (A30), POST HOC: every flash arm sits exactly one query-width
        # fp16 plane above its mem-efficient twin, static and offload, fp16 and bf16, with or
        # without native GQA (so not k/v); same peak instant. Storage unnamed.
        non['flash: one more query-width plane (unnamed)'] = s * sh.n_q * d
    return mig, non


def intercept_terms(run, platform, step0=False):
    """Bytes at the static peak: first replayed layer, no gradient pile.
    step0=True: the first step, before AdamW's state exists (it is allocated at
    the first optimizer step), so 8 B/param/rank is absent (tests/test_harness_step0.py)."""
    p_layer, p_root, p_tot = param_counts(run.shape)
    W, sh = run.world, run.shape
    t = {
        'sharded fp32 master (4 B/param/rank)': 0 if run.cpu_offload else 4 * p_tot / W,
        'sharded AdamW state (8 B/param/rank)': 0 if (run.cpu_offload or step0) else 8 * p_tot / W,
        'gathered root unit (fp16)': 2 * p_root,
        'lm_head gradient (fp16, unsharded)': 2 * sh.vocab * sh.hidden,
        'gathered replayed layer (fp16)': 2 * p_layer,
        # Post hoc on 7761; out of sample on 7763 and named by its snapshots.
        'second gathered layer (fp16; FSDP backward prefetch)': 2 * p_layer,
        'one MLP weight gradient (fp16 h x I, autograd)': 2 * sh.hidden * sh.intermediate,
        'library workspace': platform.ws_mib * MiB,
    }
    return t


def second_replay_terms(run):
    """The static peak's other instant: the backward replay of layer L-2. One boundary has been consumed and 4 B/param of a layer more is
    live. Returns (extra intercept bytes, extra per-token bytes)."""
    p_layer, _, _ = param_counts(run.shape)
    return {'third gathered layer + previous layer\'s fp16 weight gradients': 4 * p_layer}, \
        {'first boundary consumed': -run.act_bytes * run.shape.hidden}


def deep_replay_terms(run):
    """The static peak's fifth instant (634078; POST HOC, no free
    parameter): below the second replay, each layer the backward descends frees one
    boundary (act_bytes * h per token) and adds one reduced fp32 gradient shard
    (4 B * p_layer / W). Where the shard is larger (x below 2 p_layer / (W h 1024))
    the peak sits at the deepest replay that still prefetches, L - 3 layers below the
    second replay (the snapshot holds 25 shards at L = 28). Returns the extra bytes on
    top of the second-replay instant: (intercept, per token). Absent under FSDP CPU
    offload (gradient shards live on the host)."""
    p_layer, _, _ = param_counts(run.shape)
    m = run.shape.layers - 3
    return {f'{m} reduced fp32 gradient shards': m * 4 * p_layer / run.world}, \
        {f'{m} boundaries consumed': -m * run.act_bytes * run.shape.hidden}


def optimizer_terms(run, platform):
    """The optimizer-step instant (post hoc): no activations, so it
    binds only at short x or under a deep floor. Absent under FSDP CPU offload
    (the optimizer runs on the host). Returns (intercept bytes, per-token bytes)."""
    _, _, p_tot = param_counts(run.shape)
    if run.cpu_offload:
        return None
    W = run.world
    return {'sharded fp32 master': 4 * p_tot / W,
            'sharded fp32 gradient': 4 * p_tot / W,
            'sharded AdamW state': 8 * p_tot / W,
            'AdamW multi-tensor temporary (fp32; 1.28)': 4 * p_tot / W,
            'library workspace': platform.ws_mib * MiB}, {'int64 ids': 8}


def floor_extra_terms(run, restore_depth, step0=False):
    """Extra bytes at the relocated peak once the pick cap binds. From step 1 on
    the floor holds (W - 1) more shard-sized units, 4 B·p_layer·(1 - 1/W): the
    count measured at world 4 and predicted exactly at 2 and 3 (7765); storage
    named at world 2 only."""
    p_layer, _, _ = param_counts(run.shape)
    L, W = run.shape.layers, run.world
    pile = L - max(restore_depth, 1) - 2
    t = {
        'two more gathered layers (fp16)': 2 * 2 * p_layer,
        f'gradient pile ({pile} fp32 shards)': pile * 4 * p_layer / W,
    }
    if step0:
        t['step 0: (W - 1) shard units not yet live'] = -(W - 1) * 4 * p_layer / W
    return t


def backward_end_terms(run, platform, step0=False):
    """The end-of-backward instant, after layer 0: sharded
    master + AdamW, every layer's reduced fp32 gradient shard, 6 B per root
    parameter (plausibly the gathered fp16 root plus a 4 B root gradient; storage
    unnamed). Found post hoc on 7764 (3B, L=4); predicted O1 x=4 of 7763 to
    0.002 MiB. Binds only at short x with a large root. Not derived under FSDP
    CPU offload. Returns (intercept bytes, per-token bytes)."""
    if run.cpu_offload:
        return None
    p_layer, p_root, p_tot = param_counts(run.shape)
    W, L = run.world, run.shape.layers
    return {'sharded fp32 master + AdamW (no AdamW in step 0)': (4 if step0 else 12) * p_tot / W,
            'reduced fp32 gradient shards, every layer': 4 * L * p_layer / W,
            'root: 6 B per parameter (unnamed)': 6 * p_root,
            'library workspace': platform.ws_mib * MiB}, {'int64 ids': 8}
