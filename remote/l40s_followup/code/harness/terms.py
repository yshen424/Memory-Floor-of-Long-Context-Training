MiB = 2 ** 20

def param_counts(shape):
    h, I, d = shape.hidden, shape.intermediate, shape.head_dim
    p_layer = (h * shape.n_q * d
               + 2 * h * shape.n_kv * d
               + shape.n_q * d * h
               + 3 * h * I
               + 2 * h)
    p_root = 2 * shape.vocab * h + h
    return p_layer, p_root, shape.layers * p_layer + p_root

def slope_terms(run):
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
    return mig, non

def intercept_terms(run, platform, step0=False):
    p_layer, p_root, p_tot = param_counts(run.shape)
    W, sh = run.world, run.shape
    t = {
        'sharded fp32 master (4 B/param/rank)': 0 if run.cpu_offload else 4 * p_tot / W,
        'sharded AdamW state (8 B/param/rank)': 0 if (run.cpu_offload or step0) else 8 * p_tot / W,
        'gathered root unit (fp16)': 2 * p_root,
        'lm_head gradient (fp16, unsharded)': 2 * sh.vocab * sh.hidden,
        'gathered replayed layer (fp16)': 2 * p_layer,
        'second gathered layer (fp16; FSDP backward prefetch)': 2 * p_layer,
        'one MLP weight gradient (fp16 h x I, autograd)': 2 * sh.hidden * sh.intermediate,
        'library workspace': platform.ws_mib * MiB,
    }
    return t

def second_replay_terms(run):
    p_layer, _, _ = param_counts(run.shape)
    return {'third gathered layer + previous layer\'s fp16 weight gradients (1.28)': 4 * p_layer}, \
        {'first boundary consumed': -run.act_bytes * run.shape.hidden}

def optimizer_terms(run, platform):
    _, _, p_tot = param_counts(run.shape)
    if run.cpu_offload:
        return None
    W = run.world
    return {'sharded fp32 master': 4 * p_tot / W,
            'sharded fp32 gradient': 4 * p_tot / W,
            'sharded AdamW state': 8 * p_tot / W,
            'AdamW multi-tensor temporary (fp32; 1.28)': 4 * p_tot / W,
            'library workspace': platform.ws_mib * MiB}, {'int64 ids': 8}

def floor_extra_terms(run, restore_depth):
    p_layer, _, _ = param_counts(run.shape)
    L, W = run.shape.layers, run.world
    pile = L - max(restore_depth, 1) - 2
    return {
        'two more gathered layers (fp16)': 2 * 2 * p_layer,
        f'gradient pile ({pile} fp32 shards)': pile * 4 * p_layer / W,
    }

def backward_end_terms(run, platform):
    if run.cpu_offload:
        return None
    p_layer, p_root, p_tot = param_counts(run.shape)
    W, L = run.world, run.shape.layers
    return {'sharded fp32 master + AdamW': 12 * p_tot / W,
            'reduced fp32 gradient shards, every layer': 4 * L * p_layer / W,
            'root: 6 B per parameter (unnamed)': 6 * p_root,
            'library workspace': platform.ws_mib * MiB}, {'int64 ids': 8}
