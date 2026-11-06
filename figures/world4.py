"""
World-4 measured inputs shared by figures 6 and 10-14.  Jobs 7743 (W=2/W=1
brackets + time twin) and 7744 (static line and bracket, W=0 ladder,
losslessness).  L=64, chunked loss (c=256), regime B,
fp16, v=32000, b=1, world 4, fused-input, use_cache off.  Rows in
profile_results/runs_4gpu.csv.  x = tokens/1024, MiB.

Every floor below is a MEASURED intercept on the slope given with it (7744
blocks S/T/U).  Each sits exactly 12.00 MiB below the
intercept derived from parameter counts: the gradient pile grows by 3 shards
at step 1 at world 4, not 5.  No floor is a fit to more than its own rungs.
"""

import os

RESULTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "profile_results"))

# Static whole-layer checkpointing, step 1 (binding), 7744 block S.
# Slope 132.320279: the x=16-104 rungs sit on it to 0.007 MiB (their own fit
# gives 132.3201, the harness arithmetic 132.320312).
M_STATIC = 132.320279
C_STATIC = 1476.7989            # through the x=104 top rung on that slope
C_STATIC_STEP0 = C_STATIC - 861.9400   # AdamW 8 B/param / 4 ranks

# Step-1 floor lines per restore depth: (intercept, slope, usable picks).
FLOORS = {
    2: (1860.9001, 40.820279, 61),   # 7743 B2 x=328
    1: (1866.9020, 39.320279, 62),   # 7743 B1 x=340/344 and Z x=328 (spread 0.0005)
    0: (1866.8908, 37.820312, 63),   # 7744 E x=328/356 (spread 0.0000)
}
# Step-0 floors (no optimizer state yet); A-P3 held to 0.01 MiB.
FLOORS_STEP0 = {2: 980.9585, 1: 986.9540, 0: 986.9489}

M_ALL64 = 36.3203               # all-64 analytic slope; depth- and world-invariant

# Brackets: largest fit / smallest fail, same ladder, 4 x-units apart.
X_STATIC_FIT, X_STATIC_FAIL = 104, 108
X_ENGINE_FIT = {2: 328, 1: 344, 0: 356}
X_ENGINE_FAIL = {2: 332, 1: 348, 0: 360}
SPAN = X_ENGINE_FIT[0] / X_STATIC_FIT             # 3.4231
SPAN_BAND = (X_ENGINE_FIT[0] / X_STATIC_FAIL, X_ENGINE_FAIL[0] / X_STATIC_FIT)

# Measured step-1 peaks at the rungs that ran (MiB).
PEAK = {
    ("static", 104): 15238.10791,
    (2, 328): 15249.95166,
    (1, 328): 14763.95313, (1, 340): 15235.79688, (1, 344): 15393.07813,
    (0, 108): 14957.39, (0, 140): 14751.64, (0, 280): 14166.48,
    (0, 328): 14271.95313, (0, 356): 15330.92188,
}
# Step-0 peaks of the FAILING rungs (from the log; they die in step 1 bwd).
FAIL_STEP0 = {2: 14533.29, 1: 14670.42, 0: 14602.26, "static": 14905.45}
FAIL_REQ = {2: 1328.0, 1: 1392.0, 0: 1440.0, "static": 432.0}   # 4x MiB, replayed FFN

# This config's capacity: its largest ACHIEVED fit across the depth brackets
# (a lower bound; an OOM is not an upper bound).
CAPACITY_MIB = 15393.078125     # W=1 x=344

# Below-cap picks at W=0 (step 1): the gate's sequence.
PICKS_W0 = {108: 5, 140: 25, 280: 58, 328: 63, 356: 63}

# Step times, MEAN OVER THE STEP PAIR (s), chunked loss, c=256.
# (arm stdevs: W=2@328 0.27, W=1@328 5.15, W=0@328 3.54)
STEP_S = {
    2: [(328, 2427.791)],
    1: [(328, 2436.455), (340, 2618.958), (344, 2686.762)],
    0: [(108, 255.806), (140, 428.702), (280, 1746.099), (328, 2431.411),
        (356, 2884.731)],
}
X_COMMON = 328                  # the one x all three depths ran (Z arm)
# PCIe share of the engine's own step pair, W=0 (evict + restore time).
PCIE_PCT = [(140, 0.389), (280, 0.450), (328, 0.406), (356, 0.322)]
# Aggregate observed / contended-rate prediction over x=356's 252 transfers.
OBS_OVER_CONTENDED = 0.120

# Run directories (rank-0 profiles) for the event-level figures (6, 13).
RUN_W0 = {
    140: "headline_4gpu/7744-h768L64_E_engine_loose_b14.9769_w0_c256_s143360_n2"
         "_norminput_cacheoff_b1_v32000_fp16_mem_efficient_world4",
    356: "headline_4gpu/7744-h768L64_E_engine_loose_b14.9769_w0_c256_s364544_n2"
         "_norminput_cacheoff_b1_v32000_fp16_mem_efficient_world4",
}
