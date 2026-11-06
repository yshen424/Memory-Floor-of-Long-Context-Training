# a30_followup — two A30 follow-up jobs to 633912 (job A 642029, job B 642032)

**Job A, `a30f_tax.sbatch` (~6 h): why is stock offload FASTER than static at long
sequence on Ampere?** 633912 measured offload − static = −1.07 s at x=100 and −3.59 s
at x=150 (both repeats). Four candidate causes, one block each: a sweep with
per-step allocator counters and 1 Hz GPU clock/power logging on every arm (S); the
allocator's expandable segments turned OFF, which every earlier job ran ON (E);
offload carrying an idle ballast equal to static's extra resident memory (G); a
profiled pair at x=150 (Q). Plus M, two allocator snapshots at x=24 (flash vs
memory-efficient) to name the tensor behind flash's +1.5 MiB/x plane.

**Job B, `a30f_native.sbatch` (~5 h expected, ≤ 23.2 h by its guard): the native
24 GB stock-offload bracket**, predicted [564, 568) (range of largest fit 561–570),
so the A30 span is measured natively. About 56 GB of pinned host memory per rank
(~44% of the node). A rung starts only if its pinned total is predicted ≤ 55% of
host RAM, and a watchdog kills any arm when host MemAvailable falls below 30%:
the node never drains. A rung starts only if it is predicted to end by 23.2 h.

The two jobs are independent; each fits one 24 h job. Predictions are in each
script's header. Every file and output here starts with `a30f_`.

## Step 0: on the local V100 cluster (~75–100 min)

```bash
cd remote/a30_followup
./a30f_build.sh                      # refresh code/ from the tree, pack ../a30_followup.tar.gz
sbatch a30f_v100_smoke.sbatch        # runs both jobs' smoke branches on 4 V100 nodes
```
Check `a30f_out/v100smoke_tax_<jobid>/a30f_summary.txt` and
`a30f_out/v100smoke_native_<jobid>/a30f_summary.txt` against the smoke's S-P1..S-P7
before copying the package to ClusterA.

## On ClusterA (login node, inside this directory)

The venv is `~/a30_venv`, built once with `../a30_venv_setup/a30_venv_setup.sh`
(same pins as `~/l40s_venv`). Each job checks that its `torchrun` starts before
running any arm.

```bash
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:a30:4 --time=24:00:00 a30f_tax.sbatch
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:a30:4 --time=24:00:00 a30f_native.sbatch
# when each finishes:
./a30f_collect.sh tax <jobid>        # -> a30f_tax_results_<jobid>.tar.gz
./a30f_collect.sh native <jobid>     # -> a30f_native_results_<jobid>.tar.gz
```
Traces and raw snapshots stay on the cluster unless asked for (`... <jobid> full`).

## What runs

| job | block | what | answers |
|---|---|---|---|
| A | S | static/offload at x = 80, 100, 120, 150 (150 twice), ABBA | where the sign flips; counters and clocks for every hypothesis |
| A | G | offload + 13.8 GB (x=150) / 9.1 GB (x=100) idle ballast | does static's footprint alone slow it? |
| A | E | static/offload at x = 100, 150, expandable segments OFF | is it the allocator's page mapping? |
| A | Q | profiled static/offload pair at x=150 | is the gap busy (kernels) or idle time? |
| A | M | allocator snapshots, flash vs memory-efficient, x=24 | which tensor is the flash plane? |
| B | NO | stock offload, native 24 GB, from x=564 in steps of 4 | the native A30 span |

`code/` is a snapshot of the V100 tree.
