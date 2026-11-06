# l40s_followup — the L40S follow-up to 630188 (job 634078)

**What it measures, in one ~7 h exclusive job:**
- **T:** offload's time cost on the L40S, measured cleanly: static/offload pairs at
  x = 32/64/100 with four repeats each. 630188's two-repeat calibration took a noisy pair.
- **RSC, ROC:** a pre-registered capacity prediction at the realistic 3B shape: native
  brackets with FSDP CPU offload, predicted before the run from the capacity model
  fitted on 630188. 630188's registered brackets used nvidia-smi's card total, so its
  model was fitted after the fact; this is its out-of-sample test.
- **RS, RO:** 630188's 3B ladders again (in-sample replication).
- **R:** the one unexplained byte point (3B static x=16, +0.148 MiB), with an allocator snapshot.

All predictions are in the header of `l40s_followup.sbatch`. Every file and output
starts with `l40s_`.

## On ClusterA (login node)

Needs `~/l40s_venv`, built once with `../l40s_venv_setup/l40s_venv_setup.sh`. The job uses
that venv only, and stops before its first arm if the venv fails the launcher check.

```bash
cd <path>/l40s_followup
sbatch -A <account> -p <partition> --qos=<qos> --nodes=1 \
       --exclusive --gres=gpu:l40s:4 --time=24:00:00 l40s_followup.sbatch
./l40s_followup_collect.sh <jobid>   # when done -> l40s_followup_results_<jobid>.tar.gz
```

`code/` is a snapshot of the V100 tree (`./l40s_followup_build.sh`); the reader is `tools/remote_frontier.py --platform l40s`.
