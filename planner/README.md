# planner — POET's exact program on the base configuration

POET [Patil et al., ICML 2022] solves recomputation and offloading together as one integer program. Here its program runs on a graph of the base configuration (tier-A, chunked loss c=256, fp16) with recomputation fixed to whole layers, and asks for the lowest peak a schedule can reach. POET's source is used unedited, at the commit in `POET_COMMIT.txt` (github.com/ShishirPatil/poet); the graph, the class constraints and the sparse solver are ours.

**Units.** Every number is a slope in MiB per 1024 tokens. POET's graph holds only activation tensors, all proportional to the sequence length, so each optimum is a slope independent of length. The bytes no placement can change (parameters, optimizer state, gathered weights and gradients, the workspace) add to every mode alike and are not in the graph.

## Files

- `scripts/toygraph.py`: the census graph (11 forward and 10 backward operations per layer, tensor sizes from the harness's slope terms), the fixed whole-layer schedule, and an independent simulator of POET's memory accounting that re-scores every solution.
- `scripts/sparse_poet.py`: POET's own objective and constraint generators over decision matrices in which every entry the fixed schedule determines is a constant. Modes:
  - `static`: nothing offloaded;
  - `inclass`: only layer boundaries may be offloaded, and a replay keeps its own copy of its boundary (the paper's class);
  - `wide_replay`: the tensors a replay produces may also be offloaded, between whole replays (`--spread 0`) or between single operations (`--spread 1`).
- `scripts/run_poet.py`, `scripts/sweep.py`: stock POET with the same constraints added after construction, used to validate the sparse solver.
- `scripts/paper_run.py`: the paper's run (about an hour on 4 threads); `--summary` prints the table from the log.
- `scripts/census_traffic.sh`: least-traffic schedules at depths 4 and 16.
- `logs/`: the runs as made. `paper_run.jsonl` and `paper_run.out` (the paper's run), `census_traffic.jsonl` (traffic at depths 4 and 16), `validate_small.jsonl` (stock POET) and `sparse_validate.jsonl` (the sparse solver) on small graphs.

## Results (`logs/paper_run.jsonl`)

| run | minimum peak (MiB per 1024 tokens) |
|---|---|
| static, depth 64 | 132.3203, Table 3's static slope |
| in-class, depths 8, 16, 32, 64 | 37.8203 at every depth, Eq. 4's slope |
| in-class, depth 64, at 37.8203 and at 37.8193 | feasible with 94.5 MiB in and 94.5 out (63 boundaries, never the last); infeasible |
| wide_replay, between whole replays, depth 64 | 34.5547 (8.6% lower) |
| wide_replay, between single operations, depth 64 | 20.0000 |

Traffic at each mode's minimum, in-class against between whole replays: 45.0 against 123.0 MiB at depth 16 (`census_traffic.jsonl`), 93.0 against 247.3 at depth 32 (`paper_run.jsonl`), about 2.7 times. The least-traffic solves of both wide modes at depth 64 did not finish (CBC failed on one and hit its time limit on the other); the minimum peaks above do not depend on them.

**Validation.** On the 18 small configurations both solved (depths 2 and 4, two graph granularities; static, in-class, and every tensor offloadable), the sparse solver's minimum peak equals stock POET's. Every solution in every log is re-scored by `toygraph.simulate` (`sim_peak_of_solution`).

**What the in-class run does and does not show.** The rule that a replay keeps its own copy of its boundary is a constraint in the run, so the run checks that no timing of the other boundaries beats Eq. 4 and that the accounting is right; it does not test that premise. In POET's model transfers are not limited by link bandwidth and host memory is unlimited.

## Running it

```bash
cd planner
git clone https://github.com/ShishirPatil/poet poet_src && git -C poet_src checkout $(cat POET_COMMIT.txt)
python3 -m venv .venv && .venv/bin/pip install pulp==3.3.2 numpy==2.2.6 && .venv/bin/pip install -e poet_src
cd scripts && ../.venv/bin/python paper_run.py            # appends to ../logs/paper_run.jsonl
../.venv/bin/python paper_run.py --summary
```
Python 3.10, PuLP 3.3.2 with its bundled CBC, NumPy 2.2.6, CPU only.
