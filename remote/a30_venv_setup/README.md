# a30_venv_setup — the one A30 environment on ClusterA

Builds `~/a30_venv` once. Every A30 job (`a30_frontier`, `a30_followup`, and later
ones) uses it and nothing else, and stops before its first arm if `torchrun` does
not start there. `A30_VENV=<path>` overrides it for a single job.

Pins match `~/l40s_venv` and the versions 633912 ran: python 3.12 (module
`python312`), torch 2.6.0+cu124, transformers 4.50.0. The two venvs are separate
so that rebuilding one never changes the other GPU's jobs.

```bash
# on the ClusterA login node
cd remote/a30_venv_setup                     # wherever you unpacked it
bash a30_venv_setup.sh                       # ~5-10 min, no GPU needed
cat ~/a30_venv/VERSIONS                      # send this back
```
The script refuses to overwrite an existing `~/a30_venv`. Never move a working
venv: its scripts carry their absolute path.
