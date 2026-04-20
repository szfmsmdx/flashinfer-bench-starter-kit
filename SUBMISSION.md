# GDN Submission Layout

This repository uses the multi-definition layout recommended by the contest FAQ.
Do not add a root-level `config.toml`, because the evaluator scans the repository
root and immediate subdirectories for configs.

Submissions:

- `gdn_decode_qk4_v8_d128_k_last/`
  - `definition = "gdn_decode_qk4_v8_d128_k_last"`
  - `entry_point = "kernel.py::kernel"`
- `gdn_prefill_qk4_v8_d128_k_last/`
  - `definition = "gdn_prefill_qk4_v8_d128_k_last"`
  - `entry_point = "kernels.py::kernel"`

Useful local checks:

```bash
python3 scripts/pack_solution.py
python3 scripts/run_local.py
```

If `flashinfer_bench` is not installed locally, run the official checks in an
environment with the starter-kit dependencies installed.

Submission steps:

```bash
git status
git add .
git commit -m "Submit GDN decode and prefill kernels"
git push origin main
git tag submission-v1
git push origin submission-v1
```

For a private repository, grant read access to `flashinfer-bot` before sharing
the repository URL with the organizers.
