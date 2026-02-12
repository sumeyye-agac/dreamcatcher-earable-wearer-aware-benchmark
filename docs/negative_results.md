# Negative Results Log

This file records outcomes that were neutral, negative, or dominated by better settings.
The goal is to preserve decision traceability, not to optimize presentation.

## Entries

| date_utc | experiment_group | hypothesis | outcome | evidence_path | decision |
|---|---|---|---|---|---|
| 2026-02-11T01:07:15Z | KD / TinyCNN | Higher KD alpha (`alpha=0.9`) with `tau=4` should improve student F1. | `p2_kd_tinycnn_a0p9_t4_seed42` ended with `test_f1=76.65%`, below TinyCNN baseline `76.87%` (delta `-0.22pp`). | `results/leaderboard.csv` (`run_name=p2_kd_tinycnn_a0p9_t4_seed42`) | Do not treat high-alpha settings as universally safe; keep per-setting comparison mandatory. |
| 2026-02-10T14:50:57Z | KD / TinyCNN | Lower alpha (`alpha=0.3`, `tau=5`) may be enough for top TinyCNN performance. | Positive but dominated: `test_f1=77.33%` vs best TinyCNN KD `78.87%` (`p2_kd_tinycnn_a0p6_t5_seed42`). | `results/leaderboard.csv` (`run_name=p2_kd_tinycnn_a0p3_t5_seed42`) | Mark as non-preferred for deployment-focused tuning. |
| 2026-02-12T10:40:00Z | KD grid policy | Keeping `alpha=0.0` in final grid adds useful KD signal. | Rejected for final campaign policy: `alpha=0.0` removes distillation term and behaves as CE-only student training. | `experiments/manifest_repro_v1.json` + `docs/decision_log.md` (current pinned policy) | Final pinned KD grid is `alpha=[0.3,0.6,0.9]`, `tau=[3.0,4.0,5.0]`. |
