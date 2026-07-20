# Current Evaluation Process Audit

Audit time: 2026-07-18 UTC. Repository: `/workspace/RAVEN`; HEAD
`b222bb86b0266a9e55fb8e25b1134394ac7494cf`.

The requested `/workspace/raven_repro/logs` path does not exist. The repository-local
`/workspace/RAVEN/raven_repro/logs` path also does not exist. A full process search for
`raven|RAVEN|eval|watermark` returned no live process. Therefore no `SIGTERM` or
`SIGKILL` was sent and no generation, training, or unrelated job was stopped.

| PID | PPID | Start / elapsed | Command / cwd | GPU | Class | Status and evidence |
| ---: | ---: | --- | --- | --- | --- | --- |
| 476553 | unavailable | created 2026-07-17 01:47 UTC | Historical entry was `raven_repro/scripts/run_diffusiondb_chain_after_clean.py`; cwd `/workspace/RAVEN`; exact `/proc` command unavailable | workers used GPU 2/3 for generation, 1/2 for attacks | paired data generation then historical RAVEN attack/detector/quality orchestrator | stale PID file; `/proc/476553` absent; run state says completed 2026-07-17T14:21:52Z |
| 685328 | unavailable | created 2026-07-17 17:48 UTC | `quality_decomposition_experiment.py` no-color evaluation inferred from preserved output/log name; exact `/proc` command unavailable | historical log/config only | quality evaluation | stale PID file; `/proc/685328` absent; output completed 2026-07-17T20:33:12Z |

Historical worker PIDs recorded by immutable run state were generation 476711/476712
(GPU 2/3) and attacked-watermarked/attacked-clean 585854/585856 (GPU 1/2). All were
already completed. The historical chain directly invoked paired generation, manifest
merge, P1 plan, attacked-watermarked and attacked-clean workers, NFPA synchronization,
then quality/detector metrics. It wrote below
`outputs/raven_paired_formal/diffusiondb/20260717T014700Z`; its root logs remain at
`orchestrator.log`, `run.log`, and `logs/`. Derived evaluation directories were
quarantined at `outputs/legacy_invalid/20260718T072817Z/` while generated clean and
watermarked images were not moved.

Classification: the completed paired image generation is source data and was preserved.
The derived P1/NFPA/quality chain is pre-audit evaluation and is not the new formal path.
No waiter was live.
