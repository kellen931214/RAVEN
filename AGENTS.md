## Required Skills

Before reviewing, modifying, debugging, testing, rerunning, or evaluating this repository, you must first read and follow:

`skills/experiment-integrity/SKILL.md`

You must also read the entries in `DEBUG_CHANGELOG.md` that are relevant to the current task.

If the task involves any of the following:

* watermark generation or embedding
* watermark removal attacks
* clean and watermarked sample pairing
* attacked-clean and attacked-watermarked sample pairing
* Tree-Ring or other watermark detectors
* thresholds, TPR, FPR, ROC-AUC, or attack success rate
* watermark evaluation metrics such as CLIP, FID, PSNR, or SSIM
* RAVEN or another watermark removal pipeline

you must also read and follow:

`skills/watermark-evaluation/SKILL.md`

Do not begin implementation until all applicable instructions and checks have been completed.

## GPU Failure Hard Stop

This project runs inside Docker, where GPU access may occasionally disappear because of container runtime, driver, CUDA, or NVML failures.

For every task that requires a GPU, perform a GPU availability check before loading models, starting workers, resuming experiments, or running evaluation.

Treat any of the following as a GPU failure:

* `nvidia-smi` returns a non-zero exit code.
* `nvidia-smi` reports an NVML initialization error.
* `torch.cuda.is_available()` is `False`.
* `torch.cuda.device_count()` is `0`.
* The requested CUDA device cannot be initialized.
* A basic CUDA tensor allocation or kernel execution fails for a reason other
  than an isolated CUDA out-of-memory condition handled below.
* CUDA reports driver, device, initialization, illegal access, or device-lost errors.
* The GPU disappears during an experiment.

When a GPU failure is detected:

1. Stop the current task immediately.
2. Do not continue modifying code based on incomplete GPU results.
3. Do not retry repeatedly.
4. Do not switch GPUs automatically, except for the bounded CUDA OOM fallback
   defined below.
5. Do not use CPU execution as a substitute.
6. Do not use historical outputs as a substitute for the missing run.
7. Do not launch later stages, dependent experiments, or full evaluation.
8. Preserve all existing outputs, logs, manifests, and source files.
9. Report the exact failed command and full error.
10. End the current Codex task with:

`STOPPED: GPU unavailable inside Docker. No further work was performed.`

Only resume GPU work after the user explicitly confirms that Docker GPU access has been restored.

## CUDA OOM GPU Fallback

A CUDA out-of-memory error is distinct from the Docker, driver, CUDA
initialization, NVML, illegal-access, or device-lost failures covered by the
hard stop above. When an evaluation or other GPU task fails specifically with
CUDA OOM:

1. Stop the failed process and preserve its complete command, error log,
   partial output root, manifests, and source files.
2. Do not kill, pause, or modify processes owned by other work, and do not try
   to repair Docker, NVML, the driver, or GPU mounts.
3. Query `nvidia-smi` once to identify candidate GPUs. A candidate must have no
   active compute process, low utilization, and enough free memory for the
   failed workload.
4. Check candidates in descending free-memory order. For each candidate, set
   `CUDA_VISIBLE_DEVICES` explicitly and require PyTorch device discovery,
   device initialization, a basic CUDA tensor allocation, and a basic kernel
   execution to succeed.
5. Select the first candidate that passes all checks. Record both its physical
   GPU index and the process-visible CUDA index.
6. Retry the failed workflow once on that selected GPU, using a new timestamped
   output root. Never resume or reuse the partial output/cache from the OOM
   attempt.
7. Do not fall back to CPU and do not repeatedly retry the same GPU.
8. If no candidate GPU passes, stop the Codex task, report every inspected GPU
   and the exact failure, and end with:

`STOPPED: No idle usable GPU remained after CUDA OOM. No further work was performed.`

## Mandatory RAVEN Experiment Skills

For every watermark generation task, read and follow:

- .agents/skills/raven-shared-clean/SKILL.md
- skills/watermark-evaluation/SKILL.md

For every RAVEN attack or evaluation task, read and follow:

- .agents/skills/raven-attack-artifacts/SKILL.md
- .agents/skills/raven-experiment-naming/SKILL.md
- .agents/skills/raven-experiment-table/SKILL.md
- skills/watermark-evaluation/SKILL.md

After a generation, attack or evaluation experiment finishes, use:

python experiments/update_experiment_table.py --run-root <completed-run-root>

Do not manually transcribe metrics.

Do not infer metrics from logs.

Do not autonomously monitor detached jobs.

For Gaussian Shading, report bit_accuracy and official-threshold detection
rates. Do not relabel GS results as TPR@1%FPR unless a separate empirical
clean-negative 1%-FPR calibration was actually performed.

The GS bit-accuracy family and the Tree-Ring TPR family are examples, not the
complete set. Every watermark method reports its own detector metric, its own
score direction, its own threshold family and its own detection-rate
definition, and every method must satisfy these same rules. A method without a
registered method-specific extractor fails closed instead of being coerced into
another method's schema.
