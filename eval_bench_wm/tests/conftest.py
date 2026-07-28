import sys
from pathlib import Path

# ``eval_bench_wm`` modules are imported as top-level packages (``utils.…``),
# so the benchmark root must be importable regardless of the pytest rootdir.
EVAL_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_BENCH_ROOT))
