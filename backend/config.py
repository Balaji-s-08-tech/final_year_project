import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ORIGIN = os.environ.get("VLM_FRONTEND_ORIGIN", "http://localhost:3000")
OUTPUT_ROOT = (PROJECT_ROOT / "checkpoints").resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def approved_roots() -> tuple[Path, ...]:
    candidates = [
        PROJECT_ROOT / "smoke_test_output",
        PROJECT_ROOT / "test_awq_output",
        OUTPUT_ROOT,
    ]
    configured = os.environ.get("VLM_CHECKPOINT_DIRS", "")
    candidates.extend(Path(item).expanduser() for item in configured.split(os.pathsep) if item)
    roots = []
    for candidate in candidates:
        try:
            root = candidate.resolve()
            if root.exists() and root.is_dir() and root.is_relative_to(PROJECT_ROOT):
                roots.append(root)
        except OSError:
            continue
    return tuple(dict.fromkeys(roots))
