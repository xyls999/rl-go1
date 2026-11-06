"""Purpose-tagged audit of fresh V1330 training and diagnostic seeds."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import secrets

from contracts_v1330 import FORBIDDEN_SEEDS, MAX_SEED, MIN_SEED, require

TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".toml", ".py", ".sh", ".md",
                           ".receipt", ".log", ".txt"})
SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})
PURPOSES = frozenset({"training", "diagnostic-evaluation"})


def _token_pattern(seed: int) -> re.Pattern[str]:
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED,
            "V1330 seed must be in the approved range")
    return re.compile(r"(?<![0-9])" + re.escape(str(seed)) + r"(?![0-9])")


def scan_seed(seed: int, roots: list[Path],
              ignore_paths: list[Path] | None = None) -> tuple[list[str], int, int]:
    pattern = _token_pattern(seed)
    ignored = {path.resolve(strict=False) for path in (ignore_paths or [])}
    hits: list[str] = []
    files_scanned = 0
    bytes_scanned = 0
    for root in roots:
        require(root.is_dir() and not root.is_symlink(),
                "V1330 seed audit root missing/symlinked: " + str(root))
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or path.resolve(strict=False) in ignored:
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if pattern.search(path.name):
                hits.append(str(path))
            raw = path.read_bytes()
            files_scanned += 1
            bytes_scanned += len(raw)
            if pattern.search(raw.decode("utf-8", errors="replace")):
                hits.append(str(path))
    return sorted(set(hits)), files_scanned, bytes_scanned


def select_unused_seed(roots: list[Path], candidates=None) -> int:
    values = iter(candidates) if candidates is not None else None
    for _ in range(512):
        candidate = (next(values, None) if values is not None else
                     secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED)
        if candidate is None:
            break
        if type(candidate) is not int or not MIN_SEED <= candidate <= MAX_SEED:
            continue
        if candidate in FORBIDDEN_SEEDS:
            continue
        if not scan_seed(candidate, roots)[0]:
            return candidate
    raise ValueError("could not find a fresh seed in the audited V1330 corpus")


def write_seed_audit(seed: int, purpose: str, roots: list[Path], destination: Path) -> dict:
    require(purpose in PURPOSES, "unsupported V1330 seed-audit purpose")
    require(type(seed) is int and MIN_SEED <= seed <= MAX_SEED and seed not in FORBIDDEN_SEEDS,
            "V1330 audit seed is forbidden or outside the approved range")
    require(not destination.exists() and not destination.is_symlink(),
            "V1330 seed audit output already exists: " + str(destination))
    hits, files_scanned, bytes_scanned = scan_seed(seed, roots)
    require(not hits, "V1330 seed already appears in audited text: " + ", ".join(hits[:5]))
    content = (
        "V1330_SEED_AUDIT_V1\n"
        f"PURPOSE {purpose}\n"
        f"SEED {seed}\n"
        f"AUDIT_UTC {datetime.now(timezone.utc).isoformat()}\n"
        f"FILES_SCANNED {files_scanned}\n"
        f"BYTES_SCANNED {bytes_scanned}\n"
        "HITS []\n"
        "RESULT UNUSED_IN_AUDITED_TEXT\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return {"purpose": purpose, "seed": seed, "files_scanned": files_scanned,
            "bytes_scanned": bytes_scanned, "audit_text": content}
