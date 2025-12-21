from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def write_meta_json(path: Path, meta: Dict[str, Any]) -> None:
    """
    Write run metadata in a stable, human-readable format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    m = dict(meta)
    m.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    m.setdefault("python", platform.python_version())
    m.setdefault("platform", platform.platform())
    m.setdefault("torch", torch.__version__)
    path.write_text(json.dumps(m, indent=2, sort_keys=True))


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
