from __future__ import annotations

from datetime import datetime


def make_run_id(run_name: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_name.strip())
    return f"{ts}_{safe}_s{seed}"
