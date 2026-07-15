from __future__ import annotations

import sys
from pathlib import Path

APP_BUNDLE = Path(__file__).resolve().parents[1]
PIPELINE = APP_BUNDLE / "scripts" / "pipeline"

if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))
