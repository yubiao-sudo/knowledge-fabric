"""把 scripts/ 与 services/ingest 注入 sys.path，使 chunk_eval 可直接导入。"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
_INGEST = os.path.join(_REPO, "services", "ingest")

for path in (_SCRIPTS, _INGEST):
    if path not in sys.path:
        sys.path.insert(0, path)
