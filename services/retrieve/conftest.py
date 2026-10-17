"""pytest 根配置：把 services/retrieve 注入 sys.path，使 `import kf_retrieve` 可用。"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
