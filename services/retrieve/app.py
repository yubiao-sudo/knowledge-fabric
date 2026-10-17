"""供 `uvicorn app:app` 使用的入口（在 services/retrieve 目录下运行）。

真正的应用定义在 `kf_retrieve/app.py`，这里只做再导出。
"""

from kf_retrieve.app import app  # noqa: F401

__all__ = ["app"]
