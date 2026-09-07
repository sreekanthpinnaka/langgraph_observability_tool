import sys
from pathlib import Path


def get_ui_dir() -> Path:
    """Return the directory containing UI assets, resolving PyInstaller bundle if frozen."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "langgraph_observe" / "server" / "ui"
    return Path(__file__).parent.resolve()


UI_DIR = get_ui_dir()
INDEX_HTML_PATH = UI_DIR / "index.html"


def get_dashboard_html() -> str:
    """Read and return the embedded dashboard HTML string."""
    path = get_ui_dir() / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<h1>LangGraph Observability Dashboard</h1><p>UI template not found.</p>"

