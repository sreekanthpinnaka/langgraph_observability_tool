"""Build script to compile langgraph-observe into a standalone Windows .exe using PyInstaller."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
ENTRY_POINT = PROJECT_ROOT / "langgraph_observe" / "__main__.py"
UI_HTML = PROJECT_ROOT / "langgraph_observe" / "server" / "ui" / "index.html"


def build_exe() -> None:
    print("=" * 60)
    print("  [*] Building langgraph-observe.exe with PyInstaller")
    print("=" * 60)

    if not UI_HTML.exists():
        raise FileNotFoundError(f"UI index.html template not found at {UI_HTML}")

    # Hidden imports required by FastAPI, Uvicorn, SQLAlchemy, PyMySQL, etc.
    hidden_imports = [
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
        "starlette",
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "sqlalchemy.dialects.sqlite",
        "sqlalchemy.dialects.mysql",
        "sqlalchemy.sql.default_comparator",
        "pymysql",
        "dotenv",
        "anyio",
    ]

    # Data to bundle: index.html into langgraph_observe/server/ui
    data_separator = ";" if os.name == "nt" else ":"
    add_data_arg = f"{UI_HTML}{data_separator}langgraph_observe/server/ui"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "langgraph-observe",
        "--add-data",
        add_data_arg,
    ]

    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    cmd.append(str(ENTRY_POINT))

    print(f"Executing: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[!] PyInstaller build failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    exe_path = DIST_DIR / ("langgraph-observe.exe" if os.name == "nt" else "langgraph-observe")
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("  [+] Build Successful!")
        print(f"  Executable: {exe_path}")
        print(f"  Size:       {size_mb:.2f} MB")
        print("=" * 60)

        # Also copy .env as a template next to the exe if not already present
        sample_env = DIST_DIR / ".env"
        if not sample_env.exists() and (PROJECT_ROOT / ".env").exists():
            shutil.copy2(PROJECT_ROOT / ".env", sample_env)
            print(f"  Copied .env configuration to: {sample_env}")
    else:
        print(f"\n[!] Executable not found at expected location: {exe_path}")
        sys.exit(1)


if __name__ == "__main__":
    build_exe()
