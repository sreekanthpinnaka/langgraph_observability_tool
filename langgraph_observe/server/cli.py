from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
import webbrowser
from dotenv import load_dotenv
import uvicorn

from langgraph_observe.server.app import create_server_app

# Ensure Unicode output doesn't crash on standard Windows consoles (cp1252)
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def _load_environment() -> None:
    """Load .env file, prioritizing the executable's directory if packaged with PyInstaller."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        env_path = exe_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return
    load_dotenv()


_load_environment()


def _open_browser_delayed(url: str, delay: float = 1.0) -> None:
    """Open default web browser after a short delay so the server has time to start."""
    def _target():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()


def main() -> None:
    is_one_click = (len(sys.argv) == 1)
    is_interactive = bool(sys.stdin and sys.stdin.isatty())

    parser = argparse.ArgumentParser(
        prog="langgraph-observe",
        description="CLI tool for the LangGraph Observability & Tracing Server",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'server' command (with 'run' and 'start' aliases)
    server_parser = subparsers.add_parser(
        "server",
        aliases=["run", "start"],
        help="Start the standalone observability server & dashboard",
    )
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    server_parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    server_parser.add_argument("--db-url", type=str, default=None, help="Database URL (e.g. mysql+pymysql://user:pass@host:3306/db, or read from DATABASE_URL in .env)")
    server_parser.add_argument("--db", type=str, default="observe.db", help="SQLite database fallback path (default: observe.db)")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    server_parser.add_argument("--open-browser", dest="open_browser", action="store_true", default=None, help="Automatically open the dashboard in your default browser")
    server_parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="Do not automatically open browser")

    args = parser.parse_args()

    if args.command is None or args.command in ("server", "run", "start"):
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8765)
        db_url = getattr(args, "db_url", None)
        db_path = getattr(args, "db", "observe.db")
        reload_flag = getattr(args, "reload", False)
        open_browser = getattr(args, "open_browser", None)

        # In one-click mode (double-clicked .exe), default to opening browser
        if open_browser is None:
            open_browser = is_one_click

        # Interactive database URL input if launched via click with no CLI arguments
        if is_one_click and is_interactive and db_url is None:
            configured_url = os.getenv("DATABASE_URL") or os.getenv("OBSERVE_DATABASE_URL")
            default_display = configured_url if configured_url else f"SQLite ({db_path})"
            print("\n" + "=" * 60)
            print("  🔭 LangGraph Observability Server Launcher")
            print("=" * 60)
            print(f"  Current Database: {default_display}")
            print("  Press [Enter] to use current, or paste a new Database URL below.")
            try:
                user_db_input = input("  Database URL: ").strip()
                if user_db_input:
                    db_url = user_db_input
            except (EOFError, KeyboardInterrupt):
                print("\nLaunch cancelled.")
                return

        dashboard_url = f"http://{host}:{port}/"
        ingest_url = f"http://{host}:{port}/api/v1/traces"
        db_display = db_url or ("from environment / .env" if ("DATABASE_URL" in os.environ or "OBSERVE_DATABASE_URL" in os.environ) else db_path)

        print("\n" + "=" * 60)
        print("  🔭 LangGraph Observability Server")
        print(f"  👉 Dashboard:      {dashboard_url}")
        print(f"  👉 Ingestion API:  {ingest_url}")
        print(f"  👉 Database:       {db_display}")
        print("=" * 60 + "\n")

        if open_browser:
            _open_browser_delayed(dashboard_url, delay=1.0)

        exit_code = 0
        try:
            app = create_server_app(db_path=db_path, db_url=db_url)
            uvicorn.run(app, host=host, port=port, reload=reload_flag)
        except KeyboardInterrupt:
            print("\nServer stopped.")
        except Exception as e:
            print(f"\n❌ Error running server: {e}")
            exit_code = 1
        finally:
            if getattr(sys, "frozen", False) and is_interactive:
                input("\nPress Enter to exit...")
            if exit_code != 0 and not getattr(sys, "frozen", False):
                sys.exit(exit_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

