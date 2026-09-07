from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import pytest

from langgraph_observe.server.cli import main


def test_cli_server_command_args():
    """Verify CLI parses --port, --host, --db, and --reload flags for 'server'."""
    with patch.object(sys, "argv", ["langgraph-observe", "server", "--port", "9000", "--host", "0.0.0.0", "--reload"]), \
         patch("uvicorn.run") as mock_uvicorn, \
         patch("langgraph_observe.server.cli.create_server_app") as mock_create_app:

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        main()

        mock_create_app.assert_called_once_with(db_path="observe.db", db_url=None)
        mock_uvicorn.assert_called_once_with(mock_app, host="0.0.0.0", port=9000, reload=True)


def test_cli_run_and_start_aliases():
    """Verify 'run' and 'start' aliases function identically to 'server'."""
    with patch.object(sys, "argv", ["langgraph-observe", "run", "--port", "7777"]), \
         patch("uvicorn.run") as mock_uvicorn, \
         patch("langgraph_observe.server.cli.create_server_app") as mock_create_app:

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        main()
        mock_uvicorn.assert_called_once_with(mock_app, host="127.0.0.1", port=7777, reload=False)


def test_cli_lazy_import_export():
    """Verify langgraph_observe.server exposes start_server_cli via lazy __getattr__."""
    import langgraph_observe.server as server_mod

    start_fn = getattr(server_mod, "start_server_cli")
    assert callable(start_fn)

    with pytest.raises(AttributeError):
        getattr(server_mod, "non_existent_attribute_xyz")


def test_cli_one_click_interactive_url_input(monkeypatch):
    """Verify one-click interactive mode prompts and accepts a custom Database URL."""
    from langgraph_observe.server.cli import main

    monkeypatch.setattr(sys, "argv", ["langgraph-observe"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "mysql+pymysql://test:pass@localhost/custom_db")

    with patch("uvicorn.run") as mock_uvicorn, \
         patch("langgraph_observe.server.cli.create_server_app") as mock_create_app, \
         patch("langgraph_observe.server.cli._open_browser_delayed") as mock_browser:

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        main()

        mock_create_app.assert_called_once_with(db_path="observe.db", db_url="mysql+pymysql://test:pass@localhost/custom_db")
        mock_uvicorn.assert_called_once()
        mock_browser.assert_called_once()


def test_cli_one_click_interactive_default_db(monkeypatch):
    """Verify pressing Enter uses default database in one-click mode."""
    from langgraph_observe.server.cli import main

    monkeypatch.setattr(sys, "argv", ["langgraph-observe"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    with patch("uvicorn.run") as mock_uvicorn, \
         patch("langgraph_observe.server.cli.create_server_app") as mock_create_app, \
         patch("langgraph_observe.server.cli._open_browser_delayed") as mock_browser:

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        main()

        mock_create_app.assert_called_once_with(db_path="observe.db", db_url=None)
        mock_uvicorn.assert_called_once()
        mock_browser.assert_called_once()


def test_cli_no_browser_flag():
    """Verify --no-browser flag prevents browser opening."""
    with patch.object(sys, "argv", ["langgraph-observe", "server", "--no-browser"]), \
         patch("uvicorn.run"), \
         patch("langgraph_observe.server.cli.create_server_app"), \
         patch("langgraph_observe.server.cli._open_browser_delayed") as mock_browser:

        main()
        mock_browser.assert_not_called()


def test_frozen_ui_dir_resolution(monkeypatch, tmp_path):
    """Verify get_dashboard_html resolves path via sys._MEIPASS when frozen."""
    fake_meipass = tmp_path / "meipass"
    ui_dir = fake_meipass / "langgraph_observe" / "server" / "ui"
    ui_dir.mkdir(parents=True)
    (ui_dir / "index.html").write_text("<h1>Bundled Frozen UI for LangGraph</h1>", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)

    from langgraph_observe.server.ui import get_dashboard_html
    html = get_dashboard_html()
    assert "<h1>Bundled Frozen UI for LangGraph</h1>" in html


