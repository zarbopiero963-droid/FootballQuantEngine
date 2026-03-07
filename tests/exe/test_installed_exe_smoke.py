import os

import pytest

pytest.importorskip("pywinauto")

from pywinauto.application import Application

EXE_PATH = os.environ.get(
    "APP_EXE_PATH",
    r"dist\FootballQuantEngine\FootballQuantEngine.exe",
)


@pytest.mark.windows
def test_installed_exe_opens_main_window():
    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    app = Application(backend="uia").start(EXE_PATH)

    try:
        main = app.window(title_re=".*Football Quant Engine.*")
        main.wait("visible", timeout=20)

        assert main.exists()
        assert main.is_visible()
    finally:
        try:
            app.kill()
        except Exception:
            pass


@pytest.mark.windows
def test_installed_exe_has_main_buttons():
    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    app = Application(backend="uia").start(EXE_PATH)

    try:
        main = app.window(title_re=".*Football Quant Engine.*")
        main.wait("visible", timeout=20)

        run_btn = main.child_window(title="Run Cycle", control_type="Button")
        settings_btn = main.child_window(title="Settings", control_type="Button")
        about_btn = main.child_window(title="About", control_type="Button")

        assert run_btn.exists()
        assert settings_btn.exists()
        assert about_btn.exists()
    finally:
        try:
            app.kill()
        except Exception:
            pass
