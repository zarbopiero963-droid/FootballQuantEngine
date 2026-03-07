import os
import subprocess
import sys
import time

import pytest

if not sys.platform.startswith("win"):
    pytest.skip("EXE smoke tests run only on Windows", allow_module_level=True)

psutil = pytest.importorskip("psutil")

from pywinauto import Desktop

EXE_PATH = os.environ.get(
    "APP_EXE_PATH",
    r"dist\FootballQuantEngine\FootballQuantEngine.exe",
)


def _start_app():
    proc = subprocess.Popen(
        [EXE_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    time.sleep(8)
    return proc


def _find_main_window(process_id, timeout=30):
    desktop = Desktop(backend="uia")
    end_time = time.time() + timeout

    while time.time() < end_time:
        windows = []
        for win in desktop.windows():
            try:
                if win.process_id() == process_id:
                    windows.append(win)
            except Exception:
                continue

        visible_windows = []
        for win in windows:
            try:
                if win.is_visible():
                    visible_windows.append(win)
            except Exception:
                continue

        if visible_windows:
            return visible_windows[0]

        if windows:
            return windows[0]

        time.sleep(1)

    return None


@pytest.mark.windows
def test_installed_exe_opens_main_window():
    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    proc = _start_app()

    try:
        main = _find_main_window(proc.pid, timeout=30)

        assert main is not None, "No window found for EXE process"
        assert main.is_visible() or main.is_enabled()

        try:
            title = main.window_text()
        except Exception:
            title = ""

        assert isinstance(title, str)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


@pytest.mark.windows
def test_installed_exe_does_not_exit_immediately():
    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    proc = _start_app()

    try:
        assert psutil.pid_exists(proc.pid), "EXE process was not created"

        process = psutil.Process(proc.pid)
        assert process.is_running(), "EXE process is not running"

        return_code = proc.poll()
        assert return_code is None, f"EXE exited too early with code: {return_code}"
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
