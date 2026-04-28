import os
import subprocess
import sys
import time

import pytest

# Tests must be *collected* on all platforms so `-m windows` returns exit 0
# (2 skipped) rather than exit 5 (no tests collected).  The actual Windows-
# only libraries are imported lazily inside each test and helper.
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="EXE smoke tests require Windows with pywinauto and psutil",
)

EXE_PATH = os.environ.get(
    "APP_EXE_PATH",
    r"dist\FootballQuantEngine\FootballQuantEngine.exe",
)


def _start_app():
    return subprocess.Popen(
        [EXE_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _wait_for_window(timeout: float = 30.0):
    Desktop = pytest.importorskip("pywinauto").Desktop
    end_time = time.time() + timeout

    while time.time() < end_time:
        try:
            windows = Desktop(backend="uia").windows()
            for window in windows:
                try:
                    title = window.window_text() or ""
                except Exception:
                    title = ""

                if "football quant engine" in title.lower():
                    return window
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)

    return None


def _terminate_process_tree(pid: int):
    try:
        import psutil

        parent = psutil.Process(pid)
        for proc in parent.children(recursive=True):
            try:
                proc.kill()
            except Exception:
                pass
        parent.kill()
    except Exception:
        pass


@pytest.mark.windows
def test_installed_exe_opens_main_window():
    pytest.importorskip("pywinauto")
    pytest.importorskip("psutil")

    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    proc = _start_app()

    try:
        main = _wait_for_window(timeout=30.0)
        assert main is not None, "No window found for EXE process"

        try:
            title = main.window_text()
        except Exception:
            title = ""

        assert "football quant engine" in title.lower()
    finally:
        _terminate_process_tree(proc.pid)


@pytest.mark.windows
def test_installed_exe_has_main_buttons():
    pytest.importorskip("pywinauto")
    pytest.importorskip("psutil")

    if not os.path.exists(EXE_PATH):
        pytest.skip(f"EXE not found: {EXE_PATH}")

    proc = _start_app()

    try:
        main = _wait_for_window(timeout=30.0)
        assert main is not None, "No window found for EXE process"

        descendants = []
        try:
            descendants = main.descendants()
        except Exception:
            descendants = []

        texts = []
        for item in descendants:
            try:
                text = (item.window_text() or "").strip()
            except Exception:
                text = ""
            if text:
                texts.append(text.lower())

        joined = " | ".join(texts)

        assert (
            "run" in joined
            or "start" in joined
            or "backtest" in joined
            or "dashboard" in joined
            or "import" in joined
        ), f"No expected controls found. Controls: {joined}"
    finally:
        _terminate_process_tree(proc.pid)
