from __future__ import annotations

import os
import subprocess
import sys
import threading
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

# GitHub Actions Windows runners are slow on cold start due to Defender scans.
_WINDOW_TIMEOUT = float(os.environ.get("EXE_WINDOW_TIMEOUT", "60"))
_POLL_INTERVAL = 1.5


def _start_app() -> subprocess.Popen:
    proc = subprocess.Popen(
        [EXE_PATH],
        stdout=subprocess.DEVNULL,  # GUI app; stdout not useful
        stderr=subprocess.PIPE,
    )
    # Drain stderr in a background thread so the OS pipe buffer (~64 KB) never
    # fills and blocks the subprocess.  Chunks are accumulated for diagnostics.
    _buf: list[bytes] = []

    def _drain() -> None:
        assert proc.stderr
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            _buf.append(chunk)

    proc._stderr_buf = _buf  # type: ignore[attr-defined]
    threading.Thread(target=_drain, daemon=True).start()
    return proc


def _process_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def _wait_for_window(
    proc: subprocess.Popen, timeout: float = _WINDOW_TIMEOUT
) -> tuple:
    """
    Wait for the app window to appear, anchored to the process PID.

    Uses pywinauto.Application.connect() so we only look at windows owned
    by our specific process — not every window on the desktop — which is
    both faster and immune to title collisions with other apps.

    Always returns (window | None, diag: str).
    """
    Application = pytest.importorskip("pywinauto").Application
    end_time = time.time() + timeout

    while time.time() < end_time:
        if not _process_alive(proc):
            stderr = b"".join(getattr(proc, "_stderr_buf", [])).decode(errors="replace")
            pytest.fail(
                f"EXE exited with code {proc.returncode} before a window appeared.\n"
                f"stderr: {stderr[-2000:]}"
            )

        try:
            app = Application(backend="uia").connect(process=proc.pid, timeout=2)
            # top_window() returns the foreground window of this process
            win = app.top_window()
            # Confirm it has a real title (not a transient splash/loader)
            title = win.window_text() or ""
            if title:
                return win, ""
        except Exception:
            pass

        time.sleep(_POLL_INTERVAL)

    # Last-ditch: dump what's visible for CI diagnostics
    try:
        Desktop = pytest.importorskip("pywinauto").Desktop
        all_titles = [
            w.window_text()
            for w in Desktop(backend="uia").windows()
            if w.window_text()
        ]
        diag = f"Visible windows at timeout: {all_titles}"
    except Exception as exc:
        diag = f"Could not enumerate windows: {exc}"

    return None, diag


def _terminate_process_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
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
        main, diag = _wait_for_window(proc)
        assert main is not None, f"No window found after {_WINDOW_TIMEOUT}s. {diag}"

        title = ""
        try:
            title = main.window_text()
        except Exception:
            pass

        assert "football quant engine" in title.lower(), (
            f"Unexpected window title: {title!r}"
        )
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
        main, diag = _wait_for_window(proc)
        assert main is not None, f"No window found after {_WINDOW_TIMEOUT}s. {diag}"

        descendants = []
        try:
            descendants = main.descendants()
        except Exception:
            pass

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
