import pytest


def test_desktop_launcher_import():

    try:
        from app.desktop_launcher import main
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"Desktop launcher dependencies not available in CI: {exc}")

    assert main is not None
