from app.app_metadata import APP_NAME, APP_VERSION
from app.version_info import get_version_info


def test_app_metadata_exists():

    assert isinstance(APP_NAME, str)
    assert isinstance(APP_VERSION, str)


def test_version_info_runs():

    info = get_version_info()

    assert "name" in info
    assert "version" in info
