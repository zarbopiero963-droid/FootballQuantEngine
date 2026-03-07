from app.version_info import get_version_info


def test_version_info_contains_expected_keys():

    info = get_version_info()

    assert "name" in info
    assert "version" in info
    assert "author" in info
    assert "description" in info
