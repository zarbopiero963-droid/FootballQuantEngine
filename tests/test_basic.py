import pytest


def test_import_pyside6():
    pytest.importorskip("PySide6")