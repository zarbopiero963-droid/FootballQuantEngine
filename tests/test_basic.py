import pytest


def test_import_pyside6():
    pytest.importorskip("PySide6")


def test_import_numpy():
    import numpy


def test_import_pandas():
    import pandas


def test_import_sklearn():
    import sklearn


def test_import_matplotlib():
    import matplotlib