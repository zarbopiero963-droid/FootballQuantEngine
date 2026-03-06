import pandas as pd

from engine.output_manager import OutputManager


def test_output_manager_class_exists():

    manager = OutputManager()

    assert isinstance(manager, OutputManager)


def test_output_manager_export_dataframe_empty():

    manager = OutputManager()

    df = pd.DataFrame()

    manager.export_dataframe(df, "empty_test")

    assert isinstance(df, pd.DataFrame)
