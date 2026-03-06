from training.dataset_builder import DatasetBuilder


def test_dataset_builder_class_exists():

    builder = DatasetBuilder()

    assert isinstance(builder, DatasetBuilder)
