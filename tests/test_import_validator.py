from training.import_validator import ImportValidator


def test_import_validator_class_exists():

    validator = ImportValidator()

    assert isinstance(validator, ImportValidator)
