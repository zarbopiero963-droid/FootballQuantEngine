from quant.fusion.team_name_normalizer import TeamNameNormalizer


def test_team_name_normalizer_maps_aliases():
    normalizer = TeamNameNormalizer()

    assert normalizer.normalize("Inter Milan") == "inter"
    assert normalizer.normalize("AC Milan") == "milan"
