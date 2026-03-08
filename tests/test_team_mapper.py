from quant.fusion.team_mapper import TeamMapper


def test_team_mapper_returns_mapped_name():
    mapper = TeamMapper()

    assert mapper.map_name("Inter Milan") == "inter"
    assert mapper.map_name("Juve") == "juventus"
