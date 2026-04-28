from __future__ import annotations


def resolve_fixture_id(match_id, fixtures):

    for f in fixtures:
        if f"{f['home']}_vs_{f['away']}" == match_id:
            return f["fixture_id"]

    return None
