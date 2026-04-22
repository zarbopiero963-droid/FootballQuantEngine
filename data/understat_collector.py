import json
import re

import requests


class UnderstatCollector:

    BASE_URL = "https://understat.com"
    _TEAMS_PATTERN = re.compile(r"teamsData\s+=\s+JSON.parse\('(.*)'\)")
    _MATCHES_PATTERN = re.compile(r"matchesData\s+=\s+JSON.parse\('(.*)'\)")
    _REQUEST_TIMEOUT = 30

    def get_league_data(self, league):

        url = f"{self.BASE_URL}/league/{league}"

        response = requests.get(url, timeout=self._REQUEST_TIMEOUT)
        response.raise_for_status()

        match = self._TEAMS_PATTERN.search(response.text)
        if match is None:
            raise ValueError(
                f"Could not find teamsData in Understat response for league '{league}'"
            )

        decoded = json.loads(match.group(1).encode("utf-8").decode("unicode_escape"))

        return decoded

    def get_team_data(self, team, season):

        url = f"{self.BASE_URL}/team/{team}/{season}"

        response = requests.get(url, timeout=self._REQUEST_TIMEOUT)
        response.raise_for_status()

        match = self._MATCHES_PATTERN.search(response.text)
        if match is None:
            raise ValueError(
                f"Could not find matchesData in Understat response for team '{team}' season '{season}'"
            )

        decoded = json.loads(match.group(1).encode("utf-8").decode("unicode_escape"))

        return decoded
