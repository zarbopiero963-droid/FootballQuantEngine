import json
import re

import requests


class UnderstatCollector:

    BASE_URL = "https://understat.com"

    def get_league_data(self, league):

        url = f"{self.BASE_URL}/league/{league}"

        response = requests.get(url)

        pattern = re.compile(r"teamsData\s+=\s+JSON.parse\('(.*)'\)")

        data = pattern.search(response.text).group(1)

        decoded = json.loads(data.encode("utf-8").decode("unicode_escape"))

        return decoded

    def get_team_data(self, team, season):

        url = f"{self.BASE_URL}/team/{team}/{season}"

        response = requests.get(url)

        pattern = re.compile(r"matchesData\s+=\s+JSON.parse\('(.*)'\)")

        data = pattern.search(response.text).group(1)

        decoded = json.loads(data.encode("utf-8").decode("unicode_escape"))

        return decoded
