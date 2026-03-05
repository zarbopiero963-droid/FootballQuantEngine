from io import StringIO

import pandas as pd
import requests


class ClubEloCollector:

    BASE_URL = "http://api.clubelo.com"

    def get_team_history(self, team):

        url = f"{self.BASE_URL}/{team}"

        response = requests.get(url)

        data = StringIO(response.text)

        df = pd.read_csv(data)

        return df

    def get_ranking(self):

        url = f"{self.BASE_URL}/elo"

        response = requests.get(url)

        data = StringIO(response.text)

        df = pd.read_csv(data)

        return df
