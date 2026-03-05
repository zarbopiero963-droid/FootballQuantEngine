import requests

from utils.logger import get_logger

logger = get_logger()


class ApiClient:

    def __init__(self, base_url, api_key=None):

        self.base_url = base_url
        self.api_key = api_key

    def get(self, endpoint, params=None, headers=None):

        if params is None:
            params = {}

        if headers is None:
            headers = {}

        url = f"{self.base_url}/{endpoint}"

        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text}")
            raise Exception("API request failed")

        return response.json()
