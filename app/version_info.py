from app.app_metadata import (
    APP_AUTHOR,
    APP_DESCRIPTION,
    APP_NAME,
    APP_VERSION,
)


def get_version_info():

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "author": APP_AUTHOR,
        "description": APP_DESCRIPTION,
    }
