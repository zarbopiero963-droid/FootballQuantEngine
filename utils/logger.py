from __future__ import annotations

import logging


def get_logger():

    logger = logging.getLogger("quant_engine")

    if not logger.handlers:

        handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        handler.setFormatter(formatter)

        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

    return logger
