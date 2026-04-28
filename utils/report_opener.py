from __future__ import annotations

import os
import webbrowser


def open_report(report_path):

    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)

    webbrowser.open("file://" + os.path.abspath(report_path))
