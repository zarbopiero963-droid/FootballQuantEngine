import sys
import os

# aggiunge la root del repository al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from repo_update_engine import fix_whitespace


def test_whitespace():

    assert callable(fix_whitespace)