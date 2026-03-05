import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from repo_update_engine import create_file


def test_create_file(tmp_path):

    file = tmp_path / "hello.txt"

    create_file(file, "hello world")

    assert file.exists()
