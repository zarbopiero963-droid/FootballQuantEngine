import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from repo_update_engine import backup_repository


def test_backup_creation(tmp_path):

    os.chdir(tmp_path)

    os.mkdir("backups")

    with open("test.txt", "w") as f:
        f.write("hello")

    backup_repository()

    backups = os.listdir("backups")

    assert len(backups) > 0
