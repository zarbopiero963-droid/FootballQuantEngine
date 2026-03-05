from repo_update_engine import backup_repository
import os


def test_backup_creation(tmp_path):

    os.chdir(tmp_path)

    with open("file.txt", "w") as f:
        f.write("hello")

    os.mkdir("backups")

    backup_repository()

    backups = os.listdir("backups")

    assert len(backups) > 0