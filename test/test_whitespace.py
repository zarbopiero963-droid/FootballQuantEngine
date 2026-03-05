from repo_update_engine import fix_whitespace
import os


def test_fix_whitespace(tmp_path):

    file = tmp_path / "test.txt"

    file.write_text("hello    \nworld\t\n\n")

    os.chdir(tmp_path)

    fix_whitespace()

    content = file.read_text()

    assert "hello\n" in content