import os
from repo_update_engine import fix_whitespace

def test_fix_whitespace(tmp_path):

    file_path = tmp_path / "test.py"

    # file con spazi sbagliati
    file_path.write_text(
        "def run():    \n\tprint('hello')    \n\n\n"
    )

    # esegue il fixer
    os.chdir(tmp_path)
    fix_whitespace()

    content = file_path.read_text()

    # controlli
    assert "\t" not in content
    assert content.endswith("\n")
    assert "    print('hello')" in content
