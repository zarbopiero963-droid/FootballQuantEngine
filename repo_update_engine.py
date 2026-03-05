import ast
import datetime
import difflib
import os
import shutil
import subprocess
import sys

INSTRUCTIONS_FILE = "devops_update.txt"
BACKUP_DIR = "backups"
PATCH_DIR = "patches"
LOG_FILE = "logs/operations.log"
HISTORY_FILE = "backups/history.log"

DRY_RUN = "--dry-run" in sys.argv


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log(msg):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} {msg}\n")
    print(msg)


def history(msg):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} {msg}\n")


def version_backup_path():
    ts = timestamp()
    path = os.path.join(BACKUP_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path, ts


def backup_repository():

    root, ts = version_backup_path()

    for r, d, f in os.walk("."):

        if r.startswith("./backups"):
            continue

        for file in f:

            src = os.path.join(r, file)

            rel = os.path.relpath(src, ".")

            dst = os.path.join(root, rel)

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            shutil.copy2(src, dst)

    history(f"BACKUP_REPOSITORY {ts}")
    log(f"backup version {ts}")


def restore_version(version):

    root = os.path.join(BACKUP_DIR, version)

    for r, d, f in os.walk(root):

        for file in f:

            src = os.path.join(r, file)

            rel = os.path.relpath(src, root)

            dst = rel

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            shutil.copy2(src, dst)


def safe_patch_function(file_path, func_name, new_code):

    with open(file_path) as f:
        code = f.read()

    tree = ast.parse(code)

    class SafePatch(ast.NodeTransformer):

        def visit_FunctionDef(self, node):

            if node.name == func_name:

                patch = ast.parse(new_code).body

                node.body.extend(patch)

            return node

    tree = SafePatch().visit(tree)

    new_code_text = ast.unparse(tree)

    with open(file_path, "w") as f:
        f.write(new_code_text)


def run_lint():
    subprocess.run(["ruff", "check", "."])


def run_tests():
    subprocess.run(["pytest"])


def run_formatters():
    subprocess.run(["black", "."])
    subprocess.run(["isort", "."])


def fix_whitespace():

    for root, dirs, files in os.walk("."):

        if root.startswith("./.git"):
            continue

        for file in files:

            if not file.endswith((".py", ".txt", ".md", ".yml", ".json")):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            new = []

            for line in lines:
                line = line.rstrip()
                line = line.replace("\t", "    ")
                new.append(line + "\n")

            while new and new[-1].strip() == "":
                new.pop()

            new.append("\n")

            with open(path, "w") as f:
                f.writelines(new)


def process():

    if not os.path.exists(INSTRUCTIONS_FILE):
        return

    with open(INSTRUCTIONS_FILE) as f:
        lines = f.readlines()

    if DRY_RUN:
        print(lines)
        return

    for line in lines:

        parts = line.strip().split()

        if not parts:
            continue

        cmd = parts[0]

        if cmd == "BACKUP_REPOSITORY":
            backup_repository()

        elif cmd == "SAFE_PATCH_FUNZIONE":
            safe_patch_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd == "FIX_WHITESPACE":
            fix_whitespace()

    run_formatters()

    run_lint()

    run_tests()


if __name__ == "__main__":
    process()
