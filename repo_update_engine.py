import os
import ast
import shutil
import datetime
import subprocess
import difflib
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

    if not os.path.exists(root):
        log("version not found")
        return

    for r, d, f in os.walk(root):

        for file in f:

            src = os.path.join(r, file)

            rel = os.path.relpath(src, root)

            dst = rel

            os.makedirs(os.path.dirname(dst), exist_ok=True)

            shutil.copy2(src, dst)

    log("repository restored")


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

    log(f"patched function {func_name}")


def generate_patch():

    os.makedirs(PATCH_DIR, exist_ok=True)

    patch_file = os.path.join(PATCH_DIR, f"patch_{timestamp()}.diff")

    repo_files = []

    for root, dirs, files in os.walk("."):

        for file in files:

            if file.endswith(".py"):

                repo_files.append(os.path.join(root, file))

    diffs = []

    for file in repo_files:

        with open(file) as f:
            new = f.readlines()

        backup_versions = os.listdir(BACKUP_DIR)

        if not backup_versions:
            continue

        latest = sorted(backup_versions)[-1]

        backup_path = os.path.join(BACKUP_DIR, latest, file)

        if not os.path.exists(backup_path):
            continue

        with open(backup_path) as f:
            old = f.readlines()

        diff = difflib.unified_diff(old, new)

        diffs.extend(list(diff))

    with open(patch_file, "w") as f:
        f.writelines(diffs)

    log(f"patch generated {patch_file}")


def fix_whitespace():

    log("fixing whitespace")

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


def run_lint():
    subprocess.run(["ruff", "check", "."])


def run_tests():
    subprocess.run(["pytest"])


def run_formatters():
    subprocess.run(["black", "."])
    subprocess.run(["isort", "."])


def process():

    if not os.path.exists(INSTRUCTIONS_FILE):
        log("instruction file missing")
        return

    with open(INSTRUCTIONS_FILE) as f:
        lines = f.readlines()

    if DRY_RUN:
        for l in lines:
            print(l.strip())
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

        elif cmd == "GENERATE_PATCH":
            generate_patch()

        history(f"COMMAND {line.strip()}")

    run_formatters()

    run_lint()

    run_tests()


if __name__ == "__main__":
    process()
