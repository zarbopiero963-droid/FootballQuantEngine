import datetime
import os
import shutil
import subprocess

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
LOG_DIR = "logs"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")


# ------------------------
# UTIL
# ------------------------


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log(msg):
    ensure_dir(LOG_DIR)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

    print(msg)


# ------------------------
# BACKUP
# ------------------------


def backup_repository():
    ensure_dir(BACKUP_DIR)

    ts = timestamp()

    dst = os.path.join(BACKUP_DIR, f"backup_{ts}")

    shutil.make_archive(dst, "zip", ".")

    log(f"Backup created {dst}.zip")


def rollback():
    if not os.path.exists(BACKUP_DIR):
        log("No backup directory")
        return

    backups = sorted(os.listdir(BACKUP_DIR))

    if not backups:
        log("No backups found")
        return

    latest = backups[-1]

    archive = os.path.join(BACKUP_DIR, latest)

    log(f"Rollback using {archive}")

    shutil.unpack_archive(archive, ".")

    log("Rollback completed")


# ------------------------
# FILE OPS
# ------------------------


def create_folder(path):
    os.makedirs(path, exist_ok=True)

    log(f"Folder created {path}")


def create_file(path, content):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log(f"File created {path}")


def append_file(path, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + content)

    log(f"Append {path}")


# ------------------------
# CLEAN WHITESPACE
# ------------------------


def fix_whitespace():
    for root, dirs, files in os.walk("."):

        if ".git" in root:
            continue

        for file in files:

            if not file.endswith((".py", ".txt", ".md", ".yml", ".json")):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            new_lines = []

            for line in lines:
                line = line.rstrip()

                line = line.replace("\t", "    ")

                new_lines.append(line + "\n")

            while new_lines and new_lines[-1].strip() == "":
                new_lines.pop()

            new_lines.append("\n")

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

    log("Whitespace fixed")


# ------------------------
# FORMAT + LINT + TEST
# ------------------------


def run_black():
    log("Running Black")

    subprocess.run(["black", "."], check=False)


def run_isort():
    log("Running isort")

    subprocess.run(["isort", "."], check=False)


def run_ruff():
    log("Running ruff")

    subprocess.run(["ruff", "check", ".", "--fix"], check=False)


def run_pytest():
    log("Running pytest")

    result = subprocess.run(["pytest", "-v"], capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        log("Tests FAILED")

        rollback()

        raise Exception("Tests failed")

    log("Tests PASSED")


# ------------------------
# PROCESS
# ------------------------


def process():
    if not os.path.exists(INSTRUCTIONS_FILE):
        log("No instruction file")
        return

    with open(INSTRUCTIONS_FILE) as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line:
            i += 1
            continue

        parts = line.split()

        cmd = parts[0]

        if cmd == "AUTO_BACKUP_BEFORE_RUN":
            backup_repository()

        elif cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:
            create_folder(parts[1])

        elif cmd in ["CREA_FILE", "CREATE_FILE"]:

            path = parts[1]

            i += 1

            content = []

            while i < len(lines) and lines[i].strip() != "EOF":

                content.append(lines[i])

                i += 1

            create_file(path, "".join(content))

        elif cmd == "APPEND":

            path = parts[1]

            i += 1

            content = []

            while i < len(lines) and lines[i].strip() != "EOF":

                content.append(lines[i])

                i += 1

            append_file(path, "".join(content))

        elif cmd == "FIX_WHITESPACE":
            fix_whitespace()

        i += 1

    # ------------------------
    # CI PIPELINE
    # ------------------------

    fix_whitespace()

    run_black()

    run_isort()

    run_ruff()

    run_pytest()


if __name__ == "__main__":
    process()