import os
import ast
import shutil
import datetime
import zipfile

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
LOG_DIR = "logs"
PATCH_DIR = "patches"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")


# -----------------------
# UTIL
# -----------------------

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log(msg):

    ensure_dir(LOG_DIR)

    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

    print(msg)


# -----------------------
# AUTO BACKUP
# -----------------------

AUTO_BACKUP = True


# -----------------------
# BACKUP ZIP
# -----------------------

def backup_repository():

    ensure_dir(BACKUP_DIR)

    ts = timestamp()

    zip_path = os.path.join(BACKUP_DIR, f"backup_{ts}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

        for root, dirs, files in os.walk("."):

            if root.startswith("./backups"):
                continue

            if root.startswith("./.git"):
                continue

            for file in files:

                path = os.path.join(root, file)

                arc = os.path.relpath(path, ".")

                zipf.write(path, arc)

    log(f"Backup created {zip_path}")


# -----------------------
# RESTORE ZIP
# -----------------------

def restore_latest():

    files = sorted(os.listdir(BACKUP_DIR))

    if not files:
        log("No backups found")
        return

    latest = files[-1]

    path = os.path.join(BACKUP_DIR, latest)

    with zipfile.ZipFile(path, "r") as zipf:

        zipf.extractall(".")

    log(f"Restored backup {latest}")


# -----------------------
# FILE OPS
# -----------------------

def create_folder(path):

    os.makedirs(path, exist_ok=True)

    log(f"Folder created {path}")


def create_file(path, content):

    folder = os.path.dirname(path)

    if folder:
        ensure_dir(folder)

    with open(path, "w") as f:
        f.write(content)

    log(f"File created {path}")


def append_file(path, content):

    with open(path, "a") as f:
        f.write(content)

    log(f"Append {path}")


# -----------------------
# WHITESPACE FIX
# -----------------------

def fix_whitespace():

    for root, dirs, files in os.walk("."):

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

    log("Whitespace fixed")


# -----------------------
# MULTILINE PARSER
# -----------------------

def read_block(lines, start):

    content = []

    i = start

    while i < len(lines) and lines[i].strip() != "EOF":

        content.append(lines[i])

        i += 1

    return "".join(content), i


# -----------------------
# PROCESS
# -----------------------

def process():

    if not os.path.exists(INSTRUCTIONS_FILE):
        return

    if AUTO_BACKUP:
        backup_repository()

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

        if cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:

            create_folder(parts[1])

        elif cmd in ["CREA_FILE", "CREATE_FILE"]:

            path = parts[1]

            if len(parts) > 2 and parts[2] == "<<EOF":

                content, i = read_block(lines, i + 1)

                create_file(path, content)

            else:

                create_file(path, " ".join(parts[2:]))

        elif cmd in ["APPEND", "AGGIUNGI"]:

            path = parts[1]

            if len(parts) > 2 and parts[2] == "<<EOF":

                content, i = read_block(lines, i + 1)

                append_file(path, content)

            else:

                append_file(path, " ".join(parts[2:]))

        elif cmd == "BACKUP_REPOSITORY":

            backup_repository()

        elif cmd == "RESTORE_LATEST":

            restore_latest()

        elif cmd == "FIX_WHITESPACE":

            fix_whitespace()

        i += 1


if __name__ == "__main__":

    process()