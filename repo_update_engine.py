import datetime
import os
import shutil
import subprocess
import zipfile

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
LOG_DIR = "logs"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")

AUTO_BACKUP_ALWAYS = True
KEEP_LAST_BACKUPS = 10


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


def list_backups():
    ensure_dir(BACKUP_DIR)
    files = [
        os.path.join(BACKUP_DIR, f)
        for f in os.listdir(BACKUP_DIR)
        if f.startswith("backup_") and f.endswith(".zip")
    ]
    files.sort()
    return files


def cleanup_old_backups():
    backups = list_backups()

    if len(backups) <= KEEP_LAST_BACKUPS:
        return

    old = backups[:-KEEP_LAST_BACKUPS]

    for f in old:
        try:
            os.remove(f)
            log(f"Deleted old backup {f}")
        except Exception as e:
            log(f"Cannot delete {f}: {e}")


def backup_repository():

    ensure_dir(BACKUP_DIR)

    ts = timestamp()

    zip_path = os.path.join(BACKUP_DIR, f"backup_{ts}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:

        for root, dirs, files in os.walk("."):

            if ".git" in root:
                continue

            if BACKUP_DIR in root:
                continue

            for file in files:

                if file.endswith(".pyc"):
                    continue

                path = os.path.join(root, file)

                rel = os.path.relpath(path, ".")

                z.write(path, rel)

    log(f"Backup created {zip_path}")

    cleanup_old_backups()


def restore_latest_backup():

    backups = list_backups()

    if not backups:
        log("No backup found")
        return False

    latest = backups[-1]

    log(f"Rollback using {latest}")

    with zipfile.ZipFile(latest, "r") as z:

        for member in z.namelist():

            if member.startswith(".git"):
                continue

            dest = os.path.join(".", member)

            ensure_dir(os.path.dirname(dest))

            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    log("Rollback completed")

    return True


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

    ensure_dir(os.path.dirname(path))

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + content)

    log(f"Append {path}")


def replace_text(path, old, new):

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    data = data.replace(old, new)

    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

    log(f"Replace text in {path}")


def replace_line(path, old, new):

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    out = []
    changed = False

    for line in lines:

        if old in line:
            out.append(line.replace(old, new))
            changed = True
        else:
            out.append(line)

    if changed:

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out)

        log(f"REPLACE_LINE applied in {path}")

    else:

        log(f"Pattern not found in {path}")


def insert_line(path, number, text):

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = max(0, number - 1)

    if idx > len(lines):
        idx = len(lines)

    lines.insert(idx, text + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    log(f"Inserted line {number} in {path}")


# ------------------------
# WHITESPACE
# ------------------------


def fix_whitespace():

    for root, dirs, files in os.walk("."):

        if ".git" in root:
            continue

        for file in files:

            if not file.endswith((".py", ".txt", ".md", ".yml", ".json", ".ini")):
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

            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new)

    log("Whitespace fixed")


# ------------------------
# FORMAT / TEST
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
        raise Exception("Tests failed")

    log("Tests PASSED")


# ------------------------
# PROCESS
# ------------------------


def process():

    if not os.path.exists(INSTRUCTIONS_FILE):
        log("No instruction file")
        return

    if AUTO_BACKUP_ALWAYS:
        backup_repository()

    with open(INSTRUCTIONS_FILE, encoding="utf-8") as f:
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

        elif cmd in ["APPEND", "AGGIUNGI"]:

            path = parts[1]

            i += 1
            content = []

            while i < len(lines) and lines[i].strip() != "EOF":
                content.append(lines[i])
                i += 1

            append_file(path, "".join(content))

        elif cmd in ["REPLACE", "SOSTITUISCI"]:

            path = parts[1]
            old = parts[2]
            new = " ".join(parts[3:])

            replace_text(path, old, new)

        elif cmd == "REPLACE_LINE":

            path = parts[1]

            if parts[2] == "<<EOF":

                i += 1
                old = lines[i].rstrip("\n")

                i += 1
                new = lines[i].rstrip("\n")

                while i < len(lines) and lines[i].strip() != "EOF":
                    i += 1

                replace_line(path, old, new)

            else:

                old = parts[2]
                new = " ".join(parts[3:])

                replace_line(path, old, new)

        elif cmd in ["INSERT_LINE", "INSERISCI_RIGA"]:

            path = parts[1]
            number = int(parts[2])
            text = " ".join(parts[3:])

            insert_line(path, number, text)

        elif cmd == "FIX_WHITESPACE":
            fix_whitespace()

        i += 1

    try:

        fix_whitespace()
        run_black()
        run_isort()
        run_ruff()
        run_pytest()

    except Exception as e:

        log(f"CI FAILED: {e}")

        ok = restore_latest_backup()

        if not ok:
            raise

        raise


if __name__ == "__main__":
    process()
