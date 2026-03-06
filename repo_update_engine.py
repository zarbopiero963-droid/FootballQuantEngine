import os
import datetime
import subprocess
import zipfile

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
LOG_DIR = "logs"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")

AUTO_BACKUP_ALWAYS = True


# -------------------------
# UTIL
# -------------------------

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


def log_file_created(path):
    log(f"[CREATE] {path}")


def log_file_modified(path):
    log(f"[MODIFY] {path}")


def log_file_deleted(path):
    log(f"[DELETE] {path}")


# -------------------------
# BACKUP
# -------------------------

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

                src = os.path.join(root, file)

                rel = os.path.relpath(src, ".")

                z.write(src, rel)

    log(f"Backup created {zip_path}")


def restore_last_backup():

    backups = sorted(os.listdir(BACKUP_DIR))

    if not backups:
        log("No backup available")
        return

    latest = os.path.join(BACKUP_DIR, backups[-1])

    log(f"Rollback from {latest}")

    with zipfile.ZipFile(latest, "r") as z:
        z.extractall(".")


# -------------------------
# FILE OPS
# -------------------------

def create_folder(path):

    os.makedirs(path, exist_ok=True)

    log(f"Folder created {path}")


def create_file(path, content):

    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log_file_created(path)


def append_file(path, content):

    ensure_dir(os.path.dirname(path))

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + content)

    log_file_modified(path)


# -------------------------
# WHITESPACE
# -------------------------

def fix_whitespace():

    for root, dirs, files in os.walk("."):

        if ".git" in root:
            continue

        for file in files:

            if not file.endswith((".py",".txt",".md",".yml",".json",".ini")):
                continue

            path=os.path.join(root,file)

            with open(path,"r",encoding="utf-8",errors="ignore") as f:
                lines=f.readlines()

            new=[]

            for line in lines:

                line=line.rstrip()

                line=line.replace("\t","    ")

                new.append(line+"\n")

            while new and new[-1].strip()=="":
                new.pop()

            new.append("\n")

            with open(path,"w",encoding="utf-8") as f:
                f.writelines(new)

    log("Whitespace fixed")


# -------------------------
# FORMAT
# -------------------------

def run_black():

    log("Running Black")

    subprocess.run(["black","."],check=False)


def run_isort():

    log("Running isort")

    subprocess.run(["isort","."],check=False)


def run_ruff():

    log("Running ruff")

    subprocess.run(["ruff","check",".","--fix"],check=False)


# -------------------------
# TEST
# -------------------------

def run_pytest():

    log("Running pytest")

    result=subprocess.run(
        ["pytest","-v"],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode!=0:

        log("Tests FAILED")

        raise Exception("Tests failed")

    log("Tests PASSED")


# -------------------------
# PROCESS
# -------------------------

def process():

    if AUTO_BACKUP_ALWAYS:
        backup_repository()

    if not os.path.exists(INSTRUCTIONS_FILE):
        log("No instruction file")
        return

    with open(INSTRUCTIONS_FILE) as f:
        lines=f.readlines()

    i=0

    while i<len(lines):

        line=lines[i].strip()

        if not line:
            i+=1
            continue

        parts=line.split()

        cmd=parts[0]

        if cmd in ["CREA_CARTELLA","CREATE_FOLDER"]:

            create_folder(parts[1])

        elif cmd in ["CREA_FILE","CREATE_FILE"]:

            path=parts[1]

            i+=1

            content=[]

            while i<len(lines) and lines[i].strip()!="EOF":

                content.append(lines[i])

                i+=1

            create_file(path,"".join(content))

        elif cmd in ["APPEND","AGGIUNGI"]:

            path=parts[1]

            i+=1

            content=[]

            while i<len(lines) and lines[i].strip()!="EOF":

                content.append(lines[i])

                i+=1

            append_file(path,"".join(content))

        elif cmd=="FIX_WHITESPACE":

            fix_whitespace()

        i+=1

    try:

        fix_whitespace()

        run_black()

        run_isort()

        run_ruff()

        run_pytest()

    except Exception as e:

        log(f"CI FAILED: {e}")

        restore_last_backup()

        raise


if __name__=="__main__":
    process()