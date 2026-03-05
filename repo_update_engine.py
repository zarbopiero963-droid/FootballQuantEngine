import datetime
import os
import shutil
import subprocess
import zipfile

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
LOG_DIR = "logs"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")

# ✅ Backup automatico ad ogni run (oltre al comando AUTO_BACKUP_BEFORE_RUN)
AUTO_BACKUP_ALWAYS = True

# ✅ Tieni solo gli ultimi N backup zip
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


def list_backup_zips():
    ensure_dir(BACKUP_DIR)
    zips = [
        os.path.join(BACKUP_DIR, f)
        for f in os.listdir(BACKUP_DIR)
        if f.startswith("backup_") and f.endswith(".zip")
    ]
    zips.sort()
    return zips


def cleanup_old_backups():
    zips = list_backup_zips()
    if len(zips) <= KEEP_LAST_BACKUPS:
        return
    to_delete = zips[:-KEEP_LAST_BACKUPS]
    for p in to_delete:
        try:
            os.remove(p)
            log(f"Deleted old backup {p}")
        except Exception as e:
            log(f"Failed deleting old backup {p}: {e}")


# ------------------------
# BACKUP (ZIP compresso)
# ------------------------


def backup_repository():
    """
    Backup ZIP del repository:
    - pesa meno grazie alla compressione
    - esclude: .git, backups, logs, __pycache__, *.pyc
    """
    ensure_dir(BACKUP_DIR)
    ts = timestamp()
    zip_path = os.path.join(BACKUP_DIR, f"backup_{ts}.zip")

    def should_skip(rel_path: str) -> bool:
        rel = rel_path.replace("\\", "/")
        if rel.startswith(".git/") or rel == ".git":
            return True
        if rel.startswith(f"{BACKUP_DIR}/") or rel == BACKUP_DIR:
            return True
        if rel.startswith(f"{LOG_DIR}/") or rel == LOG_DIR:
            return True
        if "/__pycache__/" in rel or rel.endswith("/__pycache__"):
            return True
        if rel.endswith(".pyc"):
            return True
        return False

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("."):
            rel_root = os.path.relpath(root, ".")
            rel_root = "." if rel_root == "." else rel_root.replace("\\", "/")

            keep_dirs = []
            for d in dirs:
                rel_dir = d if rel_root == "." else f"{rel_root}/{d}"
                if not should_skip(rel_dir):
                    keep_dirs.append(d)
            dirs[:] = keep_dirs

            for file in files:
                src = os.path.join(root, file)
                rel = os.path.relpath(src, ".").replace("\\", "/")
                if should_skip(rel):
                    continue
                zf.write(src, rel)

    log(f"Backup created {zip_path}")
    cleanup_old_backups()


def restore_latest_backup():
    """
    Ripristina l'ultimo backup zip sopra la working tree (senza toccare .git e backups).
    """
    zips = list_backup_zips()
    if not zips:
        log("No backup zip found -> cannot rollback")
        return False

    latest = zips[-1]
    log(f"ROLLBACK: restoring from {latest}")

    with zipfile.ZipFile(latest, "r") as zf:
        for member in zf.namelist():
            rel = member.replace("\\", "/")
            if rel.startswith(".git/") or rel.startswith(f"{BACKUP_DIR}/"):
                continue
            if rel.startswith(f"{LOG_DIR}/"):
                continue
            if member.endswith("/"):
                continue

            dest_path = os.path.join(".", rel)
            ensure_dir(os.path.dirname(dest_path))
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    log("ROLLBACK completed")
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
    """
    REPLACE_LINE: sostituisce una stringa SOLO dentro le righe (soft)
    Esempio:
    REPLACE_LINE pytest.ini testpaths=test testpaths=test tests
    """
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

    if not changed:
        log(f"REPLACE_LINE: pattern not found in {path} -> '{old}'")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out)
        log(f"REPLACE_LINE applied in {path}")


def insert_line(path, line_number, text):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = max(0, line_number - 1)
    if idx > len(lines):
        idx = len(lines)

    lines.insert(idx, text + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    log(f"Inserted line {line_number} in {path}")


# ------------------------
# CLEAN WHITESPACE
# ------------------------


def fix_whitespace():
    for root, dirs, files in os.walk("."):
        if ".git" in root:
            continue

        for file in files:
            if not file.endswith((".py", ".txt", ".md", ".yml", ".yaml", ".json", ".ini")):
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
        raise Exception("Tests failed")
    log("Tests PASSED")


# ------------------------
# PROCESS
# ------------------------


def process():
    if not os.path.exists(INSTRUCTIONS_FILE):
        log("No instruction file")
        return

    # ✅ Backup automatico ad ogni run (1 sola volta)
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

        # AUTO BACKUP (manuale esplicito)
        if cmd == "AUTO_BACKUP_BEFORE_RUN":
            backup_repository()

        # CREATE FOLDER
        elif cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:
            create_folder(parts[1])

        # CREATE FILE MULTILINE (termina con EOF)
        elif cmd in ["CREA_FILE", "CREATE_FILE"]:
            path = parts[1]
            i += 1
            content = []
            while i < len(lines) and lines[i].strip() != "EOF":
                content.append(lines[i])
                i += 1
            create_file(path, "".join(content))

        # APPEND MULTILINE (termina con EOF)
        elif cmd in ["AGGIUNGI", "APPEND"]:
            path = parts[1]
            i += 1
            content = []
            while i < len(lines) and lines[i].strip() != "EOF":
                content.append(lines[i])
                i += 1
            append_file(path, "".join(content))

        # ✅ REPLACE / SOSTITUISCI (soft, non riscrive tutto)
        # Sintassi:
        # REPLACE file_path <old...> <new...>
        # Nota: con spazi è difficile senza EOF -> usa REPLACE_LINE o APPEND se devi mettere spazi.
        elif cmd in ["REPLACE", "SOSTITUISCI"]:
            path = parts[1]
            old = parts[2]
            new = " ".join(parts[3:])
            replace_text(path, old, new)

        # ✅ REPLACE_LINE (PER pytest.ini ecc.)
        # Sintassi:
        # REPLACE_LINE pytest.ini "testpaths = test" "testpaths = test tests"
        # (se vuoi evitare virgolette, usa parole senza spazi)
        elif cmd == "REPLACE_LINE":
            path = parts[1]
            old = parts[2]
            new = " ".join(parts[3:])
            replace_line(path, old, new)

        # ✅ INSERT_LINE
        # INSERT_LINE file 10 testo...
        elif cmd in ["INSERT_LINE", "INSERISCI_RIGA"]:
            path = parts[1]
            line_no = int(parts[2])
            text = " ".join(parts[3:])
            insert_line(path, line_no, text)

        # FIX WHITESPACE
        elif cmd == "FIX_WHITESPACE":
            fix_whitespace()

        i += 1

    # ------------------------
    # CI PIPELINE
    # ------------------------
    try:
        fix_whitespace()
        run_black()
        run_isort()
        run_ruff()
        run_pytest()
    except Exception as e:
        log(f"CI FAILED -> {e}")
        ok = restore_latest_backup()
        if not ok:
            raise
        raise


if __name__ == "__main__":
    process()
