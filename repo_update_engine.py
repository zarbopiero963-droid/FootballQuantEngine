import os
import ast
import shutil
import datetime
import difflib
import sys

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
PATCH_DIR = "patches"
LOG_FILE = "logs/operations.log"
HISTORY_FILE = "backups/history.log"

DRY_RUN = "--dry-run" in sys.argv


# -------------------------
# UTIL
# -------------------------

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def log(msg):
    ensure_dir("logs")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\n")

    print(msg)


def history(msg):
    ensure_dir("backups")

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{now()} {msg}\n")


# -------------------------
# BACKUP
# -------------------------

def version_backup_path():

    ts = timestamp()

    path = os.path.join(BACKUP_DIR, ts)

    os.makedirs(path, exist_ok=True)

    return path, ts


def backup_file(path):

    if not os.path.exists(path):
        return

    root, ts = version_backup_path()

    dst = os.path.join(root, path)

    ensure_dir(os.path.dirname(dst))

    shutil.copy2(path, dst)

    log(f"backup file {path}")
    history(f"BACKUP_FILE {path}")


def backup_repository():

    root, ts = version_backup_path()

    for r, d, f in os.walk("."):

        if r.startswith("./backups"):
            continue

        for file in f:

            src = os.path.join(r, file)

            rel = os.path.relpath(src, ".")

            dst = os.path.join(root, rel)

            ensure_dir(os.path.dirname(dst))

            shutil.copy2(src, dst)

    log(f"backup repo {ts}")
    history(f"BACKUP_REPOSITORY {ts}")


# -------------------------
# RESTORE
# -------------------------

def restore_file(path):

    versions = sorted(os.listdir(BACKUP_DIR))

    if not versions:
        return

    latest = versions[-1]

    src = os.path.join(BACKUP_DIR, latest, path)

    if not os.path.exists(src):
        return

    ensure_dir(os.path.dirname(path))

    shutil.copy2(src, path)

    log(f"restore file {path}")


def restore_repository():

    versions = sorted(os.listdir(BACKUP_DIR))

    if not versions:
        return

    latest = versions[-1]

    restore_version(latest)


def restore_version(version):

    root = os.path.join(BACKUP_DIR, version)

    if not os.path.exists(root):
        return

    for r, d, f in os.walk(root):

        for file in f:

            src = os.path.join(r, file)

            rel = os.path.relpath(src, root)

            dst = rel

            ensure_dir(os.path.dirname(dst))

            shutil.copy2(src, dst)

    log(f"restore version {version}")
    history(f"RESTORE_VERSION {version}")


# -------------------------
# FILE OPS
# -------------------------

def create_folder(path):
    os.makedirs(path, exist_ok=True)
    log(f"folder created {path}")


def create_file(path, content):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"file created {path}")


def overwrite(path, content):
    backup_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"overwrite {path}")


def append(path, content):
    backup_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + content)
    log(f"append {path}")


def replace(path, old, new):

    backup_file(path)

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    data = data.replace(old, new)

    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

    log(f"replace text {path}")


def insert_line(path, line_number, text):

    backup_file(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines.insert(line_number - 1, text + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    log(f"insert line {path}")


def delete_file(path):

    backup_file(path)

    if os.path.exists(path):
        os.remove(path)

    log(f"delete file {path}")


def move_file(src, dst):

    backup_file(src)

    ensure_dir(os.path.dirname(dst))

    shutil.move(src, dst)

    log(f"move {src} -> {dst}")


def rename_file(src, dst):

    backup_file(src)

    os.rename(src, dst)

    log(f"rename {src} -> {dst}")


# -------------------------
# AST OPS
# -------------------------

def modify_function(file_path, func, new_body):

    backup_file(file_path)

    with open(file_path) as f:
        code = f.read()

    tree = ast.parse(code)

    class Modifier(ast.NodeTransformer):

        def visit_FunctionDef(self, node):

            if node.name == func:
                node.body = ast.parse(new_body).body

            return node

    tree = Modifier().visit(tree)

    with open(file_path, "w") as f:
        f.write(ast.unparse(tree))

    log(f"modify function {func}")


def patch_function(file_path, func, code):

    backup_file(file_path)

    with open(file_path) as f:
        source = f.read()

    tree = ast.parse(source)

    class Patcher(ast.NodeTransformer):

        def visit_FunctionDef(self, node):

            if node.name == func:
                node.body.extend(ast.parse(code).body)

            return node

    tree = Patcher().visit(tree)

    with open(file_path, "w") as f:
        f.write(ast.unparse(tree))

    log(f"patch function {func}")


def safe_patch_function(file_path, func, code):

    try:
        patch_function(file_path, func, code)
    except Exception as e:
        log(f"safe patch failed {e}")


def delete_function(file_path, func):

    backup_file(file_path)

    with open(file_path) as f:
        code = f.read()

    tree = ast.parse(code)

    new_body = []

    for node in tree.body:

        if isinstance(node, ast.FunctionDef) and node.name == func:
            continue

        new_body.append(node)

    tree.body = new_body

    with open(file_path, "w") as f:
        f.write(ast.unparse(tree))

    log(f"delete function {func}")


# -------------------------
# REFACTOR
# -------------------------

def refactor_repo(action, target, new):

    for r, d, f in os.walk("."):

        for file in f:

            if not file.endswith(".py"):
                continue

            path = os.path.join(r, file)

            with open(path) as f:
                code = f.read()

            tree = ast.parse(code)

            class Refactor(ast.NodeTransformer):

                def visit_FunctionDef(self, node):

                    if action == "rename_function" and node.name == target:
                        node.name = new

                    return node

            tree = Refactor().visit(tree)

            with open(path, "w") as f:
                f.write(ast.unparse(tree))

    log(f"refactor {target}->{new}")


# -------------------------
# PATCH
# -------------------------

def generate_patch():

    ensure_dir(PATCH_DIR)

    patch_file = os.path.join(PATCH_DIR, f"patch_{timestamp()}.diff")

    diffs = []

    versions = sorted(os.listdir(BACKUP_DIR))

    if not versions:
        return

    latest = versions[-1]

    for r, d, f in os.walk("."):

        for file in f:

            if not file.endswith(".py"):
                continue

            path = os.path.join(r, file)

            backup = os.path.join(BACKUP_DIR, latest, path)

            if not os.path.exists(backup):
                continue

            with open(path) as f:
                new = f.readlines()

            with open(backup) as f:
                old = f.readlines()

            diff = difflib.unified_diff(old, new)

            diffs.extend(diff)

    with open(patch_file, "w") as f:
        f.writelines(diffs)

    log(f"patch generated {patch_file}")


# -------------------------
# CLEAN
# -------------------------

def fix_whitespace():

    for r, d, f in os.walk("."):

        for file in f:

            if not file.endswith((".py", ".txt", ".md", ".yml", ".json")):
                continue

            path = os.path.join(r, file)

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

    log("whitespace fixed")


# -------------------------
# PROCESS
# -------------------------

def process():

    if not os.path.exists(INSTRUCTIONS_FILE):
        log("instruction file missing")
        return

    with open(INSTRUCTIONS_FILE) as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if not parts:
            continue

        cmd = parts[0]

        if cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:
            create_folder(parts[1])

        elif cmd in ["CREA_FILE", "CREATE_FILE"]:
            create_file(parts[1], "")

        elif cmd in ["SOVRASCRIVI", "OVERWRITE"]:
            overwrite(parts[1], "")

        elif cmd in ["AGGIUNGI", "APPEND"]:
            append(parts[1], " ".join(parts[2:]))

        elif cmd in ["SOSTITUISCI", "REPLACE"]:
            replace(parts[1], parts[2], parts[3])

        elif cmd in ["INSERISCI_RIGA", "INSERT_LINE"]:
            insert_line(parts[1], int(parts[2]), " ".join(parts[3:]))

        elif cmd in ["DELETE_FILE", "ELIMINA_FILE"]:
            delete_file(parts[1])

        elif cmd in ["MOVE_FILE", "SPOSTA_FILE"]:
            move_file(parts[1], parts[2])

        elif cmd in ["RENAME_FILE", "RINOMINA_FILE"]:
            rename_file(parts[1], parts[2])

        elif cmd in ["MODIFY_FUNCTION", "MODIFICA_FUNZIONE"]:
            modify_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["PATCH_FUNCTION", "PATCH_FUNZIONE"]:
            patch_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["SAFE_PATCH_FUNCTION", "SAFE_PATCH_FUNZIONE"]:
            safe_patch_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["DELETE_FUNCTION", "ELIMINA_FUNZIONE"]:
            delete_function(parts[1], parts[2])

        elif cmd == "REFACTOR_REPO":
            refactor_repo(parts[1], parts[2], parts[3])

        elif cmd == "BACKUP_FILE":
            backup_file(parts[1])

        elif cmd == "BACKUP_REPOSITORY":
            backup_repository()

        elif cmd == "RIPRISTINA_FILE":
            restore_file(parts[1])

        elif cmd == "RIPRISTINA_REPOSITORY":
            restore_repository()

        elif cmd == "RIPRISTINA_VERSIONE":
            restore_version(parts[1])

        elif cmd == "GENERATE_PATCH":
            generate_patch()

        elif cmd == "FIX_WHITESPACE":
            fix_whitespace()

        elif cmd == "STORIA":

            if os.path.exists(HISTORY_FILE):

                with open(HISTORY_FILE) as f:
                    print(f.read())


if __name__ == "__main__":
    process()
