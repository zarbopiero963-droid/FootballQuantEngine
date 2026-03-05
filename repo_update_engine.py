import os
import ast
import shutil
import datetime
import difflib

INSTRUCTIONS_FILE = "devops_update.txt"

BACKUP_DIR = "backups"
PATCH_DIR = "patches"
LOG_DIR = "logs"

LOG_FILE = os.path.join(LOG_DIR, "operations.log")
HISTORY_FILE = os.path.join(BACKUP_DIR, "history.log")


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


def history(msg):
    ensure_dir(BACKUP_DIR)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def latest_backup_dir():
    if not os.path.exists(BACKUP_DIR):
        return None
    versions = [d for d in os.listdir(BACKUP_DIR) if os.path.isdir(os.path.join(BACKUP_DIR, d))]
    if not versions:
        return None
    versions.sort()
    return os.path.join(BACKUP_DIR, versions[-1])


# ------------------------
# BACKUP
# ------------------------

def backup_repository():
    ts = timestamp()
    root = os.path.join(BACKUP_DIR, ts)
    ensure_dir(root)

    for r, d, f in os.walk("."):
        if r.startswith("./" + BACKUP_DIR):
            continue
        for file in f:
            src = os.path.join(r, file)
            rel = os.path.relpath(src, ".")
            dst = os.path.join(root, rel)
            ensure_dir(os.path.dirname(dst))
            shutil.copy2(src, dst)

    history(f"BACKUP_REPOSITORY {ts}")
    log(f"Backup repository {ts}")


def backup_file(path):
    if not os.path.exists(path):
        return
    ts = timestamp()
    root = os.path.join(BACKUP_DIR, ts)
    dst = os.path.join(root, path)
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(path, dst)
    history(f"BACKUP_FILE {path} {ts}")
    log(f"Backup file {path}")


# ------------------------
# RESTORE
# ------------------------

def restore_file(path):
    root = latest_backup_dir()
    if not root:
        return
    src = os.path.join(root, path)
    if not os.path.exists(src):
        return
    ensure_dir(os.path.dirname(path))
    shutil.copy2(src, path)
    log(f"Restored file {path}")


def restore_repository():
    root = latest_backup_dir()
    if not root:
        return
    restore_version(os.path.basename(root))


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
    log(f"Restored version {version}")


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


def overwrite(path, content):
    backup_file(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"Overwrite {path}")


def append_file(path, content):
    backup_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + content)
    log(f"Append {path}")


def replace_text(path, old, new):
    backup_file(path)
    with open(path, encoding="utf-8") as f:
        data = f.read()
    data = data.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    log(f"Replace text in {path}")


def insert_line(path, line_number, text):
    backup_file(path)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    lines.insert(line_number - 1, text + "\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log(f"Insert line {path}:{line_number}")


def delete_file(path):
    backup_file(path)
    if os.path.exists(path):
        os.remove(path)
    log(f"Delete file {path}")


def move_file(src, dst):
    backup_file(src)
    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)
    log(f"Move {src} -> {dst}")


def rename_file(src, dst):
    backup_file(src)
    ensure_dir(os.path.dirname(dst))
    os.rename(src, dst)
    log(f"Rename {src} -> {dst}")


# ------------------------
# AST OPS
# ------------------------

def patch_function(file_path, func, code):
    backup_file(file_path)
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    class Patcher(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func:
                node.body.extend(ast.parse(code).body)
            return node

    tree = Patcher().visit(tree)
    ast.fix_missing_locations(tree)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(ast.unparse(tree))

    log(f"Patched function {func} in {file_path}")


def modify_function(file_path, func, new_body_code):
    backup_file(file_path)
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    class Modifier(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == func:
                node.body = ast.parse(new_body_code).body
            return node

    tree = Modifier().visit(tree)
    ast.fix_missing_locations(tree)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(ast.unparse(tree))

    log(f"Modified function {func} in {file_path}")


def delete_function(file_path, func):
    backup_file(file_path)
    with open(file_path, encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code)
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            continue
        new_body.append(node)

    tree.body = new_body
    ast.fix_missing_locations(tree)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(ast.unparse(tree))

    log(f"Deleted function {func} from {file_path}")


# ------------------------
# REFACTOR
# ------------------------

def refactor_repo(action, target, new):
    for r, d, f in os.walk("."):
        for file in f:
            if not file.endswith(".py"):
                continue
            path = os.path.join(r, file)
            with open(path, encoding="utf-8") as fh:
                code = fh.read()

            tree = ast.parse(code)

            class Refactor(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if action == "rename_function" and node.name == target:
                        node.name = new
                    return node

            tree = Refactor().visit(tree)
            ast.fix_missing_locations(tree)

            with open(path, "w", encoding="utf-8") as fh:
                fh.write(ast.unparse(tree))

    log(f"Refactor {target} -> {new}")


# ------------------------
# PATCH GENERATOR
# ------------------------

def generate_patch():
    ensure_dir(PATCH_DIR)
    patch_file = os.path.join(PATCH_DIR, f"patch_{timestamp()}.diff")

    root = latest_backup_dir()
    if not root:
        return

    diffs = []

    for r, d, f in os.walk("."):
        for file in f:
            if not file.endswith(".py"):
                continue
            path = os.path.join(r, file)
            backup = os.path.join(root, path)
            if not os.path.exists(backup):
                continue

            with open(path, encoding="utf-8") as f1:
                new = f1.readlines()
            with open(backup, encoding="utf-8") as f2:
                old = f2.readlines()

            diff = difflib.unified_diff(old, new, fromfile="old", tofile="new")
            diffs.extend(diff)

    with open(patch_file, "w", encoding="utf-8") as f:
        f.writelines(diffs)

    log(f"Patch generated {patch_file}")


# ------------------------
# CLEAN
# ------------------------

def fix_whitespace():
    for r, d, f in os.walk("."):
        for file in f:
            if not file.endswith((".py", ".txt", ".md", ".yml", ".json")):
                continue
            path = os.path.join(r, file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f1:
                lines = f1.readlines()

            new = []
            for line in lines:
                line = line.rstrip().replace("\t", "    ")
                new.append(line + "\n")

            while new and new[-1].strip() == "":
                new.pop()

            new.append("\n")

            with open(path, "w", encoding="utf-8") as f2:
                f2.writelines(new)

    log("Whitespace fixed")


# ------------------------
# INSTRUCTION PARSER
# ------------------------

def read_block(lines, start_index):
    """Reads a <<EOF ... EOF block."""
    header = lines[start_index].strip()
    if "<<EOF" not in header:
        return "", start_index

    i = start_index + 1
    content = []
    while i < len(lines) and lines[i].strip() != "EOF":
        content.append(lines[i])
        i += 1

    return "".join(content), i + 1


# ------------------------
# PROCESS
# ------------------------

def process():
    if not os.path.exists(INSTRUCTIONS_FILE):
        return

    with open(INSTRUCTIONS_FILE, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:
            create_folder(parts[1])

        elif cmd in ["CREA_FILE", "CREATE_FILE"]:
            content, i = read_block(lines, i)
            create_file(parts[1], content)
            continue

        elif cmd in ["SOVRASCRIVI", "OVERWRITE"]:
            content, i = read_block(lines, i)
            overwrite(parts[1], content)
            continue

        elif cmd in ["SOSTITUISCI", "REPLACE"]:
            replace_text(parts[1], parts[2], parts[3])

        elif cmd in ["AGGIUNGI", "APPEND"]:
            append_file(parts[1], " ".join(parts[2:]))

        elif cmd in ["INSERISCI_RIGA", "INSERT_LINE"]:
            insert_line(parts[1], int(parts[2]), " ".join(parts[3:]))

        elif cmd in ["ELIMINA_FILE", "DELETE_FILE"]:
            delete_file(parts[1])

        elif cmd in ["SPOSTA_FILE", "MOVE_FILE"]:
            move_file(parts[1], parts[2])

        elif cmd in ["RINOMINA_FILE", "RENAME_FILE"]:
            rename_file(parts[1], parts[2])

        elif cmd in ["PATCH_FUNZIONE", "PATCH_FUNCTION"]:
            patch_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["SAFE_PATCH_FUNZIONE", "SAFE_PATCH_FUNCTION"]:
            patch_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["MODIFICA_FUNZIONE", "MODIFY_FUNCTION"]:
            modify_function(parts[1], parts[2], " ".join(parts[3:]))

        elif cmd in ["ELIMINA_FUNZIONE", "DELETE_FUNCTION"]:
            delete_function(parts[1], parts[2])

        elif cmd == "REFACTOR_REPO":
            refactor_repo(parts[1], parts[2], parts[3])

        elif cmd == "BACKUP_REPOSITORY":
            backup_repository()

        elif cmd == "BACKUP_FILE":
            backup_file(parts[1])

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
                print(open(HISTORY_FILE, encoding="utf-8").read())

        i += 1


if __name__ == "__main__":
    process()

<<EOF