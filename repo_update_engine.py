import ast
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

CURRENT_COMMAND = "SYSTEM"


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


def set_current_command(cmd):
    global CURRENT_COMMAND
    CURRENT_COMMAND = cmd


def command_prefix():
    return f"[CMD: {CURRENT_COMMAND}]"


def log_file_created(path):
    log(f"{command_prefix()} [CREATE] {path}")


def log_file_modified(path):
    log(f"{command_prefix()} [MODIFY] {path}")


def log_file_deleted(path):
    log(f"{command_prefix()} [DELETE] {path}")


def log_already_done(msg):
    log(f"{command_prefix()} [SKIP] {msg} -> già fatto")


# -------------------------
# BACKUP
# -------------------------

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
            log(f"{command_prefix()} Deleted old backup {f}")
        except Exception as e:
            log(f"{command_prefix()} Cannot delete {f}: {e}")


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

                if rel.endswith(".pyc"):
                    continue

                z.write(src, rel)

    log(f"{command_prefix()} Backup created {zip_path}")
    cleanup_old_backups()
    return zip_path


def latest_backup():
    backups = list_backups()
    if not backups:
        return None
    return backups[-1]


def restore_backup(zip_path):
    if not zip_path or not os.path.exists(zip_path):
        log(f"{command_prefix()} Backup zip not found")
        return False

    log(f"{command_prefix()} RESTORE from {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")

    log(f"{command_prefix()} RESTORE completed")
    return True


def restore_last_backup():
    z = latest_backup()
    if not z:
        log(f"{command_prefix()} No backup available")
        return False
    return restore_backup(z)


def restore_version(version_name):
    zip_path = os.path.join(BACKUP_DIR, version_name)

    if not zip_path.endswith(".zip"):
        zip_path += ".zip"

    return restore_backup(zip_path)


def restore_file_from_zip(zip_path, file_path):
    if not zip_path or not os.path.exists(zip_path):
        log(f"{command_prefix()} Backup zip not found")
        return False

    normalized = file_path.replace("\\", "/")

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

        if normalized not in names:
            log(f"{command_prefix()} File not found in backup: {normalized}")
            return False

        ensure_dir(os.path.dirname(normalized))

        with z.open(normalized) as src, open(normalized, "wb") as dst:
            shutil.copyfileobj(src, dst)

    log(f"{command_prefix()} RESTORE FILE completed: {normalized}")
    return True


def restore_file_latest(file_path):
    z = latest_backup()
    if not z:
        log(f"{command_prefix()} No backup available")
        return False
    return restore_file_from_zip(z, file_path)


# -------------------------
# FILE OPS
# -------------------------

def create_folder(path):
    if os.path.exists(path):
        log_already_done(f"Folder exists {path}")
        return

    os.makedirs(path, exist_ok=True)
    log(f"{command_prefix()} Folder created {path}")


def create_file(path, content):
    ensure_dir(os.path.dirname(path))

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            current = f.read()

        if current == content:
            log_already_done(f"File already identical {path}")
            return

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log_file_created(path)


def append_file(path, content):
    ensure_dir(os.path.dirname(path))

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            current = f.read()

        normalized_content = content.strip()
        if normalized_content and normalized_content in current:
            log_already_done(f"Content already present in {path}")
            return

    with open(path, "a", encoding="utf-8") as f:
        if content and not content.startswith("\n"):
            f.write("\n")
        f.write(content)

    log_file_modified(path)


def overwrite_file(path, content):
    ensure_dir(os.path.dirname(path))

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            current = f.read()

        if current == content:
            log_already_done(f"File already identical {path}")
            return

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    log_file_modified(path)


def replace_text(path, old, new):
    if not os.path.exists(path):
        log(f"{command_prefix()} [WARN] File not found for replace: {path}")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    if old not in data:
        if new in data:
            log_already_done(f"Replace already applied in {path}")
        else:
            log(f"{command_prefix()} [WARN] Pattern not found in {path}: {old}")
        return

    data2 = data.replace(old, new)

    with open(path, "w", encoding="utf-8") as f:
        f.write(data2)

    log_file_modified(path)


def insert_line(path, line_number, text):
    if not os.path.exists(path):
        log(f"{command_prefix()} [WARN] File not found for insert line: {path}")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    idx = max(0, min(line_number - 1, len(lines)))

    if idx < len(lines) and lines[idx].rstrip("\n") == text:
        log_already_done(f"Line already present in {path}:{line_number}")
        return

    lines.insert(idx, text + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    log_file_modified(path)


def delete_file(path):
    if not os.path.exists(path):
        log_already_done(f"File already deleted {path}")
        return

    os.remove(path)
    log_file_deleted(path)


def move_file(src, dst):
    if not os.path.exists(src):
        if os.path.exists(dst):
            log_already_done(f"Move already applied {src} -> {dst}")
            return
        raise FileNotFoundError(src)

    ensure_dir(os.path.dirname(dst))
    shutil.move(src, dst)
    log(f"{command_prefix()} [MOVE] {src} -> {dst}")


def rename_file(src, dst):
    if not os.path.exists(src):
        if os.path.exists(dst):
            log_already_done(f"Rename already applied {src} -> {dst}")
            return
        raise FileNotFoundError(src)

    ensure_dir(os.path.dirname(dst))
    os.rename(src, dst)
    log(f"{command_prefix()} [RENAME] {src} -> {dst}")


def replace_line_block(path, old_block, new_block):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    if old_block not in data:
        if new_block in data:
            log_already_done(f"Block already replaced in {path}")
            return
        log(f"{command_prefix()} [WARN] old block not found in {path}")
        return

    data2 = data.replace(old_block, new_block, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(data2)

    log_file_modified(path)


# -------------------------
# AST OPS
# -------------------------

def _read_python_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_python_file(path, tree):
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code + "\n")
    log_file_modified(path)


def patch_function(path, func_name, patch_code):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    source = _read_python_file(path)
    tree = ast.parse(source)

    patch_nodes = ast.parse(patch_code).body
    patch_text = ast.unparse(ast.Module(body=patch_nodes, type_ignores=[])).strip()

    patched = False
    already_done = False

    class FunctionPatcher(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            nonlocal patched, already_done
            if node.name == func_name:
                current_body = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if patch_text in current_body:
                    already_done = True
                    return node
                node.body.extend(patch_nodes)
                patched = True
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            nonlocal patched, already_done
            if node.name == func_name:
                current_body = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if patch_text in current_body:
                    already_done = True
                    return node
                node.body.extend(patch_nodes)
                patched = True
            return self.generic_visit(node)

    tree = FunctionPatcher().visit(tree)

    if already_done:
        log_already_done(f"Patch already present in {path}:{func_name}")
        return

    if not patched:
        raise Exception(f"Function not found: {func_name} in {path}")

    _write_python_file(path, tree)
    log(f"{command_prefix()} [PATCH_FUNCTION] {path}:{func_name}")


def safe_patch_function(path, func_name, patch_code):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        ast.parse(patch_code)
    except Exception as e:
        raise Exception(f"Invalid patch code: {e}")

    source = _read_python_file(path)
    original_tree = ast.parse(source)

    patch_nodes = ast.parse(patch_code).body
    patch_text = ast.unparse(ast.Module(body=patch_nodes, type_ignores=[])).strip()

    patched = False
    already_done = False

    class SafeFunctionPatcher(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            nonlocal patched, already_done
            if node.name == func_name:
                current_body = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if patch_text in current_body:
                    already_done = True
                    return node
                node.body.extend(patch_nodes)
                patched = True
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            nonlocal patched, already_done
            if node.name == func_name:
                current_body = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if patch_text in current_body:
                    already_done = True
                    return node
                node.body.extend(patch_nodes)
                patched = True
            return self.generic_visit(node)

    new_tree = SafeFunctionPatcher().visit(original_tree)

    if already_done:
        log_already_done(f"Safe patch already present in {path}:{func_name}")
        return

    if not patched:
        raise Exception(f"Function not found: {func_name} in {path}")

    ast.fix_missing_locations(new_tree)
    generated = ast.unparse(new_tree)
    ast.parse(generated)

    with open(path, "w", encoding="utf-8") as f:
        f.write(generated + "\n")

    log_file_modified(path)
    log(f"{command_prefix()} [SAFE_PATCH_FUNCTION] {path}:{func_name}")


def refactor_repo(action, old_name, new_name):
    if action != "rename_function":
        raise Exception(f"Unsupported refactor action: {action}")

    changed_files = 0

    for root, dirs, files in os.walk("."):
        if ".git" in root:
            continue
        if BACKUP_DIR in root:
            continue

        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)

            with open(path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            changed = False

            class RefactorFunctions(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    nonlocal changed
                    if node.name == old_name:
                        node.name = new_name
                        changed = True
                    return self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    nonlocal changed
                    if node.name == old_name:
                        node.name = new_name
                        changed = True
                    return self.generic_visit(node)

                def visit_Call(self, node):
                    nonlocal changed
                    if isinstance(node.func, ast.Name) and node.func.id == old_name:
                        node.func.id = new_name
                        changed = True
                    return self.generic_visit(node)

            tree = RefactorFunctions().visit(tree)

            if changed:
                ast.fix_missing_locations(tree)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(ast.unparse(tree) + "\n")
                log_file_modified(path)
                changed_files += 1

    if changed_files == 0:
        log_already_done(f"Refactor already applied or nothing to change ({old_name} -> {new_name})")
        return

    log(f"{command_prefix()} [REFACTOR_REPO] action={action} old={old_name} new={new_name} changed_files={changed_files}")


# -------------------------
# WHITESPACE
# -------------------------

def fix_whitespace():
    for root, dirs, files in os.walk("."):
        if ".git" in root.split(os.sep):
            continue
        if root.startswith(f".{os.sep}{BACKUP_DIR}"):
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

    log(f"{command_prefix()} Whitespace fixed")


# -------------------------
# FORMAT + LINT + TEST
# -------------------------

def run_black():
    log(f"{command_prefix()} Running Black")
    subprocess.run(["black", "."], check=False)


def run_isort():
    log(f"{command_prefix()} Running isort")
    subprocess.run(["isort", "."], check=False)


def run_ruff():
    log(f"{command_prefix()} Running ruff")
    subprocess.run(["ruff", "check", ".", "--fix"], check=False)


def run_pytest():
    log(f"{command_prefix()} Running pytest")

    result = subprocess.run(
        ["pytest", "-v"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        log(f"{command_prefix()} Tests FAILED")
        return False

    log(f"{command_prefix()} Tests PASSED")
    return True


# -------------------------
# PARSER HELPERS
# -------------------------

def read_block(lines, start_index):
    content = []
    i = start_index

    while i < len(lines) and lines[i].strip() != "EOF":
        content.append(lines[i])
        i += 1

    return "".join(content), i


# -------------------------
# PROCESS
# -------------------------

def process():
    set_current_command("AUTO_BACKUP_ALWAYS")
    if AUTO_BACKUP_ALWAYS:
        backup_repository()

    if not os.path.exists(INSTRUCTIONS_FILE):
        log("No instruction file")
        return

    with open(INSTRUCTIONS_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        cmd = parts[0]
        set_current_command(line)

        if cmd == "AUTO_BACKUP_BEFORE_RUN":
            backup_repository()
            i += 1
            continue

        if cmd in ["CREA_CARTELLA", "CREATE_FOLDER"]:
            create_folder(parts[1])
            i += 1
            continue

        if cmd in ["CREA_FILE", "CREATE_FILE"]:
            path = parts[1]
            i += 1
            content, i = read_block(lines, i)
            create_file(path, content)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1
            continue

        if cmd in ["AGGIUNGI", "APPEND"]:
            path = parts[1]
            i += 1
            content, i = read_block(lines, i)
            append_file(path, content)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1
            continue

        if cmd in ["SOVRASCRIVI", "OVERWRITE"]:
            path = parts[1]
            i += 1
            content, i = read_block(lines, i)
            overwrite_file(path, content)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1
            continue

        if cmd in ["SOSTITUISCI", "REPLACE"]:
            path = parts[1]
            old = parts[2]
            new = " ".join(parts[3:])
            replace_text(path, old, new)
            i += 1
            continue

        if cmd in ["INSERISCI_RIGA", "INSERT_LINE"]:
            path = parts[1]
            line_no = int(parts[2])
            text = " ".join(parts[3:])
            insert_line(path, line_no, text)
            i += 1
            continue

        if cmd in ["REPLACE_LINE", "SOSTITUISCI_BLOCCO"]:
            path = parts[1]

            i += 1
            old_block, i = read_block(lines, i)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1

            new_block, i = read_block(lines, i)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1

            replace_line_block(path, old_block, new_block)
            continue

        if cmd in ["SPOSTA_FILE", "MOVE_FILE"]:
            move_file(parts[1], parts[2])
            i += 1
            continue

        if cmd in ["RINOMINA_FILE", "RENAME_FILE"]:
            rename_file(parts[1], parts[2])
            i += 1
            continue

        if cmd in ["ELIMINA_FILE", "DELETE_FILE"]:
            delete_file(parts[1])
            i += 1
            continue

        if cmd in ["RIPRISTINA_ULTIMO", "RESTORE_LAST", "ROLLBACK_LAST"]:
            restore_last_backup()
            i += 1
            continue

        if cmd in ["RIPRISTINA_FILE", "RESTORE_FILE"]:
            restore_file_latest(parts[1])
            i += 1
            continue

        if cmd in ["RIPRISTINA_VERSIONE", "RESTORE_VERSION"]:
            restore_version(parts[1])
            i += 1
            continue

        if cmd in ["PATCH_FUNCTION"]:
            path = parts[1]
            func_name = parts[2]
            i += 1
            patch_code, i = read_block(lines, i)
            patch_function(path, func_name, patch_code)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1
            continue

        if cmd in ["SAFE_PATCH_FUNCTION"]:
            path = parts[1]
            func_name = parts[2]
            i += 1
            patch_code, i = read_block(lines, i)
            safe_patch_function(path, func_name, patch_code)
            if i < len(lines) and lines[i].strip() == "EOF":
                i += 1
            continue

        if cmd in ["REFACTOR_REPO"]:
            action = parts[1]
            old_name = parts[2]
            new_name = parts[3]
            refactor_repo(action, old_name, new_name)
            i += 1
            continue

        if cmd == "FIX_WHITESPACE":
            fix_whitespace()
            i += 1
            continue

        log(f"{command_prefix()} [WARN] Unknown command")
        i += 1

    try:
        set_current_command("CI_PIPELINE")
        fix_whitespace()
        run_black()
        run_isort()
        run_ruff()

        ok = run_pytest()
        if not ok:
            raise RuntimeError("Tests failed")

    except Exception as e:
        set_current_command("ROLLBACK")
        log(f"{command_prefix()} CI FAILED: {e}")
        rollback_last_backup()
        raise


if __name__ == "__main__":
    process()