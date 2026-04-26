# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ["app/desktop_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
        "sqlalchemy",
        "openpyxl",
        "requests",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "PySide6.QtSvg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "matplotlib.backends",
        "tkinter",
        "_tkinter",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
        "jupyterlab",
        "pydoc",
        "doctest",
        "unittest",
        "black",
        "isort",
        "ruff",
        "pyinstaller",
        "pkg_resources._vendor",
        "setuptools",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FootballQuantEngine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=[],
    name="FootballQuantEngine",
)
