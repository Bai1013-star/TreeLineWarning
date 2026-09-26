# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / "desktop_launcher.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "web" / "templates" / "index.html"), "web/templates"),
        (str(project_root / "web" / "templates" / "help.html"), "web/templates"),
        (str(project_root / "web" / "static" / "china_regions.json"), "web/static"),
        (str(project_root / "web" / "static" / "leaflet.js"), "web/static"),
        (str(project_root / "web" / "static" / "leaflet.css"), "web/static"),
        (str(project_root / "models" / "temporal_attention.npz"), "models"),
        (str(project_root / "models" / "species_growth.json"), "models"),
    ],
    hiddenimports=[
        "multipart",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "webview",
        "webview.platforms.edgechromium",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="TreeLineWarning",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
