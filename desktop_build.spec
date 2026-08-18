# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("algorithms")
    + collect_submodules("core")
    + ["scripts", "scripts.serve_p3a_giro3d", "scripts.tree_viewer_api"]
)
datas = [
    ("viewer_poc", "viewer_poc"),
    ("scripts/serve_p3a_giro3d.py", "scripts"),
    ("models/species_growth.json", "models"),
    ("models/temporal_attention.npz", "models"),
]

a = Analysis(
    ["desktop/app.py"], pathex=["."], binaries=[], datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TreeLineWarning", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TreeLineWarning")
