"""Bootstrap copied CPython/PDAL and package wheels without a local venv."""
from pathlib import Path
import os
import sys

PROGRAM = Path(__file__).resolve().parent
RUNTIME = PROGRAM/'runtime'
sys.path.insert(0, str(PROGRAM/'packages'))
sys.path.insert(0, str(PROGRAM))
sys.path.append(str(RUNTIME/'Lib/site-packages'))
_handles = []
for directory in (RUNTIME/'Library/bin', RUNTIME/'DLLs'):
    _handles.append(os.add_dll_directory(str(directory)))
os.environ.update(TREELINE_PDAL_RUNTIME_ENV=str(RUNTIME),
                  TREELINE_PDAL_PROJ_DATA_ROOT=str(RUNTIME/'Library/share/proj'),
                  PROJ_DATA=str(RUNTIME/'Library/share/proj'), PROJ_LIB=str(RUNTIME/'Library/share/proj'),
                  GDAL_DATA=str(RUNTIME/'Library/share/gdal'), PROJ_NETWORK='OFF')
os.environ['PDAL_DRIVER_PATH'] = str(RUNTIME/'Library/bin')
if sys.stdout is None:sys.stdout=open(os.devnull,'w')
if sys.stderr is None:sys.stderr=open(os.devnull,'w')

if '--self-test' in sys.argv:
    import importlib
    import json
    for name in ('numpy','scipy','laspy','lazrs','pdal','pyproj','fastapi','webview','openpyxl','onnxruntime','PIL'):
        module=importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(PROGRAM):
            raise RuntimeError('Dependency escaped portable folder: '+name)
    from desktop.pdal_runtime import dependency_health
    health=dependency_health()
    health['portable_python']=str(sys.executable)
    health['portable_prefix']=str(sys.prefix)
    path=PROGRAM.parent/'运行日志/portable-self-test.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(health,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(health,ensure_ascii=False),flush=True)
    raise SystemExit(0 if health.get('minimal_operation_pass') and health.get('proj_minimal_operation_pass') else 1)
elif '--audit-server' in sys.argv:
    import runpy
    project=sys.argv[sys.argv.index('--project-root')+1]
    output=sys.argv[sys.argv.index('--output-root')+1]
    sys.argv=['serve_offline_audit.py','B','--project',project,'--output-root',output,'--resource-root',str(PROGRAM)]
    runpy.run_path(str(PROGRAM/'scripts/portable_audit_server.py'),run_name='__main__')
else:
    from launch import main
    raise SystemExit(main())
