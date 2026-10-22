from pathlib import Path
import json
from algorithms.prediction.cma_weather import cma_weather

result=cma_weather(39.89,116.32)
Path('D:/TreeLineWarning/verification_real_20260930_weststation/cma_online_actual_20261001.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
