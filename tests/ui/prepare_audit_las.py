from pathlib import Path
import os
import laspy
import numpy as np

root=Path(os.environ.get('REAL_OUTPUT', 'D:/TreeLineWarning/verification_real_20260930_weststation'))
source=Path('E:/treeline/9.20北京西站南路-1_colorized.las')
with laspy.open(source) as reader:
    points=reader.read_points(10000)
    cloud=laspy.LasData(reader.header.copy(),points=points)
    cloud.write(root/'audit_recognition.las')
print('Saved 10000 actual source points as recognition plumbing fixture; no species truth asserted.')
