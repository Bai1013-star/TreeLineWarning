"""Render numeric original-source evidence; requires no app installation."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
root = Path(sys.argv[1])
with np.load(root/'pole_T002_original_inspection.npz', allow_pickle=False) as data:
    xyz, base = data['xyz'], data['base']
p = xyz[np.linspace(0, len(xyz)-1, min(len(xyz), 120000), dtype=int)]-base
image = Image.new('RGB', (1500, 850), '#ffffff')
draw = ImageDraw.Draw(image)
for number, (i,j) in enumerate([(0,2),(1,2),(0,1)]):
    left, top, right, bottom = number*500+55, 65, number*500+460, 775
    lo, hi = p[:,[i,j]].min(axis=0), p[:,[i,j]].max(axis=0)
    hi = np.maximum(hi,lo+1e-6)
    draw.rectangle((left,top,right,bottom), outline='#444444')
    draw.text((left,20),'Original T002: '+ 'XYZ'[i]+' vs '+'XYZ'[j],fill='#000000')
    for t in np.linspace(0,1,6):
        x = left+t*(right-left); y=bottom-t*(bottom-top)
        draw.line((x,top,x,bottom),fill='#dddddd')
        draw.line((left,y,right,y),fill='#dddddd')
        draw.text((x-10,bottom+10),f'{lo[0]+t*(hi[0]-lo[0]):.2f}',fill='#222222')
        draw.text((left-42,y),f'{lo[1]+t*(hi[1]-lo[1]):.2f}',fill='#222222')
    mapped=(p[:,[i,j]]-lo)/(hi-lo)
    for row, z in zip(mapped,p[:,2]):
        x=left+row[0]*(right-left); y=bottom-row[1]*(bottom-top)
        color=(50, int(70+140*np.clip(z/8,0,1)), 170)
        draw.point((x,y),fill=color)
    draw.text((left,810),'Coordinates relative to confirmed seed, metres',fill='#222222')
image.save(root/'pole_T002_original_inspection.png')
print({'original_points': len(xyz), 'plot_points': len(p)})
