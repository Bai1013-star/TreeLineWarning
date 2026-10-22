from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root = Path('D:/TreeLineWarning/output/pdf')
pages = sorted(root.glob('manual-latest-*.png'))
assert len(pages) == 17
for offset in range(0, len(pages), 6):
    sheet = Image.new('RGB', (1500, 1440), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(pages[offset:offset+6]):
        thumbnail = ImageOps.contain(Image.open(path).convert('RGB'), (480, 680))
        x, y = (index % 3)*500, (index // 3)*720
        sheet.paste(thumbnail, (x, y+30))
        draw.text((x+10, y+8), path.stem, fill='black')
    sheet.save(root/f'latest-sheet-{offset//6}.png')
