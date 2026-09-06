"""Build macos/AppIcon.icns from the ClearCam camera glyph.

The glyph comes from images/android-chrome-512x512.png (white mark on blue);
it is lifted as an alpha mask and set on the app's own ink, inside the
macOS squircle at Apple's proportions (icon art fills 824 of 1024 px).
Run once when the mark changes; the .icns is committed so packaging needs
no image tooling.
"""
import subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'images/android-chrome-512x512.png'
TARGET = ROOT / 'macos/AppIcon.icns'
INK, PAPER = (15, 26, 28, 255), (232, 238, 235, 255)
SIZE, ART, RADIUS = 1024, 824, 186


def glyph_mask(size):
    src = Image.open(SOURCE).convert('RGBA')
    # white-ish pixels are the mark; everything else is the blue field
    r, g, b, a = src.split()
    mask = Image.eval(Image.merge('RGB', (r, g, b)).convert('L'), lambda v: 255 if v > 200 else 0)
    mask = Image.composite(mask, Image.new('L', src.size, 0), a)
    return mask.resize((size, size), Image.LANCZOS)


def render():
    canvas = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    plate = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    inset = (SIZE - ART) // 2
    ImageDraw.Draw(plate).rounded_rectangle((inset, inset, inset + ART, inset + ART), RADIUS, fill=INK)
    # a faint top light so the plate reads as a surface, not a flat swatch
    gradient = Image.new('L', (1, ART))
    for y in range(ART): gradient.putpixel((0, y), int(46 * (1 - y / ART)))
    light = Image.new('RGBA', (ART, ART), PAPER[:3] + (0,))
    light.putalpha(gradient.resize((ART, ART)))
    plate.alpha_composite(light, (inset, inset))
    plate = Image.composite(plate, Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0)), plate.split()[3])
    # shadow under the plate, as Finder expects of a pre-rendered icon
    shadow = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((inset, inset + 10, inset + ART, inset + ART + 10), RADIUS, fill=(0, 0, 0, 110))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    canvas.alpha_composite(plate)
    mark = glyph_mask(int(ART * 0.78))
    fill = Image.new('RGBA', mark.size, PAPER)
    fill.putalpha(mark)
    canvas.alpha_composite(fill, ((SIZE - mark.width) // 2, (SIZE - mark.height) // 2))
    return canvas


def main():
    art = render()
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / 'AppIcon.iconset'; iconset.mkdir()
        for px in (16, 32, 128, 256, 512):
            art.resize((px, px), Image.LANCZOS).save(iconset / f'icon_{px}x{px}.png')
            art.resize((px * 2, px * 2), Image.LANCZOS).save(iconset / f'icon_{px}x{px}@2x.png')
        subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(TARGET)], check=True)
    art.save(ROOT / 'macos/AppIcon-1024.png')
    print('wrote', TARGET, TARGET.stat().st_size, 'bytes')


if __name__ == '__main__':
    sys.exit(main())
