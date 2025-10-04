# tools/slice_sheet.py
from PIL import Image
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", help="Immagine con tutte le carte (4x10)")
    ap.add_argument("--out", default="src/briscola_ai/vision/templates/full", help="cartella output")
    args = ap.parse_args()

    img = Image.open(args.sheet).convert("RGB")
    W,H = img.size
    rows, cols = 4, 10
    cell_w, cell_h = W//cols, H//rows

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    n=0
    for r in range(rows):
        for c in range(cols):
            x0 = c*cell_w + 6     # piccolo margine per evitare bordi arrotondati
            y0 = r*cell_h + 6
            x1 = (c+1)*cell_w - 6
            y1 = (r+1)*cell_h - 6
            crop = img.crop((x0,y0,x1,y1))
            # normalizza a dimensione fissa (opzionale, aiuta il matching)
            crop = crop.resize((120, 180), Image.BICUBIC)
            fname = outdir / f"row{r}_col{c}.png"
            crop.save(fname)
            n += 1
    print(f"Salvate {n} carte in {outdir}")

if __name__ == "__main__":
    main()
