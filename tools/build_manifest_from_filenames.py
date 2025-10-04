from pathlib import Path
import re, json, sys

SUITS = {"denari","coppe","bastoni","spade"}
RANKS = {"A","3","K","C","F","7","6","5","4","2"}

def guess(fn:str):
    base = fn.lower()
    suit = next((s for s in SUITS if s in base), None)
    rank = next((r for r in RANKS if re.search(rf"(?:_|-){r.lower()}(?:\.|$)", base)), None)
    return suit, rank

def main():
    if len(sys.argv)<3:
        print("uso: python tools/build_manifest_from_filenames.py <templates_full_dir> <out_manifest.json>")
        sys.exit(1)
    tdir = Path(sys.argv[1])
    out = Path(sys.argv[2])

    mapping = {}
    for p in sorted(tdir.glob("*.png")):
        s,r = guess(p.name)
        mapping[p.name] = {"suit": s or "", "rank": r or ""}

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"full": mapping}, f, ensure_ascii=False, indent=2)
    print("scritto:", out)

if __name__=="__main__":
    main()
