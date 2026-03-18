import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-image-csv', required=True)
    ap.add_argument('--out-txt', required=True)
    ap.add_argument('--fn-threshold', type=int, default=30)
    ap.add_argument('--topk', type=int, default=120)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.per_image_csv).open('r', encoding='utf-8')))
    rows = [r for r in rows if int(r['fn']) >= args.fn_threshold]
    rows.sort(key=lambda r: (-int(r['fn']), -int(r['abs_err'])))
    rows = rows[: args.topk]

    out = Path(args.out_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(r['image_id'] + '\n')

    print(f'hardcase_count={len(rows)}')
    print(f'out={out}')


if __name__ == '__main__':
    main()
