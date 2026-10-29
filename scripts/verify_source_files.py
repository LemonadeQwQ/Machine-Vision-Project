"""核验已准备的官方原始文件；不会下载、解压或启动模型。"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root, selected=None, tables_only=False):
    root = Path(root).resolve()
    records = json.loads((ROOT / 'configs/data/source_files.json').read_text(encoding='utf-8'))['files']
    if selected:
        known = {r['file'] for r in records}
        if set(selected) - known:
            raise ValueError('Unknown source file requested')
        records = [r for r in records if r['file'] in selected]
    if tables_only:
        records = [r for r in records if r['file'].endswith('.csv')]
    for record in records:
        path = (root / record['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Missing / changed file: {record["file"]}')
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        if sha != record['sha256']:
            raise ValueError(f'SHA256 mismatch: {record["file"]}')
        print(f'Verified {record["file"]}', flush=True)
    return len(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'DATASETS')
    parser.add_argument('--file', action='append')
    parser.add_argument('--tables-only', action='store_true')
    args = parser.parse_args()
    verify(args.data_root, args.file, args.tables_only)
