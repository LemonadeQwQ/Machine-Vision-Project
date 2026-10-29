"""训练前相似图片审计。pHash 候选是保守隔离依据，不等于人工确认重复。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from .common import ROOT, digest, fingerprint, image_path, read_csv, write_csv, write_json


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            stack.extend(child for d, child in node[2].items() if distance - radius <= d <= distance + radius)


def hashes(root_row):
    root, row = root_row
    with Image.open(image_path(root, row['relative_path'])) as im:
        im = ImageOps.exif_transpose(im).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(im, dtype=np.float32)
    values = []
    for flipped in (pixels, np.fliplr(pixels)):
        for k in range(4):
            low = dctn(np.rot90(flipped, k), type=2, norm='ortho')[:8, :8]
            bits = low > np.median(low)
            values.append(f'{int.from_bytes(np.packbits(bits).tobytes(), "big"):016x}')
    return {**row, 'phash_d4': '|'.join(values)}


def audit(root, output, workers=8, radius=4):
    root, output = Path(root), Path(output)
    inputs = [root / '_audit' / f'{name}_Training_images.csv' for name in ('MILK10k', 'ISIC2019')]
    rows = []
    for path in inputs:
        rows.extend(read_csv(path))
    signature = fingerprint({'inputs': [digest(p) for p in inputs], 'method': 'phash64-d4-v1', 'radius': radius})
    cache = output / 'perceptual_hashes.csv'
    meta_path = output / 'perceptual_hashes.json'
    if cache.exists() and meta_path.exists() and json.loads(meta_path.read_text())['signature'] == signature:
        hashed = read_csv(cache)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashed = []
            for i, row in enumerate(pool.map(hashes, ((root, r) for r in rows)), 1):
                hashed.append(row)
                if i % 2500 == 0:
                    print(f'Images decoded: {i}/{len(rows)}', flush=True)
        write_csv(cache, hashed)
        write_json(meta_path, {'signature': signature, 'images': len(hashed), 'method': '64-bit pHash, eight rotations/reflections'})
    tree, edges = BKTree(), []
    for i, row in enumerate(hashed):
        matches = {}
        for value in set(row['phash_d4'].split('|')):
            for j, distance in tree.query(int(value, 16), radius):
                matches[j] = min(matches.get(j, 65), distance)
        edges.extend({'image_a': hashed[j]['image_id'], 'image_b': row['image_id'], 'distance': d} for j, d in sorted(matches.items()))
        tree.add(int(row['phash_d4'].split('|')[0], 16), i)
        if (i + 1) % 5000 == 0:
            print(f'Similarity search: {i+1}/{len(hashed)}, candidates={len(edges)}', flush=True)
    write_csv(output / 'similarity_candidates.csv', edges, ['image_a', 'image_b', 'distance'])
    write_json(output / 'similarity_audit.json', {'signature': signature, 'images': len(rows), 'candidate_pairs': len(edges),
               'radius': radius, 'method': '64-bit pHash with eight rotations/reflections',
               'limitations': 'Conservative candidates, not confirmed same patients. Crops and unrelated close-looking lesions can escape or collide.'})
    print(f'Audit finished: {len(edges)} candidate pairs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT.parent / 'DATASETS')
    parser.add_argument('--output', type=Path, default=ROOT / 'DATA/cache')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    audit(args.data_root, args.output, args.workers)
