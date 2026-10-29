"""只从 torchvision 固定官方地址下载 ImageNet V2；校验命名哈希前缀。"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.common import ROOT, digest, write_json

if __name__ == '__main__':
    from torchvision.models import ResNet50_Weights
    url = ResNet50_Weights.IMAGENET1K_V2.url
    if url != 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth':
        raise RuntimeError(f'Unexpected weights URL: {url}')
    target = ROOT / 'pretrained_weights/resnet50-11ad3fa6.pth'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        temp = target.with_suffix('.part')
        urllib.request.urlretrieve(url, temp)
        if not digest(temp).startswith('11ad3fa6'):
            raise ValueError('Official weights hash mismatch')
        temp.replace(target)
    sha = digest(target)
    if not sha.startswith('11ad3fa6'):
        raise ValueError('Weights hash mismatch')
    record = {'name': 'ResNet50_Weights.IMAGENET1K_V2', 'url': url, 'sha256': sha, 'bytes': target.stat().st_size}
    write_json(ROOT / 'configs/schema/pretrained_weights.json', record)
    print(json.dumps(record, indent=2))
