"""完整五折后按已约定的 best epoch 中位数 refit；不读取 holdout 指标。"""
import argparse
import json
from pathlib import Path
from .experiment import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    summary = json.loads(args.cv_summary.read_text(encoding='utf-8'))
    run(summary['config'], 0, args.output, resume=args.resume, refit_summary=summary)
