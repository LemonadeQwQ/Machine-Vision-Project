"""顺序运行单实验五折；集群并行调度由内部作业脚本维护。"""
import argparse
import json
from pathlib import Path
from .experiment import configuration, run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--data-root')
    parser.add_argument('--weights')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or any(i not in range(5) for i in args.folds):
        parser.error('Use unique fold numbers 0..4')
    overrides = {k: v for k, v in {'data_root': args.data_root, 'weights': args.weights}.items() if v is not None}
    config = configuration(args.config, overrides)
    for fold in args.folds:
        target = args.output / f'fold_{fold}'
        # 不静默跳过已存在结果，避免配置改变后把旧折混进新实验。
        run(config, fold, target, smoke=args.smoke, resume=args.resume)
