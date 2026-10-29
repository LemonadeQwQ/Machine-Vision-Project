"""生成真实状态表；未训练的实验保持 planned，短检查不充当正式结果。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.common import write_csv


def collect(root, output):
    root, output = Path(root), Path(output)
    rows = []
    for index in range(10):
        experiment = f'E{index:02}'
        for fold in range(5):
            file = root / experiment / f'fold_{fold}' / 'run.json'
            row = {'experiment': experiment, 'fold': fold, 'status': 'planned', 'best_macro_f1': '', 'best_epoch': '', 'reason': 'not started'}
            if file.exists():
                run = json.loads(file.read_text(encoding='utf-8'))
                if run.get('purpose') != 'formal' or not run.get('code', {}).get('source_sha256') or run.get('protocol') != 'resnet9-v1':
                    raise ValueError(f'Not a traceable formal result: {file}')
                if run['experiment'] != experiment or run['fold'] != fold:
                    raise ValueError('Result stored under wrong experiment/fold')
                row.update(status=run['status'], reason=run.get('error') or run.get('stop_reason') or '')
                if run['status'] == 'completed':
                    from src.evaluate import load_evaluation_data, calculate_metrics
                    from src.class_schema import NINE_CLASSES
                    _, truth, probs = load_evaluation_data(file.parent / 'validation_labels.csv', file.parent / 'predictions.csv',
                                                            class_names=NINE_CLASSES, id_column='sample_id')
                    value = calculate_metrics(truth, probs, class_names=NINE_CLASSES)[0]['macro_f1']
                    if abs(value - run['best_macro_f1']) > 1e-9:
                        raise ValueError('Score mismatch')
                    row.update(best_macro_f1=value, best_epoch=run['best_epoch'])
            rows.append(row)
    write_csv(output, rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.runs, args.output)
