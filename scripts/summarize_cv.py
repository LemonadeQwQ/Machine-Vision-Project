"""完整五折才导出正式 OOF：均值/样本标准差和 pooled OOF 分别报告。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.class_schema import NINE_CLASSES
from src.data.common import digest, fingerprint, read_csv, write_csv, write_json
from src.evaluate import evaluate_files, load_evaluation_data


def summarize(run_root, manifests, output):
    run_root, manifests, output = Path(run_root), Path(manifests), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    dev = read_csv(manifests / 'milk_dev.csv')
    runs, probabilities = [], []
    for fold in range(5):
        folder = run_root / f'fold_{fold}'
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if run['status'] != 'completed' or run['purpose'] != 'formal' or run['fold'] != fold or run.get('phase', 'cross_validation') != 'cross_validation':
            raise ValueError('Only completed formal CV folds may be summarized')
        if not run['code'].get('source_sha256') or run['inputs']['manifest_sha256'] != digest(manifests / 'version.json'):
            raise ValueError('Missing code provenance or different manifest version')
        if runs and any(run[k] != runs[0][k] for k in ('config', 'protocol', 'experiment')):
            raise ValueError('Mixed experiment / config / protocol')
        if runs and run['code']['source_sha256'] != runs[0]['code']['source_sha256']:
            raise ValueError('Mixed source versions')
        _, labels, probs = load_evaluation_data(manifests / f'folds/fold_{fold}/val.csv', folder / 'predictions.csv', class_names=NINE_CLASSES, id_column='sample_id')
        from src.evaluate import calculate_metrics
        metrics = calculate_metrics(labels, probs, class_names=NINE_CLASSES)[0]
        if abs(metrics['macro_f1'] - run['best_macro_f1']) > 1e-9:
            raise ValueError('Recorded score differs from prediction CSV')
        run['verified_metrics'] = metrics
        probabilities.extend(read_csv(folder / 'predictions.csv'))
        runs.append(run)
    if Counter(r['sample_id'] for r in probabilities) != Counter(r['sample_id'] for r in dev):
        raise ValueError('OOF must cover every development sample exactly once')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'oof_predictions.csv', probabilities, ['sample_id', *NINE_CLASSES])
    pooled = evaluate_files(manifests / 'milk_dev.csv', output / 'oof_predictions.csv', output / 'pooled_oof',
        model_name=runs[0]['experiment'] + ' pooled OOF', class_names=NINE_CLASSES, id_column='sample_id')
    metrics = {key: {'mean': statistics.mean(r['verified_metrics'][key] for r in runs),
                    'sample_std': statistics.stdev(r['verified_metrics'][key] for r in runs)}
               for key in runs[0]['verified_metrics']}
    summary = {'status': 'completed', 'purpose': 'formal', 'fold_count': 5, 'protocol': runs[0]['protocol'],
        'experiment': runs[0]['experiment'], 'config': runs[0]['config'], 'code_sha256': runs[0]['code']['source_sha256'],
        'manifest_sha256': runs[0]['inputs']['manifest_sha256'], 'fold_metrics': metrics,
        'pooled_oof_macro_f1': pooled['macro_f1'], 'n_oof': len(probabilities),
        'refit_epochs': int(statistics.median(r['best_epoch'] for r in runs)),
        'selection_note': 'Fold best Macro F1, earliest tie; development selection estimates; refit uses median best epoch',
        'source_runs': {f'fold_{i}/run.json': digest(run_root / f'fold_{i}/run.json') for i in range(5)}}
    write_json(output / 'cv_summary.json', summary)
    write_csv(output / 'fold_metrics.csv', [{'fold': r['fold'], 'best_epoch': r['best_epoch'], **r['verified_metrics']} for r in runs])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=Path('DATA/manifests'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.runs, args.manifests, args.output), indent=2))
