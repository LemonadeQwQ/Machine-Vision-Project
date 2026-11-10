"""Render aggregate CV results without loading images or case-level predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.summary.read_text(encoding='utf-8'))
    data = record['results']
    if not data or len({r['experiment'] for r in data}) != len(data):
        raise ValueError('Expected a nonempty list of distinct experiments')
    if len({r['n_oof'] for r in data}) != 1 or data[0]['n_oof'] <= 0:
        raise ValueError('Experiments must describe the same positive OOF population')
    means = np.array([r['fold_metrics']['macro_f1']['mean'] for r in data])
    std = np.array([r['fold_metrics']['macro_f1']['sample_std'] for r in data])
    oof = np.array([r['pooled_oof_metrics']['macro_f1'] for r in data])
    if not all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in (means, std, oof)):
        raise ValueError('Expected finite fractional metrics and nonnegative sample SD')
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(12, max(4.8, len(data) * .55 + 2)), layout='constrained')
    primary = (record.get('selection') or {}).get('primary')
    colors = ['#367e9c' if r['experiment'] == primary else '#9baabd' for r in data]
    ax.barh(y, means * 100, xerr=std * 100, color=colors, capsize=4,
            label='Five-fold mean +/- sample SD')
    ax.scatter(oof * 100, y, s=45, color='#182c3c', marker='D', zorder=4, label='Pooled OOF')
    legacy_labels = {'E00': 'Dermoscopy', 'E01': 'Clinical', 'E02': 'Paired fusion'}
    labels = [r['experiment'] + '  ' + (r.get('description') or legacy_labels.get(r['experiment'], ''))
              for r in data]
    ax.set_yticks(y, [label.rstrip() for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel('Nine-class Macro F1 (%)')
    ax.set_xlim(0, 100)
    ax.set_title(f"MILK development cross-validation - {data[0]['n_oof']:,} cases", loc='left', pad=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.17)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    holdout_note = ('holdout results reported separately' if record.get('holdout_evaluated')
                    else 'holdout not evaluated at this snapshot')
    fig.text(.5, -.025, f'Development CV estimates; {holdout_note}. Error bars are not confidence intervals.',
             ha='center', fontsize=9, color='#4b5563')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
