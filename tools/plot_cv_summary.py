"""Render an aggregate CV figure without loading case-level predictions."""
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
    assert [r['experiment'] for r in data] == ['E00', 'E01', 'E02']
    x = np.arange(3)
    means = [r['fold_metrics']['macro_f1']['mean'] * 100 for r in data]
    std = [r['fold_metrics']['macro_f1']['sample_std'] * 100 for r in data]
    oof = [r['pooled_oof_metrics']['macro_f1'] * 100 for r in data]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), layout='constrained')
    ax.bar(x, means, yerr=std, width=.55, color=['#3b82a0', '#8c9cb7', '#3a9188'],
           capsize=6, label='Five-fold mean ± sample SD')
    ax.scatter(x, oof, s=64, color='#172533', marker='D', zorder=4, label='Pooled OOF')
    for i, value in enumerate(means):
        ax.text(i, value + std[i] + 1.2, f'{value:.2f}%', ha='center', fontsize=11)
    ax.set_xticks(x, ['E00\nDermoscopy', 'E01\nClinical', 'E02\nPaired fusion'])
    ax.set_ylabel('Nine-class Macro F1 (%)')
    ax.set_ylim(0, 65)
    ax.set_title(f"MILK development cross-validation · {data[0]['n_oof']:,} cases", loc='left', pad=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.17)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    fig.text(.5, -.015, 'Development selection estimates; holdout not evaluated. Error bars are not confidence intervals.',
             ha='center', fontsize=9, color='#4b5563')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
