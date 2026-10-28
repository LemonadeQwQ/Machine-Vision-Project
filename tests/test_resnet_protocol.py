"""新协议的离线契约测试：人工图片 / 小模型；不下载、不训练真实病例。"""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.class_schema import NINE_CLASSES
from src.data.common import ROOT, read_csv, write_csv, write_json
from src.data.deduplicate import BKTree, UnionFind
from src.data.manifest import ManifestDataset, assert_disjoint, fit_metadata, validate_rows
from src.data.prepare import group_folds, mapping_table
from src.evaluate import load_evaluation_data
from src.experiment import configuration, epoch, load_checkpoint, run, smoke_rows
from src.losses import classification_loss
from src.models.fusion import ResNetExperiment
from scripts.collect_results import collect
from scripts.summarize_cv import summarize


def row(key, label='BCC', group=None):
    return {'sample_id': f'MILK:{key}', 'source': 'MILK', 'lesion_id': key, 'group_id': group or key,
        'label': label, 'label_id': NINE_CLASSES.index(label), 'derm_path': 'a.png', 'clinical_path': 'b.png',
        'has_derm': 1, 'has_clinical': 1, 'metadata_key': key, 'split': 'dev', 'age_years': '40',
        'sex': 'male', 'anatomical_region': 'head', 'skin_tone_class': '2', 'derm_image_id': key, 'clinical_image_id': key + '-c'}


class TinyExperiment(nn.Module):
    """测试引擎用的小模型，避免离线测试依赖 ResNet 权重。"""
    def __init__(self, *args):
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.3), nn.Linear(3*16*16, 9))

    def forward(self, batch):
        return self.classifier(batch['derm'])

    def optimizer_groups(self, a, b):
        return [{'params': self.parameters(), 'lr': a}]


class TestResNetProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        pixels = np.arange(24*32*3, dtype=np.uint8).reshape(24, 32, 3)
        Image.fromarray(pixels).save(self.root / 'a.png')
        Image.fromarray(np.flipud(pixels)).save(self.root / 'b.png')

    def tearDown(self):
        self.temp.cleanup()

    def test_mapping_explicit_unknown_rejected(self):
        mapping = mapping_table()
        self.assertEqual(mapping['ISIC2019', 'AK'], 'AKIEC')
        self.assertIsNone(mapping['ISIC2019', 'UNK'])
        self.assertIsNone(mapping['MILK', 'BEN_OTH'])
        with self.assertRaises(KeyError):
            _ = mapping['ISIC2019', 'NEW']

    def test_experiment_variable_isolation(self):
        configs = {p.stem[:3]: configuration(p) for p in (ROOT / 'configs/experiments').glob('E*.yaml')}
        comparisons = [('E00', 'E01', {'modality'}), ('E00', 'E03', {'weighted'}),
            ('E00', 'E04', {'include_isic'}), ('E03', 'E05', {'include_isic'}),
            ('E02', 'E06', {'weighted'}), ('E06', 'E07', {'include_isic'}),
            ('E02', 'E08', {'metadata'}), ('E02', 'E09', {'augment'})]
        for a, b, expected in comparisons:
            self.assertEqual({k for k in configs[a] if k != 'experiment' and configs[a][k] != configs[b][k]}, expected)

    def test_manifest_rejects_bad_label_duplicate_path_mask(self):
        good = row('a')
        for bad in [dict(good, label_id=0), dict(good, has_clinical=0), dict(good, derm_path='../escape.png')]:
            with self.assertRaises(ValueError):
                validate_rows([bad], self.root)
        with self.assertRaises(ValueError):
            validate_rows([good, good])

    def test_group_isolation_and_oof(self):
        rows = [row(f'{c}-{i}-{j}', c, f'{c}-{i}') for c in NINE_CLASSES for i in range(10) for j in range(2)]
        folds = group_folds(rows)
        observed = []
        for train, val in folds:
            assert_disjoint(train, val)
            observed.extend(r['sample_id'] for r in val)
        self.assertEqual(Counter(observed), Counter(r['sample_id'] for r in rows))
        with self.assertRaises(ValueError):
            assert_disjoint([row('a', group='shared')], [row('b', group='shared')])

    def test_rare_groups_block_folds(self):
        with self.assertRaises(ValueError):
            group_folds([row(str(i), c) for i, c in enumerate(NINE_CLASSES)])

    def test_similarity_tree_matches_brute_force(self):
        values = [0, 1, 3, 7, 15, 255, 65535, 7]
        tree = BKTree()
        for i, value in enumerate(values):
            tree.add(value, i)
        for query in [0, 4, 17, 65535]:
            expected = {(i, (query ^ value).bit_count()) for i, value in enumerate(values) if (query ^ value).bit_count() <= 3}
            self.assertEqual(set(tree.query(query, 3)), expected)
        groups = UnionFind(['z', 'a', 'b'])
        groups.union('z', 'b')
        groups.union('a', 'z')
        self.assertEqual({groups.find(k) for k in ['a', 'b', 'z']}, {'a'})

    def test_missing_modality_is_zero_and_metadata_train_only(self):
        sample = dict(row('a'), clinical_path='', has_clinical=0)
        encoder = fit_metadata([row('fit')])
        dataset = ManifestDataset([dict(sample, age_years='', sex='unseen')], self.root, metadata_encoder=encoder)
        batch = dataset[0]
        self.assertEqual(batch['modality_mask'].tolist(), [1., 0.])
        self.assertEqual(batch['clinical'].count_nonzero().item(), 0)
        self.assertEqual(encoder.age_mean, 40)
        self.assertTrue(torch.isfinite(batch['metadata']).all())
        with self.assertRaises(ValueError):
            ManifestDataset([sample], self.root, modality='clinical')

    def test_geometric_randomness_reproducible_per_epoch(self):
        dataset = ManifestDataset([row('a')], self.root, train=True)
        self.assertTrue(torch.equal(dataset[0]['derm'], dataset[0]['derm']))
        first = dataset[0]['derm']
        dataset.epoch.fill_(1)
        self.assertFalse(torch.equal(first, dataset[0]['derm']))
        val = ManifestDataset([row('a')], self.root, train=False)
        first = val[0]['derm']
        val.epoch.fill_(10)
        self.assertTrue(torch.equal(first, val[0]['derm']))

    def test_persistent_worker_sees_new_epoch(self):
        from torch.utils.data import DataLoader
        data = ManifestDataset([row('a')], self.root, train=True, size=16)
        worker = DataLoader(data, batch_size=1, num_workers=1, persistent_workers=True)
        try:
            first = next(iter(worker))['derm'].clone()
            data.epoch.fill_(1)
            second = next(iter(worker))['derm'].clone()
            expected = data[0]['derm'].unsqueeze(0)
            self.assertFalse(torch.equal(first, second))
            self.assertTrue(torch.equal(expected, second))
        finally:
            if worker._iterator is not None:
                worker._iterator._shutdown_workers()

    def test_pair_independence_bn_freeze_and_missing_branch(self):
        model = ResNetExperiment('pair')
        model.train()
        self.assertIsNot(model.encoders['derm'], model.encoders['clinical'])
        self.assertFalse(model.encoders['derm'].bn1.training)
        self.assertFalse(model.encoders['derm'].conv1.weight.requires_grad)
        self.assertTrue(model.encoders['derm'].layer4[0].conv1.weight.requires_grad)
        calls = []
        handle = model.encoders['clinical'].register_forward_hook(lambda m, i, o: calls.append(i[0].shape[0]))
        batch = {'derm': torch.randn(2, 3, 32, 32), 'clinical': torch.randn(2, 3, 32, 32),
                 'modality_mask': torch.tensor([[1., 1.], [1., 0.]])}
        before = model.encoders['derm'].layer4[0].bn1.running_mean.clone()
        logits = model(batch)
        logits.sum().backward()
        self.assertEqual(tuple(logits.shape), (2, 9))
        self.assertEqual(calls, [1])
        self.assertTrue(torch.equal(before, model.encoders['derm'].layer4[0].bn1.running_mean))
        self.assertIsNotNone(model.encoders['derm'].layer4[0].conv1.weight.grad)
        batch['modality_mask'][:, 1] = 0
        with torch.no_grad():
            model(batch)
        self.assertEqual(calls, [1])
        handle.remove()

    def test_ce_weights_exact_formula_and_zero_class_rejected(self):
        rows = [row(f'{c}-{j}', c) for i, c in enumerate(NINE_CLASSES) for j in range(i+1)]
        plain, _ = classification_loss(rows, False, 'cpu')
        self.assertIsNone(plain.weight)
        weighted, record = classification_loss(rows, True, 'cpu')
        expected = 1 / np.arange(1, 10, dtype=float)
        expected /= expected.mean()
        np.testing.assert_allclose(weighted.weight.numpy(), expected, rtol=1e-6)
        with self.assertRaises(ValueError):
            classification_loss([row('a')], True, 'cpu')

    def test_sample_id_evaluation_aligns_and_rejects_extra(self):
        labels = self.root / 'labels.csv'
        predictions = self.root / 'predictions.csv'
        write_csv(labels, [{'sample_id': 'a', 'label': 'BCC'}, {'sample_id': 'b', 'label': 'NV'}])
        probs = [dict(sample_id=k, **{c: int(c == label) for c in NINE_CLASSES}) for k, label in [('b','NV'), ('a','BCC')]]
        write_csv(predictions, probs)
        ids, y, p = load_evaluation_data(labels, predictions, class_names=NINE_CLASSES, id_column='sample_id')
        self.assertEqual(ids, ['a', 'b'])
        np.testing.assert_array_equal(y, p.argmax(1))
        write_csv(predictions, probs + [dict(probs[0], sample_id='c')])
        with self.assertRaises(ValueError):
            load_evaluation_data(labels, predictions, class_names=NINE_CLASSES, id_column='sample_id')

    def test_synthetic_engine_checkpoint_inference_and_resume(self):
        config = configuration(ROOT / 'configs/experiments/E00_derm.yaml',
            {'data_root': str(self.root), 'workers': 0, 'image_size': 16, 'batch_size': 9, 'max_epochs': 2, 'amp': 'off'})
        train = [row(f'train-{c}', c) for c in NINE_CLASSES]
        val = [row(f'val-{c}', c) for c in NINE_CLASSES]
        inputs = {'manifest_sha256': 'synthetic', 'weights_sha256': 'none', 'fold': 0}
        target = self.root / 'run'
        with patch('src.experiment.preflight', return_value=(train, val, inputs)), patch('src.experiment.ResNetExperiment', TinyExperiment):
            record = run(config, 0, target, device_name='cpu')
            self.assertEqual(record['status'], 'completed')
            checkpoint = load_checkpoint(target / 'best.pt')
            self.assertEqual(checkpoint['class_order'], list(NINE_CLASSES))
            # 重载后的输出通过正式推理接口核对。
            from src.infer import predict
            manifest = self.root / 'val.csv'
            write_csv(manifest, val)
            with patch('src.infer.ResNetExperiment', TinyExperiment):
                result = predict(target / 'best.pt', manifest, self.root / 'reloaded.csv', device_name='cpu')
            stored = read_csv(target / 'predictions.csv')
            np.testing.assert_allclose(np.asarray([[float(r[c]) for c in NINE_CLASSES] for r in stored]), result['probabilities'], atol=1e-7)
            resumed = run(config, 0, target, resume=True, device_name='cpu')
            self.assertEqual(resumed['epochs_completed'], 2)
            with self.assertRaises(ValueError):
                run(dict(config, head_lr=.1), 0, target, resume=True, device_name='cpu')
        checkpoint['class_order'] = list(reversed(NINE_CLASSES))
        torch.save(checkpoint, self.root / 'bad.pt')
        with self.assertRaises(ValueError):
            load_checkpoint(self.root / 'bad.pt')

    def test_collector_keeps_unrun_experiments_planned(self):
        rows = collect(self.root / 'empty', self.root / 'status.csv')
        self.assertEqual(len(rows), 50)
        self.assertEqual({r['status'] for r in rows}, {'planned'})
        with self.assertRaises(FileNotFoundError):
            summarize(self.root / 'empty', ROOT / 'DATA/manifests', self.root / 'summary')

    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        for workers in (0, 4):
            with self.subTest(workers=workers):
                self.check_interrupted_resume(workers)

    def check_interrupted_resume(self, workers):
        config = configuration(ROOT / 'configs/experiments/E00_derm.yaml',
            {'data_root': str(self.root), 'workers': workers, 'image_size': 16, 'batch_size': 4, 'max_epochs': 2, 'amp': 'off'})
        train = [row(f't-{c}-{i}', c) for c in NINE_CLASSES for i in range(3)]
        val = [row(f'v-{c}', c) for c in NINE_CLASSES]
        reference, interrupted = self.root / f'reference-{workers}', self.root / f'interrupted-{workers}'
        with patch('src.experiment.preflight', return_value=(train, val, {'manifest_sha256': 'artificial', 'fold': 0})), \
             patch('src.experiment.ResNetExperiment', TinyExperiment):
            run(config, 0, reference, device_name='cpu')
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise KeyboardInterrupt()
                return epoch(*args, **kwargs)
            with patch('src.experiment.epoch', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                run(config, 0, interrupted, device_name='cpu')
            run(config, 0, interrupted, resume=True, device_name='cpu')
        a = load_checkpoint(reference / 'last.pt')
        b = load_checkpoint(interrupted / 'last.pt')
        for key in a['model']:
            self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
        self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
        for key, state in a['optimizer']['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][key][name]), name)
        self.assertEqual([h['train']['sample_order_sha256'] for h in a['history']],
                         [h['train']['sample_order_sha256'] for h in b['history']])
        self.assertEqual((reference / 'predictions.csv').read_bytes(), (interrupted / 'predictions.csv').read_bytes())

    def test_smoke_selection_covers_sources_classes_and_masks(self):
        milk = [row(f'm-{c}-{i}', c) for c in NINE_CLASSES for i in range(6)]
        isic = [dict(row(f'i-{c}-{i}', c), sample_id=f'ISIC2019:i-{c}-{i}', source='ISIC2019', clinical_path='', has_clinical=0)
                for c in NINE_CLASSES for i in range(6)]
        chosen = smoke_rows(milk + isic, 4)
        self.assertEqual(Counter(r['source'] for r in chosen), {'MILK': 36, 'ISIC2019': 36})
        self.assertEqual(Counter(int(r['has_clinical']) for r in chosen), {0: 36, 1: 36})
        self.assertEqual(len({r['sample_id'] for r in chosen}), len(chosen))
        self.assertTrue({r['sample_id'] for r in chosen} <= {r['sample_id'] for r in milk + isic})
        self.assertEqual({r['source'] for r in smoke_rows(milk, 2)}, {'MILK'})

    def test_smoke_checkpoint_stop_and_resume(self):
        from src.infer import predict
        from scripts.check_smoke_run import check
        config = configuration(ROOT / 'configs/experiments/E07_pair_isic_weighted.yaml',
            {'data_root': str(self.root), 'workers': 0, 'image_size': 16, 'batch_size': 4, 'amp': 'off'})
        train = [row(f't-{c}', c) for c in NINE_CLASSES]
        train += [dict(row(f'i-{c}', c), sample_id=f'ISIC2019:i-{c}', source='ISIC2019', clinical_path='', has_clinical=0)
                  for c in NINE_CLASSES]
        val = [row(f'v-{c}', c) for c in NINE_CLASSES]
        target = self.root / 'smoke'
        with patch('src.experiment.preflight', return_value=(train, val, {'manifest_sha256': 'artificial', 'fold': 0})), \
             patch('src.experiment.ResNetExperiment', TinyExperiment):
            stopped = run(config, 0, target, device_name='cpu', smoke=True, smoke_epochs=2, smoke_stop_after_epoch=1)
            self.assertEqual(stopped['status'], 'partial')
            check(target, 'stopped')
            self.assertFalse((target / '.running').exists())
            resumed = run(config, 0, target, device_name='cpu', smoke=True, smoke_epochs=2, resume=True)
            self.assertEqual(resumed['resume_from_epoch'], 1)
            with patch('src.infer.ResNetExperiment', TinyExperiment):
                predict(target / 'best.pt', target / 'smoke_val.csv', target / 'reloaded_predictions.csv', device_name='cpu')
            self.assertEqual(check(target, 'completed')['status'], 'passed')
        with self.assertRaises(ValueError):
            run(config, 0, self.root / 'invalid', smoke_stop_after_epoch=1)

    def test_refit_has_no_validation_and_uses_declared_epochs(self):
        from src.experiment import code_record
        config = configuration(ROOT / 'configs/experiments/E00_derm.yaml',
            {'data_root': str(self.root), 'workers': 0, 'image_size': 16, 'batch_size': 9, 'max_epochs': 2, 'amp': 'off', 'manifests': str(self.root)})
        train = [row(f't-{c}', c) for c in NINE_CLASSES]
        val = [dict(row(f'v-{c}', c), split='holdout') for c in NINE_CLASSES]
        write_csv(self.root / 'milk_dev.csv', train)
        write_csv(self.root / 'milk_holdout.csv', val)
        summary = {'status': 'completed', 'fold_count': 5, 'purpose': 'formal', 'config': config,
            'manifest_sha256': 'fixture', 'code_sha256': code_record()['source_sha256'], 'refit_epochs': 1}
        with patch('src.experiment.preflight', return_value=(train, val, {'manifest_sha256': 'fixture', 'fold': 0})), \
             patch('src.experiment.ResNetExperiment', TinyExperiment):
            result = run(config, 0, self.root / 'refit', device_name='cpu', refit_summary=summary)
        self.assertEqual(result['epochs_completed'], 1)
        self.assertIsNone(result['best_macro_f1'])
        checkpoint = load_checkpoint(self.root / 'refit/best.pt')
        self.assertTrue(checkpoint['refit_complete'])
        self.assertEqual(checkpoint['phase'], 'development_refit')
        self.assertFalse((self.root / 'refit/validation_labels.csv').exists())

    def test_cv_summary_complete_and_mixed_or_smoke_rejected(self):
        from src.data.common import digest
        runs, manifests = self.root / 'runs', self.root / 'manifests'
        write_json(manifests / 'version.json', {'fixture': True})
        all_rows = []
        for fold in range(5):
            val = [row(f'{fold}-{c}', c) for c in NINE_CLASSES]
            all_rows.extend(val)
            write_csv(manifests / f'folds/fold_{fold}/val.csv', val)
            probs = [dict(sample_id=r['sample_id'], **{c: int(c == r['label']) for c in NINE_CLASSES}) for r in val]
            write_csv(runs / f'fold_{fold}/predictions.csv', probs)
            write_json(runs / f'fold_{fold}/run.json', {'status': 'completed', 'purpose': 'formal', 'fold': fold,
                'code': {'source_sha256': 'artificial-fixture'}, 'inputs': {'manifest_sha256': digest(manifests / 'version.json')},
                'config': {'fixture': True}, 'protocol': 'resnet9-v1', 'experiment': 'E00', 'best_epoch': fold+1, 'best_macro_f1': 1.})
        write_csv(manifests / 'milk_dev.csv', all_rows)
        result = summarize(runs, manifests, self.root / 'complete')
        self.assertEqual(result['n_oof'], 45)
        self.assertEqual(result['refit_epochs'], 3)
        self.assertEqual(result['fold_metrics']['macro_f1'], {'mean': 1., 'sample_std': 0.})
        path = runs / 'fold_4/run.json'
        changed = json.loads(path.read_text())
        changed['purpose'] = 'smoke'
        write_json(path, changed)
        with self.assertRaises(ValueError):
            summarize(runs, manifests, self.root / 'bad')


if __name__ == '__main__':
    unittest.main()
