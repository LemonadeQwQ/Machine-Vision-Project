"""Synthetic, offline E11-E13 contract checks; never download weights or use patient images."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import numpy as np
from PIL import Image

from src.class_schema import NINE_CLASSES
from src.data.common import ROOT, digest, training_source_files, write_csv, write_json
from src.experiment import PREPROCESS, configuration, epoch, load_checkpoint, run
from src.experiment_registry import EXPERIMENTS, validate_run_identity, comparison_contract
from src.models.factory import (NEW_CHECKPOINT_FORMAT, build_model, checkpoint_state, model_contract,
                                prepare_model, restore_model, text_features_digest)
from src.models.foundation import (FrozenMONET, MONET_FILES, MONET_REVISION, MONET_SHA256,
    PANDERM_SHA256, PANDERM_SOURCE, RESNET_SHA256, concept_scores,
    freeze_panderm, monet_assets, panderm_encoder)
from src.models.fusion import ResNetExperiment
from src.models.next_stage import ModalityGate, NextStageExperiment
from src.metadata_features import MetadataEncoder
from scripts.collect_results import collect


def config(number):
    path = next((ROOT / 'configs/experiments').glob(f'E{number:02}_*.yaml'))
    return configuration(path)


class TinyEncoder(nn.Module):
    def __init__(self, dimension=2048):
        super().__init__()
        self.projection = nn.Linear(3, dimension)

    def forward(self, image):
        return self.projection(image.mean((2, 3)))


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)
        self.dropout = nn.Dropout(.5)
        self.seen = None

    def forward(self, pixel_values):
        self.seen = pixel_values.detach().clone()
        return SimpleNamespace(pooler_output=self.dropout(self.projection(pixel_values.mean((2, 3)))))


def synthetic_monet(*args, **kwargs):
    # Exercise the real frozen forward/dtype implementation with tiny fake weights.
    # A pinned external model must reconstruct identically without consuming the
    # classifier's RNG stream (including inference before configure_runtime).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13013)
        model = FrozenMONET.__new__(FrozenMONET)
        nn.Module.__init__(model)
        model.vision, model.projection = TinyVision(), nn.Linear(8, 8, bias=False)
        model.register_buffer('targets', torch.randn(4, 3, 8))
        model.register_buffer('references', torch.randn(3, 8))
    model.requires_grad_(False).eval()
    return model


def batch(metadata_dim=7, size=8):
    return {'derm': torch.randn(2, 3, size, size), 'clinical': torch.randn(2, 3, size, size),
            'modality_mask': torch.tensor([[1., 1.], [1., 0.]]),
            'metadata': torch.randn(2, metadata_dim), 'metadata_mask': torch.ones(2, 4)}


def components(number):
    result = {'resnet': {'sha256': RESNET_SHA256}}
    if number == 12:
        result['panderm'] = {'sha256': PANDERM_SHA256, 'source_sha256': digest(ROOT / PANDERM_SOURCE)}
    if number == 13:
        result['monet'] = {'repository': 'chanwkim/monet', 'revision': MONET_REVISION,
                           'sha256': {name: MONET_SHA256 if name == 'model.safetensors' else 'a' * 64 for name in MONET_FILES}}
    return result


class E11E13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registry_and_single_change_definitions(self):
        self.assertEqual(EXPERIMENTS[-3:], ('E11', 'E12', 'E13'))
        original = config(10)
        self.assertNotIn('gate', original)
        for number, enabled in ((11, 'gate'), (12, 'panderm'), (13, 'monet')):
            current = config(number)
            self.assertEqual({key for key in ('gate', 'panderm', 'monet') if current[key]}, {enabled})
            self.assertEqual({key: current[key] for key in original if key != 'experiment'},
                             {key: value for key, value in original.items() if key != 'experiment'})
            with self.assertRaises(ValueError):
                configuration(next((ROOT / 'configs/experiments').glob(f'E{number}_*.yaml')), {'metadata': False})
        for change in ({'gate_hidden': 256}, {'panderm': True}, {'experiment': 'E14'}):
            with self.assertRaises(ValueError):
                configuration(ROOT / 'configs/experiments/E11_pair_gate.yaml', change)
        with self.assertRaises(ValueError):
            configuration(ROOT / 'configs/experiments/E10_pair_isic_weighted_metadata.yaml', {'gate': True})

    def test_legacy_factory_retains_exact_parameter_names_and_values(self):
        with patch('src.models.fusion.backbone', side_effect=lambda _: TinyEncoder()):
            torch.manual_seed(12)
            old = ResNetExperiment('pair', 7)
            torch.manual_seed(12)
            new = build_model(config(10), 7)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]), key)

    def test_gate_neutral_output_matches_e10_including_missing_clinical(self):
        with patch('src.models.fusion.backbone', side_effect=lambda _: TinyEncoder()), \
             patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()):
            torch.manual_seed(9)
            baseline = build_model(config(10), 7).eval()
            torch.manual_seed(9)
            gated = build_model(config(11), 7).eval()
        values = batch()
        self.assertTrue(torch.equal(baseline(values), gated(values)))
        self.assertTrue(torch.equal(gated.last_gate, torch.ones(2, 2)))
        gated.train()
        nn.CrossEntropyLoss()(gated(values), torch.tensor([0, 1])).backward()
        gradient = gated.gate.network[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_gate_unequal_dimensions_and_missing_path_are_fixed(self):
        gate = ModalityGate(768)
        with torch.no_grad():
            gate.network[-1].bias.fill_(2.)
        derm, clinical = torch.randn(2, 768), torch.randn(2, 2048)
        mask = torch.tensor([[1., 1.], [1., 0.]])
        d, c, scales = gate(derm, clinical, mask)
        self.assertTrue(torch.equal(d[1], derm[1]))
        self.assertEqual(c[1].count_nonzero().item(), 0)
        self.assertTrue(torch.equal(scales[1], torch.ones(2)))
        self.assertGreater(scales[0, 0].item(), 1.)
        self.assertLess(scales.max().item(), 2.)

    def test_missing_clinical_never_calls_encoder_and_no_clinical_gradient(self):
        with patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()):
            model = build_model(config(11), 7).train()
        values = batch()
        values['modality_mask'][:, 1] = 0
        with patch.object(model.encoders['clinical'], 'forward', side_effect=AssertionError('missing branch ran')):
            model(values).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoders['clinical'].parameters()))
        values['modality_mask'][0, 0] = 0
        with self.assertRaises(ValueError):
            model(values)

    def test_panderm_freezing_selects_only_last_two_blocks_and_norm(self):
        encoder = nn.Module()
        encoder.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
        encoder.norm = nn.LayerNorm(4)
        encoder.other = nn.Linear(4, 4)
        freeze_panderm(encoder)
        x = encoder.other(torch.randn(2, 4))
        for block in encoder.blocks:
            x = block(x)
        encoder.norm(x).square().sum().backward()
        self.assertFalse(encoder.training)
        for index, block in enumerate(encoder.blocks):
            self.assertEqual(block.weight.requires_grad, index >= 10)
            self.assertEqual(block.weight.grad is not None, index >= 10)
        self.assertIsNotNone(encoder.norm.weight.grad)
        self.assertIsNone(encoder.other.weight.grad)

    def test_real_panderm_architecture_without_weights(self):
        # Existing timm is sufficient; no checkpoint/network access is performed.
        encoder = panderm_encoder()
        self.assertIsInstance(encoder.head, nn.Identity)
        self.assertEqual(encoder.num_features, 768)
        with torch.no_grad():
            output = encoder(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 768))
        self.assertTrue(all(not p.requires_grad for p in encoder.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in encoder.blocks[-1].parameters()))

    def test_panderm_pair_uses_raw_dimensions_and_no_auxiliary_heads(self):
        with patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()), \
             patch('src.models.next_stage.panderm_encoder', return_value=TinyEncoder(768)):
            model = build_model(config(12), 7)
        self.assertEqual(model.classifier[0].in_features, 768 + 2048 + 2 + 64 + 4)
        self.assertEqual(tuple(model(batch()).shape), (2, 9))
        self.assertFalse(any('aux' in name or 'derm_project' in name for name, _ in model.named_parameters()))

    def test_monet_scores_are_independent_reference_comparisons(self):
        images = torch.tensor([[1., 0.]])
        refs = torch.zeros(3, 2)
        targets = torch.zeros(4, 3, 2)
        scores = concept_scores(images, targets, refs)
        self.assertTrue(torch.equal(scores, torch.full((1, 4), .5)))
        targets[0, :, 0] = .01
        changed = concept_scores(images, targets, refs)
        self.assertGreater(changed[0, 0].item(), .5)
        self.assertTrue(torch.equal(changed[:, 1:], scores[:, 1:]))
        self.assertNotAlmostEqual(changed.sum().item(), 1.)

    def test_monet_is_frozen_eval_and_receives_same_augmented_tensor(self):
        with patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()), \
             patch('src.models.next_stage.FrozenMONET', side_effect=synthetic_monet):
            model = build_model(config(13), 7)
        prepare_model(model, config(13), torch.device('cpu'))
        model.train()
        values = batch(size=224)
        model(values).sum().backward()
        self.assertFalse(model.monet.training)
        self.assertFalse(model.monet.vision.dropout.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.monet.parameters()))
        self.assertTrue(torch.equal(model.monet.vision.seen, values['derm']))
        self.assertIsNotNone(model.concept_project[0].weight.grad)
        self.assertEqual(model.monet(values['derm']).dtype, torch.float32)

    def test_monet_text_encoder_runs_once_and_no_processor_is_invoked(self):
        calls = []

        class FakeCLIP(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model, self.visual_projection = TinyVision(), nn.Linear(8, 8, bias=False)
                self.config = SimpleNamespace(projection_dim=768, vision_config=SimpleNamespace(
                    image_size=224, patch_size=14, hidden_size=1024, num_hidden_layers=24))

            def get_text_features(self, input_ids):
                calls.append(input_ids)
                return input_ids[:, :1].float() + torch.arange(8).float().unsqueeze(0)

        fake = FakeCLIP()
        tokenizer = lambda texts, **kwargs: {'input_ids': torch.tensor([[len(text)] for text in texts])}
        module = SimpleNamespace(CLIPModel=SimpleNamespace(from_pretrained=lambda *a, **k: (fake, {})),
                                 CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
        with patch.dict('sys.modules', {'transformers': module}), \
             patch('src.models.foundation.require_monet_dependencies'), \
             patch('src.models.foundation.monet_assets', return_value=components(13)['monet']):
            model = FrozenMONET('unused-local-fixture')
        self.assertEqual(len(calls), 13)
        self.assertEqual(tuple(model.targets.shape), (4, 3, 8))
        self.assertEqual(model.targets.dtype, torch.float32)
        model.train()
        first = model(torch.randn(2, 3, 224, 224))
        model(torch.randn(2, 3, 224, 224))
        self.assertEqual(len(calls), 13)
        self.assertEqual(tuple(first.shape), (2, 4))
        self.assertFalse(model.training)

    def test_monet_checkpoint_omits_only_verified_external_weights(self):
        with patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()), \
             patch('src.models.next_stage.FrozenMONET', side_effect=synthetic_monet):
            first = build_model(config(13), 7).eval()
            second = build_model(config(13), 7).eval()
        # Emulate reconstructing identical pinned external weights.
        second.monet.vision.load_state_dict(first.monet.vision.state_dict())
        second.monet.projection.load_state_dict(first.monet.projection.state_dict())
        state = checkpoint_state(first, config(13))
        self.assertIn('monet.targets', state)
        self.assertFalse(any(k.startswith(('monet.vision.', 'monet.projection.')) for k in state))
        restore_model(second, state, config(13))
        values = batch(size=224)
        self.assertTrue(torch.equal(first(values), second(values)))
        bad = dict(state)
        del bad['classifier.0.weight']
        with self.assertRaises(ValueError):
            restore_model(second, bad, config(13))
        with self.assertRaises(ValueError):
            restore_model(second, first.state_dict(), config(13))

    def test_pinned_monet_receipt_rejects_changed_files_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in MONET_FILES:
                (folder / name).write_text(name, encoding='utf-8')
            hashes = {name: digest(folder / name) for name in MONET_FILES}
            receipt = {'repository': 'chanwkim/monet', 'revision': MONET_REVISION, 'sha256': hashes}
            write_json(folder / 'download_manifest.json', receipt)
            with patch('src.models.foundation.MONET_SHA256', hashes['model.safetensors']):
                actual = monet_assets(folder)
                self.assertEqual(actual, receipt)
                (folder / '.cache').mkdir()
                self.assertEqual(monet_assets(folder), receipt)
                for name in ('added_tokens.json', 'adapter_config.json', 'adapter_model.safetensors'):
                    unexpected = folder / name
                    unexpected.write_text('unrecorded optional HF input', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Unexpected MONET input'):
                        monet_assets(folder)
                    unexpected.unlink()
                (folder / 'vocab.json').write_text('changed', encoding='utf-8')
                with self.assertRaises(ValueError):
                    monet_assets(folder)
                receipt['revision'] = 'unapproved'
                write_json(folder / 'download_manifest.json', receipt)
                with self.assertRaises(ValueError):
                    monet_assets(folder)

    def test_new_checkpoint_roundtrip_and_contract_mismatch_rejected(self):
        encoder = MetadataEncoder.fit([{'age_years': '30', 'sex': 'male', 'anatomical_region': 'head', 'skin_tone_class': '2'}], include_skin_tone=True)
        current = config(11)
        with patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()):
            model = build_model(current, encoder.output_dim)
        value = {'format': NEW_CHECKPOINT_FORMAT, 'class_order': list(NINE_CLASSES),
                 'preprocess': PREPROCESS, 'config': current, 'metadata': encoder.to_dict(),
                 'inputs': {'components': components(11)}, 'model': checkpoint_state(model, current),
                 'model_contract': model_contract(current, encoder.output_dim, components(11))}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.pt'
            torch.save(value, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded['model_contract'], value['model_contract'])
            value['model_contract']['gate']['hidden'] = 256
            torch.save(value, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_registry_collection_and_candidate_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = collect(root / 'empty', root / 'status.csv')
            self.assertEqual(len(rows), 70)
            self.assertEqual({row['status'] for row in rows}, {'planned'})
        current = config(11)
        contract = model_contract(current, 20, components(11))
        run = {'experiment': 'E11', 'configuration_id': 'E11', 'config': current,
               'inputs': {'components': components(11)}, 'model_contract': contract}
        validate_run_identity(run)
        run['configuration_id'] = 'E13'
        with self.assertRaises(ValueError):
            validate_run_identity(run)
        other = dict(contract, metadata_dim=28)
        self.assertEqual(comparison_contract(contract), comparison_contract(other))

    def test_source_fingerprint_includes_vendor_and_attribution(self):
        files = training_source_files()
        self.assertEqual(files[PANDERM_SOURCE], digest(ROOT / PANDERM_SOURCE))
        self.assertIn('stages/monet_stage/reference/MONET_LICENSE.txt', files)


    def test_v2_run_resume_infer_and_provenance_rejection(self):
        from src.infer import predict

        def row(prefix, label, source='MILK'):
            key = f'{prefix}-{label}'
            return {'sample_id': f'{source}:{key}', 'source': source, 'lesion_id': key,
                    'group_id': key, 'label': label, 'label_id': NINE_CLASSES.index(label),
                    'derm_path': 'derm.png', 'clinical_path': 'clinical.png' if source == 'MILK' else '',
                    'has_derm': 1, 'has_clinical': int(source == 'MILK'), 'metadata_key': key,
                    'split': 'dev', 'age_years': '40', 'sex': 'male',
                    'anatomical_region': 'head', 'skin_tone_class': '2',
                    'derm_image_id': key, 'clinical_image_id': key + '-c'}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
            Image.fromarray(pixels).save(root / 'derm.png')
            Image.fromarray(np.flipud(pixels)).save(root / 'clinical.png')
            train = [row('train', label) for label in NINE_CLASSES]
            train += [row('isic', label, 'ISIC2019') for label in NINE_CLASSES]
            val = [row('val', label) for label in NINE_CLASSES]
            write_csv(root / 'val.csv', val)
            for number in (11, 12, 13):
                with self.subTest(experiment=number), ExitStack() as stack:
                    current = configuration(next((ROOT / 'configs/experiments').glob(f'E{number}_*.yaml')),
                        {'data_root': str(root), 'workers': 0, 'batch_size': 18, 'max_epochs': 2, 'amp': 'off'})
                    inputs = {'manifest_sha256': 'synthetic', 'fold': 0, 'components': components(number)}
                    preflight = stack.enter_context(patch('src.experiment.preflight', return_value=(train, val, inputs)))
                    stack.enter_context(patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()))
                    stack.enter_context(patch('src.models.next_stage.panderm_encoder', side_effect=lambda _: TinyEncoder(768)))
                    stack.enter_context(patch('src.models.next_stage.FrozenMONET', side_effect=synthetic_monet))
                    reference = root / f'E{number}-reference'
                    record = run(current, 0, reference, device_name='cpu')
                    validate_run_identity(record)
                    saved = load_checkpoint(reference / 'best.pt')
                    self.assertEqual(saved['format'], NEW_CHECKPOINT_FORMAT)
                    predict(reference / 'best.pt', root / 'val.csv', reference / 'reloaded.csv', device_name='cpu')
                    self.assertEqual((reference / 'predictions.csv').read_bytes(), (reference / 'reloaded.csv').read_bytes())
                    if number == 11:
                        self.assertIn('gate_diagnostics', saved['history'][0]['train'])
                    # A changed component may not be used for the CV-derived refit,
                    # even when the old source/config/manifest claims match.
                    refit = {'status': 'completed', 'fold_count': 5, 'purpose': 'formal',
                             'config': current, 'manifest_sha256': 'synthetic', 'refit_epochs': 1,
                             'components': {}, 'code_sha256': record['code']['source_sha256']}
                    with self.assertRaisesRegex(ValueError, 'components differ'):
                        run(current, 0, root / f'E{number}-bad-refit', device_name='cpu', refit_summary=refit)
                    if number != 13:
                        continue
                    interrupted = root / 'E13-interrupted'
                    calls = 0

                    def interrupt(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise KeyboardInterrupt()
                        return epoch(*args, **kwargs)

                    with patch('src.experiment.epoch', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                        run(current, 0, interrupted, device_name='cpu')
                    run(current, 0, interrupted, device_name='cpu', resume=True)
                    first, second = load_checkpoint(reference / 'last.pt'), load_checkpoint(interrupted / 'last.pt')
                    for key in first['model']:
                        self.assertTrue(torch.equal(first['model'][key], second['model'][key]), key)
                    self.assertEqual(first['optimizer']['param_groups'], second['optimizer']['param_groups'])
                    for key, state in first['optimizer']['state'].items():
                        for name, value in state.items():
                            self.assertTrue(torch.equal(value, second['optimizer']['state'][key][name]), name)
                    self.assertEqual([item['train']['sample_order_sha256'] for item in first['history']],
                                     [item['train']['sample_order_sha256'] for item in second['history']])
                    self.assertEqual((reference / 'predictions.csv').read_bytes(), (interrupted / 'predictions.csv').read_bytes())
                    predict(interrupted / 'best.pt', root / 'val.csv', interrupted / 'reloaded.csv', device_name='cpu')
                    self.assertEqual((interrupted / 'predictions.csv').read_bytes(), (interrupted / 'reloaded.csv').read_bytes())
                    tampered = copy.deepcopy(first)
                    tampered['model']['monet.targets'][0, 0, 0] += 1
                    torch.save(tampered, root / 'altered-text.pt')
                    with self.assertRaisesRegex(ValueError, 'text feature identity'):
                        load_checkpoint(root / 'altered-text.pt')
                    changed = copy.deepcopy(inputs)
                    changed['components']['monet']['sha256']['vocab.json'] = 'b' * 64
                    preflight.return_value = (train, val, changed)
                    with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                        run(current, 0, interrupted, device_name='cpu', resume=True)


if __name__ == '__main__':
    unittest.main()
