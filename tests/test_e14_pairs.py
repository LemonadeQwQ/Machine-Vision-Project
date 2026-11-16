"""Targeted offline pair contracts, using artificial inputs and tiny encoders."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.class_schema import NINE_CLASSES
from src.data.common import ROOT, write_csv
from src.experiment import configuration, epoch, load_checkpoint, run
from src.experiment_registry import EXPERIMENTS, EXPERIMENT_GROUPS, validate_run_identity
from src.infer import predict
from src.models.factory import build_model, model_contract
from scripts.collect_results import collect
from test_e11_e13 import TinyEncoder, synthetic_monet, batch, components


PAIRS = ('E14-GP', 'E14-GM', 'E14-PM')


def config(name, overrides=None):
    return configuration(next((ROOT / 'configs/experiments').glob(name + '_*.yaml')), overrides)


def assets(name):
    result = components(11)
    if 'P' in name.split('-')[1]:
        result.update(components(12))
    if 'M' in name.split('-')[1]:
        result.update(components(13))
    return result


def tiny_models(stack):
    stack.enter_context(patch('src.models.next_stage.backbone', side_effect=lambda _: TinyEncoder()))
    stack.enter_context(patch('src.models.next_stage.panderm_encoder', side_effect=lambda _: TinyEncoder(768)))
    stack.enter_context(patch('src.models.next_stage.FrozenMONET', side_effect=synthetic_monet))


class E14PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_pairs_preserve_protocol_and_reject_other_combinations(self):
        self.assertEqual(EXPERIMENT_GROUPS['singles'], ('E11', 'E12', 'E13'))
        self.assertEqual(EXPERIMENT_GROUPS['pairs'], PAIRS)
        self.assertEqual(EXPERIMENTS[-4:-1], PAIRS)
        reference = config('E10')
        for name, enabled in zip(PAIRS, ({'gate', 'panderm'}, {'gate', 'monet'}, {'panderm', 'monet'})):
            current = config(name)
            self.assertEqual({key for key in ('gate', 'panderm', 'monet') if current[key]}, enabled)
            self.assertEqual({k:current[k] for k in reference if k != 'experiment'},
                             {k:v for k,v in reference.items() if k != 'experiment'})
            for key in ('gate', 'panderm', 'monet'):
                with self.assertRaises(ValueError):
                    config(name, {key:not current[key]})
            contract = model_contract(current, 28, assets(name))
            self.assertEqual(contract['configuration_id'], name)
            self.assertEqual(contract['derm_dim'], 768 if current['panderm'] else 2048)
        for invalid in ('E14', 'E15', 'E14-PG'):
            with self.assertRaises(ValueError):
                config('E14-GP', {'experiment':invalid})

    def test_neutral_gate_preserves_single_component_initialization(self):
        data = batch(size=224)
        for pair, reference in (('E14-GP','E12'), ('E14-GM','E13')):
            with self.subTest(pair=pair), ExitStack() as stack:
                tiny_models(stack)
                torch.manual_seed(42)
                baseline = build_model(config(reference), 7).eval()
                torch.manual_seed(42)
                combined = build_model(config(pair), 7).eval()
                common = baseline.state_dict()
                for key, value in common.items():
                    self.assertTrue(torch.equal(value, combined.state_dict()[key]), key)
                self.assertTrue(torch.equal(baseline(data), combined(data)))
                combined(data).sum().backward()
                self.assertGreater(combined.gate.network[-1].weight.grad.abs().sum().item(), 0)

    def test_collector_keeps_pair_folds_separate_and_checks_contract_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = collect(root/'empty', root/'status.csv')
            self.assertEqual(len(rows), 90)
            for name in PAIRS:
                self.assertEqual([r['fold'] for r in rows if r['experiment']==name], list(range(5)))
        current = config('E14-GP')
        record = {'experiment':'E14-GP','configuration_id':'E14-GP','config':current,
                  'inputs':{'components':assets('E14-GP')},
                  'model_contract':model_contract(current, 28, assets('E14-GP'))}
        validate_run_identity(record)
        for bad in ('E14', 'E14-GM', 'E14-PM'):
            with self.assertRaises(ValueError):
                validate_run_identity({**record,'configuration_id':bad})

    def test_all_pairs_run_restart_replay_and_external_checkpoint_contract(self):
        def row(prefix, label, source='MILK'):
            key = f'{prefix}-{label}'
            return {'sample_id':f'{source}:{key}','source':source,'lesion_id':key,'group_id':key,
                'label':label,'label_id':NINE_CLASSES.index(label),'derm_path':'derm.png',
                'clinical_path':'clinical.png' if source=='MILK' else '', 'has_derm':1,
                'has_clinical':int(source=='MILK'),'metadata_key':key,'split':'dev','age_years':'40',
                'sex':'male','anatomical_region':'head','skin_tone_class':'2','derm_image_id':key,
                'clinical_image_id':key+'-c'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.arange(24*32*3,dtype=np.uint8).reshape(24,32,3)
            Image.fromarray(pixels).save(root/'derm.png')
            Image.fromarray(np.flipud(pixels)).save(root/'clinical.png')
            train = [row('train',label) for label in NINE_CLASSES]
            train += [row('isic',label,'ISIC2019') for label in NINE_CLASSES]
            val = [row('val',label) for label in NINE_CLASSES]
            write_csv(root/'val.csv',val)
            for name in PAIRS:
                with self.subTest(pair=name), ExitStack() as stack:
                    tiny_models(stack)
                    current = config(name,{'data_root':str(root),'workers':0,'batch_size':18,
                                           'max_epochs':2,'amp':'off'})
                    inputs = {'manifest_sha256':'synthetic','fold':0,'components':assets(name)}
                    checked = stack.enter_context(patch('src.experiment.preflight',return_value=(train,val,inputs)))
                    reference, interrupted = root/(name+'-reference'), root/(name+'-interrupted')
                    record = run(current,0,reference,device_name='cpu')
                    validate_run_identity(record)
                    calls = 0
                    def interrupt(*args,**kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 3:
                            raise KeyboardInterrupt()
                        return epoch(*args,**kwargs)
                    with patch('src.experiment.epoch',side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                        run(current,0,interrupted,device_name='cpu')
                    run(current,0,interrupted,device_name='cpu',resume=True)
                    first,second = load_checkpoint(reference/'last.pt'),load_checkpoint(interrupted/'last.pt')
                    self.assertEqual(first['model_contract']['configuration_id'],name)
                    for key,value in first['model'].items():
                        self.assertTrue(torch.equal(value,second['model'][key]),key)
                    for key,state in first['optimizer']['state'].items():
                        for field,value in state.items():
                            self.assertTrue(torch.equal(value,second['optimizer']['state'][key][field]),field)
                    self.assertEqual(first['optimizer']['param_groups'],second['optimizer']['param_groups'])
                    self.assertEqual((reference/'predictions.csv').read_bytes(),(interrupted/'predictions.csv').read_bytes())
                    predict(interrupted/'best.pt',root/'val.csv',interrupted/'reloaded.csv',device_name='cpu')
                    self.assertEqual((reference/'predictions.csv').read_bytes(),(interrupted/'reloaded.csv').read_bytes())
                    if current['gate']:
                        self.assertIn('gate_diagnostics',first['history'][0]['train'])
                    if current['monet']:
                        self.assertFalse(any(key.startswith(('monet.vision.','monet.projection.')) for key in first['model']))
                        self.assertIn('monet.targets',first['model'])
                    changed = copy.deepcopy(inputs)
                    changed['manifest_sha256'] = 'changed'
                    checked.return_value = (train,val,changed)
                    with self.assertRaisesRegex(ValueError,'Resume rejected'):
                        run(current,0,interrupted,device_name='cpu',resume=True)


if __name__ == '__main__':
    unittest.main()
