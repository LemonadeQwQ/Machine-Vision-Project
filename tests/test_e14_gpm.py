"""Synthetic three-component contracts; cluster scheduling stays outside public."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import torch
from src.experiment_registry import EXPERIMENTS, EXPERIMENT_GROUPS, validate_run_identity
from src.models.factory import build_model, model_contract
from test_e11_e13 import batch
import test_e14_pairs as pairs


class E14GPMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_identity_and_unchanged_pair_protocol(self):
        current, reference = pairs.config('E14-GPM'), pairs.config('E14-PM')
        self.assertEqual(EXPERIMENT_GROUPS['triple'], ('E14-GPM',))
        self.assertEqual(EXPERIMENTS[-1], 'E14-GPM')
        self.assertEqual(len(EXPERIMENTS), 18)
        self.assertEqual({k:v for k,v in current.items() if k not in ('gate','experiment')},
                         {k:v for k,v in reference.items() if k not in ('gate','experiment')})
        for flag in ('gate','panderm','monet'):
            self.assertIs(current[flag], True)
            with self.assertRaises(ValueError):
                pairs.config('E14-GPM', {flag:False})
        contract = model_contract(current, 28, pairs.assets('E14-GPM'))
        self.assertEqual(contract['derm_dim'], 768)
        self.assertEqual(contract['external_checkpoint_components'], ['monet.vision','monet.projection'])
        run = {'experiment':'E14-GPM','configuration_id':'E14-GPM','config':current,
               'inputs':{'components':pairs.assets('E14-GPM')},'model_contract':contract}
        validate_run_identity(run)
        with self.assertRaises(ValueError):
            validate_run_identity({**run,'configuration_id':'E14-PM'})

    def test_neutral_gate_preserves_pm_initialization_and_missing_clinical(self):
        with ExitStack() as stack:
            pairs.tiny_models(stack)
            torch.manual_seed(42)
            baseline = build_model(pairs.config('E14-PM'),7).eval()
            torch.manual_seed(42)
            triple = build_model(pairs.config('E14-GPM'),7).eval()
            for key,value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value,triple.state_dict()[key]),key)
            data=batch(size=224)
            self.assertTrue(torch.equal(baseline(data),triple(data)))
            triple(data).sum().backward()
            self.assertTrue(torch.isfinite(triple.gate.network[-1].weight.grad).all())
            self.assertGreater(triple.gate.network[-1].weight.grad.abs().sum().item(),0)
            self.assertTrue(all(p.grad is None for p in triple.monet.parameters()))
            data['modality_mask'][:,1]=0
            with patch.object(triple.encoders['clinical'],'forward',side_effect=AssertionError('missing clinical encoded')):
                self.assertTrue(torch.equal(baseline(data),triple(data)))
            self.assertTrue(torch.equal(triple.last_gate,torch.ones_like(triple.last_gate)))

    def test_gpm_restart_replay_and_external_checkpoint_contract(self):
        with patch.object(pairs,'PAIRS',('E14-GPM',)):
            pairs.E14PairTests.test_all_pairs_run_restart_replay_and_external_checkpoint_contract(self)


if __name__ == '__main__':
    unittest.main()
