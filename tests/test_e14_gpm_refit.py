"""Public GPM refit semantics with artificial data and tiny encoders."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from src.class_schema import NINE_CLASSES
from src.data.common import write_csv
from src.experiment import code_record, epoch, load_checkpoint, run
from src.models.factory import build_model
import test_e14_pairs as pairs


class GPMRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_pool_no_holdout_images_and_fixed_epoch_completion(self):
        def row(prefix, label, source='MILK'):
            key = prefix + '-' + label
            return dict(sample_id=source+':'+key, source=source, lesion_id=key, group_id=key,
                        label=label, label_id=NINE_CLASSES.index(label), derm_path='derm.png',
                        clinical_path='clinical.png' if source=='MILK' else '', has_derm=1,
                        has_clinical=int(source=='MILK'), metadata_key=key, split='dev', age_years='40',
                        sex='male', anatomical_region='head', skin_tone_class='2',
                        derm_image_id=key, clinical_image_id=key+'-c')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            Image.new('RGB', (24, 32), 'red').save(root / 'derm.png')
            Image.new('RGB', (24, 32), 'blue').save(root / 'clinical.png')
            dev = [row(prefix, label) for prefix in ('fold-train', 'fold-val') for label in NINE_CLASSES]
            isic = [row('external', label, 'ISIC2019') for label in NINE_CLASSES]
            holdout = [{**row('holdout', label), 'split': 'holdout', 'derm_path': 'DO_NOT_OPEN.png',
                        'clinical_path': 'DO_NOT_OPEN.png'} for label in NINE_CLASSES]
            for name, rows in (('milk_dev', dev), ('isic_train', isic), ('milk_holdout', holdout)):
                write_csv(root / (name + '.csv'), rows)
            config = pairs.config('E14-GPM', {'data_root': str(root), 'manifests': str(root),
                                            'workers': 0, 'batch_size': 27, 'max_epochs': 2, 'amp': 'off'})
            inputs = dict(manifest_sha256='synthetic', fold=0, components=pairs.assets('E14-GPM'))
            summary = dict(status='completed', purpose='formal', fold_count=5, config=config,
                           manifest_sha256='synthetic', components=inputs['components'],
                           code_sha256=code_record()['source_sha256'], refit_epochs=2)
            pairs.tiny_models(stack)
            stack.enter_context(patch('src.experiment.preflight', return_value=(dev[:9]+isic, dev[9:], inputs)))
            model_call = stack.enter_context(patch('src.experiment.build_model', wraps=build_model))
            target = root / 'refit'
            count = 0
            def interrupt(*args, **kwargs):
                nonlocal count
                count += 1
                self.assertIsNotNone(args[5])  # Every epoch has an optimizer: no validation pass.
                if count == 2:
                    raise KeyboardInterrupt()
                return epoch(*args, **kwargs)
            with patch('src.experiment.epoch', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                run(config, 0, target, refit_summary=summary, device_name='cpu')
            first = load_checkpoint(target / 'last.pt')
            self.assertFalse(first['refit_complete'])
            self.assertEqual(first['epoch'], 1)
            result = run(config, 0, target, refit_summary=summary, device_name='cpu', resume=True)
            self.assertEqual((result['train_samples'], result['val_samples'], result['epochs_completed']), (27, 0, 2))
            self.assertEqual(result['phase'], 'development_refit')
            self.assertIsNone(result['best_macro_f1'])
            self.assertTrue(all(call.kwargs['initialize'] for call in model_call.call_args_list))
            last = load_checkpoint(target / 'last.pt')
            self.assertTrue(last['refit_complete'])
            self.assertTrue(all(item['val'] is None for item in last['history']))
            self.assertFalse((target / 'predictions.csv').exists())
            self.assertFalse((target / 'validation_labels.csv').exists())
            for field, value in [('code_sha256', 'changed'), ('components', {})]:
                bad = copy.deepcopy(summary)
                bad[field] = value
                with self.assertRaises(ValueError):
                    run(config, 0, root / ('bad-' + field), refit_summary=bad, device_name='cpu')


if __name__ == '__main__':
    unittest.main()
