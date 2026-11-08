"""Plan integrity tests; no estimator fit/transform/predict is called."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src import pilot_extension_plan as extension
from src.model_training import _json_text, read_json, atomic_json


class ExtensionPlanTests(unittest.TestCase):
    def test_sampling_is_fixed_unique_and_old_five_are_prefix(self):
        spaces = {model: extension.parameter_space(model, technical=False)
                  for model in extension.MODEL_ORDER}
        first = extension._sample_configurations(spaces)
        self.assertEqual(first, extension._sample_configurations(spaces))
        for model in extension.MODEL_ORDER:
            self.assertEqual(len(first[model]), 20)
            self.assertEqual(len({x['config_sha256'] for x in first[model]}), 20)
            legacy = list(extension.ParameterSampler(spaces[model], n_iter=5, random_state=42))
            self.assertEqual(_json_text(legacy), _json_text([x['params'] for x in first[model][:5]]))
            self.assertEqual([x['candidate_number'] for x in first[model][5:10]], list(range(6, 11)))

    def test_saved_fold_results_must_match_fixed_prefix_and_be_finite(self):
        entries = [{'params': {'model__C': index + 1}} for index in range(20)]
        frame = pd.DataFrame([{'candidate': index, 'params_json': _json_text(entry['params']),
            'mean_neg_log_loss': -.2, **{f'fold_{fold}_neg_log_loss': -.2 for fold in range(3)}}
            for index, entry in enumerate(entries[:5])])
        extension._validate_cv_frame(frame, entries, 'unit test')
        changed = frame.copy(); changed.loc[0, 'params_json'] = _json_text({'model__C': 99})
        with self.assertRaisesRegex(ValueError, 'differs'):
            extension._validate_cv_frame(changed, entries, 'unit test')
        changed = frame.copy(); changed.loc[0, 'fold_0_neg_log_loss'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            extension._validate_cv_frame(changed, entries, 'unit test')
        with self.assertRaisesRegex(ValueError, 'exactly saved'):
            extension._validate_cv_frame(pd.concat([frame, frame.iloc[:1]]), entries, 'unit test')

    def test_existing_fixed_plan_cannot_be_overwritten_or_written_into_pilot(self):
        plan = {'fingerprint': 'fixed', 'configurations': {'example': [1, 2, 3]}}
        with tempfile.TemporaryDirectory(prefix='extension_plan_') as temporary:
            root = Path(temporary)
            prepared, pilot, output = root/'prepared', root/'pilot', root/'extension'
            with patch.object(extension, 'build_extension_plan', return_value=plan):
                self.assertEqual(extension.ensure_extension_plan(prepared, pilot, output), plan)
                self.assertEqual(extension.ensure_extension_plan(prepared, pilot, output), plan)
                file = output / extension.PLAN_FILE
                atomic_json(file, {'fingerprint': 'tampered'})
                with self.assertRaisesRegex(ValueError, 'Saved fixed plan changed'):
                    extension.ensure_extension_plan(prepared, pilot, output)
                self.assertEqual(read_json(file), {'fingerprint': 'tampered'})
                with self.assertRaisesRegex(ValueError, 'separate from the original'):
                    extension.ensure_extension_plan(prepared, pilot, pilot/'nested')


if __name__ == '__main__':
    unittest.main()
