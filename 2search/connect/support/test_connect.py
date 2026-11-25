"""Check exact thresholds, the 29-rule family, input completeness and portable entry points."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from connect_common import (AI_TYPES, default_document, inspect_inputs,
                            read_document_rules, select_validation_family, write_csv, reference_labels, reference_profile,
                            load_label_units, read_csv)

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_label_selection_matches_original_discovery_coverage(self):
        units = load_label_units()
        for rule in read_csv(ROOT / '04_pooled_all_4734/outputs/pooled_all_fixed_rules.csv'):
            profile = reference_profile(rule['conditions_str_exact'], units=units)
            for feature, item in profile.items():
                if feature != 'multivariate_structure_score':
                    self.assertEqual(item['datasets'], int(rule['pooled_n_datasets']), rule['rule_id'])

    def test_median_labels_use_joint_conditions_and_unique_datasets(self):
        reference = {'features': {'trend': {'cuts': [0.1, 0.3]}, 'noise': {'cuts': [0.1, 0.3]}}}
        units = ([{'Dataset': 'a', 'trend': 0.05, 'noise': 0.9}] * 10
                 + [{'Dataset': 'b', 'trend': 0.25, 'noise': 0.9},
                    {'Dataset': 'c', 'trend': 0.4, 'noise': 0.1}])
        result = reference_profile('trend:≤:0.9|noise:>:0.3', reference, units=units)
        self.assertEqual(result['trend']['datasets'], 2)
        self.assertAlmostEqual(result['trend']['median'], 0.15)
        self.assertEqual(result['trend']['label'], 'Mid')
        self.assertEqual(result['noise']['label'], 'High')

    def test_median_labels_use_closed_upper_boundaries(self):
        reference = {'features': {'trend': {'cuts': [0.1, 0.3]}}}
        for value, expected in [(0.1, 'Low'), (0.3, 'Mid'), (0.31, 'High')]:
            self.assertEqual(reference_labels('trend:>:0', reference, units=[{'Dataset': 'a', 'trend': value}]),
                             {'trend': expected})

    def test_univariate_coherence_is_not_labelled_as_low(self):
        reference = {'features': {'multivariate_structure_score': {'cuts': [0.2, 0.5]}}}
        units = [{'Dataset': str(i), 'data_type': 0, 'multivariate_structure_score': 0} for i in range(10)]
        units.append({'Dataset': 'multi', 'data_type': 1, 'multivariate_structure_score': 0.4})
        result = reference_profile('multivariate_structure_score:≤:0.5', reference, units=units)
        self.assertEqual(result['multivariate_structure_score'], {'label': 'Mid', 'median': 0.4, 'datasets': 1})
        with self.assertRaisesRegex(ValueError, 'No covered datasets available'):
            reference_labels('multivariate_structure_score:≤:0.1', reference, units=units)

    def test_renamed_document_and_ambiguous_choices(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'law.docx').touch()
            (folder / '~$law.docx').touch()
            self.assertEqual(default_document(folder).name, 'law.docx')
            (folder / 'another.docx').touch()
            with self.assertRaises(ValueError):
                default_document(folder)

    def test_document_filter_does_not_change_fdr_family(self):
        family = json.loads((ROOT / '01_exact_rule_validation/rule_family.json').read_text(encoding='utf-8'))['rule_ids']
        displayed = read_document_rules(default_document(ROOT))
        fixed = [{'rule_id': f'P{i:02d}'} for i in range(1, 31)]
        candidates = select_validation_family(fixed, family, displayed)
        self.assertEqual(len(candidates), 29)
        self.assertEqual(sum(r['in_document'] for r in candidates), 27)
        self.assertEqual({r['rule_id'] for r in candidates if not r['in_document']}, {'P18', 'P22'})
        self.assertNotIn('P19', {r['rule_id'] for r in candidates})
        with self.assertRaises(ValueError):
            select_validation_family(fixed, family, [{'rule_id': 'P19', 'table_order': 1}])
        with self.assertRaises(ValueError):
            select_validation_family(fixed[1:], family, displayed)

    def test_top_three_checked_for_each_ai_not_only_pooled_total(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for mode in ('univariate', 'multivariate'):
                for horizon in (3, 12):
                    for ai in AI_TYPES:
                        rows = [{'Dataset': mode, 'TIME_LAG': 12, 'PRE_LEN': horizon}] * 3
                        write_csv(folder / f'{ai}_12_{horizon}_{mode}.csv', rows)
            self.assertEqual(inspect_inputs(folder)['candidate_rows'], 36)
            row = {'Dataset': 'univariate', 'TIME_LAG': 12, 'PRE_LEN': 3}
            write_csv(folder / 'deepseek_12_3_univariate.csv', [row] * 2)
            write_csv(folder / 'chatgpt_12_3_univariate.csv', [row] * 4)
            with self.assertRaisesRegex(ValueError, 'expected 3 candidates'):
                inspect_inputs(folder)

    def test_stage_hooks_restored_after_error(self):
        spec = importlib.util.spec_from_file_location('pooled_workflow', ROOT / '04_pooled_all_4734/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_discover = module.dr.discover_rules
        original_select = module.blt.select_deployment_rules
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with module.stage_aware_search():
                self.assertIsNot(module.dr.discover_rules, original_discover)
                raise RuntimeError('test failure')
        self.assertIs(module.dr.discover_rules, original_discover)
        self.assertIs(module.blt.select_deployment_rules, original_select)

    def test_check_only_works_outside_project_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('04_pooled_all_4734', '01_exact_rule_validation'):
                destination = Path(temp) / name
                result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / name / 'run.py'),
                                         '--check-only', '--output-dir', str(destination)],
                                        cwd=temp, capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(destination.exists())

    def test_document_check_accepts_blank_lift_for_uncovered_settings(self):
        spec = importlib.util.spec_from_file_location('validation_workflow', ROOT / '01_exact_rule_validation/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = {}
            for (mode, horizon), lift, count in zip(module.config.CELLS, ['1.5', '1.0', '', ''], [5, 5, 0, 0]):
                source[f'lift_{mode}_{horizon}'] = lift
                source[f'n_dataset_{mode}_{horizon}'] = count
            write_csv(folder / 'pooled_all_law_table.csv', [source])
            displayed = [{'rule_id': 'P01', 'docx_cells': [''] * 10 + ['✓', '×', '—', '—']}]
            result = module.document_check(displayed, [{'rule_id': 'P01'}], folder)
            self.assertEqual(result['status'], 'passed')
            displayed[0]['docx_cells'][12] = '✓'
            self.assertEqual(len(module.document_check(displayed, [{'rule_id': 'P01'}], folder)['mismatches']), 1)


if __name__ == '__main__':
    unittest.main()
