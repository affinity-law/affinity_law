"""Validate stage 04 full-precision thresholds independently for each of the three providers."""
import argparse
import io
import json
import math
import sys
import time
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
CONNECT = HERE.parent
SEARCH = CONNECT.parent
sys.path.insert(0, str(CONNECT / 'support'))
sys.path.insert(0, str(SEARCH / 'law_search'))
from connect_common import (default_document, inspect_inputs, read_csv, read_document_rules,
                            select_validation_family, sha256, write_json, reference_labels, reference_profile,
                            load_label_units, manifest_path, REFERENCE_PATH)
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import config
    import build_law_table as stats
    from prepare_data import build_unit_table, build_binary, rule_mask, parse_conditions, W_COL
except ModuleNotFoundError as exc:
    raise SystemExit(f'Missing dependency {exc.name}. Install it with python -m pip install -r "{CONNECT / "requirements.txt"}"') from exc


def save_csv(output_dir, name, rows):
    pd.DataFrame(rows).to_csv(output_dir / name, index=False, encoding='utf-8-sig')


def load_inputs(args):
    fixed_path = args.pooled_dir / 'pooled_all_fixed_rules.csv'
    if not fixed_path.is_file():
        raise FileNotFoundError(f'Missing {fixed_path}. Run stage 04 first or specify its output with --pooled-dir.')
    info = inspect_inputs(args.input_dir)
    discovery = json.loads((args.pooled_dir / 'run_summary.json').read_text(encoding='utf-8'))
    if info['input_sha256'] != discovery['input_sha256']:
        raise ValueError('Candidate inputs differ from stage 04. Check --input-dir and --pooled-dir')
    if sha256(fixed_path) != discovery['fixed_rules_sha256']:
        raise ValueError('Full-precision rule file changed. Rerun stage 04')
    if sha256(SEARCH / 'law_search/config.py') != discovery['config_sha256']:
        raise ValueError('Statistical configuration differs from stage 04. Rerun stage 04')
    family = json.loads((HERE / 'rule_family.json').read_text(encoding='utf-8'))
    displayed = read_document_rules(args.docx)
    fixed = read_csv(fixed_path)
    rules = select_validation_family(fixed, family['rule_ids'], displayed)
    lookup = {r['rule_id']: r for r in rules}
    feature_order = list(json.loads(REFERENCE_PATH.read_text(encoding='utf-8'))['features'])
    label_units = load_label_units(args.input_dir)
    for row in displayed:
        rule = lookup[row['rule_id']]
        labels = reference_labels(rule['conditions_str_exact'], units=label_units)
        expected = [labels.get(feature, '') for feature in feature_order]
        actual = row['docx_cells'][3:10]
        if expected != actual:
            raise ValueError(f"{row['rule_id']} has document labels inconsistent with median feature bands of the covered datasets")
    return info, rules, displayed


def source_hashes(args):
    paths = [args.docx, args.pooled_dir / 'pooled_all_fixed_rules.csv',
             args.pooled_dir / 'pooled_all_law_table.csv', args.pooled_dir / 'run_summary.json',
             HERE / 'rule_family.json', Path(__file__), CONNECT / 'support/connect_common.py', REFERENCE_PATH]
    paths += sorted((SEARCH / 'law_search').glob('*.py'))
    paths += sorted(args.input_dir.glob('*.csv'))
    return {manifest_path(p): sha256(p) for p in paths}


def document_check(displayed, summaries, pooled_dir):
    by_id = {r['rule_id']: r for r in summaries}
    pooled = read_csv(pooled_dir / 'pooled_all_law_table.csv')
    mismatches, ai_checks, cell_checks = [], 0, 0
    for item in displayed:
        rid = item['rule_id']
        source = pooled[int(rid[1:]) - 1]
        for i, (mode, horizon) in enumerate(config.CELLS):
            n = int(source[f'n_dataset_{mode}_{horizon}'])
            value = source[f'lift_{mode}_{horizon}']
            lift = float(value) if value else float('nan')
            expected = '—' if n < config.MIN_CELL_DATASETS or not np.isfinite(lift) else ('✓' if lift > 1 else '×')
            actual = item['docx_cells'][10 + i]
            if actual != expected:
                mismatches.append({'rule_id': rid, 'column': f'{mode}_{horizon}', 'document': actual, 'computed': expected})
            cell_checks += 1
        # Some document cells intentionally show only the check mark, without the grade.
        if len(item['docx_cells']) >= 17:
            for ai, value in zip(config.AI_TYPES, item['docx_cells'][-3:]):
                result = by_id[rid]
                mark = '✓' if result[f'{ai}_passes_revalidation'] else '×'
                expected = mark + ' ' + result[f'{ai}_grade']
                if value not in (mark, expected):
                    mismatches.append({'rule_id': rid, 'column': ai, 'document': value, 'computed': expected})
                ai_checks += 1
    return {'displayed_rules': len(displayed), 'ai_cells_checked': ai_checks,
            'forecast_cells_checked': cell_checks, 'mismatches': mismatches,
            'status': 'passed' if not mismatches else 'mismatch'}


def validate_ai(ai, candidates, input_dir, input_info):
    began = time.monotonic()
    with TemporaryDirectory(prefix='table2-ai-') as temp, redirect_stdout(io.StringIO()):
        build_unit_table(temp, ai, input_dir)
        units = pd.read_csv(Path(temp) / '00_unit_table.csv', encoding='utf-8-sig')
    assert len(units) == input_info['rows_by_ai'][ai] and units.Dataset.nunique() == input_info['datasets']
    assert units.groupby('unit_id', sort=False).size().eq(3).all()
    ids = units.unit_id.to_numpy().reshape(-1, 3)
    assert np.all(ids == ids[:, :1])
    assert np.isfinite(units[W_COL]).all() and (units[W_COL] > 0).all()
    rules = [dict(stage=r['stage'], module=r['module'], target=r['target'],
                  conditions=parse_conditions(r['conditions_str_exact'])) for r in candidates]
    print(f'{ai}: {config.B_PERM} permutations for {len(rules)} fixed rules', flush=True)
    permute = stats._unit_permutation(units, np.random.default_rng(config.PERM_SEED))
    results = [stats.rule_permutation_stats(units, r['stage'], r['module'], r['target'],
                                           r['conditions'], permute, config.B_PERM) for r in rules]
    q = stats.bh_fdr([r['p_rule_wald'] for r in results])
    print(f'{ai}: {config.N_BOOT} dataset bootstrap replicates', flush=True)
    repro, med, lo, hi, valid = stats.bootstrap_repro(units, rules, config.N_BOOT)
    details = []
    min_stable = math.ceil(units.Dataset.nunique() * config.MIN_STABLE_DATASET_FRACTION)
    for i, (candidate, rule, result) in enumerate(zip(candidates, rules, results)):
        mask = rule_mask(units, rule['conditions'])
        y, weights = build_binary(units, rule['stage'], rule['module'], rule['target']).to_numpy(), units[W_COL].to_numpy()
        assert result['n_rule_confirm'] == mask.sum()
        assert math.isclose(result['p_rule_confirm'], np.sum(y[mask] * weights[mask]) / weights[mask].sum(), rel_tol=1e-12)
        lifts, supports, supported, effective = stats.cell_stability(units, **rule)
        pure, pockets = stats.rule_purity(units, **rule)
        # Match the original pipeline's rounding before assign_grade; retain raw values too.
        rounded_repro, rounded_lo = float(np.round(repro[i], 3)), float(np.round(lo[i], 2))
        grade = stats.assign_grade(q[i], rounded_repro, rounded_lo, effective,
                                   result['coverage_dataset_confirm'], result['n_dataset_confirm'],
                                   list(supports.values()), pure, min_stable)
        reasons = []
        if not np.isfinite(q[i]) or q[i] >= config.FDR_EXPLORATORY:
            reasons.append('q>=0.10 or unavailable')
        if not np.isfinite(rounded_repro) or rounded_repro < config.REPRO_MIN:
            reasons.append('Bootstrap directional support<0.80 or unavailable')
        if effective < 2:
            reasons.append('Fewer than two valid cells')
        if result['n_dataset_confirm'] < config.MIN_CANDIDATE_DATASETS:
            reasons.append('Fewer than 12 covered datasets')
        grade_reasons = []
        if q[i] >= config.FDR_STRONG or not np.isfinite(q[i]): grade_reasons.append('q>=0.05 or unavailable')
        if rounded_repro < config.REPRO_MIN or not np.isfinite(rounded_repro): grade_reasons.append('Insufficient bootstrap support')
        if rounded_lo <= 1 or not np.isfinite(rounded_lo): grade_reasons.append('95% CI lower bound<=1 or unavailable')
        if not pure: grade_reasons.append('Counter-preference region present')
        if effective < 2: grade_reasons.append('Fewer than two valid cells')
        if result['n_dataset_confirm'] < min_stable: grade_reasons.append(f'Insufficient grade A coverage: fewer than{min_stable}datasets')
        detail = {'rule_id': candidate['rule_id'], 'ai': ai, 'stage': rule['stage'], 'module': rule['module'],
                  'condition': candidate['condition'], 'conditions_str_exact': candidate['conditions_str_exact'],
                  **result, 'q_rule': q[i], 'repro': rounded_repro, 'repro_raw': repro[i],
                  'n_boot_valid': int(valid[i]), 'lift_med_boot': float(np.round(med[i], 2)),
                  'lift_ci_lo': rounded_lo, 'lift_ci_hi': float(np.round(hi[i], 2)),
                  'lift_ci_lo_raw': lo[i], 'lift_ci_hi_raw': hi[i],
                  **lifts, **supports, 'n_cells_supported': supported, 'n_cells_effective': effective,
                  'purity': 'Pure' if pure else 'Contains a counter-preference region',
                  'impure_detail': json.dumps(pockets, ensure_ascii=False), 'grade': grade,
                  'passes_S_or_A': int(grade in ('S', 'A')), 'passes_revalidation': int(not reasons),
                  'failure_reason': '; '.join(reasons), 'SA_failure_reason': '; '.join(grade_reasons)}
        details.append(detail)
    print(f'{ai}: pass={sum(d["passes_revalidation"] for d in details)}/{len(details)}, '
          f'S/A={sum(d["passes_S_or_A"] for d in details)}, {time.monotonic()-began:.1f}s', flush=True)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validate fixed stage 04 thresholds by provider using the original 29-rule BH family.')
    parser.add_argument('--input-dir', type=Path, default=config.INPUT_DIR, help='Directory with the 12 candidate CSVs')
    parser.add_argument('--pooled-dir', type=Path, default=CONNECT / '04_pooled_all_4734/outputs', help='Stage 04 output directory')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs', help='Write both results here; defaults are the summary CSV under connect and the manifest under the stage outputs directory')
    parser.add_argument('--docx', type=Path, help='Read-only display table; defaults to the sole .docx under connect')
    parser.add_argument('--check-only', action='store_true', help='Check only inputs, full-precision thresholds, test family and document conditions')
    args = parser.parse_args(argv)
    if args.docx is None:
        try:
            args.docx = default_document(CONNECT)
        except ValueError as exc:
            parser.error(str(exc))
    for name in ('input_dir', 'pooled_dir', 'output_dir', 'docx'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        input_info, candidates, displayed = load_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Input checks passed:{input_info["datasets"]} datasets; test family: {len(candidates)} rules; document displays {len(displayed)} rules.', flush=True)
    if args.check_only:
        return
    before = source_hashes(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Stage results in a temporary directory so a failed calculation cannot mix old and new tables.
    with TemporaryDirectory(prefix='.validation-', dir=args.output_dir) as temp:
        output = Path(temp)
        all_details = []
        for ai in config.AI_TYPES:
            all_details += validate_ai(ai, candidates, args.input_dir, input_info)
        indexed = {(r['rule_id'], r['ai']): r for r in all_details}
        summaries = []
        label_units = load_label_units(args.input_dir)
        for candidate in candidates:
            summary = dict(candidate)
            # Retain the original condition_qual for provenance; use explicit reference-band columns for the new table.
            profile = reference_profile(candidate['conditions_str_exact'], units=label_units)
            for feature in json.loads(REFERENCE_PATH.read_text(encoding='utf-8'))['features']:
                item = profile.get(feature, {})
                summary[feature + '_reference_label'] = item.get('label', '')
                summary[feature + '_reference_median'] = item.get('median', '')
                summary[feature + '_reference_dataset_count'] = item.get('datasets', '')
            for ai in config.AI_TYPES:
                detail = indexed[candidate['rule_id'], ai]
                for field in ['grade', 'passes_revalidation', 'passes_S_or_A', 'lift_confirm', 'q_rule', 'repro',
                              'lift_ci_lo', 'lift_ci_hi', 'n_cells_effective', 'n_dataset_confirm', 'failure_reason', 'SA_failure_reason']:
                    summary[f'{ai}_{field}'] = detail[field]
            summary['validated_ai_count'] = sum(summary[f'{ai}_passes_revalidation'] for ai in config.AI_TYPES)
            summary['SA_ai_count'] = sum(summary[f'{ai}_passes_S_or_A'] for ai in config.AI_TYPES)
            summaries.append(summary)
        visible = sorted((r for r in summaries if r['in_document']), key=lambda r: r['table_order'])
        save_csv(output, 'document_summary.csv', visible)
        assert len(all_details) == len(candidates) * 3
        if source_hashes(args) != before:
            raise RuntimeError('Inputs changed during computation. Results were not published')
        check = document_check(displayed, summaries, args.pooled_dir)
        manifest = {'status': 'completed', 'source_docx': manifest_path(args.docx),
                    'n_rules': len(candidates), 'n_displayed_rules': len(displayed),
                    'family_rule_ids': [c['rule_id'] for c in candidates],
                    'displayed_rule_ids': [r['rule_id'] for r in displayed],
                    'per_ai': {ai: {'passes_revalidation': sum(d['passes_revalidation'] for d in all_details if d['ai'] == ai),
                                   'passes_S_or_A': sum(d['passes_S_or_A'] for d in all_details if d['ai'] == ai),
                                   'grades': dict(Counter(d['grade'] for d in all_details if d['ai'] == ai)),
                                   'candidate_rows': input_info['rows_by_ai'][ai],
                                   'datasets': input_info['datasets'], 'dataset_config_units': input_info['dataset_config_units']}
                               for ai in config.AI_TYPES},
                    'validated_ai_distribution': dict(Counter(s['validated_ai_count'] for s in summaries)),
                    'fdr_family': 'Each AI separately; all 29 original fixed rules, ascending P ID order; document filtering happens after correction.',
                    'threshold_policy': 'Use the full-precision thresholds exported by 04; do not rediscover or use rounded display thresholds.',
                    'parameters': {k: v for k, v in vars(config).items() if k.isupper() and isinstance(v, (int, float, str, bool, list, tuple))},
                    'purity_min_frac_used': stats.PURITY_MIN_FRAC,
                    'validation_criteria': 'q<0.10; repro>=0.80; effective cells>=2; covered datasets>=12',
                    'source_files_unchanged': True, 'input_sha256': before,
                    'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                    'document_check': check,
                    'limitations': 'Same-source per-AI revalidation; the AI datasets also contributed to pooled discovery, so this is not independent validation.'}
        write_json(output / 'run_manifest.json', manifest)
        for path in output.iterdir():
            destination = (CONNECT / path.name if path.name == 'document_summary.csv'
                           and args.output_dir == (HERE / 'outputs').resolve() else args.output_dir / path.name)
            path.replace(destination)
    for ai, result in manifest['per_ai'].items():
        print(f'{ai}: {result["passes_revalidation"]}/{len(candidates)} passed revalidation;{result["passes_S_or_A"]} reached grade S/A.')
    summary_path = ((CONNECT if args.output_dir == (HERE / 'outputs').resolve() else args.output_dir)
                    / 'document_summary.csv')
    print(f'Summary data:{summary_path}\nRun manifest:{args.output_dir / "run_manifest.json"}')
    print(f'Document check:{check["status"]}(read-only Word document)')
    if check['mismatches']:
        raise SystemExit('Document differs from computed results. Inspect document_check in outputs/run_manifest.json.')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
