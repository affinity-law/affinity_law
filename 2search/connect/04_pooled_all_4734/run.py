"""Pool all 4,734 candidates from three providers (nine per dataset and configuration). Use stage-specific tree search and representative selection.

Process-local settings (law_search/config.py stays on disk):
1. Stage-specific tree parameters:
   - A (preprocessing): MAX_DEPTH=3, MIN_REL_GAIN=0.01 for concise rules driven by individual features.
   - B/C/E (backbone, head, composition): MAX_DEPTH=4, MIN_REL_GAIN=0.002 to capture feature interactions.
2. Statistical protocol:
   - Audit, split and normalize all rules (04); run 1,000 unit permutations and global BH-FDR correction (05);
     then 1,000 dataset bootstrap resamples (06), four-cell stability tests (07) and S/A/B/C/D grading (08).
3. Within-module representative selection (top two):
   - Rank deployable S/A rules for each (stage, module, target) by the original priority and retain up to two.
     This limits redundant related rules for one motif, targeting approximately 20-30 rules overall.
   - Keep the main table, full-precision thresholds and run configuration. Compute intermediates in a temporary directory.
"""
import argparse
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
LAW_SEARCH = HERE.parents[1] / 'law_search'
sys.path.insert(0, str(HERE.parent / 'support'))
sys.path.insert(0, str(LAW_SEARCH))
from connect_common import inspect_inputs, manifest_path, sha256, write_csv
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import config
    import discover_rules as dr
    import build_law_table as blt
    from prepare_data import build_binary, parse_conditions, rule_mask, W_COL
except ModuleNotFoundError as exc:
    raise SystemExit(f'Missing dependency {exc.name}. Install it with python -m pip install -r "{HERE.parent / "requirements.txt"}"') from exc

TAG = 'combined'
OUT_TAG = 'pooled_all'
STAGE_SEARCH_PARAMS = {
    'A': {'MAX_DEPTH': 3, 'MIN_REL_GAIN': 0.01},
    'B': {'MAX_DEPTH': 4, 'MIN_REL_GAIN': 0.002},
    'C': {'MAX_DEPTH': 4, 'MIN_REL_GAIN': 0.002},
    'E': {'MAX_DEPTH': 4, 'MIN_REL_GAIN': 0.002},
}
MAX_RULES_PER_MODULE = 2


@contextmanager
def stage_aware_search():
    orig_discover_rules = dr.discover_rules
    orig_select = blt.select_deployment_rules

    def stage_aware_discover_rules(units, stage, module, target):
        params = STAGE_SEARCH_PARAMS[stage]
        old_depth, old_gain = dr.MAX_DEPTH, dr.MIN_REL_GAIN
        try:
            dr.MAX_DEPTH = params['MAX_DEPTH']
            dr.MIN_REL_GAIN = params['MIN_REL_GAIN']
            return orig_discover_rules(units, stage, module, target)
        finally:
            dr.MAX_DEPTH, dr.MIN_REL_GAIN = old_depth, old_gain

    def topk_select_deployment_rules(law):
        pool = law[law['grade'].isin(['S', 'A'])].copy()
        pool['_grade_rank'] = pool['grade'].map({'S': 0, 'A': 1})
        pool = pool.sort_values(
            ['_grade_rank', 'lift_ci_lo', 'coverage', 'lift', 'n_features'],
            ascending=[True, False, False, False, True],
        )
        top_pool = pool.groupby(['stage', 'module', 'target'], sort=False).head(MAX_RULES_PER_MODULE)
        representatives = top_pool.drop_duplicates(['stage', 'module', 'target'], keep='first')
        return list(top_pool.index), set(representatives.index)

    dr.discover_rules = stage_aware_discover_rules
    blt.select_deployment_rules = topk_select_deployment_rules
    try:
        yield
    finally:
        dr.discover_rules = orig_discover_rules
        blt.select_deployment_rules = orig_select


def export_fixed_rules(work, deploy):
    """Export original tree thresholds so validation can reuse them directly."""
    pure = pd.read_csv(work / '04_pure_rules.csv', encoding='utf-8-sig')
    units = pd.read_csv(work / '00_unit_table.csv', encoding='utf-8-sig')
    rules = []
    for number, (_, row) in enumerate(deploy.iterrows(), 1):
        matches = pure[(pure.stage == row.stage) & (pure.module == row.module)
                       & (pure.target == row.target) & (pure.condition == row.condition)]
        if len(matches) != 1:
            raise ValueError(f'P{number:02d} No unique match for the full-precision threshold')
        conditions = matches.iloc[0].conditions_str
        mask = rule_mask(units, parse_conditions(conditions))
        labels = build_binary(units, row.stage, row.module, row.target).to_numpy()
        weights = units[W_COL].to_numpy()
        if (int(mask.sum()) != int(row.n_rule)
                or units.loc[mask, 'Dataset'].nunique() != row.n_dataset_confirm
                or not np.isclose(np.sum(labels[mask] * weights[mask]) / weights[mask].sum(), row.p_rule, rtol=1e-12, atol=0)):
            raise ValueError(f'P{number:02d} Full-precision thresholds did not reproduce discovery coverage or selection rate')
        rules.append({'rule_id': f'P{number:02d}', 'source_csv_row': number + 1,
                      'stage': row.stage, 'module': row.module, 'target': row.target,
                      'condition': row.condition, 'condition_qual': row.condition_qual,
                      'conditions_str_exact': conditions, 'pooled_grade': row.grade,
                      'pooled_lift': row.lift, 'pooled_n_datasets': int(row.n_dataset_confirm)})
    write_csv(work / 'pooled_all_fixed_rules.csv', rules)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Pool all three providers, discover laws and export full-precision thresholds.')
    parser.add_argument('--input-dir', type=Path, default=config.INPUT_DIR, help='Directory with the 12 candidate CSVs (default ../law_search/inputs)')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs', help='Output directory (default outputs beside this script)')
    parser.add_argument('--check-only', action='store_true', help='Check input completeness only')
    args = parser.parse_args(argv)
    args.input_dir, args.output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    summary = inspect_inputs(args.input_dir)
    if args.check_only:
        print(f"Checks passed:{summary['datasets']} datasets,{summary['candidate_rows']} candidate records.")
        print(f'Input:{args.input_dir}\nOutput:{args.output_dir}')
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"pooled {summary['candidate_rows']} candidates ({summary['dataset_config_units']} units x 9),"
          f"using stage-specific tree search and within-module top-{MAX_RULES_PER_MODULE} representative selection.", flush=True)
    with TemporaryDirectory(prefix='.pooled-all-run-', dir=args.output_dir) as temp, stage_aware_search():
        work = Path(temp)
        steps = [('Prepare candidate data', lambda: blt.prepare_data(work, TAG, args.input_dir)),
                 ('Stage-specific tree search and purity audit', lambda: dr.discover(work)),
                 ('1,000 permutation tests and BH correction', lambda: blt.compute_significance(work)),
                 ('1,000 dataset bootstrap resamples', lambda: blt.compute_reproducibility(work)),
                 ('Four-cell stability test', lambda: blt.compute_stability(work)),
                 ('Grade and select representative rules', lambda: blt.export_law_table(work))]
        with (work / 'pipeline.log').open('w', encoding='utf-8') as log:
            for number, (label, action) in enumerate(steps, 1):
                print(f'[{number}/{len(steps)}] {label}……', flush=True)
                with redirect_stdout(log):
                    action()
                log.flush()

        for name, checksum in summary['input_sha256'].items():
            if sha256(args.input_dir / name) != checksum:
                raise RuntimeError(f'Input changed during the run:{name}')

        deploy_df = pd.read_csv(work / '08_cookbook_law_table_deploy.csv', encoding='utf-8-sig')
        audit_df = pd.read_csv(work / '08_cookbook_law_table_audit.csv', encoding='utf-8-sig')
        full_sa_df = audit_df[audit_df['grade'].isin(['S', 'A'])].copy()
        export_fixed_rules(work, deploy_df)

        summary.update(
            scheme='Stage-specific tree search (original A settings; B/C/E gain gate 0.002, depth 4) and top-two representatives per module',
            model_selection='Include the top three candidates from each provider for each dataset and configuration (nine total)',
            stage_search_params=STAGE_SEARCH_PARAMS,
            max_rules_per_module=MAX_RULES_PER_MODULE,
            n_candidate_pure_rules=int(len(audit_df)),
            n_full_sa_rules_before_top2=int(len(full_sa_df)),
            n_rules=int(len(deploy_df)),
            grades={k: int(v) for k, v in deploy_df['grade'].value_counts().items()},
            rules_by_stage={k: int(v) for k, v in deploy_df['stage'].value_counts().items()},
            b_stage_modules_covered=sorted(deploy_df.loc[deploy_df['stage'] == 'B', 'module'].unique().tolist()),
            config_sha256=hashlib.sha256((LAW_SEARCH / 'config.py').read_bytes()).hexdigest(),
            tree_seed=config.SEED,
            min_leaf=config.MIN_LEAF,
            permutation_count=config.B_PERM,
            bootstrap_count=config.N_BOOT,
            input_dir=manifest_path(args.input_dir),
            fixed_rules_sha256=sha256(work / 'pooled_all_fixed_rules.csv'),
            code_sha256={p.name: sha256(p) for p in [Path(__file__), HERE.parent / 'support/connect_common.py', *LAW_SEARCH.glob('*.py')]},
            runtime={'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        )
        (work / 'run_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )

        moves = {
            '08_cookbook_law_table_deploy.csv': f'{OUT_TAG}_law_table.csv',
            'pooled_all_fixed_rules.csv': 'pooled_all_fixed_rules.csv',
            'run_summary.json': 'run_summary.json',
        }
        for src, dst in moves.items():
            shutil.copyfile(work / src, args.output_dir / dst)

    print(f"Completed. Main table: {summary['n_rules']} rules; distribution by stage {summary['rules_by_stage']}, grades {summary['grades']}.",
          flush=True)
    print(f'Results:{args.output_dir}\nNext: python "{HERE.parent / "01_exact_rule_validation" / "run.py"}"', flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
