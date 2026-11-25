"""Local workflow adapter for the current L and 3par protocols."""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
L_RESULTS = ROOT / '2search/compar_result/models_L'


def rule_hits(law, meta, horizon):
    import select_arch
    return [int(i) for i, row in law.iterrows()
            if row['grade'] in ('S', 'A')
            and select_arch.rule_triggers(select_arch.rule_conditions(row), meta, horizon)
            and not select_arch._in_corner(row, meta)]


def recorded_sample(sample, horizon):
    from common import discover_datasets
    ds = next(row for row in discover_datasets() if row['name'] == sample)
    original = L_RESULTS / sample / f'h{horizon}'
    selection = json.loads((original / 'selection.json').read_text(encoding='utf-8'))
    return ds, selection['spec'], original


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def status(folder, message, **extra):
    save(folder / 'status.json', {'status':'running', 'message':message, **extra})


def read_data(path):
    import numpy as np
    import pandas as pd
    if path.suffix == '.npy':
        # Refuse object arrays: browser uploads are untrusted and must never unpickle.
        values = np.load(path, allow_pickle=False)
        if values.dtype.kind not in 'fiu':
            raise ValueError('NPY must contain numeric values. Object, complex, Boolean, and date arrays are not supported.')
    else:
        import csv
        with path.open(encoding='utf-8-sig', newline='') as file:
            records = list(csv.reader(file))
        if not records or not records[0]:
            raise ValueError('The CSV file is empty.')
        count = len(records[0])
        if any(len(row) != count for row in records):
            raise ValueError('CSV rows have inconsistent column counts or contain blank rows.')
        def numeric(value):
            try:
                float(value)
                return value.strip() != ''
            except ValueError:
                return False
        first = records[0]
        numeric_tail = all(numeric(value) for value in first[1:]) if count > 1 else numeric(first[0])
        timestamp_first = count > 1 and not numeric(first[0]) and numeric_tail
        if timestamp_first:
            try:
                pd.to_datetime(first[0], errors='raise')
            except (ValueError, TypeError):
                timestamp_first = False
        if not all(numeric(value) for value in first) and not timestamp_first:
            if any(numeric(value) for value in first):
                raise ValueError('The first row mixes numeric and invalid fields; the header cannot be identified reliably.')
            records = records[1:]
        if not records:
            raise ValueError('The CSV contains a header but no data.')
        frame = pd.DataFrame(records)
        if count > 1 and not all(numeric(value) for value in frame.iloc[:, 0]):
            try:
                times = pd.to_datetime(frame.iloc[:, 0], errors='raise', format='mixed')
            except (ValueError, TypeError) as exc:
                raise ValueError('The first CSV column must contain numeric values or recognizable timestamps.') from exc
            if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
                raise ValueError('Timestamps are missing, duplicated, or out of order. Correct them before importing; the app does not reorder data.')
            frame = frame.iloc[:, 1:]
        if frame.apply(lambda col: col.str.strip().eq('').any()).any():
            raise ValueError('The CSV contains missing values. Clean the data before importing.')
        try:
            values = frame.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError('CSV variable columns contain nonnumeric values.') from exc
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError('Data must have shape [time] or [time, variables].')
    if not np.isfinite(values).all():
        raise ValueError('The data contains NaN or infinite values. Clean it before importing.')
    return values


def result_for(folder, values, meta, spec, code, horizon, name, source='computed'):
    import hashlib
    import numpy as np
    from config import LAW_TABLE
    from select_arch import load_law
    hits = rule_hits(load_law(), meta, horizon)
    indices = np.linspace(0, len(values) - 1, min(240, len(values)), dtype=int)
    return {'id':folder.name, 'job_id':folder.name, 'name':name, 'source':source,
            'rows':len(values), 'channels':int(values.shape[1]), 'horizon':horizon,
            'indices':indices.tolist(), 'preview':values[indices, :min(3, values.shape[1])].tolist(),
            'meta':meta, 'spec':spec, 'hits':hits, 'model_code':code, 'tuned':False,
            'law_table_sha256':hashlib.sha256(LAW_TABLE.read_bytes()).hexdigest(),
            'selection_status':'L architecture selected on validation data'}


def analyze(folder, task):
    import numpy as np
    import pandas as pd
    from model_assembler import build_model_code
    import compute_metrics
    import exp_L
    from config import TIME_LAG, TRAIN_RATIO, VAL_RATIO
    horizon = task['horizon']
    status(folder, 'Validating the data format and temporal splits')
    values = read_data(folder / ('upload' + task['suffix']))
    train_end = int(len(values) * TRAIN_RATIO)
    val_end = train_end + int(len(values) * VAL_RATIO)
    if min(train_end, val_end-train_end, len(values)-val_end) < TIME_LAG + horizon:
        raise ValueError('After the 60/20/20 split, there are too few observations for a 12-step input and the selected horizon. Provide more time points.')
    path = folder / 'data.npy'
    np.save(path, values)
    status(folder, 'Computing 39 indicators and 7 meta-features')
    compute_metrics.compute_metrics_for_npy(str(path), str(folder / 'metrics'), seasonal_period=0)
    meta_table = pd.read_csv(folder / 'metrics/meta_features.csv', encoding='utf-8-sig')
    meta = dict(zip(meta_table.indicator_name, meta_table.value.astype(float)))
    status(folder, 'Selecting among up to six L architecture candidates using validation RMSE')
    exp_L.RUN_ROOT = folder / 'selection'
    ds = {'name':'uploaded_data', 'npy_path':str(path), 'station_num':values.shape[1],
          'meta_csv':str(folder / 'metrics/meta_features.csv')}
    spec, _ = exp_L.build(ds, horizon)
    code = build_model_code(spec, TIME_LAG, horizon, values.shape[1])
    (folder / 'candidate_model.py').write_text(code, encoding='utf-8')
    return result_for(folder, values, meta, spec, code, horizon, task['filename'])


def tune_job(folder, task, prepared_result=None):
    import csv
    import tune
    from hpo.config import load_settings
    horizon = task['horizon']
    status(folder, 'Preparing inputs for fixed-architecture tuning')
    if task['action'] == 'tune_sample' and prepared_result is None:
        prepared_result = prepare_default_model(folder, task)
    if prepared_result is not None:
        result = prepared_result
        data_path = folder / 'data.npy'
    else:
        source = HERE / 'runs' / task['source_job']
        if source.resolve().parent != (HERE / 'runs').resolve():
            raise ValueError('Invalid source run directory.')
        shutil.copyfile(source / 'data.npy', folder / 'data.npy')
        shutil.copyfile(source / 'candidate_model.py', folder / 'candidate_model.py')
        shutil.copytree(source / 'metrics', folder / 'metrics')
        result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
        data_path = folder / 'data.npy'
    if result.get('sample_id'):
        from config import SEEDS
        sample = result['sample_id']
        _, _, original = recorded_sample(sample, horizon)
        original_code = original / f'law_s{SEEDS[0]}' / 'candidate_model.py'
        if result['model_code'] != original_code.read_text(encoding='utf-8-sig'):
            raise ValueError('This model does not match the archived L evaluation. Start again from a project example.')
        data_path = folder / (sample + '.npy')
        shutil.copyfile(folder / 'data.npy', data_path)
    settings = load_settings()
    result['tuning_settings'] = settings
    tune.run_folder = lambda name: folder / 'tuning'
    status(folder, 'Running the original 32-configuration tuning protocol with two-seed screening and three-seed confirmation')
    tune.main(['--run','web_run','--model',str(folder / 'candidate_model.py'),
               '--data',str(data_path),'--features',str(folder / 'metrics/meta_features.csv'),
               '--horizons',str(horizon),'--trials',str(settings['search']['trials'])])
    candidates = list((folder / 'tuning').glob('tasks/*/h*/recommended/recommended_model.py'))
    if len(candidates) != 1:
        raise ValueError('Tuning did not produce a unique recommended model. Check the run logs.')
    model_path = candidates[0]
    result['base_result'] = {key: value for key, value in result.items() if key != 'base_result'}
    result.update(model_code=model_path.read_text(encoding='utf-8-sig'), tuned=True,
                  job_id=folder.name, id=folder.name, source='computed_tuning',
                  selection_status=result['selection_status'] + ' · Fixed-architecture tuning complete')
    params_path = model_path.parent / 'recommended_params.txt'
    result['tuning_params_text'] = params_path.read_text(encoding='utf-8-sig') if params_path.exists() else ''
    result['tuning_result'] = extract_tuned_configuration(result['tuning_params_text'])
    result['recommendation_source'] = next(
        line.split(':', 1)[1].strip() for line in result['tuning_params_text'].splitlines()
        if line.startswith('Recommendation source:'))
    with tune.summary_path(folder / 'tuning').open(encoding='utf-8-sig', newline='') as handle:
        row = next(csv.DictReader(handle))
    result['tuning_comparison'] = {
        'historical_l_rmse': float(row['l_original_test_rmse_mean']) if row['l_original_test_rmse_mean'] else None,
        'selected_rmse': float(row['best_test_rmse_mean']) if row['status'] == 'COMPLETE' else None,
        'rmse_gain': float(row['test_rmse_improvement_vs_l_original'])
        if row['test_rmse_improvement_vs_l_original'] not in ('', 'N/A') else None,
    }
    result.pop('evaluation', None)
    result.pop('evaluation_error', None)
    try:
        from evaluation import tuned_evaluation
        result['evaluation'] = tuned_evaluation(folder, model_path, result)
    except (ValueError, KeyError, OSError) as exc:
        result['evaluation_error'] = f'Model code is ready, but evaluation results could not be loaded: {exc}'
    return result


def prepare_default_model(folder, task):
    from common import load_meta
    from config import SEEDS
    horizon = task['horizon']
    status(folder, 'Preparing default-parameter evaluation')
    if task.get('sample_id'):
        sample = task['sample_id']
        ds, spec, original = recorded_sample(sample, horizon)
        shutil.copyfile(ds['npy_path'], folder / 'data.npy')
        (folder / 'metrics').mkdir()
        shutil.copyfile(ds['meta_csv'], folder / 'metrics/meta_features.csv')
        values = read_data(folder / 'data.npy')
        meta = load_meta(ds['meta_csv'])
        code = (original / f'law_s{SEEDS[0]}' / 'candidate_model.py').read_text(encoding='utf-8-sig')
        result = result_for(folder, values, meta, spec, code, horizon, sample, 'recorded_then_exported')
        result.update(sample_id=sample, selection_status='Archived L architecture')
        (folder / 'candidate_model.py').write_text(code, encoding='utf-8')
    else:
        source = HERE / 'runs' / task['source_job']
        if source.resolve().parent != (HERE / 'runs').resolve():
            raise ValueError('Invalid source run directory.')
        result = json.loads((source / 'result.json').read_text(encoding='utf-8'))
        if result.get('tuned'):
            raise ValueError('Tuned models use evaluation results from the original tuning workflow.')
        for filename in ('data.npy', 'candidate_model.py'):
            shutil.copyfile(source / filename, folder / filename)
        shutil.copytree(source / 'metrics', folder / 'metrics')
    result.update(id=folder.name, job_id=folder.name)
    result.pop('evaluation', None)
    result.pop('evaluation_error', None)
    return result


def evaluate_job(folder, task, prepared_result=None):
    import train_one
    from config import SEEDS, EPOCHS, EARLY_STOP_PATIENCE
    from evaluation import collect_evaluation, recorded_l_sources
    result = prepared_result if prepared_result is not None else prepare_default_model(folder, task)
    horizon = task['horizon']
    if result.get('sample_id'):
        sources = recorded_l_sources(folder, result)
    else:
        sources = []
        cfg, _, _ = train_one._load_pipeline()
        for i, seed in enumerate(SEEDS, 1):
            status(folder, f'Training and evaluation: seed {seed}({i}/{len(SEEDS)}), up to {EPOCHS} epochs, patience {EARLY_STOP_PATIENCE}')
            metrics = train_one.train_test(result['spec'], folder / 'data.npy', horizon, seed,
                                          result['channels'], folder / 'evaluation', f'seed_{seed}')
            if metrics is None:
                raise ValueError('Training or testing failed. Check the run logs.')
            test = folder / 'evaluation' / f'seed_{seed}' / 'test'
            sources.append((seed, test / cfg.TEST_METRICS_FILE, test / cfg.FINAL_PREDS_FILE, test / cfg.FINAL_TARGETS_FILE))
    result['evaluation'] = collect_evaluation(folder, sources, result, 'L default parameters · Three-seed evaluation', SEEDS)
    result['selection_status'] += ' · Default evaluation complete'
    return result


def model_job(folder, task):
    """Chain unchanged research stages in one run, without further browser actions."""
    from result_files import save_result_files
    result = prepare_default_model(folder, task) if task.get('sample_id') else analyze(folder, task)
    # Preserve the assembled model even when subsequent training is interrupted.
    save_result_files(folder, result)
    save(folder / 'result.json', result)
    if task['tuning']:
        return tune_job(folder, task, prepared_result=result)
    return evaluate_job(folder, task, prepared_result=result)


def extract_tuned_configuration(params_text):
    """Display-only reading of the original recommended parameter description."""
    marker = 'Recommended parameters:\n'
    if marker not in params_text:
        raise ValueError('Recommended parameters are missing; the model input window cannot be confirmed.')
    config, _ = json.JSONDecoder().raw_decode(params_text.split(marker, 1)[1].lstrip())
    return config


def main():
    sys.stdout.reconfigure(line_buffering=True)
    folder = Path(sys.argv[1]).resolve()
    if folder.parent != (HERE / 'runs').resolve() or not folder.is_dir():
        raise ValueError('Worker output must remain within 4html/runs.')
    sys.path[:0] = [str(ROOT / '2search/compar_exp'), str(ROOT / '1experiment'), str(ROOT / '3par')]
    try:
        task = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        if task['action'] == 'model':
            result = model_job(folder, task)
        elif task['action'] == 'analyze':
            result = analyze(folder, task)
        elif task['action'] in ('evaluate', 'evaluate_sample'):
            result = evaluate_job(folder, task)
        elif task['action'] in ('export', 'export_sample'):
            result = prepare_default_model(folder, task)
        else:
            result = tune_job(folder, task)
        from result_files import save_result_files
        status(folder, 'Saving run artifacts, model dimensions, and the architecture diagram')
        save_result_files(folder, result)
        save(folder / 'result.json', result)
        status(folder, 'Run completed', status='completed')
    except Exception as exc:
        status(folder, 'Computation did not complete', status='failed', error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
