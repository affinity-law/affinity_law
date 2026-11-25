"""Read original test artifacts for display; never train or select a model here."""
import json
import math
import shutil
import statistics
from pathlib import Path
from output_language import english

METRICS = ('rmse', 'mae', 'wmape')
EVALUATION_SEED = 2026


def local_path(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError('Evaluation files must remain within this run directory.')
    return path


def collect_evaluation(folder, sources, result, label, expected_seeds):
    """Keep metrics from complete original tests, with the same seed's curves."""
    import numpy as np
    entries = []
    if [s[0] for s in sources] != list(expected_seeds):
        raise ValueError('Evaluation seeds are incomplete; results cannot be displayed.')
    shape = None
    for seed, metrics_path, pred_path, target_path in sources:
        record = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
        space = record.get('evaluation_space')
        space = space.get('data_space') if isinstance(space, dict) else space
        if space != 'original':
            raise ValueError('Evaluation results are not in the original data scale.')
        if record.get('seed', seed) != seed:
            raise ValueError('The metrics do not match the random seed.')
        lag = record.get('time_lag', record.get('task', {}).get('time_lag'))
        if lag is not None and lag != result.get('tuning_result', {}).get('time_lag', 12):
            raise ValueError('The evaluation window does not match the recommended model.')
        values = {key: record['overall_metrics'][key] for key in METRICS}
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('Evaluation metrics are missing or contain nonfinite values.')
        pred = np.load(pred_path, mmap_mode='r', allow_pickle=False)
        target = np.load(target_path, mmap_mode='r', allow_pickle=False)
        if (pred.ndim != 2 or pred.shape != target.shape or not len(pred)
                or pred.shape[1] != result['channels'] or pred.dtype.kind not in 'fiu'
                or target.dtype.kind not in 'fiu' or not np.isfinite(pred).all() or not np.isfinite(target).all()
                or (shape is not None and pred.shape != shape)):
            raise ValueError('Observed and predicted arrays have invalid shapes or values.')
        shape = pred.shape
        entries.append({'seed': seed, 'metrics': values,
                        'predictions': pred_path.relative_to(folder).as_posix(),
                        'targets': target_path.relative_to(folder).as_posix()})
    if not entries or shape[0] > result['rows']:
        raise ValueError('Complete test-set evaluation results are not available.')
    summary = {'label': english(label), 'split': 'test', 'evaluation_space': 'original',
               'aggregation': 'overlap_average', 'seed_count': len(entries),
               'point_count': shape[0], 'channels': shape[1],
               'row_offset': result['rows'] - shape[0],
               'time_lag': result.get('tuning_result', {}).get('time_lag', 12),
               'horizon': result['horizon'], 'wmape_unit': 'ratio',
               'per_seed': [{'seed': e['seed'], 'metrics': e['metrics']} for e in entries],
               # Keep the legacy field name for saved-result compatibility; single runs use the exact value.
               'metrics': {key: {'mean': entries[0]['metrics'][key] if len(entries) == 1 else statistics.mean(e['metrics'][key] for e in entries),
                                 'std': statistics.stdev(e['metrics'][key] for e in entries) if len(entries) > 1 else None}
                           for key in METRICS}}
    manifest = folder / 'evaluation.json'
    manifest.write_text(json.dumps({'summary': summary, 'entries': entries}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return summary


def tuned_evaluation(folder, model_path, result):
    """Read metrics for the exact model that 3par exports."""
    from config import SEEDS

    task_folder = model_path.parent.parent
    lines = result['tuning_params_text'].splitlines()
    source = next((line.split(':', 1)[1].strip() for line in lines if line.startswith('Recommendation source:')), None)
    if source == 'L_DEFAULT':
        return collect_evaluation(folder, recorded_l_sources(folder, result), result,
                                  'Recommended model: archived L default · Three-seed evaluation', SEEDS)
    if source == 'R_DEFAULT':
        directories = [(EVALUATION_SEED, task_folder / 'test/default')]
        label = 'Recommended model: default parameters · Single-run evaluation'
    elif source == 'TUNED_TOP3':
        directories = [(seed, task_folder / 'best' / f'seed_{seed}' / 'test') for seed in SEEDS]
        label = 'Recommended model: tuned parameters · Three-seed evaluation'
    else:
        raise ValueError('The recommendation source cannot be identified. Results from other configurations are not shown.')
    sources = [(seed, path / 'metrics.json', path / 'averaged_predictions.npy', path / 'averaged_targets.npy')
               for seed, path in directories]
    return collect_evaluation(folder, sources, result, label, [seed for seed, _ in directories])


def recorded_l_sources(folder, result):
    from config import SEEDS
    from worker import recorded_sample

    _, _, original = recorded_sample(result['sample_id'], result['horizon'])
    original_code = original / f'law_s{SEEDS[0]}' / 'candidate_model.py'
    if result['model_code'] != original_code.read_text(encoding='utf-8-sig'):
        raise ValueError('This model does not match the archived L evaluation. Start again from a project example.')
    filenames = ('test_metrics.json', 'averaged_final_predictions.npy', 'averaged_final_targets.npy')
    sources = []
    for seed in SEEDS:
        source_run = original / f'law_s{seed}'
        target_run = folder / 'evaluation' / f'seed_{seed}'
        target = target_run / 'test'
        target.mkdir(parents=True, exist_ok=True)
        paths = [target / name for name in filenames]
        for path in paths:
            shutil.copyfile(source_run / 'test' / path.name, path)
        (target_run / 'train').mkdir(exist_ok=True)
        for name in ('best_model_weights.pth', 'train_standardization_stats.npz'):
            shutil.copyfile(source_run / 'train' / name, target_run / 'train' / name)
        sources.append((seed, *paths))
    return sources


def read_curve(folder, seed, channel, start=0):
    """Page through consecutive original points; metrics always use the full test set."""
    import numpy as np
    manifest = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
    summary = manifest['summary']
    if not 0 <= channel < summary['channels'] or not 0 <= start < summary['point_count']:
        raise ValueError('The selected variable or plot range is outside the evaluation data.')
    entry = next((entry for entry in manifest['entries'] if entry['seed'] == seed), None)
    if entry is None:
        raise ValueError('This evaluation has no results for the selected seed.')
    stop = min(start + 600, summary['point_count'])
    pred = np.load(local_path(folder, entry['predictions']), mmap_mode='r', allow_pickle=False)
    target = np.load(local_path(folder, entry['targets']), mmap_mode='r', allow_pickle=False)
    return {'seed': seed, 'channel': channel, 'start': start, 'stop': stop,
            'total': summary['point_count'], 'indices': list(range(summary['row_offset'] + start + 1, summary['row_offset'] + stop + 1)),
            'predicted': pred[start:stop, channel].tolist(), 'actual': target[start:stop, channel].tolist()}


def curve_png(curve, summary):
    """Standalone scientific figure from the exact points shown in the browser."""
    from io import BytesIO
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(11, 4.8), dpi=180, layout='constrained')
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.plot(curve['indices'], curve['actual'], color='#268c87', linewidth=1.4, label='Ground truth')
    ax.plot(curve['indices'], curve['predicted'], color='#ee8833', linewidth=1.4, linestyle='--', label='Prediction')
    ax.set(title=f"Test predictions | Variable {curve['channel'] + 1} | Seed {curve['seed']}",
           xlabel='Original data row (1-based)', ylabel='Value (original scale)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    figure.suptitle(f"Input {summary['time_lag']} / Horizon {summary['horizon']} | Overlap-averaged forecasts | Test points {curve['start'] + 1}-{curve['stop']} of {curve['total']}", fontsize=9)
    output = BytesIO()
    figure.savefig(output, format='png')
    return output.getvalue()
