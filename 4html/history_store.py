"""Read the local run archive without importing the training environment."""
import json
import re
from datetime import datetime, timezone

JOB_ID = re.compile(r'[a-f0-9]{32}')


def read_record(path):
    if not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(record, dict):
        raise ValueError('Invalid saved run record.')
    return record


def iso_time(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def list_history(runs, active_jobs):
    groups, skipped = {}, 0
    if not runs.is_dir():
        return {'datasets': [], 'total_runs': 0, 'skipped': 0}
    for folder in runs.iterdir():
        if not folder.is_dir() or folder.is_symlink() or not JOB_ID.fullmatch(folder.name):
            continue
        try:
            task = read_record(folder / 'request.json')
            saved = read_record(folder / 'status.json')
            result = read_record(folder / 'result.json')
            name = result.get('sample_id') or task.get('sample_id') or result.get('name') or task.get('filename')
            if not name and JOB_ID.fullmatch(str(task.get('source_job', ''))):
                parent = read_record(runs / task['source_job'] / 'result.json')
                name = parent.get('sample_id') or parent.get('name')
            name = str(name or 'Unknown dataset')
            # Uploads and project examples of the same named dataset share a group.
            dataset = re.sub(r'\.(npy|csv)$', '', name, flags=re.I)
            status = saved.get('status', 'interrupted')
            entry = active_jobs.get(folder.name)
            if entry:
                code = entry['process'].poll()
                if entry.get('cancelled'):
                    status = 'cancelled'
                elif code is None:
                    status = 'running'
                else:
                    status = 'completed' if code == 0 and result else 'failed'
            elif status in ('running', 'queued'):
                status = 'interrupted'
            timestamp_file = next((folder / filename for filename in ('status.json', 'result.json', 'request.json')
                                   if (folder / filename).is_file()), folder)
            evaluation = result.get('evaluation') or {}
            record = {
                'id': folder.name, 'name': name, 'status': status,
                'created_at': iso_time(folder / 'request.json') if (folder / 'request.json').is_file() else iso_time(folder),
                'updated_at': iso_time(timestamp_file), 'horizon': result.get('horizon', task.get('horizon')),
                'rows': result.get('rows'), 'channels': result.get('channels'),
                'tuned': bool(result.get('tuned', task.get('tuning', str(task.get('action', '')).startswith('tune')))),
                'has_result': status == 'completed' and bool(result.get('spec') and result.get('model_code')),
                'has_evaluation': bool(evaluation), 'seed_count': evaluation.get('seed_count'),
                'metrics': {key: evaluation.get('metrics', {}).get(key, {}).get('mean') for key in ('rmse', 'mae', 'wmape')},
            }
            group = groups.setdefault(dataset, {'id': dataset, 'name': dataset, 'runs': []})
            group['runs'].append(record)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            skipped += 1
    for group in groups.values():
        group['runs'].sort(key=lambda row: row['updated_at'], reverse=True)
        group['updated_at'] = group['runs'][0]['updated_at']
    datasets = sorted(groups.values(), key=lambda group: group['updated_at'], reverse=True)
    return {'datasets': datasets, 'total_runs': sum(len(group['runs']) for group in datasets), 'skipped': skipped}
