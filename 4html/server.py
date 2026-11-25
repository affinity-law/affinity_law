"""Local HTTP connection layer. Original research files are only imported/read.

Run with the project's Python environment: python -B 4html/server.py
Only 4html static assets and this application's API are exposed.
"""
import argparse
import errno
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import uuid
import webbrowser
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from history_store import list_history, read_record
from output_language import public_result, english_report, matched_rules

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAX_UPLOAD = 50 * 1024 * 1024
JOBS = {}
LOCK = threading.RLock()
COMPUTE_DEPENDENCIES = ('numpy', 'pandas', 'scipy', 'statsmodels', 'sklearn', 'torch', 'fvcore', 'matplotlib')


def dependencies(names):
    return [name for name in names if importlib.util.find_spec(name) is None]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def health():
    missing = dependencies(COMPUTE_DEPENDENCIES)
    tuning_missing = dependencies(COMPUTE_DEPENDENCIES + ('openai',))
    return {'application': 'tslab-project-bridge', 'compute_ready': not missing,
            'tuning_ready': not tuning_missing,
            'message': 'The local computation service is ready.' if not missing else 'Missing Python dependencies: ' + ', '.join(missing) + '. Start the app in the project environment with requirements.txt installed.',
            'tuning_message': 'Select Automatic Tuning to enter your DeepSeek API key.' if not tuning_missing else
            'Tuning is not ready: missing ' + ', '.join(tuning_missing)}


def start_job(folder, task, api_key=None):
    """Each job uses a separate process to isolate the original global configs."""
    write_json(folder / 'request.json', task)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1',
           'RESULT_ROOT': str(folder / 'pipeline'), 'TSF_RESULT_ROOT': str(folder / 'selection')}
    if api_key is not None:
        env['DEEPSEEK_API_KEY'] = api_key
    log = (folder / 'run.log').open('wb')
    try:
        process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(HERE / 'worker.py'), str(folder)],
                                   cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    finally:
        log.close()
    with LOCK:
        JOBS[folder.name] = {'process': process, 'folder': folder, 'cancelled': False}
    return {'id': folder.name, 'status': 'queued'}


def get_job(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Invalid run ID.')
    with LOCK:
        entry = JOBS.get(job_id)
    if not entry:
        folder = HERE / 'runs' / job_id
        status_file = folder / 'status.json'
        if status_file.is_file():
            saved = read_record(status_file)
            result = {**saved, 'id': job_id}
            if saved.get('status') == 'completed' and (folder / 'result.json').is_file():
                result['result'] = read_record(folder / 'result.json')
                result['result']['output_folder'] = str(folder.resolve())
            elif saved.get('status') in ('running', 'queued'):
                result.update(status='interrupted', message='This run stopped before completion. Saved files are preserved.')
            return result
        raise ValueError('Run not found. If the service was restarted, check the runs folder for saved results.')
    folder, process = entry['folder'], entry['process']
    result = {'id': job_id, 'status': 'running', 'message': 'Preparing the local computation environment'}
    status_file = folder / 'status.json'
    if status_file.exists():
        result.update(json.loads(status_file.read_text(encoding='utf-8')))
    code = process.poll()
    if entry['cancelled']:
        result.update(status='cancelled', message='The run was cancelled. Existing artifacts remain in its output folder.')
    elif code is not None:
        if code != 0:
            result.update(status='failed', error=result.get('error') or 'The computation process exited. Expand the logs to check the runtime.')
        elif (folder / 'result.json').exists():
            result.update(status='completed', result=json.loads((folder / 'result.json').read_text(encoding='utf-8')))
        else:
            result.update(status='failed', error='Computation did not return complete results.')
    elif result.get('status') == 'completed':
        result.update(status='running', message='Saving the model and report')
    log_path = folder / 'run.log'
    if log_path.exists():
        with log_path.open('rb') as log:
            log.seek(max(0, log_path.stat().st_size - 8000))
            result['log'] = log.read().decode('utf-8', errors='replace')
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = 'AffinityLaw/1.0'

    def log_message(self, format, *args):
        # Do not log posted data or personal file contents.
        pass

    def origin_allowed(self):
        host = self.headers.get('Host', '')
        if host not in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'):
            return False
        origin = self.headers.get('Origin')
        if not origin:
            return True
        return origin == 'http://' + host

    def reply(self, code, payload, content_type='application/json; charset=utf-8'):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8') if isinstance(payload, (dict, list)) else payload
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    def do_OPTIONS(self):
        self.reply(204 if self.origin_allowed() else 403, b'')

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if not self.origin_allowed():
            return self.reply(403, {'error': 'Access the workspace using the local startup URL.'})
        try:
            if path == '/api/health':
                return self.reply(200, health())
            if path == '/api/history':
                with LOCK:
                    active = dict(JOBS)
                return self.reply(200, list_history(HERE / 'runs', active))
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/report', path)
            if match:
                job = get_job(match[1])
                if job['status'] != 'completed' or not job.get('result'):
                    raise ValueError('This run has no completed result.')
                result = job['result']
                bundle = json.loads((HERE / 'assets/research-data.js').read_text(encoding='utf-8').removeprefix('window.RESEARCH_DATA = ').strip().removesuffix(';'))
                rules = matched_rules(result, bundle)
                return self.reply(200, english_report(result, rules))
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/curve(\.png)?', path)
            if match:
                job = get_job(match[1])
                if job['status'] != 'completed' or not job['result'].get('evaluation'):
                    raise ValueError('Complete test results are not available for this model.')
                from evaluation import read_curve, curve_png
                query = parse_qs(urlsplit(self.path).query)
                curve = read_curve(HERE / 'runs' / match[1], int(query.get('seed', ['2026'])[0]),
                                   int(query.get('channel', ['0'])[0]), int(query.get('start', ['0'])[0]))
                if match[2]:
                    png = curve_png(curve, job['result']['evaluation'])
                    filename = f"prediction_seed{curve['seed']}_variable{curve['channel'] + 1}_{curve['start'] + 1}-{curve['stop']}.png"
                    (HERE / 'runs' / match[1] / filename).write_bytes(png)
                    return self.reply(200, png, 'image/png')
                return self.reply(200, curve)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})', path)
            if match:
                job = get_job(match[1])
                if job.get('result'):
                    job['result'] = public_result(job['result'])
                return self.reply(200, job)
            if path.startswith('/api/'):
                return self.reply(404, {'error': 'API endpoint not found.'})
            if path in ('/', '/index.html'):
                file = HERE / 'index.html'
            elif path in ('/app.js', '/performance.js', '/styles.css'):
                file = HERE / path[1:]
            elif re.fullmatch(r'/assets/[A-Za-z0-9_.-]+\.(js|py)', path):
                file = HERE / path[1:]
            else:
                return self.reply(404, b'Not found', 'text/plain; charset=utf-8')
            if not file.is_file():
                return self.reply(404, b'Not found', 'text/plain; charset=utf-8')
            content_type = {'.js': 'text/javascript', '.css': 'text/css', '.py': 'text/plain', '.html': 'text/html'}.get(file.suffix, 'application/octet-stream')
            return self.reply(200, file.read_bytes(), content_type + '; charset=utf-8')
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            return self.reply(400, {'error': str(exc)})

    def body(self, limit):
        try:
            length = int(self.headers.get('Content-Length', '-1'))
        except ValueError:
            raise ValueError('Invalid request length.') from None
        if length < 0 or length > limit:
            raise ValueError('The request has no valid length or exceeds 50 MB.')
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError('The upload is incomplete.')
        return raw

    def create_folder(self):
        folder = HERE / 'runs' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        return folder

    def tuning_api_key(self):
        """Read a per-job credential without putting it in saved request data."""
        key = self.headers.get('X-DeepSeek-API-Key', os.environ.get('DEEPSEEK_API_KEY', '')).strip()
        if not key:
            raise ValueError('Enter your DeepSeek API key in the tuning dialog.')
        if len(key) > 512 or any(not 33 <= ord(char) <= 126 for char in key):
            raise ValueError('Enter a valid DeepSeek API key without spaces or line breaks.')
        settings = json.loads((HERE.parent / '3par/settings.json').read_text(encoding='utf-8'))
        endpoint = urlsplit(settings['agent']['base_url'])
        if (endpoint.scheme != 'https' or endpoint.hostname != 'api.deepseek.com'
                or endpoint.port not in (None, 443) or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment):
            raise ValueError('Workbench tuning supports the official DeepSeek API endpoint only.')
        return key

    def do_POST(self):
        if not self.origin_allowed():
            return self.reply(403, {'error': 'Access the workspace using the local startup URL.'})
        path = urlsplit(self.path).path
        try:
            if path == '/api/jobs':
                h = health()
                if not h['compute_ready']:
                    return self.reply(503, {'error': h['message']})
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    raise ValueError('Uploads must use multipart/form-data.')
                raw = self.body(MAX_UPLOAD + 65536)
                message = BytesParser(policy=default).parsebytes(('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode() + raw)
                fields = {}
                for part in message.iter_parts():
                    name = part.get_param('name', header='content-disposition')
                    if name in fields:
                        raise ValueError('Provide exactly one file and one forecast horizon per request.')
                    fields[name] = part
                file = fields.get('file')
                if file is None:
                    raise ValueError('Provide a time series file.')
                filename = (file.get_filename() or '').replace('\\', '/').split('/')[-1]
                suffix = Path(filename).suffix.lower()
                payload = file.get_payload(decode=True)
                if suffix not in ('.npy', '.csv') or not payload or len(payload) > MAX_UPLOAD:
                    raise ValueError('Upload a nonempty NPY or CSV file of at most 50 MB.')
                horizon = int(fields['horizon'].get_content().strip()) if 'horizon' in fields else 3
                if horizon not in (3, 12):
                    raise ValueError('The forecast horizon must be 3 or 12.')
                run_model = fields['run_model'].get_content().strip() if 'run_model' in fields else 'false'
                tuning = fields['tuning'].get_content().strip() if 'tuning' in fields else 'false'
                if run_model not in ('true', 'false') or tuning not in ('true', 'false'):
                    raise ValueError('Invalid modeling settings.')
                if run_model == 'true' and tuning == 'true' and not h['tuning_ready']:
                    return self.reply(503, {'error':h['tuning_message']})
                api_key = self.tuning_api_key() if run_model == 'true' and tuning == 'true' else None
                folder = self.create_folder()
                (folder / ('upload' + suffix)).write_bytes(payload)
                return self.reply(202, start_job(folder, {'action':'model' if run_model == 'true' else 'analyze',
                    'tuning':tuning == 'true', 'filename':filename, 'suffix':suffix, 'horizon':horizon}, api_key=api_key))
            if path == '/api/modeling':
                task = json.loads(self.body(4096))
                if not isinstance(task, dict) or type(task.get('tuning')) is not bool:
                    raise ValueError('Select a tuning mode.')
                if type(task.get('horizon')) is not int or task['horizon'] not in (3, 12):
                    raise ValueError('The forecast horizon must be 3 or 12.')
                if task.get('sample_id') not in ('0102_ETTh2_9580_7', '0230_capitalbike_count_9649_1', '0925_MaunaLoa_CO2_2284_1'):
                    raise ValueError('Unknown example dataset.')
                h = health()
                if not h['tuning_ready' if task['tuning'] else 'compute_ready']:
                    return self.reply(503, {'error':h['tuning_message' if task['tuning'] else 'message']})
                api_key = self.tuning_api_key() if task['tuning'] else None
                return self.reply(202, start_job(self.create_folder(), {'action':'model', 'sample_id':task['sample_id'],
                    'horizon':task['horizon'], 'tuning':task['tuning']}, api_key=api_key))
            if path in ('/api/tuning', '/api/evaluation', '/api/export'):
                evaluating = path == '/api/evaluation'
                exporting = path == '/api/export'
                action = 'evaluate' if evaluating else 'export' if exporting else 'tune'
                h = health()
                if not h['compute_ready' if evaluating or exporting else 'tuning_ready']:
                    return self.reply(503, {'error': h['message' if evaluating or exporting else 'tuning_message']})
                api_key = self.tuning_api_key() if action == 'tune' else None
                task = json.loads(self.body(4096))
                horizon = task.get('horizon')
                if type(horizon) is not int or horizon not in (3, 12):
                    raise ValueError('The forecast horizon must be 3 or 12.')
                if task.get('sample_id'):
                    allowed = ('0102_ETTh2_9580_7', '0230_capitalbike_count_9649_1', '0925_MaunaLoa_CO2_2284_1')
                    if task['sample_id'] not in allowed:
                        raise ValueError('Unknown example dataset.')
                    request = {'action':action + '_sample', 'sample_id':task['sample_id'], 'horizon':horizon}
                else:
                    source = get_job(task.get('job_id', ''))
                    if source['status'] != 'completed' or source['result']['horizon'] != horizon:
                        raise ValueError('Complete data processing for this forecast horizon first.')
                    if (evaluating or exporting) and source['result'].get('tuned'):
                        raise ValueError('Use the existing test results for the tuned model.')
                    request = {'action':action, 'source_job':source['id'], 'horizon':horizon}
                return self.reply(202, start_job(self.create_folder(), request, api_key=api_key))
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/cancel', path)
            if match:
                with LOCK:
                    entry = JOBS.get(match[1])
                    if not entry:
                        raise ValueError('Run not found.')
                    if entry['process'].poll() is None:
                        entry['cancelled'] = True
                        entry['process'].terminate()
                return self.reply(200, {'id':match[1], 'message':'Cancellation request processed.'})
            return self.reply(404, {'error':'API endpoint not found.'})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return self.reply(400, {'error':str(exc)})


class LocalServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let a second interpreter steal an active listener.
    allow_reuse_address = os.name != 'nt'

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind_local_server(port):
    try:
        return LocalServer(('127.0.0.1', port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE and getattr(exc, 'winerror', None) != 10048:
            raise
        return LocalServer(('127.0.0.1', 0), Handler)


def main(open_browser=False):
    parser = argparse.ArgumentParser(description='Serve the HTML frontend and call the unchanged research project.')
    parser.add_argument('--port', type=int, default=8040)
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser automatically.')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be in the range 1-65535.')
    try:
        server = bind_local_server(args.port)
    except OSError as exc:
        parser.exit(1, f'Unable to start the local service: {exc}\n')
    if server.server_port != args.port:
        print(f'Port {args.port} is in use; switched to available port {server.server_port}.', flush=True)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'affinity law (local access): {url}', flush=True)
    print(health()['message'], flush=True)
    print('Keep this terminal open. Press Ctrl+C to stop. Run files are saved under 4html/runs.', flush=True)
    if open_browser and not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for entry in JOBS.values():
            if entry['process'].poll() is None:
                entry['process'].terminate()


if __name__ == '__main__':
    main()
