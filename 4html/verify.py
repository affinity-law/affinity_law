"""Meaningful adapter checks; does not train or edit the research project."""
import ast
import hashlib
import json
import re
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import urllib.error
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / '2search/compar_exp'), str(ROOT / '3par')]
import numpy as np
import config
import select_arch
from hpo.models import extract_spec
from worker import read_data, extract_tuned_configuration
import server
import worker
from evaluation import collect_evaluation, read_curve, tuned_evaluation, curve_png
from result_files import model_io, topology_svg, save_result_files
from output_language import english, english_report, matched_rules, public_result
from history_store import list_history


class ResearchContract(unittest.TestCase):
    def test_tuning_keys_are_isolated_and_excluded_from_saved_requests(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(server.JOBS, clear=True), \
                patch.dict(server.os.environ, {'DEEPSEEK_API_KEY': 'parent-test-value'}), \
                patch('server.subprocess.Popen') as spawn:
            for index, key in enumerate(('first-test-credential', 'second-test-credential')):
                folder = Path(temp) / str(index)
                folder.mkdir()
                task = {'action': 'model', 'tuning': True, 'horizon': 3}
                server.start_job(folder, task, api_key=key)
                self.assertEqual(spawn.call_args.kwargs['env']['DEEPSEEK_API_KEY'], key)
                self.assertEqual(json.loads((folder / 'request.json').read_text()), task)
                self.assertNotIn(key, (folder / 'run.log').read_text())
                self.assertNotIn(key, repr(spawn.call_args.args))
                self.assertNotIn(key, repr(server.JOBS))
            self.assertEqual(server.os.environ['DEEPSEEK_API_KEY'], 'parent-test-value')

    def test_tuning_request_requires_a_key_and_official_deepseek_endpoint(self):
        handler = object.__new__(server.Handler)
        handler.headers = {}
        with patch.dict(server.os.environ, {'DEEPSEEK_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'tuning dialog'):
                handler.tuning_api_key()
            handler.headers = {'X-DeepSeek-API-Key': 'job-test-credential'}
            self.assertEqual(handler.tuning_api_key(), 'job-test-credential')
            handler.headers = {'X-DeepSeek-API-Key': 'bad key'}
            with self.assertRaises(ValueError):
                handler.tuning_api_key()
            handler.headers = {'X-DeepSeek-API-Key': 'job-test-credential'}
            settings = {'agent': {'base_url': 'https://unrelated.example'}}
            with patch('pathlib.Path.read_text', return_value=json.dumps(settings)):
                with self.assertRaisesRegex(ValueError, 'official DeepSeek'):
                    handler.tuning_api_key()
        with patch('server.dependencies', return_value=[]), \
                patch.dict(server.os.environ, {'DEEPSEEK_API_KEY': ''}):
            self.assertTrue(server.health()['tuning_ready'])
            self.assertNotIn('api_key', server.health())

    def test_modeling_and_upload_forward_keys_outside_task_data(self):
        from email.message import Message
        from io import BytesIO
        for sample, tuning in ((True, True), (False, True), (True, False), (False, False)):
            with self.subTest(sample=sample, tuning=tuning):
                handler = object.__new__(server.Handler)
                handler.headers = Message()
                handler.headers['Host'] = '127.0.0.1:8040'
                handler.headers['Origin'] = 'http://127.0.0.1:8040'
                handler.headers['X-DeepSeek-API-Key'] = 'request-test-credential'
                handler.server = SimpleNamespace(server_port=8040)
                if sample:
                    handler.path = '/api/modeling'
                    body = json.dumps({'sample_id': '0102_ETTh2_9580_7', 'horizon': 3, 'tuning': tuning}).encode()
                else:
                    handler.path = '/api/jobs'
                    handler.headers['Content-Type'] = 'multipart/form-data; boundary=fixture'
                    fields = {'horizon': '3', 'run_model': 'true', 'tuning': str(tuning).lower()}
                    text = ''.join('--fixture\r\nContent-Disposition: form-data; name="' + name + '"\r\n\r\n' + value + '\r\n' for name, value in fields.items())
                    text += '--fixture\r\nContent-Disposition: form-data; name="file"; filename="series.csv"\r\nContent-Type: text/csv\r\n\r\nx\r\n1\r\n2\r\n--fixture--\r\n'
                    body = text.encode()
                handler.headers['Content-Length'] = str(len(body))
                handler.rfile = BytesIO(body)
                with tempfile.TemporaryDirectory() as temp, patch.object(handler, 'create_folder', return_value=Path(temp)), \
                        patch.object(handler, 'reply') as reply, patch('server.start_job') as start, \
                        patch('server.dependencies', return_value=[]):
                    handler.do_POST()
                    self.assertEqual(reply.call_args.args[0], 202)
                    self.assertEqual(start.call_args.kwargs['api_key'], 'request-test-credential' if tuning else None)
                    self.assertNotIn('request-test-credential', json.dumps(start.call_args.args[1]))

    def test_default_evaluation_uses_current_l_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            result = {'rows':100, 'channels':1, 'horizon':3, 'spec':{}, 'selection_status':'Assembled'}
            calls = []
            def train(spec, data, horizon, seed, channels, output, name):
                calls.append(seed)
                path = output / name / 'test'
                path.mkdir(parents=True)
                (path/'metrics.json').write_text(json.dumps({'seed':seed, 'evaluation_space':'original',
                    'overall_metrics':{'rmse':float(seed - 2024),'mae':1.5,'wmape':.1}}))
                np.save(path/'pred.npy', np.full((5,1),3.))
                np.save(path/'target.npy', np.ones((5,1)))
                return {'rmse':2.}
            cfg = SimpleNamespace(TEST_METRICS_FILE='metrics.json', FINAL_PREDS_FILE='pred.npy', FINAL_TARGETS_FILE='target.npy')
            fake = SimpleNamespace(train_test=train, _load_pipeline=lambda:(cfg,None,None))
            with patch.dict(sys.modules, {'train_one':fake}):
                evaluation = worker.evaluate_job(folder, {'horizon':3}, prepared_result=result)['evaluation']
            self.assertEqual(calls,config.SEEDS)
            self.assertEqual(evaluation['seed_count'],3)
            self.assertEqual(evaluation['metrics']['rmse'],{'mean':3.,'std':1.})
            self.assertEqual(read_curve(folder,2026,0)['predicted'],[3.]*5)
            self.assertEqual(read_curve(folder,2028,0)['predicted'],[3.]*5)

    def test_one_click_runs_the_selected_original_stages_in_one_folder(self):
        for sample in (False, True):
            for tuning in (False, True):
                with self.subTest(sample=sample, tuning=tuning), tempfile.TemporaryDirectory() as temp:
                    folder = Path(temp)
                    prepared = {'spec':{'A':['Differencing'],'B':['TCN Block'],'C':'MLP Head','E':'Single Block'}}
                    task = {'action':'model', 'horizon':3, 'tuning':tuning}
                    if sample: task['sample_id'] = '0925_MaunaLoa_CO2_2284_1'
                    def finish(output, request, prepared_result):
                        self.assertEqual(output, folder)
                        self.assertIs(prepared_result, prepared)
                        self.assertEqual(json.loads((folder/'result.json').read_text(encoding='utf-8')), prepared)
                        return {'finished': True}
                    with patch.object(worker, 'analyze', return_value=prepared) as analyze, \
                         patch.object(worker, 'prepare_default_model', return_value=prepared) as prepare, \
                         patch.object(worker, 'evaluate_job', side_effect=finish) as evaluate, \
                         patch.object(worker, 'tune_job', side_effect=finish) as tune, \
                         patch('result_files.save_result_files'):
                        self.assertEqual(worker.model_job(folder, task), {'finished':True})
                        self.assertEqual(prepare.call_count, int(sample))
                        self.assertEqual(analyze.call_count, int(not sample))
                        self.assertEqual(tune.call_count, int(tuning))
                        self.assertEqual(evaluate.call_count, int(not tuning))

    def test_final_model_window_and_isolated_saved_artifacts(self):
        import xml.etree.ElementTree as ET
        bundle = json.loads((HERE / 'assets/research-data.js').read_text(encoding='utf-8').removeprefix('window.RESEARCH_DATA = ').strip().removesuffix(';'))
        example = bundle['examples'][0]
        result = {**example, **example['variants']['3'], 'horizon':3, 'tuned':True, 'tuning_result':{'time_lag':64}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / 'first', root / 'second'
            first.mkdir(); second.mkdir()
            for folder, lag in ((first,64),(second,32)):
                result['tuning_result']['time_lag'] = lag
                save_result_files(folder, result)
                report = json.loads((folder / 'research_report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['model_io']['input_steps'], lag)
                self.assertEqual(report['model_io']['output_steps'], 3)
                self.assertEqual((folder / 'recommended_model.py').read_text(encoding='utf-8'), english(result['model_code']))
                self.assertTrue((folder / 'result_notes.txt').exists())
                for filename in ('recommended_model.py', 'model_topology.svg', 'research_report.json', 'result_notes.txt'):
                    self.assertIsNone(re.search(r'[\u4e00-\u9fff]', (folder / filename).read_text(encoding='utf-8')))
                text = ''.join(ET.fromstring((folder / 'model_topology.svg').read_text(encoding='utf-8')).itertext())
                self.assertIn(f'Input {lag} steps', text)
            self.assertEqual(json.loads((first/'research_report.json').read_text(encoding='utf-8'))['model_io']['input_steps'],64)
        for mode in ('Single Block','Sequential Composition','Parallel Additive Branches','Residual Modeling','Gated Fusion'):
            result['spec'] = {'A':['RevIN'], 'B':['TCN Block'] if mode=='Single Block' else ['TCN Block','Linear Block'], 'C':'Linear Head','E':mode}
            text = ''.join(ET.fromstring(topology_svg(result)).itertext())
            self.assertIn(mode,text)
            self.assertIn('Inverse RevIN',text)
            if mode=='Gated Fusion': self.assertIn('softmax(g)',text)
            if mode=='Residual Modeling': self.assertIn('Correction',text)

    def test_history_groups_datasets_recovers_restarts_and_skips_corrupt_records(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / 'runs'
            self.assertEqual(list_history(runs, {})['datasets'], [])
            def record(job_id, task, status, result=None):
                folder = runs / job_id
                folder.mkdir(parents=True)
                server.write_json(folder / 'request.json', task)
                server.write_json(folder / 'status.json', {'status': status})
                if result is not None:
                    server.write_json(folder / 'result.json', result)
                return folder
            first = record('a'*32, {'filename':'dataset.npy','horizon':3}, 'completed',
                           {'name':'dataset.npy','horizon':3,'spec':{'E':'Single Block'},'model_code':'pass',
                            'evaluation':{'metrics':{'rmse':{'mean':2.5}},'seed_count':3}})
            record('b'*32, {'sample_id':'dataset','horizon':12}, 'completed',
                   {'sample_id':'dataset','spec':{'E':'Single Block'},'model_code':'pass'})
            record('c'*32, {'source_job':'a'*32,'horizon':3}, 'running')
            broken = record('d'*32, {}, 'failed')
            (broken / 'result.json').write_text('{broken', encoding='utf-8')
            archive = list_history(runs, {})
            self.assertEqual((archive['total_runs'], archive['skipped'], len(archive['datasets'])), (3,1,1))
            records = {row['id']:row for row in archive['datasets'][0]['runs']}
            self.assertEqual(records['c'*32]['status'], 'interrupted')
            self.assertFalse(records['c'*32]['has_result'])
            self.assertEqual(records['a'*32]['metrics']['rmse'], 2.5)
            self.assertNotIn('model_code', records['a'*32])
            active = {'c'*32:{'process':SimpleNamespace(poll=lambda:None), 'cancelled':False}}
            self.assertEqual(next(row for row in list_history(runs, active)['datasets'][0]['runs'] if row['id']=='c'*32)['status'], 'running')
            with patch.object(server, 'HERE', Path(temp)), patch.object(server, 'JOBS', {}):
                self.assertEqual(server.get_job('a'*32)['status'], 'completed')
                self.assertEqual(server.get_job('c'*32)['status'], 'interrupted')
                with self.assertRaises(ValueError): server.get_job('../outside')

    def test_english_exports_preserve_models_and_original_history(self):
        import model_assembler
        bundle = json.loads((HERE / 'assets/research-data.js').read_text(encoding='utf-8').removeprefix('window.RESEARCH_DATA = ').strip().removesuffix(';'))
        class WithoutDocstrings(ast.NodeTransformer):
            def visit_Expr(self, node):
                return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else node
        for source in model_assembler._MODULE_SOURCES.values():
            self.assertIsNone(re.search(r'[\u4e00-\u9fff]', english(source)))
            self.assertEqual(ast.dump(WithoutDocstrings().visit(ast.parse(source))),
                             ast.dump(WithoutDocstrings().visit(ast.parse(english(source)))))
        for example in bundle['examples']:
            for horizon, variant in example['variants'].items():
                result = {**example, **variant, 'horizon':int(horizon), 'selection_status':'Archived L architecture'}
                before = json.dumps(result, ensure_ascii=False)
                output = public_result(result)
                self.assertEqual(before, json.dumps(result, ensure_ascii=False))
                self.assertIsNone(re.search(r'[\u4e00-\u9fff]', output['model_code'] + output['topology_svg']))
                self.assertEqual(ast.dump(WithoutDocstrings().visit(ast.parse(result['model_code']))),
                                 ast.dump(WithoutDocstrings().visit(ast.parse(output['model_code']))))
                report = english_report(result, bundle['rules'])
                self.assertIsNone(re.search(r'[\u4e00-\u9fff]', json.dumps(report, ensure_ascii=False)))
                self.assertEqual(output['spec'], result['spec'])

    def test_evaluation_keeps_full_metrics_and_aligned_consecutive_curves(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            targets = np.arange(1406, dtype=float).reshape(703, 2) + 1
            sources = []
            for seed, offset in ((2026, 1), (2027, 2), (2028, 3)):
                path = folder / str(seed)
                path.mkdir()
                predictions = targets + offset
                metrics = {'rmse':offset, 'mae':offset, 'wmape':float(offset*targets.size/targets.sum())}
                (path / 'metrics.json').write_text(json.dumps({'evaluation_space':'original','overall_metrics':metrics}))
                np.save(path / 'pred.npy', predictions)
                np.save(path / 'target.npy', targets)
                sources.append((seed, path / 'metrics.json', path / 'pred.npy', path / 'target.npy'))
            result = {'rows':1000, 'channels':2, 'horizon':3}
            summary = collect_evaluation(folder, sources, result, 'test', [2026, 2027, 2028])
            self.assertEqual(summary['metrics']['rmse'], {'mean':2,'std':1})
            first = read_curve(folder, 2027, 1)
            last = read_curve(folder, 2027, 1, 600)
            self.assertEqual(first['indices'], list(range(298, 898)))
            self.assertEqual(last['indices'], list(range(898, 1001)))
            np.testing.assert_array_equal(last['actual'], targets[600:, 1])
            np.testing.assert_array_equal(last['predicted'], targets[600:, 1] + 2)
            self.assertEqual(curve_png(last, summary)[:8], b'\x89PNG\r\n\x1a\n')
            for seed, channel, start in ((9999,0,0),(2026,2,0),(2026,0,-1),(2026,0,703)):
                with self.assertRaises(ValueError):
                    read_curve(folder, seed, channel, start)
            with self.assertRaises(ValueError):
                collect_evaluation(folder, sources[:2], result, 'incomplete', [2026,2027,2028])

    def test_tuned_fallback_uses_default_model_test_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            task = folder / 'tuning/tasks/example/h3'
            model = task / 'recommended/recommended_model.py'
            paths = [task / 'test/default'] + [task / 'best' / f'seed_{s}' / 'test' for s in (2026,2027,2028)]
            for i, path in enumerate(paths):
                path.mkdir(parents=True)
                (path / 'metrics.json').write_text(json.dumps({'evaluation_space':'original', 'overall_metrics':{'rmse':i+1,'mae':i+1,'wmape':.01*(i+1)}}))
                np.save(path / 'averaged_predictions.npy', np.full((5,1), i+1, dtype=float))
                np.save(path / 'averaged_targets.npy', np.ones((5,1)))
            result = {'rows':100, 'channels':1, 'horizon':3, 'tuning_params_text':'Recommendation source: R_DEFAULT\n'}
            summary = tuned_evaluation(folder, model, result)
            self.assertEqual(summary['seed_count'], 1)
            self.assertEqual(summary['metrics']['rmse'], {'mean':1,'std':None})
            self.assertEqual(read_curve(folder, 2026, 0)['predicted'], [1]*5)
            result['tuning_params_text'] = 'Recommendation source: TUNED_TOP3\n'
            summary = tuned_evaluation(folder, model, result)
            self.assertEqual(summary['seed_count'], 3)
            self.assertEqual(summary['metrics']['rmse'], {'mean':3,'std':1})
            self.assertEqual(read_curve(folder, 2026, 0)['predicted'], [2]*5)
            self.assertEqual(read_curve(folder, 2028, 0)['predicted'], [4]*5)

    def test_export_matches_original_selector_and_assembler(self):
        bundle = (HERE / 'assets/research-data.js').read_text(encoding='utf-8')
        data = json.loads(bundle.removeprefix('window.RESEARCH_DATA = ').strip().removesuffix(';'))
        original_law = config.LAW_TABLE
        self.assertEqual(data['law_sha256'], hashlib.sha256(original_law.read_bytes()).hexdigest())
        law = select_arch.load_law()
        self.assertEqual(len(data['rules']), len(law))
        for example in data['examples']:
            for h in (3, 12):
                variant = example['variants'][str(h)]
                _, spec, archived = worker.recorded_sample(example['id'], h)
                self.assertEqual(variant['spec'], spec)
                code = (archived / f'law_s{config.SEEDS[0]}' / 'candidate_model.py').read_text(encoding='utf-8-sig')
                self.assertEqual(extract_spec(code), spec)
                self.assertEqual(variant['model_code'], code)
                self.assertEqual((HERE / 'assets' / variant['model_file']).read_text(encoding='utf-8'), code)
                ast.parse(code)
                expected_hits = [int(i) for i, row in law.iterrows()
                                 if row['grade'] in ('S', 'A')
                                 and select_arch.rule_triggers(select_arch.rule_conditions(row), example['meta'], h)
                                 and not select_arch._in_corner(row, example['meta'])]
                self.assertEqual(variant['hits'], expected_hits)

    def test_current_l_fallback_uses_matching_archived_three_seed_metrics(self):
        sample = '0925_MaunaLoa_CO2_2284_1'
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            result = worker.prepare_default_model(folder, {'sample_id':sample, 'horizon':3})
            result['tuning_params_text'] = 'Recommendation source: L_DEFAULT\n'
            model = folder / 'tuning/tasks/example/h3/recommended/recommended_model.py'
            expected = worker.evaluate_job(folder, {'horizon':3}, prepared_result=result)['evaluation']
            actual = tuned_evaluation(folder, model, result)
            self.assertEqual(actual['seed_count'], 3)
            self.assertEqual(actual['metrics'], expected['metrics'])
            self.assertEqual(read_curve(folder, 2028, 0)['seed'], 2028)

    def test_csv_preserves_numeric_rows_and_optional_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.csv'
            for content in ('1,2\n3,4\n', 'a,b\n1,2\n3,4\n',
                            'time,a,b\n2026-01-01,1,2\n2026-01-02,3,4\n',
                            '2026-01-01,1,2\n2026-01-02,3,4\n'):
                path.write_text(content, encoding='utf-8')
                np.testing.assert_array_equal(read_data(path), [[1,2],[3,4]])

    def test_invalid_data_is_rejected_without_silent_cleaning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.csv'
            for content in ('a,b\n1,\n3,4\n', 'a,b\n1,NaN\n', '1,2\n3\n',
                            'date,x\n2026-01-02,2\n2026-01-01,1\n',
                            'date,x\n2026-01-01,1\n2026-01-01,2\n'):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_data(path)
            path = Path(temp) / 'input.npy'
            for values in (np.array([{'not':'numeric'}], dtype=object), np.array([1,np.inf]),
                           np.zeros((2,3,4)), np.array([1+2j])):
                np.save(path, values)
                with self.assertRaises(ValueError):
                    read_data(path)
            np.save(path, np.arange(10))
            self.assertEqual(read_data(path).shape, (10,1))

    def test_tuned_window_and_parameters_come_from_original_output(self):
        config = {'time_lag':64, 'lr':0.003, 'batch_size':32}
        text = 'Dataset: example\nRecommended parameters:\n' + json.dumps(config, indent=2) + '\n\nOther results'
        self.assertEqual(extract_tuned_configuration(text), config)
        with self.assertRaises(ValueError):
            extract_tuned_configuration('missing output')

    def test_server_does_not_expose_project_or_runs(self):
        http = server.LocalServer(('127.0.0.1', 0), server.Handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{http.server_port}'
        try:
            with urllib.request.urlopen(base + '/api/health') as response:
                h = json.load(response)
                self.assertEqual(h['application'], 'tslab-project-bridge')
                self.assertNotIn('api_key', h)
            with self.assertRaises(OSError):
                server.LocalServer(('127.0.0.1', http.server_port), server.Handler)
            with server.bind_local_server(http.server_port) as fallback:
                self.assertNotEqual(fallback.server_port, http.server_port)
                self.assertEqual(fallback.server_address[0], '127.0.0.1')
            for path in ('/../README.md', '/%2e%2e/requirements.txt', '/worker.py', '/runs/secret.json'):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + path)
                self.assertEqual(caught.exception.code, 404)
            request = urllib.request.Request(base + '/api/tuning', data=b'{}', headers={'Origin':'https://unrelated.example'})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            request = urllib.request.Request(base + '/api/health', headers={'Host':'unrelated.example'})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            self.assertEqual(caught.exception.code, 403)
            request = urllib.request.Request(base + '/api/health', headers={'Origin':base})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(response.status, 200)
        finally:
            http.shutdown()
            http.server_close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
