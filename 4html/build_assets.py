"""Export read-only research fixtures using the project's original selector/assembler."""
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / '2search/compar_exp'))
import numpy as np
import config
import select_arch
from common import load_meta
from rule_revisions import ENGLISH_TABLES
from result_files import topology_svg, model_io
from worker import recorded_sample, rule_hits


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def main():
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    law_path = config.LAW_TABLE
    law = select_arch.load_law()
    pooled = rows(config.POOLED_LAW_TABLE)
    rules = []
    for rule in json.loads(law.to_json(orient='records', double_precision=15)):
        if 'source_csv_row' in rule:
            evidence = pooled[int(rule['source_csv_row']) - 2]
            rule = {**evidence, **rule}
        rules.append(rule)
    records = []
    examples = [('0102_ETTh2_9580_7', 'ETTh2 · Electricity Transformer', 'Multivariate / Energy'),
                ('0230_capitalbike_count_9649_1', 'Capital Bikeshare · Bike Demand', 'Univariate / Transport'),
                ('0925_MaunaLoa_CO2_2284_1', 'Mauna Loa · CO2 Concentration', 'Univariate / Environment')]
    for name, title, category in examples:
        ds, _, _ = recorded_sample(name, 3)
        values = np.load(ds['npy_path'], allow_pickle=False)
        if values.ndim == 1:
            values = values[:, None]
        meta = load_meta(ds['meta_csv'])
        indices = np.linspace(0, len(values) - 1, min(240, len(values)), dtype=int)
        variants = {}
        for horizon in (3, 12):
            _, spec, original = recorded_sample(name, horizon)
            code = (original / f'law_s{config.SEEDS[0]}' / 'candidate_model.py').read_text(encoding='utf-8-sig')
            ast.parse(code)
            filename = f'{name}_h{horizon}.py'
            (assets / filename).write_text(code, encoding='utf-8')
            hits = rule_hits(law, meta, horizon)
            variants[str(horizon)] = {'spec': spec, 'hits': hits, 'model_file': filename, 'model_code': code}
            shape = {'spec':spec, 'horizon':horizon, 'channels':int(values.shape[1])}
            variants[str(horizon)].update(topology_svg=topology_svg(shape), model_io=model_io(shape))
        records.append({'id': name, 'name': title, 'category': category, 'rows': len(values),
                        'channels': int(values.shape[1]), 'meta': meta, 'indices': indices.tolist(),
                        'preview': values[indices, :min(3, values.shape[1])].tolist(),
                        'variants': variants, 'source': 'recorded'})
    bundle = {'examples': records, 'rules': rules,
              'overview': json.loads(select_arch.load_overview().to_json(orient='records')),
              'library': json.loads((ROOT / '1experiment/module_set/module_library_candidates.json').read_text(encoding='utf-8'))['modules'],
              'law_sha256': hashlib.sha256(law_path.read_bytes()).hexdigest(),
              'protocol': json.loads((ROOT / '3par/settings.json').read_text(encoding='utf-8'))}
    bundle['equivalent_law_sha256'] = [old for old, current in ENGLISH_TABLES.items()
                                       if current == bundle['law_sha256']]
    (assets / 'research-data.js').write_text('window.RESEARCH_DATA = ' + json.dumps(bundle, ensure_ascii=False, allow_nan=False) + ';\n', encoding='utf-8')
    print(f'Exported {len(records)} examples, {len(bundle["rules"])} original rules, 6 original model files.')


if __name__ == '__main__':
    main()
