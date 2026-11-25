"""File handling, input checks and rule-family selection for pooled discovery and fixed-threshold validation."""
import csv
import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from statistics import median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT.parent
AI_TYPES = ('deepseek', 'chatgpt', 'gemini')
REFERENCE_PATH = Path(__file__).resolve().with_name('label_reference.json')


def manifest_path(path):
    """Record repository inputs relative to its root and external inputs by absolute path."""
    path = Path(path).resolve()
    project = SEARCH_ROOT.parent
    return (path.relative_to(project) if path.is_relative_to(project) else path).as_posix()


def load_label_units(input_dir=None, reference=None):
    """Verify feature consistency and use one row per dataset and horizon to avoid reweighting by candidate or provider."""
    import pandas as pd

    reference = reference or json.loads(REFERENCE_PATH.read_text(encoding='utf-8'))
    input_dir = Path(input_dir) if input_dir is not None else SEARCH_ROOT / 'law_search/inputs'
    units, dataset_features = {}, {}
    for ai in AI_TYPES:
        for horizon in (3, 12):
            for mode in ('univariate', 'multivariate'):
                frame = pd.read_csv(input_dir / f'{ai}_12_{horizon}_{mode}.csv', encoding='utf-8-sig')
                # Match the original raw CSV -> unit CSV -> readback path exactly. A plain float()
                # parse can differ by one ULP at a closed rule boundary and change membership.
                frame = pd.read_csv(io.StringIO(frame[['Dataset', *reference['features']]].to_csv(index=False)))
                for row in frame.to_dict('records'):
                    values = {feature: float(row[feature]) for feature in reference['features']}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError(f"{row['Dataset']} has invalid feature values")
                    previous = dataset_features.setdefault(row['Dataset'], values)
                    if any(abs(values[f] - previous[f]) > 1e-12 for f in values):
                        raise ValueError(f"{row['Dataset']} has inconsistent features across candidates, providers or horizons")
                    units[row['Dataset'], horizon] = dict(values, Dataset=row['Dataset'],
                                                          config=int(horizon == 12), data_type=int(mode == 'multivariate'))
    return list(units.values())


def reference_profile(conditions_str, reference=None, *, units=None):
    """Select on the entire rule, deduplicate datasets and use median features for display labels."""
    reference = reference or json.loads(REFERENCE_PATH.read_text(encoding='utf-8'))
    units = load_label_units(reference=reference) if units is None else units
    conditions = [(f, direction, float(value)) for part in conditions_str.split('|')
                  for f, direction, value in [part.split(':')]]
    selected = {}
    for row in units:
        if all((row[f] > value if direction == '>' else row[f] <= value) for f, direction, value in conditions):
            selected[row['Dataset']] = row
    profile = {}
    for feature in dict.fromkeys(f for f, _, _ in conditions if f in reference['features']):
        values = [row[feature] for row in selected.values()
                  if feature != 'multivariate_structure_score' or row['data_type'] == 1]
        if not values:
            raise ValueError(f'{feature} No covered datasets available to assign Low/Mid/High labels')
        value = median(values)
        q1, q2 = reference['features'][feature]['cuts']
        profile[feature] = {'label': 'Low' if value <= q1 else 'Mid' if value <= q2 else 'High',
                            'median': value, 'datasets': len(values)}
    return profile


def reference_labels(conditions_str, reference=None, *, units=None):
    return {feature: item['label'] for feature, item in
            reference_profile(conditions_str, reference, units=units).items()}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows and not fields:
        raise ValueError('Specify column names explicitly for an empty table')
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_inputs(input_dir):
    """Check all 12 inputs and each provider's top three before starting computation."""
    input_dir = Path(input_dir)
    counts, ai_counts, modes, hashes = Counter(), Counter(), {}, {}
    for horizon in (3, 12):
        for mode in ('univariate', 'multivariate'):
            for ai in AI_TYPES:
                path = input_dir / f'{ai}_12_{horizon}_{mode}.csv'
                if not path.is_file():
                    raise FileNotFoundError(f'Missing candidate input:{path}')
                rows = read_csv(path)
                if not rows:
                    raise ValueError(f'Empty candidate input:{path}')
                hashes[path.name] = sha256(path)
                for row in rows:
                    if int(row['TIME_LAG']) != 12 or int(row['PRE_LEN']) != horizon:
                        raise ValueError(f'{path.name} has a forecast configuration inconsistent with its filename')
                    dataset = row['Dataset']
                    if dataset in modes and modes[dataset] != mode:
                        raise ValueError(f'Dataset appears as both univariate and multivariate:{dataset}')
                    modes[dataset] = mode
                    counts[dataset, horizon, ai] += 1
                    ai_counts[ai] += 1
    for dataset in modes:
        for horizon in (3, 12):
            for ai in AI_TYPES:
                if counts[dataset, horizon, ai] != 3:
                    raise ValueError(f'{dataset} h{horizon} {ai} expected 3 candidates, found {counts[dataset, horizon, ai]} records')
    return {'datasets': len(modes), 'dataset_config_units': len(modes) * 2,
            'candidate_rows': sum(ai_counts.values()), 'rows_by_ai': dict(ai_counts),
            'input_sha256': hashes}


def default_document(folder):
    """Allow a renamed reference document. Use --docx to choose among multiple files."""
    candidates = [p for p in Path(folder).glob('*.docx') if not p.name.startswith('~$')]
    if len(candidates) != 1:
        raise ValueError('Expected one Word document in connect. Specify --docx if the directory has zero or multiple documents.')
    return candidates[0]


def read_document_rules(docx):
    """Read the table with P-prefixed rule IDs independently of titles, positions and table counts."""
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    matches = []
    for table in root.findall('.//w:tbl', ns):
        records = []
        for row in table.findall('w:tr', ns)[1:]:
            cells = [''.join(t.text or '' for t in cell.findall('.//w:t', ns)).strip()
                     for cell in row.findall('w:tc', ns)]
            if len(cells) >= 14 and re.fullmatch(r'P\d+', cells[2]):
                records.append({'rule_id': cells[2], 'table_order': len(records) + 1,
                                'docx_cells': cells})
        if records:
            matches.append(records)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one table with P-prefixed rule IDs; found {len(matches)} tables')
    records = matches[0]
    if len({r['rule_id'] for r in records}) != len(records):
        raise ValueError('Duplicate rule IDs in document')
    return records


def select_validation_family(fixed_rules, family_ids, displayed):
    """The BH correction family includes all original tests independently of which rows appear in the document."""
    lookup = {r['rule_id']: r for r in fixed_rules}
    if len(lookup) != len(fixed_rules) or len(set(family_ids)) != len(family_ids):
        raise ValueError('Duplicate rule IDs')
    if set(family_ids) - lookup.keys():
        raise ValueError('Stage 04 output lacks original test-family rules. Run stage 04 first')
    if {r['rule_id'] for r in displayed} - set(family_ids):
        raise ValueError('Document contains rules outside the test family. Check rule_family.json')
    order = {r['rule_id']: r['table_order'] for r in displayed}
    return [dict(lookup[rid], table_order=order.get(rid, ''), in_document=int(rid in order))
            for rid in sorted(family_ids)]
