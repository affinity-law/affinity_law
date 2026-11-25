"""English presentation copies; original computation records remain untouched."""
import copy
import json
import re
from pathlib import Path

_asset = (Path(__file__).parent / 'assets/output-english.js').read_text(encoding='utf-8')
_messages = json.loads(_asset.split('window.OUTPUT_ENGLISH = ', 1)[1].strip().removesuffix(';'))
_phrases = re.compile('|'.join(re.escape(key) for key in sorted(_messages, key=len, reverse=True)))


def english(text):
    return _phrases.sub(lambda match: _messages[match[0]], text).replace('；', '; ').replace('。', '. ').replace('：', ': ')


def english_value(value):
    if isinstance(value, dict):
        return {key: english_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [english_value(item) for item in value]
    return english(value) if isinstance(value, str) else value


def public_result(result):
    """Localize saved results on read, including diagrams produced by old versions."""
    from result_files import topology_svg
    output = copy.deepcopy(result)
    output.pop('base_result', None)
    output['model_code'] = english(output['model_code'])
    output['selection_status'] = english(output.get('selection_status', ''))
    if output.get('evaluation'):
        output['evaluation']['label'] = english(output['evaluation'].get('label', 'Test evaluation'))
    if output.get('evaluation_error'):
        output['evaluation_error'] = english(output['evaluation_error'])
    output['topology_svg'] = topology_svg(output)
    return output


def matched_rules(result, bundle):
    digest = result.get('law_table_sha256')
    if digest != bundle['law_sha256'] and digest not in bundle.get('equivalent_law_sha256', []):
        return []
    return [bundle['rules'][index] for index in result.get('hits', [])]


def english_report(result, rules):
    excluded = {'model_code', 'topology_svg', 'base_result', 'preview', 'indices', 'variants'}
    report = {key: value for key, value in result.items() if key not in excluded}
    report.update(application='affinity law', output_language='en', matched_rules=rules)
    # Reports use a portable location; the UI retains the exact absolute path for copying.
    if result.get('job_id'):
        report['output_folder'] = 'runs/' + result['job_id']
    return english_value(report)
