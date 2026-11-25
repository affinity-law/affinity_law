"""Save readable outputs and a topology diagram inside each isolated run folder."""
import csv
import html
import json
import textwrap
from pathlib import Path
from output_language import english, english_report, matched_rules

INVERSES = {'Z-score':'Inverse Z-score', 'RevIN':'Inverse RevIN',
            'Differencing':'Inverse Differencing', 'Series Decomposition':'Decomposition Recomposition'}


def model_io(result):
    lag = result.get('tuning_result', {}).get('time_lag', 12)
    return {'input_steps':lag, 'output_steps':result['horizon'], 'variables':result['channels'],
            'input_shape':['batch',lag,result['channels']], 'output_shape':['batch',result['horizon'],result['channels']]}


def topology_svg(result):
    spec, io = result['spec'], model_io(result)
    e = spec['E']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1320 340" role="img" aria-labelledby="topology-title">',
             f'<title id="topology-title">Final architecture: input {io["input_steps"]} steps, output {io["output_steps"]} steps, {io["variables"]} variables</title>',
             '<defs><marker id="topology-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#a0acba"/></marker></defs>',
             '<rect width="1320" height="340" rx="16" fill="#fbfcfd"/>',
             '<g font-family="Segoe UI,Microsoft YaHei,sans-serif">']
    def text(x, y, value, size=14, color='#394657'):
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" fill="{color}" font-size="{size}">{html.escape(str(value))}</text>')
    def node(x, y, w, h, title, labels, color='#ffffff'):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{color}" stroke="#dde3ea"/>')
        text(x+w/2, y+25, title, 15)
        lines = [line for label in labels for line in textwrap.wrap(str(label), width=max(12, int(w/7.5)), break_long_words=False)]
        for i, line in enumerate(lines):
            text(x+w/2, y+49+i*18, line, 12, '#758292')
    def arrow(d):
        parts.append(f'<path d="{d}" fill="none" stroke="#a0acba" stroke-width="1.7" marker-end="url(#topology-arrow)"/>')
    text(660, 27, f'Input {io["input_steps"]} steps → Output {io["output_steps"]} steps · {io["variables"]} variables', 18)
    node(15,130,120,90,'Input sequence',[f'{io["input_steps"]} steps',f'{io["variables"]} variables'],'#fff3e8')
    node(160,112,165,126,'A · Preprocessing',spec['A'] or ['Identity'])
    parts.append('<rect x="350" y="53" width="425" height="245" rx="13" fill="#f3f6f9" stroke="#d4dde7" stroke-dasharray="5 4"/>')
    text(562, 80, 'B / E · ' + e, 14)
    arrow('M135 175H160'); arrow('M325 175H350')
    if e == 'Single Block':
        node(450,126,210,98,'B · Backbone',spec['B'])
        arrow('M350 175H450'); arrow('M660 175H775')
    elif e == 'Sequential Composition':
        node(370,126,175,98,'B1',[spec['B'][0]])
        node(575,126,175,98,'B2',[spec['B'][1]])
        arrow('M350 175H370'); arrow('M545 175H575'); arrow('M750 175H775')
    else:
        node(395,96,180,82,'B1 · Main branch' if e == 'Residual Modeling' else 'B1',[spec['B'][0]])
        node(395,201,180,82,'B2 · Correction' if e == 'Residual Modeling' else 'B2',[spec['B'][1]])
        node(632,140,120,76,'Gated fusion' if e == 'Gated Fusion' else 'Add',['softmax(g)' if e == 'Gated Fusion' else 'B1(x) + B2(x)'])
        arrow('M350 175H375V137H395'); arrow('M375 175V242H395')
        arrow('M575 137H605V164H632'); arrow('M575 242H605V194H632'); arrow('M752 175H775')
    node(800,126,140,98,'C · Head',[spec['C']])
    inverse = [INVERSES[a] for a in reversed(spec['A']) if a in INVERSES]
    node(965,103,190,144,'D · Restoration',inverse or ['Identity'])
    node(1180,130,125,90,'Forecast',[f'{io["output_steps"]} steps',f'{io["variables"]} variables'],'#edf7f3')
    arrow('M775 175H800'); arrow('M940 175H965'); arrow('M1155 175H1180')
    text(660, 324, f'Tensor shape: [batch, {io["input_steps"]}, {io["variables"]}] → [batch, {io["output_steps"]}, {io["variables"]}] · Original module connections', 13, '#758292')
    return ''.join(parts) + '</g></svg>'


def save_result_files(folder, result):
    """The run directory holds code, reports, figures and original training subfolders."""
    result['model_io'] = model_io(result)
    result['topology_svg'] = topology_svg(result)
    result['output_folder'] = str(folder.resolve())
    code_name = 'recommended_model.py' if result.get('tuned') else 'candidate_model.py'
    (folder / code_name).write_text(english(result['model_code']), encoding='utf-8')
    (folder / 'model_topology.svg').write_text(result['topology_svg'], encoding='utf-8')
    io = result['model_io']
    files = [code_name, 'model_topology.svg', 'research_report.json']
    if result.get('evaluation'):
        from evaluation import curve_png, read_curve
        evaluation = result['evaluation']
        with (folder / 'evaluation_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['seed_or_statistic','rmse','mae','wmape_ratio'])
            for entry in evaluation['per_seed']:
                writer.writerow([entry['seed'], *(entry['metrics'][key] for key in ('rmse','mae','wmape'))])
            if evaluation['seed_count'] > 1:
                for statistic in ('mean','std'):
                    writer.writerow([statistic, *(evaluation['metrics'][key][statistic] for key in ('rmse','mae','wmape'))])
        first = evaluation['per_seed'][0]['seed']
        figure_name = f'prediction_seed{first}_variable1.png'
        (folder / figure_name).write_bytes(curve_png(read_curve(folder, first, 0), evaluation))
        files.extend(['evaluation_metrics.csv', figure_name])
    (folder / 'result_notes.txt').write_text(
        f"Dataset: {english(result['name'])}\nInput: {io['input_steps']} steps\nForecast: {io['output_steps']} steps\nVariables: {io['variables']}\n"
        f"Model: {code_name}\nComposition: {result['spec']['E']}\n"
        'This folder contains the model, diagram and report for this run.\n'
        'Evaluation CSV files contain RMSE, MAE and WMAPE for each seed, plus mean and standard deviation for multiple seeds.\n'
        'After evaluation, the prediction PNG shows up to 600 consecutive test points for variable 1. Export other variables and ranges from the workspace.\n'
        'Full predictions, targets, weights and normalization statistics are retained in the evaluation or tuning subfolders.\n'
        'WMAPE is stored as a ratio in CSV and JSON: 0.1 means 10%. Metrics use the full test set, independently of the chart range.\n', encoding='utf-8')
    result['output_files'] = files + ['result_notes.txt']
    bundle = json.loads((Path(__file__).parent / 'assets/research-data.js').read_text(encoding='utf-8').removeprefix('window.RESEARCH_DATA = ').strip().removesuffix(';'))
    rules = matched_rules(result, bundle)
    report = english_report(result, rules)
    (folder / 'research_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
