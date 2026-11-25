"""Update the Word table's band labels and scope text while retaining all statistics."""
import argparse
import json
import math
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from connect_common import ROOT, REFERENCE_PATH, default_document, read_csv, read_document_rules, reference_labels, load_label_units

APPENDIX_TITLE = 'Fixed reference bands and original rule ranges'
FEATURE_NAMES = ['Distributional shape', 'Trend strength', 'Non-stationarity', 'Periodicity',
                 'Temporal dependence', 'Volatility', 'Cross-variable coherence']


def set_text(cell, value, size=8):
    cell.text = value
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.line_spacing = 1
        for run in paragraph.runs:
            run.font.name = 'Arial'
            run.font.size = Pt(size)
            run.font.color.rgb = RGBColor(0, 0, 0)


def style_table(table, widths, size=8):
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for row_index, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if row_index == 0:
            trpr.append(OxmlElement('w:tblHeader'))
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
            set_text(cell, cell.text, size)
            pr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for side in ('top', 'left', 'bottom', 'right'):
                element = OxmlElement('w:' + side)
                for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    element.set(qn('w:' + key), value)
                borders.append(element)
            pr.append(borders)
            margin = OxmlElement('w:tcMar')
            for side, value in [('top', '12'), ('bottom', '12'), ('left', '35'), ('right', '35')]:
                element = OxmlElement('w:' + side)
                element.set(qn('w:w'), value)
                element.set(qn('w:type'), 'dxa')
                margin.append(element)
            pr.append(margin)
            if row_index == 0:
                shading = OxmlElement('w:shd')
                shading.set(qn('w:fill'), 'E7E6E6')
                pr.append(shading)
                for run in cell.paragraphs[0].runs:
                    run.bold = True


def numeric_ranges(conditions):
    bounds = {}
    for part in conditions.split('|'):
        feature, direction, value = part.split(':')
        value = float(value)
        low, high = bounds.get(feature, (-math.inf, math.inf))
        bounds[feature] = (max(low, value), high) if direction == '>' else (low, min(high, value))
    return {feature: (f'{low:.6f} < x\n≤ {high:.6f}' if math.isfinite(low) and math.isfinite(high)
                      else f'x > {low:.6f}' if math.isfinite(low) else f'x ≤ {high:.6f}')
            for feature, (low, high) in bounds.items()}


def update_document(source, output):
    reference = json.loads(REFERENCE_PATH.read_text(encoding='utf-8'))
    features = list(reference['features'])
    fixed = {r['rule_id']: r for r in read_csv(ROOT / '04_pooled_all_4734/outputs/pooled_all_fixed_rules.csv')}
    label_units = load_label_units(reference=reference)
    before = read_document_rules(source)
    document = Document(source)
    # Replace only our own appendix on repeated runs.
    body = document._element.body
    removing = False
    for child in list(body):
        if child.tag == qn('w:p') and ''.join(child.itertext()).startswith(APPENDIX_TITLE):
            removing = True
        if removing and child.tag != qn('w:sectPr'):
            body.remove(child)
    table = next(t for t in document.tables if len(t.columns) == 17)
    for row, record in zip(table.rows[1:], before):
        labels = reference_labels(fixed[record['rule_id']]['conditions_str_exact'], reference, units=label_units)
        for cell, feature in zip(row.cells[3:10], features):
            set_text(cell, labels.get(feature, ''))
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    document.paragraphs[0].text = 'Table 2 Pooled meta feature to model motif affinity laws'
    document.paragraphs[0].style = 'Title'
    document.paragraphs[0].paragraph_format.space_before = Pt(0)
    document.paragraphs[0].paragraph_format.space_after = Pt(2)
    for run in document.paragraphs[0].runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.size = Pt(12)
    for paragraph in document.paragraphs:
        if paragraph.text.startswith('Rules were derived'):
            paragraph.text = ('Rules were derived from 4,734 pooled candidate models (263 datasets; 1,578 per AI). '
                              f'All {len(before)} original rules, thresholds and results are unchanged. '
                              'Each Low, Mid or High label classifies the median feature value across unique datasets satisfying the entire rule. '
                              'It describes the typical level, not the full numerical range. '
                              'Use the original numerical ranges on the next page to determine membership. '
                              'Blank cells are unconstrained; row conditions apply jointly. '
                              'Coherence medians use covered multivariate datasets only; univariate placeholder zeros are excluded. '
                              'Higher stationarity scores indicate greater non-stationarity.')
        if paragraph.style.name != 'Title':
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.line_spacing = 1
            for run in paragraph.runs:
                run.font.name = 'Arial'
                run.font.size = Pt(7.5)
    heading = document.add_paragraph(APPENDIX_TITLE)
    heading.paragraph_format.page_break_before = True
    heading.paragraph_format.space_after = Pt(4)
    for run in heading.runs:
        run.bold = True
        run.font.size = Pt(12)
    note = document.add_paragraph('Reference bands classify the median feature value of unique datasets covered by each complete rule. Cut points are the 1/3 and 2/3 quantiles of 263 unique datasets, rounded to six decimals and held fixed; coherence uses multivariate datasets only. Labels describe typical levels and do not replace original rule thresholds.')
    note.paragraph_format.space_after = Pt(4)
    for run in note.runs:
        run.font.size = Pt(8)
    bands = document.add_table(rows=1, cols=4)
    for cell, text in zip(bands.rows[0].cells, ['Feature', 'Low', 'Mid', 'High']):
        cell.text = text
    for feature, name in zip(features, FEATURE_NAMES):
        q1, q2 = reference['features'][feature]['cuts']
        for cell, text in zip(bands.add_row().cells, [name, f'x ≤ {q1:.6f}', f'{q1:.6f} < x ≤ {q2:.6f}', f'x > {q2:.6f}']):
            cell.text = text
    style_table(bands, [2.4, 2.5, 3.4, 2.5])
    caption = document.add_paragraph('Original rule ranges')
    caption.paragraph_format.space_before = Pt(5)
    caption.paragraph_format.space_after = Pt(3)
    caption.paragraph_format.keep_with_next = True
    for run in caption.runs:
        run.bold = True
        run.font.size = Pt(10)
    ranges = document.add_table(rows=1, cols=8)
    for cell, text in zip(ranges.rows[0].cells, ['Law', *FEATURE_NAMES]):
        cell.text = text
    for record in before:
        values = numeric_ranges(fixed[record['rule_id']]['conditions_str_exact'])
        for cell, text in zip(ranges.add_row().cells, [record['rule_id'], *[values.get(f, '') for f in features]]):
            cell.text = text
    style_table(ranges, [0.44, *[1.48] * 7], size=7.5)
    footnote = document.add_paragraph('Only the displayed numbers are rounded to six decimals. Calculations retain the full precision in conditions_str_exact in the accompanying CSV files. Lower bounds are open and upper bounds are closed. A blank cell imposes no constraint; these numerical conditions, not the reference labels, determine membership.')
    footnote.paragraph_format.space_before = Pt(3)
    footnote.paragraph_format.space_after = Pt(0)
    for run in footnote.runs:
        run.font.size = Pt(7.5)
    document.save(output)
    after = read_document_rules(output)
    assert [r['rule_id'] for r in before] == [r['rule_id'] for r in after]
    for old, new in zip(before, after):
        assert old['docx_cells'][:3] == new['docx_cells'][:3]
        assert old['docx_cells'][10:] == new['docx_cells'][10:]
    print(f'Updated labels for {len(after)} original rules; all rule IDs and result marks preserved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    source = args.docx or default_document(ROOT)
    update_document(source, args.output or source)
