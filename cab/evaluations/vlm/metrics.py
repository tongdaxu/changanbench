"""Metrics for paired original/compressed VLM outputs (run with --help for CLI options)."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cab.evaluations.vlm.benchmark_tasks import (TASKS, SEATS, CABIN_EXACT, CABIN_TEXT, fields_for,
                             prepare_output, partial_output, validate_field)


def parse_output(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('```') and value.endswith('```'):
            value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        value = json.loads(value)
    if not isinstance(value, (dict, list)):
        raise ValueError('output must be a JSON object or array')
    return value


def read_records(path):
    records = {}
    with open(path, encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['id']
            if not isinstance(key, str) or not key or key in records:
                raise ValueError(f'{path}:{line_no}: id must be a unique nonempty string')
            records[key] = row.get('output')
    return records


MISSING = object()


def field(obj, name):
    if name == '$':
        return obj
    for part in name.split('.'):
        if not isinstance(obj, dict) or part not in obj:
            return MISSING
        obj = obj[part]
    return obj


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def valid(obj, args):
    if obj is MISSING:
        return False
    for name in args.exact_fields:
        if field(obj, name) is MISSING:
            return False
    for name in args.text_fields:
        if not isinstance(field(obj, name), str):
            return False
    for name in args.set_fields:
        value = field(obj, name)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return False
    for name in args.detection_fields:
        value = field(obj, name)
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get('target_type'), str):
                return False
            box = item.get('bbox')
            if not isinstance(box, list) or len(box) != 4:
                return False
            if not all(type(x) in (int, float) and math.isfinite(x) for x in box):
                return False
            if box[2] <= box[0] or box[3] <= box[1]:
                return False
    try:
        canonical(obj)
    except (ValueError, TypeError):
        return False
    return True


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def detection_score(a, b, threshold):
    # Maximum-cardinality bipartite matching; prefer high-IoU edges locally.
    edges = [[j for j in sorted(range(len(b)), key=lambda j: iou(x['bbox'], b[j]['bbox']), reverse=True)
              if x['target_type'] == b[j]['target_type'] and iou(x['bbox'], b[j]['bbox']) >= threshold] for x in a]
    matched = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(a)):
        augment(i, set())
    overlaps = [iou(a[i]['bbox'], b[j]['bbox']) for j, i in matched.items()]
    return (2*len(matched)/(len(a)+len(b)) if a or b else 1.0), overlaps


def average(values):
    return mean(values) if values else None


def inspect_output(raw, args, exact_fields, text_fields):
    """Keep global conformance separate from independently usable metric fields."""
    result = {'parsed': False, 'valid': False, 'values': {}, 'field_valid': {}, 'field_errors': {}}
    names = exact_fields + text_fields + args.set_fields + args.detection_fields
    try:
        obj = parse_output(raw)
        result['parsed'] = True
    except (ValueError, TypeError, IndexError) as exc:
        result['error'] = str(exc)
        result['field_valid'] = {name: False for name in names}
        result['field_errors'] = {name: 'output is not parseable JSON object/array' for name in names}
        return result
    try:
        whole = prepare_output(obj, args.task) if args.task else obj
        if not valid(whole, args):
            raise ValueError('configured field schema validation failed')
        result['valid'] = True
        result['canonical'] = canonical(whole)
    except (ValueError, TypeError, IndexError) as exc:
        result['error'] = str(exc)
    normalized = partial_output(obj, args.task) if args.task else obj
    result['normalized'] = normalized
    for name in names:
        value = field(normalized, name)
        result['field_valid'][name] = False
        if value is MISSING:
            result['field_errors'][name] = 'field missing'
            continue
        result['values'][name] = value
        try:
            canonical(value)
            validate_field(obj, args.task, name, value)
            if name in text_fields and not isinstance(value, str):
                raise ValueError('text field must be a string')
            if name in args.set_fields and (not isinstance(value, list) or
                    not all(isinstance(x, str) for x in value)):
                raise ValueError('set field must be a string array')
            if name in args.detection_fields:
                # Reuse generic bbox validation, scoped to this field alone.
                scoped = argparse.Namespace(exact_fields=[], text_fields=[], set_fields=[], detection_fields=['$'])
                if not valid(value, scoped):
                    raise ValueError('invalid detection field')
            result['field_valid'][name] = True
        except (ValueError, TypeError, IndexError) as exc:
            result['field_errors'][name] = str(exc)
    return result


def evaluate_records(reference, candidate, args, model=None):
    """Compare ID-keyed outputs in memory without writing files."""
    if not reference:
        raise ValueError("reference is empty")
    cabin_text_fields = [f"attributes.{s}.{f}" for s in SEATS for f in CABIN_TEXT] if args.task == "rec_cabin" else []
    all_text_fields = args.text_fields + cabin_text_fields
    all_exact_fields = args.exact_fields + ([f"attributes.{s}.{f}" for s in SEATS for f in CABIN_EXACT] if args.task == "rec_cabin" else [])
    if all_text_fields and model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model, device=args.device)
    rows, pending = [], {name: [] for name in all_text_fields}
    for key, original in reference.items():
        row = {'id': key, 'candidate_present': key in candidate, 'metrics': {}}
        inspected = []
        for side, raw in [('reference', original), ('candidate', candidate.get(key))]:
            info = inspect_output(raw, args, all_exact_fields, all_text_fields)
            row[side + '_valid'] = info['valid']
            row[side + '_parsed'] = info['parsed']
            row[side + '_field_valid'] = info['field_valid']
            row[side + '_field_errors'] = info['field_errors']
            if 'error' in info:
                row[side + '_error'] = info['error']
            inspected.append(info)
        a, b = inspected
        if a['valid']:
            row['metrics']['json_exact_match'] = int(b['valid'] and a['canonical'] == b['canonical'])
        for name in all_exact_fields:
            if a['field_valid'][name]:
                row['metrics']['exact:' + name] = int(b['field_valid'][name] and
                    canonical(a['values'][name]) == canonical(b['values'][name]))
        for name in args.set_fields:
            if not a['field_valid'][name]:
                continue
            ok = b['field_valid'][name]
            sa, sb = set(a['values'][name]), set(b['values'][name]) if ok else set()
            row['metrics']['set_f1:' + name] = (2*len(sa & sb)/(len(sa)+len(sb)) if sa or sb else 1.0) if ok else 0.0
            row['metrics']['jaccard:' + name] = (len(sa & sb)/len(sa | sb) if sa or sb else 1.0) if ok else 0.0
        for name in args.detection_fields:
            if not a['field_valid'][name]:
                continue
            score, overlaps = detection_score(a['values'][name], b['values'][name], args.iou_threshold) if b['field_valid'][name] else (0.0, [])
            row['metrics']['detection_f1:' + name] = score
            row['metrics']['matched_iou:' + name] = average(overlaps)
        if args.task == 'rec_cabin':
            for seat in SEATS:
                def seat_attributes(info):
                    root = info.get('normalized')
                    attrs = field(root, 'attributes.' + seat)
                    return attrs if isinstance(attrs, dict) else None
                aa, bb = seat_attributes(a), seat_attributes(b)
                if aa is not None or bb is not None:
                    row['metrics'][f'attribute_keys:{seat}'] = int(aa is not None and bb is not None and set(aa) == set(bb))
                if a['parsed']:
                    for attribute in CABIN_TEXT:
                        name = f'attributes.{seat}.{attribute}'
                        if name not in a['values']:
                            row['metrics']['extra_field:' + name] = int(name in b['values'])
        for name in all_text_fields:
            if not a['field_valid'][name]:
                status = 'reference_missing' if name not in a['values'] else 'reference_invalid'
                row.setdefault('text_status', {})[name] = status
                continue
            ta, tb = a['values'][name], b['values'].get(name, MISSING)
            ok = b['field_valid'][name]
            if name in cabin_text_fields:
                row['metrics']['field_presence:' + name] = int(tb is not MISSING)
                row['metrics']['exact:' + name] = int(ok and ta == tb)
            status = ('candidate_invalid' if not b['parsed'] else 'candidate_missing' if tb is MISSING else
                      'candidate_invalid' if not ok else 'both_empty' if ta == tb == '' else
                      'one_empty' if ta == '' or tb == '' else 'scored')
            row.setdefault('text_status', {})[name] = status
            if status == 'scored':
                pending[name].append((len(rows), ta, tb))
        rows.append(row)
    for name, pairs in pending.items():
        if not pairs:
            continue
        # Encode only aligned pairs and take row-wise dot products (no N x N matrix).
        left = model.encode([x[1] for x in pairs], batch_size=args.batch_size, normalize_embeddings=True)
        right = model.encode([x[2] for x in pairs], batch_size=args.batch_size, normalize_embeddings=True)
        scores = (left * right).sum(axis=1)
        for (index, _, _), score in zip(pairs, scores):
            rows[index]['metrics']['sbert:' + name] = float(score)
    names = sorted({name for row in rows for name in row['metrics']})
    report = {
        'interpretation': 'Agreement with reference outputs, not task accuracy.',
        'configuration': vars(args),
        'reference_samples': len(rows),
        'candidate_extra_ids': sorted(set(candidate) - set(reference)),
        'candidate_missing_count': sum(not row['candidate_present'] for row in rows),
        'reference_valid_rate': mean(row['reference_valid'] for row in rows),
        'candidate_valid_rate': mean(row['candidate_valid'] for row in rows),
        'metrics': {}, 'text_coverage': {}, 'field_validity': {},
    }
    for name in names:
        values = [row['metrics'][name] for row in rows if row['metrics'].get(name) is not None]
        report['metrics'][name] = {'mean': average(values), 'count': len(values)}
    for name in all_text_fields:
        statuses = [row['text_status'][name] for row in rows]
        report['text_coverage'][name] = {status: statuses.count(status) for status in ['scored', 'both_empty', 'one_empty', 'candidate_invalid', 'candidate_missing', 'reference_missing', 'reference_invalid']}
        eligible = sum(row['reference_field_valid'][name] for row in rows)
        report['text_coverage'][name]['reference_field_valid_count'] = eligible
        report['text_coverage'][name]['scored_fraction_of_valid_reference'] = statuses.count('scored')/eligible if eligible else None
    for name in all_exact_fields + all_text_fields + args.set_fields + args.detection_fields:
        report['field_validity'][name] = {
            side + '_valid_count': sum(row[side + '_field_valid'][name] for row in rows)
            for side in ('reference', 'candidate')
        }
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--task', choices=TASKS, help='Original prompt/task name; enables schema validation and metrics')
    parser.add_argument('--exact-fields', nargs='*', default=[])
    parser.add_argument('--text-fields', nargs='*', default=[])
    parser.add_argument('--set-fields', nargs='*', default=[])
    parser.add_argument('--detection-fields', nargs='*', default=[])
    parser.add_argument('--embedding-model', help='Chinese-capable SentenceTransformer model path or ID')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    if args.task:
        if any((args.exact_fields, args.text_fields, args.set_fields, args.detection_fields)):
            parser.error('--task cannot be combined with manual field options')
        args.exact_fields, args.text_fields, args.set_fields, args.detection_fields = fields_for(args.task)
    cabin_text_fields = [f'attributes.{s}.{f}' for s in SEATS for f in CABIN_TEXT] if args.task == 'rec_cabin' else []
    all_text_fields = args.text_fields + cabin_text_fields
    all_exact_fields = args.exact_fields + ([f'attributes.{s}.{f}' for s in SEATS for f in CABIN_EXACT]
                                           if args.task == 'rec_cabin' else [])
    if all_text_fields and not args.embedding_model:
        parser.error('--text-fields requires --embedding-model')
    if not 0 < args.iou_threshold <= 1 or args.batch_size < 1:
        parser.error('IoU threshold must be in (0, 1]; batch size must be positive')
    reference, candidate = read_records(args.reference), read_records(args.candidate)
    if not reference:
        parser.error('reference is empty')
    report, rows = evaluate_records(reference, candidate, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output_dir/'details.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
