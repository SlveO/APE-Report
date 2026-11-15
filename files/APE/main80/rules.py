"""Predeclared 144 configurations and validation-only rule selection."""
import math
from itertools import product
from statistics import mean

LENGTHS = (4096, 8192, 16384)
COUNTS = (4, 8, 16)
PARAMS = tuple(product((0.5, 0.7, 0.9, 1.0), repeat=2))
DEFAULT = (0.9, 0.9)
DATASETS = ('qasper', 'hotpotqa', 'musique')


def tie_key(param):
    t, s = param
    return ((t - 0.9) ** 2 + (s - 0.9) ** 2, t, s)


def nearest(length, count):
    if length <= 0 or count <= 0:
        raise ValueError('L and N must be positive')
    return min(product(LENGTHS, COUNTS), key=lambda cell: (
        (math.log2(length) - math.log2(cell[0])) ** 2 + (math.log2(count) - math.log2(cell[1])) ** 2,
        cell[0], cell[1]))


def balanced_score(records, cells=None):
    cells = list(cells or product(LENGTHS, COUNTS))
    groups = {(d, l, n): [] for d in DATASETS for l, n in cells}
    for r in records:
        key = (r['dataset'], r['L'], r['N'])
        if key in groups:
            groups[key].append(r['score'])
    missing = [key for key, values in groups.items() if not values]
    if missing:
        raise ValueError(f'incomplete balanced rule selection: {missing}')
    return mean(mean(v) for v in groups.values())


def select_exploration(records):
    if any(r['split'] != 'explore' for r in records):
        raise ValueError('only exploration records may choose candidates')
    by_param = {p: [r for r in records if (r['T'], r['S']) == p] for p in PARAMS}
    # Every parameter must use the same IDs in each cell.
    identities = [sorted((r['dataset'], r['sample_id'], r['L'], r['N']) for r in rows) for rows in by_param.values()]
    if not all(i == identities[0] for i in identities):
        raise ValueError('unpaired exploration results')
    global_best = min(PARAMS, key=lambda p: (-balanced_score(by_param[p]), tie_key(p)))
    per_cell = {f'{l}/{n}': min(PARAMS, key=lambda p: (-balanced_score(by_param[p], [(l, n)]), tie_key(p))) for l, n in product(LENGTHS, COUNTS)}
    return {'global': list(global_best), 'cells': {k: list(v) for k, v in per_cell.items()}}


def freeze_rule(records, candidates):
    if any(r['split'] != 'validation' for r in records):
        raise ValueError('only validation may freeze the rule')
    def rows(param, cell=None):
        return [r for r in records if (r['T'], r['S']) == tuple(param) and (cell is None or (r['L'], r['N']) == cell)]
    fixed_options = {DEFAULT, tuple(candidates['global'])}
    # Check pairing separately per candidate / cell, including deduplicated runs.
    for l, n in product(LENGTHS, COUNTS):
        choices = fixed_options | {tuple(candidates['cells'][f'{l}/{n}'])}
        ids = [sorted((r['dataset'], r['sample_id']) for r in rows(p, (l, n))) for p in choices]
        if not ids[0] or not all(x == ids[0] for x in ids):
            raise ValueError('unpaired validation results')
    fixed = min(fixed_options, key=lambda p: (-balanced_score(rows(p)), tie_key(p)))
    cells, scores = {}, {}
    for l, n in product(LENGTHS, COUNTS):
        p = tuple(candidates['cells'][f'{l}/{n}'])
        best_score = balanced_score(rows(p, (l, n)), [(l, n)])
        fixed_score = balanced_score(rows(fixed, (l, n)), [(l, n)])
        cells[f'{l}/{n}'] = list(p if best_score > fixed_score else fixed)
        scores[f'{l}/{n}'] = {'candidate': best_score, 'fixed': fixed_score}
    return {'protocol': 'ape-v3-nine-cell', 'fixed': list(fixed), 'cells': cells, 'validation_scores': scores,
            'mapping': 'Euclidean log2(L),log2(actual_N); ties smaller L then N; clamp boundaries'}


def choose(rule, length, count):
    l, n = nearest(length, count)
    return tuple(rule['cells'][f'{l}/{n}'])
