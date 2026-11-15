"""Frozen stage DAG; only preauthorized nested sample counts may change."""
from itertools import product

from .common import identity, read_json
from .rules import COUNTS, DATASETS, DEFAULT, LENGTHS, PARAMS

STAGES = ('baseline', 'explore', 'validation', 'length_check', 'attention', 'longbench_final', 'litm', 'doc2dial')
DEPENDENCIES = {'baseline': [], 'explore': ['baseline'], 'validation': ['explore'],
                'length_check': ['validation'], 'attention': ['validation'],
                'longbench_final': ['length_check', 'attention'], 'litm': ['longbench_final'], 'doc2dial': ['litm']}


class JobFactory:
    def __init__(self, frozen):
        self.directory = frozen
        self.manifest = read_json(f'{frozen}/frozen.json')
        self.tables = {name: read_json(f'{frozen}/{name}.json') for name in ('longbench', 'development', 'litm', 'doc2dial')}
        self.orders = self.manifest['orders']

    def job(self, stage, table, i, method, param=None, n=None, cache='cold', labels=None):
        row = self.tables[table][i]
        key = {'stage': stage, 'sample_id': row['sample_id'], 'method': method,
               'L': row.get('L', row.get('_length')), 'N': n, 'position': row.get('position'),
               'parameters': list(param) if param is not None else None, 'cache': cache}
        return {'key': key, 'table': table, 'index': i, 'method': method, 'param': list(param) if param else None,
                'N': n, 'cache': cache, 'logical_labels': labels or [method], 'max_new_tokens': row['max_new_tokens']}

    def make(self, stage, limits, candidates=None):
        jobs = []
        if stage == 'baseline':
            for method in ('sequential', 'pcw', 'ape_default'):
                jobs.extend(self.job(stage, 'longbench', i, method) for i in range(len(self.tables['longbench'])))
        elif stage in {'explore', 'validation'}:
            split = stage if stage == 'explore' else 'validation'
            selected = self.orders[split][:limits[split]]
            for sid in selected:
                for i, row in enumerate(self.tables['development']):
                    if row['sample_id'] != sid or row['L'] not in LENGTHS:
                        continue
                    for n in COUNTS:
                        if stage == 'explore':
                            choices = {p: ['grid_search'] for p in PARAMS}
                        else:
                            choices = {}
                            for label, p in (('default', DEFAULT), ('global', candidates['global']), ('cell', candidates['cells'][f'{row["L"]}/{n}'])):
                                choices.setdefault(tuple(p), []).append(label)
                        jobs.extend(self.job(stage, 'development', i, 'ape_fixed', p, n, 'search', labels) for p, labels in choices.items())
        elif stage == 'length_check':
            # Fixed first ten validation IDs, independent of validation scores or shrinking.
            selected = self.orders['validation'][:10]
            for method in ('ape_fixed', 'ape_dynamic'):
                for i, row in enumerate(self.tables['development']):
                    if row['sample_id'] in selected and row['L'] in (6144, 12288):
                        jobs.extend(self.job(stage, 'development', i, method, n=n) for n in COUNTS)
        elif stage == 'attention':
            for d in DATASETS:
                for length, n in ((4096, 4), (16384, 16)):
                    eligible = {r['sample_id']: i for i, r in enumerate(self.tables['development']) if r['dataset'] == d and r['split'] == 'explore' and r['L'] == length}
                    first = next(s for s in self.orders['explore'] if s in eligible)
                    choices = {}
                    for label, p in (('default', DEFAULT), ('exploration_cell_best', tuple(candidates['cells'][f'{length}/{n}']))):
                        choices.setdefault(p, []).append(label)
                    for p, labels in choices.items():
                        jobs.append(self.job(stage, 'development', eligible[first], 'ape_fixed', p, n, 'cold', ['diagnostic_only', *labels]))
        elif stage == 'longbench_final':
            selected = self.orders['test'][:limits['test']]
            for method in ('ape_fixed', 'ape_dynamic'):
                jobs.extend(self.job(stage, 'longbench', i, method) for i, row in enumerate(self.tables['longbench']) if row['sample_id'] in selected)
        elif stage in {'litm', 'doc2dial'}:
            selected = self.orders[stage][:limits[stage]]
            for method in ('sequential', 'ape_fixed', 'ape_dynamic'):
                for sid in selected:
                    for i, row in enumerate(self.tables[stage]):
                        if row['sample_id'] == sid:
                            jobs.append(self.job(stage, stage, i, method))
                            if stage == 'doc2dial':
                                jobs.append(self.job(stage, stage, i, method, cache='hot'))
        else:
            raise ValueError(stage)
        keys = [identity(j['key']) for j in jobs]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate scheduled key')
        return jobs

    def initial_limits(self):
        return {**{s: len(self.orders[s]) for s in ('explore', 'validation', 'test')}, 'litm': 500, 'doc2dial': 500}

    def lower(self, limits, locked=()):
        for names, levels in ((('explore',), (35, 20, 10, 6)), (('validation',), (30, 20, 10)),
                              (('test',), (300, 150, 90)), (('litm', 'doc2dial'), (300, 200, 100))):
            if any(name in locked for name in names):
                continue
            current = min(limits[n] for n in names)
            below = [n for n in levels if n < current]
            if below:
                updated = dict(limits)
                for name in names:
                    updated[name] = below[0]
                return updated
        return None

    def selected_ids(self, limits):
        return {name: self.orders[name][:n] for name, n in limits.items()}
