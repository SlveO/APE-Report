"""Standalone SVG scientific figures from verified diagnostic numbers."""
from collections import defaultdict

from .common import identity, inside, read_records, write_json


def render(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory = inside(directory)
    records, incomplete = read_records(directory / 'attention.jsonl')
    cases = defaultdict(list)
    for record in records:
        cases[identity(record['key'])].append(record)
    output = directory / 'attention-figures'
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (key, rows) in enumerate(cases.items()):
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
        for ax, layer in zip(axes, (0, 15, 31)):
            layer_rows = sorted((r for r in rows if r['layer'] == layer), key=lambda r: r['step'])
            if not layer_rows:
                ax.set_title(f'Layer {layer}: no recorded steps')
                continue
            bins = (max(len(r['token_probabilities'][0]) for r in layer_rows) + 63) // 64
            matrix = np.full((len(layer_rows), bins), np.nan)
            for i, row in enumerate(layer_rows):
                weights = np.asarray(row['token_probabilities'], dtype=float).mean(axis=0)
                for b in range(bins):
                    segment = weights[b*64:(b+1)*64]
                    if len(segment):
                        matrix[i, b] = segment.sum()
            image = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0)
            ax.set_title(f'Layer {layer}; mean of heads 0, 8, 16, 24; mass per 64-token bin')
            ax.set_ylabel('Actual query/decode step')
            ax.set_xlabel('Physical cache token index / 64 (prefix, documents, request)')
            fig.colorbar(image, ax=ax, label='Attention mass')
        metadata = rows[0]['key']
        fig.suptitle(f'{metadata["sample_id"]} | L={metadata["L"]}, N={metadata["N"]}, T/S={metadata["parameters"]}', fontsize=11)
        name = f'attention-{number:03d}.svg'
        fig.savefig(output / name, format='svg')
        plt.close(fig)
        index.append({'figure': name, 'key': metadata, 'raw_records': len(rows), 'aggregation': 'mean of four specified heads; sum over contiguous64 token bins',
            'maximum_output_reconstruction_error': max(r['max_abs_output_error'] for r in rows)})
    write_json(output / 'index.json', {'figures': index, 'incomplete_raw_lines': incomplete, 'raw_numbers': '../attention.jsonl'})
    return len(index)
