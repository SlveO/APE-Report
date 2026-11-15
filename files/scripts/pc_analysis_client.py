"""Portable, standard-library-only reader for ape-pc-export/v1. No GPU/imports."""
import argparse
import codecs
import json
from collections import Counter
from pathlib import Path


class Bundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['format'] != 'ape-pc-export/v1':
            raise ValueError('unknown export format')
        self.entries = {e['path']: e for e in self.manifest['files']}

    def confined(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('manifest path leaves export')
        return path

    def chunks(self, relative):
        entry = self.entries[relative]
        if entry['storage'] == 'alias':
            yield from self.chunks(entry['alias_of'])
            return
        pieces = entry['parts'] if entry['storage'] == 'parts' else [{'path': entry['stored_path'], 'bytes': entry['bytes']}]
        total = 0
        for piece in pieces:
            path = self.confined(piece['path'])
            if path.stat().st_size != piece['bytes']:
                raise ValueError('stored size mismatch: ' + piece['path'])
            with path.open('rb') as stream:
                while True:
                    data = stream.read(1024*1024)
                    if not data:
                        break
                    total += len(data)
                    yield data
        if total != entry['bytes']:
            raise ValueError('reconstructed size mismatch: ' + relative)

    def lines(self, relative):
        decoder = codecs.getincrementaldecoder('utf-8')()
        pending = ''
        for chunk in self.chunks(relative):
            pending += decoder.decode(chunk)
            lines = pending.split('\n')
            pending = lines.pop()
            yield from lines
        pending += decoder.decode(b'', final=True)
        if pending:
            yield pending

    def json(self, relative):
        return json.loads(b''.join(self.chunks(relative)))

    def verify(self):
        for relative in self.entries:
            for _ in self.chunks(relative):
                pass
        run = 'output/runs/' + self.manifest['campaign'] + '/'
        rows = [json.loads(line) for line in self.lines(run + 'attempts.jsonl') if line.strip()]
        keys = [json.dumps(r['key'], sort_keys=True) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate result key')
        summary = self.json(run + 'summary.json')
        success = sum(r['status'] == 'success' for r in rows)
        if success != summary['successful_records']:
            raise ValueError('raw record/summary count mismatch')
        print(json.dumps({'files': len(self.entries), 'records': len(rows), 'successful': success,
            'statuses': dict(Counter(r['status'] for r in rows)), 'checks': 'sizes, JSON, IDs and counts; no digests'}, ensure_ascii=False, indent=2))

    def summary(self):
        run = 'output/runs/' + self.manifest['campaign'] + '/'
        result = self.json(run + 'summary.json')
        print('Campaign:', self.manifest['campaign'], '|', result['status'], '| pending:', result['pending'])
        print('Successful:', result['successful_records'], '| failed:', result['failed_attempts'])
        print('\nLongBench common510 test: dataset F1 ×100')
        for row in result['longbench_five_method_paired']:
            print(row['dataset'], 'n=', row['paired_n'], {m: round(v*100, 4) for m,v in row['mean_f1'].items()})
        print('\nLost in the Middle:')
        print(json.dumps(result['litm_paired']['methods'], ensure_ascii=False, indent=2))
        print('\nDoc2Dial:')
        print(json.dumps(result['doc2dial_paired']['conditions'], ensure_ascii=False, indent=2))

    def restore(self, output):
        output = Path(output).resolve()
        for relative in self.entries:
            target = (output / relative).resolve()
            if not target.is_relative_to(output):
                raise ValueError('output path leaves restore root')
            if target.exists():
                with target.open('rb') as existing:
                    for data in self.chunks(relative):
                        if existing.read(len(data)) != data:
                            raise ValueError('existing PC file differs; not overwritten: ' + relative)
                    if existing.read(1):
                        raise ValueError('existing PC file has extra content: ' + relative)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restoring')
            if temporary.exists():
                raise ValueError('previous incomplete restore retained: ' + str(temporary))
            with temporary.open('xb') as stream:
                for data in self.chunks(relative):
                    stream.write(data)
            temporary.replace(target)
        print('Restored', len(self.entries), 'original paths into', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('verify', 'summary', 'restore'))
    parser.add_argument('--output', default='restored')
    args = parser.parse_args()
    bundle = Bundle(Path(__file__).resolve().parent)
    if args.action == 'verify':
        bundle.verify()
    elif args.action == 'summary':
        bundle.summary()
    else:
        bundle.restore(args.output)


if __name__ == '__main__':
    main()
