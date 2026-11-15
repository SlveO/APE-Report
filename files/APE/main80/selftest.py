"""Finite correctness tests, separate from formal campaign and research timing."""
import argparse
import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from .common import Ledger, ROOT, append_json, inside, read_json, read_records, write_json
from .data import construct, split_groups
from .metrics import best_subspan_em, corpus_doc2dial, qa_f1
from .rules import COUNTS, DATASETS, DEFAULT, LENGTHS, PARAMS, choose, freeze_rule, nearest, select_exploration


class ProtocolTests(unittest.TestCase):
    def test_torn_resume_and_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.jsonl'
            key = {'sample': 'q1', 'T': 0.9}
            ledger = Ledger(path)
            ledger.add({'key': key, 'attempt': 1, 'status': 'sample_error'})
            with path.open('ab') as f:
                f.write(b'{"key":')
            ledger = Ledger(path)
            self.assertEqual(len(ledger.incomplete), 1)
            ledger.add({'key': key, 'attempt': 2, 'status': 'success'})
            restored = Ledger(path)
            self.assertTrue(restored.done(key))
            self.assertEqual(len(restored.incomplete), 1)
            self.assertEqual(restored.next_attempt(key), 3)
            self.assertFalse(restored.done({'sample': 'q1', 'T': 0.7}))
            with self.assertRaises(ValueError):
                restored.add({'key': key, 'attempt': 3, 'status': 'success'})
            self.assertIn(b'{"key":\n', path.read_bytes())

    def test_workspace_escape(self):
        with self.assertRaises(ValueError):
            inside(ROOT / '../outside')

    def test_groups_and_duplicate_questions(self):
        rows = []
        for d in DATASETS:
            for i in range(100):
                rows.append({'sample_id': f'{d}:{i}', 'dataset': d, 'question': f'{d} q{i}', 'support_documents': [f'{d}-doc{i//2}']})
        rows[0]['question'] = rows[100]['question'] = 'Same normalized question'
        order, report = split_groups(rows)
        self.assertEqual(rows[0]['split'], rows[100]['split'])
        self.assertEqual(rows[0]['split'], rows[1]['split'])
        self.assertEqual(sum(map(len, order.values())), 300)
        self.assertTrue(all(not v['evidence_group_overlap'] for k, v in report.items() if '/' in k))

    def test_exact_construction_and_unreviewed_supplement(self):
        class Tokenizer:
            def __call__(self, text, **kw):
                return {'input_ids': list(range(len(text))), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
            def encode(self, text, **kw):
                return list(range(len(text)))
        row = {'document': 'abcdefghijklmnop', 'evidence_char_spans': [[4, 7], [10, 12]], 'sample_id': 'a', 'answers': ['xyz'], 'split': 'explore', 'group': ['a'], 'support_documents': ['a']}
        result = construct(row, 9, Tokenizer(), [], {})
        self.assertTrue(result['applicable'])
        self.assertEqual(len(result['document_tokens']), 9)
        self.assertTrue(set([4, 5, 6, 10, 11]).issubset(result['document_tokens']))
        self.assertEqual(result['document_tokens'], sorted(result['document_tokens']))
        donor = {**row, 'sample_id': 'b', 'group': ['b'], 'support_documents': ['b']}
        self.assertFalse(construct(row, 20, Tokenizer(), [donor], {})['applicable'])

    def test_144_balanced_selection_and_validation_fallback(self):
        records = []
        for d in DATASETS:
            for length in LENGTHS:
                for n in COUNTS:
                    for t, s in PARAMS:
                        records.append({'dataset': d, 'sample_id': d, 'split': 'explore', 'L': length, 'N': n, 'T': t, 'S': s,
                            'score': 1.0 if (t, s) == (0.7, 0.9) else 0.0})
        candidates = select_exploration(records)
        self.assertEqual(candidates['global'], [0.7, 0.9])
        validation = [dict(r, split='validation', score=0.5) for r in records if (r['T'], r['S']) in {DEFAULT, (0.7, 0.9)}]
        rule = freeze_rule(validation, candidates)
        self.assertEqual(rule['fixed'], list(DEFAULT))
        self.assertTrue(all(p == list(DEFAULT) for p in rule['cells'].values()))
        self.assertEqual(nearest(1, 1), (4096, 4))
        self.assertEqual(nearest(1000000, 100), (16384, 16))
        with self.assertRaises(ValueError):
            select_exploration([dict(records[0], split='test')])

    def test_scoring_id_alignment(self):
        self.assertEqual(qa_f1('The red red', 'red'), 2/3)
        self.assertEqual(best_subspan_em('It is New York city.', ['New York']), 1)
        predictions = [{'sample_id': 'a', 'answer': 'The cat sat on the mat.'}, {'sample_id': 'b', 'answer': 'A bird sings a lovely song.'}]
        refs = {'b': 'A bird sings a lovely song.', 'a': 'The cat sat on the mat.'}
        self.assertAlmostEqual(corpus_doc2dial(predictions, refs)['score'], 100)
        with self.assertRaises(ValueError):
            corpus_doc2dial(predictions, {'a': refs['a']})
        with self.assertRaises(ValueError):
            corpus_doc2dial([predictions[0], predictions[0]], refs)

    def test_interrupted_attempt_keeps_unique_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'attempts.jsonl'
            key = {'sample': 'interrupted', 'T': 0.9}
            append_json(Path(tmp) / 'events.jsonl', {'event': 'request_started', 'key': key, 'attempt': 1})
            restored = Ledger(path)
            self.assertEqual(restored.next_attempt(key), 2)
            restored.add({'key': key, 'attempt': 2, 'status': 'success'})
            self.assertTrue(Ledger(path).done(key))

    def test_generated_rule_identity_and_config_snapshot(self):
        from unittest.mock import patch
        from .campaign import snapshot, seal_generated
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / 'code.py'
            source.write_text('value = 1\n')
            with patch('main80.campaign.source_files', return_value=[source]):
                snapshot(directory, {})
                seal_generated(directory, 'rule.json', {'fixed': [0.9, 0.9]})
                snapshot(directory, {}, resume=True)
                (directory / 'rule.json').unlink()
                snapshot(directory, {}, resume=True)
                self.assertEqual(read_json(directory / 'rule.json')['fixed'], [0.9, 0.9])
                write_json(directory / 'rule.json', {'fixed': [0.5, 0.5]})
                with self.assertRaises(ValueError):
                    snapshot(directory, {}, resume=True)
                (directory / 'rule.json').write_bytes((directory / 'snapshot/generated/rule.json').read_bytes())
                source.write_text('value = 2\n')
                with self.assertRaises(ValueError):
                    snapshot(directory, {}, resume=True)

    def test_unknown_hot_speed_does_not_hide_known_budget_cost(self):
        from .campaign import estimate
        class Empty:
            def done(self, key):
                return False
            def finished(self, key):
                return False
        jobs = [{'key': {'L': 4096}, 'method': 'ape_fixed', 'cache': 'hot'}, {'key': {'L': 8192}, 'method': 'pcw', 'cache': 'cold'}]
        speed = [{'method': 'ape', 'length_bucket': 4096, 'cache_mode': 'cold', 'count': 3, 'low': 4.0, 'high': 8.0}]
        self.assertIsNone(estimate(jobs, Empty(), speed))
        self.assertIsNone(estimate(jobs, Empty(), speed, conservative=True))
        self.assertEqual(estimate(jobs, Empty(), speed, conservative=True, partial=True), (0, 8))

    def test_empty_outputs_remain_failed_and_do_not_remove_search_pairs(self):
        from .campaign import selected_records
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ledger = Ledger(directory / 'attempts.jsonl')
            first = {'sample_id': 'q1', 'L': 4096, 'N': 4, 'parameters': [0.5, 0.5]}
            second = dict(first, parameters=[0.9, 0.9])
            ledger.add({'key': first, 'attempt': 1, 'status': 'sample_error', 'answer': '  ', 'output_token_ids': [0]})
            ledger.add({'key': second, 'attempt': 1, 'status': 'success', 'answer': 'valid', 'score': 0.5})
            restored = Ledger(directory / 'attempts.jsonl')
            self.assertFalse(restored.done(first))
            self.assertTrue(restored.finished(first))
            self.assertEqual(restored.outcome(first)['score'], 0)
            paired = selected_records(restored, [{'key': first}, {'key': second}], directory)
            self.assertEqual(len(paired), 2)
            self.assertEqual(paired[0]['status'], 'sample_error')

    def test_worker_resume_after_empty_cold_preserves_zero_and_recovery_time(self):
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        from .worker import execute
        from .methods import SampleError
        calls = []
        class FakeEngine:
            def __init__(self):
                self.encoded = None
                self.device = SimpleNamespace(type='cpu')
            def sync(self):
                pass
            def restore_cache(self, row, param, rule):
                calls.append('KV_only')
                self.encoded = object()
                return 5.0
            def run(self, row, **kwargs):
                calls.append(kwargs['cache_mode'])
                self.encoded = object()
                raise SampleError('empty model output', {'answer': '\n\n', 'output_token_ids': [0],
                    'output_tokens': 1, 'ttft_seconds': 1.0, 'total_seconds': 2.0, 'L': 4096, 'N': 4,
                    'T': 0.9, 'S': 0.9, 'cache_mode': kwargs['cache_mode']})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            row = {'sample_id': 'd1', 'dataset': 'doc2dial', 'split': 'test', 'group': ['doc'], 'answers': ['reference']}
            write_json(directory / 'doc2dial.json', [row])
            key = {'stage': 'doc2dial', 'sample_id': 'd1', 'L': 4096, 'N': None, 'cache': 'cold'}
            cold = {'key': key, 'table': 'doc2dial', 'index': 0, 'method': 'sequential', 'param': None, 'N': None, 'cache': 'cold', 'logical_labels': ['sequential']}
            hot = dict(cold, key=dict(key, cache='hot'), cache='hot')
            bundle = {'model': str(directory), 'data': str(directory), 'method': 'sequential', 'ledger': str(directory / 'attempts.jsonl'),
                      'events': str(directory / 'events.jsonl'), 'heartbeat': str(directory / 'heartbeat.json'), 'deadline_epoch': time.time()+30}
            with patch('main80.worker.Engine.load', side_effect=lambda *args: FakeEngine()):
                execute(dict(bundle, jobs=[cold]))
                execute(dict(bundle, jobs=[hot]))
            ledger = Ledger(directory / 'attempts.jsonl')
            self.assertEqual(calls, ['cold', 'KV_only', 'hot'])
            self.assertEqual(len(ledger.scored), 2)
            self.assertFalse(ledger.done(key))
            self.assertEqual(ledger.outcome(hot['key'])['score'], 0)
            self.assertEqual(ledger.outcome(hot['key'])['ttft_seconds'], 6)
            self.assertEqual(ledger.outcome(hot['key'])['total_seconds'], 7)

    def test_single_instance_lock_survives_launcher_exit(self):
        import fcntl
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'campaign.lock'
            lock = path.open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            child = subprocess.Popen([sys.executable, '-B', '-c', 'import time;time.sleep(10)'],
                start_new_session=True, pass_fds=(lock.fileno(),))
            lock.close()
            try:
                with path.open('a+') as second:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    child.terminate()
                    child.wait(timeout=5)
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)


class StockMethodTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from .methods import Engine
        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, bos_token_id=1, eos_token_id=63,
            rope_scaling={'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 64})
        config._attn_implementation = 'eager'
        model = LlamaForCausalLM(config).eval()
        class Tokenizer:
            bos_token_id = 1
            def encode(self, text, **kw):
                return [int(x) for x in text.split()]
            def decode(self, tokens, **kw):
                return ' '.join(map(str, tokens))
        self.engine = Engine(model, Tokenizer(), 'pcw', device='cpu')

    def tearDown(self):
        self.engine.hook.remove()

    def test_pcw_single_window_equivalence(self):
        import torch
        with torch.inference_mode():
            encoded = self.engine.encode([1], [[3, 4, 5]], [6, 7], ())
            _, logits = self.engine.forward([6, 7], encoded.cache, encoded.next_position, True)
            _, reference = self.engine.forward([1, 3, 4, 5, 6, 7], position=0, logits=True)
        torch.testing.assert_close(logits, reference, atol=1e-6, rtol=1e-5)
        self.assertEqual(encoded.next_position, 4)

    def test_pcw_unequal_windows_against_full_mask_reference(self):
        import torch
        windows = [[3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        query = [13, 14]
        with torch.inference_mode():
            encoded = self.engine.encode([1], windows, query, ())
            self.assertEqual(encoded.order, [1, 0, 2])
            self.assertEqual(encoded.next_position, 5)
            self.assertEqual(encoded.cache[0][0].shape, (1, 2, 11, 8))
            _, actual = self.engine.forward(query, encoded.cache, encoded.next_position, True)
            tokens, positions, starts = [], [], []
            for i in encoded.order:
                starts.append(len(tokens))
                part = [1] + windows[i]
                tokens.extend(part)
                positions.extend(range(len(part)))
            query_start = len(tokens)
            tokens.extend(query)
            positions.extend(range(5, 7))
            mask = torch.full((1, 1, len(tokens), len(tokens)), torch.finfo(torch.float32).min)
            for start, end in zip(starts, starts[1:] + [query_start]):
                for i in range(start, end):
                    mask[0, 0, i, start:i+1] = 0
            for i in range(query_start, len(tokens)):
                mask[0, 0, i, :i+1] = 0
                mask[0, 0, i, starts[1:]] = torch.finfo(torch.float32).min
            output = self.engine.model(input_ids=torch.tensor([tokens]), position_ids=torch.tensor([positions]), attention_mask=mask, use_cache=False)
            reference = output.logits[:, -1]
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)

    def test_hot_cache_immutable_and_new_history_regenerates(self):
        import torch
        row = {'prefix': '1', 'document_tokens': [3, 4, 5, 6, 7], 'suffix': '8 9', 'N': 2, 'max_new_tokens': 4}
        cold = self.engine.run(row)
        snapshot = tuple(tuple(t.clone() for t in layer) for layer in self.engine.encoded.cache)
        hot = self.engine.run(row, cache_mode='hot')
        self.assertEqual(cold['output_token_ids'], hot['output_token_ids'])
        self.assertTrue(hot['cache_hit'])
        for a, b in zip(snapshot, self.engine.encoded.cache):
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))
        row['suffix'] = '10 11 12'
        changed = self.engine.run(row, cache_mode='hot')
        self.assertTrue(changed['cache_hit'])
        self.assertEqual(changed['request_tokens'], 3)
        with self.assertRaises(ValueError):
            self.engine.run(dict(row, document_tokens=[22, 23, 24]), cache_mode='hot')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=('protocol', 'methods_cpu'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests if args.suite == 'protocol' else StockMethodTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    inside(args.output).parent.mkdir(parents=True, exist_ok=True)
    inside(args.output).write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
