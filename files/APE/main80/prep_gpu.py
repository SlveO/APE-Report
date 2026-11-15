"""Finite, synthetic, full-model correctness checks. Never a research stage."""
import argparse
import json
import time
from functools import partial

import torch

from ape import enable_attention_prefill_query
from .common import inside, now, write_json
from .diagnostics import AttentionCapture
from .freeze import CHAT_END, CHAT_START
from .methods import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('sequential', 'pcw', 'ape_default', 'ape_fixed'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = inside(args.output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = time.perf_counter()
    engine = Engine.load(inside('models/Llama-3.1-8B-Instruct'), args.method)
    engine.sync()
    load_seconds = time.perf_counter() - loaded
    row = {'prefix': CHAT_START + 'Use the following two sources.\n',
           'document': 'Source A: The blue lake is named Azure Lake.\nSource B: The green river is named Emerald River.\n',
           'suffix': '\nQuestion: What is the blue lake called? Answer with its name only.\n' + CHAT_END,
           'N': 2, 'max_new_tokens': 8, 'original_document_count': 2}
    prefix, parts, suffix = engine.prepare(row)
    total_input = len(prefix) + sum(map(len, parts)) + len(suffix)
    assert total_input <= 256, 'finite short correctness input only'
    assert engine.model.dtype == torch.bfloat16
    assert engine.model.config.num_key_value_heads == 8
    with torch.inference_mode():
        if args.method == 'ape_fixed':
            with AttentionCapture(engine.model, output / 'token-attention.jsonl', {'test': 'synthetic_full_model_correctness'}):
                cold = engine.run(row)
        else:
            cold = engine.run(row)
        original = tuple(tuple(t.clone() if torch.is_tensor(t) else t for t in layer) for layer in engine.encoded.cache)
        hot = engine.run(row, cache_mode='hot')
        assert cold['output_token_ids'] == hot['output_token_ids']
        for old_layer, layer in zip(original, engine.encoded.cache):
            for old, new in zip(old_layer, layer):
                if torch.is_tensor(old):
                    assert torch.equal(old, new), 'query mutated reusable document KV'
        # Simulate interruption after cold success: rebuild KV only, then hot decode.
        engine.encoded = None
        rebuild_seconds = engine.restore_cache(row)
        recovered = engine.run(row, cache_mode='hot')
        assert recovered['output_token_ids'] == cold['output_token_ids']
        changed = engine.run(dict(row, suffix='\nQuestion: What is the green river called? Give its name only.\n' + CHAT_END), cache_mode='hot')
        checks = {'cold_hot_same_tokens': True, 'immutable_document_KV': True,
                  'resume_KV_only_rebuild': True, 'changed_question_fresh_generation': changed['request_tokens'] > 0}
        if args.method == 'pcw':
            document = sum(parts, [])
            encoded = engine.encode([engine.tokenizer.bos_token_id], [document], suffix, ())
            _, separate = engine.forward(suffix, encoded.cache, encoded.next_position, True)
            _, complete = engine.forward([engine.tokenizer.bos_token_id] + document + suffix, position=0, logits=True)
            delta = float((separate.float() - complete.float()).abs().max())
            torch.testing.assert_close(separate, complete, atol=0.3, rtol=0.01)
            assert separate.argmax(-1).item() == complete.argmax(-1).item()
            checks.update(single_window_equivalence=True, single_window_max_BF16_logit_delta=delta,
                          unique_BOS_and_longest_start=encoded.cache[0][0].shape[2] == 1 + len(document) and encoded.next_position == 1 + len(document))
        if args.method.startswith('ape_'):
            assert cold['stage_calls']['prefix'] == 1 and cold['stage_calls']['context'] == 2 and cold['stage_calls']['query'] >= 1
            encoded = engine.encoded
            logits = []
            for t, s in ((0.9, 0.9), (0.5, 0.7)):
                enable_attention_prefill_query('llama3.1-8b-instruct', engine.model, t, s)
                bound = engine.model.model.layers[0].self_attn.forward.__func__
                assert isinstance(bound, partial) and bound.keywords == {'temperature': t, 'scale': s}
                _, value = engine.forward(suffix, encoded.cache, encoded.next_position, True)
                logits.append(value)
            delta = float((logits[0].float() - logits[1].float()).abs().max())
            assert delta > 0, 'T/S did not affect full-model query logits'
            checks.update(official_three_stage_calls=True, public_T_S_interface=True, T_S_logit_delta=delta)
    result = {'role': 'EXECUTOR', 'status': 'SUBMITTED', 'method': args.method, 'at': now(),
        'model': 'models/Llama-3.1-8B-Instruct', 'physical_gpu': 0, 'dtype': str(engine.model.dtype),
        'load_seconds': load_seconds, 'input_tokens': total_input, 'max_new_tokens': 8,
        'request_generations': 4, 'common_content_token_ids': prefix + sum(parts, []) + suffix,
        'cold': cold, 'hot': hot, 'recovered_hot': recovered, 'changed_question': changed,
        'recovery_build_seconds': rebuild_seconds, 'checks': checks,
        'scope': 'finite synthetic correctness; no benchmark answers or formal speed conclusions'}
    write_json(output / 'result.json', result)
    print(json.dumps({'method': args.method, 'checks': checks, 'input_tokens': total_input, 'generations': 4}), flush=True)


if __name__ == '__main__':
    main()
