"""Diagnostic-only token probabilities, checked against actual APE output.

The branch masses alone are not token attention. Reconstruct the within-branch
softmax, multiply by actual BF16 branch mass, then compare weighted values.
"""
import math

import torch

from .common import append_json


class AttentionCapture:
    def __init__(self, model, output, metadata):
        self.model, self.output, self.metadata = model, output, metadata
        self.step = {}

    def __enter__(self):
        for i in (0, 15, 31):
            attention = self.model.model.layers[i].self_attn
            attention.ape_observer = self.callback(i)
        return self

    def callback(self, layer):
        def capture(module, q, kc, ko, vc, vo, mass, actual, temperature):
            step = self.step.get(layer, 0)
            self.step[layer] = step + 1
            heads = [0, 8, 16, 24]
            qi = q[0, heads, -1].float()
            context_logits = torch.einsum('hd,hkd->hk', qi, kc[0, heads].float()) / (math.sqrt(module.head_dim) * temperature)
            other_logits = torch.einsum('hd,hkd->hk', qi, ko[0, heads].float()) / math.sqrt(module.head_dim)
            # The last query position sees every cached/request token (causal tail).
            pc, po = context_logits.softmax(-1), other_logits.softmax(-1)
            branch = mass[0, -1, heads, 0].float()
            wc, wo = pc * branch[:, :1], po * branch[:, 1:]
            reconstruction = torch.einsum('hk,hkd->hd', wc, vc[0, heads].float()) + torch.einsum('hk,hkd->hd', wo, vo[0, heads].float())
            target = actual[0, -1, heads].float()
            max_abs = float((reconstruction - target).abs().max())
            # Quantized flash outputs and BF16 branch mixing introduce rounding.
            tolerance = 0.03 + 0.02 * float(target.abs().max())
            if max_abs > tolerance:
                raise RuntimeError(f'token attention reconstruction mismatch: {max_abs} > {tolerance}')
            p = module.len_prefix
            # Restore physical cache order: shared prefix, all documents, request.
            weights = torch.cat([wo[:, :p], wc, wo[:, p:]], dim=-1)
            append_json(self.output, {**self.metadata, 'layer': layer, 'heads': heads, 'step': step,
                'query_kind': 'query_last_token' if step == 0 else 'decode_token',
                'max_abs_output_error': max_abs, 'tolerance': tolerance,
                'token_probabilities': weights.cpu().tolist(), 'branch_masses': branch.cpu().tolist()})
        return capture

    def __exit__(self, *exc):
        for i in (0, 15, 31):
            del self.model.model.layers[i].self_attn.ape_observer
