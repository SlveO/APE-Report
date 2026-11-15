"""Definitions from LongBench, Lost in the Middle, and DialDoc21.

Sources and downloaded originals are listed in SOURCES.md. All scoring joins by ID.
"""
import re
import string
from collections import Counter


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def qa_f1(prediction, answer):
    pred, ref = normalize(prediction).split(), normalize(answer).split()
    shared = sum((Counter(pred) & Counter(ref)).values())
    if shared == 0:
        return 0.0
    precision, recall = shared / len(pred), shared / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_subspan_em(prediction, answers):
    return float(any(normalize(answer) in normalize(prediction) for answer in answers))


def score(dataset, prediction, answers):
    if not prediction.strip() or not answers:
        raise ValueError('empty prediction or reference cannot be marked successful')
    if dataset in {'qasper', 'hotpotqa', 'musique'}:
        return max(qa_f1(prediction, a) for a in answers)
    if dataset == 'litm':
        return best_subspan_em(prediction, answers)
    if dataset == 'doc2dial':
        import sacrebleu
        return sacrebleu.corpus_bleu([prediction], [[answers[0]]]).score
    raise ValueError(f'unknown scoring protocol: {dataset}')


def corpus_doc2dial(records, reference_by_id):
    """Official sacrebleu defaults; explicit ID join avoids order-based mis-scoring."""
    import sacrebleu
    ids = [r['sample_id'] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(reference_by_id):
        raise ValueError('Doc2Dial prediction/reference IDs do not align')
    metric = sacrebleu.metrics.BLEU()
    result = metric.corpus_score([r['answer'] for r in records], [[reference_by_id[i] for i in ids]])
    return {'score': result.score, 'signature': str(metric.get_signature())}
