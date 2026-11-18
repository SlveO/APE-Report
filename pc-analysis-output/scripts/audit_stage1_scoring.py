"""Stage 1 only: input pairing, reference joins, and scoring-definition audit.

Evidence stays read-only. Outputs are restricted to stages/01_audit/tables,reports.
Run with Python 3.12.7 -B; standard library only, no installed scorer required.
BLEU is an independent implementation of SacreBLEU 2.4.3 default 13a/exp
sufficient statistics; exact upstream URLs are emitted in the audit report.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import functools
import gzip
import json
import math
import platform
import re
import string
import sys
from pathlib import Path

CAMPAIGN = "main80-20261114T041503"
METHODS = ("sequential", "ape_fixed", "ape_dynamic")
POSITIONS = (0, 9, 19)
CHAT_START = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
CHAT_END = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SIGNATURE = "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.3"
TOLERANCE = 1e-8
SOURCES = {
    "longbench_metrics": "https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py",
    "longbench_eval": "https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py",
    "litm_metrics": "https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py",
    "litm_first_line": "https://github.com/nelson-liu/lost-in-the-middle/blob/main/scripts/evaluate_qa_responses.py",
    "doc2dial_eval": "https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/sharedtask_utils.py",
    "doc2dial_template": "https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/scripts/subtask2/seq2seq_utils.py",
    "doc2dial_historical_versions": "https://github.com/doc2dial/sharedtask-dialdoc2021/blob/master/requirements.txt",
    "historical_hf_defaults_and_example": "https://github.com/huggingface/datasets/blob/1.3.0/metrics/sacrebleu/sacrebleu.py",
    "bleu_243": "https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/metrics/bleu.py",
    "tokenizer_13a_243": "https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/tokenizers/tokenizer_13a.py",
    "tokenizer_regexp_243": "https://github.com/mjpost/sacrebleu/blob/v2.4.3/sacrebleu/tokenizers/tokenizer_re.py",
    "sacrebleu_example_component": "https://github.com/mjpost/sacrebleu/blob/v2.4.3/README.md#using-sacrebleu-from-python",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_f1(prediction, references):
    pred = normalize(prediction).split()
    scores = []
    for reference in references:
        ref = normalize(reference).split()
        common = sum((collections.Counter(pred) & collections.Counter(ref)).values())
        scores.append(2 * common / (len(pred) + len(ref)) if common else 0.0)
    return max(scores, default=0.0)


def em(prediction, references):
    prediction = normalize(prediction)
    return float(any(normalize(ref) in prediction for ref in references))


# The regex sequence follows the version-pinned TokenizerRegexp source above.
TOKENIZER_REGEX = (
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
)


@functools.lru_cache(maxsize=8192)
def tokens_13a(text):
    text = text.rstrip().replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    for source, replacement in (("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(source, replacement)
    text = f" {text} "
    for regex, replacement in TOKENIZER_REGEX:
        text = regex.sub(replacement, text)
    return tuple(text.split())


def bleu_stats(prediction, references):
    """[system length, closest reference length, correct 1..4, total 1..4]."""
    pred = tokens_13a(prediction)
    refs = [tokens_13a(ref) for ref in references if ref is not None and ref != ""]
    if not refs:
        raise ValueError("BLEU requires a nonempty reference set")
    ref_len = min((len(ref) for ref in refs), key=lambda n: (abs(n - len(pred)), n))
    correct, total = [], []
    for order in range(1, 5):
        pred_ngrams = collections.Counter(tuple(pred[i:i + order]) for i in range(len(pred) - order + 1))
        ref_ngrams = collections.Counter()
        for ref in refs:
            current = collections.Counter(tuple(ref[i:i + order]) for i in range(len(ref) - order + 1))
            for ngram, count in current.items():
                ref_ngrams[ngram] = max(ref_ngrams[ngram], count)
        correct.append(sum((pred_ngrams & ref_ngrams).values()))
        total.append(sum(pred_ngrams.values()))
    return [len(pred), ref_len] + correct + total


def add_stats(vectors):
    result = [0] * 10
    for vector in vectors:
        for i, value in enumerate(vector):
            result[i] += value
    return result


def bleu_from_stats(vector):
    sys_len, ref_len = vector[:2]
    correct, total = vector[2:6], vector[6:10]
    if not sys_len or not any(correct):
        return 0.0
    bp = math.exp(1 - ref_len / sys_len) if sys_len < ref_len else 1.0
    precisions, smoothing = [], 1.0
    for hit, count in zip(correct, total):
        if not count:  # effective_order=False leaves an absent order at zero.
            return 0.0
        if hit:
            precisions.append(100.0 * hit / count)
        else:
            smoothing *= 2.0
            precisions.append(100.0 / (smoothing * count))
    return bp * math.exp(sum(math.log(p) for p in precisions) / 4)


def text2line(text):
    return text.replace("\n", "\t").replace("\r", "\t").strip()


def tag(role, text):
    return f"<{role}>\t{text2line(text)}"


def write_csv(path, rows, empty_fields=()):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict)) else v for k, v in row.items()})


def official_longbench_functions(path):
    """Load only the three pure official functions, without third-party imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"normalize_answer", "f1_score", "qa_f1_score"}]
    namespace = {"re": re, "string": string, "Counter": collections.Counter}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def audit(root, output):
    tables, reports = output / "tables", output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    run = root / "output" / "runs" / CAMPAIGN
    snapshot = run / "snapshot" / "APE" / "main80"
    checks, issues, parsed, bad_lines = [], [], [], []

    def issue(kind, **data):
        issues.append({"kind": kind, **data})

    def check(name, denominator, failures, scope, evidence, note=""):
        checks.append({"check": name, "scope": scope, "denominator": denominator,
                       "failure_count": failures, "unit": "count", "status": "PASS" if not failures else "FINDING",
                       "evidence": evidence, "note": note})

    print("Reading attempts, frozen inputs, and request events", flush=True)
    with (run / "attempts.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines.append(line_no)
                continue
            row["_line"] = line_no
            parsed.append(row)
    check("attempts JSON decoding", len(parsed) + len(bad_lines), len(bad_lines), "all physical records", "attempts.jsonl", str(bad_lines[:10]))
    by_key = collections.defaultdict(list)
    for row in parsed:
        by_key[identity(row["key"])].append(row)
    duplicate_success = [key for key, rows in by_key.items() if sum(r["status"] == "success" for r in rows) > 1]
    for key in duplicate_success:
        issue("duplicate_success", key=json.loads(key), lines=[r["_line"] for r in by_key[key]])
    check("unique successful task key", len(by_key), len(duplicate_success), "all physical task keys", "attempts.jsonl:key,status")
    successes = [row for row in parsed if row["status"] == "success"]
    lb = load_json(root / "datasets/frozen-main80/longbench.json")
    litm = load_json(root / "datasets/frozen-main80/litm.json")
    doc = load_json(root / "datasets/frozen-main80/doc2dial.json")
    lb_by_id = {row["sample_id"]: row for row in lb}
    litm_by = {(row["sample_id"], row["position"]): row for row in litm}
    doc_by_id = {row["sample_id"]: row for row in doc}
    config = load_json(root / "data_sources.json")
    starts, ends, rebuilds = collections.defaultdict(list), collections.defaultdict(list), []
    with (run / "events.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            event = json.loads(line)
            event["_line"] = line_no
            if event.get("event") == "request_started":
                starts[(identity(event["key"]), event["attempt"])].append(event)
            elif event.get("event") == "request_ended":
                ends[(identity(event["key"]), event["attempt"])].append(event)
            elif event.get("event") == "resume_cache_rebuilt":
                rebuilds.append(event)

    # Validate saved per-generation definitions, with independent reference joins.
    official = official_longbench_functions(root / "datasets/raw-main80/references/longbench_metrics.py.txt")
    score_groups = collections.defaultdict(lambda: {"n": 0, "mismatches": 0, "max_abs_difference": 0.0})
    score_differences, ref_issues, invalid = [], [], []
    bleu_vectors = {}
    for row in parsed:
        dataset, sid = row["dataset"], row["sample_id"]
        frozen = (lb_by_id.get(sid) if dataset in {"qasper", "hotpotqa", "musique"}
                  else litm_by.get((sid, row.get("position"))) if dataset == "litm" else doc_by_id.get(sid))
        if frozen is None or row.get("reference_answers") != frozen.get("answers"):
            ref_issues.append({"line": row["_line"], "sample_id": sid, "dataset": dataset,
                               "kind": "missing_frozen_ID" if frozen is None else "reference_mismatch"})
        references, prediction = row.get("reference_answers"), row.get("answer")
        saved = row.get("score")
        if not isinstance(prediction, str) or not isinstance(references, list) or not references or any(not isinstance(r, str) or not r.strip() for r in references):
            invalid.append({"line": row["_line"], "sample_id": sid, "kind": "missing_answer_or_reference"})
            continue
        if not isinstance(saved, (int, float)) or isinstance(saved, bool) or not math.isfinite(saved):
            invalid.append({"line": row["_line"], "sample_id": sid, "kind": "nonfinite_or_missing_score"})
            continue
        if not 0 <= saved <= (100 if dataset == "doc2dial" else 1):
            invalid.append({"line": row["_line"], "sample_id": sid, "kind": "score_out_of_range"})
        if not prediction.strip() and row["status"] == "success":
            invalid.append({"line": row["_line"], "sample_id": sid, "kind": "successful_empty_prediction"})
        if dataset in {"qasper", "hotpotqa", "musique"}:
            recomputed = qa_f1(prediction, references)
            upstream = max(official["qa_f1_score"](prediction, r) for r in references)
            if abs(recomputed - upstream) > TOLERANCE:
                issue("independent_vs_official_F1", line=row["_line"], difference=recomputed - upstream)
            metric = "LongBench_v1_F1"
        elif dataset == "litm":
            recomputed, metric = em(prediction, references), "campaign_full_answer_EM"
        elif dataset == "doc2dial":
            vector = bleu_stats(prediction, references[:1])
            bleu_vectors[row["_line"]] = vector
            recomputed, metric = bleu_from_stats(vector), "single_record_corpus_BLEU_defaults"
        else:
            issue("unknown_dataset", line=row["_line"], dataset=dataset)
            continue
        key = (dataset, row["key"]["stage"], metric)
        group = score_groups[key]
        group["n"] += 1
        difference = recomputed - saved
        group["max_abs_difference"] = max(group["max_abs_difference"], abs(difference))
        if abs(difference) > TOLERANCE:
            group["mismatches"] += 1
            score_differences.append({"line": row["_line"], "dataset": dataset, "sample_id": sid,
                                      "metric": metric, "saved": saved, "recomputed": recomputed, "difference": difference})
    definition_rows = [{"dataset": d, "stage": stage, "metric": metric, "n_physical_records": g["n"],
                        "mismatch_count": g["mismatches"], "max_abs_difference": g["max_abs_difference"],
                        "tolerance": TOLERANCE, "unit": "raw F1/EM or BLEU point", "scope": "saved score definition consistency"}
                       for (d, stage, metric), g in sorted(score_groups.items())]
    check("saved generation score reproduction", sum(g["n"] for g in score_groups.values()), len(score_differences), "all scored physical records", "attempts.jsonl:score,answer,reference_answers; snapshot/metrics.py")
    check("frozen sample/reference ID join", len(parsed), len(ref_issues), "all physical generation records", "frozen-main80/{longbench,litm,doc2dial}.json")
    check("score and reference validity", len(parsed), len(invalid), "all physical generation records", "attempts.jsonl")

    # LITM: exact method/position pairing and source-document multiset checks.
    print("Auditing LITM pairing and raw document sets", flush=True)
    litm_records = [r for r in successes if r["key"]["stage"] == "litm"]
    litm_condition = collections.defaultdict(list)
    for row in litm_records:
        litm_condition[(row["sample_id"], row["method"], row["position"])].append(row)
    litm_ids = sorted({r["sample_id"] for r in litm})
    pairing_rows, litm_pair_errors = [], 0
    for sid in litm_ids:
        counts = [len(litm_condition[(sid, method, pos)]) for method in METHODS for pos in POSITIONS]
        errors = sum(n != 1 for n in counts)
        litm_pair_errors += errors
        pairing_rows.append({"sample_id": sid, "expected_conditions": 9, "successful_records": sum(counts),
                             "missing_conditions": sum(n == 0 for n in counts), "duplicate_conditions": sum(n > 1 for n in counts),
                             "position_values": POSITIONS, "unit": "condition", "status": "PASS" if not errors else "FINDING"})
    check("LITM 500 questions x 3 positions x 3 methods", len(litm_ids) * 9, litm_pair_errors + abs(len(litm_ids) - 500), "sequential/fixed/dynamic cold requests", "attempts.jsonl; frozen-main80/litm.json")
    selected_questions = {" ".join(r["question"].casefold().split()) for r in litm}
    raw_litm, raw_duplicates = {}, []
    for pos, path in config["litm"].items():
        rows = {}
        with gzip.open(root / path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = " ".join(row["question"].casefold().split())
                if key in selected_questions:
                    if key in rows:
                        raw_duplicates.append({"question": row["question"], "position": int(pos)})
                    rows[key] = row
        raw_litm[int(pos)] = rows
    litm_input_rows = []
    prompt = (root / config["litm_prompt"]).read_text(encoding="utf-8").rstrip("\n")
    before, after = prompt.split("{search_results}")
    for sid in litm_ids:
        variants = [litm_by[(sid, pos)] for pos in POSITIONS]
        qkey = " ".join(variants[0]["question"].casefold().split())
        raw = [raw_litm[pos].get(qkey) for pos in POSITIONS]
        if any(r is None for r in raw):
            issue("litm_missing_raw_question", sample_id=sid)
            continue
        documents = [sorted((d["title"], d["text"]) for d in r["ctxs"]) for r in raw]
        same_set = all(d == documents[0] for d in documents)
        same_question = all(r["question"] == raw[0]["question"] for r in raw)
        same_references = all(sorted(r["answers"]) == sorted(raw[0]["answers"]) for r in raw)
        gold_identities = [(r["ctxs"][p]["title"], r["ctxs"][p]["text"]) for p, r in zip(POSITIONS, raw)]
        same_gold = all(g == gold_identities[0] for g in gold_identities)
        for pos, frozen, original in zip(POSITIONS, variants, raw):
            gold_ok = len(original["ctxs"]) == 20 and sum(bool(d.get("isgold")) for d in original["ctxs"]) == 1 and bool(original["ctxs"][pos].get("isgold"))
            expected_doc = "\n".join(f'Document [{i+1}](Title: {d["title"]}) {d["text"]}' for i, d in enumerate(original["ctxs"]))
            answer_ok = frozen["answers"] == original["answers"]
            template_ok = (frozen["prefix"] == CHAT_START + before.format(question=original["question"])
                           and frozen["suffix"] == after.format(question=original["question"]) + CHAT_END)
            document_ok = frozen["document"] == expected_doc
            litm_input_rows.append({"sample_id": sid, "position": pos, "document_count": len(original["ctxs"]),
                                    "same_document_multiset_all_positions": same_set,
                                    "same_question_all_positions": same_question, "same_references_all_positions": same_references,
                                    "same_gold_document_identity_all_positions": same_gold, "unique_gold_at_expected_position": gold_ok,
                                    "frozen_document_exact_match": document_ok, "reference_exact_match": answer_ok,
                                    "prompt_exact_match": template_ok, "status": "PASS" if all((same_set, same_question, same_references, same_gold, gold_ok, document_ok, answer_ok, template_ok)) else "FINDING"})
    litm_input_errors = sum(r["status"] != "PASS" for r in litm_input_rows)
    check("LITM raw/frozen documents, gold position, references, and template", 1500, litm_input_errors + abs(1500 - len(litm_input_rows)) + len(raw_duplicates), "500 questions and three raw positions", "data_sources.json:litm; frozen-main80/litm.json")
    changed, minimum_examples = [], {}
    change_counts = collections.defaultdict(lambda: {"records": 0, "changed": 0, "multiline": 0, "question_ids": set()})
    for row in sorted(litm_records, key=lambda r: (r["sample_id"], r["method"], r["position"])):
        refs = litm_by[(row["sample_id"], row["position"])]["answers"]
        full = em(row["answer"], refs)
        first_line = row["answer"].split("\n")[0].strip()
        first = em(first_line, refs)
        group = change_counts[(row["method"], row["position"])]
        group["records"] += 1
        group["multiline"] += "\n" in row["answer"]
        if full != first:
            group["changed"] += 1
            group["question_ids"].add(row["sample_id"])
            record = {"sample_id": row["sample_id"], "method": row["method"], "position": row["position"],
                      "attempts_line": row["_line"], "attempt": row["attempt"], "saved_score": row["score"],
                      "full_answer_em": full, "official_first_line_em": first, "difference": first - full,
                      "score_unit": "EM_0_1", "L": row["L"], "N": row["N"], "T": row["T"], "S": row["S"]}
            changed.append(record)
            minimum_examples.setdefault((row["method"], row["position"]), {**record, "answer": row["answer"], "first_line": first_line, "reference_answers": refs})
    change_summary = [{"method": method, "position": pos, "n_generations": g["records"], "multiline_generations": g["multiline"],
                       "changed_generations": g["changed"], "changed_questions": len(g["question_ids"]),
                       "changed_fraction": g["changed"] / g["records"], "unit": "count/fraction", "scope": "scoring-entrypoint difference; no method-quality comparison"}
                      for (method, pos), g in sorted(change_counts.items())]
    check("LITM official first-line entrypoint agreement", len(litm_records), len(changed), "full answer versus official first-line best_subspan_em", "snapshot/metrics.py:score(litm); official evaluate_qa_responses.py:get_metrics_for_example", "Preserve saved scores; publish a separate official first-line score during the later quality stage.")

    # Doc2Dial: target IDs, every history turn, and exact common template adaptation.
    print("Auditing Doc2Dial raw target IDs, history, references and cache requests", flush=True)
    raw_dialogues = load_json(root / config["doc2dial_dialogues"])
    raw_documents = load_json(root / config["doc2dial_documents"])
    raw_targets = {}
    for domain, docs in raw_dialogues["dial_data"].items():
        for docid, dialogues in docs.items():
            for dialogue in dialogues:
                for index, turn in enumerate(dialogue["turns"]):
                    if turn["role"] == "agent":
                        raw_targets[f'{dialogue["dial_id"]}_{turn["turn_id"] - 1}'] = (domain, docid, dialogue, turn, index)
    doc_input_rows, doc_history_conditions, retained_ungrounded = [], [], 0
    for row in doc:
        sid = row["sample_id"]
        if sid not in raw_targets:
            issue("doc2dial_missing_raw_target_ID", sample_id=sid)
            continue
        domain, docid, dialogue, target, index = raw_targets[sid]
        previous = dialogue["turns"][:index]
        expected_history = [{k: t[k] for k in ("turn_id", "role", "utterance")} for t in previous]
        ungrounded = sum(not t["references"] for t in previous)
        retained_ungrounded += ungrounded
        expected_suffix = "\t" + "\t".join([tag("last_turn", previous[-1]["utterance"])] + [tag(t["role"], t["utterance"]) for t in reversed(previous)]) + CHAT_END
        expected_prefix = CHAT_START + tag("title", docid) + "\t<doc_context>\t"
        expected_source = {"domain": domain, "doc_id": docid, "dial_id": dialogue["dial_id"], "target_turn": target["turn_id"]}
        flags = {
            "grounded_agent_target": target["role"] == "agent" and bool(target["references"]),
            "source_and_target_ID_match": row["source"] == expected_source,
            "preceding_turn_ID_match": previous[-1]["turn_id"] == target["turn_id"] - 1,
            "all_previous_history_match": row["history"] == expected_history and row["history_turn_ids"] == [t["turn_id"] for t in previous],
            "adapted_reference_match": row["answers"] == [text2line(target["utterance"])],
            "raw_reference_13a_tokens_match": tokens_13a(target["utterance"]) == tokens_13a(row["answers"][0]),
            "full_document_match": row["document"] == text2line(raw_documents["doc_data"][domain][docid]["doc_text"]),
            "common_template_exact_match": row["prefix"] == expected_prefix and row["suffix"] == expected_suffix,
        }
        doc_input_rows.append({"sample_id": sid, **expected_source, "history_turns": len(previous), "ungrounded_history_turn_occurrences": ungrounded,
                               "last_turn_role": previous[-1]["role"], **flags, "status": "PASS" if all(flags.values()) else "FINDING"})
        repeated_turns = [t["turn_id"] for t in previous if text2line(t["utterance"]) == row["answers"][0]]
        if previous[-1]["role"] != "user" or repeated_turns:
            latest_user = next((t for t in reversed(previous) if t["role"] == "user"), None)
            doc_history_conditions.append({"sample_id": sid, **expected_source,
                                           "last_turn_role": previous[-1]["role"], "last_turn_id": previous[-1]["turn_id"],
                                           "consecutive_agent_target": previous[-1]["role"] == "agent",
                                           "reference_exactly_repeated_in_history": bool(repeated_turns),
                                           "matching_history_turn_ids": repeated_turns,
                                           "previous_utterance": previous[-1]["utterance"],
                                           "target_reference": row["answers"][0], "latest_user_turn": latest_user,
                                           "handling": "preserve frozen target; annotate original dialogue condition"})
    doc_input_errors = sum(r["status"] != "PASS" for r in doc_input_rows)
    check("Doc2Dial original target ID, complete history, template and reference", 500, doc_input_errors + abs(500 - len(doc_input_rows)), "500 frozen target turns", "data_sources.json:doc2dial_dialogues,doc2dial_documents; frozen-main80/doc2dial.json")
    doc_records = [r for r in successes if r["key"]["stage"] == "doc2dial"]
    doc_condition = collections.defaultdict(list)
    for row in doc_records:
        doc_condition[(row["sample_id"], row["method"], row["cache_mode"])].append(row)
    doc_pair_rows, hot_rows = [], []
    doc_pair_errors = 0
    for sid in sorted(doc_by_id):
        counts = [len(doc_condition[(sid, m, c)]) for m in METHODS for c in ("cold", "hot")]
        errors = sum(n != 1 for n in counts)
        doc_pair_errors += errors
        doc_pair_rows.append({"sample_id": sid, "expected_conditions": 6, "successful_records": sum(counts),
                              "missing_conditions": sum(n == 0 for n in counts), "duplicate_conditions": sum(n > 1 for n in counts),
                              "unit": "condition", "status": "PASS" if not errors else "FINDING"})
        for method in METHODS:
            cold_rows, warm_rows = doc_condition[(sid, method, "cold")], doc_condition[(sid, method, "hot")]
            if len(cold_rows) != 1 or len(warm_rows) != 1:
                continue
            cold, hot = cold_rows[0], warm_rows[0]
            event_lists = []
            for row in (cold, hot):
                k = (identity(row["key"]), row["attempt"])
                event_lists.extend((starts[k], ends[k]))
            events_complete = all(len(events) == 1 for events in event_lists)
            event_order = False
            if events_complete:
                times = [dt.datetime.fromisoformat(events[0]["at"]) for events in event_lists]
                event_order = times[0] <= times[1] <= times[2] <= times[3]
            cold_key, hot_key = dict(cold["key"]), dict(hot["key"])
            cold_key.pop("cache"); hot_key.pop("cache")
            ape_hook = method.startswith("ape_")
            hot_flags = {
                "same_task_except_cache": cold_key == hot_key,
                "same_output_token_ids": cold["output_token_ids"] == hot["output_token_ids"],
                "same_answer_and_reference": cold["answer"] == hot["answer"] and cold["reference_answers"] == hot["reference_answers"],
                "cold_miss_hot_hit": not cold["cache_hit"] and hot["cache_hit"],
                "request_event_pairs_complete": events_complete,
                "cold_then_hot_event_order": event_order,
                "hot_document_build_zero": hot["cache_build_seconds"] == 0,
                "same_actual_input_config": all(cold.get(k) == hot.get(k) for k in ("L", "N", "T", "S", "block_lengths", "prefix_tokens", "request_tokens")),
                "hot_output_token_count_match": hot["output_tokens"] == len(hot["output_token_ids"]),
                "APE_hot_query_calls_match_output_tokens": hot["stage_calls"]["query"] == hot["output_tokens"] if ape_hook else True,
                "APE_hot_no_prefix_context_calls": hot["stage_calls"]["prefix"] == hot["stage_calls"]["context"] == 0 if ape_hook else True,
            }
            hot_rows.append({"sample_id": sid, "method": method, "cold_attempts_line": cold["_line"], "hot_attempts_line": hot["_line"],
                             "cold_start_event_line": event_lists[0][0]["_line"] if event_lists[0] else None,
                             "hot_start_event_line": event_lists[2][0]["_line"] if event_lists[2] else None,
                             "hot_query_calls": hot["stage_calls"]["query"], "hot_output_tokens": hot["output_tokens"],
                             "query_hook_scope": "APE-only; sequential hook remains zero", "hot_timing_condition": hot["timing_condition"],
                             "resume_cache_rebuild_seconds": hot.get("resume_cache_rebuild_seconds", 0), **hot_flags,
                             "status": "PASS" if all(hot_flags.values()) else "FINDING"})
    check("Doc2Dial 500 targets x 3 methods x 2 cache conditions", 3000, doc_pair_errors + abs(len(doc_by_id) - 500), "paired cold/hot requests", "attempts.jsonl")
    hot_failures = sum(row["status"] != "PASS" for row in hot_rows)
    check("Doc2Dial hot cache and repeated-generation evidence", 1500, hot_failures + abs(1500 - len(hot_rows)), "same target/method cold-hot pairs", "events.jsonl; snapshot/methods.py:run; snapshot/worker.py:execute; attempts.jsonl:stage_calls,output_token_ids")

    # Score aggregation consistency only; no method-level quality results are emitted.
    summary = load_json(run / "summary.json")
    corpus_rows = []
    for saved in summary["doc2dial_paired"]["conditions"]:
        method, cache = saved["method"], saved["cache_mode"]
        rows = [r for r in doc_records if r["method"] == method and r["cache_mode"] == cache]
        vectors = [bleu_vectors[r["_line"]] for r in rows]
        stats = add_stats(vectors)
        reproduced = bleu_from_stats(stats)
        original = saved["corpus_bleu"]
        corpus_rows.append({"method": method, "cache_mode": cache, "n_requests": len(rows),
                            "absolute_server_difference": abs(reproduced - original["score"]), "tolerance": TOLERANCE,
                            "recorded_signature": original["signature"], "expected_server_signature": SIGNATURE,
                            "signature_match": original["signature"] == SIGNATURE,
                            "system_length": stats[0], "reference_length": stats[1], "correct_1_4": stats[2:6], "total_1_4": stats[6:10],
                            "unit": "BLEU point difference / token and ngram counts", "scope": "scoring consistency, method quality withheld",
                            "status": "PASS" if abs(reproduced - original["score"]) <= TOLERANCE and original["signature"] == SIGNATURE else "FINDING"})
    check("Doc2Dial corpus sufficient-statistics agreement with server", 6, sum(r["status"] != "PASS" for r in corpus_rows), "six saved method/cache aggregates", "summary.json:doc2dial_paired.conditions; independently joined requests")

    # Two upstream examples/probes with external or hand-derived expected values.
    fixtures = []
    preds = ["hello there general kenobi", "foo bar foobar"]
    refs = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
    value = bleu_from_stats(add_stats(bleu_stats(p, r) for p, r in zip(preds, refs)))
    fixtures.append({"fixture": "HF 1.3.0 official multi-reference corpus example", "expected": 100.0, "actual": value,
                     "absolute_difference": abs(value - 100.0), "tolerance": TOLERANCE, "source": SOURCES["historical_hf_defaults_and_example"]})
    # A single component of SacreBLEU's public example; expected value hand-derived.
    value = bleu_from_stats(bleu_stats("The dog bit the man.", ["The dog had bit the man."]))
    expected = 100 * math.exp(1 - 7 / 6) * (1 * (4 / 5) * (2 / 4) * (1 / 3)) ** 0.25
    fixtures.append({"fixture": "upstream example component; independently derived counts and BP", "expected": expected, "actual": value,
                     "absolute_difference": abs(value - expected), "tolerance": TOLERANCE, "source": SOURCES["sacrebleu_example_component"],
                     "expected_stats": [6, 7, 6, 4, 2, 1, 6, 5, 4, 3]})
    check("upstream BLEU examples and independent component arithmetic", len(fixtures), sum(r["absolute_difference"] > TOLERANCE for r in fixtures), "ported 13a/exp scorer", "scoring_official_fixtures.csv")

    write_csv(tables / "scoring_checks.csv", checks)
    write_csv(tables / "scoring_definition_consistency.csv", definition_rows)
    write_csv(tables / "scoring_saved_value_differences.csv", score_differences, ("line", "dataset", "sample_id", "metric", "saved", "recomputed", "difference"))
    write_csv(tables / "scoring_reference_join_issues.csv", ref_issues, ("line", "sample_id", "dataset", "kind"))
    write_csv(tables / "scoring_invalid_values.csv", invalid, ("line", "sample_id", "kind"))
    write_csv(tables / "scoring_other_issues.csv", issues, ("kind", "line", "sample_id"))
    write_csv(tables / "scoring_litm_pairing.csv", pairing_rows)
    write_csv(tables / "scoring_litm_input_checks.csv", litm_input_rows)
    write_csv(tables / "scoring_litm_first_line_impact.csv", change_summary)
    write_csv(tables / "scoring_litm_first_line_changed_IDs.csv", changed, ("sample_id", "method", "position", "attempts_line"))
    write_csv(tables / "scoring_litm_first_line_minimal_evidence.csv", minimum_examples.values(), ("sample_id", "answer", "reference_answers"))
    write_csv(tables / "scoring_doc2dial_input_checks.csv", doc_input_rows)
    write_csv(tables / "scoring_doc2dial_history_condition_evidence.csv", doc_history_conditions,
              ("sample_id", "last_turn_role", "consecutive_agent_target", "reference_exactly_repeated_in_history"))
    write_csv(tables / "scoring_doc2dial_pairing.csv", doc_pair_rows)
    write_csv(tables / "scoring_doc2dial_hot_generation_evidence.csv", hot_rows)
    write_csv(tables / "scoring_doc2dial_corpus_consistency.csv", corpus_rows)
    write_csv(tables / "scoring_official_fixtures.csv", fixtures)
    changed_questions = len({r["sample_id"] for r in changed})
    metadata = {"scope": "stage 1 input/reference/scoring audit only", "campaign": CAMPAIGN,
                "research_status": load_json(run / "delivery.json").get("status"), "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "dependencies": "Python standard library only; SacreBLEU not installed or imported",
                "randomness": "none; no resampling", "n_physical_records": len(parsed), "status_counts": dict(collections.Counter(r["status"] for r in parsed)),
                "n_litm_questions": len(litm_ids), "n_litm_generations": len(litm_records), "litm_changed_generations": len(changed),
                "litm_changed_questions": changed_questions, "litm_minimal_evidence_rows": len(minimum_examples),
                "n_doc2dial_targets": len(doc), "n_doc2dial_generations": len(doc_records), "n_cold_hot_pairs": len(hot_rows),
                "doc2dial_unique_dialogues": len({r["source"]["dial_id"] for r in doc}),
                "doc2dial_unique_documents": len({(r["source"]["domain"], r["source"]["doc_id"]) for r in doc}),
                "doc2dial_consecutive_agent_targets": sum(r["consecutive_agent_target"] for r in doc_history_conditions),
                "doc2dial_reference_repeated_in_history_targets": sum(r["reference_exactly_repeated_in_history"] for r in doc_history_conditions),
                "doc2dial_input_issue_count": doc_input_errors,
                "litm_input_issue_count": litm_input_errors,
                "ungrounded_history_turn_occurrences_retained": retained_ungrounded,
                "doc2dial_resume_rebuild_events": sum(e.get("key", {}).get("stage") == "doc2dial" for e in rebuilds),
                "score_mismatch_count": len(score_differences), "score_max_abs_difference": max(g["max_abs_difference"] for g in score_groups.values()),
                "reference_join_issue_count": len(ref_issues), "invalid_value_count": len(invalid), "other_issue_count": len(issues),
                "hot_evidence_failed_pairs": hot_failures, "sources": SOURCES}
    (reports / "STAGE1_SCORING_METADATA.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 第1阶段：LITM、Doc2Dial 与评分实现审核", "", "研究状态：SUBMITTED。", "",
             "本次仅审核输入、配对、参考连接、评分定义及热请求生成记录。正式方法质量比较、时延比较、置信区间及图表留待后续阶段。", "",
             "## 1 执行与证据", "", f"运行环境：`{sys.executable}`，Python `{platform.python_version()}`；仅用标准库，SacreBLEU 未安装或导入。本阶段无随机抽样。",
             "", "```powershell", r"C:\Users\Osheepne\anaconda3\python.exe -B pc-analysis-output/scripts/audit_stage1_scoring.py", "```", "",
             f"输入：`{root}`，campaign `{CAMPAIGN}`。我们读取 {len(parsed):,} 条物理生成记录，使用 `snapshot/APE/main80/` 解释实际执行，按冻结 ID 连接参考。`logical_labels` 保留为生成标签，同一物理生成只审核一次。", "",
             "## 2 已发现的 LITM 评分入口差异", "",
             f"campaign 将完整答案传入 `best_subspan_em`；官方评测入口先取第一行并去首尾空白，再做规范化子串匹配。专项复算发现 {len(changed):,}/{len(litm_records):,} 条生成的评分发生变化，涉及 {changed_questions} 个问题。受影响范围为 LITM 的逐题 EM、位置汇总、总体汇总以及后续方法配对比较。原始答案及保存分数保留；后续质量分析需另列官方首行 EM，并保留全文口径作为审核对照。",
             "", "|方法|位置|生成数|首行规则改变评分的生成数|", "|---|---:|---:|---:|"]
    for r in change_summary:
        lines.append(f'|{r["method"]}|{r["position"]}|{r["n_generations"]}|{r["changed_generations"]}|')
    lines += ["", "本表描述评分入口的影响范围。全部受影响 ID 和源行号见 `tables/scoring_litm_first_line_changed_IDs.csv`；每个方法／位置取最小样本 ID 形成一条原答案证据，共 " + str(len(minimum_examples)) + " 条，见 `tables/scoring_litm_first_line_minimal_evidence.csv`。", "",
              "## 3 输入、配对与参考连接", "",
              f"LITM 读取 {len(litm_ids)} 个问题、三种位置和三方法，共 {len(litm_records):,} 条成功生成。我们从原始压缩文件流式读取选中问题，核对20篇文档的多重集合、唯一 gold 及位置、冻结文档全文、问题与参考答案，以及共同输入模板。输入核对异常 {litm_input_errors} 条；方法／位置配对异常 {litm_pair_errors} 项。", "",
              f"Doc2Dial 读取 {len(doc)} 个目标及 {len(doc_records):,} 条成功生成，目标分布于 {metadata['doc2dial_unique_dialogues']} 个对话和 {metadata['doc2dial_unique_documents']} 篇资料。我们以官方 `{{dial_id}}_{{agent_turn_id-1}}` 连接原始目标，核对grounded agent角色、紧邻前一轮的ID与角色、每个前序历史轮次、完整资料及参考答案。输入／参考异常 {doc_input_errors} 条；六条件配对异常 {doc_pair_errors} 项。", "",
              f"实际对话条件：{metadata['doc2dial_consecutive_agent_targets']}/500 个目标的紧邻前一轮也是agent，因而模板last_turn在这些目标中是前一个agent发言；它们共对应 {metadata['doc2dial_consecutive_agent_targets'] * 6} 条生成。另有 {metadata['doc2dial_reference_repeated_in_history_targets']}/500 个目标参考在前序历史中出现完全相同的text2line文本，涉及 {metadata['doc2dial_reference_repeated_in_history_targets'] * 6} 条生成。官方参考ID连接及原模板按agent轮次处理，均保留这些原始目标。后续分析须标记连续agent与历史重复条件，并按对话／资料处理相关性。最小原文及匹配轮次见 `scoring_doc2dial_history_condition_evidence.csv`，其中样本 `22198e14bb6cd2a93ec1f6540101934b_2` 的原始目标和上一agent文本均为`gh`、最近user文本为`hg`。", "",
              f"选中目标历史中无grounding前序轮次的出现次数为 {retained_ungrounded}，同一历史轮次随多个目标出现时分别计次。" + ("这500个目标的所有前序轮次均带references，完整历史策略与旧脚本筛选后的历史内容一致。" if not retained_ungrounded else "冻结输入保留这些前序轮次。") + "官方 `text2line`、标签、last_turn 与逆序历史继续采用；本次将全文资料放在历史前面、保留全部前序历史并加入统一 Llama chat envelope。官方历史模板先跳过无references轮次，再累计历史，全文资料追加在历史后面。上述适配写入冻结记录，三方法共享。目标参考经 text2line 转换，其13a分词与原始目标 utterance 逐项对照。", "",
              "## 4 保存评分与官方定义", "",
              "LongBench：小写、去ASCII标点、去冠词a/an/the、合并空白，按词袋重叠计算F1，多参考取最大。我们以独立实现复算，同时从包内官方源仅加载三个纯评分函数作逐条对照。F1／EM原值范围为0–1，Doc2Dial BLEU为0–100。", "",
              f"保存逐条分数与其campaign定义的复算差异超过 {TOLERANCE:g} 的记录共 {len(score_differences)} 条；最大绝对数值差 {metadata['score_max_abs_difference']:.3g}。冻结参考连接异常 {len(ref_issues)} 条，非法分数／缺失输入检查异常 {len(invalid)} 条。明细按数据集和阶段记录于 `scoring_definition_consistency.csv`，只展示一致性和差值。", "",
              "Doc2Dial逐条 `score` 使用单条输入的 corpus_bleu 默认设置；汇总另调用 corpus_score。因此审核对逐条和语料聚合分别核对。独立实现采用13a分词、1–4gram截断匹配、exp平滑、mixed case和effective order off，先累加充分统计量，再计算语料BLEU及长度惩罚。我们验证官方HF例子与SacreBLEU例子中单句组件的独立手算，并对照全部保存逐条分数和六个服务器语料聚合。", "",
              f"服务器签名为 `{SIGNATURE}`；当前实现是独立标准库复算实现，签名用于对照服务器设置。官方历史环境记录SacreBLEU1.4.3、datasets1.3.0，本campaign记录SacreBLEU2.4.3。实际设置及六组充分统计／绝对差保存在 `scoring_doc2dial_corpus_consistency.csv`。", "",
              "## 5 热请求重新生成证据", "",
              f"我们核对 {len(hot_rows):,} 个目标／方法冷热请求对，联合检查独立request_started/ended事件及时间顺序、cold miss/hot hit、相同资料与配置、输出token一致性和APE query调用。未通过联合检查的请求对 {hot_failures} 个。Doc2Dial缓存恢复事件 {metadata['doc2dial_resume_rebuild_events']} 个。", "",
              "实际执行代码每次将history/current request的suffix重新forward，从当前logits重新逐token生成；热缓存只复用资料KV。APE热请求prefix/context调用为0，query调用与本次输出token数对应。顺序方法的stage hook按代码保持0，其重复生成链路由统一run循环和独立请求事件登记。冷热输出token相同另行作为相同贪心请求的一致性检查，证据表保留源记录及事件行号。恢复重建成本的字段及代码已登记，计时比较在后续阶段进行。", "",
              "## 6 本阶段结论与交接", "",
              "本阶段交付评分差异和数据连接审核。LITM后续质量分析须采用官方首行处理，并明确原campaign全文计分造成的影响范围。输入及评分一致性检查结果均列入 `scoring_checks.csv`；存在异常时按具体ID／源行号复核相关范围。研究状态保持SUBMITTED。", "",
              "## 7 官方来源与本地定位", ""]
    for title, url in SOURCES.items():
        lines.append(f"- [{title}]({url})")
    lines += ["", "本地执行链路：`snapshot/APE/main80/metrics.py`、`freeze.py`、`methods.py`、`worker.py`、`report.py`；官方保存源：`datasets/raw-main80/references/longbench_metrics.py.txt` 与 `litm_metrics.py.txt`。表中 `attempts_line`、事件行号均为UTF-8原文件一基行号。", ""]
    (reports / "STAGE1_SCORING.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "sources"}, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--restored", type=Path, default=repo / "restored")
    args = parser.parse_args()
    audit(args.restored.resolve(), repo / "pc-analysis-output/stages/01_audit")


if __name__ == "__main__":
    main()
