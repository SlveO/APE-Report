# Development supplement semantic review

Role: `EXECUTOR`. Authorization: `U_MAIN80`. Status: `SUBMITTED`; PLANNER review and final data-freeze decisions remain pending.

Read all 88 target questions/reference-answer lists, every ordered proposed selection, and the complete text of all 395 catalog passages. Shared passages were read once, then assessed against each target selecting them. No model output or score was available or used.

The 7,533 proposed target–paragraph pairs partition into 7,281 retained and 252 rejected pairs. Eighty-seven targets retain reviewed passages. Five targets have rejections:

| Target index | Target | Retained / rejected passages | Retained supplied interval tokens | Reason |
|---|---|---:|---:|---|
| 0 | `qasper:b805e336d2e8cce895100cfde3a536e632ddd5296ddece21` | 217 / 23 | 11,441 | Body/appearance taunts and gendered treatment can supply indirect personal-attack/sexism cues. |
| 3 | `qasper:c6464e3b2dbf1c8412496fdef56cafcccd0ccb9dd1937886` | 241 / 10 | 12,589 | Visual/gender portrayals, stereotypes and a racial insult are too close to the broad bias/inference question. |
| 6 | `qasper:901f735b1582acacb606002ab77c6c7a3fe6017d38349aa2` | 219 / 1 | 12,415 | A clinical-diagnosis/laboratory-results biography is withheld for the EHR-extraction question. |
| 27 | `qasper:df3b5e07ec132472a8fc5b7f30b6ce3d942c8488a3d8ff7c` | 0 / 216 | 0 | “Are the images from a specific domain?” has only Yes references and no domain description. The visual donor subjects cannot be verified against that unknown domain. |
| 51 | `qasper:a49d38fea6e6ffd631d227f17753ce20624d17bd8a04ee03` | 245 / 2 | 13,485 | Fictional despair/self-destruction and moping can supply symptom-like cues for depression annotation. |

These rejections express unresolved neutrality; they do not assert that the target papers discussed the donor examples. `review.json` gives concrete target-specific bases, exact approved IDs in original order, and a reason for every rejected ID. `contains_additional_answer_evidence=false` applies only to the actually read retained paragraphs; it is `null` for target 27, which retains none.

Retained passages have identifiable topic separation: encyclopedia donors concern song provenance/performances, fiction, East European geography/biography, banking and federalism; paper donors concern cyberbullying detection, image-caption bias, biomedical extraction, word representations and MOOC intervention. Per-target bases explain why these do not supply the requested research method/result or the specific encyclopedia entity relationship.

`validation.json` records direct input-content equality, ID/order/partition checks, metadata-only document/split checks, and no repeated normalized paragraph within a proposal. All catalog metadata says `explore`; no proposed donor document intersects its target's supplied document set. Upstream mapping provenance was not reconstructed by this review.

Token counts are sums of supplied interval lengths. Target 0 retains fewer than its proposed 12,181-token supplement requirement; target 27 has no approved supplement. PLANNER must recompute feasibility and mark affected constructions inapplicable where needed. This review adds no replacement donors and proves no final exact-length construction.

Deliveries: `review.json`, `input.snapshot.json`, `execution.json`, `validation.json`; the read/review/validation/submission scripts preserve the actual processing logic. No GPU/model run, download, install, score-based selection, hash/checksum, experiment-code or stage-document edit was performed.
