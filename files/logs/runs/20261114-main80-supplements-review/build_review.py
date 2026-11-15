"""Materialize the executor's manually reasoned semantic decisions, without scoring."""

import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
STARTED = datetime.now(timezone.utc).isoformat()

# Entries correspond to the explicitly read target order in input.snapshot.json.
# These are human/agent semantic judgments, not outputs of a lexical classifier.
BASIS = [
    "The target asks which cyberbullying topics the paper treats: personal attack, racism and sexism. Retained Wikipedia passages concern folk-song publication/recording history, fictional television events, East European places and biography, banking and federal government; they do not describe Internet harassment or the paper's topic taxonomy. Body-mocking lyrics and passages that evaluate women through gender roles are withheld because their semantic overlap with personal attack/sexism prevents confident neutrality.",
    "The target requires linking a Star Trek episode title to a nursery rhyme and its Roud index 821. Proposed donors discuss cyberbullying detection, Flickr30K descriptions and biomedical information extraction. They give neither that title-to-rhyme relationship nor folk-song index numbers.",
    "The target requires Dmitry Borisovich's birthplace and the Russian region containing it, Yaroslavl Oblast. The five donors describe cyberbullying harms, cross-platform language, classifier families and social-media datasets. They do not identify his birthplace or a Russian administrative location.",
    "The target concerns ways to discover biases/unwarranted inferences in image descriptions, with POS tagging, adjective analysis and Louvain clustering in the references. Retained song provenance/performance, fictional plots without the withheld bias examples, geography, banking and political history do not teach a procedure for inspecting an image-caption corpus. Passages involving gendered visual portrayals, appearance comparisons, racial insult, or stereotype-like judgments are withheld because they could give indirect examples relevant to that broad bias question.",
    "The target asks for the platform game led by Christian Whitehead, Sonic Mania. Donors are a cyberbullying introduction, a list of classifiers, and Formspring annotation counts. They contain no game development credits or platform-game identity.",
    "The target asks who awards the Adam Smith prize, University of Cambridge, through a colonial-policy clue. Its sole donor explains variation in cyberbullying language across social-media platforms; it supplies neither an award institution nor an Adam Smith/colonies relationship.",
    "The target asks whether the paper extracts information from electronic health records. Retained donors concern song/film narratives and the history, places, politics and banking of Eastern Europe; they do not describe electronic medical records or extracting clinical information. A real medical case involving neurological diagnosis and laboratory results is withheld conservatively because its clinical-record-like content is too close to the target's subject.",
    "The target compares Tongshanjiabu and Himalchuli by mountain height. Proposed passages concern cyberbullying models, biased image descriptions and biomedical NLP. Their numbers are study counts, model measures and publication references, not elevations or facts about either Himalayan peak.",
    "The target asks for the president of the group containing Kim Un-yong, Thomas Bach. The sole donor defines cyberbullying and mentions an advertising incident involving Arvida Bystrom. It gives no Olympic organization membership or presidency.",
    "The target compares a WSD/word-embedding approach against alternatives on SCWS, with GM_KL's better correlations in the reference. Retained encyclopedia passages discuss song versions, fictional television plots and real geographic/political/banking history. They give no word-sense-disambiguation experiment, embedding method comparison or SCWS correlation result.",
    "The target places the town hosting the 1960 FIBT championship in the Veneto region of northern Italy. Donors discuss NLP experiments and annotation biases; none locates that championship or Cortina d'Ampezzo. Biomedical or classifier statistics cannot provide the required town-to-region relation.",
    "The target asks which label signed the Live Life Loud band's artist, Fair Trade Services. Donors concern cyberbullying definitions, social-media corpora and classifiers. They contain no band signing or music-label relationship.",
    "The target's previous state-of-the-art system is hLSTM. Retained donors are nursery-rhyme/film history and geographic, biographic and financial encyclopedia text. They make no NLP baseline or neural-system comparison and cannot establish hLSTM as the earlier best system.",
    "The target asks which pope gave Marozia, Alberic II's mother, the title patricia of Rome: Pope John X. Proposed donors cover cyberbullying, image-caption bias, biomedical extraction and mathematical word representations. They have no medieval papal appointment, Marozia or Roman patricia title relationship.",
    "The target asks the record label of Permission to Fly's recording artist, Hollywood Records. Donors are cyberbullying research paragraphs, including dataset and neural-model descriptions, with no music artist contract or label history.",
    "The target identifies the paper's social-media input as Facebook status updates. Retained encyclopedia passages describe folk songs and film adaptations, towns, political biography, banks and orchestra websites. A named orchestra's website is not a social-media post corpus; none of these passages describes the paper's use of Facebook messages.",
    "The target locates the city containing Kellyville Ridge on the Cumberland Plain. Proposed cyberbullying research paragraphs describe online behavior, algorithms and corpora, without a Sydney/Blacktown geography relation or a plain containing that city.",
    "The target identifies the New Orleans Saints emblem through Ten High's manufacturer, fleur-de-lis. The sole donor concerns cyberbullying harms and an advertising incident. It contains no distillery headquarters, Saints emblem or heraldic symbol relationship.",
    "The target asks which learning methods are used for recognizing question entailment, logistic regression and neural networks. Retained song/film, biography and geographic passages do not describe question entailment or computational classifiers. Their stories and place descriptions cannot supply the requested RQE method list.",
    "The target asks the population of Hong Kong through Yau Ma Tei North, 7.2 million. The sole donor's percentages concern Internet users affected by cyberbullying; it gives neither a Hong Kong population nor the constituency-to-city relation.",
    "The target asks what MV Miralda's owner intended in April 2010, finding a buyer for Finnish operations. The two donors discuss cyberbullying harms and platform-specific vocabulary, without a ship, an owner, a sale announcement or Finnish operations.",
    "The target's references mention PPDB/book corpus and also contain unanswerable annotations. Retained donors discuss folk-song collections, fictional programs, Eastern European history and banks, without a research data selection statement. A literary collection mentioned in a nursery-rhyme history does not assert use of that collection as this paper's dataset, nor add a PPDB/book-corpus relation.",
    "The target asks the completion year of the arena hosting the 53rd NHL All-Star Game, 1998. Donors are cyberbullying model/data descriptions; any dates refer to online platforms or studies, not arena construction or NHL venues.",
    "The target asks the sibling of Frances Freeling Broderip, Tom/Thomas Hood. The sole donor defines cyberbullying and describes an advertisement; it gives no Hood family or sibling relationship.",
    "The target asks how words are matched before reordering, with CFILT-preorder/unanswerable references. Retained donors describe song variants and grammar, fictional plots, geography, biographies and banking. The apo koinou remark is a historical meter construction, not a computational alignment/reordering pipeline, and none supplies the CFILT procedure.",
    "The target asks the country containing Alsa Mall and Spencer Plaza, India. Proposed donors discuss cyberbullying and its classifiers/datasets. They contain no location of either mall or Chennai-to-country relationship.",
    "The target asks the institution attended by Pocahontas's John Smith voice actor, NIDA. The sole donor discusses platform differences in cyberbullying. It provides no actor biography or drama-school attendance.",
    "Unresolved target domain: the question only asks whether images are from a specific domain and both references are Yes; it never names that domain or its image source. Proposed Wikipedia material includes Dutch-girl illustrations, pictures of women in an advertisement and visual/film narratives. With no target-domain description, I cannot establish that those visual subjects are disjoint, and cannot certify the proposal as evidence-neutral. The entire proposed selection is withheld pending domain evidence from the target paper.",
    "The target asks the ethnic origins of Santeria through Migene Gonzalez-Wippler, West African descendants. The three donors define cyberbullying, discuss its detection bottlenecks and list classifiers. Their discussion of racism does not give Santeria's origin, adherents or the expert's religious specialty.",
    "The target classifies the organization in Thursday Night Football as a sports league. The sole donor defines cyberbullying and describes its prevalence; it does not discuss football competitions or sports-league organization.",
    "The target asks for the research sample size, 29,794 Wikipedia articles and 2,794 arXiv papers. Retained donors concern individual folk songs, fictional programs and Eastern European places, people and banks. Their dates/populations/areas do not describe a sample of research articles or the paper's dataset size.",
    "The target asks the title held by the author of Det norske Theater's first production, Baron of Holberg. Proposed donors discuss NLP datasets, classifiers, extraction systems and word-vector models, without Norwegian theatrical production history, Ludvig Holberg or a baronial title.",
    "The target asks the league containing the operator of Al Janoub Stadium, Qatar Stars League. The two donors discuss cyberbullying and linguistic variation on social media. They contain no Qatar club, stadium operator or football league membership.",
    "The target asks whether pretrained embeddings are used. Retained encyclopedia donors describe folk songs/films, places, political biography and bank history. They contain no initialization, embedding vectors, pretrained representation or neural-training method, so cannot provide the requested method evidence.",
    "The target identifies the Kentucky rule-class city containing Plainview, Jeffersontown. The donors describe social-media corpora and NLP learning methods; no Kentucky neighborhood-to-city relation or municipal population 26,595 is supplied.",
    "The target follows Kraai River to Orange River and asks its source, Thaba Putsoa. Donors cover cyberbullying detection and image-description biases. They give no river tributary network or geographic headwater; mathematical/data terms in these texts do not describe the target watercourse.",
    "The target distinguishes the shrub Mimosa from Cryptocoryne. Donors concern cyberbullying, Flickr30K bias, biomedical extraction and introductory word representations. Biomedical discussions concern protein/gene/drug text extraction rather than botany and do not describe either plant's growth form.",
    "The target asks the role voiced by the Nothing Suits Me Like a Suit performer in Batman: Under the Red Hood. The two donors describe cyberbullying harms and online-platform language. They include no voice actor credits, Batman characters or the song's performer.",
    "The target asks for the proposed system's DL-PS/EC-MT/EC-UQ F1 values. Retained donors contain song history, television criticism and place/biographic facts. The television B+ rating and historic numeric facts are not NLP-system accuracy/F1 measurements on the named datasets and do not provide those scores.",
    "The target links The 50th Law to Socrates and the philosopher whose writings record him, Plato. The sole donor defines cyberbullying and discusses an advertisement. It has no ancient philosophy, Socrates-to-Plato relation or book anecdote.",
    "The target asks the Canadian party's federal leader through Pierre-Hugues Boisvenu, Andrew Scheer. The sole donor discusses cyberbullying harms and an advertising example. It has no Canadian party membership or leadership evidence.",
    "The target asks G. V. Prakash Kumar's first film score through Udhayam NH4, Veyyil. Proposed donors discuss cyberbullying, image captions, biomedical extraction and word-vector models. They contain no Tamil film soundtrack credits or composer career chronology.",
    "The target asks how many plague outbreaks occurred in the city where Titian died, 22. The two donors concern cyberbullying prevalence and language variation; their figures are online-behavior observations, not Venice plague chronology or a painter's death location.",
    "The target asks how training data were translated, using Apertium. Retained encyclopedia material contains foreign names/transliterations and song variants but no training-corpus translation process or machine-translation platform. It does not add an Apertium-based transformation of the target data.",
    "The target asks where the Dallas Cowboys' 1980s punter played college football, Arizona State University. Donors concern cyberbullying datasets/models and image-caption stereotypes, with no punter career or college-football affiliation. Generic references to student/athletic scenes are not the required biographical relation.",
    "The target identifies the Saints symbol via Blanton's manufacturer and headquarters city, fleur-de-lis. The two donors describe cyberbullying harms and social-media variation. They give no whiskey company headquarters, New Orleans sports emblem or heraldry.",
    "The target asks for system scores located in the Ens Test column of Table TABREF19. Retained song/film and geographical/biographical material has no ensemble-test result table. Numeric encyclopedia facts and a critic's episode grade do not supply the target paper's evaluation-column contents.",
    "The target asks the US Hot 100 peak of the song on CeCe Peniston's Finally, number five. Donors concern cyberbullying, image-caption bias and biomedical extraction. They contain no CeCe Peniston recording, chart placement or US Hot 100 history.",
    "The target asks for eight biomedical NER benchmark tasks, BC5CDR/NCBI/BC4CHEMD/BC2GM/JNLPBA/LINNAEUS/Species-800. Retained donors are song/film, geographic and historical encyclopedia passages. Their disease metaphor and health-biography snippets are not named-entity benchmarks and provide none of the eight-task evaluation setup.",
    "The target identifies the artist with Marvel Team-Up and Batman: Son of the Demon credits, Jerry Bingham. The sole donor defines cyberbullying and gives an advertising incident. It contains no comics artist or illustration-credit relation for those works.",
    "The target asks which watercourse receives the river on which Lostock Dam sits, Hunter River. Donors discuss cyberbullying research, image-caption bias and biomedical NLP. They have no Australian river flow direction, Paterson River mouth or dam geography.",
    "The target asks the depression annotation scheme: absence of evidence, depressed mood, sleep disturbance and fatigue/loss of energy. Retained donors give song provenance, fictional and historical/geographical facts, with no dataset-label definitions or psychiatric annotation process. Fictional despair/self-destruction and the Dutch doll's moping are withheld because those emotional examples could supply symptom-like hints for this broad annotation question.",
    "The target asks the broadcaster for whom the 2013 Silverstone winner commentates, BBC Formula One. Proposed donors cover NLP research, mathematical word representations and MOOC intervention. They give no motor-racing winner, commentator employment or BBC broadcast relation.",
    "The target asks FC Barcelona's 2009 series of achievements, continental treble. The sole donor describes social-media cyberbullying and its vocabulary. It has no Spanish football scorer, Barcelona season or trophy sequence.",
    "The target asks the Twitter dataset size, 1,873 conversation threads/about 14k tweets. Retained encyclopedia donors concern song/film, geographic and historical subjects. They describe no Twitter conversation collection and none of their counts refer to tweets or research threads.",
    "The target asks the animal category of the waterfowl at Hardley Flood, duck. Proposed donors concern NLP classifiers, image-caption annotation, biomedical extraction and mathematical word representations, plus MOOCs. They contain no Hardley Flood ecology, common pochard or taxonomic statement classifying its waterfowl; biomedical entity types and word-model equations are unrelated to that animal identity.",
    "The target asks the date Terry Twell's team last beat the relevant FA Cup winner, 1 December 2010. The two donors concern cyberbullying harms and platform-specific communication. They give no club match record, FA Cup result or football fixture chronology.",
    "The target asks for speech-system baselines such as LF-MMI, Seq2Seq, RNN-T and CTC/Gram-CTC. Retained donors concern folk songs, television fiction, place history, people and banks. Descriptions of singing or musical arrangements do not state acoustic-model baselines or speech-recognition experiments.",
    "The target asks the Miami-area home city of Primetime Race Group, Hollywood, Florida. The donors discuss cyberbullying models/datasets. They contain no racing organization headquarters or Florida metropolitan geography.",
    "The target asks the British commander in the Battle of New Orleans through the Mother-in-Law singer's birth city, Edward Pakenham. Its sole donor discusses cyberbullying research bottlenecks and target topics, without a battle, military commander or singer biography.",
    "The target asks for classification baselines including Maximum Entropy, SVM, Recursive/Convolutional Neural Networks and BERT. Retained encyclopedia passages cover songs/fiction, geography, history and banking, without describing computational classification experiments or those baseline architectures. Political or architectural comparisons in those passages are not classifier baselines.",
    "The target asks whether Werther and The Odyssey are both operas. The sole donor defines cyberbullying and describes an advertisement; it supplies no genre, composer or literary-form fact for either named work.",
    "The target asks the president of East Timor through the Indonesia-Timor Leste truth commission, Francisco Guterres. Donors concern cyberbullying detection and datasets, not state formation, Indonesia's commission or national political officeholders.",
    "The target asks the sentiment-classification benchmark, IMDb movie reviews. Retained donors include film/television appearances and plots but no labeled review corpus, sentiment task or use of IMDb as data. Film subject matter alone does not establish which review dataset this paper used.",
    "The target identifies the Francis Lawrence film with Rich Lee previsualizations, Constantine. The five donors concern cyberbullying definitions, methods and datasets, giving no director/previsualization credit or target film identity.",
    "The target asks the capital of a county bordering WAPL's county, Green Bay. The three donors discuss cyberbullying and research bottlenecks; no radio licensing location, Wisconsin county boundary or county-seat relation is supplied.",
    "The target asks whether NER is learned from both text and images. Retained encyclopedia donors contain song/fiction narratives, visual depictions and place/biography facts, but no named-entity labels, multimodal model, visual feature encoding or learning procedure. An illustration or advertisement depicted in an encyclopedia passage is not a statement that the target NER model learns from visual inputs.",
    "The target asks Noelle Scaggs's earlier band, The Rebirth, through Fitz and the Tantrums. Donors discuss NLP detection, image bias and biomedical extraction. They give no singer membership, frontwoman tenure or band career history.",
    "The target asks The Beach's filming location in Thailand through Pao Sarasin, Ko Phi Phi Leh. The two donors concern cyberbullying and online vocabulary, with no Thai geography, filming credits or politician birthplace.",
    "The target asks the improvement from WAP 0.688 to 0.690 for MDREA. Retained encyclopedia material concerns song/film, geography, biography and banking; it contains no attention-model result or WAP benchmark. Population, area, financial and historical values cannot establish that research comparison.",
    "The target asks the capital of the province containing Baraki Barak District, Puli Alam. Donors concern cyberbullying datasets/classifiers and image-description bias. They give no Afghan district/province relation or Logar capital.",
    "The target asks the number of plague outbreaks at Vivaldi's birthplace through La Silvia, 22. The two donors discuss cyberbullying prevalence and platform language, without Venice plague chronology or composer birthplace.",
    "The target asks how dual audio/text RNNs are fused, with a feed-forward network. Retained encyclopedia passages describe songs, voices in entertainment, geography and people/banks, but no audio feature sequence, dual-RNN encoder or neural fusion layer. Singing/character dialogue does not describe a multimodal learning architecture.",
    "The target distinguishes Keith Morris from Rome Ramirez by punk-band work. Donors discuss cyberbullying corpora and neural/classical models. They include no musician biography or punk-band membership for either person.",
    "The target asks who awards the Adam Smith prize through Lectures on Jurisprudence, University of Cambridge. The four donors describe cyberbullying platform issues, learning methods and datasets; they give no economics prize or awarding university.",
    "The target asks the conflict method's evaluation tasks, Quora duplicate-question detection and question ranking/Bing People Also Ask. Retained encyclopedia donors contain story conflicts and fictional duplication but no question-pair dataset, search-ranking task, Quora or Bing experiment. Narrative conflict/duplicates do not supply the research evaluation relationship.",
    "The target asks the birth year of Walter Dornberger, von Braun's collaborator in Operation Lighthouse, 1895. Donors cover cyberbullying, image annotation and biomedical extraction. Their dates concern studies/platforms or publication references rather than rocket-experiment participants and birth records.",
    "The target asks the parent organization of Lawrence Landweber's school, City University of New York. The two donors discuss cyberbullying and online communication. They provide no Landweber education or Brooklyn College institutional affiliation.",
    "The target asks classification evaluation metrics, precision, recall, F-measure/accuracy. Retained encyclopedia text concerns songs/fiction, geography and historical biography; it gives no classifier evaluation definitions or metric reporting rule. Counts and dates there do not measure prediction correctness.",
    "The target distinguishes which of Matthew Good Band and Switchfoot remains together. Donors concern cyberbullying models and image-caption bias. They do not provide either band's membership, dissolution or continued activity.",
    "The target asks which undertaking included the discoverer of unique atomic numbers, Manhattan Project. The sole donor describes cyberbullying vocabulary across social platforms; it contains no atomic physics, proton discovery or wartime scientific project membership.",
    "The target asks which summarization algorithms were tested, LSA/TextRank/LexRank/ILP. Retained encyclopedia donors describe songs, fictional programs and geographic/historical facts, without extractive summarization, graph ranking or an algorithm experiment list. Narrative prose itself does not provide the named summarizer evidence.",
    "The target asks the type of Dianne Morgan and Joe Wilkinson's group in Two Episodes of Mash, a deadpan sketch group. Donors discuss NLP classification, image-description analysis and biomedical information extraction. They have no British comedy act, performers or sketch-group identity.",
    "The target links Eutychides's birth city to Greek mythology. Donors concern cyberbullying research and its classifiers/corpora, with no Hellenistic sculptor, Sicyon or mythological genealogy. Research categories and online incidents cannot supply that city-to-mythology relation.",
    "The target asks what word subspaces can represent, word vectors and within-class contexts. Retained encyclopedia passages discuss song/fiction, places, people and banking; they do not define mathematical word representations, vector classes or subspace operations. Ordinary words appearing in stories are not a technical representation claim.",
    "The target asks whether Yut and Tsuro are both board games. Donors concern cyberbullying and image-description research. They give no rules, game medium or classification of either named game, despite generic mentions of a photographed sporting event.",
    "The target asks a county bordering the birthplace county of She Did It's performer, Cabarrus County. The sole donor describes cyberbullying platform language. It contains no singer birthplace or North Carolina county boundary.",
    "The target asks a county bordering the county associated with a Redan High School/state/city chain, Presque Isle County. The sole donor discusses cyberbullying detection bottlenecks, giving no school location, US city population rank or county-border relation.",
]

TAUNT = "Conservative rejection: this lyric mocks a boyfriend's body, appearance or hygiene, or depicts interpersonal humiliation/aggression. That can provide an indirect example of personal attack, one of the target's reference topics; I cannot confidently certify it as neutral for the cyberbullying-topic question. This does not claim that the original paper discussed this song."
GENDER = "Conservative rejection: the passage evaluates or portrays women/boys through gendered attractiveness, domestic roles or stereotyped treatment. That is semantically adjacent to sexism in the target's reference topics, so absence of an explicit keyword is insufficient to certify neutrality."
BIAS = "Conservative rejection: the passage supplies a concrete appearance/gender portrayal, racial insult, or an interpretation based on visual/behavioral categories. Those examples could support the broad question about biases and unwarranted inferences even though the passage does not name the reference POS/Louvain procedure; its neutrality is unresolved."
REJECTIONS = {
    0: {
        **{i: TAUNT for i in [16,17,26,40,43,54,55,56,57,58,59,62,64,68,72,78,83,123]},
        **{i: GENDER for i in [91,99,103,117,118]},
    },
    3: {i: BIAS for i in [83,91,99,103,117,118,123,304,311,312]},
    6: {321: "Conservative rejection: the paragraph reports hospitalization, neurological diagnosis, possible poisoning and blood toxicology results. It resembles clinical-record content, so I withhold it for a target explicitly asking about extraction from electronic health records, even though it is a biography rather than an extraction study."},
    27: 'all',
    51: {
        131: "Conservative rejection: fictional despair ending in self-destruction can provide an emotional/symptom-like cue near the target's depressed-mood annotation labels. The passage does not state an annotation scheme, but evidence neutrality for that broad annotation question is not sufficiently certain.",
        304: "Conservative rejection: the doll's emotional distress and moping can provide a low-mood-like example near the target's depression annotation labels. It is an entertainment account, not the paper's annotation scheme, but I cannot confidently rule out indirect answer cues.",
    },
}


def donor_topics(indexes):
    groups = [
        ('nursery-rhyme publication/lyrics, performance, adaptations and Star Trek fiction', lambda i: 0 <= i <= 132 or 301 <= i <= 312 or i in [327,328,329,330,331]),
        ('East European geography, political/biographic history, Cyprus banking and federalism', lambda i: 133 <= i <= 239 or 313 <= i <= 326 or i == 336),
        ('cyberbullying definitions, social-media datasets and computational classifiers', lambda i: 240 <= i <= 264 or i == 337),
        ('Flickr30K image descriptions, stereotype analysis and annotation methods', lambda i: 265 <= i <= 290 or i in [332,333]),
        ('biomedical entity, relation and event extraction', lambda i: 291 <= i <= 300 or i in [334,335] or 338 <= i <= 350),
        ('Gaussian-mixture word representations, KL divergence and NLP evaluation', lambda i: 351 <= i <= 389 or i == 394),
        ('MOOC discussion forums and instructor intervention prediction', lambda i: 390 <= i <= 393),
    ]
    return [name for name, predicate in groups if any(predicate(i) for i in indexes)]


data = json.loads((BASE / 'input.snapshot.json').read_text())
catalog = list(data['catalog'].items())
lookup = {cid: i for i, (cid, _) in enumerate(catalog)}
assert len(BASIS) == len(data['proposals']) == 88
assert len(catalog) == 395
review = []
for i, proposal in enumerate(data['proposals']):
    original = proposal['catalog_ids']
    specified = REJECTIONS.get(i, {})
    if specified == 'all':
        rejected = {lookup[cid]: "Target-domain uncertainty: references only say Yes and the target image domain is unspecified. The selected corpus includes visual/illustration/advertising/film subjects; without the target paper's domain evidence I cannot certify domain disjointness or absence of indirect answering evidence. Entire proposal withheld conservatively." for cid in original}
    else:
        rejected = specified
        assert all(catalog[j][0] in original for j in rejected), (i, list(rejected))
    approved = [cid for cid in original if lookup[cid] not in rejected]
    review.append({
        'target_sample': proposal['target_sample'],
        'target_index': i,
        'dataset': proposal['dataset'],
        'split': proposal['split'],
        'question': proposal['question'],
        'reference_answers': proposal['reference_answers'],
        'approved_catalog_ids': approved,
        'rejected_catalog_ids': [{'catalog_id': cid, 'catalog_index': lookup[cid], 'reason': rejected[lookup[cid]]} for cid in original if lookup[cid] in rejected],
        'contains_additional_answer_evidence': False if approved else None,
        'evidence_assessment_scope': 'Only the actually read retained paragraphs, interpreted against this target question and references. No assertion of neutrality is made for rejected paragraphs.' if approved else 'No retained paragraph is certified; target-domain ambiguity remains unresolved.',
        'basis': BASIS[i],
        'proposed_donor_topics': donor_topics([lookup[cid] for cid in original]),
        'proposed_paragraph_count': len(original),
        'approved_paragraph_count': len(approved),
        'rejected_paragraph_count': len(original) - len(approved),
        'proposed_token_count': sum(data['catalog'][cid]['token_interval'][1] - data['catalog'][cid]['token_interval'][0] for cid in original),
        'approved_token_count': sum(data['catalog'][cid]['token_interval'][1] - data['catalog'][cid]['token_interval'][0] for cid in approved),
        'reviewer_role': 'EXECUTOR',
        'status': 'SUBMITTED',
    })

review_path = BASE / 'review.json'
assert not review_path.exists(), 'Initial creation only; read current contents before any revision.'
review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')

record_path = BASE / 'execution.json'
record = json.loads(record_path.read_text())
record['commands'].append({
    'cmd': 'bash scripts/with_env.sh /home/shenxiaoyang/miniconda3/envs/ape/bin/python -B logs/runs/20261114-main80-supplements-review/build_review.py',
    'started_at': STARTED,
    'ended_at': datetime.now(timezone.utc).isoformat(),
    'exit_code': 0,
    'purpose': 'Materialize manually recorded semantic decisions; no model or score involved.',
})
record['semantic_review_completed_at'] = datetime.now(timezone.utc).isoformat()
record['reviewed_catalog_ranges'] = [[0,79],[80,159],[160,239],[240,270],[271,300],[301,337],[338,365],[366,394]]
record['reviewed_target_ranges'] = [[0,29],[30,59],[60,87]]
record['coverage_note'] = 'Ranges are inclusive and refer to insertion order in input.snapshot.json. Complete text was emitted in finite batches and actually read; semantic verdicts are recorded per selected target, not inferred from the read commands or lexical filters.'
record['summary'] = {
    'target_count': len(review),
    'catalog_paragraphs_read': len(catalog),
    'targets_with_retained_safe_paragraphs': sum(bool(r['approved_catalog_ids']) for r in review),
    'targets_with_rejections': sum(bool(r['rejected_catalog_ids']) for r in review),
    'targets_with_no_approved_paragraphs': [r['target_sample'] for r in review if not r['approved_catalog_ids']],
    'proposed_target_paragraph_pairs': sum(r['proposed_paragraph_count'] for r in review),
    'approved_target_paragraph_pairs': sum(r['approved_paragraph_count'] for r in review),
    'rejected_target_paragraph_pairs': sum(r['rejected_paragraph_count'] for r in review),
}
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(record['summary'], ensure_ascii=False, indent=2))
