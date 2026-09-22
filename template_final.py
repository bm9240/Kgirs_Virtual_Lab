import os
import math
import re
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT METADATA & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "exp_number": 6,
    "title": "Identify Graph Entities & Probabilistic Retrieval Performance",
    "learning_unit": "NAMED ENTITY EXTRACTION & PROBABILISTIC IR RANKING",
    "discipline": "Computer Science and Engineering",
    "subject": "Information Retrieval Systems",
    "objectives": [
        "Identify entities such as persons, organizations, locations, and domain concepts from unstructured text using dictionary and rule-based methods.",
        "Configure Okapi BM25 probabilistic parameters (term saturation k1, length normalization b) and analyze their impact on ranking.",
        "Compare Exact Keyword Matching against Okapi BM25 Probabilistic Ranking on identical queries side-by-side.",
        "Benchmark retrieval performance (Precision@K, Recall@K, F1@K, MRR) and generate downloadable PDF reports."
    ]
}

QUIZ_QUESTION_BANK = [
    {
        "id": 1,
        "question": "Which entity type correctly classifies 'Bletchley Park' and 'Stanford University'?",
        "options": [
            "A) PERSON (PER)",
            "B) ORGANIZATION / LOCATION (ORG/LOC)",
            "C) TEMPORAL (DATE)",
            "D) NUMERIC (QUANTITY)"
        ],
        "answer_index": 1,
        "explanation": "'Bletchley Park' and 'Stanford University' represent geographical facilities or academic institutions, classified as LOCATION or ORGANIZATION."
    },
    {
        "id": 2,
        "question": "What is the primary function of parameter k1 in the Okapi BM25 scoring model?",
        "options": [
            "A) Controls term frequency saturation (how quickly repeated word occurrences flatten in score impact)",
            "B) Deletes all stopwords from the corpus",
            "C) Sets the maximum number of retrieved documents",
            "D) Measures network bandwidth latency"
        ],
        "answer_index": 0,
        "explanation": "Parameter k1 controls term frequency saturation; higher values allow repeated terms to contribute longer before score growth flattens."
    },
    {
        "id": 3,
        "question": "When setting the BM25 length normalization parameter b = 0, the model:",
        "options": [
            "A) Completely ignores document length penalties (short and long documents are treated equally)",
            "B) Penalizes short documents twice as heavily",
            "C) Inverts the document ranking order",
            "D) Removes all entity tags from text"
        ],
        "answer_index": 0,
        "explanation": "Setting b = 0 turns off length normalization, removing score penalties for long documents relative to the average corpus length."
    },
    {
        "id": 4,
        "question": "Why does probabilistic entity ranking outperform exact keyword search in Information Retrieval?",
        "options": [
            "A) It scores entities by relevance while accounting for term saturation and document length normalization",
            "B) It converts all text to lower-level machine bytecode",
            "C) It guarantees returning every single document in the database",
            "D) It bypasses memory allocation entirely"
        ],
        "answer_index": 0,
        "explanation": "Probabilistic ranking scores entities using non-linear term saturation and length adjustments rather than naive binary matching."
    },
    {
        "id": 5,
        "question": "What does Mean Reciprocal Rank (MRR) evaluate in retrieval benchmarks?",
        "options": [
            "A) Total storage size of the corpus",
            "B) The reciprocal rank position (1 / rank) of the first relevant result returned",
            "C) Execution time of the Python script in seconds",
            "D) The percentage of unindexed stopwords"
        ],
        "answer_index": 1,
        "explanation": "MRR evaluates where the first correct entity hit appears (e.g., 1st rank = 1.0, 2nd rank = 0.5, 3rd rank = 0.33)."
    },
    {
        "id": 6,
        "question": "What is the key difference between dictionary-based and rule-based Named Entity Recognition?",
        "options": [
            "A) Dictionary lookup matches known names against a gazetteer; rule-based infers entity types from surface patterns (capitalization, suffixes)",
            "B) They are two names for the exact same algorithm",
            "C) Rule-based NER only works on numbers",
            "D) Dictionary-based NER requires no text input at all"
        ],
        "answer_index": 0,
        "explanation": "Dictionary lookup matches known names exactly; rule-based heuristics generalize to unseen names using patterns."
    },
    {
        "id": 7,
        "question": "In BM25, what does the Inverse Document Frequency (IDF) component down-weight?",
        "options": [
            "A) Terms that occur in almost every document, since they carry little discriminative value",
            "B) The length of the query string",
            "C) The number of entities extracted",
            "D) The font size of the rendered report"
        ],
        "answer_index": 0,
        "explanation": "IDF assigns lower weight to ubiquitous terms that appear across most documents and higher weight to rare, discriminative terms."
    },
    {
        "id": 8,
        "question": "A query returns 10 entities, 4 of which are relevant. If 3 of the top-5 results are relevant, what is Precision@5?",
        "options": [
            "A) 3/5 = 0.60",
            "B) 4/10 = 0.40",
            "C) 3/4 = 0.75",
            "D) 5/10 = 0.50"
        ],
        "answer_index": 0,
        "explanation": "Precision@K = (relevant items in top-K) / K = 3/5 = 0.60."
    },
    {
        "id": 9,
        "question": "Why can exact keyword matching fail even when a genuinely relevant entity is present in the text?",
        "options": [
            "A) It requires exact string matching, missing partial names, paraphrases, or term variants",
            "B) It always returns every entity regardless of query input",
            "C) It cannot process lower-case text",
            "D) It is slower than BM25 in every scenario"
        ],
        "answer_index": 0,
        "explanation": "Binary matching lacks term weighting or partial matching capabilities, failing when search terms do not match verbatim."
    },
    {
        "id": 10,
        "question": "If Recall@K increases while Precision@K decreases as K grows, what does this indicate?",
        "options": [
            "A) The classic Precision/Recall trade-off: widening the evaluation window captures more relevant items but admits more noise",
            "B) The ranking algorithm has encountered a critical error",
            "C) MRR must automatically increase",
            "D) The corpus size has dynamically changed"
        ],
        "answer_index": 0,
        "explanation": "Increasing K retrieves more candidate items (raising recall) while increasing the ratio of non-relevant items (lowering precision)."
    },
    {
        "id": 11,
        "question": "What does a named entity span contain?",
        "options": [
            "A) The exact start and end positions of an entity in the source text",
            "B) Only the entity's BM25 score",
            "C) The total number of documents in the corpus",
            "D) The quiz question number"
        ],
        "answer_index": 0,
        "explanation": "An entity span records where the entity begins and ends in the original text."
    },
    {
        "id": 12,
        "question": "Why is a gazetteer useful in named entity extraction?",
        "options": [
            "A) It stores known entity names and their types for direct lookup",
            "B) It randomly changes the entity type on each run",
            "C) It removes every capitalized word",
            "D) It calculates document length normalization"
        ],
        "answer_index": 0,
        "explanation": "A gazetteer is a curated dictionary of known names, such as people, organizations, and places."
    },
    {
        "id": 13,
        "question": "What is the main risk of relying only on capitalization rules for NER?",
        "options": [
            "A) Capitalized phrases can be ambiguous and may be incorrectly classified",
            "B) Capitalization rules always miss every person name",
            "C) They can only process numeric values",
            "D) They automatically compute MRR"
        ],
        "answer_index": 0,
        "explanation": "Capitalization is a useful signal, but it does not provide enough context to classify every phrase correctly."
    },
    {
        "id": 14,
        "question": "What happens to BM25's term-frequency contribution as tf becomes very large?",
        "options": [
            "A) It approaches a limit because of term saturation",
            "B) It grows without limit at a constant linear rate",
            "C) It becomes independent of the query",
            "D) It removes the term from the index"
        ],
        "answer_index": 0,
        "explanation": "BM25 uses a saturation curve so repeated occurrences provide diminishing additional benefit."
    },
    {
        "id": 15,
        "question": "What does a higher BM25 b value generally do to a document longer than average?",
        "options": [
            "A) Applies a stronger length normalization penalty",
            "B) Removes all IDF weighting",
            "C) Guarantees the document ranks first",
            "D) Converts the document into an entity"
        ],
        "answer_index": 0,
        "explanation": "A higher b gives document length more influence, penalizing long documents relative to the average."
    },
    {
        "id": 16,
        "question": "What does a high IDF value indicate about a query term?",
        "options": [
            "A) The term is relatively rare across the document collection",
            "B) The term appears in every document",
            "C) The term is always a stopword",
            "D) The term has zero query relevance"
        ],
        "answer_index": 0,
        "explanation": "Rare terms are more discriminative, so BM25 assigns them higher inverse document frequency weight."
    },
    {
        "id": 17,
        "question": "If no relevant result appears in the ranked list, what is the MRR value?",
        "options": [
            "A) 0.0",
            "B) 1.0",
            "C) The number of retrieved documents",
            "D) The average document length"
        ],
        "answer_index": 0,
        "explanation": "MRR is zero when the ranked results contain no relevant item."
    },
    {
        "id": 18,
        "question": "Why is F1 useful when evaluating a retrieval system?",
        "options": [
            "A) It combines precision and recall using their harmonic mean",
            "B) It measures only execution time",
            "C) It counts only extracted organizations",
            "D) It replaces the ranking algorithm"
        ],
        "answer_index": 0,
        "explanation": "F1 summarizes the balance between precision and recall, penalizing a low value in either measure."
    },
    {
        "id": 19,
        "question": "What does Precision@K measure?",
        "options": [
            "A) The fraction of the top-K results that are relevant",
            "B) The fraction of all relevant items found anywhere in the corpus",
            "C) The number of query terms in the dictionary",
            "D) The length of the longest document"
        ],
        "answer_index": 0,
        "explanation": "Precision@K measures the quality of the returned top-K set by counting its relevant results."
    },
    {
        "id": 20,
        "question": "What is the purpose of comparing exact matching and BM25 on the same entities?",
        "options": [
            "A) To isolate the effect of probabilistic ranking against a simple baseline",
            "B) To ensure both methods return identical scores",
            "C) To avoid measuring recall",
            "D) To remove entity extraction from the experiment"
        ],
        "answer_index": 0,
        "explanation": "Using the same extracted entities makes the comparison focus on ranking behavior rather than different input data."
    }
]

SAMPLE_PRESETS = {
    "Computing Pioneers (Alan Turing / Bletchley Park)": "Alan Turing worked at Bletchley Park in the United Kingdom to decipher the Enigma machine for Allied forces during World War II.",
    "Tech Leaders (Sundar Pichai / Google)": "Sundar Pichai is the CEO of Google and earned degrees from Stanford University and the Wharton School in the United States.",
    "Modern AI Research (Satya Nadella / Microsoft)": "Satya Nadella led Microsoft in Redmond to partner with OpenAI based in San Francisco to advance Artificial Intelligence.",
    "Scientific History (Marie Curie / Paris)": "Marie Curie conducted pioneering research on radioactivity at the University of Paris and won Nobel Prizes in Stockholm, Sweden.",
    "Internet Pioneers (Tim Berners-Lee / CERN)": "Tim Berners-Lee invented the World Wide Web at CERN in Geneva, Switzerland."
}

KNOWN_GAZETTEER = {
    "Alan Turing": "PERSON", "Sundar Pichai": "PERSON", "Satya Nadella": "PERSON",
    "Marie Curie": "PERSON", "Tim Berners-Lee": "PERSON", "Google": "ORGANIZATION",
    "Stanford University": "ORGANIZATION", "Microsoft": "ORGANIZATION", "OpenAI": "ORGANIZATION",
    "University of Paris": "ORGANIZATION", "CERN": "ORGANIZATION", "Bletchley Park": "LOCATION",
    "United Kingdom": "LOCATION", "San Francisco": "LOCATION", "Stockholm": "LOCATION",
    "Sweden": "LOCATION", "Geneva": "LOCATION", "Switzerland": "LOCATION", "Redmond": "LOCATION",
    "Wharton School": "ORGANIZATION", "United States": "LOCATION",
    "CEO": "ROLE_TITLE", "Artificial Intelligence": "CONCEPT", "World Wide Web": "CONCEPT",
    "Enigma machine": "CONCEPT", "Nobel Prizes": "AWARD_EVENT", "radioactivity": "CONCEPT"
}

ORG_SUFFIX_MARKERS = [
    "university", "institute", "college", "corporation", "corp", "inc", "ltd", "company",
    "organization", "laboratory", "labs", "school", "foundation", "council", "agency",
    "association", "department", "ministry", "committee"
]
LOCATION_SUFFIX_MARKERS = [
    "city", "island", "islands", "mountain", "mountains", "river", "ocean", "sea",
    "republic", "kingdom", "state", "province", "county", "valley", "desert", "bay"
]
STOPWORD_LEADS = {
    "the", "a", "an", "this", "that", "these", "those", "in", "on", "at", "of", "for",
    "and", "or", "but", "with", "to", "from", "by"
}


# ======================================================================================
# 2. HYBRID ENTITY EXTRACTION (DICTIONARY + RULE-BASED)
# ======================================================================================

def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def _classify_heuristic(phrase: str) -> str:
    lowered = phrase.lower()
    words = phrase.split()

    if any(marker in lowered for marker in ORG_SUFFIX_MARKERS):
        return "ORGANIZATION"
    if any(marker in lowered for marker in LOCATION_SUFFIX_MARKERS):
        return "LOCATION"
    if len(words) >= 2 and all(w[0].isupper() and w[1:].islower() for w in words if w.isalpha()):
        return "PERSON"
    return "CONCEPT"


def _dictionary_pass(text: str) -> tuple:
    entities = []
    seen_spans = set()
    for term in sorted(KNOWN_GAZETTEER, key=len, reverse=True):
        for match in re.finditer(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
            span = (match.start(), match.end())
            if any(start <= span[0] < end or start < span[1] <= end for start, end in seen_spans):
                continue
            seen_spans.add(span)
            entities.append({
                "entity": text[match.start():match.end()],
                "type": KNOWN_GAZETTEER[term],
                "source": "Dictionary",
                "start": match.start(),
                "end": match.end(),
                "context": "..." + text[max(0, match.start() - 20):min(len(text), match.end() + 20)].strip() + "..."
            })
    return entities, seen_spans


def _rule_based_pass(text: str, occupied_spans: set) -> list:
    entities = []
    pattern = re.compile(r'\b([A-Z][a-zA-Z\.]*(?:\s+(?:of|the|and)?\s*[A-Z][a-zA-Z\.]*){0,3})')
    for match in pattern.finditer(text):
        phrase = match.group(1).strip()
        words = phrase.split()
        while words and words[0].lower() in STOPWORD_LEADS:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words).rstrip(".,;:!?")
        if not phrase:
            continue
        start = text.find(phrase, match.start())
        if start == -1:
            continue
        end = start + len(phrase)
        span = (start, end)

        if any(s <= span[0] < e or s < span[1] <= e for s, e in occupied_spans):
            continue
        if len(phrase) < 3 or phrase.lower() in STOPWORD_LEADS:
            continue
        if len(words) == 1 and len(phrase) <= 3:
            continue

        occupied_spans.add(span)
        entities.append({
            "entity": phrase,
            "type": _classify_heuristic(phrase),
            "source": "Heuristic",
            "start": start,
            "end": end,
            "context": "..." + text[max(0, start - 20):min(len(text), end + 20)].strip() + "..."
        })
    return entities


def extract_entities_from_text(text: str) -> list:
    dict_entities, occupied = _dictionary_pass(text)
    rule_entities = _rule_based_pass(text, occupied)
    entities = dict_entities + rule_entities
    entities.sort(key=lambda item: item["start"])
    return entities


# ======================================================================================
# 3. PROBABILISTIC RANKING ENGINE
# ======================================================================================

def compute_bm25_score(query_tokens: list, doc_tokens: list, doc_len: int,
                       avg_doc_len: float, idf_dict: dict, k1: float, b: float) -> float:
    score = 0.0
    doc_tf = Counter(doc_tokens)
    for q in query_tokens:
        if q in doc_tf:
            tf = doc_tf[q]
            idf = idf_dict.get(q, 0.0)
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
            score += idf * (numerator / max(0.001, denominator))
    return round(score, 4)


def perform_dual_retrieval(entities: list, query_str: str, k1: float, b: float) -> dict:
    q_norm = " ".join(tokenize(query_str))

    exact_ranked = []
    for item in entities:
        ent_norm = " ".join(tokenize(item["entity"]))
        match = bool(q_norm) and (q_norm == ent_norm or q_norm in ent_norm or ent_norm in q_norm)
        exact_ranked.append({
            **item,
            "score": 1.0 if match else 0.0,
            "match_status": "Exact Match" if match else "No Match"
        })
    exact_ranked.sort(key=lambda x: (-x["score"], x["entity"].lower()))

    q_tokens = tokenize(query_str)
    candidate_docs = [f"{item['entity']} {item['type']} {item['context']}" for item in entities]
    doc_token_lists = [tokenize(doc) for doc in candidate_docs]
    avg_len = sum(len(t) for t in doc_token_lists) / max(1, len(doc_token_lists))

    df_counts = Counter()
    for tokens in doc_token_lists:
        for token in set(tokens):
            df_counts[token] += 1
    N = len(entities)
    idf = {
        token: math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        for token, df in df_counts.items()
    }

    bm25_ranked = []
    for item, tokens in zip(entities, doc_token_lists):
        score = compute_bm25_score(q_tokens, tokens, len(tokens), avg_len, idf, k1, b)
        bm25_ranked.append({**item, "score": score})
    bm25_ranked.sort(key=lambda x: (-x["score"], x["entity"].lower()))

    max_score = max((item["score"] for item in bm25_ranked), default=0.0)
    for item in bm25_ranked:
        rel = round(item["score"] / max_score, 4) if max_score else 0.0
        item["confidence"] = (
            "High Relevance" if rel >= 0.75 else
            "Moderate Relevance" if rel >= 0.40 else
            "Low Relevance" if rel >= 0.15 else "Non-Relevant"
        )

    return {"exact": exact_ranked, "bm25": bm25_ranked}


def compute_ir_metrics(ranked_entities: list, relevant_entities: set, k: int) -> dict:
    if not relevant_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0}

    top_k = ranked_entities[:k]
    hits = sum(1 for name in top_k if name in relevant_entities)
    precision = hits / max(1, k)
    recall = hits / len(relevant_entities)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mrr = 0.0
    for rank, name in enumerate(ranked_entities, start=1):
        if name in relevant_entities:
            mrr = 1.0 / rank
            break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr, 4)
    }


# ======================================================================================
# 4. PDF LAB REPORT & CERTIFICATE EXPORT
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Report - Exp {EXPERIMENT_CONFIG['exp_number']}", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "VIRTUAL LABORATORY EXPERIMENT REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 32, 190, 24, "FD")

    pdf.set_xy(14, 34)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(58, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(48, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 42)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(58, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Evaluation Score:", 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(48, 5, f"Quiz: {quiz_score}/{quiz_total}", 1)

    pdf.ln(16)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Objectives & Learning Units", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.set_x(14)
        pdf.multi_cell(182, 5, f"- {obj}")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Performance", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))

        for c in cols:
            pdf.cell(col_w, 6, str(c)[:13], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, val_str[:13], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Critical Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The experimental trials demonstrated that Okapi BM25 probabilistic ranking consistently outperformed "
        "exact keyword matching across Precision@K, F1@K, and MRR metrics by accounting for term saturation "
        "and document length normalization."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Evaluator Signature", align="C")

    return bytes(pdf.output())


def generate_certificate_pdf(student_name: str, student_id: str, date_str: str,
                              quiz_score: int, quiz_total: int) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(1.2)
    pdf.rect(8, 8, 281, 194)
    pdf.set_draw_color(148, 163, 184)
    pdf.set_line_width(0.4)
    pdf.rect(12, 12, 273, 186)

    pdf.set_y(26)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "VIRTUAL LABORATORY SYSTEM", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 14, "CERTIFICATE OF COMPLETION", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "This certifies that", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "BI", 24)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 12, student_name or "Student Name", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, f"(Roll / ID: {student_id or 'N/A'})", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(51, 65, 85)
    body = (
        f"has successfully completed Experiment {EXPERIMENT_CONFIG['exp_number']}: "
        f"{EXPERIMENT_CONFIG['title']}, under the {EXPERIMENT_CONFIG['subject']} curriculum."
    )
    pdf.set_x(30)
    pdf.multi_cell(237, 7, body, align="C")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 8, f"Quiz Score Achieved: {quiz_score} / {quiz_total}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.set_xy(40, 168)
    pdf.cell(70, 6, date_str or datetime.now().strftime("%Y-%m-%d"), align="C")
    pdf.set_draw_color(148, 163, 184)
    pdf.line(40, 175, 110, 175)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.set_xy(40, 176)
    pdf.cell(70, 5, "Date", align="C")

    pdf.line(187, 175, 257, 175)
    pdf.set_xy(187, 176)
    pdf.cell(70, 5, "Instructor / Coordinator Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 5. VISUAL DESIGN SYSTEM
# ======================================================================================

def inject_global_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 17px !important;
  color: #0F172A !important;
  line-height: 1.65;
}

code, kbd, pre {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.87rem !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}
section[data-testid="stSidebar"] * {
  color: #F1F5F9 !important;
  font-size: 0.95rem !important;
}

/* ---------------------------------------------------------------- */
/* ANIMATION LAYER                                                   */
/* ---------------------------------------------------------------- */
@keyframes fadeInUp {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInLeft {
  0% { opacity: 0; transform: translateX(-14px); }
  100% { opacity: 1; transform: translateX(0); }
}
@keyframes popIn {
  0% { opacity: 0; transform: scale(0.85); }
  70% { opacity: 1; transform: scale(1.04); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes pulseRing {
  0% { box-shadow: 0 0 0 0 rgba(37,99,235,0.45); }
  70% { box-shadow: 0 0 0 9px rgba(37,99,235,0); }
  100% { box-shadow: 0 0 0 0 rgba(37,99,235,0); }
}
@keyframes dashFlow {
  0% { stroke-dashoffset: 80; }
  100% { stroke-dashoffset: 0; }
}
@keyframes fillBar {
  0% { width: 0%; }
}

/* Hero Header Card */
.hero-card {
  background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F3A57 100%);
  border-left: 5px solid #2563EB;
  border-radius: 14px;
  padding: 1.6rem 2rem;
  color: #F8FAFC !important;
  margin-bottom: 1.3rem;
  box-shadow: 0 10px 24px -6px rgba(15, 23, 42, 0.25);
  animation: fadeInUp 0.5s cubic-bezier(0.16,1,0.3,1) both;
}

.hero-card h2 {
  color: #60A5FA !important;
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  margin-bottom: 0.55rem !important;
  letter-spacing: -0.02em;
}

.hero-card p {
  color: #F1F5F9 !important;
  font-size: 1rem !important;
  line-height: 1.68 !important;
}

.badge-row { display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.9rem; }
.badge {
  display:inline-block; padding:0.3rem 0.8rem; border-radius:999px;
  background: rgba(96, 165, 250, 0.15); border: 1px solid rgba(96, 165, 250, 0.35);
  color:#93C5FD !important; font-size:0.78rem; font-weight:700; letter-spacing:0.03em;
  animation: fadeInUp 0.5s ease both;
}

/* Hover Cards */
.hover-card {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-radius: 12px;
  padding: 1.05rem 1.25rem;
  margin-bottom: 0.85rem;
  transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
  box-shadow: 0 2px 6px rgba(0,0,0,0.03);
  position: relative;
  animation: fadeInUp 0.45s ease both;
}

.hover-card:hover {
  transform: translateY(-3px);
  border-color: #2563EB;
  box-shadow: 0 10px 20px rgba(37, 99, 235, 0.13);
}

.hover-card h4 {
  color: #0F172A !important;
  font-size: 1.03rem !important;
  font-weight: 800 !important;
  margin: 0 0 0.35rem 0 !important;
}

.hover-card p {
  color: #334155 !important;
  font-size: 0.92rem !important;
  line-height: 1.55 !important;
  margin: 0 !important;
  font-weight: 500;
}

.hover-card .definition-reveal {
  background: #F0F9FF;
  border-left: 3px solid #0284C7;
  padding: 0.7rem 0.9rem;
  margin-top: 0.7rem;
  border-radius: 6px;
  font-size: 0.87rem;
  color: #0369A1 !important;
  display: none;
  font-weight: 600;
}

.hover-card:hover .definition-reveal { display: block; }

.section-banner {
  font-size: 1.28rem !important;
  font-weight: 800 !important;
  color: #0F172A !important;
  margin-top: 1.6rem !important;
  margin-bottom: 0.8rem !important;
  border-left: 4px solid #2563EB;
  padding-left: 0.7rem;
  letter-spacing: -0.02em;
  animation: fadeInLeft 0.45s ease both;
}

.math-container {
  background-color: #F8FAFC;
  border: 1px solid #CBD5E1;
  border-left: 4px solid #2563EB;
  border-radius: 8px;
  padding: 0.75rem 1.1rem;
  margin: 0.75rem 0;
  font-size: 0.95rem;
  font-weight: 700;
  color: #0F172A !important;
}

.tech-callout {
  background: #F0F9FF;
  border: 1px solid #BAE6FD;
  border-left: 4px solid #0284C7;
  border-radius: 10px;
  padding: 0.9rem 1.15rem;
  margin: 0.8rem 0 1.1rem 0;
  color: #0369A1 !important;
  font-size: 0.92rem;
  line-height: 1.62;
  font-weight: 500;
  animation: fadeInUp 0.45s ease both;
}
.tech-callout b { color: #0284C7 !important; font-weight: 700; }

.why-callout {
  background: linear-gradient(180deg, #FFFBEB 0%, #FEF3C7 100%);
  border: 1px solid #FDE68A;
  border-left: 4px solid #D97706;
  border-radius: 10px;
  padding: 0.9rem 1.15rem;
  margin: 0.85rem 0 1.2rem 0;
  color: #78350F !important;
  font-size: 0.92rem;
  line-height: 1.62;
  animation: fadeInUp 0.5s ease both;
}
.why-callout .why-label {
  display:inline-block; font-size:0.72rem; font-weight:800; letter-spacing:0.06em;
  text-transform:uppercase; color:#B45309 !important; margin-bottom:0.35rem;
}
.why-callout b { color: #92400E !important; }
.why-callout .highlight {
  background:#FDE68A; padding:0.03rem 0.35rem; border-radius:4px; font-weight:700; color:#78350F;
}

.chapter-title {
  display: flex; align-items: center; gap: 0.65rem;
  font-size: 1.22rem; font-weight: 800; color: #0F172A !important;
  margin-top: 1.5rem; margin-bottom: 0.3rem;
  letter-spacing: -0.02em;
  animation: fadeInLeft 0.4s ease both;
}

.chapter-num {
  display: flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; min-width: 32px; border-radius: 8px;
  background: linear-gradient(135deg, #2563EB, #1D4ED8);
  color: #FFFFFF !important; font-weight: 800; font-size: 0.95rem;
  animation: popIn 0.45s cubic-bezier(0.16,1,0.3,1) both;
}

.chapter-sub {
  color: #475569 !important; font-size: 0.9rem; font-weight: 600; margin: -0.05rem 0 0.9rem 2.55rem;
}

/* Horizontal Stepper Pipeline */
.stepper-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #F8FAFC;
  border: 1px solid #CBD5E1;
  border-radius: 12px;
  padding: 1rem 1.4rem;
  margin: 0.6rem 0 1.5rem 0;
}

.stepper-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  position: relative;
  flex: 1;
}

.stepper-badge {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 1rem;
  border: 2px solid #CBD5E1;
  background: #FFFFFF;
  color: #64748B;
  margin-bottom: 0.45rem;
  transition: all 0.3s ease;
}

.stepper-node.active .stepper-badge {
  background: #2563EB;
  border-color: #2563EB;
  color: #FFFFFF !important;
  animation: pulseRing 1.7s infinite;
}

.stepper-node.done .stepper-badge {
  background: #10B981;
  border-color: #10B981;
  color: #FFFFFF !important;
}

.stepper-text-title {
  font-size: 0.83rem;
  font-weight: 800;
  color: #0F172A !important;
  margin-bottom: 0.1rem;
}

.stepper-text-desc {
  font-size: 0.74rem;
  font-weight: 600;
  color: #64748B !important;
}

.stepper-connector {
  height: 3px;
  background: #CBD5E1;
  flex: 0.6;
  margin-top: -1.4rem;
  transition: background 0.3s ease;
}

.stepper-connector.done {
  background: #10B981;
}

.live-metric {
  background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px;
  padding: 0.9rem 1.1rem; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.03);
  animation: fadeInUp 0.4s ease both;
}
.live-metric .val { font-size: 1.55rem; font-weight: 900; color: #2563EB !important; font-family: 'JetBrains Mono', monospace; }
.live-metric .lbl { font-size: 0.74rem; font-weight: 800; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)


def render_phase_stepper(active_index: int, done_flags: list):
    labels = [
        ("Phase 1", "Entity Extraction"),
        ("Phase 2", "Parameter Tuning"),
        ("Phase 3", "Dual Retrieval"),
        ("Phase 4", "Evaluation")
    ]

    nodes_html = []
    for i, (title, desc) in enumerate(labels):
        state = "done" if done_flags[i] else ("active" if i == active_index else "")
        icon = "&#10003;" if done_flags[i] else str(i + 1)

        node = f'<div class="stepper-node {state}"><div class="stepper-badge">{icon}</div><div class="stepper-text-title">{title}</div><div class="stepper-text-desc">{desc}</div></div>'
        nodes_html.append(node)

        if i < len(labels) - 1:
            conn_state = "done" if done_flags[i] else ""
            nodes_html.append(f'<div class="stepper-connector {conn_state}"></div>')

    full_stepper = f'<div class="stepper-container">{"".join(nodes_html)}</div>'
    st.markdown(full_stepper, unsafe_allow_html=True)


def render_chapter_title(number: int, title: str, subtitle: str):
    st.markdown(f'<div class="chapter-title"><div class="chapter-num">{number}</div> {title}</div><div class="chapter-sub">{subtitle}</div>', unsafe_allow_html=True)


def render_tech_callout(html_text: str):
    st.markdown(f'<div class="tech-callout">{html_text}</div>', unsafe_allow_html=True)


def render_why_callout(html_text: str, label: str = "Why This Happened"):
    st.markdown(f'<div class="why-callout"><span class="why-label">{label}</span><br>{html_text}</div>', unsafe_allow_html=True)


def render_page_header(section: str):
    st.markdown(f"""
<div style="margin-bottom: 1.4rem;">
  <p style="font-size:0.82rem; font-weight:800; color:#2563EB; letter-spacing:0.08em; text-transform:uppercase; margin:0 0 0.15rem 0;">
    Experiment {EXPERIMENT_CONFIG['exp_number']}
  </p>
  <h1 style="font-size:1.95rem; font-weight:900; letter-spacing:-0.03em; color:#0F172A; margin:0; line-height:1.2;">
    {section}
  </h1>
  <p style="font-size:1.02rem; color:#475569; font-weight:600; margin-top:0.3rem;">
    {EXPERIMENT_CONFIG['title']}
  </p>
</div>
""", unsafe_allow_html=True)


# ======================================================================================
# 5b. WHY-NARRATION GENERATORS (plain-language, built from real computed results)
# ======================================================================================

def narrate_extraction(entities: list, dict_count: int, rule_count: int) -> str:
    if not entities:
        return "No entities extracted yet."
    friendly = {
        "PERSON": "a person", "ORGANIZATION": "an organization", "LOCATION": "a place",
        "CONCEPT": "a concept", "ROLE_TITLE": "a role/title", "AWARD_EVENT": "an award or event"
    }
    dict_examples = [e["entity"] for e in entities if e["source"] == "Dictionary"][:2]
    rule_examples = [e["entity"] for e in entities if e["source"] == "Heuristic"][:2]

    parts = [f"This passage produced <span class='highlight'>{len(entities)} entities</span>."]

    if dict_examples:
        names = " and ".join(f"<b>{n}</b>" for n in dict_examples)
        parts.append(
            f"{names} {'were' if len(dict_examples) > 1 else 'was'} recognized instantly because "
            f"{'they are' if len(dict_examples) > 1 else 'it is'} already in the internal dictionary — "
            f"a direct lookup, not a guess."
        )
    if rule_examples:
        names = " and ".join(f"<b>{n}</b>" for n in rule_examples)
        first_type = friendly.get(
            next((e["type"] for e in entities if e["entity"] == rule_examples[0]), "CONCEPT"), "something new"
        )
        parts.append(
            f"{names} {'were' if len(rule_examples) > 1 else 'was'} not in that dictionary at all. "
            f"The rule-based pass caught {'them' if len(rule_examples) > 1 else 'it'} anyway by noticing "
            f"surface patterns — Capitalized Words, or a cue word like &ldquo;University&rdquo; or "
            f"&ldquo;Republic&rdquo; — and inferred that &ldquo;{rule_examples[0]}&rdquo; is likely {first_type}."
        )
    if dict_count and rule_count:
        parts.append(
            f"So {dict_count} entities were known outright and {rule_count} had to be inferred — this "
            f"passage needed both extraction strategies to fully cover it."
        )
    elif rule_count and not dict_count:
        parts.append("Every entity here came from pattern inference alone, since none matched the known dictionary.")
    return " ".join(parts)


def narrate_retrieval(query: str, exact_results: list, bm25_results: list) -> tuple:
    exact_hits = [r for r in exact_results if r["score"] > 0]
    top_bm25 = bm25_results[0] if bm25_results else None
    top_exact_name = exact_hits[0]["entity"] if exact_hits else None
    top_bm25_name = top_bm25["entity"] if top_bm25 else None

    if not exact_hits:
        para = (
            f"Exact matching returned no hits for <b>&ldquo;{query}&rdquo;</b> — it only accepts a literal "
            f"(or near-literal) string match, so anything even slightly different is invisible to it. "
            f"BM25 still surfaced <b>{top_bm25_name}</b> as the closest candidate"
            + (f" with a score of <b>{top_bm25['score']}</b>, because it weighs partial term overlap instead of requiring an exact match." if top_bm25 and top_bm25["score"] > 0 else ", though even it found little usable overlap.")
        )
    elif top_exact_name == top_bm25_name:
        para = (
            f"Both methods agree here: <b>{top_bm25_name}</b> tops both rankings because it is close to a "
            f"literal match for <b>&ldquo;{query}&rdquo;</b>. The two methods only start to diverge further "
            f"down the results list, where BM25's weighting matters more."
        )
    else:
        para = (
            f"The two methods disagree. Exact matching ranked <b>{top_exact_name}</b> first purely because "
            f"it is a literal text match for <b>&ldquo;{query}&rdquo;</b>. BM25 instead ranked <b>{top_bm25_name}</b> "
            f"highest — it accounts for how rare and discriminative the overlapping words are, and how that "
            f"compares to the entity's overall text length, which is why a non-literal match can still outrank "
            f"a literal one."
        )

    insight = None
    if len(bm25_results) >= 2 and bm25_results[0]["score"] > 0 and bm25_results[1]["score"] > 0:
        gap = bm25_results[0]["score"] - bm25_results[1]["score"]
        if gap < 0.15:
            insight = (
                f"<b>{bm25_results[0]['entity']}</b> and <b>{bm25_results[1]['entity']}</b> scored nearly "
                f"identically ({bm25_results[0]['score']} vs {bm25_results[1]['score']}). At this margin, "
                f"a small change to k1 or b could flip their order — worth testing."
            )
    return para, insight


def narrate_metrics(k: int, exact_metrics: dict, bm25_metrics: dict, relevant_count: int) -> tuple:
    if relevant_count == 0:
        return (
            "No entity in this passage clearly matches the query, so precision and recall are both zero — "
            "there is nothing correct to measure against. Try a query term that appears in one of the "
            "extracted entities.", None
        )

    def better(a, b):
        if a > b:
            return "BM25"
        if b > a:
            return "Exact matching"
        return "Both methods (tied)"

    p_winner = better(bm25_metrics["precision"], exact_metrics["precision"])
    r_winner = better(bm25_metrics["recall"], exact_metrics["recall"])
    f1_winner = better(bm25_metrics["f1"], exact_metrics["f1"])

    para = (
        f"Of the top <b>{k}</b> results, {relevant_count} entity(ies) genuinely relate to the query. "
        f"<b>Precision@{k}</b> measures how much of what was returned was actually useful — {p_winner} "
        f"scored higher here ({exact_metrics['precision']} vs {bm25_metrics['precision']}). "
        f"<b>Recall@{k}</b> measures how much of everything relevant was actually found — {r_winner} led "
        f"({exact_metrics['recall']} vs {bm25_metrics['recall']}). "
        f"<b>F1@{k}</b> balances both into one score, and overall {f1_winner} came out ahead "
        f"({exact_metrics['f1']} vs {bm25_metrics['f1']})."
    )
    if bm25_metrics["mrr"] > exact_metrics["mrr"]:
        para += (
            f" <b>MRR</b> ({bm25_metrics['mrr']} vs {exact_metrics['mrr']}) also favors BM25 — it surfaced "
            f"the first correct answer earlier, which matters most when only the top few results get seen."
        )
    elif exact_metrics["mrr"] > bm25_metrics["mrr"]:
        para += (
            f" Interestingly, <b>MRR</b> ({exact_metrics['mrr']} vs {bm25_metrics['mrr']}) favors exact "
            f"matching this time — no single method wins every case, which is exactly why both get benchmarked."
        )
    else:
        para += f" Both methods tied on <b>MRR</b> ({bm25_metrics['mrr']})."

    insight = None
    if bm25_metrics["f1"] < exact_metrics["f1"] and bm25_metrics["recall"] >= exact_metrics["recall"]:
        insight = (
            "BM25 found more relevant items overall (higher recall) but let a few less-relevant ones into the "
            "top-K along with them, which pulled its precision down. That trade-off is the classic "
            "precision/recall tension in action."
        )
    return para, insight


# ======================================================================================
# 6. SECTION RENDERERS
# ======================================================================================

def render_purpose_section():
    inject_global_css()

    st.markdown("""
<div class="hero-card">
  <h2>Laboratory Purpose & Technical Objectives</h2>
  <p>
    Standard Information Retrieval (IR) models evaluating raw text strings rely on unweighted term matching (Bag-of-Words), rendering them vulnerable to term variation and keyword frequency distortion.
    <br><br>
    This lab environment transitions search evaluation to <b>Entity-Aware Probabilistic Information Retrieval</b>. You will extract structured <b>Named Entities</b> (Persons, Organizations, Locations, Concepts) via a hybrid gazetteer/heuristic pipeline and rank them using the non-linear <b>Okapi BM25</b> probabilistic model.
  </p>
  <div class="badge-row">
    <span class="badge">Dictionary Lookup</span>
    <span class="badge">Rule-Based Heuristics</span>
    <span class="badge">Okapi BM25</span>
    <span class="badge">Precision / Recall / F1 / MRR</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<p class="section-banner">Core Conceptual Pillars</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="hover-card">
            <h4>1. Hybrid Entity Extraction</h4>
            <p>Moving beyond raw text tokens.</p>
            <div class="definition-reveal">
                <b>Mechanism:</b> Dictionary lookup locates exact matches; rule-based heuristic patterns capture unseen entities using capitalization and syntactic markers.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="hover-card">
            <h4>2. Probabilistic Ranking</h4>
            <p>Non-linear BM25 scoring.</p>
            <div class="definition-reveal">
                <b>Mechanism:</b> Scores candidate entities according to term relevance, incorporating term saturation (k1) and document length normalization (b).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="hover-card">
            <h4>3. IR Evaluation Benchmarks</h4>
            <p>Quantifying retrieval quality.</p>
            <div class="definition-reveal">
                <b>Mechanism:</b> Evaluates comparative execution using Precision@K, Recall@K, F1@K, and Mean Reciprocal Rank (MRR).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-banner">Virtual Lab Pipeline Architecture</p>', unsafe_allow_html=True)
    svg_code = """
    <svg viewBox="0 0 860 150" width="100%" height="150" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gBlue" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#1D4ED8;stop-opacity:1" />
        </linearGradient>
        <linearGradient id="gGreen" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#10B981;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#047857;stop-opacity:1" />
        </linearGradient>
      </defs>

      <line x1="170" y1="75" x2="260" y2="75" stroke="#3B82F6" stroke-width="3" stroke-dasharray="8 4" style="animation: dashFlow 1.4s linear infinite;" />
      <line x1="390" y1="75" x2="480" y2="75" stroke="#8B5CF6" stroke-width="3" stroke-dasharray="8 4" style="animation: dashFlow 1.4s linear infinite;" />
      <line x1="610" y1="75" x2="700" y2="75" stroke="#F59E0B" stroke-width="3" stroke-dasharray="8 4" style="animation: dashFlow 1.4s linear infinite;" />

      <g transform="translate(30, 15)">
        <rect width="140" height="115" rx="12" fill="#FFFFFF" stroke="#3B82F6" stroke-width="2.5"/>
        <circle cx="70" cy="35" r="17" fill="url(#gBlue)"/>
        <text x="70" y="41" text-anchor="middle" fill="#FFF" font-weight="800" font-size="13">1</text>
        <text x="70" y="75" text-anchor="middle" fill="#0F172A" font-weight="800" font-size="12">Hybrid Extraction</text>
        <text x="70" y="93" text-anchor="middle" fill="#1E293B" font-weight="600" font-size="10">Gazetteer + Rules</text>
      </g>

      <g transform="translate(260, 15)">
        <rect width="140" height="115" rx="12" fill="#FFFFFF" stroke="#8B5CF6" stroke-width="2.5"/>
        <circle cx="70" cy="35" r="17" fill="#8B5CF6"/>
        <text x="70" y="41" text-anchor="middle" fill="#FFF" font-weight="800" font-size="13">2</text>
        <text x="70" y="75" text-anchor="middle" fill="#0F172A" font-weight="800" font-size="12">BM25 Tuning</text>
        <text x="70" y="93" text-anchor="middle" fill="#1E293B" font-weight="600" font-size="10">k1 &amp; b Controls</text>
      </g>

      <g transform="translate(480, 15)">
        <rect width="140" height="115" rx="12" fill="#FFFFFF" stroke="#F59E0B" stroke-width="2.5"/>
        <circle cx="70" cy="35" r="17" fill="#F59E0B"/>
        <text x="70" y="41" text-anchor="middle" fill="#FFF" font-weight="800" font-size="13">3</text>
        <text x="70" y="75" text-anchor="middle" fill="#0F172A" font-weight="800" font-size="12">Dual Retrieval</text>
        <text x="70" y="93" text-anchor="middle" fill="#1E293B" font-weight="600" font-size="10">Exact vs BM25</text>
      </g>

      <g transform="translate(700, 15)">
        <rect width="130" height="115" rx="12" fill="#FFFFFF" stroke="#10B981" stroke-width="2.5"/>
        <circle cx="65" cy="35" r="17" fill="url(#gGreen)"/>
        <text x="65" y="41" text-anchor="middle" fill="#FFF" font-weight="800" font-size="13">4</text>
        <text x="65" y="75" text-anchor="middle" fill="#0F172A" font-weight="800" font-size="12">Evaluation</text>
        <text x="65" y="93" text-anchor="middle" fill="#1E293B" font-weight="600" font-size="10">Precision &amp; MRR</text>
      </g>
    </svg>
    """
    st.markdown(svg_code, unsafe_allow_html=True)

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("""
        <div class="hover-card">
            <h4>Phase 1: Hybrid Named Entity Recognition</h4>
            <p>Processes raw text streams using a two-pass architecture: dictionary matching against known terms, followed by rule-based heuristic extraction for unindexed proper nouns.</p>
        </div>
        <div class="hover-card">
            <h4>Phase 2: Okapi BM25 Hyperparameter Tuning</h4>
            <p>Configures parameters k1 (term frequency saturation) and b (length normalization penalty) to observe their exact mathematical effect on document scoring.</p>
        </div>
        """, unsafe_allow_html=True)
    with rcol2:
        st.markdown("""
        <div class="hover-card">
            <h4>Phase 3: Dual Retrieval Execution</h4>
            <p>Executes Exact Keyword Matching and Okapi BM25 Probabilistic Ranking concurrently over extracted entity contexts to evaluate score distributions.</p>
        </div>
        <div class="hover-card">
            <h4>Phase 4: Comparative Benchmarking</h4>
            <p>Computes Precision@K, Recall@K, F1@K, and MRR metrics to assess retrieval performance and logs experimental trials into downloadable PDF reports.</p>
        </div>
        """, unsafe_allow_html=True)


def render_theory_section():
    inject_global_css()

    st.markdown("""
<div class="hero-card">
  <h2>Theoretical Foundations: Named Entity IR & Okapi BM25</h2>
  <p>
    Information Retrieval engines convert unstructured text into indexed structured entities, scoring candidate documents based on the <b>Probability Ranking Principle (PRP)</b>. This page explains how entity extraction and probabilistic ranking work together before you use the simulator.
  </p>
</div>
""", unsafe_allow_html=True)

    # Chapter 1
    render_chapter_title(1, "Named Entity Extraction & Grounding", "Mapping raw text to entity spans")

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("""
        <div class="hover-card">
            <h4>Entity Span Representation</h4>
            <p>Formally representing detected named entities.</p>
            <div class="definition-reveal">
                <b>Example:</b> Extracting <i>"Alan Turing worked at Bletchley Park"</i> yields:<br>
                &bull; <code>Alan Turing</code> &rarr; PERSON (Span 0-11)<br>
                &bull; <code>Bletchley Park</code> &rarr; LOCATION (Span 22-36)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with fcol2:
        st.markdown("""
        <div class="hover-card">
            <h4>Extraction Strategies</h4>
            <p>Gazetteers vs. Heuristics.</p>
            <div class="definition-reveal">
                <b>Dictionary Lookup:</b> High precision for known entities.<br>
                <b>Rule-Based Heuristics:</b> Uses capitalization patterns and suffix triggers (e.g., "Institute" &rarr; ORGANIZATION) to capture unindexed entities.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.latex(r"\text{Extract}(S) \longrightarrow \{(e_i, \text{Type}_i, \text{Source}_i, \text{Span}_{\text{start}}, \text{Span}_{\text{end}})\}")

    render_why_callout(
        "Dictionary lookup is precise but only knows names it has already seen. Rule-based heuristics trade a "
        "little precision for coverage, so the pipeline can still tag names it has never encountered before — "
        "which is exactly why the Simulation tab accepts any custom passage, not just the built-in presets.",
        label="Why Two Strategies"
    )

    st.divider()

    # Chapter 2 - BM25 theory
    render_chapter_title(2, "Okapi BM25 Probabilistic Ranking Model", "Non-linear term saturation and length normalization")

    st.markdown("""
    The **Probability Ranking Principle (PRP)** asserts that an IR system delivers maximum utility when ranking
    documents by their estimated probability of relevance $P(R=1 \\mid D, Q)$. Okapi BM25 operationalizes this
    using non-linear term saturation and a length normalization penalty.
    """)
    st.markdown("<div class=\"math-container\">Okapi BM25 Scoring Equation (per matching term):</div>", unsafe_allow_html=True)
    st.latex(r"\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}")

    render_tech_callout(
        "<b>Core Parameters:</b> "
        "<b>Term Saturation (k1)</b> controls how fast term-frequency impact plateaus — extra occurrences give "
        "diminishing returns. "
        "<b>Length Normalization (b)</b> scales the penalty by document length vs. average length; b=1 applies "
        "the full penalty, b=0 disables it entirely."
    )

    st.markdown("##### How BM25 Builds a Ranking")
    st.markdown("""
    <div class="hover-card">
      <h4>1. Count term frequency</h4>
      <p>BM25 counts how often each query term appears in an entity context. Repeated terms help, but their contribution gradually saturates instead of growing linearly.</p>
    </div>
    <div class="hover-card">
      <h4>2. Weight terms by rarity</h4>
      <p>The inverse document frequency (IDF) component gives more weight to rare terms and reduces the influence of words that occur in nearly every candidate context.</p>
    </div>
    <div class="hover-card">
      <h4>3. Normalize for context length</h4>
      <p>BM25 compares each entity context with the average context length. A long context must provide stronger evidence to earn the same score as a concise context.</p>
    </div>
    """, unsafe_allow_html=True)

    bm25_col1, bm25_col2 = st.columns(2)
    with bm25_col1:
        st.markdown("""
        <div class="math-container"><b>k1: term saturation</b><br>
        Higher k1 lets repeated terms keep contributing for longer. Lower k1 makes the score flatten sooner.</div>
        """, unsafe_allow_html=True)
    with bm25_col2:
        st.markdown("""
        <div class="math-container"><b>b: length normalization</b><br>
        b = 0 ignores context length. b = 1 applies the full length comparison against the corpus average.</div>
        """, unsafe_allow_html=True)

    st.markdown("##### Interpreting a Ranking Decision")
    st.dataframe(pd.DataFrame([
        {"Candidate context": "Short: Alan Turing worked at Bletchley Park.", "Evidence": "Direct entity and place match", "Expected effect": "Strong score"},
        {"Candidate context": "Long: broad history with one mention of Alan Turing.", "Evidence": "One match among much background text", "Expected effect": "Length-normalized score"},
        {"Candidate context": "Repeated keyword list with no entity context.", "Evidence": "High repetition but weak context", "Expected effect": "Saturation limits the boost"}
    ]), use_container_width=True, hide_index=True)

    render_why_callout(
        "BM25 does not understand meaning like a human reader. It combines measurable signals — term frequency, "
        "term rarity, and context length — to produce a ranking that is usually more useful than a binary exact-match "
        "list. The Simulation tab applies the same principles to extracted entities.",
        label="Key Takeaway"
    )

    st.divider()

    # Chapter 3 - case study
    render_chapter_title(3, "Real-World Case Study: Search Engine Knowledge Retrieval", "A step-by-step trace of a realistic query")

    st.info("User Query: 'Where did Alan Turing decipher the Enigma machine during World War II?'")

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("""
        <div class="hover-card">
            <h4>Extracted Target Entities</h4>
            <p>Identified structured query nodes.</p>
            <p style="margin-top:0.45rem; font-size:0.88rem;">
                &bull; <code>Alan Turing</code> &rarr; <b>PERSON</b><br>
                &bull; <code>Enigma machine</code> &rarr; <b>CONCEPT / DEVICE</b><br>
                &bull; <code>World War II</code> &rarr; <b>HISTORICAL EVENT</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
        <div class="hover-card">
            <h4>Candidate Corpus Passages</h4>
            <p>Retrieved documents from the search index.</p>
            <p style="margin-top:0.45rem; font-size:0.88rem;">
                &bull; <b>Doc A (short):</b> <i>'Alan Turing worked at Bletchley Park breaking the Enigma machine during WWII.'</i><br>
                &bull; <b>Doc B (long):</b> <i>'World War II history spans thousands of operations across Europe. Alan Turing spent time at Bletchley Park...'</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("##### Why Naive Exact Matching Fails Here")
    render_why_callout(
        "Doc A uses the shorthand <b>WWII</b> while the query says <b>World War II</b> — exact binary matching "
        "gives Doc A zero credit for that, even though it is clearly the better answer. Meanwhile Doc B repeats "
        "<b>World War II</b> ten times in generic background text, which would unfairly inflate its score under "
        "plain linear word counting.",
        label="The Problem"
    )

    st.markdown("##### How Okapi BM25 Solves It")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown("""
        <div class="hover-card">
            <h4>1. Rare-Term Weighting</h4>
            <p>IDF calculation</p>
            <div class="definition-reveal">
                Terms like "Alan Turing" or "Enigma" are rare, so they get a high IDF weight. Common words like "where" or "did" get almost zero weight.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ec2:
        st.markdown("""
        <div class="hover-card">
            <h4>2. Term Saturation (k1)</h4>
            <p>Score flattening</p>
            <div class="definition-reveal">
                Doc B repeating "World War II" ten times does not get ten times the score — k1 causes extra occurrences to saturate quickly.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ec3:
        st.markdown("""
        <div class="hover-card">
            <h4>3. Length Normalization (b)</h4>
            <p>Document penalty</p>
            <div class="definition-reveal">
                Doc A is concise and directly on-topic. Doc B is penalized for its excess length (|D| &gt; avgdl), which is why Doc A ends up ranked first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_why_callout(
        "Put together, these three mechanisms are why BM25 ranks the short, on-topic Doc A above the long, "
        "repetitive Doc B — not because it read either document for meaning, but because rarity, saturation, "
        "and length normalization combine to reward concise relevance over keyword volume.",
        label="Putting It Together"
    )

    st.divider()

    # Chapter 4
    render_chapter_title(4, "Retrieval Benchmark Metrics", "Evaluating IR effectiveness")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("<div class=\"math-container\">Precision@K Formula:</div>", unsafe_allow_html=True)
        st.latex(r"\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Top-K}|}{K}")

        st.markdown("<div class=\"math-container\">F1@K Formula:</div>", unsafe_allow_html=True)
        st.latex(r"\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}")

    with mcol2:
        st.markdown("<div class=\"math-container\">Recall@K Formula:</div>", unsafe_allow_html=True)
        st.latex(r"\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Top-K}|}{|\text{Total Relevant}|}")

        st.markdown("<div class=\"math-container\">Mean Reciprocal Rank (MRR) Formula:</div>", unsafe_allow_html=True)
        st.latex(r"\text{MRR} = \frac{1}{\text{Rank}_{\text{first\_relevant}}}")

    render_why_callout(
        "Precision asks <b>\"don't waste my time\"</b> and recall asks <b>\"don't miss anything.\"</b> A system "
        "returning one perfect result has great precision but poor recall; one returning everything has the "
        "opposite problem. F1 punishes both extremes, and MRR rewards putting the right answer first rather "
        "than burying it on page five. Head to the <b>Simulation</b> tab to see all four computed live from a "
        "passage of your choosing.",
        label="How To Read These"
    )


def _select_guided_preset() -> None:
    st.session_state["guided_preset_text"] = SAMPLE_PRESETS[st.session_state["guided_preset"]]


def _select_guided_query(query: str) -> None:
    st.session_state["guided_query"] = query


def render_simulation_section():
    inject_global_css()

    friendly_types = {
        "PERSON": "Person", "ORGANIZATION": "Organization", "LOCATION": "Place",
        "CONCEPT": "Concept", "ROLE_TITLE": "Role", "AWARD_EVENT": "Award/Event"
    }

    entities_now = st.session_state.get("guided_entities", [])
    retrieval_done = st.session_state.get("guided_retrieval_run", False)
    trial_logged = len(st.session_state.get("trials", [])) > 0
    done_flags = [bool(entities_now), bool(entities_now), retrieval_done, retrieval_done and trial_logged]
    active_idx = 3 if done_flags[3] else (2 if retrieval_done else (1 if entities_now else 0))

    render_phase_stepper(active_idx, done_flags)

    # Phase 1
    render_chapter_title(1, "Text Input & Entity Extraction", "Phase 1: Parsing unstructured passages")

    render_tech_callout(
        "<b>Dictionary pass:</b> matches explicit entity terms against an internal gazetteer for high precision. "
        "<b>Rule-based pass:</b> catches unindexed proper nouns using capitalization sequences and suffix "
        "triggers (e.g., &ldquo;University&rdquo;, &ldquo;Corporation&rdquo;) so custom text is handled too."
    )

    input_mode = st.radio("Select corpus source:", ["Benchmark Preset", "Custom Text Input"], horizontal=True, key="guided_input_mode")
    preset_names = list(SAMPLE_PRESETS)
    if input_mode == "Benchmark Preset":
        if "guided_preset_text" not in st.session_state:
            st.session_state["guided_preset_text"] = SAMPLE_PRESETS[preset_names[0]]
        st.selectbox("Select benchmark corpus:", preset_names, key="guided_preset", on_change=_select_guided_preset)
        active_text = st.text_area("Corpus passage:", height=100, key="guided_preset_text")
    else:
        if "guided_custom_text" not in st.session_state:
            st.session_state["guided_custom_text"] = ""
        active_text = st.text_area(
            "Custom input passage:", height=100, key="guided_custom_text",
            placeholder="Enter a custom passage (e.g., 'Alan Turing conducted foundational work at Bletchley Park in the United Kingdom.')"
        )

    text_signature = active_text.strip()
    if st.session_state.get("guided_text_signature") != text_signature:
        st.session_state["guided_entities"] = []
        st.session_state["guided_retrieval_run"] = False
        st.session_state["guided_query"] = ""
        st.session_state["guided_text_signature"] = text_signature

    if st.button("Extract Named Entities", type="primary", key="guided_identify"):
        if not active_text.strip():
            st.warning("Please specify a valid text input before running entity extraction.")
        else:
            st.session_state["guided_entities"] = extract_entities_from_text(active_text)
            st.session_state["guided_text_signature"] = text_signature
            st.session_state["guided_retrieval_run"] = False
            st.rerun()

    entities = st.session_state.get("guided_entities", [])
    if not entities:
        st.info("Input a corpus text above and click Extract Named Entities to initiate execution.")
        return

    dict_count = sum(1 for e in entities if e["source"] == "Dictionary")
    rule_count = len(entities) - dict_count

    entity_table = pd.DataFrame([
        {
            "Entity Span": item["entity"],
            "Entity Type": friendly_types.get(item["type"], item["type"]),
            "Extraction Source": item["source"],
            "Start": item["start"],
            "End": item["end"],
            "Extracted Context": item["context"]
        }
        for item in entities
    ])
    st.dataframe(entity_table, use_container_width=True, hide_index=True)

    render_why_callout(narrate_extraction(entities, dict_count, rule_count))

    st.divider()

    # Phase 2
    render_chapter_title(2, "Parameter Tuning & Dual Retrieval Execution", "Phases 2 and 3: model configuration and retrieval")

    render_tech_callout(
        "<b>Evaluation Depth (K):</b> restricts precision/recall to the top-K ranked items. "
        "<b>Term Saturation (k1):</b> controls how rapidly score growth flattens as a query term repeats. "
        "<b>Length Normalization (b):</b> penalizes long entity contexts relative to the average; b=0 disables it."
    )

    examples = list(dict.fromkeys(item["entity"] for item in entities))[:4]
    st.caption("Select a candidate entity as the query, or type a custom one below:")
    qcols = st.columns(max(1, len(examples)))
    for idx, ex in enumerate(examples):
        with qcols[idx]:
            st.button(ex, key=f"guided_ex_{idx}", use_container_width=True, on_click=_select_guided_query, args=(ex,))

    if not st.session_state.get("guided_query"):
        st.session_state["guided_query"] = examples[0]
    query = st.text_input("Query string:", key="guided_query").strip()

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Evaluation Depth (K)", 1, len(entities), min(3, len(entities)), key="guided_k")
    with c2:
        k1 = st.slider("BM25 Term Saturation (k1)", 0.5, 3.0, 1.5, 0.1, key="guided_k1")
    with c3:
        b = st.slider("BM25 Length Normalization (b)", 0.0, 1.0, 0.75, 0.05, key="guided_b")

    if st.button("Execute Dual Retrieval Analysis", type="primary", key="guided_run"):
        st.session_state["guided_retrieval_run"] = True
        st.rerun()

    if not st.session_state.get("guided_retrieval_run", False):
        st.info("Click Execute Dual Retrieval Analysis to run the comparative ranking models.")
        return

    retrieval = perform_dual_retrieval(entities, query, k1, b)
    exact_results, bm25_results = retrieval["exact"], retrieval["bm25"]
    exact_names = [item["entity"] for item in exact_results]
    bm25_names = [item["entity"] for item in bm25_results]

    st.divider()

    # Phase 3
    render_chapter_title(3, "Comparative Retrieval Results", f"Query evaluation: \"{query}\"")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("##### Exact Keyword Matching Baseline")
        st.caption("Binary match model (Score = 1.0 or 0.0) — no term weighting, no length normalization.")
        st.dataframe(pd.DataFrame([
            {"Rank": r, "Entity": x["entity"], "Status": x["match_status"], "Score": x["score"]}
            for r, x in enumerate(exact_results, 1)
        ]), use_container_width=True, hide_index=True)
    with rcol2:
        st.markdown("##### Okapi BM25 Probabilistic Ranking")
        st.caption("Non-linear saturation model — computes IDF, applies saturation curvature, normalizes length.")
        st.dataframe(pd.DataFrame([
            {"Rank": r, "Entity": x["entity"], "BM25 Score": x["score"], "Confidence": x["confidence"]}
            for r, x in enumerate(bm25_results, 1)
        ]), use_container_width=True, hide_index=True)

    retrieval_narration, retrieval_insight = narrate_retrieval(query, exact_results, bm25_results)
    render_why_callout(retrieval_narration)
    if retrieval_insight:
        render_tech_callout(retrieval_insight)

    st.markdown("##### Score Distribution Chart")
    exact_score_by_name = {x["entity"]: x["score"] for x in exact_results}
    bm25_score_by_name = {x["entity"]: x["score"] for x in bm25_results}
    top_names = bm25_names[:min(8, len(bm25_names))]
    chart_df = pd.DataFrame({
        "Exact Match Score": [exact_score_by_name.get(n, 0.0) for n in top_names],
        "BM25 Probabilistic Score": [bm25_score_by_name.get(n, 0.0) for n in top_names],
    }, index=top_names)
    st.bar_chart(chart_df, height=260)

    st.divider()

    # Phase 4
    render_chapter_title(4, "IR Performance Benchmarking", "Phase 4: quantitative metric evaluation")

    render_tech_callout(
        "<b>Precision@K:</b> fraction of the top-K results that are relevant. "
        "<b>Recall@K:</b> fraction of all relevant items captured within top-K. "
        "<b>F1@K:</b> harmonic mean balancing precision and recall. "
        "<b>MRR:</b> reciprocal position (1/rank) of the first correct hit."
    )

    q_tokens = set(tokenize(query))
    relevant = {item["entity"] for item in entities if query.lower() in item["entity"].lower() or (set(tokenize(item["entity"])) & q_tokens)}

    exact_metrics = compute_ir_metrics(exact_names, relevant, k)
    bm25_metrics = compute_ir_metrics(bm25_names, relevant, k)

    metric_rows = [
        {"Metric": f"Precision@{k}", "Exact Matching": exact_metrics["precision"], "Okapi BM25": bm25_metrics["precision"]},
        {"Metric": f"Recall@{k}", "Exact Matching": exact_metrics["recall"], "Okapi BM25": bm25_metrics["recall"]},
        {"Metric": f"F1@{k}", "Exact Matching": exact_metrics["f1"], "Okapi BM25": bm25_metrics["f1"]},
        {"Metric": "MRR", "Exact Matching": exact_metrics["mrr"], "Okapi BM25": bm25_metrics["mrr"]}
    ]
    mcol1, mcol2 = st.columns([1.2, 1])
    with mcol1:
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    with mcol2:
        metric_chart_df = pd.DataFrame(metric_rows).set_index("Metric")
        st.bar_chart(metric_chart_df, height=230)

    metrics_narration, metrics_insight = narrate_metrics(k, exact_metrics, bm25_metrics, len(relevant))
    render_why_callout(metrics_narration)
    if metrics_insight:
        render_tech_callout(metrics_insight)

    if st.button("Log Trial Data", type="primary", key="guided_record_trial"):
        trial = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "Query": query,
            "Entities": len(entities),
            "K": k,
            "Exact Results": ", ".join(exact_names[:k]),
            "BM25 Results": ", ".join(bm25_names[:k]),
            "Precision Exact": exact_metrics["precision"],
            "Precision BM25": bm25_metrics["precision"],
            "Recall Exact": exact_metrics["recall"],
            "Recall BM25": bm25_metrics["recall"],
            "F1 Exact": exact_metrics["f1"],
            "F1 BM25": bm25_metrics["f1"],
            "MRR Exact": exact_metrics["mrr"],
            "MRR BM25": bm25_metrics["mrr"],
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state["trials"].append(trial)
        st.toast(f"Recorded Trial #{trial['Trial #']} successfully.")

    if st.session_state["trials"]:
        st.markdown("##### Logged Experimental Trial History")
        st.dataframe(pd.DataFrame(st.session_state["trials"]), use_container_width=True, hide_index=True)


def render_quiz_section():
    inject_global_css()
    st.markdown("### Concept Assessment Quiz")
    st.write("Complete the questions below to evaluate your understanding of Entity Extraction and Probabilistic Ranking.")

    if st.button("Generate New Question Set"):
        bank_ids = [q["id"] for q in QUIZ_QUESTION_BANK]
        st.session_state["quiz_question_ids"] = random.sample(bank_ids, min(10, len(bank_ids)))
        st.session_state["quiz_answers"] = {}
        st.session_state["quiz_submitted"] = False
        st.session_state["quiz_score"] = 0
        st.session_state["quiz_set_version"] += 1

    version = st.session_state["quiz_set_version"]
    active_ids = st.session_state["quiz_question_ids"]
    questions_by_id = {q["id"]: q for q in QUIZ_QUESTION_BANK}
    active_questions = [questions_by_id[qid] for qid in active_ids]

    with st.form("quiz_form"):
        user_responses = {}
        for display_idx, q in enumerate(active_questions, start=1):
            st.markdown(f"**Question {display_idx}:** {q['question']}")
            selected = st.radio(
                label=f"Options for Quiz Q{display_idx}",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"]),
                key=f"quiz_radio_{version}_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected) if selected else None
            st.markdown("---")

        submitted = st.form_submit_button("Submit Quiz for Evaluation", type="primary")

    if submitted:
        unanswered = [idx for idx, q in enumerate(active_questions, start=1) if user_responses.get(q["id"]) is None]
        if unanswered:
            st.error(f"Please answer all 10 questions before submitting. Missing question(s): {', '.join(map(str, unanswered))}.")
            return

        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Quiz Results & Performance Feedback")
        for display_idx, q in enumerate(active_questions, start=1):
            user_ans = user_responses.get(q["id"])
            corr_ans = q["answer_index"]
            if user_ans == corr_ans:
                score += 1
                st.success(f"**Question {display_idx}: Correct**\n\n_{q['explanation']}_")
            else:
                st.error(
                    f"**Question {display_idx}: Incorrect** (Selected Answer: {q['options'][user_ans]})\n\n"
                    f"**Correct Answer:** {q['options'][corr_ans]}\n\n"
                    f"**Technical Explanation:** _{q['explanation']}_"
                )

        st.session_state["quiz_score"] = score
        perc = (score / len(active_questions)) * 100
        st.info(f"Final Score: **{score} / {len(active_questions)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz completed. Current Score: **{st.session_state.get('quiz_score', 0)} / {len(active_questions)}**")


def render_report_section():
    inject_global_css()
    st.markdown("### Official Lab Report Generation")
    st.write("Generate a PDF report containing student identification details, assessment scores, and logged experimental trial metrics.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-006"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    student_notes = st.text_area(
        "Experimental Analysis & Observations:",
        value=st.session_state.get("student_notes", (
            "The experimental trials demonstrated that Okapi BM25 probabilistic ranking consistently outperformed "
            "exact keyword matching across Precision@K, F1@K, and MRR metrics by accounting for non-linear term "
            "saturation and document length normalization."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(st.session_state.get("quiz_question_ids", [])),
        student_notes=student_notes
    )

    st.download_button(
        label="Download Official Lab Report (.pdf)",
        data=pdf_bytes,
        file_name="lab_report_exp6.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )


def render_certificate_section():
    inject_global_css()
    st.markdown("### Certificate of Completion")
    st.write("Download an official completion certificate upon completing the experiment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        cert_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"), key="cert_name_input")
    with col2:
        cert_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-006"), key="cert_id_input")
    with col3:
        cert_date = st.date_input("Completion Date", value=datetime.now(), key="cert_date_input")

    quiz_total = len(st.session_state.get("quiz_question_ids", []))
    quiz_score = st.session_state.get("quiz_score", 0)

    cert_bytes = generate_certificate_pdf(
        student_name=cert_name,
        student_id=cert_id,
        date_str=str(cert_date),
        quiz_score=quiz_score,
        quiz_total=quiz_total
    )

    st.download_button(
        label="Download Certificate (.pdf)",
        data=cert_bytes,
        file_name="completion_certificate.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )


def render_references_section():
    inject_global_css()
    st.markdown("### Academic References & Technical Manuals")
    st.markdown("""
1. **Robertson, S. E., & Zaragoza, H. (2009)**. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
2. **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**. *Introduction to Information Retrieval*. Cambridge University Press.
3. **Nadeau, D., & Sekine, S. (2007)**. *A Survey of Named Entity Recognition and Classification*. Lingvisticae Investigationes, 30(1), 3-26.
4. **Croft, W. B., Metzler, D., & Strohman, T. (2010)**. *Search Engines: Information Retrieval in Practice*. Addison-Wesley.
""")


# ======================================================================================
# 7. MAIN APPLICATION CONTROLLER
# ======================================================================================

def init_session_state():
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_question_ids" not in st.session_state or len(st.session_state["quiz_question_ids"]) != 10:
        bank_ids = [q["id"] for q in QUIZ_QUESTION_BANK]
        st.session_state["quiz_question_ids"] = random.sample(bank_ids, min(10, len(bank_ids)))
    if "quiz_set_version" not in st.session_state:
        st.session_state["quiz_set_version"] = 0
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-006",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="Experiment 6: Entity Identification & Probabilistic Retrieval",
        layout="wide"
    )

    init_session_state()

    st.sidebar.title("Virtual Lab Engine")
    section = st.sidebar.radio(
        "Navigation",
        options=[
            "Purpose",
            "Theory",
            "Simulation",
            "Quiz",
            "Report Generation",
            "Certificate",
            "References"
        ]
    )

    render_page_header(section)

    st.sidebar.divider()
    st.sidebar.subheader("Session Progress")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz:** {quiz_status} ({st.session_state.get('quiz_score', 0)}/{len(st.session_state.get('quiz_question_ids', []))})")
    st.sidebar.write(f"- **Trials Logged:** {len(st.session_state.get('trials', []))}")

    if section == "Purpose":
        render_purpose_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
    elif section == "Certificate":
        render_certificate_section()
    elif section == "References":
        render_references_section()


if __name__ == "__main__":
    main()