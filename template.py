"""
Knowledge Graph Information Retrieval System (KGIRS) Virtual Laboratory
Experiment 6: Identify Graph Entities

Four Core Sections:
  1. Theory: Graph entities, NER, entity typing, candidate generation, Exact vs. Probabilistic IR.
  2. Simulation: Interactive entity extraction, dual retrieval engine (Exact vs. Probabilistic),
                 score ranking, Plotly visual comparison, and trial logger.
  3. Quiz: 10-question self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student metadata, recorded trials, discussion notes, and downloadable PDF report.

Note: Native Streamlit components used throughout without custom CSS for full theme compatibility.
"""

import os
import re
import math
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Experiment 6: Identify Graph Entities",
    "subject": "Knowledge Graph Information Retrieval System",
    "objectives": [
        "Understand the principles of Named Entity Recognition (NER) for identifying graph entities (Person, Organization, Location, Role, Concept) from raw text.",
        "Formulate a candidate entity repository representing potential nodes for Knowledge Graph population.",
        "Implement and evaluate Exact / Keyword Retrieval based on deterministic boolean string and token containment.",
        "Calculate Probabilistic Ranked Retrieval scores using token overlap, term coverage, and n-gram similarity under the Probability Ranking Principle (PRP).",
        "Critically compare Exact vs. Probabilistic retrieval in terms of recall, ranking discriminability, and resilience to query ambiguity."
    ]
}

THEORY_CONTENT = {
    "background": """
### 1. Graph Entities in Knowledge Graphs
In a Knowledge Graph $\\mathcal{G} = (\\mathcal{E}, \\mathcal{R}, \\mathcal{T})$, **entities** ($e \\in \\mathcal{E}$) represent unique, real-world objects, abstract concepts, or semantic instances (e.g., *Sundar Pichai*, *Google*, *Stanford University*). Entities serve as the fundamental nodes connected by directed relation edges ($r \\in \\mathcal{R}$) forming factual knowledge triples:
$$\\langle \\text{subject entity}, \\text{predicate relation}, \\text{object entity} \\rangle$$

### 2. Entity Identification (Named Entity Recognition - NER)
Before knowledge can be indexed or linked into a graph, raw unstructured text must undergo **Entity Identification**:
1. **Span Detection**: Locating the start and end character boundaries $[i, j]$ of potential entity mentions within the sentence.
2. **Entity Classification / Typing**: Assigning each identified mention to an ontology category:
   - **PERSON**: Individual human beings (e.g., *Sundar Pichai*, *Alan Turing*).
   - **ORGANIZATION**: Companies, universities, institutions (e.g., *Google*, *Stanford University*, *CERN*).
   - **LOCATION**: Geopolitical entities, cities, countries (e.g., *California*, *San Francisco*, *Geneva*).
   - **ROLE_TITLE**: Professional titles or executive designations (e.g., *CEO*, *Inventor*, *Scientist*).
   - **TECH_CONCEPT / EVENT**: Domains, artifacts, technologies, or historic events (e.g., *Artificial Intelligence*, *World Wide Web*).

### 3. Candidate Entities in a Knowledge Graph
Identified entity mentions from raw text represent **candidate graph entities**. In large-scale information retrieval systems, these candidates must be mapped to existing knowledge bases or indexed for fast entity retrieval. Because natural language is inherently ambiguous (polysemy, homonymy, and surface variations), candidate generation provides the search space upon which retrieval algorithms operate.

### 4. Information Retrieval Paradigms: Exact vs. Probabilistic Retrieval

#### A. Exact / Keyword Retrieval (Deterministic Boolean Matching)
- **Mechanism**: Evaluates whether the user's search query $Q$ strictly matches the candidate entity surface name $E$ via exact string equality or token containment ($Q = E$ or $Q \\subseteq E$).
- **Output**: Binary relevance score $S_{\\text{exact}} \\in \\{0.0, 1.0\\}$.
- **Advantages**: Highly precise for known canonical names, negligible computational overhead, zero false positives on strict matches.
- **Limitations**: **Extremely brittle**. Fails entirely when queries contain multiple keywords not found in a single entity name (e.g., querying `"Google CEO"` against candidates returns zero matches), suffers zero recall on spelling variants or abbreviations, and cannot provide graded relevance ranking.

#### B. Probabilistic Ranked Retrieval (Probability Ranking Principle - PRP)
- **Mechanism**: Founded on Robertson's **Probability Ranking Principle (PRP)**, which posits that an information retrieval system achieves maximum effectiveness if it ranks candidates in descending order of their estimated probability of relevance to the query:
$$P(R=1 \\mid Q, E)$$
- **Score Calculation**: Combines multiple probabilistic evidence signals:
  1. **Token Jaccard Probability**:
     $$J(Q, E) = \\frac{|Tokens(Q) \\cap Tokens(E)|}{|Tokens(Q) \\cup Tokens(E)|}$$
  2. **Query Term Coverage**:
     $$\\text{Coverage}(Q, E) = \\frac{|Tokens(Q) \\cap Tokens(E)|}{|Tokens(Q)|}$$
  3. **Character n-gram String Similarity (Fuzzy Stem Matching)**:
     $$\\text{Sim}_{\\text{ngram}}(Q, E) = \\frac{2 \\cdot |Ngrams(Q) \\cap Ngrams(E)|}{|Ngrams(Q)| + |Ngrams(E)|}$$
  4. **Contextual Relevance Prior**: Boost awarded when query terms appear in the surrounding entity context window or matching entity type.
- **Combined Relevance Score**:
  $$S_{\\text{prob}}(Q, E) = \\min\\Big(1.0,\\, w_1 \\cdot \\text{Coverage} + w_2 \\cdot J(Q, E) + w_3 \\cdot \\text{Sim}_{\\text{ngram}} + w_4 \\cdot \\text{ContextBoost}\\Big)$$
- **Advantages**: **High recall and graded ranking discriminability**. Gracefully handles multi-token queries (`"Google CEO"` ranks both *Google* and *CEO* near the top), tolerates minor typos or partial stems (`"Sundar"` matches *Sundar Pichai* with high confidence), and ranks the entire candidate space smoothly.

### 5. Comparative Performance Summary
| Feature / Dimension | Exact / Keyword Retrieval | Probabilistic Ranked Retrieval |
| :--- | :--- | :--- |
| **Output Type** | Binary decision $\\{0, 1\\}$ | Continuous probability score $[0.0, 1.0]$ |
| **Ranking Capability** | Unranked (all matches equal) | Fine-grained descending ranking (PRP) |
| **Partial / Multi-word Queries** | Fails (Zero recall on compound terms) | High recall (Scores individual token contributions) |
| **Fuzzy / Typo Robustness** | Zero tolerance | High (n-gram similarity softens string boundaries) |
| **KGIRS Application** | Quick exact dictionary lookup | Exploratory entity search & entity disambiguation |
    """,
    "procedure": [
        "Step 1: Review the theoretical foundations of Graph Entities, NER, and Retrieval models.",
        "Step 2: Navigate to the Simulation section in the sidebar menu.",
        "Step 3: Select one of the preset benchmark texts (e.g., Sundar Pichai at Google/Stanford) or enter your own custom text.",
        "Step 4: Click 'Identify Entities' to execute the NER pipeline and extract candidate graph entities.",
        "Step 5: Inspect the detected entities table with their assigned types (PERSON, ORGANIZATION, LOCATION, etc.).",
        "Step 6: Enter a search query (e.g., 'Google', 'Sundar', 'Google CEO', or 'Stanford') or click a quick-fill sample query.",
        "Step 7: Compare the side-by-side results of Exact / Keyword Retrieval versus Probabilistic Ranked Retrieval.",
        "Step 8: Observe the interactive Plotly ranking comparison chart showing graded relevance scores.",
        "Step 9: Click 'Record Current Trial' to capture the experimental trial into your session logbook.",
        "Step 10: Repeat for at least 3-4 distinct queries (including exact terms, single words, and multi-word queries).",
        "Step 11: Complete the 10-question self-grading Quiz to verify your conceptual mastery.",
        "Step 12: Open Report Generation, enter your student details, and generate your downloadable PDF lab report."
    ],
    "key_terms": {
        "Graph Entity": "A distinct node in a knowledge graph representing a concrete or abstract real-world object (e.g., Person, Org).",
        "Entity Identification (NER)": "The natural language processing task of locating mention spans in text and classifying their semantic category.",
        "Entity Type": "The ontological classification assigned to an entity (e.g., PERSON, ORGANIZATION, LOCATION, ROLE_TITLE).",
        "Candidate Graph Entity": "An extracted entity mention treated as a candidate entry for knowledge graph population or retrieval index.",
        "Exact / Keyword Retrieval": "Deterministic information retrieval that requires strict boolean string or token containment.",
        "Probabilistic Ranking": "Ranking retrieval candidates in descending order of their calculated probability of relevance P(R=1 | Q, E).",
        "Probability Ranking Principle": "Fundamental IR principle stating systems maximize utility by ranking candidates by decreasing probability of relevance.",
        "Relevance Score": "A normalized numerical value in [0.0, 1.0] reflecting the calculated relevance of a candidate entity to a query.",
        "Ranking Discriminability": "The ability of a retrieval model to separate highly relevant entities from marginally relevant or non-relevant ones."
    }
}

SAMPLE_PRESETS = {
    "Tech Leaders (Sundar Pichai / Google)": (
        "Sundar Pichai is the CEO of Google and studied at Stanford University in California."
    ),
    "Computing History (Alan Turing / Bletchley Park)": (
        "Alan Turing worked at Bletchley Park in the United Kingdom to break the Enigma machine for Allied forces."
    ),
    "Modern AI (Satya Nadella / Microsoft / OpenAI)": (
        "Satya Nadella led Microsoft to partner with OpenAI based in San Francisco to advance Artificial Intelligence."
    ),
    "Scientific Discoveries (Marie Curie / Paris)": (
        "Marie Curie conducted pioneering research on radioactivity at the University of Paris and won Nobel Prizes in Stockholm, Sweden."
    ),
    "Internet Pioneers (Tim Berners-Lee / CERN)": (
        "Tim Berners-Lee invented the World Wide Web at CERN in Geneva, Switzerland."
    )
}

KNOWN_GAZETTEER = {
    "Sundar Pichai": "PERSON",
    "Alan Turing": "PERSON",
    "Satya Nadella": "PERSON",
    "Marie Curie": "PERSON",
    "Tim Berners-Lee": "PERSON",
    "Google": "ORGANIZATION",
    "Stanford University": "ORGANIZATION",
    "Microsoft": "ORGANIZATION",
    "OpenAI": "ORGANIZATION",
    "University of Paris": "ORGANIZATION",
    "CERN": "ORGANIZATION",
    "Bletchley Park": "ORGANIZATION",
    "California": "LOCATION",
    "United Kingdom": "LOCATION",
    "San Francisco": "LOCATION",
    "Stockholm": "LOCATION",
    "Sweden": "LOCATION",
    "Geneva": "LOCATION",
    "Switzerland": "LOCATION",
    "CEO": "ROLE_TITLE",
    "Artificial Intelligence": "TECH_CONCEPT",
    "World Wide Web": "TECH_CONCEPT",
    "Enigma machine": "TECH_CONCEPT",
    "Nobel Prizes": "AWARD_EVENT",
    "radioactivity": "TECH_CONCEPT"
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "In a Knowledge Graph (KG), what does an 'entity' represent?",
        "options": [
            "A) A relation label connecting two existing graph nodes",
            "B) A discrete node representing a real-world object, person, organization, or concept",
            "C) A retrieval query used to search the graph",
            "D) A score assigned to a candidate during ranking"
        ],
        "answer_index": 1,
        "explanation": "Entities are the nodes in a Knowledge Graph representing unique real-world objects or concepts (e.g., Sundar Pichai, Google)."
    },
    {
        "id": 2,
        "question": "What are the two primary sub-tasks involved in Named Entity Recognition (NER)?",
        "options": [
            "A) Boundary span detection and semantic type classification",
            "B) Query expansion and result ranking",
            "C) Candidate graph construction and relation extraction",
            "D) Token indexing and PDF report generation"
        ],
        "answer_index": 0,
        "explanation": "NER first detects the start and end boundary spans of entity mentions in the text, then assigns them to semantic types (e.g., PERSON, ORG)."
    },
    {
        "id": 3,
        "question": "Why are identified entity mentions referred to as 'candidate graph entities' prior to graph insertion?",
        "options": [
            "A) Because they are already confirmed as unique graph nodes",
            "B) Because natural language mentions are ambiguous and must be evaluated for linking and retrieval",
            "C) Because they contain only numerical identifiers",
            "D) Because candidate entities cannot be used by an information retrieval system"
        ],
        "answer_index": 1,
        "explanation": "Mention spans extracted from raw text may refer to multiple real-world entities (ambiguity/polysemy), so they form a candidate space for linking and retrieval."
    },
    {
        "id": 4,
        "question": "What is the primary limitation of Exact / Keyword Retrieval when searching candidate graph entities?",
        "options": [
            "A) It may return several partial matches with different relevance strengths",
            "B) It requires a probabilistic model for every exact string match",
            "C) It suffers from extreme brittleness, returning zero results on multi-word or compound queries not matching a single entity name",
            "D) It cannot identify the entity type before retrieval"
        ],
        "answer_index": 2,
        "explanation": "Exact keyword matching is binary and brittle: a query like 'Google CEO' fails completely because no single candidate entity is named 'Google CEO'."
    },
    {
        "id": 5,
        "question": "What does Robertson's Probability Ranking Principle (PRP) dictate for optimal information retrieval?",
        "options": [
            "A) Candidates should be displayed in the order they appeared in the source text",
            "B) Candidates should be ranked in descending order of their estimated probability of relevance to the query",
            "C) Candidates with the same entity type should always be ranked together",
            "D) Only candidates with an exact score of 1.0 should be displayed"
        ],
        "answer_index": 1,
        "explanation": "The Probability Ranking Principle (PRP) states that retrieval effectiveness is maximized when items are presented in decreasing order of their probability of relevance P(R=1 | Q, E)."
    },
    {
        "id": 6,
        "question": "In probabilistic entity retrieval, how is Token Jaccard Similarity between query Q and candidate E calculated?",
        "options": [
            "A) |Tokens(Q) ∩ Tokens(E)| / |Tokens(Q) ∪ Tokens(E)|",
            "B) |Tokens(Q)| × |Tokens(E)|",
            "C) |Tokens(Q) ∩ Tokens(E)| / |Tokens(Q)|",
            "D) |Tokens(Q) ∪ Tokens(E)| / |Tokens(Q) ∩ Tokens(E)|"
        ],
        "answer_index": 0,
        "explanation": "Jaccard similarity is the size of the token intersection divided by the size of the token union: |Q ∩ E| / |Q ∪ E|."
    },
    {
        "id": 7,
        "question": "Suppose the extracted candidate entities are [Sundar Pichai, CEO, Google, Stanford University]. If the query is 'Google CEO', what occurs?",
        "options": [
            "A) Exact retrieval returns Google and CEO because it combines separate token matches",
            "B) Exact retrieval returns 0 matches, while Probabilistic retrieval successfully ranks Google and CEO with high relevance scores",
            "C) Exact retrieval returns only Stanford University because it has the longest name",
            "D) Both methods reject the query because it contains more than one token"
        ],
        "answer_index": 1,
        "explanation": "Exact match requires the entire string 'Google CEO' in a candidate name, yielding 0 matches. Probabilistic retrieval scores token overlap and correctly ranks Google and CEO at the top."
    },
    {
        "id": 8,
        "question": "Why is character n-gram similarity included alongside token matching in probabilistic entity scoring?",
        "options": [
            "A) To give every candidate the same relevance score",
            "B) To identify the semantic type of an entity mention",
            "C) To provide resilience against minor typographical errors, morphological stems, and partial spelling variations",
            "D) To replace token coverage and Jaccard similarity completely"
        ],
        "answer_index": 2,
        "explanation": "Sub-word n-gram overlap softens strict word boundaries, allowing queries with slight typos (e.g. 'Googel') or partial stems to still match relevant entities."
    },
    {
        "id": 9,
        "question": "Which retrieval paradigm typically provides higher Recall when querying Knowledge Graph entities?",
        "options": [
            "A) Exact / Keyword Retrieval",
            "B) Probabilistic Ranked Retrieval",
            "C) Both methods always produce the same recall for compound queries",
            "D) Recall cannot be evaluated for candidate graph entities"
        ],
        "answer_index": 1,
        "explanation": "Probabilistic retrieval captures partial matches, token subsets, and fuzzy similarities, retrieving relevant entities that exact boolean filters omit (higher recall)."
    },
    {
        "id": 10,
        "question": "Why is 'graded relevance' advantageous over binary (0 or 1) match status in knowledge graph search?",
        "options": [
            "A) It allows the system to prioritize candidates that are strongly relevant over those that are only tangentially related",
            "B) It makes every candidate an exact match for the query",
            "C) It removes the need to identify candidate entities first",
            "D) It guarantees that the top candidate is a unique real-world entity"
        ],
        "answer_index": 0,
        "explanation": "Graded relevance scores (e.g. 0.95 vs 0.40) provide fine ranking discrimination, allowing users and downstream reasoning engines to prioritize high-confidence matches."
    }
]


# ======================================================================================
# 2. CORE ENGINES: ENTITY EXTRACTION & RETRIEVAL LOGIC
# ======================================================================================

def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts named entities from text using contextual rules, gazetteer matching,
    and regex patterns for candidate graph entity generation.
    Returns a structured list of entity dictionaries.
    """
    if not text or not text.strip():
        return []

    entities = []
    seen_spans = set()

    # 1. Gazetteer & multi-word match (longest first to avoid greedy subsets)
    sorted_known = sorted(KNOWN_GAZETTEER.keys(), key=lambda x: len(x), reverse=True)
    for term in sorted_known:
        pattern = r'\b' + re.escape(term) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = (match.start(), match.end())
            if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                seen_spans.add(span)
                original_str = text[match.start():match.end()]
                ent_type = KNOWN_GAZETTEER[term]
                c_start = max(0, match.start() - 25)
                c_end = min(len(text), match.end() + 25)
                entities.append({
                    "entity": original_str,
                    "type": ent_type,
                    "start": match.start(),
                    "end": match.end(),
                    "context": "..." + text[c_start:c_end].strip() + "..."
                })

    # 2. Rule & Regex pattern matching for organizations, institutions, and roles
    org_suffixes = r'(?:University|College|Institute|Corp|Corporation|Inc|Ltd|Company|Organization|Lab|Laboratory|Group|Foundation)'
    loc_suffixes = r'(?:City|State|Country|Street|Avenue|Park|Valley|Republic|Kingdom|Islands|Ocean|River)'
    role_words = r'(?:CEO|CTO|CFO|President|Founder|Director|Professor|Scientist|Engineer|Minister|Inventor|Lead|Chief|Chairman)'

    # Formal Organizations (e.g., Stanford University, MIT Institute)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+' + org_suffixes + r')\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "ORGANIZATION",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start()-20):min(len(text), m.end()+20)].strip() + "..."
            })

    # Location phrases (e.g., United Kingdom, Silicon Valley)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+' + loc_suffixes + r')\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "LOCATION",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start()-20):min(len(text), m.end()+20)].strip() + "..."
            })

    # Executive & Professional Roles
    for m in re.finditer(r'\b(' + role_words + r')\b', text, re.IGNORECASE):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": text[m.start():m.end()],
                "type": "ROLE_TITLE",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start()-20):min(len(text), m.end()+20)].strip() + "..."
            })

    # Multi-word Proper Nouns (Person names / entities)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entity_str = m.group(1)
            entities.append({
                "entity": entity_str,
                "type": "PERSON",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start()-20):min(len(text), m.end()+20)].strip() + "..."
            })

    # Single capitalized words appearing mid-sentence
    for m in re.finditer(r'(?<=[a-z0-9,]\s)([A-Z][a-z]{2,})\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entity_str = m.group(1)
            entities.append({
                "entity": entity_str,
                "type": "OTHER_ENTITY",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start()-20):min(len(text), m.end()+20)].strip() + "..."
            })

    # Sort sequentially by text appearance offset
    entities.sort(key=lambda x: x["start"])
    return entities


def compute_ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Calculates character n-gram Dice similarity coefficient between two strings."""
    s1_clean = s1.lower().strip()
    s2_clean = s2.lower().strip()
    if not s1_clean or not s2_clean:
        return 0.0
    if s1_clean == s2_clean:
        return 1.0

    s1_ng = [s1_clean[i:i+n] for i in range(len(s1_clean) - n + 1)] if len(s1_clean) >= n else [s1_clean]
    s2_ng = [s2_clean[i:i+n] for i in range(len(s2_clean) - n + 1)] if len(s2_clean) >= n else [s2_clean]

    s1_set = set(s1_ng)
    s2_set = set(s2_ng)

    overlap = len(s1_set.intersection(s2_set))
    total = len(s1_set) + len(s2_set)
    return (2.0 * overlap) / total if total > 0 else 0.0


def run_exact_retrieval(query: str, candidate_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Method 1: Exact / Keyword Retrieval Engine.
    Evaluates whether the candidate entity contains the exact query string or exact token match.
    Returns binary relevance scores (1.0 or 0.0).
    """
    q_clean = query.lower().strip()
    results = []

    for ent in candidate_entities:
        e_name = ent["entity"].lower().strip()
        e_tokens = [t for t in re.findall(r'\w+', e_name)]

        # Boolean match condition: exact string identity, whole-token match, or exact substring
        is_exact = (q_clean == e_name) or (q_clean in e_name) or any(q_clean == tok for tok in e_tokens)

        results.append({
            "entity": ent["entity"],
            "type": ent["type"],
            "exact_match": is_exact,
            "match_status": "MATCH FOUND" if is_exact else "NO MATCH",
            "score": 1.0 if is_exact else 0.0,
            "context": ent.get("context", "")
        })

    return results


def run_probabilistic_retrieval(query: str, candidate_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Method 2: Probabilistic Ranked Retrieval Engine.
    Calculates P(Relevant | Query, Candidate) based on:
      - Token Jaccard similarity
      - Query term coverage
      - Character n-gram fuzzy similarity
    - transparent token coverage, Jaccard, and n-gram evidence
    Ranks candidates in descending order of relevance score.
    """
    q_clean = query.lower().strip()
    q_tokens = set(re.findall(r'\w+', q_clean))
    results = []

    for ent in candidate_entities:
        e_name = ent["entity"].lower().strip()
        e_tokens = set(re.findall(r'\w+', e_name))

        # 1. Token Jaccard Probability
        intersection = q_tokens.intersection(e_tokens)
        union = q_tokens.union(e_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0

        # 2. Query Term Coverage
        coverage = len(intersection) / len(q_tokens) if q_tokens else 0.0

        # 3. Fuzzy n-gram String Similarity
        ngram_sim = compute_ngram_similarity(q_clean, e_name, n=2)

        # Weighted normalized relevance score, shown as evidence rather than probability.
        if q_clean == e_name:
            final_score = 1.0
        else:
            raw_score = (0.45 * coverage) + (0.30 * jaccard) + (0.25 * ngram_sim)
            # Bonus if query is contained as an exact substring
            if q_clean in e_name:
                raw_score = min(1.0, raw_score + 0.25)
            final_score = min(1.0, max(0.0, raw_score))

        calibrated_score = round(final_score, 4)

        # Qualitative confidence classification
        if calibrated_score >= 0.85:
            confidence = "High Confidence"
        elif calibrated_score >= 0.40:
            confidence = "Moderate Relevance"
        elif calibrated_score >= 0.15:
            confidence = "Marginal Association"
        else:
            confidence = "Non-Relevant"

        results.append({
            "entity": ent["entity"],
            "type": ent["type"],
            "relevance_score": calibrated_score,
            "confidence": confidence,
            "token_coverage": round(coverage, 2),
            "jaccard": round(jaccard, 2),
            "ngram_sim": round(ngram_sim, 2),
            "score_explanation": (
                f"Coverage {coverage:.2f} x 0.45 + token similarity {jaccard:.2f} x 0.30 + "
                f"n-gram similarity {ngram_sim:.2f} x 0.25"
            ),
            "context": ent.get("context", "")
        })

    # Sort descending by relevance score (Probability Ranking Principle)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    for idx, r in enumerate(results, 1):
        r["rank"] = idx

    return results


# ======================================================================================
# 3. LAB REPORT PDF EXPORTER
# ======================================================================================

def sanitize_pdf_text(text: str) -> str:
    """Replaces non-ASCII or unsupported characters for standard FPDF Helvetica font."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2192": "->",
        "\u2208": "in", "\u2229": "AND", "\u222a": "OR", "\u2264": "<=",
        "\u2265": ">=", "\u2260": "!=", "\u03b1": "alpha", "\u03b2": "beta",
        "\u2022": "-", "\u25cf": "*", "\xb7": "-", "\xb0": " deg"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("ascii", "replace").decode("ascii")


class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Knowledge Graph Information Retrieval System Virtual Lab", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment benchmark records into an official, formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Header & Subject
    pdf.set_text_color(30, 58, 138)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, sanitize_pdf_text(EXPERIMENT_CONFIG["title"]), align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 6, sanitize_pdf_text(f"Subject: {EXPERIMENT_CONFIG['subject']}"), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 27, 190, 22, "FD")

    pdf.set_xy(14, 29)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, sanitize_pdf_text(student_name or "N/A"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(46, 5, sanitize_pdf_text(student_id or "N/A"), 1)

    pdf.set_xy(14, 37)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, sanitize_pdf_text(date_str or datetime.now().strftime("%Y-%m-%d")), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pct = int((quiz_score / quiz_total) * 100 if quiz_total else 0)
    pdf.cell(46, 5, f"{quiz_score} / {quiz_total} ({pct}%)", 1)

    pdf.ln(15)

    # 1. Learning Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Experiment Objectives & Context", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = sanitize_pdf_text(obj)
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials (Exact vs. Probabilistic Retrieval)", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No experimental trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))

        for c in cols:
            pdf.cell(col_w, 6, sanitize_pdf_text(str(c))[:16], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, sanitize_pdf_text(val_str)[:16], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(5)

    # 3. Observations & Student Discussion
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Student Observations & Retrieval Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)

    notes_text = student_notes.strip() if student_notes.strip() else (
        "In this experiment, entity identification extracted candidate graph entities (Person, Organization, "
        "Location, Role) from unstructured text. Exact/keyword retrieval demonstrated zero recall on multi-word "
        "or partial queries (e.g. 'Google CEO'), while probabilistic ranked retrieval successfully computed "
        "token Jaccard probabilities and n-gram overlap to rank the most relevant candidate entities at the top."
    )
    pdf.multi_cell(0, 5, sanitize_pdf_text(notes_text))
    pdf.ln(10)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Lab Instructor / Evaluator Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. SECTION RENDERERS
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, and Procedure."""
    st.header("Theoretical Framework: Graph Entities & Retrieval")
    st.markdown(THEORY_CONTENT["background"])

    st.divider()
    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology & Theoretical Reference", expanded=False):
        terms_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Concept / Term", "Definition & Role in KGIRS"]
        )
        st.table(terms_df)


def render_candidate_graph(entities: List[Dict[str, Any]]):
    """Render a compact candidate-node graph without inferring relationships."""
    display_entities = entities[:8]
    if not display_entities:
        st.info("No candidate nodes are available yet.")
        return

    type_colors = {
        "PERSON": "#2563eb",
        "ORGANIZATION": "#059669",
        "LOCATION": "#d97706",
        "ROLE_TITLE": "#7c3aed",
        "TECH_CONCEPT": "#dc2626",
        "AWARD_EVENT": "#db2777",
        "OTHER_ENTITY": "#64748b",
    }
    x_positions = list(range(len(display_entities)))
    fig = go.Figure()
    for index, entity in enumerate(display_entities):
        fig.add_trace(go.Scatter(
            x=[x_positions[index]],
            y=[0],
            mode="markers+text",
            text=[f"{entity['entity']}<br><sup>{entity['type']}</sup>"],
            textposition="top center",
            marker={"size": 28, "color": type_colors.get(entity["type"], "#64748b"), "line": {"width": 2, "color": "white"}},
            hovertemplate=f"<b>{entity['entity']}</b><br>Type: {entity['type']}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        height=250,
        xaxis={"visible": False, "range": [-1, max(1, len(display_entities))]},
        yaxis={"visible": False, "range": [-1, 1]},
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        title="Candidate Graph (identified entity nodes)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="candidate_graph")
    if len(entities) > len(display_entities):
        st.caption(f"Showing the first {len(display_entities)} nodes to keep the candidate graph readable ({len(entities)} total candidates).")


def render_simulation_section():
    """Renders Section 2: Interactive Entity Identification & Comparative Retrieval."""
    st.header("Simulation: Identify, Retrieve, Compare")
    st.info("Laboratory sequence: Input -> Run identification -> Observe candidates -> Run retrieval -> Compare -> Understand")

    # Step 1: Input Text Configuration
    st.subheader("Step 1: Text Input")
    input_mode = st.radio(
        "Choose Text Input Method:",
        options=["Select from Benchmark Presets", "Enter Custom Unstructured Text"],
        horizontal=True
    )

    if input_mode == "Select from Benchmark Presets":
        selected_preset = st.selectbox(
            "Benchmark Domain Presets:",
            options=list(SAMPLE_PRESETS.keys()),
            index=0
        )
        active_text = SAMPLE_PRESETS[selected_preset]
        st.text_area("Selected Text Preview:", value=active_text, height=70, disabled=True)
    else:
        active_text = st.text_area(
            "Enter Custom Text (containing Persons, Organizations, Locations, Roles, etc.):",
            value=st.session_state.get("custom_text_val", "Sundar Pichai is the CEO of Google and studied at Stanford University in California."),
            height=90
        )
        st.session_state["custom_text_val"] = active_text

    text_signature = active_text.strip()
    if st.session_state.get("last_extracted_text") != text_signature:
        st.session_state["identified_entities"] = []
        st.session_state["retrieval_run"] = False

    # Step 2: Entity Identification Button
    col_btn, col_info = st.columns([1.5, 3.5])
    with col_btn:
        if st.button("Identify Graph Entities", type="primary", use_container_width=True):
            detected = extract_entities_from_text(active_text)
            st.session_state["identified_entities"] = detected
            st.session_state["last_extracted_text"] = text_signature
            st.session_state["retrieval_run"] = False
            st.toast(f"Successfully identified {len(detected)} candidate graph entities!")

    with col_info:
        if "identified_entities" in st.session_state and st.session_state["identified_entities"]:
            cnt = len(st.session_state["identified_entities"])
            types = set(e["type"] for e in st.session_state["identified_entities"])
            st.success(f"**Entity Identification Active**: {cnt} candidate entities detected across {len(types)} distinct types.")
        else:
            st.caption("Click **Identify Graph Entities** to create the candidate entity set.")

    # Automatically identify if not yet done for default text
    if "identified_entities" not in st.session_state:
        st.session_state["identified_entities"] = extract_entities_from_text(active_text)
        st.session_state["last_extracted_text"] = text_signature

    identified_entities = st.session_state.get("identified_entities", [])

    if not identified_entities:
        st.warning("No named entities identified in the text. Please ensure the text includes capitalized proper nouns, institutions, locations, or roles.")
        return

    # Display Identified Entities as Candidate Graph Entities
    st.divider()
    st.subheader("Step 3: Show Entity Types and Candidate Graph")
    st.caption("Each extracted mention is a candidate node. The graph is intentionally small and displays nodes only; no relationships are inferred.")

    ent_df = pd.DataFrame([
        {
            "Entity Mention": e["entity"],
            "Entity Type": e["type"],
            "Span [Start, End]": f"[{e['start']}, {e['end']}]",
            "Mention Context Window": e["context"]
        } for e in identified_entities
    ])
    st.dataframe(ent_df, use_container_width=True, hide_index=True)
    render_candidate_graph(identified_entities)

    # Step 3: Query & Dual Retrieval Execution
    st.divider()
    st.subheader("Step 4: Enter Query and Run Retrieval")
    st.write("Enter a query, then run both retrieval methods over the same candidate set.")

    col_q1, col_q2 = st.columns([3, 2])
    with col_q1:
        query_input = st.text_input(
            "Search Query:",
            value=st.session_state.get("active_query", "Google"),
            placeholder="e.g., Google, Sundar Pichai, CEO, Stanford, California, Google CEO"
        )
        st.session_state["active_query"] = query_input

    with col_q2:
        st.caption("Quick-Test Query Suggestions:")
        # Generate quick query options based on entities found
        sample_queries = [identified_entities[0]["entity"]] if identified_entities else ["Google"]
        if len(identified_entities) > 1:
            sample_queries.append(identified_entities[1]["entity"])
        sample_queries.extend(["Google CEO", "Sundar", "Stanford", "CEO"])
        # Dedup preserving order
        unique_samples = list(dict.fromkeys(sample_queries))[:4]

        quick_cols = st.columns(len(unique_samples))
        for idx, sq in enumerate(unique_samples):
            with quick_cols[idx]:
                if st.button(sq, key=f"quick_q_{idx}", use_container_width=True):
                    st.session_state["active_query"] = sq
                    st.rerun()

    query = st.session_state.get("active_query", "Google").strip()

    if st.button("Run Retrieval Experiment", type="primary", use_container_width=True):
        st.session_state["retrieval_run"] = True

    if not st.session_state.get("retrieval_run", False):
        st.info("Retrieval results will appear here after you click **Run Retrieval Experiment**.")
        return

    # Step 5: Run Both Retrieval Methods
    exact_results = run_exact_retrieval(query, identified_entities)
    prob_results = run_probabilistic_retrieval(query, identified_entities)

    # Step 5: Side-by-Side Display of Methods
    st.divider()
    st.subheader("Step 5: Scores and Ranking")

    col_exact, col_prob = st.columns(2)

    with col_exact:
        st.markdown("### Method 1: Exact / Keyword Retrieval")
        st.caption("Deterministic boolean matching. Score is binary (1.0 = Match, 0.0 = No Match).")

        exact_matches = [r for r in exact_results if r["exact_match"]]
        st.metric("Exact Matches Found", f"{len(exact_matches)} / {len(exact_results)}")

        exact_display_df = pd.DataFrame([
            {
                "Candidate Entity": r["entity"],
                "Type": r["type"],
                "Match Status": r["match_status"],
                "Binary Score": r["score"]
            } for r in exact_results
        ])
        st.dataframe(exact_display_df, use_container_width=True, hide_index=True)

        if not exact_matches:
            st.error(f"**Zero Exact Matches**: The exact query '{query}' does not strictly match any candidate entity name as a whole string or token.")
        else:
            st.success(f"Exact match found for: **{', '.join([m['entity'] for m in exact_matches])}**")

    with col_prob:
        st.markdown("### Method 2: Probabilistic Ranked Retrieval")
        st.caption("Ranks candidates using a normalized Probabilistic Relevance Score based on observable text evidence.")

        top_prob = prob_results[0] if prob_results else None
        st.metric(
            "Top Ranked Entity",
            f"{top_prob['entity'] if top_prob else 'None'}",
            delta=f"Relevance: {top_prob['relevance_score'] if top_prob else 0.0}"
        )

        prob_display_df = pd.DataFrame([
            {
                "Rank": f"#{r['rank']}",
                "Candidate Entity": r["entity"],
                "Type": r["type"],
                "Relevance Score": f"{r['relevance_score']:.4f}",
                "Confidence Level": r["confidence"]
            } for r in prob_results
        ])
        st.dataframe(prob_display_df, use_container_width=True, hide_index=True)

        st.caption("Score evidence for each candidate")
        factor_df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Candidate": r["entity"],
                "Query Coverage": r["token_coverage"],
                "Token Similarity (Jaccard)": r["jaccard"],
                "N-gram Similarity": r["ngram_sim"],
                "Why this score": r["score_explanation"],
            } for r in prob_results
        ])
        st.dataframe(factor_df, use_container_width=True, hide_index=True)

    # Step 6: Visual Comparison of Relevance Scores
    st.divider()
    st.subheader("Step 6: Compare Both Methods")
    st.caption("Direct visual comparison between Exact Retrieval (Binary 0 or 1) and Probabilistic Relevance Scores across all candidate entities.")

    # Create Plotly Bar Chart
    entities_names = [r["entity"] for r in prob_results]
    prob_scores = [r["relevance_score"] for r in prob_results]

    # Map exact scores to the same entity order as prob_results
    exact_score_map = {r["entity"]: r["score"] for r in exact_results}
    exact_scores = [exact_score_map.get(name, 0.0) for name in entities_names]

    fig = go.Figure()

    # Exact Retrieval Bars
    fig.add_trace(go.Bar(
        name="Exact / Keyword Retrieval (Binary)",
        x=entities_names,
        y=exact_scores,
        marker_color="#94a3b8",
        opacity=0.75,
        text=[f"{s:.1f}" for s in exact_scores],
        textposition="auto"
    ))

    # Probabilistic Retrieval Bars
    fig.add_trace(go.Bar(
        name="Probabilistic Ranked Retrieval (PRP Score)",
        x=entities_names,
        y=prob_scores,
        marker_color="#2563eb",
        opacity=0.9,
        text=[f"{s:.4f}" for s in prob_scores],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"Comparative Retrieval Score Distribution for Query: '{query}'",
        xaxis_title="Candidate Graph Entities (Ordered by Probabilistic Rank)",
        yaxis_title="Retrieval Score [0.0 - 1.0]",
        yaxis=dict(range=[0, 1.15], dtick=0.2),
        barmode="group",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Step 7: Automated Analytical Comparison Insights
    with st.expander("Comparative Retrieval Performance Analysis", expanded=True):
        observation = (
            f"Exact retrieval returned {len(exact_matches)} match(es), while ranked retrieval placed "
            f"'{top_prob['entity'] if top_prob else 'no candidate'}' first with a relevance score of "
            f"{top_prob['relevance_score'] if top_prob else 0.0:.4f}. "
            "The ranked method preserves partial evidence through coverage, token similarity, and n-gram similarity."
        )
        st.success(f"**Observation:** {observation}")
        st.markdown(f"""
**Key Experimental Observations for Query: `{query}`**
1. **Ranking Discriminability**:
   - **Exact Retrieval**: Produces a flat, binary outcome ($\\{{0.0, 1.0\\}}$). Non-matching entities are indiscriminately assigned $0.0$, offering zero distinction between partially relevant and completely irrelevant candidates.
    - **Probabilistic Retrieval**: Produces a continuous, graded relevance spectrum ($[0.0, 1.0]$). It successfully discriminates the top match (**{top_prob['entity']}**, Score: `{top_prob['relevance_score']:.4f}`) from lower-ranked candidates.
2. **Compound / Multi-Word Query Behavior**:
   - When searching multi-word queries like `"Google CEO"`, exact string match fails completely because no single entity has that combined name.
   - Probabilistic retrieval separates the query into constituent tokens (`"google"`, `"ceo"`), scoring coverage against both *Google* (Organization) and *CEO* (Role/Title).
3. **Fuzzy & Substring Resilience**:
   - Queries matching partial names (e.g., `"Sundar"` for *Sundar Pichai* or `"Stanford"` for *Stanford University*) maintain high probabilistic relevance while exact keyword matching may fail if strict equality is enforced.
""")

    # Step 8: Trial Data Logger
    st.divider()
    st.subheader("Step 8: Experimental Data Log Book")
    col_log_act, col_log_table = st.columns([1.5, 3.5])

    with col_log_act:
        st.caption("Record this query trial into your session logbook:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            top_entity_name = top_prob["entity"] if top_prob else "N/A"
            top_entity_score = top_prob["relevance_score"] if top_prob else 0.0
            exact_count = len(exact_matches)

            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query": query,
                "Entities Identified": len(identified_entities),
                "Exact Matches": exact_count,
                "Top Ranked Entity": top_entity_name,
                "Relevance Score": top_entity_score,
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully logged!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Experimental trial log cleared.")

    with col_log_table:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_bytes = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_bytes,
                file_name="experiment_6_entity_retrieval_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to capture your experimental retrieval results.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Concept Assessment Quiz: Graph Entities & Retrieval")
    st.write("Complete the 10 conceptual questions below to assess your understanding of entity identification, candidate generation, and retrieval models.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Detailed Rationale")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        pct = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({pct:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already completed. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Lab Report Generation")
    st.write("Compile your student details, recorded retrieval trials, and quiz evaluation into an official PDF report.")

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

    st.subheader("Discussion & Experimental Observations")
    default_discussion = (
        "In this experiment, entity identification extracted candidate graph entities (Person, Organization, "
        "Location, Role) from unstructured text. Exact/keyword retrieval demonstrated zero recall on multi-word "
        "or partial queries (e.g. 'Google CEO'), while probabilistic ranked retrieval successfully computed "
        "token Jaccard probabilities and n-gram overlap to rank the most relevant candidate entities at the top."
    )
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", default_discussion),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['title']} ({EXPERIMENT_CONFIG['subject']})")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: No trials recorded in the Simulation section yet. The report will show 0 recorded trials.")

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    # Save to local files for guaranteed download
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report_exp6_graph_entities.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


# ======================================================================================
# 5. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
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
    if "active_query" not in st.session_state:
        st.session_state["active_query"] = "Google"


def main():
    st.set_page_config(
        page_title="Experiment 6: Identify Graph Entities | KGIRS Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # Title Banner
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(f"Virtual Laboratory: **{EXPERIMENT_CONFIG['subject']}**")

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")
    st.sidebar.write(f"- **Trials Logged:** `{len(st.session_state.get('trials', []))}`")

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
