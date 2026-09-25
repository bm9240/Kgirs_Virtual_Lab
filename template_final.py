"""
IIT Kharagpur Virtual Laboratory
Discipline: Computer Science and Engineering
Subject: Knowledge Graph and Information Retrieval System (KGIRS)
Experiment 6: Identify Graph Entities & Probabilistic Retrieval Performance

Sections:
  1. Purpose
  2. Theory
  3. Simulation
  4. Quiz
  5. Report Generation
  6. Certificate
  7. References

Designed following IIT Kharagpur Virtual Lab format and the modular architecture of template.py.
"""

import os
import math
import re
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
import networkx as nx


# ======================================================================================
# 1. EXPERIMENT METADATA & EDUCATIONAL CONTENT (IIT KGP VLAB STANDARD)
# ======================================================================================

EXPERIMENT_CONFIG = {
    "exp_number": 6,
    "title": "Identify Graph Entities & Probabilistic Retrieval Performance",
    "learning_unit": "KGIRS MODULE 1 & MODULE 2: STRUCTURED KNOWLEDGE EXTRACTION & PROBABILISTIC IR RANKING",
    "discipline": "Computer Science and Engineering",
    "subject": "Knowledge Graph and Information Retrieval System (KGIRS)",
    "institute": "Indian Institute of Technology Kharagpur (IIT KGP)",
    "objectives": [
        "Extract domain-specific entities (Persons, Organizations, Locations, Concepts) from unstructured text to build structured knowledge (Module 1.2).",
        "Construct an interactive Knowledge Graph topology using relational triples with fewer, focused nodes for semantic clarity.",
        "Formulate and evaluate probabilistic retrieval principles using the Probabilistic Relevance Framework / Okapi BM25 Model (Module 2.1).",
        "Benchmark comparative retrieval performance (Precision@K, Recall@K, MAP, NDCG) between baseline keyword search and entity-aware probabilistic ranking."
    ]
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
    )
}

KNOWN_GAZETTEER = {
    "Sundar Pichai": "PERSON",
    "Alan Turing": "PERSON",
    "Satya Nadella": "PERSON",
    "Marie Curie": "PERSON",
    "Google": "ORGANIZATION",
    "Stanford University": "ORGANIZATION",
    "Microsoft": "ORGANIZATION",
    "OpenAI": "ORGANIZATION",
    "University of Paris": "ORGANIZATION",
    "Bletchley Park": "ORGANIZATION",
    "California": "LOCATION",
    "United Kingdom": "LOCATION",
    "San Francisco": "LOCATION",
    "Stockholm": "LOCATION",
    "Sweden": "LOCATION",
    "Paris": "LOCATION",
    "CEO": "ROLE_TITLE",
    "Artificial Intelligence": "CONCEPT",
    "World Wide Web": "CONCEPT",
    "Enigma machine": "CONCEPT",
    "Nobel Prizes": "AWARD",
    "radioactivity": "CONCEPT",
    "CERN": "ORGANIZATION",
    "Geneva": "LOCATION",
    "Switzerland": "LOCATION"
}

# Quiz Question Bank: the Quiz section randomly samples 10 of these each time,
# so pressing "Get New Question Set" produces a fresh mix of questions.
QUIZ_QUESTION_BANK = [
    {
        "id": 1,
        "question": "Which of the following entity categories correctly classifies 'Cortex Labs' and 'Arclight AI'?",
        "options": [
            "A) LOCATION (LOC)",
            "B) ORGANIZATION (ORG)",
            "C) PERSON (PER)",
            "D) TEMPORAL (DATE)"
        ],
        "answer_index": 1,
        "explanation": "'Cortex Labs' and 'Arclight AI' are research labs and corporate entities, classified under the ORGANIZATION (ORG) category."
    },
    {
        "id": 2,
        "question": "What is the primary role of the parameter k1 in the Okapi BM25 scoring formula?",
        "options": [
            "A) It controls the term frequency saturation non-linearity (how fast score plateaus with repeated term occurrences)",
            "B) It sets the absolute document length in bytes",
            "C) It determines the network port used for graph database connections",
            "D) It eliminates all stop words from the query vector"
        ],
        "answer_index": 0,
        "explanation": "The parameter k1 (typically between 1.2 and 2.0) calibrates term frequency saturation; higher values allow repeated terms to contribute more to the score before leveling off."
    },
    {
        "id": 3,
        "question": "In BM25, setting the document length normalization parameter b = 0 causes the model to:",
        "options": [
            "A) Completely disable term frequency weighting",
            "B) Completely eliminate document length normalization (short and long documents treated equally)",
            "C) Invert the ranking order from worst to best",
            "D) Restrict retrieval to only single-word documents"
        ],
        "answer_index": 1,
        "explanation": "Parameter b controls length normalization; when b = 0, no length penalty is applied, whereas b = 1 applies full scaling by document length relative to average document length."
    },
    {
        "id": 4,
        "question": "How does incorporating Knowledge Graph entities improve probabilistic document ranking over pure BM25?",
        "options": [
            "A) By converting all natural language text into binary machine code",
            "B) By boosting documents that share disambiguated entities and relational context with query concepts",
            "C) By guaranteeing that every search query returns 100% of all documents in the corpus",
            "D) By bypassing the index and reading directly from disk linearly"
        ],
        "answer_index": 1,
        "explanation": "Entity-aware ranking leverages structured semantics: matching query entities against verified graph entities in documents overcomes surface-level lexical mismatch and improves ranking precision."
    },
    {
        "id": 5,
        "question": "What does Mean Average Precision (MAP) measure in an Information Retrieval evaluation benchmark?",
        "options": [
            "A) The average file size of relevant documents in kilobytes",
            "B) The mean of the Average Precision scores evaluated across a set of queries, emphasizing top-ranked relevant hits",
            "C) The clock execution time taken by the ranking algorithm in milliseconds",
            "D) The percentage of hardware RAM utilized during graph construction"
        ],
        "answer_index": 1,
        "explanation": "MAP evaluates the quality of ranked lists by averaging precision at each relevant document rank across multiple queries, heavily penalizing relevant documents ranked far down."
    },
    {
        "id": 6,
        "question": "Which evaluation metric incorporates graded relevance judgments with logarithmic positional discounting?",
        "options": [
            "A) Normalized Discounted Cumulative Gain (NDCG)",
            "B) Raw Term Frequency (TF)",
            "C) Graph Degree Centrality",
            "D) Corpus Vocabulary Count"
        ],
        "answer_index": 0,
        "explanation": "NDCG uses graded relevance levels and applies a logarithmic discount (1/log2(rank+1)) to prioritize placing highly relevant documents near the very top."
    },
    {
        "id": 7,
        "question": "In graph entity extraction, what problem does Entity Disambiguation (Entity Linking) resolve?",
        "options": [
            "A) Determining whether an entity mention like 'Paris' refers to the city of Paris or a person named Paris",
            "B) Formatting JSON files into comma-separated text files",
            "C) Deleting stop words from the query string",
            "D) Preventing network graph edges from crossing each other"
        ],
        "answer_index": 0,
        "explanation": "Entity Linking maps ambiguous textual surface mentions to canonical Knowledge Graph nodes based on surrounding semantic context."
    },
    {
        "id": 8,
        "question": "In our Knowledge Graph visualization with fewer nodes, what do the nodes and directed edges represent?",
        "options": [
            "A) Nodes represent documents; edges represent file download speeds",
            "B) Nodes represent identified domain entities; edges represent typed semantic relationships between them",
            "C) Nodes represent pixels; edges represent color gradients",
            "D) Nodes represent database servers; edges represent network latency"
        ],
        "answer_index": 1,
        "explanation": "In an entity knowledge graph, vertices represent entities (Person, Org, Loc, Concept) and directed edges signify relations such as 'affiliated_with' or 'developed'."
    },
    {
        "id": 9,
        "question": "What happens to the Precision@K metric if the top K retrieved documents contain 2 relevant items when K = 4?",
        "options": [
            "A) Precision@4 = 0.25",
            "B) Precision@4 = 0.50 (50%)",
            "C) Precision@4 = 0.75",
            "D) Precision@4 = 1.00"
        ],
        "answer_index": 1,
        "explanation": "Precision@K = (Number of relevant documents in top K) / K. Here, 2 / 4 = 0.50 or 50%."
    },
    {
        "id": 10,
        "question": "When designing knowledge graph experiments with fewer nodes, why is node sparsity advantageous in virtual lab interfaces?",
        "options": [
            "A) It prevents visual cognitive overload, ensuring students can clearly trace entity relations and semantic paths",
            "B) It makes the web page load slower to test student patience",
            "C) Sparse graphs are required because Streamlit cannot display more than 3 colors",
            "D) Graphs with more than 10 nodes are illegal in virtual labs"
        ],
        "answer_index": 0,
        "explanation": "Targeted graphs with 8-15 nodes maintain visual clarity, allowing learners to clearly trace entity types, relationships, and degree centrality without confusing cluttered layouts."
    },
    {
        "id": 11,
        "question": "Which of the following best defines a Named Entity in knowledge graph construction?",
        "options": [
            "A) Any arbitrary stopword or punctuation mark in a sentence",
            "B) A real-world object or abstract concept with distinct identity and semantic type (e.g., Person, Org, Loc)",
            "C) A mathematical syntax error occurring during text tokenization",
            "D) A random float value assigned to word frequency counters"
        ],
        "answer_index": 1,
        "explanation": "A Named Entity represents a discrete real-world entity (such as an individual, company, place, or concept) that can be linked to a node in a Knowledge Graph."
    },
    {
        "id": 12,
        "question": "What is the primary limitation of pure keyword-based Bag-of-Words (BoW) retrieval?",
        "options": [
            "A) It cannot store strings in computer memory",
            "B) It ignores word polysemy, synonymy, semantic entity types, and relational context",
            "C) It executes too quickly to calculate relevance scores",
            "D) It only processes numerical equations rather than natural language"
        ],
        "answer_index": 1,
        "explanation": "Bag-of-Words models treat text as unorganized tokens, failing to recognize that a term like 'Mercury' could mean a planet or an organization, and missing underlying entity relationships."
    },
    {
        "id": 13,
        "question": "According to the Probability Ranking Principle (PRP) formulated by Robertson (1977), how should documents be ordered?",
        "options": [
            "A) In random order to ensure statistical variance",
            "B) In decreasing order of their estimated probability of relevance to the information need",
            "C) Alphabetically by document author name",
            "D) Strictly by increasing order of document length in bytes"
        ],
        "answer_index": 1,
        "explanation": "The Probability Ranking Principle asserts that overall retrieval effectiveness is maximized when documents are returned in decreasing order of their estimated probability of relevance."
    },
    {
        "id": 14,
        "question": "In a Knowledge Graph, how is factual information structured at the foundational level?",
        "options": [
            "A) As unstructured binary blobs",
            "B) As relational subject-predicate-object (head, relation, tail) triples",
            "C) As isolated single float values",
            "D) As recursive HTML document trees without attributes"
        ],
        "answer_index": 1,
        "explanation": "Knowledge Graphs represent domain facts using directed relational triples (head entity, relation predicate, tail entity), e.g., (Elena Voss, affiliated_with, Nimbus Labs)."
    },
    {
        "id": 15,
        "question": "In the Okapi BM25 retrieval model, what purpose does the Inverse Document Frequency (IDF) factor serve?",
        "options": [
            "A) It penalizes rare, highly informative terms",
            "B) It assigns higher discriminative weight to terms that appear in fewer documents across the collection",
            "C) It sets all term frequencies to a constant value of 1.0",
            "D) It deletes documents whose length exceeds the average"
        ],
        "answer_index": 1,
        "explanation": "IDF penalizes widespread common words (like 'the', 'system') while giving high discriminative weight to rare, salient terms and specific entities."
    },
    {
        "id": 16,
        "question": "In the Vector Space Model vs. the Probabilistic Model of Information Retrieval, what distinguishes the probabilistic approach?",
        "options": [
            "A) It represents documents as points in Euclidean space ranked by cosine similarity",
            "B) It estimates and ranks documents by their probability of relevance to the query, grounded in relevance theory",
            "C) It requires no term weighting scheme at all",
            "D) It only works for numeric datasets, not natural language text"
        ],
        "answer_index": 1,
        "explanation": "Unlike the geometric Vector Space Model, the Probabilistic Model explicitly estimates P(relevance | document, query) and ranks accordingly, per the Probability Ranking Principle."
    },
    {
        "id": 17,
        "question": "What does Recall@K measure in a retrieval evaluation?",
        "options": [
            "A) The fraction of the top K retrieved documents that are relevant",
            "B) The fraction of all relevant documents in the collection that were successfully retrieved within the top K",
            "C) The average time taken to retrieve K documents",
            "D) The number of irrelevant documents mistakenly excluded from the corpus"
        ],
        "answer_index": 1,
        "explanation": "Recall@K = (Relevant documents retrieved in top K) / (Total relevant documents in the collection), measuring retrieval completeness rather than precision."
    },
    {
        "id": 18,
        "question": "What is the maximum possible value of NDCG@K, and what does achieving it indicate?",
        "options": [
            "A) NDCG@K can exceed 1.0 when all documents are relevant",
            "B) NDCG@K = 1.0, indicating the ranking matches the ideal order of relevance",
            "C) NDCG@K is unbounded and has no maximum",
            "D) NDCG@K = K, one point per correctly ranked document"
        ],
        "answer_index": 1,
        "explanation": "NDCG@K is normalized against the Ideal DCG@K, so a perfect ranking (most relevant documents placed highest) always yields NDCG@K = 1.0."
    },
    {
        "id": 19,
        "question": "As the Entity Boost Factor (alpha) is increased significantly in the entity-aware scoring formula, what effect does this have on ranking?",
        "options": [
            "A) It has no effect since alpha only scales document length",
            "B) Entity/relational matches increasingly dominate the final score, potentially overriding lexical BM25 differences",
            "C) It reduces the influence of all query terms to zero immediately",
            "D) It converts the ranking algorithm into a purely alphabetical sort"
        ],
        "answer_index": 1,
        "explanation": "Since the final score is BM25 + alpha * entity_salience, increasing alpha proportionally increases the weight of verified entity/relation matches relative to the lexical BM25 component."
    },
    {
        "id": 20,
        "question": "Why does combining structured Knowledge Graph triples with lexical BM25 scoring create a 'hybrid search' system?",
        "options": [
            "A) Because it merges two unrelated database engines into a single binary file",
            "B) Because it combines unstructured lexical term-matching signals with structured, verified relational entity signals in one ranking score",
            "C) Because it retrieves documents from two separate physical servers simultaneously",
            "D) Because it hybridizes image search with text search"
        ],
        "answer_index": 1,
        "explanation": "Hybrid search fuses lexical evidence (term frequency/IDF) with structured semantic evidence (verified entities and relations), addressing weaknesses that either signal alone would miss."
    }
]


# ======================================================================================
# 2. CURATED DOMAIN CORPORA & KNOWLEDGE BASE
# ======================================================================================

DATA_CORPORA = {
    "AI & Deep Learning Pioneers (Domain 1)": {
        "description": "Corpus on fictional Artificial Intelligence researchers, invented research institutions, breakthrough algorithms, and global research hubs.",
        "documents": [
            {
                "doc_id": "DOC-101",
                "title": "Deep Learning Breakthroughs at Lakeside University",
                "text": "Elena Voss and her research group at Lakeside University pioneered Deep Learning and artificial neural network backpropagation. Voss later joined Nimbus AI Labs to scale distributed neural representations in Toronto.",
                "entities": [
                    {"name": "Elena Voss", "type": "PERSON"},
                    {"name": "Lakeside University", "type": "ORGANIZATION"},
                    {"name": "Deep Learning", "type": "CONCEPT"},
                    {"name": "Nimbus AI Labs", "type": "ORGANIZATION"},
                    {"name": "Toronto", "type": "LOCATION"}
                ],
                "triples": [
                    ("Elena Voss", "affiliated_with", "Lakeside University"),
                    ("Elena Voss", "pioneered", "Deep Learning"),
                    ("Elena Voss", "joined", "Nimbus AI Labs"),
                    ("Lakeside University", "located_in", "Toronto")
                ],
                "relevant_to": ["deep learning", "elena voss", "toronto neural network", "nimbus ai labs pioneer"]
            },
            {
                "doc_id": "DOC-102",
                "title": "Convolutional Neural Networks and Horizon AI Research",
                "text": "Marcus Lindqvist developed Convolutional Neural Networks for computer vision at Hudson University. Lindqvist subsequently became Chief AI Scientist at Horizon AI in New York, collaborating on self-supervised machine learning.",
                "entities": [
                    {"name": "Marcus Lindqvist", "type": "PERSON"},
                    {"name": "Convolutional Networks", "type": "CONCEPT"},
                    {"name": "Hudson University", "type": "ORGANIZATION"},
                    {"name": "Horizon AI", "type": "ORGANIZATION"},
                    {"name": "New York", "type": "LOCATION"}
                ],
                "triples": [
                    ("Marcus Lindqvist", "affiliated_with", "Hudson University"),
                    ("Marcus Lindqvist", "developed", "Convolutional Networks"),
                    ("Marcus Lindqvist", "leads", "Horizon AI"),
                    ("Horizon AI", "located_in", "New York")
                ],
                "relevant_to": ["marcus lindqvist", "convolutional networks", "horizon ai research", "computer vision new york"]
            },
            {
                "doc_id": "DOC-103",
                "title": "Cortex Labs and Reinforcement Learning in London",
                "text": "Kavi Rajan co-founded Cortex Labs in London, revolutionizing Deep Reinforcement Learning with StrategoNet and ProteoMind. Cortex Labs was acquired by Nimbus Corp, strengthening the London artificial intelligence ecosystem.",
                "entities": [
                    {"name": "Kavi Rajan", "type": "PERSON"},
                    {"name": "Cortex Labs", "type": "ORGANIZATION"},
                    {"name": "Reinforcement Learning", "type": "CONCEPT"},
                    {"name": "Nimbus Corp", "type": "ORGANIZATION"},
                    {"name": "London", "type": "LOCATION"}
                ],
                "triples": [
                    ("Kavi Rajan", "founded", "Cortex Labs"),
                    ("Kavi Rajan", "pioneered", "Reinforcement Learning"),
                    ("Cortex Labs", "acquired_by", "Nimbus Corp"),
                    ("Cortex Labs", "located_in", "London")
                ],
                "relevant_to": ["cortex labs", "kavi rajan", "reinforcement learning", "london ai nimbus corp", "strategonet"]
            },
            {
                "doc_id": "DOC-104",
                "title": "Generative Pre-trained Transformers at Arclight AI",
                "text": "Ethan Cole leads Arclight AI in San Francisco, which introduced Generative Transformers and large language models. Arclight AI partnered with Meridian Systems to deploy generative intelligence across enterprise cloud architectures.",
                "entities": [
                    {"name": "Ethan Cole", "type": "PERSON"},
                    {"name": "Arclight AI", "type": "ORGANIZATION"},
                    {"name": "Generative Transformers", "type": "CONCEPT"},
                    {"name": "Meridian Systems", "type": "ORGANIZATION"},
                    {"name": "San Francisco", "type": "LOCATION"}
                ],
                "triples": [
                    ("Ethan Cole", "leads", "Arclight AI"),
                    ("Arclight AI", "developed", "Generative Transformers"),
                    ("Arclight AI", "partnered_with", "Meridian Systems"),
                    ("Arclight AI", "located_in", "San Francisco")
                ],
                "relevant_to": ["ethan cole", "arclight ai", "generative transformers", "san francisco cloud meridian systems"]
            },
            {
                "doc_id": "DOC-105",
                "title": "Aurora Institute for Learning Algorithms and Renata Okafor",
                "text": "Renata Okafor founded the Aurora Institute in Montreal to advance Deep Learning and generative adversarial modeling. Okafor collaborates with Riverside University to promote ethical artificial intelligence frameworks.",
                "entities": [
                    {"name": "Renata Okafor", "type": "PERSON"},
                    {"name": "Aurora Institute", "type": "ORGANIZATION"},
                    {"name": "Deep Learning", "type": "CONCEPT"},
                    {"name": "Montreal", "type": "LOCATION"}
                ],
                "triples": [
                    ("Renata Okafor", "founded", "Aurora Institute"),
                    ("Renata Okafor", "researches", "Deep Learning"),
                    ("Aurora Institute", "located_in", "Montreal")
                ],
                "relevant_to": ["renata okafor", "deep learning", "montreal aurora institute", "neural models"]
            }
        ]
    },
    "Enterprise Cloud & Systems (Domain 2)": {
        "description": "Corpus on fictional enterprise operating platforms, distributed cloud infrastructure, leadership, and headquarters.",
        "documents": [
            {
                "doc_id": "DOC-201",
                "title": "Meridian Systems Cloud Platform Transformation",
                "text": "Priya Anand directs Meridian Systems from Redmond, steering the growth of the Meridian Cloud ecosystem. Meridian Systems integrates Distributed Computing to support scalable hybrid enterprise infrastructures.",
                "entities": [
                    {"name": "Priya Anand", "type": "PERSON"},
                    {"name": "Meridian Systems", "type": "ORGANIZATION"},
                    {"name": "Meridian Cloud", "type": "CONCEPT"},
                    {"name": "Redmond", "type": "LOCATION"},
                    {"name": "Distributed Computing", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Priya Anand", "leads", "Meridian Systems"),
                    ("Meridian Systems", "developed", "Meridian Cloud"),
                    ("Meridian Systems", "located_in", "Redmond"),
                    ("Meridian Cloud", "implements", "Distributed Computing")
                ],
                "relevant_to": ["priya anand", "meridian systems cloud", "redmond cloud", "distributed computing"]
            },
            {
                "doc_id": "DOC-202",
                "title": "Vantage Cloud Services and Global Cloud Scale",
                "text": "Derek Simmons established Vantage Cloud Services in Seattle, deploying Elastic Cloud architecture worldwide. Vantage Cloud Services delivers Cloud Virtualization for millions of enterprise software applications.",
                "entities": [
                    {"name": "Derek Simmons", "type": "PERSON"},
                    {"name": "Vantage Cloud Services", "type": "ORGANIZATION"},
                    {"name": "Cloud Virtualization", "type": "CONCEPT"},
                    {"name": "Seattle", "type": "LOCATION"}
                ],
                "triples": [
                    ("Derek Simmons", "leads", "Vantage Cloud Services"),
                    ("Vantage Cloud Services", "engineered", "Cloud Virtualization"),
                    ("Vantage Cloud Services", "located_in", "Seattle")
                ],
                "relevant_to": ["derek simmons", "vantage cloud services", "seattle cloud virtualization"]
            },
            {
                "doc_id": "DOC-203",
                "title": "Nimbus Cloud Platform and Distributed Data Processing",
                "text": "Arjun Mehta leads Nimbus Corp in Mountain View, advancing Nimbus Cloud and container orchestration technology. Nimbus Corp builds Distributed Computing systems for global web infrastructure.",
                "entities": [
                    {"name": "Arjun Mehta", "type": "PERSON"},
                    {"name": "Nimbus Corp", "type": "ORGANIZATION"},
                    {"name": "Nimbus Cloud", "type": "CONCEPT"},
                    {"name": "Mountain View", "type": "LOCATION"},
                    {"name": "Distributed Computing", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Arjun Mehta", "leads", "Nimbus Corp"),
                    ("Nimbus Corp", "operates", "Nimbus Cloud"),
                    ("Nimbus Corp", "located_in", "Mountain View"),
                    ("Nimbus Cloud", "utilizes", "Distributed Computing")
                ],
                "relevant_to": ["arjun mehta", "nimbus cloud", "mountain view", "distributed computing"]
            },
            {
                "doc_id": "DOC-204",
                "title": "Open Kernel Alliance and Open Source Operating Kernels",
                "text": "Viktor Petrov created the Solstice Kernel, supported by the Open Kernel Alliance in San Francisco. The Solstice Kernel powers modern Cloud Virtualization across major corporate data centers.",
                "entities": [
                    {"name": "Viktor Petrov", "type": "PERSON"},
                    {"name": "Open Kernel Alliance", "type": "ORGANIZATION"},
                    {"name": "Solstice Kernel", "type": "CONCEPT"},
                    {"name": "San Francisco", "type": "LOCATION"},
                    {"name": "Cloud Virtualization", "type": "CONCEPT"}
                ],
                "triples": [
                    ("Viktor Petrov", "created", "Solstice Kernel"),
                    ("Open Kernel Alliance", "located_in", "San Francisco"),
                    ("Solstice Kernel", "enables", "Cloud Virtualization")
                ],
                "relevant_to": ["viktor petrov", "solstice kernel", "san francisco", "cloud virtualization"]
            }
        ]
    }
}


# ======================================================================================
# 3. GRAPH & PROBABILISTIC RANKING ENGINE
# ======================================================================================

def tokenize(text: str) -> list:
    """Basic lowercased alphanumeric tokenizer."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def compute_bm25_score(query_tokens: list, doc_tokens: list, doc_len: int,
                       avg_doc_len: float, idf_dict: dict, k1: float, b: float) -> float:
    """Calculates Okapi BM25 score for a document against query tokens."""
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


def build_knowledge_graph(corpus_docs: list):
    """
    Constructs a NetworkX graph with fewer, focused nodes (8-15 nodes max)
    representing Persons, Organizations, Locations, and Concepts.
    """
    G = nx.DiGraph()
    entity_metadata = {}

    for doc in corpus_docs:
        for ent in doc["entities"]:
            ename = ent["name"]
            etype = ent["type"]
            if not G.has_node(ename):
                G.add_node(ename, type=etype, docs=set())
            G.nodes[ename]["docs"].add(doc["doc_id"])
            entity_metadata[ename] = etype

        for h, r, t in doc.get("triples", []):
            if G.has_node(h) and G.has_node(t):
                G.add_edge(h, t, relation=r)

    return G, entity_metadata


# Lightweight keyword/gazetteer heuristics used only for user-pasted custom text,
# since custom documents have no hand-annotated entities/triples like DATA_CORPORA.
CUSTOM_ORG_KEYWORDS = [
    "university", "institute", "college", "corporation", "corp", "inc",
    "company", "labs", "laboratory", "laboratories", "foundation", "ltd",
    "llc", "association", "school", "academy", "organization", "agency"
]

CUSTOM_LOCATION_GAZETTEER = {
    "london", "toronto", "new york", "san francisco", "seattle", "montreal",
    "mountain view", "redmond", "paris", "berlin", "tokyo", "beijing",
    "mumbai", "delhi", "bangalore", "kharagpur", "boston", "chicago",
    "los angeles", "washington", "singapore", "dubai", "sydney"
}


def extract_custom_entities_and_triples(text: str):
    """
    Heuristic entity/relation extraction for user-supplied text: flags
    capitalized noun phrases as candidate entities, classifies them via
    keyword/gazetteer matching, and links entities that co-occur within
    the same sentence. Not a trained NER model, so results are approximate.
    """
    entities = {}
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentence_entities = []
    phrase_pattern = re.compile(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b')

    for sent in sentences:
        found_in_sentence = []
        for match in phrase_pattern.finditer(sent):
            phrase = match.group().strip()
            if len(phrase) < 3:
                continue
            lower = phrase.lower()
            if lower in CUSTOM_LOCATION_GAZETTEER:
                etype = "LOCATION"
            elif any(kw in lower for kw in CUSTOM_ORG_KEYWORDS):
                etype = "ORGANIZATION"
            elif len(phrase.split()) == 2:
                etype = "PERSON"
            else:
                etype = "CONCEPT"

            entities.setdefault(phrase, etype)
            found_in_sentence.append(phrase)

        sentence_entities.append(list(dict.fromkeys(found_in_sentence)))

    triples = []
    seen_pairs = set()
    for ents_in_sent in sentence_entities:
        for i in range(len(ents_in_sent)):
            for j in range(i + 1, len(ents_in_sent)):
                pair = tuple(sorted((ents_in_sent[i], ents_in_sent[j])))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    triples.append((ents_in_sent[i], "related_to", ents_in_sent[j]))

    entity_list = [{"name": name, "type": etype} for name, etype in entities.items()]
    return entity_list, triples


def build_custom_corpus_from_text(raw_text: str) -> list:
    """
    Splits user-pasted text into documents (separated by a line containing
    only '---') and runs heuristic entity/relation extraction on each.
    """
    blocks = [b.strip() for b in re.split(r'\n\s*-{3,}\s*\n', raw_text) if b.strip()]
    docs = []
    for idx, block in enumerate(blocks, start=1):
        first_line = block.splitlines()[0].strip()
        title = first_line[:80] if 0 < len(first_line) < 100 else f"Custom Document {idx}"
        entities, triples = extract_custom_entities_and_triples(block)
        docs.append({
            "doc_id": f"CUSTOM-{idx}",
            "title": title,
            "text": block,
            "entities": entities,
            "triples": triples,
            "relevant_to": []
        })
    return docs


def evaluate_comparative_retrieval(corpus_docs: list, query_str: str,
                                   k1: float = 1.5, b: float = 0.75,
                                   alpha_entity: float = 1.2, top_k: int = 3):
    """
    Performs dual retrieval: Standard Okapi BM25 vs. Entity-Aware Probabilistic Ranking.
    Computes Precision@K, Recall@K, MAP, and NDCG@K for both methods.
    """
    q_tokens = tokenize(query_str)
    N = len(corpus_docs)
    doc_lens = [len(tokenize(d["text"])) for d in corpus_docs]
    avg_doc_len = sum(doc_lens) / max(1, N)

    # Compute IDF
    df_counts = Counter()
    for d in corpus_docs:
        unique_tokens = set(tokenize(d["text"]))
        for t in unique_tokens:
            df_counts[t] += 1

    idf_dict = {}
    for token, df in df_counts.items():
        # Standard probabilistic BM25 IDF formulation
        idf_dict[token] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    # Identify query entities
    all_known_entities = {}
    for d in corpus_docs:
        for ent in d["entities"]:
            all_known_entities[ent["name"].lower()] = ent

    matched_query_entities = []
    q_lower = query_str.lower()
    for ename_lower, ent_obj in all_known_entities.items():
        if ename_lower in q_lower:
            matched_query_entities.append(ent_obj["name"])

    # Ground truth relevance determination for this query
    ground_truth = set()
    for d in corpus_docs:
        # Check relevance tags or substring matching
        is_rel = False
        for tag in d.get("relevant_to", []):
            if any(q in tag or tag in query_str.lower() for q in q_tokens if len(q) > 2):
                is_rel = True
                break
        if not is_rel:
            # Fallback heuristic: query tokens overlap with doc title/entities
            doc_entity_names = [e["name"].lower() for e in d["entities"]]
            overlap = any(q in d["title"].lower() or any(q in en for en in doc_entity_names) for q in q_tokens if len(q) > 2)
            if overlap:
                is_rel = True
        if is_rel:
            ground_truth.add(d["doc_id"])

    # If ground truth is empty, no documents are genuinely relevant to this query
    # (Previously this fallback forced the first doc as relevant, causing misleading labels)

    # 1. Standard BM25 Scoring
    bm25_results = []
    for d in corpus_docs:
        d_tokens = tokenize(d["text"])
        score = compute_bm25_score(q_tokens, d_tokens, len(d_tokens), avg_doc_len, idf_dict, k1, b)
        bm25_results.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "score": score,
            "is_relevant": d["doc_id"] in ground_truth,
            "entities": [e["name"] for e in d["entities"]]
        })

    bm25_ranked = sorted(bm25_results, key=lambda x: x["score"], reverse=True)

    # 2. Entity-Aware Probabilistic Scoring
    # Score = BM25 + alpha * (Entity Match Salience)
    entity_results = []
    for d in corpus_docs:
        d_tokens = tokenize(d["text"])
        base_bm25 = compute_bm25_score(q_tokens, d_tokens, len(d_tokens), avg_doc_len, idf_dict, k1, b)
        doc_entity_names = [e["name"] for e in d["entities"]]

        entity_salience = 0.0
        matching_entities = []
        for q_ent in matched_query_entities:
            if q_ent in doc_entity_names:
                entity_salience += 1.5
                matching_entities.append(q_ent)

        # Relation context boost
        for h, r, t in d.get("triples", []):
            if any(qe in [h, t] for qe in matched_query_entities):
                entity_salience += 0.5

        final_score = round(base_bm25 + (alpha_entity * entity_salience), 4)

        entity_results.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "score": final_score,
            "base_bm25": base_bm25,
            "entity_boost": round(alpha_entity * entity_salience, 4),
            "matched_entities": matching_entities,
            "is_relevant": d["doc_id"] in ground_truth,
            "entities": doc_entity_names
        })

    entity_ranked = sorted(entity_results, key=lambda x: x["score"], reverse=True)

    # Calculate Metrics: Precision@K, Recall@K, MAP, NDCG@K
    def calculate_metrics(ranked_list, K, total_rel_count):
        top_k_items = ranked_list[:K]
        rel_in_k = sum(1 for item in top_k_items if item["is_relevant"])
        precision_k = round(rel_in_k / max(1, K), 4)
        recall_k = round(rel_in_k / max(1, total_rel_count), 4)

        # Average Precision (AP)
        running_rel = 0
        ap_sum = 0.0
        for rank_idx, item in enumerate(ranked_list):
            if item["is_relevant"]:
                running_rel += 1
                ap_sum += running_rel / (rank_idx + 1)
        ap = round(ap_sum / max(1, total_rel_count), 4)

        # NDCG@K
        dcg = 0.0
        for idx, item in enumerate(top_k_items):
            rel_grade = 1.0 if item["is_relevant"] else 0.0
            dcg += rel_grade / math.log2(idx + 2)

        # Ideal DCG@K
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(K, total_rel_count)))
        ndcg_k = round(dcg / max(0.0001, idcg), 4) if idcg > 0 else 0.0

        return {
            "precision_k": precision_k,
            "recall_k": recall_k,
            "map": ap,
            "ndcg_k": ndcg_k
        }

    rel_count = len(ground_truth)
    bm25_metrics = calculate_metrics(bm25_ranked, top_k, rel_count)
    entity_metrics = calculate_metrics(entity_ranked, top_k, rel_count)

    return {
        "bm25_ranked": bm25_ranked,
        "entity_ranked": entity_ranked,
        "bm25_metrics": bm25_metrics,
        "entity_metrics": entity_metrics,
        "ground_truth": list(ground_truth),
        "matched_query_entities": matched_query_entities
    }


def generate_plotly_knowledge_graph(G: nx.DiGraph):
    """
    Renders an uncluttered, publication-grade interactive network graph
    with fewer nodes (8-15) using Plotly and NetworkX spring layout.
    """
    pos = nx.spring_layout(G, seed=42, k=0.85)

    # Color palette for entity categories
    color_map = {
        "PERSON": "#8B5CF6",       # Purple
        "ORGANIZATION": "#2563EB", # Royal Blue
        "LOCATION": "#10B981",     # Emerald Green
        "CONCEPT": "#F59E0B"       # Amber Orange
    }

    # Extract edge coordinates
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} &rarr; <i>{edge[2].get('relation', 'rel')}</i> &rarr; {edge[1]}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#94A3B8"),
        hoverinfo='none',
        mode='lines'
    )

    # Extract node coordinates and properties
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_labels = []
    node_sizes = []

    degrees = dict(G.degree())

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("type", "CONCEPT")
        node_colors.append(color_map.get(ntype, "#6B7280"))
        node_labels.append(node)
        deg = degrees.get(node, 1)
        node_sizes.append(max(18, min(36, 18 + deg * 4)))
        node_text.append(
            f"<b>Entity:</b> {node}<br>"
            f"<b>Category:</b> {ntype}<br>"
            f"<b>Degree Centrality:</b> {deg}<br>"
            f"<b>Associated Docs:</b> {len(G.nodes[node].get('docs', []))}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, family="Arial, sans-serif"),
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='#1E293B')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Domain Knowledge Graph Topology (Fewer Nodes for Clarity)",
                font=dict(size=15)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=45),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ======================================================================================
# 4. LAB REPORT PDF EXPORTER (FPDF COMPATIBLE WITH template.py)
# ======================================================================================

class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | IIT Kharagpur Virtual Laboratory - KGIRS Exp 6", align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment benchmark records into an official PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Institute & Department Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR - VIRTUAL LABS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING | KGIRS", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, f"Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student & Session Info Box
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
    pdf.cell(38, 5, "Evaluation Scores:", 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(16, 185, 129)
    score_str = f"Quiz: {quiz_score}/{quiz_total}"
    pdf.cell(48, 5, score_str, 1)

    pdf.ln(16)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Experiment Objectives & Learning Units", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = str(obj).replace("$", "").replace("\\", "")
        pdf.cell(5, 5, "-", 0)
        pdf.cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Experimental Trials & Comparative Performance", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No simulation trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = list(trials_df.columns)
        num_cols = len(cols)
        col_w = max(18, int(190 / max(1, num_cols)))

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

    # 3. Observations & Analysis
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations, Inferences & Critical Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else (
        "The experimental trials demonstrated that entity-aware probabilistic ranking consistently outperformed "
        "the standard Okapi BM25 keyword baseline across Precision@K, MAP, and NDCG@K metrics by capturing "
        "disambiguated semantic relationships between persons, organizations, locations, and concepts."
    )
    pdf.multi_cell(0, 5, notes_text)
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Evaluator Signature", align="C")

    return bytes(pdf.output())


def generate_certificate_pdf(student_name: str, student_id: str, date_str: str,
                             quiz_score: int, quiz_total: int) -> bytes:
    """Compiles a landscape Certificate of Completion PDF for the experiment."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # Decorative double border
    pdf.set_draw_color(37, 99, 235)
    pdf.set_line_width(1.2)
    pdf.rect(8, 8, 281, 194)
    pdf.set_draw_color(148, 163, 184)
    pdf.set_line_width(0.4)
    pdf.rect(12, 12, 273, 186)

    pdf.set_y(26)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR - VIRTUAL LABORATORIES", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING | KGIRS", align="C", new_x="LMARGIN", new_y="NEXT")

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
        f"{EXPERIMENT_CONFIG['title']}, under the {EXPERIMENT_CONFIG['subject']} "
        "curriculum of the IIT Kharagpur Virtual Laboratory."
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
    pdf.cell(70, 5, "Instructor / Lab Coordinator Signature", align="C")

    return bytes(pdf.output())


# ======================================================================================
# 5. IIT KGP VLAB SECTION RENDERERS
# ======================================================================================

def render_page_header(section: str):
    """Renders the shared page header. Purpose gets the full Experiment title as its heading;
    every other section gets a small grey Experiment line and the section name as its heading."""
    if section == "Purpose":
        st.markdown("""
<style>
[data-testid="stMainBlockContainer"] { padding-top: 21px; }
</style>
""", unsafe_allow_html=True)
        st.markdown(f"""
<h1 style="font-size:1.95rem; font-weight:800; color:#000000; margin:0 0 0.35rem 0; line-height:1.3;">
Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}
</h1>
<p style="font-size:1.5rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase;
color:#000000; margin:0 0 1.3rem 0;">
{section}
</p>
""", unsafe_allow_html=True)
    else:
        heading_class = "kg-sim-page-heading" if section == "Simulation" else ""
        st.markdown("""
<style>
[data-testid="stMainBlockContainer"] { padding-top: 40px; }
</style>
""", unsafe_allow_html=True)
        if section == "Simulation":
            st.markdown("""
<style>
        .kg-sim-page-heading { margin-bottom:1.4rem !important; }
</style>
""", unsafe_allow_html=True)
        st.markdown(f"""
<p style="font-size:0.82rem; font-weight:600; color:#000000; margin:0 0 0.3rem 0;">
Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}
</p>
        <h1 class="{heading_class}" style="font-size:1.5rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase;
color:#000000; margin:0 0 1.3rem 0; line-height:1.3;">
{section}
</h1>
""", unsafe_allow_html=True)


# Groups of staggered elements that should reveal on scroll rather than all at once on load.
SCROLL_REVEAL_SELECTOR = (
    ".kg-storyboard, .kg-term-grid, .kg-compare-grid, .kg-illus-card, "
    ".kg-obj-list, .kg-metric-card"
)


def inject_scroll_reveal():
    """Attaches an IntersectionObserver to the main app document so the kg-* animated
    groups above play as each one scrolls into view, instead of all firing at once on load."""
    components.html(f"""
<script>
(function() {{
  const doc = window.parent.document;
  const targets = doc.querySelectorAll('{SCROLL_REVEAL_SELECTOR}');
  const io = new IntersectionObserver((entries) => {{
    entries.forEach((entry) => {{
      if (entry.isIntersecting) {{
        entry.target.classList.add('kg-inview');
        io.unobserve(entry.target);
      }}
    }});
  }}, {{ root: null, threshold: 0.12, rootMargin: '0px 0px -80px 0px' }});
  targets.forEach((el) => io.observe(el));
}})();
</script>
""", height=0, width=0)


def render_purpose_section():
    """Renders Section 1: Purpose, an animated, mechanism-first walkthrough of why this experiment matters."""

    st.markdown("""
<p style="font-size:1rem; color:#000000; line-height:1.75; width:100%; margin:0 0 1.2rem 0; text-align:justify;">
The purpose of this experiment is to evaluate how effectively graph-based entity identification can organize
large, unstructured datasets by capturing relationships between entities.
It also assesses whether probabilistic retrieval can improve the relevance and accuracy of information
retrieved from this structured representation.
</p>
""", unsafe_allow_html=True)
    st.image("image.png", width="stretch")
    st.markdown('<hr class="kg-divider"/>', unsafe_allow_html=True)

    st.markdown("""
<style>
@keyframes kgRise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes kgPop { 0% { opacity:0; transform:scale(0.35); } 70% { opacity:1; transform:scale(1.08); } 100% { opacity:1; transform:scale(1); } }
@keyframes kgDraw { to { stroke-dashoffset:0; } }
@keyframes kgPulse { 0%,100% { opacity:0.55; } 50% { opacity:1; } }
@keyframes kgGrow { to { width:var(--w); } }
@keyframes kgFadeIn { from { opacity:0; } to { opacity:1; } }

.kg-eyebrow { display:inline-block; font-size:1.15rem; font-weight:700; letter-spacing:0.05em;
  color:#3B82F6; margin-bottom:0.35rem; }
.kg-eyebrow.kg-eyebrow-teal { color:#0E7C7B; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.15em; }
.kg-eyebrow.kg-eyebrow-ink { color:#000000; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.15em; }

.kg-section-title { font-size:1.3rem !important; font-weight:700; color:#1E3A8A; margin:0 0 0.6rem 0; }
.kg-section-body { font-size:0.95rem; line-height:1.7; color:#000000; width:100%; }
.kg-section-body b { color:#000000; }
.kg-example { margin-top:0.9rem; padding:0.8rem 1rem; background:#FEFDFB; border:1px solid #E7E2D3;
  border-left:3px solid #B3261E; border-radius:6px; font-size:0.87rem; color:#000000; line-height:1.6; width:100%; }
.kg-example.kg-example-teal { border-left-color:#0E7C7B; }
.kg-example code { background:rgba(31,41,55,0.06); padding:1px 5px; border-radius:4px; font-size:0.85em; }

.kg-divider { height:1px; background:#E7E2D3; margin:2.1rem 0; border:none; }

/* --- storyboard shell -------------------------------------------------- */
.kg-storyboard { display:flex; gap:0.7rem; margin-top:1.3rem; align-items:stretch; flex-wrap:wrap; }
.kg-story-step { flex:1 1 230px; min-width:220px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px;
  padding:0.95rem 1rem 1.1rem; display:flex; flex-direction:column; box-shadow:0 1px 2px rgba(31,41,55,0.04); }
.kg-story-step.kg-story-step-win { border-color:#BFE3E1; background:#F6FBFA; }
.kg-story-step.kg-story-step-lose { border-color:#F1D6D2; background:#FEFAF9; }
.kg-story-eyebrow { font-size:0.8rem; font-weight:800; letter-spacing:0.1em; color:#000000; margin-bottom:0.25rem; }
.kg-story-title { font-size:0.85rem; font-weight:700; color:#000000; margin-bottom:0.75rem; min-height:2.3em; }
.kg-story-stage { flex:1; display:flex; align-items:center; justify-content:center; min-height:118px; }
.kg-story-connector { display:flex; align-items:center; justify-content:center; flex:0 0 22px; }
.kg-story-connector svg { opacity:0; animation:kgPop 0.4s ease forwards; animation-delay:0.9s; }

/* --- step 1: plain keyword chips (no meaning attached) ------------------ */
.kg-chip-row { display:flex; flex-wrap:wrap; gap:0.32rem; justify-content:center; align-content:center; }
.kg-chip { padding:0.28rem 0.6rem; background:#F3F4F6; border:1px solid #D1D5DB; border-radius:999px;
  font-size:0.72rem; font-weight:600; color:#000000; opacity:0; animation:kgPop 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards; }

/* --- step 1 (method two): entities recognized inline, NER-style -------- */
.kg-ner-sentence { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:center;
  column-gap:0.4rem; row-gap:0.7rem; font-size:0.86rem; color:#000000; text-align:center; }
.kg-ner-word { padding-bottom:0.3rem; }
.kg-ent { display:flex; flex-direction:column; align-items:center; gap:0.22rem; opacity:0; animation:kgRise 0.4s ease forwards; }
.kg-ent-value { padding:0.15rem 0.4rem; border-radius:5px; font-weight:700; }
.kg-ent-tag { font-size:0.72rem; font-weight:800; letter-spacing:0.05em; padding:1px 5px; border-radius:4px;
  color:#fff; white-space:nowrap; opacity:0; animation:kgPop 0.35s ease forwards; }
.kg-ent-per .kg-ent-value { background:rgba(139,92,246,0.16); color:#6D28D9; } .kg-ent-per .kg-ent-tag { background:#8B5CF6; }
.kg-ent-org .kg-ent-value { background:rgba(37,99,235,0.16); color:#1D4ED8; } .kg-ent-org .kg-ent-tag { background:#2563EB; }
.kg-ent-loc .kg-ent-value { background:rgba(16,185,129,0.16); color:#047857; } .kg-ent-loc .kg-ent-tag { background:#10B981; }

/* --- step 2 (method one): raw tally cards ------------------------------- */
.kg-doc-mini { width:100%; background:#FFFFFF; border:1px solid #E5E7EB; border-radius:9px;
  padding:0.5rem 0.65rem; margin-bottom:0.45rem; opacity:0; animation:kgRise 0.4s ease forwards; }
.kg-doc-mini-title { font-size:0.85rem; font-weight:700; color:#000000; margin-bottom:0.3rem; }
.kg-doc-mini.kg-doc-spam { border-color:#F1D6D2; background:#FFFAF9; }
.kg-tally { display:inline-block; font-size:0.8rem; font-weight:600; color:#000000; background:#F3F4F6;
  border-radius:5px; padding:1px 6px; margin:1px 3px 1px 0; }
.kg-tally-hot { color:#B3261E; background:rgba(179,38,30,0.1); animation:kgPulse 1.6s ease-in-out infinite; }

/* --- step 2 (method two): mini knowledge-graph triple ------------------- */
.kg-graph-node { opacity:0; animation:kgPop 0.45s cubic-bezier(0.34,1.56,0.64,1) forwards; }
.kg-graph-label { font-size:17px; font-weight:700; fill:#000000; opacity:0; animation:kgFadeIn 0.3s ease forwards; }
.kg-graph-edge-label { font-size:16px; font-weight:600; fill:#000000; opacity:0; animation:kgFadeIn 0.3s ease forwards; }
.kg-graph-edge { stroke-dasharray:140; stroke-dashoffset:140; animation:kgDraw 0.55s ease forwards; }

/* --- step 3 (both methods): ranking bar race ---------------------------- */
.kg-barlist { width:100%; display:flex; flex-direction:column; gap:0.6rem; }
.kg-bar-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.2rem; }
.kg-bar-doc { font-size:0.85rem; font-weight:600; color:#000000; }
.kg-bar-rank { font-size:0.85rem; font-weight:800; }
.kg-bar-track { background:#EFEEE8; border-radius:6px; height:12px; overflow:hidden; }
.kg-bar { height:100%; border-radius:6px; width:0; animation:kgGrow 0.9s cubic-bezier(.22,1,.36,1) forwards; }
.kg-bar-good { background:#0E7C7B; } .kg-bar-spam { background:#B3261E; }
.kg-bar-tag { display:inline-block; margin-top:0.28rem; font-size:0.78rem; font-weight:700; padding:1px 6px;
  border-radius:4px; opacity:0; animation:kgPop 0.4s ease forwards; }
.kg-bar-tag-bad { color:#B3261E; background:rgba(179,38,30,0.09); }
.kg-bar-tag-good { color:#0E7C7B; background:rgba(14,124,123,0.09); }

/* --- section 3: outcome recap ------------------------------------------- */
.kg-obj-list { display:flex; flex-direction:column; gap:0.5rem; margin-top:0.7rem; }
.kg-obj-item { display:flex; align-items:flex-start; gap:0.65rem; padding:0.65rem 0.9rem;
  background:#FEFDFB; border:1px solid #E7E2D3; border-radius:8px;
  font-size:0.89rem; color:#000000; line-height:1.5; opacity:0; animation:kgRise 0.45s ease forwards; }
.kg-obj-mark { flex-shrink:0; margin-top:2px; }

/* --- scroll-triggered reveal: paused until scrolled into view ----------- */
:is(.kg-storyboard, .kg-obj-list),
:is(.kg-storyboard, .kg-obj-list) * { animation-play-state: paused; }
:is(.kg-storyboard, .kg-obj-list).kg-inview,
:is(.kg-storyboard, .kg-obj-list).kg-inview * { animation-play-state: running; }
</style>
""", unsafe_allow_html=True)

    arrow_svg = (
        '<div class="kg-story-connector"><svg width="20" height="20" viewBox="0 0 20 20">'
        '<path d="M5 3 L14 10 L5 17" fill="none" stroke="#9CA3AF" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    )

    # --- Section 1: Plain Keyword Search --------------------------------------------
    st.markdown("""
<div class="kg-purpose-scope">
<p class="kg-section-title">Plain Keyword Search</p>
<p class="kg-section-body">
A traditional search system reads a query as a bag of words and ranks documents by how often those
words appear, nothing more. It cannot tell that a document repeating a term nine times is less useful
than one that mentions it once, in exactly the right context. Term frequency becomes a proxy for
relevance, and that proxy breaks easily, as the walkthrough below shows.
</p>
</div>
""", unsafe_allow_html=True)

    keywords = ["Elena", "Voss", "Deep", "Learning", "Nimbus", "Toronto"]
    chips_html = "".join(
        f'<span class="kg-chip" style="animation-delay:{0.1 + i * 0.08:.2f}s">{w}</span>'
        for i, w in enumerate(keywords)
    )

    st.markdown(f"""
<div class="kg-purpose-scope">
<div class="kg-storyboard">

  <div class="kg-story-step">
    <p class="kg-story-eyebrow">STEP 1</p>
    <p class="kg-story-title">The query becomes a bag of words</p>
    <div class="kg-story-stage"><div class="kg-chip-row">{chips_html}</div></div>
  </div>

  {arrow_svg}

  <div class="kg-story-step">
    <p class="kg-story-eyebrow">STEP 2</p>
    <p class="kg-story-title">Every document is scored by raw word count</p>
    <div class="kg-story-stage" style="flex-direction:column; align-items:stretch;">
      <div class="kg-doc-mini" style="animation-delay:0.15s;">
        <div class="kg-doc-mini-title">Article: "Voss joins Nimbus Labs"</div>
        <span class="kg-tally">Voss &times; 1</span><span class="kg-tally">Nimbus &times; 1</span><span class="kg-tally">Toronto &times; 1</span>
      </div>
      <div class="kg-doc-mini kg-doc-spam" style="animation-delay:0.35s;">
        <div class="kg-doc-mini-title">Blog: keyword-stuffed page</div>
        <span class="kg-tally kg-tally-hot">Nimbus &times; 9</span>
      </div>
    </div>
  </div>

  {arrow_svg}

  <div class="kg-story-step kg-story-step-lose">
    <p class="kg-story-eyebrow">STEP 3</p>
    <p class="kg-story-title">Ranked purely by frequency</p>
    <div class="kg-story-stage">
      <div class="kg-barlist">
        <div>
          <div class="kg-bar-head"><span class="kg-bar-doc">Blog: keyword-stuffed page</span><span class="kg-bar-rank" style="color:#B3261E;">#1</span></div>
          <div class="kg-bar-track"><div class="kg-bar kg-bar-spam" style="--w:92%; animation-delay:0.5s;"></div></div>
          <span class="kg-bar-tag kg-bar-tag-bad" style="animation-delay:1.3s;">highest count, wrong document</span>
        </div>
        <div>
          <div class="kg-bar-head"><span class="kg-bar-doc">Article: "Voss joins Nimbus Labs"</span><span class="kg-bar-rank" style="color:#000000;">#2</span></div>
          <div class="kg-bar-track"><div class="kg-bar kg-bar-good" style="--w:34%; animation-delay:0.65s;"></div></div>
        </div>
      </div>
    </div>
  </div>

</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="kg-purpose-scope">
<div class="kg-example">
<b>The genuinely relevant article loses to a page that simply repeats <code>"Nimbus"</code>, because raw
word count has no concept of <i>who</i> is being talked about, or whether the mention is real.</b>
</div>
</div>
""", unsafe_allow_html=True)

    # --- Section 2: Entity-Aware, Probabilistic Retrieval ---------------------------
    st.markdown("""
<div class="kg-purpose-scope" style="margin-top:26px;">
<p class="kg-section-title">Entity-Aware, Probabilistic Retrieval</p>
<p class="kg-section-body">
This experiment adds two ideas on top of keyword matching. <b>Entity extraction</b> identifies which
words are actually people, organizations, or places, and how they relate to one another. <b>Probabilistic
ranking</b> (Okapi BM25, extended with an entity-salience boost) then reorders documents by their
estimated probability of relevance, not their word count. Same query, same two documents: watch the
ranking flip.
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="kg-purpose-scope">
<div class="kg-storyboard">

  <div class="kg-story-step">
    <p class="kg-story-eyebrow">STEP 1</p>
    <p class="kg-story-title">Entities are recognized, not just words</p>
    <div class="kg-story-stage">
<div class="kg-ner-sentence">
<span class="kg-ent kg-ent-per" style="animation-delay:0.1s;"><span class="kg-ent-tag" style="animation-delay:0.3s;">PERSON</span><span class="kg-ent-value">Elena Voss</span></span>
<span class="kg-ner-word">joined</span>
<span class="kg-ent kg-ent-org" style="animation-delay:0.25s;"><span class="kg-ent-tag" style="animation-delay:0.45s;">ORGANIZATION</span><span class="kg-ent-value">Nimbus Labs</span></span>
<span class="kg-ner-word">in</span>
<span class="kg-ent kg-ent-loc" style="animation-delay:0.4s;"><span class="kg-ent-tag" style="animation-delay:0.6s;">LOCATION</span><span class="kg-ent-value">Toronto</span></span>
</div>
    </div>
  </div>

  {arrow}

  <div class="kg-story-step">
    <p class="kg-story-eyebrow">STEP 2</p>
    <p class="kg-story-title">Entities link into a small knowledge graph</p>
    <div class="kg-story-stage">
<svg viewBox="0 0 300 200" width="100%" height="200">
<line x1="55" y1="48" x2="235" y2="85" stroke="#9CA3AF" stroke-width="1.85" class="kg-graph-edge" style="animation-delay:0.75s;"/>
<line x1="235" y1="85" x2="150" y2="155" stroke="#9CA3AF" stroke-width="1.85" class="kg-graph-edge" style="animation-delay:0.95s;"/>
<text x="145" y="50" text-anchor="middle" class="kg-graph-edge-label" style="animation-delay:1.1s;">affiliated_with</text>
<text x="175" y="86" text-anchor="middle" class="kg-graph-edge-label" style="animation-delay:1.3s;">located_in</text>
<g class="kg-graph-node" style="animation-delay:0.1s;">
<circle cx="55" cy="48" r="11.5" fill="#8B5CF6"/>
<text x="55" y="24" text-anchor="middle" class="kg-graph-label">Elena Voss</text>
</g>
<g class="kg-graph-node" style="animation-delay:0.3s;">
<circle cx="235" cy="85" r="11.5" fill="#2563EB"/>
<text x="235" y="117" text-anchor="middle" class="kg-graph-label">Nimbus Labs</text>
</g>
<g class="kg-graph-node" style="animation-delay:0.5s;">
<circle cx="150" cy="155" r="11.5" fill="#10B981"/>
<text x="150" y="187" text-anchor="middle" class="kg-graph-label">Toronto</text>
</g>
</svg>
    </div>
  </div>

  {arrow}

  <div class="kg-story-step kg-story-step-win">
    <p class="kg-story-eyebrow">STEP 3</p>
    <p class="kg-story-title">Ranked by probability of relevance</p>
    <div class="kg-story-stage">
      <div class="kg-barlist">
        <div>
          <div class="kg-bar-head"><span class="kg-bar-doc">Article: "Voss joins Nimbus Labs"</span><span class="kg-bar-rank" style="color:#0E7C7B;">#1</span></div>
          <div class="kg-bar-track"><div class="kg-bar kg-bar-good" style="--w:88%; animation-delay:0.5s;"></div></div>
          <span class="kg-bar-tag kg-bar-tag-good" style="animation-delay:1.3s;">verified relationship, correct document</span>
        </div>
        <div>
          <div class="kg-bar-head"><span class="kg-bar-doc">Blog: keyword-stuffed page</span><span class="kg-bar-rank" style="color:#000000;">#2</span></div>
          <div class="kg-bar-track"><div class="kg-bar kg-bar-spam" style="--w:24%; animation-delay:0.65s;"></div></div>
        </div>
      </div>
    </div>
  </div>

</div>
</div>
""".format(arrow=arrow_svg), unsafe_allow_html=True)

    st.markdown("""
<div class="kg-purpose-scope">
<div class="kg-example kg-example-teal">
<b>The system now recognizes that one document is genuinely about <code>Elena Voss</code> joining
<code>Nimbus Labs</code> in <code>Toronto</code>: a verified relationship, not a coincidence of
repeated words, and ranks it first because of it.</b>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="kg-divider"/>', unsafe_allow_html=True)

    # --- Section 3: What We Will Learn ----------------------------------------------
    st.markdown("""
<div class="kg-purpose-scope">
<p class="kg-section-title">What We Will Learn</p>
<p class="kg-section-body">
By the end of this experiment, you will have built and compared both retrieval methods on the same
corpus, and measured the difference with standard benchmarks rather than intuition alone.
</p>
</div>
""", unsafe_allow_html=True)
    objectives_html = "".join(
        f'<div class="kg-obj-item" style="animation-delay:{0.15 + idx * 0.13:.2f}s">'
        f'<svg class="kg-obj-mark" width="16" height="16" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="10" cy="10" r="8.5" fill="none" stroke="#0E7C7B" stroke-width="1.6"/>'
        f'<path d="M6 10.2 L9 13.2 L14 7" fill="none" stroke="#0E7C7B" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'<span>{obj}</span></div>'
        for idx, obj in enumerate(EXPERIMENT_CONFIG["objectives"])
    )
    st.markdown(f'<div class="kg-purpose-scope"><div class="kg-obj-list">{objectives_html}</div></div>', unsafe_allow_html=True)

    inject_scroll_reveal()


def render_triple_diagram():
    """Illustrates one sample (head, relation, tail) triple color-coded by entity type."""
    color_map = {"PERSON": "#8B5CF6", "ORGANIZATION": "#2563EB", "LOCATION": "#10B981"}
    nodes = [
        {"name": "Elena Voss", "type": "PERSON", "x": 0.0},
        {"name": "Lakeside University", "type": "ORGANIZATION", "x": 1.0},
        {"name": "Toronto", "type": "LOCATION", "x": 2.0},
    ]
    edges = [(0, 1, "affiliated_with"), (1, 2, "located_in")]

    fig = go.Figure()
    for i, j, rel in edges:
        fig.add_trace(go.Scatter(
            x=[nodes[i]["x"], nodes[j]["x"]], y=[0, 0],
            mode="lines", line=dict(width=2.5, color="#CBD5E1"), hoverinfo="none", showlegend=False
        ))
        fig.add_annotation(
            x=(nodes[i]["x"] + nodes[j]["x"]) / 2, y=0.15,
            text=f"<i>{rel}</i>", showarrow=False, font=dict(size=13, color="#0E7C7B", family="Georgia, serif")
        )

    fig.add_trace(go.Scatter(
        x=[n["x"] for n in nodes], y=[0] * len(nodes),
        mode="markers+text",
        marker=dict(size=48, color=[color_map[n["type"]] for n in nodes], line=dict(width=2.5, color="#1F2937")),
        text=[n["name"] for n in nodes], textposition="bottom center",
        textfont=dict(size=14, color="#1F2937", family="Georgia, serif"),
        hovertext=[f"{n['name']} ({n['type']})" for n in nodes], hoverinfo="text", showlegend=False
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=45),
        xaxis=dict(visible=False, range=[-0.5, 2.5]), yaxis=dict(visible=False, range=[-0.3, 0.35]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_prp_ranking_diagram():
    """Illustrates PRP: documents ordered by decreasing probability of relevance."""
    docs = [f"D{i}" for i in range(1, 7)]
    probs = [0.92, 0.81, 0.63, 0.44, 0.27, 0.11]
    fig = go.Figure(go.Bar(
        x=docs, y=probs, marker_color="#0E7C7B",
        text=[f"{p:.2f}" for p in probs], textposition="outside",
        textfont=dict(size=13, color="#1F2937")
    ))
    fig.update_layout(
        height=290, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="Estimated probability of relevance", range=[0, 1.08],
                   gridcolor="#EFEEE8", tickfont=dict(size=12, color="#4B5563"),
                   title_font=dict(size=13, color="#4B5563")),
        xaxis=dict(title="Documents, in ranked order", tickfont=dict(size=13, color="#1F2937"),
                   title_font=dict(size=13, color="#4B5563")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_bm25_saturation_diagram():
    """Illustrates BM25 term-frequency saturation for a few k1 values."""
    tf_vals = list(range(0, 11))
    fig = go.Figure()
    for k1, color in [(1.0, "#94A3B8"), (1.5, "#0E7C7B"), (2.5, "#B3261E")]:
        y_vals = [tf * (k1 + 1.0) / (tf + k1) for tf in tf_vals]
        fig.add_trace(go.Scatter(
            x=tf_vals, y=y_vals, mode="lines+markers", name=f"k1 = {k1}",
            line=dict(color=color, width=2.5), marker=dict(size=6)
        ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Raw term frequency, f(q, D)", gridcolor="#EFEEE8",
                   tickfont=dict(size=12, color="#4B5563"), title_font=dict(size=13, color="#4B5563")),
        yaxis=dict(title="TF component of the BM25 score", gridcolor="#EFEEE8",
                   tickfont=dict(size=12, color="#4B5563"), title_font=dict(size=13, color="#4B5563")),
        legend=dict(orientation="h", y=-0.22, font=dict(size=12, color="#1F2937")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_ndcg_discount_diagram():
    """Illustrates the logarithmic position discount used in NDCG."""
    ranks = list(range(1, 11))
    discounts = [1.0 / math.log2(r + 1) for r in ranks]
    fig = go.Figure(go.Bar(
        x=ranks, y=discounts, marker_color="#10B981",
        text=[f"{d:.2f}" for d in discounts], textposition="outside",
        textfont=dict(size=11, color="#1F2937")
    ))
    fig.update_layout(
        height=290, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Rank position", dtick=1, tickfont=dict(size=12, color="#4B5563"),
                   title_font=dict(size=13, color="#4B5563")),
        yaxis=dict(title="Discount weight, 1 / log2(rank + 1)", gridcolor="#EFEEE8",
                   tickfont=dict(size=12, color="#4B5563"), title_font=dict(size=13, color="#4B5563")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_theory_section():
    """Renders Section 2: In-Depth Theoretical Foundations, from first principles to evaluation."""
    st.markdown("""
<style>
@keyframes kgThRise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes kgThPop { 0% { opacity:0; transform:scale(0.4); } 70% { opacity:1; transform:scale(1.06); } 100% { opacity:1; transform:scale(1); } }
@keyframes kgThDraw { to { stroke-dashoffset:0; } }
@keyframes kgThFade { from { opacity:0; } to { opacity:1; } }

.kg-th-intro { font-size:1rem; line-height:1.8; color:#000000; width:100%; margin-bottom:0.3rem; margin-top:-20px !important; }

.kg-th-head { display:flex; align-items:center; gap:0.65rem; margin:0.3rem 0 0.9rem 0; }
.kg-th-num { flex-shrink:0; width:1.7rem; height:1.7rem; border-radius:50%; background:#1F2937; color:#fff;
  font-weight:700; font-size:0.78rem; display:flex; align-items:center; justify-content:center; }
.kg-th-title { font-size:1.12rem; font-weight:700; color:#000000; line-height:1.35; letter-spacing:0.01em; }

.kg-th-body { font-size:1rem; line-height:1.8; color:#000000; width:100%; }
.kg-th-body b { color:#000000; }
.kg-th-body code { background:rgba(31,41,55,0.07); padding:2px 7px; border-radius:5px; font-size:0.9em; color:#000000; }

.kg-th-quote { margin:1.1rem 0; padding:1.1rem 1.4rem; background:#F6FBFA; border-left:4px solid #0E7C7B;
  border-radius:8px; font-style:italic; color:#000000; font-size:1.02rem; line-height:1.75; width:100%; }

.kg-formula-label { font-size:0.7rem; font-weight:800; letter-spacing:0.13em; text-transform:uppercase;
  color:#B3261E; margin:0 0 0.6rem 0; }
.kg-formula-label.teal { color:#0E7C7B; }

.kg-term-grid { display:flex; flex-wrap:wrap; gap:0.75rem; margin:1.1rem 0 1.4rem 0; }
.kg-term-card { flex:1 1 210px; min-width:200px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:11px;
  border-left:4px solid #0E7C7B; padding:0.95rem 1.1rem; opacity:0; animation:kgThRise 0.4s ease forwards; }
.kg-term-symbol { font-size:1.05rem; font-weight:800; color:#0E7C7B; margin-bottom:0.35rem; font-family:"Courier New", monospace; }
.kg-term-label { font-size:0.88rem; font-weight:700; color:#000000; margin-bottom:0.3rem; }
.kg-term-desc { font-size:0.88rem; color:#000000; line-height:1.55; }

.kg-compare-grid { display:flex; flex-wrap:wrap; gap:1rem; margin:1.1rem 0 1.4rem 0; }
.kg-compare-card { flex:1 1 300px; min-width:270px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px;
  padding:1.2rem 1.3rem; opacity:0; animation:kgThRise 0.45s ease forwards; }
.kg-compare-icon { margin-bottom:0.7rem; }
.kg-compare-title { font-size:1.05rem; font-weight:800; color:#000000; margin-bottom:0.5rem; }
.kg-compare-desc { font-size:0.92rem; color:#000000; line-height:1.65; }

.kg-metric-card { background:#FEFDFB; border:1px solid #E7E2D3; border-left:4px solid #B3261E; border-radius:11px;
  padding:1rem 1.2rem; margin-bottom:0.8rem; opacity:0; animation:kgThRise 0.4s ease forwards; }
.kg-metric-name { font-size:1rem; font-weight:800; color:#000000; margin-bottom:0.5rem; }
.kg-metric-desc { font-size:0.9rem; color:#000000; line-height:1.6; margin-top:0.55rem; }

.kg-illus-card { background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px; padding:1.2rem 1.3rem 1rem;
  box-shadow:0 1px 2px rgba(31,41,55,0.05); margin:1rem 0 1.4rem 0; }
.kg-illus-caption { font-size:0.85rem; color:#000000; margin-top:0.5rem; line-height:1.5; }

.kg-gnode { opacity:0; animation:kgThPop 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards; }
.kg-gedge { stroke-dasharray:220; stroke-dashoffset:220; animation:kgThDraw 0.6s ease forwards; }
.kg-glabel { opacity:0; animation:kgThFade 0.4s ease forwards; }
.kg-gpill { opacity:0; animation:kgThPop 0.4s ease forwards; }

.katex-display { overflow-x:auto; overflow-y:hidden; padding-bottom:6px; }

/* --- scroll-triggered reveal: paused until scrolled into view ----------- */
:is(.kg-term-grid, .kg-compare-grid, .kg-illus-card, .kg-metric-card),
:is(.kg-term-grid, .kg-compare-grid, .kg-illus-card, .kg-metric-card) * { animation-play-state: paused; }
:is(.kg-term-grid, .kg-compare-grid, .kg-illus-card, .kg-metric-card).kg-inview,
:is(.kg-term-grid, .kg-compare-grid, .kg-illus-card, .kg-metric-card).kg-inview * { animation-play-state: running; }
</style>
<p class="kg-th-intro">Before comparing the two retrieval methods hands-on, this section builds up the
vocabulary and math they rely on, starting from what "data" and a "graph" even mean, through entity
extraction and Knowledge Graphs, to the probabilistic ranking formulas the Simulation section runs live.</p>
""", unsafe_allow_html=True)

    # --- 0. Foundations ---------------------------------------------------------------
    st.markdown(
        '<div class="kg-th-head" style="margin-top:20px;"><span class="kg-th-num">0</span>'
        '<span class="kg-th-title">Foundations: From Raw Text to Structured Knowledge</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body"><b>Data</b> is any recorded fact (a word, a number, a timestamp) before it has '
        'been organized into something a program can reason about. How that data is organized determines what a '
        'computer can and cannot do with it:</p>',
        unsafe_allow_html=True
    )

    compare_cards = [
        {
            "title": "Unstructured data",
            "desc": "Free-form text, images, or audio with no predefined schema: a news article, an email, "
                    "a PDF. A computer sees only characters or pixels, not meaning, until it is processed.",
            "icon": '<svg width="46" height="46" viewBox="0 0 46 46"><rect x="8" y="4" width="30" height="38" rx="4" '
                    'fill="#FFFFFF" stroke="#B3261E" stroke-width="2"/><line x1="14" y1="14" x2="32" y2="14" '
                    'stroke="#F1D6D2" stroke-width="3" stroke-linecap="round"/><line x1="14" y1="21" x2="32" y2="21" '
                    'stroke="#F1D6D2" stroke-width="3" stroke-linecap="round"/><line x1="14" y1="28" x2="26" y2="28" '
                    'stroke="#F1D6D2" stroke-width="3" stroke-linecap="round"/><line x1="14" y1="35" x2="30" y2="35" '
                    'stroke="#F1D6D2" stroke-width="3" stroke-linecap="round"/></svg>',
        },
        {
            "title": "Structured data",
            "desc": "Information organized into a fixed schema (rows and columns, or nodes and typed "
                    "relationships) so a program can query it directly, e.g. a spreadsheet or a Knowledge Graph.",
            "icon": '<svg width="46" height="46" viewBox="0 0 46 46"><rect x="4" y="6" width="38" height="34" rx="4" '
                    'fill="#FFFFFF" stroke="#0E7C7B" stroke-width="2"/><line x1="4" y1="17" x2="42" y2="17" '
                    'stroke="#0E7C7B" stroke-width="1.6"/><line x1="4" y1="28" x2="42" y2="28" stroke="#0E7C7B" '
                    'stroke-width="1.6"/><line x1="17" y1="6" x2="17" y2="40" stroke="#0E7C7B" stroke-width="1.6"/>'
                    '<line x1="30" y1="6" x2="30" y2="40" stroke="#0E7C7B" stroke-width="1.6"/></svg>',
        },
    ]
    compare_html = "".join(
        f'<div class="kg-compare-card" style="animation-delay:{0.1 + i * 0.15:.2f}s">'
        f'<div class="kg-compare-icon">{c["icon"]}</div>'
        f'<div class="kg-compare-title">{c["title"]}</div>'
        f'<div class="kg-compare-desc">{c["desc"]}</div></div>'
        for i, c in enumerate(compare_cards)
    )
    st.markdown(f'<div class="kg-compare-grid">{compare_html}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="kg-th-body">A <b>Knowledge Graph</b> is one way to structure data: a graph is simply a set of '
        '<b>nodes</b> (things) connected by <b>edges</b> (relationships between them). The terms below are used '
        'throughout the rest of this experiment:</p>',
        unsafe_allow_html=True
    )

    graph_terms = [
        ("Node / Vertex", "A single entity in the graph: a person, organization, location, or concept. Drawn as a circle."),
        ("Edge", "A connection between two nodes, representing a relationship between the entities it joins."),
        ("Directed edge", "An edge with direction: A &rarr; B means &ldquo;A relates to B&rdquo;, drawn as an arrow."),
        ("Labeled edge", "An edge tagged with the relation it represents, e.g. affiliated_with or located_in."),
        ("Degree", "The number of edges connected to a node: how many relationships that entity has."),
        ("Path", "A sequence of edges connecting one node to another through the graph."),
        ("Triple", "The smallest unit of a Knowledge Graph: (head entity, relation, tail entity)."),
    ]
    terms_html = "".join(
        f'<div class="kg-term-card" style="animation-delay:{0.06 * i:.2f}s">'
        f'<div class="kg-term-label">{label}</div><div class="kg-term-desc">{desc}</div></div>'
        for i, (label, desc) in enumerate(graph_terms)
    )
    st.markdown(f'<div class="kg-term-grid">{terms_html}</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="kg-illus-card">
<svg viewBox="0 0 380 210" width="100%" height="230">
<defs>
<marker id="kgArrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
<path d="M0,0 L9,4.5 L0,9 Z" fill="#94A3B8"/>
</marker>
</defs>
<line x1="82" y1="55" x2="255" y2="45" stroke="#94A3B8" stroke-width="2" marker-end="url(#kgArrow)" class="kg-gedge" style="animation-delay:0.5s;"/>
<line x1="255" y1="55" x2="270" y2="150" stroke="#94A3B8" stroke-width="2" marker-end="url(#kgArrow)" class="kg-gedge" style="animation-delay:0.7s;"/>
<line x1="70" y1="70" x2="65" y2="150" stroke="#94A3B8" stroke-width="2" marker-end="url(#kgArrow)" class="kg-gedge" style="animation-delay:0.9s;"/>
<text x="165" y="38" text-anchor="middle" class="kg-glabel" style="animation-delay:1.1s; font-size:14px; fill:#0E7C7B; font-weight:600;">affiliated_with</text>
<text x="290" y="105" text-anchor="middle" class="kg-glabel" style="animation-delay:1.25s; font-size:14px; fill:#0E7C7B; font-weight:600;">located_in</text>
<text x="30" y="115" text-anchor="middle" class="kg-glabel" style="animation-delay:1.4s; font-size:14px; fill:#0E7C7B; font-weight:600;">pioneered</text>
<g class="kg-gnode" style="animation-delay:0.1s;">
<circle cx="60" cy="60" r="16" fill="#8B5CF6" stroke="#1F2937" stroke-width="2"/>
<text x="60" y="94" text-anchor="middle" style="font-size:15px; font-weight:700; fill:#000000;">Node A</text>
</g>
<g class="kg-gpill" style="animation-delay:1.6s;">
<rect x="82" y="60" width="76" height="20" rx="10" fill="#1F2937"/>
<text x="120" y="74" text-anchor="middle" style="font-size:13px; font-weight:700; fill:#FFFFFF;">DEGREE = 2</text>
</g>
<g class="kg-gnode" style="animation-delay:0.3s;">
<circle cx="260" cy="45" r="16" fill="#2563EB" stroke="#1F2937" stroke-width="2"/>
<text x="260" y="20" text-anchor="middle" style="font-size:15px; font-weight:700; fill:#000000;">Node B</text>
</g>
<g class="kg-gnode" style="animation-delay:0.5s;">
<circle cx="275" cy="160" r="16" fill="#10B981" stroke="#1F2937" stroke-width="2"/>
<text x="275" y="188" text-anchor="middle" style="font-size:15px; font-weight:700; fill:#000000;">Node C</text>
</g>
<g class="kg-gnode" style="animation-delay:0.7s;">
<circle cx="60" cy="160" r="16" fill="#F59E0B" stroke="#1F2937" stroke-width="2"/>
<text x="60" y="188" text-anchor="middle" style="font-size:15px; font-weight:700; fill:#000000;">Node D</text>
</g>
<g class="kg-gpill" style="animation-delay:1.8s;">
<rect x="95" y="118" width="150" height="22" rx="11" fill="#B3261E"/>
<text x="170" y="133" text-anchor="middle" style="font-size:13px; font-weight:700; fill:#FFFFFF;">TRIPLE: (A, affiliated_with, B)</text>
</g>
</svg>
<p class="kg-illus-caption">A directed edge points from one node to another; a node's degree counts how many edges touch it; a labeled edge plus its two endpoints together form one triple.</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # --- 1. Entity Extraction & Knowledge Graph Construction --------------------------
    st.markdown(
        '<div class="kg-th-head"><span class="kg-th-num">1</span>'
        '<span class="kg-th-title">Entity Extraction &amp; Knowledge Graph Construction</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">A Knowledge Graph formalizes the terms above into a single structure: a set of '
        'entities, the relation types that can connect them, and the factual triples actually observed in the '
        'text.</p>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.markdown('<p class="kg-formula-label teal">DEFINITION</p>', unsafe_allow_html=True)
        st.latex(r"\boldsymbol{\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})}")
        st.caption("A set of entities E, a set of relation types R, and a set of triples T ⊆ E × R × E.")

    st.markdown(
        '<p class="kg-th-body">Every entity mention in the source text is tagged with one of four semantic '
        'categories:</p>',
        unsafe_allow_html=True
    )
    entity_types = [
        ("PERSON", "#8B5CF6", "Key researchers, executives, pioneers.", "Elena Voss, Kavi Rajan"),
        ("ORGANIZATION", "#2563EB", "Companies, research labs, universities.", "Cortex Labs, Lakeside University"),
        ("LOCATION", "#10B981", "Headquarters, lab sites, cities.", "London, Toronto, New York"),
        ("CONCEPT", "#F59E0B", "Technical domains, algorithmic paradigms.", "Deep Learning, Reinforcement Learning"),
    ]
    etype_html = "".join(
        f'<div class="kg-term-card" style="border-left-color:{color}; animation-delay:{0.08 * i:.2f}s">'
        f'<div class="kg-term-symbol" style="color:{color};">{name}</div>'
        f'<div class="kg-term-desc">{desc}<br/><i>e.g. {ex}</i></div></div>'
        for i, (name, color, desc, ex) in enumerate(entity_types)
    )
    st.markdown(f'<div class="kg-term-grid">{etype_html}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="kg-formula-label teal">RELATIONAL TRIPLE</p>', unsafe_allow_html=True)
        st.latex(r"\text{Triple} = (\text{Head},\ \text{Relation},\ \text{Tail})")
        st.caption("Example: (Elena Voss, affiliated_with, Nimbus AI Labs)")
    render_triple_diagram()

    st.divider()

    # --- 2. Probability Ranking Principle ----------------------------------------------
    st.markdown(
        '<div class="kg-th-head"><span class="kg-th-num">2</span>'
        '<span class="kg-th-title">The Probability Ranking Principle (PRP)</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">Formulated by Stephen E. Robertson in 1977, the Probability Ranking Principle is '
        'the theoretical basis for every ranked search system, including this one:</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="kg-th-quote">&ldquo;If a reference retrieval system\'s response to each request is a '
        'ranking of the documents in order of decreasing probability of relevance to the user, the overall '
        'effectiveness of the system will be maximized.&rdquo;</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">Formally, let <code>R &isin; {0, 1}</code> denote binary relevance. Documents '
        '<code>D</code> are ranked by the posterior odds of relevance given query <code>Q</code>:</p>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.markdown('<p class="kg-formula-label">FORMULA</p>', unsafe_allow_html=True)
        st.latex(r"\boldsymbol{\text{Odds}(R{=}1 \mid D, Q) = \dfrac{P(R{=}1 \mid D, Q)}{P(R{=}0 \mid D, Q)}}")
    render_prp_ranking_diagram()

    st.divider()

    # --- 3. Okapi BM25 -------------------------------------------------------------------
    st.markdown(
        '<div class="kg-th-head"><span class="kg-th-num">3</span>'
        '<span class="kg-th-title">Robertson&ndash;Sp&auml;rck Jones Okapi BM25 Model</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">BM25 is the non-linear, term-saturating probabilistic model this experiment uses '
        'as its plain-keyword baseline:</p>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.markdown('<p class="kg-formula-label">FORMULA</p>', unsafe_allow_html=True)
        st.latex(r"""\begin{gathered}
\boldsymbol{\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot {}} \\[6pt]
\boldsymbol{\frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}}
\end{gathered}""")

    st.markdown('<p class="kg-th-body">Each term in that formula plays a distinct role:</p>', unsafe_allow_html=True)
    bm25_terms = [
        ("f(q, D)", "Term frequency", "How many times query term q occurs in document D."),
        ("|D|, avgdl", "Length ratio", "Document length vs. the average document length across the collection."),
        ("k1 &isin; [1.2, 2.0]", "Term saturation", "How quickly extra occurrences of a term stop adding score."),
        ("b &asymp; 0.75", "Length normalization", "How strongly long documents are penalized; b=0 disables it."),
    ]
    bm25_html = "".join(
        f'<div class="kg-term-card" style="animation-delay:{0.08 * i:.2f}s">'
        f'<div class="kg-term-symbol">{sym}</div><div class="kg-term-label">{label}</div>'
        f'<div class="kg-term-desc">{desc}</div></div>'
        for i, (sym, label, desc) in enumerate(bm25_terms)
    )
    st.markdown(f'<div class="kg-term-grid">{bm25_html}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="kg-formula-label teal">SUPPORTING FORMULA &middot; INVERSE DOCUMENT FREQUENCY</p>', unsafe_allow_html=True)
        st.latex(r"\text{IDF}(q) = \ln\left(1 + \frac{N - n(q) + 0.5}{n(q) + 0.5}\right)")
        st.caption("N = total documents in the collection; n(q) = documents containing term q.")
    render_bm25_saturation_diagram()

    st.divider()

    # --- 4. Entity-Aware Probabilistic Ranking -------------------------------------------
    st.markdown(
        '<div class="kg-th-head"><span class="kg-th-num">4</span>'
        '<span class="kg-th-title">Entity-Aware Probabilistic Ranking</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">To fold Knowledge Graph semantics into ranking, the entity-aware score adds two '
        'more terms on top of the lexical BM25 score: a boost for verified shared entities, and a boost for '
        'verified shared relationships:</p>',
        unsafe_allow_html=True
    )
    with st.container(border=True):
        st.markdown('<p class="kg-formula-label">FORMULA</p>', unsafe_allow_html=True)
        st.latex(r"""\begin{gathered}
\boldsymbol{\text{Score}(D, Q) = \text{BM25}(D, Q)} \\[6pt]
\boldsymbol{{}+ \alpha \cdot \text{EntityBoost} + \beta \cdot \text{RelationBoost}}
\end{gathered}""")
        st.caption("α and β are tunable boost factors that control how much verified entities and relations can shift the lexical BM25 score.")

    col_eb, col_rb = st.columns(2)
    with col_eb:
        with st.container(border=True):
            st.markdown('<p class="kg-formula-label teal">ENTITY BOOST</p>', unsafe_allow_html=True)
            st.latex(r"\sum_{e \,\in\, \mathcal{E}_Q \cap \mathcal{E}_D} \text{Salience}(e)")
            st.caption("Sums the salience of every entity that appears in both the query and the document.")
    with col_rb:
        with st.container(border=True):
            st.markdown('<p class="kg-formula-label teal">RELATION BOOST</p>', unsafe_allow_html=True)
            st.latex(r"\sum_{(h,r,t) \,\in\, \mathcal{T}_D} \text{RelWeight}(r)")
            st.caption("Sums the weight of every document triple whose head h and tail t are both verified query entities.")

    st.divider()

    # --- 5. Comparative Evaluation Metrics -----------------------------------------------
    st.markdown(
        '<div class="kg-th-head"><span class="kg-th-num">5</span>'
        '<span class="kg-th-title">Comparative Evaluation Metrics</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="kg-th-body">Four standard Information Retrieval metrics are used to compare the two ranking '
        'methods objectively, rather than by eye:</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="kg-metric-card" style="animation-delay:0.05s">'
                '<div class="kg-metric-name">Precision@K</div></div>', unsafe_allow_html=True)
    st.latex(r"\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Top K}|}{K}")
    st.caption("Of the top K results returned, what fraction are actually relevant?")

    st.markdown('<div class="kg-metric-card" style="animation-delay:0.15s">'
                '<div class="kg-metric-name">Recall@K</div></div>', unsafe_allow_html=True)
    st.latex(r"\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Top K}|}{|\text{Total Relevant}|}")
    st.caption("Of all relevant documents that exist, what fraction did the top K actually surface?")

    st.markdown('<div class="kg-metric-card" style="animation-delay:0.25s">'
                '<div class="kg-metric-name">Mean Average Precision (MAP)</div></div>', unsafe_allow_html=True)
    st.latex(r"\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{|R_q|} \sum_{k=1}^{N} P_q(k) \times \text{rel}_q(k)")
    st.caption("Averages precision at every relevant hit, across every test query, rewarding rankings that place relevant items early.")

    with st.container(border=True):
        st.markdown('<p class="kg-formula-label">FORMULA &middot; NORMALIZED DISCOUNTED CUMULATIVE GAIN</p>', unsafe_allow_html=True)
        col_dcg, col_ndcg = st.columns(2)
        with col_dcg:
            st.latex(r"\boldsymbol{\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}}")
        with col_ndcg:
            st.latex(r"\boldsymbol{\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}}")
        st.caption("IDCG@K is the DCG@K of the ideal, perfectly-sorted ranking; NDCG@K = 1.0 means the ranking is perfect.")
    render_ndcg_discount_diagram()

    inject_scroll_reveal()


def extract_entities_from_text(text: str):
    """Extract candidate graph entities from raw text using a gazetteer and regex heuristics."""
    if not text or not text.strip():
        return []

    entities = []
    seen_spans = set()

    sorted_known = sorted(KNOWN_GAZETTEER.keys(), key=lambda x: len(x), reverse=True)
    for term in sorted_known:
        pattern = r"\b" + re.escape(term) + r"\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = (match.start(), match.end())
            if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                seen_spans.add(span)
                context_start = max(0, match.start() - 25)
                context_end = min(len(text), match.end() + 25)
                entities.append({
                    "entity": match.group(0),
                    "type": KNOWN_GAZETTEER[term],
                    "start": match.start(),
                    "end": match.end(),
                    "context": "..." + text[context_start:context_end].strip() + "..."
                })

    org_suffixes = r'(?:University|College|Institute|Corp|Corporation|Inc|Ltd|Company|Organization|Lab|Laboratory|Labs|Group|Foundation)'
    loc_suffixes = r'(?:City|State|Country|Street|Avenue|Park|Valley|Republic|Kingdom|Islands|Ocean|River)'
    role_words = r'(?:CEO|CTO|CFO|President|Founder|Director|Professor|Scientist|Engineer|Minister|Inventor|Lead|Chief|Chairman)'

    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+' + org_suffixes + r')\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "ORGANIZATION",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start() - 20):min(len(text), m.end() + 20)].strip() + "..."
            })

    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+' + loc_suffixes + r')\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "LOCATION",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start() - 20):min(len(text), m.end() + 20)].strip() + "..."
            })

    for m in re.finditer(r'\b(' + role_words + r')\b', text, re.IGNORECASE):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": text[m.start():m.end()],
                "type": "ROLE_TITLE",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start() - 20):min(len(text), m.end() + 20)].strip() + "..."
            })

    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "PERSON",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start() - 20):min(len(text), m.end() + 20)].strip() + "..."
            })

    for m in re.finditer(r'(?<=[a-z0-9,\s])([A-Z][a-z]{2,})\b', text):
        span = (m.start(), m.end())
        if not any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
            seen_spans.add(span)
            entities.append({
                "entity": m.group(1),
                "type": "OTHER_ENTITY",
                "start": m.start(),
                "end": m.end(),
                "context": "..." + text[max(0, m.start() - 20):min(len(text), m.end() + 20)].strip() + "..."
            })

    entities.sort(key=lambda x: x["start"])
    return entities


def compute_ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """Character n-gram Dice similarity between two strings."""
    s1_clean = s1.lower().strip()
    s2_clean = s2.lower().strip()
    if not s1_clean or not s2_clean:
        return 0.0
    if s1_clean == s2_clean:
        return 1.0

    s1_ng = [s1_clean[i:i + n] for i in range(len(s1_clean) - n + 1)] if len(s1_clean) >= n else [s1_clean]
    s2_ng = [s2_clean[i:i + n] for i in range(len(s2_clean) - n + 1)] if len(s2_clean) >= n else [s2_clean]

    s1_set = set(s1_ng)
    s2_set = set(s2_ng)
    overlap = len(s1_set.intersection(s2_set))
    total = len(s1_set) + len(s2_set)
    return (2.0 * overlap) / total if total > 0 else 0.0


def run_exact_retrieval(query: str, candidate_entities):
    """Exact / keyword retrieval: binary match only."""
    q_clean = query.lower().strip()
    results = []
    for ent in candidate_entities:
        e_name = ent["entity"].lower().strip()
        e_tokens = [t for t in re.findall(r"\w+", e_name)]
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


def run_probabilistic_retrieval(query: str, candidate_entities, k1: float = 1.5, b: float = 0.75):
    """BM25-based ranked retrieval with query-token coverage and document normalization."""
    q_tokens = re.findall(r"\w+", query.lower())
    if not q_tokens or not candidate_entities:
        return []

    corpus = []
    for ent in candidate_entities:
        doc_text = f"{ent['entity']} {ent.get('context', '')}"
        corpus.append({
            "entity": ent["entity"],
            "type": ent["type"],
            "context": ent.get("context", ""),
            "tokens": re.findall(r"\w+", doc_text.lower())
        })

    doc_lens = [len(item["tokens"]) for item in corpus]
    avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0

    df_counts = Counter()
    for item in corpus:
        for token in set(item["tokens"]):
            df_counts[token] += 1

    idf_dict = {}
    N = len(corpus)
    for term, df in df_counts.items():
        idf_dict[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    scored = []
    for item in corpus:
        doc_tokens = item["tokens"]
        doc_len = len(doc_tokens)
        score = compute_bm25_score(q_tokens, doc_tokens, doc_len, avg_doc_len, idf_dict, k1, b)
        if score < 0:
            score = 0.0
        scored.append({
            "entity": item["entity"],
            "type": item["type"],
            "relevance_score": round(score, 4),
            "confidence": "High Confidence" if score >= 0.85 else "Moderate Relevance" if score >= 0.40 else "Marginal Association" if score >= 0.15 else "Non-Relevant",
            "context": item["context"],
            "bm25_score": round(score, 4)
        })

    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    for idx, row in enumerate(scored, 1):
        row["rank"] = idx
    return scored


def render_candidate_graph(entities):
    """Render a compact candidate graph showing detected entities as nodes."""
    display_entities = entities[:8]
    if not display_entities:
        st.info("No candidate nodes are available yet.")
        return

    type_colors = {
        "PERSON": "#8B5CF6",
        "ORGANIZATION": "#2563EB",
        "LOCATION": "#10B981",
        "ROLE_TITLE": "#7C3AED",
        "CONCEPT": "#F59E0B",
        "AWARD": "#EC4899",
        "OTHER_ENTITY": "#64748B"
    }
    positions = [index * 2.2 for index in range(len(display_entities))]
    fig = go.Figure()
    for index, entity in enumerate(display_entities):
        node_y = 0.18 if index % 2 == 0 else -0.08
        label_position = "top center" if index % 2 == 0 else "bottom center"
        fig.add_trace(go.Scatter(
            x=[positions[index]],
            y=[node_y],
            mode="markers+text",
            text=[f"{entity['entity']}<br><sup>{entity['type']}</sup>"],
            textposition=label_position,
            textfont={"size": 10, "color": "#1F2937"},
            marker={"size": 26, "color": type_colors.get(entity["type"], "#64748b"), "line": {"width": 2, "color": "white"}},
            hovertemplate=f"<b>{entity['entity']}</b><br>Type: {entity['type']}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        height=235,
        xaxis={"visible": False, "range": [-1.0, max(2.0, positions[-1] + 1.0)]},
        yaxis={"visible": False, "range": [-0.55, 0.8]},
        margin={"l": 10, "r": 10, "t": 42, "b": 12},
        title={"text": "Candidate Entity Space", "x": 0.01, "font": {"size": 16, "color": "#1F2937"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Georgia, serif", "color": "#1F2937"}
    )
    st.plotly_chart(fig, use_container_width=True, key="candidate_graph", config={"displayModeBar": False})
    if len(entities) > len(display_entities):
        st.caption(f"Showing the first {len(display_entities)} nodes to keep the graph readable ({len(entities)} total candidates).")


def _compute_retrieval_metrics(retrieved_rows, relevant_set, top_k):
    """Calculate Precision@K, Recall@K, F1@K, and MRR for a ranked retrieval result list."""
    if not retrieved_rows:
        return {"precision_k": 0.0, "recall_k": 0.0, "f1_k": 0.0, "mrr": 0.0}

    rel_in_top_k = 0
    for row in retrieved_rows[:top_k]:
        if row["entity"] in relevant_set:
            rel_in_top_k += 1

    precision_k = rel_in_top_k / max(1, top_k)
    recall_k = rel_in_top_k / max(1, len(relevant_set))
    f1_k = (2 * precision_k * recall_k / (precision_k + recall_k)) if (precision_k + recall_k) > 0 else 0.0

    mrr = 0.0
    for idx, row in enumerate(retrieved_rows, 1):
        if row["entity"] in relevant_set:
            mrr = 1.0 / idx
            break

    return {
        "precision_k": round(precision_k, 4),
        "recall_k": round(recall_k, 4),
        "f1_k": round(f1_k, 4),
        "mrr": round(mrr, 4)
    }


def render_simulation_section():
    """Beautiful, app-consistent simulation workflow with the original logic preserved."""
    st.markdown("""
    <style>
    .kg-sim-shell {
        background: #FEFDFB;
        border: 1px solid #E7E2D3;
        border-radius: 14px;
        padding: 1rem 1.15rem 1.05rem;
        box-shadow: 0 1px 2px rgba(31,41,55,0.04);
        margin-bottom: 0.85rem;
    }
    .kg-sim-kicker {
        display: inline-block;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 800;
        font-size: 0.72rem;
        color: #0E7C7B;
        background: #F6FBFA;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.7rem;
    }
    .kg-sim-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #000000;
        margin: 0;
        line-height: 1.2;
    }
    .kg-sim-sub {
        font-size: 0.96rem;
        color: #000000;
        margin-top: 0.45rem;
        line-height: 1.7;
        max-width: 980px;
    }
    .kg-sim-grid {
        display: flex;
        align-items: stretch;
        gap: 0.55rem;
        margin: 0.8rem 0 1.15rem;
    }
    .kg-sim-step-card {
        flex: 1 1 0;
        background: #FEFDFB;
        border: 1px solid #E7E2D3;
        border-top: 3px solid #2563eb;
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
        box-shadow: 0 1px 2px rgba(31,41,55,0.03);
    }
    .kg-sim-step-card.kg-sim-green { border-top-color: #10b981; }
    .kg-sim-step-card.kg-sim-teal { border-top-color: #0E7C7B; }
    .kg-sim-step-card.kg-sim-amber { border-top-color: #f59e0b; }
    .kg-sim-step-head {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }
    .kg-sim-step-badge {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #000000;
        background: transparent;
        border: 0;
        border-radius: 999px;
        padding: 0;
        white-space: nowrap;
    }
    .kg-sim-step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #000000;
        margin: 0;
    }
    .kg-sim-step-copy {
        font-size: 0.9rem;
        line-height: 1.7;
        color: #000000;
        margin: 0;
    }
    .kg-sim-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
        margin: 0.75rem 0 0.95rem;
    }
    .kg-sim-stat {
        background: #F6FBFA;
        border: 1px solid #BFE3E1;
        border-radius: 10px;
        padding: 0.72rem 0.85rem;
    }
    .kg-sim-stat-label {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #000000;
        font-weight: 800;
    }
    .kg-sim-stat-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #000000;
        margin-top: 0.35rem;
    }
    .kg-sim-panel {
        background: #FEFDFB;
        border: 1px solid #E7E2D3;
        border-radius: 11px;
        padding: 0.85rem 0.9rem 0.3rem;
        box-shadow: 0 1px 2px rgba(31,41,55,0.04);
        height: 100%;
    }
    .kg-sim-panel h3 {
        margin: 0 0 0.65rem 0;
        color: #000000;
        font-size: 1rem;
        font-weight: 700;
    }
    .kg-sim-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0E7C7B;
        background: #F6FBFA;
        border-radius: 999px;
        padding: 0.38rem 0.6rem;
        margin-bottom: 0.55rem;
    }
    .kg-sim-query-badges {
        display:flex;
        flex-wrap:wrap;
        gap:0.45rem;
        margin-top:0.65rem;
    }
    .kg-sim-query-badge {
        background: #F3F4F6;
        color: #000000;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .kg-sim-inline-btn .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
    }
    @media (max-width: 900px) {
        .kg-sim-grid { flex-direction: column; }
        .kg-sim-title { font-size: 1.45rem; }
        .kg-sim-summary { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('''
    <div class="kg-sim-shell">
      <div class="kg-sim-kicker">Simulation</div>
      <div class="kg-sim-title">Entity Identification &amp; Retrieval Lab</div>
      <div class="kg-sim-sub">This workflow isolates the core retrieval pipeline: identify entity candidates from the source text, set the search query, tune the BM25 ranking model, and compare the exact match baseline with the probabilistic retrieval score.</div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="kg-sim-grid">
      <div class="kg-sim-step-card kg-sim-teal">
        <div class="kg-sim-step-head">
                    <span class="kg-sim-step-badge">01</span>
          <h3 class="kg-sim-step-title">Input text and entity extraction</h3>
        </div>
                <p class="kg-sim-step-copy">Choose a passage and identify people, organizations, places, roles, and concepts. These become the candidate graph nodes.</p>
      </div>

      <div class="kg-sim-step-card kg-sim-green">
        <div class="kg-sim-step-head">
                    <span class="kg-sim-step-badge">02</span>
          <h3 class="kg-sim-step-title">Search query and retrieval controls</h3>
        </div>
                <p class="kg-sim-step-copy">Set the query, Top-K, and BM25 parameters. The same candidates are then scored by both methods.</p>
      </div>

      <div class="kg-sim-step-card kg-sim-amber">
        <div class="kg-sim-step-head">
                    <span class="kg-sim-step-badge">03</span>
          <h3 class="kg-sim-step-title">Evaluate and compare retrieval quality</h3>
        </div>
                <p class="kg-sim-step-copy">Compare matches, ranked scores, the chart, and Precision@K, Recall@K, F1@K, and MRR.</p>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input mode",
        ["Use preset text", "Paste custom text"],
        horizontal=True,
        help="Select a built-in benchmark sentence or enter your own entity-rich paragraph."
    )

    if input_mode == "Use preset text":
        preset_name = st.selectbox("Benchmark preset", list(SAMPLE_PRESETS.keys()), index=0)
        active_text = SAMPLE_PRESETS[preset_name]
    else:
        active_text = st.text_area(
            "Custom text",
            value=st.session_state.get("custom_text_val", "Sundar Pichai is the CEO of Google and studied at Stanford University in California."),
            height=120
        )
        st.session_state["custom_text_val"] = active_text

    if st.session_state.get("last_extracted_text") != active_text.strip():
        st.session_state["identified_entities"] = extract_entities_from_text(active_text)
        st.session_state["last_extracted_text"] = active_text.strip()
        st.session_state["retrieval_run"] = False

    st.markdown('<div class="kg-sim-tag">01 &middot; Text source</div>', unsafe_allow_html=True)
    text_col, action_col = st.columns([4, 1])
    with text_col:
        st.text_area("Current text", value=active_text, height=110, disabled=True)
    with action_col:
        if st.button("Identify Graph Entities", type="primary", use_container_width=True):
            detected = extract_entities_from_text(active_text)
            st.session_state["identified_entities"] = detected
            st.session_state["last_extracted_text"] = active_text.strip()
            st.session_state["retrieval_run"] = False
            st.toast(f"Detected {len(detected)} candidate entities")

    if "identified_entities" not in st.session_state:
        st.session_state["identified_entities"] = extract_entities_from_text(active_text)
        st.session_state["last_extracted_text"] = active_text.strip()

    identified_entities = st.session_state.get("identified_entities", [])
    if not identified_entities:
        st.warning("No entities were detected. Try a sentence with names, organizations, locations, or role titles.")
        return

    st.markdown(
        '<div class="kg-sim-summary">'
        f'<div class="kg-sim-stat"><div class="kg-sim-stat-label">Entity count</div><div class="kg-sim-stat-value">{len(identified_entities)}</div></div>'
        f'<div class="kg-sim-stat"><div class="kg-sim-stat-label">Entity types</div><div class="kg-sim-stat-value">{len(set(e["type"] for e in identified_entities))}</div></div>'
        '<div class="kg-sim-stat"><div class="kg-sim-stat-label">Candidate space</div><div class="kg-sim-stat-value">Graph</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    entity_col, graph_col = st.columns([2, 1.1])
    with entity_col:
        st.markdown('<div class="kg-sim-tag">Candidate entities</div>', unsafe_allow_html=True)
        entity_df = pd.DataFrame([
            {
                "Entity": e["entity"],
                "Type": e["type"],
                "Context": e["context"]
            } for e in identified_entities
        ])
        st.dataframe(entity_df, width="stretch", hide_index=True)
    with graph_col:
        st.markdown('<div class="kg-sim-tag">Graph view</div>', unsafe_allow_html=True)
        render_candidate_graph(identified_entities)

    st.markdown('<div class="kg-sim-tag">02 &middot; Retrieval setup</div>', unsafe_allow_html=True)
    candidate_suggestions = list(dict.fromkeys([e["entity"] for e in identified_entities]))[:6]
    query_col, controls_col = st.columns([2.2, 1.2])
    with query_col:
        query = st.text_input(
            "Search query",
            value=st.session_state.get("active_query", candidate_suggestions[0] if candidate_suggestions else "Google"),
            help="Try queries such as Google, Sundar Pichai, Stanford University, CEO, or California."
        )
        st.session_state["active_query"] = query
        if candidate_suggestions:
            st.markdown('<div class="kg-sim-tag">Suggested queries</div>', unsafe_allow_html=True)
            badges = "".join(f'<span class="kg-sim-query-badge">{s}</span>' for s in candidate_suggestions)
            st.markdown(f'<div class="kg-sim-query-badges">{badges}</div>', unsafe_allow_html=True)
    with controls_col:
        st.markdown('<div class="kg-sim-tag">Parameters</div>', unsafe_allow_html=True)
        top_k = st.number_input("Top-K (K)", min_value=1, max_value=10, value=3, step=1)
        bm25_k1 = st.number_input("BM25 k1", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        bm25_b = st.number_input("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

    st.markdown('<div class="kg-sim-tag">03 &middot; Run and evaluate</div>', unsafe_allow_html=True)
    run_col, status_col = st.columns([1, 3])
    with run_col:
        if st.button("Run Retrieval", type="primary", use_container_width=True):
            st.session_state["retrieval_run"] = True
    with status_col:
        if not st.session_state.get("retrieval_run", False):
            st.info("Click Run Retrieval to compare Exact Retrieval and BM25 ranking on the current query.")
            return

    exact_results = run_exact_retrieval(query, identified_entities)
    bm25_results = run_probabilistic_retrieval(query, identified_entities, k1=bm25_k1, b=bm25_b)
    exact_matches = [r for r in exact_results if r["exact_match"]]
    top_bm25 = bm25_results[0] if bm25_results else None

    relevant_names = set()
    query_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]
    for ent in identified_entities:
        name = ent["entity"].lower()
        context = ent.get("context", "").lower()
        if query.lower().strip() == ent["entity"].lower().strip():
            relevant_names.add(ent["entity"])
        elif any(tok in name for tok in query_tokens) or any(tok in context for tok in query_tokens):
            relevant_names.add(ent["entity"])

    if not relevant_names:
        for ent in identified_entities:
            if any(tok in ent["entity"].lower() for tok in query_tokens):
                relevant_names.add(ent["entity"])

    if not relevant_names and query_tokens:
        relevant_names = {ent["entity"] for ent in identified_entities[:min(len(identified_entities), 2)]}

    exact_metrics = _compute_retrieval_metrics([{"entity": r["entity"]} for r in exact_results], relevant_names, top_k)
    bm25_metrics = _compute_retrieval_metrics([{"entity": r["entity"]} for r in bm25_results], relevant_names, top_k)

    st.markdown(
        '<div class="kg-sim-summary">'
        f'<div class="kg-sim-stat"><div class="kg-sim-stat-label">Exact matches</div><div class="kg-sim-stat-value">{len(exact_matches)}</div></div>'
        f'<div class="kg-sim-stat"><div class="kg-sim-stat-label">Top BM25</div><div class="kg-sim-stat-value">{top_bm25["entity"] if top_bm25 else "N/A"}</div></div>'
        f'<div class="kg-sim-stat"><div class="kg-sim-stat-label">Top-K</div><div class="kg-sim-stat-value">{top_k}</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="kg-sim-panel"><h3>Exact / keyword retrieval</h3></div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([
                {
                    "Entity": r["entity"],
                    "Type": r["type"],
                    "Direct Match": r["match_status"],
                    "Binary Score": r["score"]
                }
                for r in exact_results
            ]),
            width="stretch",
            hide_index=True
        )

    with right:
        st.markdown('<div class="kg-sim-panel"><h3>BM25 retrieval</h3></div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([
                {
                    "Rank": r["rank"],
                    "Entity": r["entity"],
                    "Type": r["type"],
                    "BM25 Score": round(r["relevance_score"], 4),
                    "Confidence": r["confidence"]
                }
                for r in bm25_results
            ]),
            width="stretch",
            hide_index=True
        )

    st.divider()
    st.markdown('<div class="kg-sim-tag">Performance view</div>', unsafe_allow_html=True)
    st.subheader("Compare retrieval results", divider="gray")
    names = [r["entity"] for r in bm25_results]
    exact_scores = [next((x["score"] for x in exact_results if x["entity"] == name), 0.0) for name in names]
    prob_scores = [r["relevance_score"] for r in bm25_results]

    comparison_chart = go.Figure()
    comparison_chart.add_trace(go.Bar(name="Exact Retrieval", x=names, y=exact_scores, marker_color="#94a3b8"))
    comparison_chart.add_trace(go.Bar(name="BM25 Retrieval", x=names, y=prob_scores, marker_color="#2563eb"))
    comparison_chart.update_layout(
        title=f"Retrieval comparison for query: '{query}'",
        xaxis_title="Candidate entity",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(comparison_chart, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Evaluate retrieval quality", divider="gray")
    evaluation_df = pd.DataFrame([
        {"Metric": "Precision@K", "Exact Retrieval": exact_metrics["precision_k"], "BM25 Retrieval": bm25_metrics["precision_k"]},
        {"Metric": "Recall@K", "Exact Retrieval": exact_metrics["recall_k"], "BM25 Retrieval": bm25_metrics["recall_k"]},
        {"Metric": "F1@K", "Exact Retrieval": exact_metrics["f1_k"], "BM25 Retrieval": bm25_metrics["f1_k"]},
        {"Metric": "MRR", "Exact Retrieval": exact_metrics["mrr"], "BM25 Retrieval": bm25_metrics["mrr"]}
    ])
    st.dataframe(evaluation_df, width="stretch", hide_index=True)

    if exact_metrics["precision_k"] > bm25_metrics["precision_k"] and exact_metrics["recall_k"] <= bm25_metrics["recall_k"]:
        interpretation = "Exact Retrieval has the higher Precision@K for this query, while BM25 has the higher Recall@K over the current candidate set."
    elif bm25_metrics["precision_k"] >= exact_metrics["precision_k"] and bm25_metrics["recall_k"] > exact_metrics["recall_k"]:
        interpretation = "BM25 gives the stronger ranking for this query and candidate set, trading some exactness for broader coverage and a better recall curve."
    else:
        interpretation = "For this query and candidate set, the two methods are close; the difference is driven by the exact token overlap and the BM25 score distribution among candidate entities."

    with st.expander("Interpretation", expanded=True):
        st.markdown(
            f"{interpretation} Exact retrieval performs direct matching, while BM25 provides a ranked probabilistic retrieval score. "
            f"Precision measures correctness among the top K results, recall measures coverage of relevant entities, F1 balances both, and MRR measures how early the first relevant entity appears for the current query '{query}'."
        )

    if st.button("Record Current Trial", type="primary", use_container_width=True):
        trial_record = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "Query": query,
            "Entities": len(identified_entities),
            "K": top_k,
            "Exact Results": len(exact_matches),
            "BM25 Results": len(bm25_results),
            "Precision@K": bm25_metrics["precision_k"],
            "Recall@K": bm25_metrics["recall_k"],
            "F1@K": bm25_metrics["f1_k"],
            "MRR": bm25_metrics["mrr"],
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state["trials"].append(trial_record)
        st.toast("Trial recorded")

    if st.session_state.get("trials"):
        st.dataframe(pd.DataFrame(st.session_state["trials"]), width="stretch", hide_index=True)
        csv_bytes = pd.DataFrame(st.session_state["trials"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download trial data as CSV",
            data=csv_bytes,
            file_name="kgirs_exp6_simulation_trials.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_simulation_workspace():
    """Render the Simulation as a spacious, structured lab workspace."""
    st.markdown("""
<style>
.kg-work-intro { margin:0 0 1.6rem 0; padding:1.15rem 1.35rem; background:#F6FBFA;
  border-left:4px solid #0E7C7B; border-radius:8px; color:#000000; font-size:0.95rem; line-height:1.75; }
.kg-work-heading { display:flex; align-items:center; gap:0.85rem; margin:2.4rem 0 0.85rem; }
.kg-first-work-heading { margin-top: 0.6rem !important;}
.kg-work-number { width:1.8rem; height:1.8rem; border-radius:50%; background:#1F2937; color:#FFFFFF;
  display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:800; flex-shrink:0; }
.kg-work-title { font-size:1.18rem; font-weight:700; color:#000000; line-height:1.35; }
.kg-work-explain { margin:0 0 1.5rem; color:#000000; font-size:0.92rem; line-height:1.7; }
.kg-work-label { color:#0E7C7B; font-size:0.72rem; font-weight:800; letter-spacing:0.13em;
    text-transform:uppercase; margin:1.3rem 0 0.75rem; }
.kg-work-note { margin:0.75rem 0 0; color:#000000; font-size:0.88rem; line-height:1.65; }
.kg-work-stat { padding:0.85rem 1.1rem; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:8px; margin-bottom:0.75rem; }
.kg-work-stat-label { font-size:0.7rem; color:#000000; font-weight:800; letter-spacing:0.08em; text-transform:uppercase; }
.kg-work-stat-value { font-size:1.3rem; color:#000000; font-weight:800; margin-top:0.25rem; }
.kg-work-panel { background:#FEFDFB; border:1px solid #E7E2D3; border-radius:10px; padding:1.15rem 1.35rem 0.85rem; margin:1.3rem 0 1.5rem; }
.kg-work-panel-title { font-size:1rem; color:#000000; font-weight:700; margin:0 0 0.75rem; }
.stTabs [data-baseweb="tab-list"] { gap:0.85rem; border-bottom:2px solid #BFE3E1; padding-bottom:0.45rem; margin-top:0.8rem; margin-bottom:1.4rem; }
.stTabs [data-baseweb="tab"] { min-height:2.85rem; padding:0.7rem 1.25rem; color:#1F2937;
    background:#FEFDFB; border:1px solid #E7E2D3; border-radius:8px; font-weight:800; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color:#FFFFFF !important; background:#0E7C7B !important;
    border-color:#0E7C7B; box-shadow:0 3px 8px rgba(14,124,123,0.2); }
.stTabs [data-baseweb="tab-highlight"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.3rem !important; margin-bottom:1.6rem !important; }
.kg-eval-intro { margin:0.6rem 0 1.4rem; padding:0.95rem 1.15rem; background:#F6FBFA;
    border-left:3px solid #0E7C7B; border-radius:6px; color:#000000; font-size:0.9rem; line-height:1.65; }
.kg-eval-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1.2rem; margin-top:1.4rem; margin-bottom:1.5rem; }
.kg-eval-card { background:#FEFDFB; border:1px solid #E7E2D3; border-left:3px solid #B3261E;
    border-radius:8px; padding:0.95rem 1.1rem; }
.kg-eval-card-title { color:#000000; font-size:0.86rem; font-weight:800; margin-bottom:0.35rem; }
.kg-eval-card-copy { color:#000000; font-size:0.8rem; line-height:1.55; margin:0; }
@media (max-width: 800px) { .kg-eval-grid { grid-template-columns:1fr; } }
.kg-work-slider-help { color:#000000; font-size:0.84rem; line-height:1.55; margin:0.4rem 0 1.2rem; }
.kg-work-neutral { margin:1.2rem 0 1.5rem; padding:0.95rem 1.2rem; background:#F8F7F2; border:1px solid #E7E2D3;
    border-left:3px solid #0E7C7B; border-radius:6px; color:#000000; font-size:0.9rem; line-height:1.6; }
@media (max-width: 800px) { .kg-work-heading { margin-top:1.4rem; } }

/* Generous spacing around Streamlit widgets in the simulation workspace */
div[data-testid="stVerticalBlockBorderWrapper"] {
    margin-top: 0.9rem !important;
    margin-bottom: 1.85rem !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stContainer"] {
    padding: 1.35rem 1.55rem !important;
}
div[data-testid="stDataFrame"] {
    margin-top: 0.85rem !important;
    margin-bottom: 1.65rem !important;
}
div[data-testid="stPlotlyChart"] {
    margin-top: 1rem !important;
    margin-bottom: 1.8rem !important;
}
div[data-testid="stTextInput"], 
div[data-testid="stTextArea"], 
div[data-testid="stSelectbox"], 
div[data-testid="stRadio"] {
    margin-bottom: 1.35rem !important;
}
div[data-testid="stSlider"] {
    margin-top: 0.45rem !important;
    margin-bottom: 1rem !important;
}
div[data-testid="stButton"] button {
    margin-top: 0.55rem !important;
    margin-bottom: 1.1rem !important;
}
div[data-testid="stDownloadButton"] button {
    margin-top: 0.95rem !important;
    margin-bottom: 1.35rem !important;
}
hr[data-testid="stDivider"] {
    margin-top: 2.2rem !important;
    margin-bottom: 2.2rem !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="kg-work-heading kg-first-work-heading"><span class="kg-work-number">1</span><span class="kg-work-title">'
        'Prepare the source text</span></div>'
        '<p class="kg-work-explain">Choose a benchmark passage or provide custom text. Entity extraction is '
        'performed only after the source is selected, making the input-to-graph transition visible.</p>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        mode_col, source_col = st.columns([1, 2.2], gap="large")
        with mode_col:
            st.markdown('<p class="kg-work-label">Source mode</p>', unsafe_allow_html=True)
            input_mode = st.radio(
                "Source mode",
                ["Benchmark preset", "Custom text"],
                label_visibility="collapsed",
                key="simulation_input_mode"
            )
        with source_col:
            if input_mode == "Benchmark preset":
                preset_name = st.selectbox(
                    "Benchmark passage", list(SAMPLE_PRESETS.keys()), key="simulation_preset"
                )
                active_text = SAMPLE_PRESETS[preset_name]
            else:
                active_text = st.text_area(
                    "Custom source text",
                    value=st.session_state.get(
                        "custom_text_val",
                        "Sundar Pichai is the CEO of Google and studied at Stanford University in California."
                    ),
                    height=105,
                    key="simulation_custom_text"
                )
                st.session_state["custom_text_val"] = active_text

        if st.session_state.get("last_extracted_text") != active_text.strip():
            st.session_state["identified_entities"] = extract_entities_from_text(active_text)
            st.session_state["last_extracted_text"] = active_text.strip()
            st.session_state["retrieval_run"] = False

        preview_col, action_col = st.columns([4, 1], gap="medium")
        with preview_col:
            st.text_area("Selected passage", value=active_text, height=92, disabled=True)
        with action_col:
            st.markdown('<p class="kg-work-label">Action</p>', unsafe_allow_html=True)
            if st.button("Extract entities", type="primary", width="stretch", key="extract_entities"):
                st.session_state["identified_entities"] = extract_entities_from_text(active_text)
                st.session_state["last_extracted_text"] = active_text.strip()
                st.session_state["retrieval_run"] = False
                st.toast("Candidate entities refreshed")

    identified_entities = st.session_state.get("identified_entities", [])
    if not identified_entities:
        st.markdown('<p class="kg-work-neutral">No entities were detected. Try a passage containing names, organizations, locations, or roles.</p>', unsafe_allow_html=True)
        return

    st.markdown(
        '<div class="kg-work-heading"><span class="kg-work-number">2</span><span class="kg-work-title">'
        'Inspect the extracted graph candidates</span></div>'
        '<p class="kg-work-explain">The table preserves the detected mention and its local context. The graph view '
        'summarizes the same candidates visually; it is a compact inspection aid, not a second data source.</p>',
        unsafe_allow_html=True
    )

    stat_cols = st.columns(3, gap="medium")
    stats = [
        ("Entity count", str(len(identified_entities))),
        ("Entity types", str(len(set(entity["type"] for entity in identified_entities)))),
        ("Candidate space", "Graph")
    ]
    for col, (label, value) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f'<div class="kg-work-stat"><div class="kg-work-stat-label">{label}</div>'
                f'<div class="kg-work-stat-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<p class="kg-work-label" style="margin-top:1.4rem;">Detected entities and context</p>', unsafe_allow_html=True)
    entity_df = pd.DataFrame([
        {"Entity": entity["entity"], "Type": entity["type"], "Context": entity["context"]}
        for entity in identified_entities
    ])
    st.dataframe(entity_df, width="stretch", hide_index=True)
    st.markdown('<p class="kg-work-label" style="margin-top:1.5rem;">Candidate graph</p>', unsafe_allow_html=True)
    render_candidate_graph(identified_entities)

    st.markdown(
        '<div class="kg-work-heading"><span class="kg-work-number">3</span><span class="kg-work-title">'
        'Configure the retrieval experiment</span></div>'
        '<p class="kg-work-explain">Use an entity name or a related term as the query. Exact retrieval reports '
        'binary matches; BM25 ranks candidates using term frequency, IDF, and document-length normalization.</p>',
        unsafe_allow_html=True
    )

    candidate_suggestions = list(dict.fromkeys(entity["entity"] for entity in identified_entities))[:6]
    with st.container(border=True):
        query = st.text_input(
            "Search query",
            value=st.session_state.get("active_query", candidate_suggestions[0] if candidate_suggestions else "Google"),
            help="Try a full entity, a single entity token, or a related context term.",
            key="simulation_query"
        )
        st.session_state["active_query"] = query
        if candidate_suggestions:
            st.markdown('<p class="kg-work-label">Suggested queries</p>', unsafe_allow_html=True)
            st.caption("  |  ".join(candidate_suggestions))

        st.markdown('<p class="kg-work-label" style="margin-top:1.4rem;">BM25 controls</p>', unsafe_allow_html=True)
        control_cols = st.columns(3, gap="large")
        with control_cols[0]:
            top_k = st.slider(
                "Top-K results", min_value=1, max_value=10, value=3, step=1,
                key="simulation_top_k",
                help="Controls how many ranked candidates are included in the evaluation metrics."
            )
            st.markdown('<p class="kg-work-slider-help"><b>Top-K:</b> highest-ranked candidates evaluated.</p>', unsafe_allow_html=True)
        with control_cols[1]:
            bm25_k1 = st.slider(
                "k1 &middot; term saturation", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                key="simulation_k1",
                help="Higher k1 lets repeated query terms contribute more before the score saturates."
            )
            st.markdown('<p class="kg-work-slider-help"><b>k1:</b> higher values preserve repeated-term influence.</p>', unsafe_allow_html=True)
        with control_cols[2]:
            bm25_b = st.slider(
                "b &middot; length normalization", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                key="simulation_b",
                help="Higher b penalizes long candidate contexts more strongly; 0 disables length normalization."
            )
            st.markdown('<p class="kg-work-slider-help"><b>b:</b> higher values penalize long contexts.</p>', unsafe_allow_html=True)

        run_col, status_col = st.columns([1, 3], gap="medium")
        with run_col:
            if st.button("Run comparison", type="primary", width="stretch", key="run_retrieval"):
                st.session_state["retrieval_run"] = True
        with status_col:
            if not st.session_state.get("retrieval_run", False):
                st.markdown('<p class="kg-work-neutral">Set the controls, then run the comparison to generate ranked results and evaluation metrics.</p>', unsafe_allow_html=True)
                return

    exact_results = run_exact_retrieval(query, identified_entities)
    bm25_results = run_probabilistic_retrieval(query, identified_entities, k1=bm25_k1, b=bm25_b)
    exact_matches = [result for result in exact_results if result["exact_match"]]
    top_bm25 = bm25_results[0] if bm25_results else None

    relevant_names = set()
    query_tokens = [token for token in re.findall(r"\w+", query.lower()) if len(token) > 1]
    for entity in identified_entities:
        name = entity["entity"].lower()
        context = entity.get("context", "").lower()
        if query.lower().strip() == name or any(token in name or token in context for token in query_tokens):
            relevant_names.add(entity["entity"])
    if not relevant_names and query_tokens:
        relevant_names = {entity["entity"] for entity in identified_entities[:min(len(identified_entities), 2)]}

    exact_metrics = _compute_retrieval_metrics(
        [{"entity": result["entity"]} for result in exact_results], relevant_names, top_k
    )
    bm25_metrics = _compute_retrieval_metrics(
        [{"entity": result["entity"]} for result in bm25_results], relevant_names, top_k
    )

    st.markdown(
        '<div class="kg-work-heading"><span class="kg-work-number">4</span><span class="kg-work-title">'
        'Read and evaluate the retrieval results</span></div>'
        '<p class="kg-work-explain">The result tabs separate the two retrieval outputs from the visual comparison '
        'and quality evaluation, keeping each answer readable while preserving the full experiment trace.</p>',
        unsafe_allow_html=True
    )

    result_stats = st.columns(3, gap="medium")
    result_values = [
        ("Exact matches", str(len(exact_matches))),
        ("Top BM25 entity", top_bm25["entity"] if top_bm25 else "N/A"),
        ("Evaluation depth", f"Top-{top_k}")
    ]
    for col, (label, value) in zip(result_stats, result_values):
        with col:
            st.markdown(
                f'<div class="kg-work-stat"><div class="kg-work-stat-label">{label}</div>'
                f'<div class="kg-work-stat-value">{value}</div></div>', unsafe_allow_html=True
            )

    result_tab, chart_tab, evaluation_tab = st.tabs(
        ["Ranked outputs", "Score comparison", "Quality evaluation"]
    )
    with result_tab:
        exact_col, bm25_col = st.columns(2, gap="large")
        with exact_col:
            st.markdown('<p class="kg-work-label">Exact / keyword retrieval</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Entity": result["entity"], "Type": result["type"], "Match": result["match_status"], "Score": result["score"]}
                for result in exact_results
            ]), width="stretch", hide_index=True)
        with bm25_col:
            st.markdown('<p class="kg-work-label">BM25 ranked retrieval</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Rank": result["rank"], "Entity": result["entity"], "Type": result["type"],
                 "Score": result["relevance_score"], "Confidence": result["confidence"]}
                for result in bm25_results
            ]), width="stretch", hide_index=True)

    names = [result["entity"] for result in bm25_results]
    exact_scores = [next((item["score"] for item in exact_results if item["entity"] == name), 0.0) for name in names]
    probabilistic_scores = [result["relevance_score"] for result in bm25_results]
    with chart_tab:
        comparison_chart = go.Figure()
        comparison_chart.add_trace(go.Bar(name="Exact Retrieval", x=names, y=exact_scores, marker_color="#B3261E"))
        comparison_chart.add_trace(go.Bar(name="BM25 Retrieval", x=names, y=probabilistic_scores, marker_color="#0E7C7B"))
        comparison_chart.update_layout(
            title=f"Retrieval scores for query: {query}", xaxis_title="Candidate entity", yaxis_title="Score",
            yaxis=dict(range=[0, max(1.1, max(probabilistic_scores or [1]) * 1.12)]), barmode="group", height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(comparison_chart, use_container_width=True, config={"displayModeBar": False})
        st.caption("Exact Retrieval is binary. BM25 preserves relative relevance differences across candidates.")

    with evaluation_tab:
        st.markdown(
            '<div class="kg-eval-intro"><b>How to read this comparison.</b> '
            'The table measures two different ranking behaviors on the same extracted candidates. '
            'Exact Retrieval rewards direct name or token matches, while BM25 gives graded scores and can rank partial contextual matches. '
            'Higher values indicate stronger retrieval quality for the current query and Top-K setting.</div>',
            unsafe_allow_html=True
        )
        evaluation_df = pd.DataFrame([
            {"Metric": "Precision@K", "Exact Retrieval": exact_metrics["precision_k"], "BM25 Retrieval": bm25_metrics["precision_k"]},
            {"Metric": "Recall@K", "Exact Retrieval": exact_metrics["recall_k"], "BM25 Retrieval": bm25_metrics["recall_k"]},
            {"Metric": "F1@K", "Exact Retrieval": exact_metrics["f1_k"], "BM25 Retrieval": bm25_metrics["f1_k"]},
            {"Metric": "MRR", "Exact Retrieval": exact_metrics["mrr"], "BM25 Retrieval": bm25_metrics["mrr"]}
        ])
        st.dataframe(evaluation_df, width="stretch", hide_index=True)
        st.markdown(
            '<div class="kg-eval-grid">'
            '<div class="kg-eval-card"><div class="kg-eval-card-title">Precision@K · correctness</div>'
            '<p class="kg-eval-card-copy">Among the first K candidates, the proportion judged relevant. It answers: “Did the returned results match the query?”</p></div>'
            '<div class="kg-eval-card"><div class="kg-eval-card-title">Recall@K · coverage</div>'
            '<p class="kg-eval-card-copy">The proportion of all relevant candidates found within the first K positions. It answers: “How much relevant information was surfaced?”</p></div>'
            '<div class="kg-eval-card"><div class="kg-eval-card-title">F1@K · balance</div>'
            '<p class="kg-eval-card-copy">The harmonic mean of precision and recall. It is high only when the ranking is both accurate and sufficiently complete.</p></div>'
            '<div class="kg-eval-card"><div class="kg-eval-card-title">MRR · first useful result</div>'
            '<p class="kg-eval-card-copy">The reciprocal rank of the first relevant candidate. A value of 1.0 means the first result is already relevant.</p></div>'
            '</div>',
            unsafe_allow_html=True
        )
        if exact_metrics["precision_k"] > bm25_metrics["precision_k"] and exact_metrics["recall_k"] <= bm25_metrics["recall_k"]:
            interpretation = "Exact Retrieval is more precise for this query, while BM25 provides broader candidate coverage."
        elif bm25_metrics["precision_k"] >= exact_metrics["precision_k"] and bm25_metrics["recall_k"] > exact_metrics["recall_k"]:
            interpretation = "BM25 gives the stronger ranking for this candidate set because it preserves graded lexical evidence."
        else:
            interpretation = "The methods are close here; the difference is driven by token overlap and BM25 normalization."
        st.markdown(
            f'<div class="kg-work-panel"><div class="kg-work-panel-title">Interpretation for this trial</div>'
            f'<p class="kg-work-note">{interpretation}</p>'
            f'<p class="kg-work-note"><b>Reading the result:</b> compare the two columns rather than looking only at one score. '
            f'If precision rises, the method is returning fewer incorrect candidates; if recall rises, it is surfacing more of the relevant candidate space. '
            f'F1 summarizes that trade-off, while MRR tells you how quickly a useful result appears.</p></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">5</span>'
        '<span class="kg-work-title">Record and review trial history</span></div>'
        '<p class="kg-work-explain">Save the current query, parameters, and evaluation metrics as one reproducible trial. Trial history stays separate from the live comparison so the results remain easy to read.</p>',
        unsafe_allow_html=True
    )
    if st.button("Record current trial", type="primary", width="stretch", key="record_trial"):
        st.session_state["trials"].append({
            "Trial #": len(st.session_state["trials"]) + 1, "Query": query,
            "Entities": len(identified_entities), "K": top_k, "Exact Results": len(exact_matches),
            "BM25 Results": len(bm25_results), "Precision@K": bm25_metrics["precision_k"],
            "Recall@K": bm25_metrics["recall_k"], "F1@K": bm25_metrics["f1_k"], "MRR": bm25_metrics["mrr"],
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.toast("Trial recorded")
    if st.session_state.get("trials"):
        trials_df = pd.DataFrame(st.session_state["trials"])
        st.dataframe(trials_df, width="stretch", hide_index=True)
        st.download_button(
            "Download trial data as CSV", data=trials_df.to_csv(index=False).encode("utf-8"),
            file_name="kgirs_exp6_simulation_trials.csv", mime="text/csv", width="stretch"
        )
    else:
        st.markdown('<p class="kg-work-neutral">No trials have been recorded in this session.</p>', unsafe_allow_html=True)


def render_quiz_section():
    """Renders Section 4: Quiz Assessment (10 questions randomly sampled from the question bank)."""
    col_btn, _ = st.columns([1.5, 3])
    with col_btn:
        if st.button("Generate new question set", type="secondary", width="stretch", key="refresh_quiz"):
            bank_ids = [q["id"] for q in QUIZ_QUESTION_BANK]
            st.session_state["quiz_question_ids"] = random.sample(bank_ids, min(10, len(bank_ids)))
            st.session_state["quiz_answers"] = {}
            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_score"] = 0
            st.session_state["quiz_set_version"] += 1
            st.rerun()

    version = st.session_state["quiz_set_version"]
    active_ids = st.session_state["quiz_question_ids"]
    questions_by_id = {q["id"]: q for q in QUIZ_QUESTION_BANK}
    active_questions = [questions_by_id[qid] for qid in active_ids]

    st.markdown(
        '<div class="kg-work-heading kg-first-work-heading"><span class="kg-work-number">1</span>'
        '<span class="kg-work-title">Assessment questions</span></div>'
        '<p class="kg-work-explain">Select the best answer for each question, then submit the form for automated grading and concept rationales.</p>',
        unsafe_allow_html=True
    )

    with st.form("quiz_form"):
        user_responses = {}
        for display_idx, q in enumerate(active_questions, start=1):
            st.markdown(
                f'<div class="kg-quiz-card">'
                f'<div><span class="kg-quiz-qnum">{display_idx}</span>'
                f'<span class="kg-work-label" style="margin:0;">Question {display_idx} of {len(active_questions)}</span></div>'
                f'<p class="kg-quiz-qtitle">{q["question"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            selected = st.radio(
                label=f"Options for Quiz Q{display_idx}",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{version}_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary", width="stretch")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.markdown(
            '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">2</span>'
            '<span class="kg-work-title">Evaluation results and detailed feedback</span></div>'
            '<p class="kg-work-explain">Review your performance alongside concept rationales for each question.</p>',
            unsafe_allow_html=True
        )

        for display_idx, q in enumerate(active_questions, start=1):
            user_ans = user_responses.get(q["id"])
            corr_ans = q["answer_index"]
            if user_ans == corr_ans:
                score += 1
                st.markdown(
                    f'<div class="kg-feedback-card correct">'
                    f'<span class="kg-work-label" style="color:#0E7C7B; margin:0 0 0.35rem 0; display:block;">Question {display_idx}: Correct</span>'
                    f'<p style="font-size:0.95rem; font-weight:700; color:#000000; margin:0 0 0.4rem 0;">{q["question"]}</p>'
                    f'<p style="font-size:0.9rem; color:#000000; margin:0 0 0.3rem 0;"><b>Selected:</b> {q["options"][user_ans]}</p>'
                    f'<p style="font-size:0.88rem; color:#000000; line-height:1.6; margin:0;"><b>Reasoning:</b> <i>{q["explanation"]}</i></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="kg-feedback-card incorrect">'
                    f'<span class="kg-work-label" style="color:#B3261E; margin:0 0 0.35rem 0; display:block;">Question {display_idx}: Incorrect</span>'
                    f'<p style="font-size:0.95rem; font-weight:700; color:#000000; margin:0 0 0.4rem 0;">{q["question"]}</p>'
                    f'<p style="font-size:0.9rem; color:#000000; margin:0 0 0.25rem 0;"><b>Your answer:</b> {q["options"][user_ans]}</p>'
                    f'<p style="font-size:0.9rem; color:#000000; margin:0 0 0.3rem 0;"><b>Correct answer:</b> {q["options"][corr_ans]}</p>'
                    f'<p style="font-size:0.88rem; color:#000000; line-height:1.6; margin:0;"><b>Reasoning:</b> <i>{q["explanation"]}</i></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.session_state["quiz_score"] = score
        perc = (score / len(active_questions)) * 100
        status_label = "Distinction" if perc >= 80 else ("Passed" if perc >= 50 else "Review Needed")
        stat_cols = st.columns(3, gap="medium")
        with stat_cols[0]:
            st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Final Score</div><div class="kg-work-stat-value">{score} / {len(active_questions)}</div></div>', unsafe_allow_html=True)
        with stat_cols[1]:
            st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Percentage</div><div class="kg-work-stat-value">{perc:.0f}%</div></div>', unsafe_allow_html=True)
        with stat_cols[2]:
            st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Assessment Result</div><div class="kg-work-stat-value" style="font-size:1.15rem;">{status_label}</div></div>', unsafe_allow_html=True)

    elif st.session_state.get("quiz_submitted", False):
        st.markdown(
            f'<div class="kg-work-panel">'
            f'<div class="kg-work-panel-title">Assessment status</div>'
            f'<p class="kg-work-note">Quiz already completed in this session. Recorded score: <b>{st.session_state.get("quiz_score", 0)} / {len(active_questions)}</b>. You can submit again using the form above or click "Generate new question set" to attempt a new question sample.</p>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_references_section():
    """Renders Section 7: Academic References & Textbooks."""
    st.markdown("### Academic References & Recommended Reading")
    st.markdown("""
1. **Robertson, S. E., & Zaragoza, H. (2009)**. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.
2. **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**. *Introduction to Information Retrieval*. Cambridge University Press.
3. **Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2021)**. *A Survey on Knowledge Graphs: Representation, Acquisition, and Applications*. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494-514.
4. **Xiong, C., Power, R., & Callan, J. (2017)**. *Explicit Semantic Ranking for Academic Search via Knowledge Graph Embedding*. Proceedings of the 26th International Conference on World Wide Web (WWW '17), 1271-1279.
5. **IIT Kharagpur Virtual Laboratories Project**. *Virtual Laboratory System Architecture and Pedagogical Guidelines*. Ministry of Education, Government of India.
""")


def render_report_section():
    """Renders Section 5: Dynamic Lab Report Generator with PDF Export."""
    st.markdown(
        '<div class="kg-work-heading"><span class="kg-work-number">1</span>'
        '<span class="kg-work-title">Student credentials and laboratory metadata</span></div>'
        '<p class="kg-work-explain">Enter your official student name and roll number to associate with this experiment report.</p>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"), key="rep_name_input")
        with col2:
            student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-006"), key="rep_id_input")
        with col3:
            lab_date = st.date_input("Experiment Date", value=datetime.now(), key="rep_date_input")

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">2</span>'
        '<span class="kg-work-title">Observations and analysis notes</span></div>'
        '<p class="kg-work-explain">Record your interpretation of results, observations, and conclusions for inclusion in the final report.</p>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        student_notes = st.text_area(
            "Enter your interpretation of results, observations, and conclusions:",
            value=st.session_state.get("student_notes", (
                "The experimental trials demonstrated that entity-aware probabilistic ranking consistently outperformed "
                "the standard Okapi BM25 keyword baseline across Precision@K, MAP, and NDCG@K metrics by capturing "
                "disambiguated semantic relationships between persons, organizations, locations, and concepts."
            )),
            height=120,
            key="rep_notes_input"
        )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">3</span>'
        '<span class="kg-work-title">Report summary preview</span></div>'
        '<p class="kg-work-explain">Verify the session metadata, diagnostic assessment score, and recorded simulation trials.</p>',
        unsafe_allow_html=True
    )

    quiz_total = len(st.session_state.get("quiz_question_ids", []))
    stat_cols = st.columns(4, gap="medium")
    with stat_cols[0]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Student</div><div class="kg-work-stat-value" style="font-size:1.05rem;">{student_name}</div></div>', unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Roll / ID</div><div class="kg-work-stat-value" style="font-size:1.05rem;">{student_id}</div></div>', unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Quiz Score</div><div class="kg-work-stat-value">{st.session_state.get("quiz_score", 0)} / {quiz_total}</div></div>', unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Trials Logged</div><div class="kg-work-stat-value">{len(trials_df)}</div></div>', unsafe_allow_html=True)

    if not trials_df.empty:
        st.markdown('<p class="kg-work-label">Recorded experimental trials</p>', unsafe_allow_html=True)
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.markdown('<p class="kg-work-neutral">Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.</p>', unsafe_allow_html=True)

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(st.session_state.get("quiz_question_ids", [])),
        student_notes=student_notes
    )

    # Save to local files for guaranteed download
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">4</span>'
        '<span class="kg-work-title">Download official lab report (.pdf)</span></div>'
        '<p class="kg-work-explain">Export your complete experimental report for academic verification and grading.</p>',
        unsafe_allow_html=True
    )

    col_btn1, col_btn2 = st.columns(2, gap="medium")
    with col_btn1:
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            width="stretch"
        )

    with col_btn2:
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="kgirs_exp6_lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


def render_certificate_section():
    """Renders Section 6: Certificate of Completion."""
    st.markdown(
        '<div class="kg-work-heading"><span class="kg-work-number">1</span>'
        '<span class="kg-work-title">Recipient credentials and completion date</span></div>'
        '<p class="kg-work-explain">Confirm the recipient name and roll number to be embossed on the official certificate.</p>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            cert_name = st.text_input(
                "Student Name", value=st.session_state["student_info"].get("name", "Student Name"), key="cert_name_input"
            )
        with col2:
            cert_id = st.text_input(
                "Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-006"), key="cert_id_input"
            )
        with col3:
            cert_date = st.date_input("Completion Date", value=datetime.now(), key="cert_date_input")

    st.session_state["student_info"]["name"] = cert_name
    st.session_state["student_info"]["id"] = cert_id
    st.session_state["student_info"]["date"] = str(cert_date)

    quiz_total = len(st.session_state.get("quiz_question_ids", []))
    quiz_score = st.session_state.get("quiz_score", 0)

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">2</span>'
        '<span class="kg-work-title">Assessment status verification</span></div>'
        '<p class="kg-work-explain">The certificate includes your quiz score achieved during this session.</p>',
        unsafe_allow_html=True
    )

    cert_stats = st.columns(3, gap="medium")
    with cert_stats[0]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Student Name</div><div class="kg-work-stat-value" style="font-size:1.05rem;">{cert_name}</div></div>', unsafe_allow_html=True)
    with cert_stats[1]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Roll / ID</div><div class="kg-work-stat-value" style="font-size:1.05rem;">{cert_id}</div></div>', unsafe_allow_html=True)
    with cert_stats[2]:
        st.markdown(f'<div class="kg-work-stat"><div class="kg-work-stat-label">Quiz Score on Record</div><div class="kg-work-stat-value">{quiz_score} / {quiz_total}</div></div>', unsafe_allow_html=True)

    if not st.session_state.get("quiz_submitted", False):
        st.markdown(
            '<p class="kg-work-neutral">You have not submitted the Quiz yet; your certificate will show a score of 0 until you complete it. '
            '(The certificate itself is available regardless.)</p>',
            unsafe_allow_html=True
        )

    cert_bytes = generate_certificate_pdf(
        student_name=cert_name,
        student_id=cert_id,
        date_str=str(cert_date),
        quiz_score=quiz_score,
        quiz_total=quiz_total
    )

    st.markdown(
        '<div class="kg-work-heading" style="margin-top:2.4rem;"><span class="kg-work-number">3</span>'
        '<span class="kg-work-title">Download certificate (.pdf)</span></div>'
        '<p class="kg-work-explain">Generate and export your verified completion certificate in landscape PDF format.</p>',
        unsafe_allow_html=True
    )

    st.download_button(
        label="Download Certificate (.pdf)",
        data=cert_bytes,
        file_name="kgirs_exp6_certificate.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch"
    )


# ======================================================================================
# 6. MAIN NAVIGATION & SESSION STATE
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_question_ids" not in st.session_state:
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
        page_title="Virtual Lab Experiment 6",
        page_icon=None,
        layout="wide"
    )

    st.markdown("""
<style>
html { font-size: 125%; }
body, [class*="css"], .stMarkdown, p, span, label, div, ol, ul, li {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: #000000;
  line-height: 1.7;
}
h1, h2, h3, h4, h5, h6 {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
  color: #000000 !important;
  font-weight: 700 !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * ,
small {
  opacity: 1 !important;
  color: #000000 !important;
}

/* Shared UI Components matching Purpose, Theory, Simulation */
.kg-work-intro {
  margin: 0 0 1.6rem 0;
  padding: 1.15rem 1.35rem;
  background: #F6FBFA;
  border-left: 4px solid #0E7C7B;
  border-radius: 8px;
  color: #000000;
  font-size: 0.95rem;
  line-height: 1.75;
}
.kg-work-heading {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  margin: 2.4rem 0 0.85rem;
}
.kg-work-number {
  width: 1.8rem;
  height: 1.8rem;
  border-radius: 50%;
  background: #1F2937;
  color: #FFFFFF;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 800;
  flex-shrink: 0;
}
.kg-work-title {
  font-size: 1.18rem;
  font-weight: 700;
  color: #000000;
  line-height: 1.35;
}
.kg-work-explain {
  margin: 0 0 1.5rem;
  color: #000000;
  font-size: 0.92rem;
  line-height: 1.7;
}
.kg-work-label {
  color: #0E7C7B;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin: 1.3rem 0 0.75rem;
}
.kg-work-stat {
  padding: 0.85rem 1.1rem;
  background: #FEFDFB;
  border: 1px solid #E7E2D3;
  border-radius: 8px;
  margin-bottom: 0.75rem;
}
.kg-work-stat-label {
  font-size: 0.7rem;
  color: #000000;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.kg-work-stat-value {
  font-size: 1.3rem;
  color: #000000;
  font-weight: 800;
  margin-top: 0.25rem;
}
.kg-work-panel {
  background: #FEFDFB;
  border: 1px solid #E7E2D3;
  border-radius: 10px;
  padding: 1.15rem 1.35rem 0.85rem;
  margin: 1.3rem 0 1.5rem;
}
.kg-work-panel-title {
  font-size: 1rem;
  color: #000000;
  font-weight: 700;
  margin: 0 0 0.75rem;
}
.kg-work-note {
  margin: 0.75rem 0 0;
  color: #000000;
  font-size: 0.88rem;
  line-height: 1.65;
}
.kg-work-neutral {
  margin: 1.2rem 0 1.5rem;
  padding: 0.95rem 1.2rem;
  background: #F8F7F2;
  border: 1px solid #E7E2D3;
  border-left: 3px solid #0E7C7B;
  border-radius: 6px;
  color: #000000;
  font-size: 0.9rem;
  line-height: 1.6;
}
.kg-quiz-card {
  background: #FEFDFB;
  border: 1px solid #E7E2D3;
  border-radius: 10px;
  padding: 1.15rem 1.35rem 0.75rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 1px 2px rgba(31,41,55,0.03);
}
.kg-quiz-qnum {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  background: #1F2937;
  color: #FFFFFF;
  font-size: 0.72rem;
  font-weight: 800;
  margin-right: 0.5rem;
  vertical-align: middle;
}
.kg-quiz-qtitle {
  font-size: 1.02rem;
  font-weight: 700;
  color: #000000;
  margin: 0.55rem 0 0.75rem;
  line-height: 1.5;
}
.kg-feedback-card {
  border-radius: 8px;
  padding: 1.1rem 1.35rem;
  margin-bottom: 1rem;
}
.kg-feedback-card.correct {
  background: #F6FBFA;
  border: 1px solid #BFE3E1;
  border-left: 4px solid #0E7C7B;
}
.kg-feedback-card.incorrect {
  background: #FEFAF9;
  border: 1px solid #F1D6D2;
  border-left: 4px solid #B3261E;
}
.kg-ref-card {
  background: #FEFDFB;
  border: 1px solid #E7E2D3;
  border-left: 4px solid #0E7C7B;
  border-radius: 10px;
  padding: 1.15rem 1.35rem;
  margin-bottom: 1.15rem;
  box-shadow: 0 1px 2px rgba(31,41,55,0.03);
}
.kg-ref-head {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  margin-bottom: 0.45rem;
}
.kg-ref-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.55rem;
  height: 1.55rem;
  border-radius: 50%;
  background: #1F2937;
  color: #FFFFFF;
  font-size: 0.75rem;
  font-weight: 800;
  flex-shrink: 0;
}
.kg-ref-author {
  font-size: 0.98rem;
  font-weight: 700;
  color: #000000;
}
.kg-ref-title {
  font-size: 0.92rem;
  font-style: italic;
  color: #000000;
  line-height: 1.55;
  margin: 0.25rem 0;
}
.kg-ref-venue {
  font-size: 0.85rem;
  color: #4B5563;
  line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

    init_session_state()

    # Navigation Sidebar matching IIT Kharagpur VLab page navigation
    section = st.sidebar.radio(
        "Lab Navigation",
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

    # Section Dispatcher
    if section == "Purpose":
        render_purpose_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_workspace()
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
