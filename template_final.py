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
            "explanation": "Recall@K measures retrieval completeness: relevant entities retrieved in the top K divided by all known relevant entities."
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
            "explanation": "NDCG is normalized against the ideal ranking, so a perfect ranking has a value of 1.0."
        },
        {
            "id": 19,
            "question": "What effect does increasing an entity boost factor have on a hybrid ranking?",
            "options": [
                "A) It has no effect on ranking",
                "B) It makes entity matches dominate lexical scores",
                "C) It removes all query terms",
                "D) It sorts candidates alphabetically"
            ],
            "answer_index": 1,
            "explanation": "A larger entity boost gives structured entity evidence more influence than lexical evidence."
        },
        {
            "id": 20,
            "question": "Why does combining structured graph information with lexical retrieval create hybrid search?",
            "options": [
                "A) It merges unrelated database engines",
                "B) It combines lexical term matching with structured entity signals",
                "C) It retrieves from two physical servers",
                "D) It combines image and text search"
            ],
            "answer_index": 1,
            "explanation": "Hybrid search combines unstructured lexical evidence with structured entity evidence."
        }
    ]

# The following corpus remains available to the existing non-Simulation helpers.
DATA_CORPORA = {
    "AI & Deep Learning Pioneers (Domain 1)": {
        "description": "Corpus on fictional Artificial Intelligence researchers and research institutions.",
        "documents": []
    },
    "Enterprise Cloud & Systems (Domain 2)": {
        "description": "Corpus on fictional enterprise cloud infrastructure and leadership.",
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
                "triples": [],
                "relevant_to": ["priya anand", "meridian systems cloud", "redmond cloud", "distributed computing"]
            },
            {
                "doc_id": "DOC-202",
                "title": "Recovered legacy placeholder",
                "text": "",
                "entities": [],
                "triples": [],
                "relevant_to": [],
                "legacy_note": """st.header("Simulation: Identify Graph Entities")
                st.info("Follow the experiment in order: identify entities, query the candidate set, compare retrieval methods, then evaluate the results.")

                st.subheader("1. Identifying Graph Entities")
                st.write("Scenario: a short passage contains people, organizations, places, and concepts. Your first task is to identify the graph entities that can later be searched.")
                input_mode = st.radio("Choose the experiment input:", ["Benchmark preset", "Custom text"], horizontal=True, key="template_input_mode")
                if input_mode == "Benchmark preset":
                    preset_names = list(SAMPLE_PRESETS)
                    if "template_preset_text" not in st.session_state:
                        st.session_state["template_preset_text"] = SAMPLE_PRESETS[preset_names[0]]
                    st.selectbox("Select a benchmark passage:", preset_names, key="template_preset", on_change=_select_template_preset)
                    active_text = st.text_area("Passage to identify:", height=100, key="template_preset_text")
                else:
                    active_text = st.text_area(
                        "Paste a passage to identify:",
                        value=st.session_state.get("custom_text_val", SAMPLE_PRESETS[preset_names[0]] if "preset_names" in locals() else SAMPLE_PRESETS[list(SAMPLE_PRESETS)[0]]),
                        height=100,
                        key="template_custom_text"
                    )
                    st.session_state["custom_text_val"] = active_text

                text_signature = active_text.strip()
                if st.session_state.get("last_extracted_text") != text_signature:
                    st.session_state["identified_entities"] = []
                    st.session_state["retrieval_run"] = False
                    st.session_state["active_query"] = ""
                    st.session_state["template_query"] = ""
                if st.button("Identify Graph Entities", type="primary", key="template_identify"):
                    st.session_state["identified_entities"] = extract_template_entities(active_text)
                    st.session_state["last_extracted_text"] = text_signature
                    st.session_state["retrieval_run"] = False

                identified_entities = st.session_state.get("identified_entities", [])
                if not identified_entities:
                    st.info("Choose or enter a passage, then click **Identify Graph Entities**. The extracted entities will become the candidate search space.")
                    return

                st.success(f"Identification complete: {len(identified_entities)} candidate graph entities found.")
                entity_df = pd.DataFrame([
                    {
                        "Entity": item["entity"],
                        "Type": item["type"],
                        "Context / Span": f"{item['context']} [{item['start']}, {item['end']}]"
                    }
                    for item in identified_entities
                ])
                st.dataframe(entity_df, use_container_width=True, hide_index=True)
                st.caption("Observation: these extracted entities now form the candidate search space. No relationships are inferred in this experiment; relationship extraction belongs to Experiment 7.")

                graph = go.Figure()
                type_colors = {"PERSON": "#2563EB", "ORGANIZATION": "#059669", "LOCATION": "#D97706", "CONCEPT": "#DC2626", "ROLE_TITLE": "#7C3AED", "AWARD_EVENT": "#DB2777"}
                entity_types = list(dict.fromkeys(item["type"] for item in identified_entities))
                for entity_type in entity_types:
                    typed_entities = [item for item in identified_entities if item["type"] == entity_type]
                    graph.add_trace(go.Scatter(
                        x=list(range(len(typed_entities))),
                        y=[entity_types.index(entity_type)] * len(typed_entities),
                        mode="markers+text",
                        text=[item["entity"] for item in typed_entities],
                        textposition="top center",
                        name=entity_type,
                        marker={"size": 24, "color": type_colors.get(entity_type, "#64748B"), "line": {"width": 2, "color": "white"}},
                        hovertemplate="<b>%{text}</b><br>Type: " + entity_type + "<extra></extra>"
                    ))
                graph.update_layout(
                    title="Candidate Knowledge Graph: Identified Entities",
                    height=340,
                    xaxis={"visible": False},
                    yaxis={"visible": False},
                    showlegend=True,
                    legend={"orientation": "h", "y": -0.12},
                    margin={"l": 20, "r": 20, "t": 55, "b": 65}
                )
                st.plotly_chart(graph, use_container_width=True, key="template_candidate_graph")

                st.divider()
                st.subheader("2. Querying")
                st.write("Enter the information need you want to find among the candidate graph entities. The examples below are taken directly from the entities identified above.")
                quick_queries = _template_quick_queries(identified_entities)
                query_columns = st.columns(max(1, len(quick_queries)))
                for index, quick_query in enumerate(quick_queries):
                    with query_columns[index]:
                        st.button(quick_query, key=f"template_quick_{index}", use_container_width=True, on_click=_select_template_query, args=(quick_query,))
                if not st.session_state.get("template_query"):
                    st.session_state["template_query"] = quick_queries[0]
                query = st.text_input("Query:", key="template_query").strip()
                st.session_state["active_query"] = query
                if not query:
                    st.info("Enter a query or choose one of the entity examples to continue.")
                    return

                if st.button("Run Retrieval", type="primary", key="template_run"):
                    st.session_state["retrieval_run"] = True
                if not st.session_state.get("retrieval_run", False):
                    st.info("Run retrieval when you are ready to compare Exact Retrieval and BM25 Retrieval.")
                    return

                st.divider()
                st.subheader("3. Retrieval")
                st.write("Both methods now search the same identified candidate entities. Exact Retrieval checks direct matching; BM25 ranks query-term relevance while normalizing for candidate length.")
                k_col, k1_col, b_col = st.columns(3)
                with k_col:
                    top_k = st.slider("K for evaluation", 1, len(identified_entities), min(3, len(identified_entities)), key="template_k")
                with k1_col:
                    k1 = st.slider("BM25 k1", 0.5, 3.0, 1.5, 0.1, key="template_k1")
                with b_col:
                    b = st.slider("BM25 length normalization (b)", 0.0, 1.0, 0.75, 0.05, key="template_b")

                retrieval = _template_retrieval(identified_entities, query, k1, b)
                exact_results = retrieval["exact"]
                bm25_results = retrieval["bm25"]
                exact_names = [item["entity"] for item in exact_results]
                bm25_names = [item["entity"] for item in bm25_results]
                exact_matches = [item for item in exact_results if item["score"] > 0]
                col_exact, col_bm25 = st.columns(2)
                with col_exact:
                    st.markdown("**Exact Retrieval**")
                    st.caption("Looks for a direct query match in the identified entity names.")
                    st.dataframe(pd.DataFrame([
                        {"Rank": rank, "Entity": item["entity"], "Type": item["type"], "Score": item["score"], "Result": item["match"]}
                        for rank, item in enumerate(exact_results, 1)
                    ]), use_container_width=True, hide_index=True)
                with col_bm25:
                    st.markdown("**BM25 Retrieval**")
                    st.caption("Ranks candidates using query-term relevance/frequency and candidate-length normalization.")
                    st.dataframe(pd.DataFrame([
                        {"Rank": rank, "Entity": item["entity"], "Type": item["type"], "BM25 Score": item["score"]}
                        for rank, item in enumerate(bm25_results, 1)
                    ]), use_container_width=True, hide_index=True)

                st.divider()
                st.subheader("4. Evaluation")
                st.write("Now we compare the retrieved entities with the known relevant entities to measure retrieval quality.")
                relevant_entities = _template_entity_relevance(identified_entities, query)
                st.info(f"Known relevant entities for this query: {', '.join(sorted(relevant_entities)) or 'None identified'}")
                exact_metrics = _retrieval_metrics(exact_names, relevant_entities, top_k)
                bm25_metrics = _retrieval_metrics(bm25_names, relevant_entities, top_k)
                metric_rows = [
                    {"Metric": f"Precision@{top_k}", "Exact Retrieval": _format_metric(exact_metrics["precision"]), "BM25 Retrieval": _format_metric(bm25_metrics["precision"]), "Meaning": "Fraction of the top K results that are relevant."},
                    {"Metric": f"Recall@{top_k}", "Exact Retrieval": _format_metric(exact_metrics["recall"]), "BM25 Retrieval": _format_metric(bm25_metrics["recall"]), "Meaning": "Fraction of all known relevant entities retrieved in the top K."},
                    {"Metric": f"F1@{top_k}", "Exact Retrieval": _format_metric(exact_metrics["f1"]), "BM25 Retrieval": _format_metric(bm25_metrics["f1"]), "Meaning": "Harmonic mean of precision and recall."},
                    {"Metric": "MRR", "Exact Retrieval": _format_metric(exact_metrics["mrr"]), "BM25 Retrieval": _format_metric(bm25_metrics["mrr"]), "Meaning": "Reciprocal rank of the first relevant result."}
                ]
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
                metric_labels = [row["Metric"] for row in metric_rows]
                exact_values = [exact_metrics["precision"] or 0.0, exact_metrics["recall"] or 0.0, exact_metrics["f1"] or 0.0, exact_metrics["mrr"] or 0.0]
                bm25_values = [bm25_metrics["precision"] or 0.0, bm25_metrics["recall"] or 0.0, bm25_metrics["f1"] or 0.0, bm25_metrics["mrr"] or 0.0]
                metric_chart = go.Figure([go.Bar(name="Exact Retrieval", x=metric_labels, y=exact_values), go.Bar(name="BM25 Retrieval", x=metric_labels, y=bm25_values)])
                metric_chart.update_layout(barmode="group", title="Retrieval Quality Comparison", yaxis={"title": "Metric value", "range": [0, 1.05]}, height=360)
                st.plotly_chart(metric_chart, use_container_width=True, key="template_evaluation_chart")

                st.subheader("5. Observation / Learning")
                best_precision = "Exact Retrieval" if (exact_metrics["precision"] or 0) > (bm25_metrics["precision"] or 0) else "BM25 Retrieval" if (bm25_metrics["precision"] or 0) > (exact_metrics["precision"] or 0) else "Both methods equally"
                best_recall = "Exact Retrieval" if (exact_metrics["recall"] or 0) > (bm25_metrics["recall"] or 0) else "BM25 Retrieval" if (bm25_metrics["recall"] or 0) > (exact_metrics["recall"] or 0) else "Both methods equally"
                st.success(f"For this query, {best_precision.lower()} has the higher Precision@K result and {best_recall.lower()} has the higher Recall@K result. F1 summarizes the balance between those two measures, while MRR shows how early the first relevant entity appears. The result is specific to this candidate set and query; neither method is assumed to be better in every experiment.")
                st.markdown("**What did you learn?**")
                st.markdown("- Text was converted into structured graph entities.\n- The entities became the candidate search space.\n- Exact retrieval performs direct matching.\n- BM25 provides ranked retrieval.\n- Precision, Recall, F1 and MRR evaluate retrieval quality.")

                st.divider()
                st.subheader("6. Trial Log")
                if st.button("Record Current Trial", type="primary", key="template_record_trial"):
                    trial = {
                        "Trial #": len(st.session_state["trials"]) + 1,
                        "Query": query,
                        "Entities": len(identified_entities),
                        "K": top_k,
                        "Exact Results": ", ".join(exact_names),
                        "BM25 Results": ", ".join(bm25_names),
                        "Precision@K Exact": exact_metrics["precision"],
                        "Precision@K BM25": bm25_metrics["precision"],
                        "Recall@K Exact": exact_metrics["recall"],
                        "Recall@K BM25": bm25_metrics["recall"],
                        "F1@K Exact": exact_metrics["f1"],
                        "F1@K BM25": bm25_metrics["f1"],
                        "MRR Exact": exact_metrics["mrr"],
                        "MRR BM25": bm25_metrics["mrr"],
                        "Timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state["trials"].append(trial)
                    st.toast(f"Trial #{trial['Trial #']} recorded.")
                if st.session_state["trials"]:
                    trial_df = pd.DataFrame(st.session_state["trials"])
                    st.dataframe(trial_df, use_container_width=True, hide_index=True)
                    st.download_button("Download Trial Log", trial_df.to_csv(index=False).encode("utf-8"), "experiment_6_trials.csv", "text/csv", key="template_download_trials")
                else:
                    st.info("Record this completed experiment to add it to the trial log.")"""
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

SAMPLE_PRESETS = {
    "Tech Leaders (Sundar Pichai / Google)": "Sundar Pichai is the CEO of Google and studied at Stanford University in California.",
    "Computing History (Alan Turing / Bletchley Park)": "Alan Turing worked at Bletchley Park in the United Kingdom to break the Enigma machine for Allied forces.",
    "Modern AI (Satya Nadella / Microsoft / OpenAI)": "Satya Nadella led Microsoft to partner with OpenAI based in San Francisco to advance Artificial Intelligence.",
    "Scientific Discoveries (Marie Curie / Paris)": "Marie Curie conducted pioneering research on radioactivity at the University of Paris and won Nobel Prizes in Stockholm, Sweden.",
    "Internet Pioneers (Tim Berners-Lee / CERN)": "Tim Berners-Lee invented the World Wide Web at CERN in Geneva, Switzerland."
}

KNOWN_GAZETTEER = {
    "Sundar Pichai": "PERSON", "Alan Turing": "PERSON", "Satya Nadella": "PERSON",
    "Marie Curie": "PERSON", "Tim Berners-Lee": "PERSON", "Google": "ORGANIZATION",
    "Stanford University": "ORGANIZATION", "Microsoft": "ORGANIZATION", "OpenAI": "ORGANIZATION",
    "University of Paris": "ORGANIZATION", "CERN": "ORGANIZATION", "Bletchley Park": "ORGANIZATION",
    "California": "LOCATION", "United Kingdom": "LOCATION", "San Francisco": "LOCATION",
    "Stockholm": "LOCATION", "Sweden": "LOCATION", "Geneva": "LOCATION", "Switzerland": "LOCATION",
    "CEO": "ROLE_TITLE", "Artificial Intelligence": "CONCEPT", "World Wide Web": "CONCEPT",
    "Enigma machine": "CONCEPT", "Nobel Prizes": "AWARD_EVENT", "radioactivity": "CONCEPT"
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
        st.markdown(f"""
<h1 style="font-size:1.95rem; font-weight:800; color:#1F2937; margin:0 0 0.35rem 0; line-height:1.3;">
Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}
</h1>
<p style="font-size:1.5rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase;
color:#1F2937; margin:0 0 1.3rem 0;">
{section}
</p>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<p style="font-size:0.82rem; font-weight:600; color:#9CA3AF; margin:0 0 0.3rem 0;">
Experiment {EXPERIMENT_CONFIG['exp_number']}: {EXPERIMENT_CONFIG['title']}
</p>
<h1 style="font-size:1.5rem; font-weight:800; letter-spacing:0.06em; text-transform:uppercase;
color:#1F2937; margin:0 0 1.3rem 0; line-height:1.3;">
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
.kg-eyebrow.kg-eyebrow-ink { color:#4B5563; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.15em; }

.kg-section-title { font-size:1.5rem; font-weight:700; color:#1F2937; margin:0 0 0.6rem 0; }
.kg-section-body { font-size:0.95rem; line-height:1.7; color:#4B5563; width:100%; }
.kg-section-body b { color:#1F2937; }
.kg-example { margin-top:0.9rem; padding:0.8rem 1rem; background:#FEFDFB; border:1px solid #E7E2D3;
  border-left:3px solid #B3261E; border-radius:6px; font-size:0.87rem; color:#4B5563; line-height:1.6; width:100%; }
.kg-example.kg-example-teal { border-left-color:#0E7C7B; }
.kg-example code { background:rgba(31,41,55,0.06); padding:1px 5px; border-radius:4px; font-size:0.85em; }

.kg-divider { height:1px; background:#E7E2D3; margin:2.1rem 0; border:none; }

/* --- storyboard shell -------------------------------------------------- */
.kg-storyboard { display:flex; gap:0.7rem; margin-top:1.3rem; align-items:stretch; flex-wrap:wrap; }
.kg-story-step { flex:1 1 230px; min-width:220px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px;
  padding:0.95rem 1rem 1.1rem; display:flex; flex-direction:column; box-shadow:0 1px 2px rgba(31,41,55,0.04); }
.kg-story-step.kg-story-step-win { border-color:#BFE3E1; background:#F6FBFA; }
.kg-story-step.kg-story-step-lose { border-color:#F1D6D2; background:#FEFAF9; }
.kg-story-eyebrow { font-size:0.62rem; font-weight:800; letter-spacing:0.1em; color:#9CA3AF; margin-bottom:0.25rem; }
.kg-story-title { font-size:0.85rem; font-weight:700; color:#1F2937; margin-bottom:0.75rem; min-height:2.3em; }
.kg-story-stage { flex:1; display:flex; align-items:center; justify-content:center; min-height:118px; }
.kg-story-connector { display:flex; align-items:center; justify-content:center; flex:0 0 22px; }
.kg-story-connector svg { opacity:0; animation:kgPop 0.4s ease forwards; animation-delay:0.9s; }

/* --- step 1: plain keyword chips (no meaning attached) ------------------ */
.kg-chip-row { display:flex; flex-wrap:wrap; gap:0.32rem; justify-content:center; align-content:center; }
.kg-chip { padding:0.28rem 0.6rem; background:#F3F4F6; border:1px solid #D1D5DB; border-radius:999px;
  font-size:0.72rem; font-weight:600; color:#4B5563; opacity:0; animation:kgPop 0.4s cubic-bezier(0.34,1.56,0.64,1) forwards; }

/* --- step 1 (method two): entities recognized inline, NER-style -------- */
.kg-ner-sentence { display:flex; flex-wrap:wrap; align-items:flex-end; justify-content:center;
  column-gap:0.4rem; row-gap:0.7rem; font-size:0.86rem; color:#1F2937; text-align:center; }
.kg-ner-word { padding-bottom:0.3rem; }
.kg-ent { display:flex; flex-direction:column; align-items:center; gap:0.22rem; opacity:0; animation:kgRise 0.4s ease forwards; }
.kg-ent-value { padding:0.15rem 0.4rem; border-radius:5px; font-weight:700; }
.kg-ent-tag { font-size:0.5rem; font-weight:800; letter-spacing:0.05em; padding:1px 5px; border-radius:4px;
  color:#fff; white-space:nowrap; opacity:0; animation:kgPop 0.35s ease forwards; }
.kg-ent-per .kg-ent-value { background:rgba(139,92,246,0.16); color:#6D28D9; } .kg-ent-per .kg-ent-tag { background:#8B5CF6; }
.kg-ent-org .kg-ent-value { background:rgba(37,99,235,0.16); color:#1D4ED8; } .kg-ent-org .kg-ent-tag { background:#2563EB; }
.kg-ent-loc .kg-ent-value { background:rgba(16,185,129,0.16); color:#047857; } .kg-ent-loc .kg-ent-tag { background:#10B981; }

/* --- step 2 (method one): raw tally cards ------------------------------- */
.kg-doc-mini { width:100%; background:#FFFFFF; border:1px solid #E5E7EB; border-radius:9px;
  padding:0.5rem 0.65rem; margin-bottom:0.45rem; opacity:0; animation:kgRise 0.4s ease forwards; }
.kg-doc-mini-title { font-size:0.68rem; font-weight:700; color:#1F2937; margin-bottom:0.3rem; }
.kg-doc-mini.kg-doc-spam { border-color:#F1D6D2; background:#FFFAF9; }
.kg-tally { display:inline-block; font-size:0.64rem; font-weight:600; color:#4B5563; background:#F3F4F6;
  border-radius:5px; padding:1px 6px; margin:1px 3px 1px 0; }
.kg-tally-hot { color:#B3261E; background:rgba(179,38,30,0.1); animation:kgPulse 1.6s ease-in-out infinite; }

/* --- step 2 (method two): mini knowledge-graph triple ------------------- */
.kg-graph-node { opacity:0; animation:kgPop 0.45s cubic-bezier(0.34,1.56,0.64,1) forwards; }
.kg-graph-label { font-size:7.5px; font-weight:700; fill:#1F2937; opacity:0; animation:kgFadeIn 0.3s ease forwards; }
.kg-graph-edge-label { font-size:6.5px; font-weight:600; fill:#6B7280; opacity:0; animation:kgFadeIn 0.3s ease forwards; }
.kg-graph-edge { stroke-dasharray:140; stroke-dashoffset:140; animation:kgDraw 0.55s ease forwards; }

/* --- step 3 (both methods): ranking bar race ---------------------------- */
.kg-barlist { width:100%; display:flex; flex-direction:column; gap:0.6rem; }
.kg-bar-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.2rem; }
.kg-bar-doc { font-size:0.68rem; font-weight:600; color:#4B5563; }
.kg-bar-rank { font-size:0.68rem; font-weight:800; }
.kg-bar-track { background:#EFEEE8; border-radius:6px; height:12px; overflow:hidden; }
.kg-bar { height:100%; border-radius:6px; width:0; animation:kgGrow 0.9s cubic-bezier(.22,1,.36,1) forwards; }
.kg-bar-good { background:#0E7C7B; } .kg-bar-spam { background:#B3261E; }
.kg-bar-tag { display:inline-block; margin-top:0.28rem; font-size:0.6rem; font-weight:700; padding:1px 6px;
  border-radius:4px; opacity:0; animation:kgPop 0.4s ease forwards; }
.kg-bar-tag-bad { color:#B3261E; background:rgba(179,38,30,0.09); }
.kg-bar-tag-good { color:#0E7C7B; background:rgba(14,124,123,0.09); }

/* --- section 3: outcome recap ------------------------------------------- */
.kg-obj-list { display:flex; flex-direction:column; gap:0.5rem; margin-top:0.7rem; }
.kg-obj-item { display:flex; align-items:flex-start; gap:0.65rem; padding:0.65rem 0.9rem;
  background:#FEFDFB; border:1px solid #E7E2D3; border-radius:8px;
  font-size:0.89rem; color:#4B5563; line-height:1.5; opacity:0; animation:kgRise 0.45s ease forwards; }
.kg-obj-mark { flex-shrink:0; margin-top:2px; }

/* --- scroll-triggered reveal: paused until scrolled into view ----------- */
:is(.kg-storyboard, .kg-obj-list),
:is(.kg-storyboard, .kg-obj-list) * { animation-play-state: paused; }
:is(.kg-storyboard, .kg-obj-list).kg-inview,
:is(.kg-storyboard, .kg-obj-list).kg-inview * { animation-play-state: running; }
</style>

<div class="kg-purpose-scope">

<h1 style="font-size:1.4rem; font-weight:800; color:#1E3A8A; margin:0 0 0.5rem 0; line-height:1.3;">
  Two Ways to Search the Same Document
</h1>
<p style="font-size:1rem; color:#4B5563; line-height:1.65; width:100%; margin:0;">
  Keyword search counts word matches; entity-aware retrieval understands who or what those words refer to.
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="kg-divider"/>', unsafe_allow_html=True)

    arrow_svg = (
        '<div class="kg-story-connector"><svg width="20" height="20" viewBox="0 0 20 20">'
        '<path d="M5 3 L14 10 L5 17" fill="none" stroke="#9CA3AF" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    )

    # --- Section 1: Plain Keyword Search --------------------------------------------
    st.markdown("""
<div class="kg-purpose-scope">
<p class="kg-eyebrow">Method One</p>
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
          <div class="kg-bar-head"><span class="kg-bar-doc">Article: "Voss joins Nimbus Labs"</span><span class="kg-bar-rank" style="color:#4B5563;">#2</span></div>
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

    st.markdown('<hr class="kg-divider"/>', unsafe_allow_html=True)

    # --- Section 2: Entity-Aware, Probabilistic Retrieval ---------------------------
    st.markdown("""
<div class="kg-purpose-scope">
<p class="kg-eyebrow kg-eyebrow-teal">METHOD TWO</p>
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
<svg viewBox="0 0 220 140" width="100%" height="120">
<line x1="42" y1="34" x2="168" y2="56" stroke="#9CA3AF" stroke-width="1.6" class="kg-graph-edge" style="animation-delay:0.75s;"/>
<line x1="168" y1="56" x2="88" y2="118" stroke="#9CA3AF" stroke-width="1.6" class="kg-graph-edge" style="animation-delay:0.95s;"/>
<text x="90" y="38" text-anchor="middle" class="kg-graph-edge-label" style="animation-delay:1.1s;">affiliated_with</text>
<text x="150" y="94" text-anchor="middle" class="kg-graph-edge-label" style="animation-delay:1.3s;">located_in</text>
<g class="kg-graph-node" style="animation-delay:0.1s;">
<circle cx="42" cy="34" r="10" fill="#8B5CF6"/>
<text x="42" y="16" text-anchor="middle" class="kg-graph-label">Elena Voss</text>
</g>
<g class="kg-graph-node" style="animation-delay:0.3s;">
<circle cx="168" cy="56" r="10" fill="#2563EB"/>
<text x="168" y="76" text-anchor="middle" class="kg-graph-label">Nimbus Labs</text>
</g>
<g class="kg-graph-node" style="animation-delay:0.5s;">
<circle cx="88" cy="118" r="10" fill="#10B981"/>
<text x="88" y="135" text-anchor="middle" class="kg-graph-label">Toronto</text>
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
          <div class="kg-bar-head"><span class="kg-bar-doc">Blog: keyword-stuffed page</span><span class="kg-bar-rank" style="color:#4B5563;">#2</span></div>
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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


def render_theory_section():
    """Renders Section 2: In-Depth Theoretical Foundations, from first principles to evaluation."""
    st.markdown("""
<style>
@keyframes kgThRise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes kgThPop { 0% { opacity:0; transform:scale(0.4); } 70% { opacity:1; transform:scale(1.06); } 100% { opacity:1; transform:scale(1); } }
@keyframes kgThDraw { to { stroke-dashoffset:0; } }
@keyframes kgThFade { from { opacity:0; } to { opacity:1; } }

.kg-th-intro { font-size:1rem; line-height:1.8; color:#4B5563; width:100%; margin-bottom:0.3rem; }

.kg-th-head { display:flex; align-items:center; gap:0.65rem; margin:0.3rem 0 0.9rem 0; }
.kg-th-num { flex-shrink:0; width:1.7rem; height:1.7rem; border-radius:50%; background:#1F2937; color:#fff;
  font-weight:700; font-size:0.78rem; display:flex; align-items:center; justify-content:center; }
.kg-th-title { font-size:1.12rem; font-weight:700; color:#1F2937; line-height:1.35; letter-spacing:0.01em; }

.kg-th-body { font-size:1rem; line-height:1.8; color:#4B5563; width:100%; }
.kg-th-body b { color:#1F2937; }
.kg-th-body code { background:rgba(31,41,55,0.07); padding:2px 7px; border-radius:5px; font-size:0.9em; color:#1F2937; }

.kg-th-quote { margin:1.1rem 0; padding:1.1rem 1.4rem; background:#F6FBFA; border-left:4px solid #0E7C7B;
  border-radius:8px; font-style:italic; color:#1F2937; font-size:1.02rem; line-height:1.75; width:100%; }

.kg-formula-label { font-size:0.7rem; font-weight:800; letter-spacing:0.13em; text-transform:uppercase;
  color:#B3261E; margin:0 0 0.6rem 0; }
.kg-formula-label.teal { color:#0E7C7B; }

.kg-term-grid { display:flex; flex-wrap:wrap; gap:0.75rem; margin:1.1rem 0 1.4rem 0; }
.kg-term-card { flex:1 1 210px; min-width:200px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:11px;
  border-left:4px solid #0E7C7B; padding:0.95rem 1.1rem; opacity:0; animation:kgThRise 0.4s ease forwards; }
.kg-term-symbol { font-size:1.05rem; font-weight:800; color:#0E7C7B; margin-bottom:0.35rem; font-family:"Courier New", monospace; }
.kg-term-label { font-size:0.88rem; font-weight:700; color:#1F2937; margin-bottom:0.3rem; }
.kg-term-desc { font-size:0.88rem; color:#4B5563; line-height:1.55; }

.kg-compare-grid { display:flex; flex-wrap:wrap; gap:1rem; margin:1.1rem 0 1.4rem 0; }
.kg-compare-card { flex:1 1 300px; min-width:270px; background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px;
  padding:1.2rem 1.3rem; opacity:0; animation:kgThRise 0.45s ease forwards; }
.kg-compare-icon { margin-bottom:0.7rem; }
.kg-compare-title { font-size:1.05rem; font-weight:800; color:#1F2937; margin-bottom:0.5rem; }
.kg-compare-desc { font-size:0.92rem; color:#4B5563; line-height:1.65; }

.kg-metric-card { background:#FEFDFB; border:1px solid #E7E2D3; border-left:4px solid #B3261E; border-radius:11px;
  padding:1rem 1.2rem; margin-bottom:0.8rem; opacity:0; animation:kgThRise 0.4s ease forwards; }
.kg-metric-name { font-size:1rem; font-weight:800; color:#1F2937; margin-bottom:0.5rem; }
.kg-metric-desc { font-size:0.9rem; color:#4B5563; line-height:1.6; margin-top:0.55rem; }

.kg-illus-card { background:#FEFDFB; border:1px solid #E7E2D3; border-radius:14px; padding:1.2rem 1.3rem 1rem;
  box-shadow:0 1px 2px rgba(31,41,55,0.05); margin:1rem 0 1.4rem 0; }
.kg-illus-caption { font-size:0.85rem; color:#6B7280; margin-top:0.5rem; line-height:1.5; }

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
        '<div class="kg-th-head"><span class="kg-th-num">0</span>'
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
<text x="165" y="38" text-anchor="middle" class="kg-glabel" style="animation-delay:1.1s; font-size:11px; fill:#0E7C7B; font-weight:600;">affiliated_with</text>
<text x="290" y="105" text-anchor="middle" class="kg-glabel" style="animation-delay:1.25s; font-size:11px; fill:#0E7C7B; font-weight:600;">located_in</text>
<text x="30" y="115" text-anchor="middle" class="kg-glabel" style="animation-delay:1.4s; font-size:11px; fill:#0E7C7B; font-weight:600;">pioneered</text>
<g class="kg-gnode" style="animation-delay:0.1s;">
<circle cx="60" cy="60" r="16" fill="#8B5CF6" stroke="#1F2937" stroke-width="2"/>
<text x="60" y="94" text-anchor="middle" style="font-size:12px; font-weight:700; fill:#1F2937;">Node A</text>
</g>
<g class="kg-gpill" style="animation-delay:1.6s;">
<rect x="82" y="60" width="76" height="20" rx="10" fill="#1F2937"/>
<text x="120" y="74" text-anchor="middle" style="font-size:10px; font-weight:700; fill:#FFFFFF;">DEGREE = 2</text>
</g>
<g class="kg-gnode" style="animation-delay:0.3s;">
<circle cx="260" cy="45" r="16" fill="#2563EB" stroke="#1F2937" stroke-width="2"/>
<text x="260" y="20" text-anchor="middle" style="font-size:12px; font-weight:700; fill:#1F2937;">Node B</text>
</g>
<g class="kg-gnode" style="animation-delay:0.5s;">
<circle cx="275" cy="160" r="16" fill="#10B981" stroke="#1F2937" stroke-width="2"/>
<text x="275" y="188" text-anchor="middle" style="font-size:12px; font-weight:700; fill:#1F2937;">Node C</text>
</g>
<g class="kg-gnode" style="animation-delay:0.7s;">
<circle cx="60" cy="160" r="16" fill="#F59E0B" stroke="#1F2937" stroke-width="2"/>
<text x="60" y="188" text-anchor="middle" style="font-size:12px; font-weight:700; fill:#1F2937;">Node D</text>
</g>
<g class="kg-gpill" style="animation-delay:1.8s;">
<rect x="95" y="118" width="150" height="22" rx="11" fill="#B3261E"/>
<text x="170" y="133" text-anchor="middle" style="font-size:10px; font-weight:700; fill:#FFFFFF;">TRIPLE: (A, affiliated_with, B)</text>
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


def _build_entity_candidates(corpus_docs: list) -> list:
    """Create one retrieval candidate per unique graph entity.

    Each candidate is represented by its entity name, type, and the text of the
    source document(s) in which the entity occurs. This lets BM25 rank entities
    rather than ranking whole documents.
    """
    candidates = {}
    for doc in corpus_docs:
        for ent in doc.get("entities", []):
            name = ent["name"]
            key = name.lower()
            if key not in candidates:
                candidates[key] = {
                    "entity": name,
                    "type": ent.get("type", "CONCEPT"),
                    "contexts": [],
                    "source_docs": []
                }
            candidates[key]["contexts"].append(doc.get("text", ""))
            candidates[key]["source_docs"].append(doc.get("doc_id", ""))

    output = []
    for item in candidates.values():
        context = " ".join(dict.fromkeys(item["contexts"]))
        representation = f"{item['entity']} {item['type']} {context}"
        output.append({
            "entity": item["entity"],
            "type": item["type"],
            "context": context,
            "source_docs": list(dict.fromkeys(item["source_docs"])),
            "text": representation
        })
    return output


def _entity_ground_truth(corpus_docs: list, query_str: str) -> set:
    """Derive benchmark relevance from the corpus' annotated relevant_to tags.

    For custom text, no relevance labels exist, so an exact entity mentioned in
    the query is used as the available ground truth when possible.
    """
    q = query_str.strip().lower()
    q_tokens = set(tokenize(q))
    relevant_entities = set()

    for doc in corpus_docs:
        doc_relevant = False
        for tag in doc.get("relevant_to", []):
            tag_lower = tag.lower()
            tag_tokens = set(tokenize(tag_lower))
            if tag_lower in q or q in tag_lower or (q_tokens and q_tokens.issubset(tag_tokens)):
                doc_relevant = True
                break
            if len(q_tokens & tag_tokens) >= max(1, min(2, len(q_tokens))):
                doc_relevant = True
                break
        if doc_relevant:
            for ent in doc.get("entities", []):
                relevant_entities.add(ent["name"])

    # If the query directly names an entity, that named entity is relevant.
    all_entities = [ent["name"] for doc in corpus_docs for ent in doc.get("entities", [])]
    for name in all_entities:
        if name.lower() in q or q in name.lower():
            relevant_entities.add(name)

    return relevant_entities


def _retrieval_metrics(ranked_names: list, relevant_names: set, k: int) -> dict:
    """Calculate Precision@K, Recall@K, F1@K and MRR."""
    if not relevant_names:
        return {"precision": None, "recall": None, "f1": None, "mrr": None}

    top_k = ranked_names[:k]
    hits = sum(1 for name in top_k if name in relevant_names)
    precision = hits / max(1, k)
    recall = hits / len(relevant_names)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    reciprocal_rank = 0.0
    for rank, name in enumerate(ranked_names, start=1):
        if name in relevant_names:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(reciprocal_rank, 4)
    }


def _retrieve_graph_entities(candidates: list, query_str: str, k1: float, b: float) -> dict:
    """Run exact entity retrieval and BM25 entity retrieval."""
    q_norm = " ".join(tokenize(query_str))

    exact_ranked = []
    for item in candidates:
        entity_norm = " ".join(tokenize(item["entity"]))
        exact_match = bool(q_norm) and (q_norm == entity_norm or q_norm in entity_norm or entity_norm in q_norm)
        exact_ranked.append({
            **item,
            "score": 1.0 if exact_match else 0.0,
            "match": "Exact match" if exact_match else "No exact match"
        })
    exact_ranked.sort(key=lambda x: (-x["score"], x["entity"].lower()))

    q_tokens = tokenize(query_str)
    doc_token_lists = [tokenize(item["text"]) for item in candidates]
    avg_len = sum(len(tokens) for tokens in doc_token_lists) / max(1, len(doc_token_lists))

    df_counts = Counter()
    for tokens in doc_token_lists:
        for token in set(tokens):
            df_counts[token] += 1
    N = len(candidates)
    idf = {
        token: math.log(1.0 + (N - df + 0.5) / (df + 0.5))
        for token, df in df_counts.items()
    }

    bm25_ranked = []
    for item, tokens in zip(candidates, doc_token_lists):
        score = compute_bm25_score(
            q_tokens, tokens, len(tokens), avg_len, idf, k1, b
        )
        bm25_ranked.append({**item, "score": score})
    bm25_ranked.sort(key=lambda x: (-x["score"], x["entity"].lower()))

    return {"exact": exact_ranked, "bm25": bm25_ranked}


def _format_metric(value):
    return "N/A" if value is None else f"{value:.3f}"


def render_legacy_simulation_section():
    """Interactive Experiment 6 simulation: Identifying -> Querying -> Retrieval -> Evaluation."""
    st.markdown("### Interactive Simulation Sandbox")
    st.info(
        "Identify graph entities from text, query the candidate entities, compare Exact Retrieval with BM25 ranking, "
        "and evaluate the ranked results using Precision@K, Recall@K, F1@K, and MRR."
    )

    with st.expander("📋 Step-by-Step Procedure", expanded=True):
        st.markdown("""
1. **Identifying:** Select a benchmark corpus or enter custom text. Identify the Persons, Organizations, Locations, and Concepts present in the text.
2. **Querying:** Enter a query describing the graph entity you want to retrieve.
3. **Retrieval:** Compare **Exact Retrieval** with **Okapi BM25 Retrieval** over the identified entity candidates.
4. **Evaluation:** Select a value of K and examine Precision@K, Recall@K, F1@K, and MRR using the available benchmark relevance labels.
5. **Record the Trial:** Save the query, rankings, scores, and evaluation results in the experimental log.
        """)

    # ------------------------------------------------------------------
    # 1. IDENTIFYING
    # ------------------------------------------------------------------
    st.markdown("## 1. Identifying")
    corpus_names = list(DATA_CORPORA.keys()) + ["Add custom text"]
    corpus_choice = st.selectbox("Select input text / corpus:", corpus_names, key="exp6_corpus")

    if corpus_choice == "Add custom text":
        raw_text = st.text_area(
            "Enter text (separate multiple documents with a line containing ---):",
            height=180,
            placeholder=(
                "Example: Sundar Pichai leads Google from California.\n"
                "Google develops artificial intelligence systems."
            ),
            key="exp6_custom_text"
        )
        if not raw_text.strip():
            st.info("Enter text above to identify graph entities.")
            return
        selected_corpus = build_custom_corpus_from_text(raw_text)
    else:
        selected_corpus = DATA_CORPORA[corpus_choice]["documents"]
        st.caption(DATA_CORPORA[corpus_choice]["description"])

    G, entity_meta = build_knowledge_graph(selected_corpus)
    candidates = _build_entity_candidates(selected_corpus)

    col_graph, col_entities = st.columns([1.55, 1.45])
    with col_graph:
        st.markdown("**Candidate Knowledge Graph**")
        st.plotly_chart(generate_plotly_knowledge_graph(G), use_container_width=True)
    with col_entities:
        st.markdown("**Identified Graph Entities**")
        entity_rows = [
            {
                "Entity": item["entity"],
                "Type": item["type"],
                "Source": ", ".join(item["source_docs"])
            }
            for item in candidates
        ]
        st.dataframe(
            pd.DataFrame(entity_rows),
            use_container_width=True,
            hide_index=True,
            height=360
        )

    st.caption(
        f"{len(candidates)} unique candidate entities identified. Each candidate is represented by its name, type, and source context for retrieval."
    )

    # ------------------------------------------------------------------
    # 2. QUERYING
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("## 2. Querying")

    benchmark_queries = sorted({tag for doc in selected_corpus for tag in doc.get("relevant_to", [])})
    query_options = benchmark_queries + ["Custom Query"]
    query_choice = st.selectbox("Select a benchmark query:", query_options, key="exp6_query_choice") if query_options else "Custom Query"

    if query_choice == "Custom Query":
        selected_query = st.text_input(
            "Enter your query:",
            placeholder="e.g. deep learning or Lakeside University",
            key="exp6_custom_query"
        )
    else:
        selected_query = query_choice
        st.write(f"**Active Query:** `{selected_query}`")

    if not selected_query or not selected_query.strip():
        st.info("Enter a query to continue to retrieval.")
        return

    k_col, k1_col, b_col = st.columns(3)
    with k_col:
        top_k = st.slider("Evaluation K", min_value=1, max_value=min(5, len(candidates)), value=min(3, len(candidates)), step=1, key="exp6_k")
    with k1_col:
        k1 = st.slider("BM25 k1", min_value=0.5, max_value=3.0, value=1.5, step=0.1, key="exp6_k1")
    with b_col:
        b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05, key="exp6_b")

    # ------------------------------------------------------------------
    # 3. RETRIEVAL
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("## 3. Retrieval")
    st.caption("The retrieval unit is a graph entity, not a whole document. BM25 uses each entity's name, type, and source context as its retrieval representation.")

    retrieval = _retrieve_graph_entities(candidates, selected_query, k1, b)
    relevant_entities = _entity_ground_truth(selected_corpus, selected_query)

    col_exact, col_bm25 = st.columns(2)
    with col_exact:
        st.markdown("### Exact Retrieval")
        exact_rows = []
        for rank, item in enumerate(retrieval["exact"], start=1):
            exact_rows.append({
                "Rank": rank,
                "Entity": item["entity"],
                "Type": item["type"],
                "Score": item["score"],
                "Match": item["match"]
            })
        st.dataframe(pd.DataFrame(exact_rows), use_container_width=True, hide_index=True)

    with col_bm25:
        st.markdown("### BM25 Retrieval")
        bm25_rows = []
        for rank, item in enumerate(retrieval["bm25"], start=1):
            bm25_rows.append({
                "Rank": rank,
                "Entity": item["entity"],
                "Type": item["type"],
                "BM25 Score": item["score"]
            })
        st.dataframe(pd.DataFrame(bm25_rows), use_container_width=True, hide_index=True)

    # Score visualization
    plot_n = min(8, len(candidates))
    plot_exact = retrieval["exact"][:plot_n]
    plot_bm25 = retrieval["bm25"][:plot_n]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Exact", x=[x["entity"] for x in plot_exact], y=[x["score"] for x in plot_exact]
    ))
    fig.add_trace(go.Bar(
        name="BM25", x=[x["entity"] for x in plot_bm25], y=[x["score"] for x in plot_bm25]
    ))
    fig.update_layout(
        barmode="group",
        title="Entity Retrieval Scores",
        xaxis_title="Candidate Entity",
        yaxis_title="Score",
        height=380,
        margin=dict(l=20, r=20, t=60, b=120)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. EVALUATION
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("## 4. Evaluation")

    exact_names = [item["entity"] for item in retrieval["exact"]]
    bm25_names = [item["entity"] for item in retrieval["bm25"]]
    exact_metrics = _retrieval_metrics(exact_names, relevant_entities, top_k)
    bm25_metrics = _retrieval_metrics(bm25_names, relevant_entities, top_k)

    if relevant_entities:
        st.caption(
            "Benchmark ground truth is derived from the corpus' annotated relevance tags. "
            f"Relevant entities for this query: {', '.join(sorted(relevant_entities))}"
        )
    else:
        st.warning(
            "No benchmark relevance labels are available for this custom query, so the evaluation metrics are shown as N/A."
        )

    metric_names = [f"Precision@{top_k}", f"Recall@{top_k}", f"F1@{top_k}", "MRR"]
    exact_values = [exact_metrics["precision"], exact_metrics["recall"], exact_metrics["f1"], exact_metrics["mrr"]]
    bm25_values = [bm25_metrics["precision"], bm25_metrics["recall"], bm25_metrics["f1"], bm25_metrics["mrr"]]

    mcols = st.columns(4)
    for col, name, exact_value, bm25_value in zip(mcols, metric_names, exact_values, bm25_values):
        with col:
            st.metric(f"{name} — Exact", _format_metric(exact_value))
            st.metric(f"{name} — BM25", _format_metric(bm25_value))

    fig_eval = go.Figure(data=[
        go.Bar(name="Exact", x=metric_names, y=[v if v is not None else 0 for v in exact_values]),
        go.Bar(name="BM25", x=metric_names, y=[v if v is not None else 0 for v in bm25_values])
    ])
    fig_eval.update_layout(
        barmode="group",
        title=f"Retrieval Evaluation Comparison (@K={top_k})",
        yaxis=dict(title="Score", range=[0, 1.05]),
        height=360
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. EXPERIMENTAL DATA LOG
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("## 5. Experimental Data Log Book")
    if st.button("Record Current Trial", type="primary", use_container_width=True, key="exp6_record_trial"):
        trial_record = {
            "Trial #": len(st.session_state["trials"]) + 1,
            "Query": selected_query,
            "# Entities": len(candidates),
            "Top Exact": exact_names[0] if exact_names else "None",
            "Top BM25": bm25_names[0] if bm25_names else "None",
            f"Exact P@{top_k}": exact_metrics["precision"],
            f"BM25 P@{top_k}": bm25_metrics["precision"],
            f"Exact R@{top_k}": exact_metrics["recall"],
            f"BM25 R@{top_k}": bm25_metrics["recall"],
            f"Exact F1@{top_k}": exact_metrics["f1"],
            f"BM25 F1@{top_k}": bm25_metrics["f1"],
            "Exact MRR": exact_metrics["mrr"],
            "BM25 MRR": bm25_metrics["mrr"],
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state["trials"].append(trial_record)
        st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

    if st.session_state["trials"]:
        df_trials = pd.DataFrame(st.session_state["trials"])
        st.dataframe(df_trials, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Trials as CSV",
            data=df_trials.to_csv(index=False).encode("utf-8"),
            file_name="kgirs_experiment6_trials.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No trials recorded yet. Run a query and click 'Record Current Trial'.")

def extract_template_entities(text: str) -> list:
    """Extract preset-style graph entities with spans and context windows."""
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
                "start": match.start(),
                "end": match.end(),
                "context": "..." + text[max(0, match.start() - 25):min(len(text), match.end() + 25)].strip() + "..."
            })
    entities.sort(key=lambda item: item["start"])
    return entities


def _template_entity_relevance(entities: list, query: str) -> set:
    """Create transparent entity relevance labels for the final evaluation section."""
    query_tokens = set(tokenize(query))
    relevant = set()
    for item in entities:
        entity_tokens = set(tokenize(item["entity"]))
        if query.lower().strip() in item["entity"].lower() or query_tokens & entity_tokens:
            relevant.add(item["entity"])
    return relevant


def _template_retrieval(entities: list, query: str, k1: float, b: float) -> dict:
    """Run exact retrieval beside BM25 and expose a normalized relevance score."""
    candidates = [
        {
            "entity": item["entity"],
            "type": item["type"],
            "context": item["context"],
            "source_docs": [],
            "text": f"{item['entity']} {item['type']} {item['context']}"
        }
        for item in entities
    ]
    retrieval = _retrieve_graph_entities(candidates, query, k1, b)
    max_score = max((item["score"] for item in retrieval["bm25"]), default=0.0)
    for item in retrieval["bm25"]:
        item["relevance_score"] = round(item["score"] / max_score, 4) if max_score else 0.0
        item["confidence"] = (
            "High Confidence" if item["relevance_score"] >= 0.75 else
            "Moderate Relevance" if item["relevance_score"] >= 0.40 else
            "Marginal Association" if item["relevance_score"] >= 0.15 else
            "Non-Relevant"
        )
    return retrieval


def _template_quick_queries(entities: list, limit: int = 4) -> list:
    """Return query suggestions that are all present in the active entity set."""
    return list(dict.fromkeys(item["entity"] for item in entities))[:limit]


def _select_template_query(query: str) -> None:
    """Update the query widget before Streamlit reruns the page."""
    st.session_state["template_query"] = query
    st.session_state["active_query"] = query


def _select_template_preset() -> None:
    """Load the selected preset into the editable text input before rerun."""
    preset_name = st.session_state["template_preset"]
    st.session_state["template_preset_text"] = SAMPLE_PRESETS[preset_name]


def render_legacy_simulation_section():
    """Renders the template-style stimulation flow with BM25 probabilistic ranking."""
    st.header("Simulation: Identify, Retrieve, Compare")
    st.info("Laboratory sequence: Input -> Run identification -> Observe candidates -> Run retrieval -> Compare -> Understand")

    st.subheader("Step 1: Text Input")
    input_mode = st.radio(
        "Choose Text Input Method:",
        options=["Select from Benchmark Presets", "Enter Custom Unstructured Text"],
        horizontal=True,
        key="template_input_mode"
    )
    if input_mode == "Select from Benchmark Presets":
        preset_names = list(SAMPLE_PRESETS)
        if "template_preset_text" not in st.session_state:
            st.session_state["template_preset_text"] = SAMPLE_PRESETS[preset_names[0]]
        selected_preset = st.selectbox(
            "Benchmark Domain Presets:",
            preset_names,
            index=0,
            key="template_preset",
            on_change=_select_template_preset,
        )
        active_text = st.text_area(
            "Selected Text Preview / Editable Input:",
            height=90,
            key="template_preset_text"
        )
    else:
        active_text = st.text_area(
            "Enter Custom Text (containing Persons, Organizations, Locations, Roles, etc.):",
            value=st.session_state.get("custom_text_val", SAMPLE_PRESETS["Tech Leaders (Sundar Pichai / Google)"]),
            height=90,
            key="template_custom_text"
        )
        st.session_state["custom_text_val"] = active_text

    text_signature = active_text.strip()
    if st.session_state.get("last_extracted_text") != text_signature:
        st.session_state["identified_entities"] = []
        st.session_state["retrieval_run"] = False
        st.session_state["active_query"] = ""
        st.session_state["template_query"] = ""

    col_btn, col_info = st.columns([1.5, 3.5])
    with col_btn:
        if st.button("Identify Graph Entities", type="primary", use_container_width=True, key="template_identify"):
            st.session_state["identified_entities"] = extract_template_entities(active_text)
            st.session_state["last_extracted_text"] = text_signature
            st.session_state["retrieval_run"] = False
            st.toast(f"Successfully identified {len(st.session_state['identified_entities'])} candidate graph entities!")
    with col_info:
        if st.session_state.get("identified_entities"):
            entities = st.session_state["identified_entities"]
            st.success(f"**Entity Identification Active**: {len(entities)} candidate entities detected across {len({e['type'] for e in entities})} distinct types.")
        else:
            st.caption("Click **Identify Graph Entities** to create the candidate entity set.")

    if "identified_entities" not in st.session_state:
        st.session_state["identified_entities"] = extract_template_entities(active_text)
        st.session_state["last_extracted_text"] = text_signature
    identified_entities = st.session_state["identified_entities"]
    if not identified_entities:
        st.warning("No named entities identified in the text. Please ensure the text includes capitalized proper nouns, institutions, locations, or roles.")
        return

    st.divider()
    st.subheader("Step 3: Show Entity Types and Candidate Graph")
    st.caption("Each extracted mention is a candidate node. The graph displays nodes only; no relationships are inferred.")
    entity_df = pd.DataFrame([
        {"Entity Mention": e["entity"], "Entity Type": e["type"], "Span [Start, End]": f"[{e['start']}, {e['end']}]", "Mention Context Window": e["context"]}
        for e in identified_entities
    ])
    st.dataframe(entity_df, use_container_width=True, hide_index=True)
    graph_docs = [{"doc_id": "TEXT-1", "title": "Input Text", "text": active_text, "entities": [{"name": e["entity"], "type": e["type"]} for e in identified_entities], "triples": []}]
    st.plotly_chart(generate_plotly_knowledge_graph(build_knowledge_graph(graph_docs)[0]), use_container_width=True, key="template_candidate_graph")

    st.divider()
    st.subheader("Step 4: Enter Query and Run Retrieval")
    st.write("Enter a query, then run both retrieval methods over the same candidate set.")
    col_query, col_quick = st.columns([3, 2])
    with col_query:
        default_query = st.session_state.get("active_query") or identified_entities[0]["entity"]
        if not st.session_state.get("template_query"):
            st.session_state["template_query"] = default_query
        query_input = st.text_input("Search Query:", placeholder="Select a query suggestion or enter your own query", key="template_query")
        st.session_state["active_query"] = query_input
    with col_quick:
        st.caption("Quick-Test Query Suggestions:")
        quick_queries = _template_quick_queries(identified_entities)
        quick_cols = st.columns(len(quick_queries))
        for index, quick_query in enumerate(quick_queries):
            with quick_cols[index]:
                st.button(
                    quick_query,
                    key=f"template_quick_{index}",
                    use_container_width=True,
                    on_click=_select_template_query,
                    args=(quick_query,)
                )
    query = st.session_state.get("template_query", "").strip()
    if st.button("Run Retrieval Experiment", type="primary", use_container_width=True, key="template_run"):
        st.session_state["retrieval_run"] = True
    if not st.session_state.get("retrieval_run", False):
        st.info("Retrieval results will appear here after you click **Run Retrieval Experiment**.")
        return

    k_col, k1_col, b_col = st.columns(3)
    with k_col:
        top_k = st.slider("Evaluation K", 1, len(identified_entities), min(3, len(identified_entities)), key="template_k")
    with k1_col:
        k1 = st.slider("BM25 k1", 0.5, 3.0, 1.5, 0.1, key="template_k1")
    with b_col:
        b = st.slider("BM25 b", 0.0, 1.0, 0.75, 0.05, key="template_b")

    retrieval = _template_retrieval(identified_entities, query, k1, b)
    exact_results = retrieval["exact"]
    bm25_results = retrieval["bm25"]
    exact_matches = [item for item in exact_results if item["score"]]

    st.divider()
    st.subheader("Step 5: Scores and Ranking")
    col_exact, col_bm25 = st.columns(2)
    with col_exact:
        st.markdown("### Method 1: Exact / Keyword Retrieval")
        st.caption("Deterministic boolean matching. Score is binary (1.0 = Match, 0.0 = No Match).")
        st.metric("Exact Matches Found", f"{len(exact_matches)} / {len(exact_results)}")
        st.dataframe(pd.DataFrame([{"Candidate Entity": x["entity"], "Type": x["type"], "Match Status": x["match"], "Binary Score": x["score"]} for x in exact_results]), use_container_width=True, hide_index=True)
    with col_bm25:
        st.markdown("### Method 2: Probabilistic Ranked Retrieval")
        st.caption("Okapi BM25 scores are normalized to a 0.0-1.0 relevance scale for comparison.")
        top_result = bm25_results[0] if bm25_results else None
        st.metric("Top Ranked Entity", top_result["entity"] if top_result else "None", delta=f"Relevance: {top_result['relevance_score'] if top_result else 0.0:.4f}")
        st.dataframe(pd.DataFrame([{"Rank": rank, "Candidate Entity": x["entity"], "Type": x["type"], "BM25 Score": x["score"], "Relevance Score": x["relevance_score"], "Confidence Level": x["confidence"]} for rank, x in enumerate(bm25_results, 1)]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Step 6: Compare Both Methods")
    exact_map = {x["entity"]: x["score"] for x in exact_results}
    fig = go.Figure()
    fig.add_bar(name="Exact / Keyword Retrieval (Binary)", x=[x["entity"] for x in bm25_results], y=[exact_map.get(x["entity"], 0.0) for x in bm25_results])
    fig.add_bar(name="BM25 Relevance Score", x=[x["entity"] for x in bm25_results], y=[x["relevance_score"] for x in bm25_results])
    fig.update_layout(barmode="group", title=f"Comparative Retrieval Score Distribution for Query: '{query}'", yaxis=dict(title="Retrieval Score [0.0 - 1.0]", range=[0, 1.15]), height=400)
    st.plotly_chart(fig, use_container_width=True, key="template_score_chart")

    st.divider()
    st.subheader("Step 7: Final Evaluation")
    relevant_entities = _template_entity_relevance(identified_entities, query)
    exact_metrics = _retrieval_metrics([x["entity"] for x in exact_results], relevant_entities, top_k)
    bm25_metrics = _retrieval_metrics([x["entity"] for x in bm25_results], relevant_entities, top_k)
    st.caption(f"Relevant entities from the identified text: {', '.join(sorted(relevant_entities)) or 'None'}")
    metric_names = [f"Precision@{top_k}", f"Recall@{top_k}", f"F1@{top_k}", "MRR"]
    exact_values = [exact_metrics["precision"], exact_metrics["recall"], exact_metrics["f1"], exact_metrics["mrr"]]
    bm25_values = [bm25_metrics["precision"], bm25_metrics["recall"], bm25_metrics["f1"], bm25_metrics["mrr"]]
    metric_cols = st.columns(4)
    for col, name, exact_value, bm25_value in zip(metric_cols, metric_names, exact_values, bm25_values):
        with col:
            st.metric(f"{name} - Exact", _format_metric(exact_value))
            st.metric(f"{name} - BM25", _format_metric(bm25_value))

    st.divider()
    st.subheader("Step 8: Experimental Data Log Book")
    if st.button("Record Current Trial", type="primary", use_container_width=True, key="template_record_trial"):
        trial = {"Trial #": len(st.session_state["trials"]) + 1, "Query": query, "Entities Identified": len(identified_entities), "Exact Matches": len(exact_matches), "Top Ranked Entity": top_result["entity"] if top_result else "N/A", "Relevance Score": top_result["relevance_score"] if top_result else 0.0, "BM25 P@K": bm25_metrics["precision"], "BM25 R@K": bm25_metrics["recall"], "Timestamp": datetime.now().strftime("%H:%M:%S")}
        st.session_state["trials"].append(trial)
        st.toast(f"Trial #{trial['Trial #']} successfully logged!")
    if st.session_state["trials"]:
        trials_df = pd.DataFrame(st.session_state["trials"])
        st.dataframe(trials_df, use_container_width=True, hide_index=True)
        st.download_button("Download Trials as CSV", trials_df.to_csv(index=False).encode("utf-8"), "experiment_6_entity_retrieval_trials.csv", "text/csv", use_container_width=True, key="template_download_trials")
    else:
        st.info("No trials recorded yet. Click 'Record Current Trial' to capture your experimental retrieval results.")


def render_simulation_section():
    """Guide the student through identification, retrieval, evaluation, and observation."""
    st.markdown("""
<style>
.sim-step { display:flex; align-items:center; gap:0.65rem; margin:1.8rem 0 0.45rem; }
.sim-step-num { display:inline-flex; align-items:center; justify-content:center; width:1.7rem; height:1.7rem;
    border-radius:50%; background:#0E7C7B; color:#fff; font-size:0.78rem; font-weight:800; }
.sim-step-title { color:#1F2937; font-size:1.18rem; font-weight:800; }
.sim-help { margin:0 0 0.8rem 2.35rem; color:#4B5563; font-size:0.98rem; line-height:1.6; }
.sim-callout { margin:0.6rem 0 1.1rem; padding:0.8rem 1rem; background:#F6FBFA; border:1px solid #BFE3E1;
    border-left:3px solid #0E7C7B; border-radius:6px; color:#4B5563; line-height:1.55; }
.sim-callout strong { color:#1F2937; }
.sim-eval-shell { margin:0.6rem 0 1rem; padding:1rem 1.1rem; background:#FEFDFB; border:1px solid #E7E2D3;
    border-left:4px solid #B3261E; border-radius:8px; }
.sim-eval-label { margin:0 0 0.25rem; color:#B3261E; font-size:0.78rem; font-weight:800;
    letter-spacing:0.13em; text-transform:uppercase; }
.sim-eval-title { margin:0 0 0.35rem; color:#1F2937; font-size:1.08rem; font-weight:800; }
.sim-eval-copy { margin:0 0 0.8rem; color:#4B5563; font-size:0.96rem; line-height:1.6; }
.sim-eval-grid { display:flex; flex-wrap:wrap; gap:0.6rem; }
.sim-eval-card { flex:1 1 145px; min-width:130px; padding:0.65rem 0.75rem; background:#F7F5EF;
    border:1px solid #E7E2D3; border-radius:6px; }
.sim-eval-name { margin:0 0 0.35rem; color:#1F2937; font-size:0.9rem; font-weight:800; }
.sim-eval-values { display:flex; justify-content:space-between; gap:0.5rem; color:#4B5563; font-size:0.84rem; line-height:1.5; }
.sim-eval-exact { color:#B3261E; font-weight:800; }
.sim-eval-bm25 { color:#0E7C7B; font-weight:800; }
.sim-eval-note { margin:0.85rem 0 0; color:#4B5563; font-size:0.9rem; line-height:1.55; }
div[data-testid="stButton"] button[kind="primary"] { background:#0E7C7B; border-color:#0E7C7B; }
div[data-testid="stButton"] button[kind="primary"]:hover { background:#0A6261; border-color:#0A6261; }
</style>
""", unsafe_allow_html=True)
    st.markdown('<div class="sim-callout"><strong>How this works:</strong> identify the entities first, choose a query, run both retrieval methods, and use the evaluation to interpret the difference.</div>', unsafe_allow_html=True)
    friendly_types = {
        "PERSON": "Person", "ORGANIZATION": "Organization", "LOCATION": "Place",
        "CONCEPT": "Idea or technology", "ROLE_TITLE": "Job title", "AWARD_EVENT": "Award or event"
    }

    st.markdown('<div class="sim-step"><span class="sim-step-num">1</span><span class="sim-step-title">Identify Graph Entities (Named Entity Recognition / NER)</span></div><p class="sim-help">This passage contains people, organizations, places, and ideas. Mark them first so the computer knows what it is allowed to search.</p>', unsafe_allow_html=True)
    input_mode = st.radio("Choose the experiment input:", ["Benchmark preset", "Custom text"], horizontal=True, key="guided_input_mode")
    preset_names = list(SAMPLE_PRESETS)
    if input_mode == "Benchmark preset":
        if "guided_preset_text" not in st.session_state:
            st.session_state["guided_preset_text"] = SAMPLE_PRESETS[preset_names[0]]
        st.selectbox("Select a benchmark passage:", preset_names, key="guided_preset", on_change=_select_guided_preset)
        active_text = st.text_area("Passage to identify:", height=100, key="guided_preset_text")
    else:
        if "guided_custom_text" not in st.session_state:
            st.session_state["guided_custom_text"] = SAMPLE_PRESETS[preset_names[0]]
        active_text = st.text_area("Paste a passage to identify:", height=100, key="guided_custom_text")

    text_signature = active_text.strip()
    if st.session_state.get("guided_text_signature") != text_signature:
        st.session_state["guided_entities"] = []
        st.session_state["guided_retrieval_run"] = False
        st.session_state["guided_query"] = ""
        st.session_state["guided_text_signature"] = text_signature
    if st.button("Identify Graph Entities", type="primary", key="guided_identify"):
        with st.status("Reading the passage and finding named items...", expanded=True) as status:
            st.write("Looking for known people, organizations, places, ideas, and job titles.")
            st.session_state["guided_entities"] = extract_template_entities(active_text)
            st.session_state["guided_text_signature"] = text_signature
            st.session_state["guided_retrieval_run"] = False
            status.update(label="Entity identification complete", state="complete")

    entities = st.session_state.get("guided_entities", [])
    if not entities:
        st.info("Choose or enter a passage, then click **Identify Graph Entities**.")
        return
    st.success(f"Result: we found {len(entities)} searchable items in the passage.")
    st.dataframe(pd.DataFrame([
        {"Entity": item["entity"], "What it is": friendly_types.get(item["type"], item["type"]), "Where it appeared": f"{item['context']} [{item['start']}, {item['end']}]"}
        for item in entities
    ]), use_container_width=True, hide_index=True)
    st.caption("Why this matters: these named items are now the complete list of things the search methods will consider. We do not draw relationships between them in this experiment.")

    graph = go.Figure()
    colors = {
        "PERSON": "#8B5CF6", "ORGANIZATION": "#2563EB", "LOCATION": "#10B981",
        "CONCEPT": "#F59E0B", "ROLE_TITLE": "#B3261E", "AWARD_EVENT": "#0E7C7B"
    }
    types = list(dict.fromkeys(item["type"] for item in entities))
    for type_index, entity_type in enumerate(types):
        typed = [item for item in entities if item["type"] == entity_type]
        graph.add_shape(
            type="line", x0=-0.45, x1=max(0.45, len(typed) - 0.55),
            y0=type_index, y1=type_index,
            line={"color": "#E7E2D3", "width": 1}, layer="below"
        )
        graph.add_trace(go.Scatter(
            x=list(range(len(typed))), y=[type_index] * len(typed), mode="markers+text",
            text=[item["entity"] for item in typed], textposition="top center",
            textfont={"family": "Georgia, serif", "size": 12, "color": "#1F2937"},
            name=friendly_types.get(entity_type, entity_type),
            marker={"size": 30, "color": colors.get(entity_type, "#64748B"), "line": {"width": 2, "color": "#1F2937"}},
            hovertemplate="<b>%{text}</b><br>Type: " + friendly_types.get(entity_type, entity_type) + "<extra></extra>"
        ))
    graph.update_layout(
        height=max(330, 115 * len(types)),
        title={
            "text": "<b>CANDIDATE ENTITY SPACE</b><br><sup>Identified items available for retrieval</sup>",
            "font": {"family": "Georgia, serif", "size": 16, "color": "#1F2937"},
            "x": 0.02, "xanchor": "left"
        },
        xaxis={"visible": False, "range": [-0.65, max(1.0, max(len([item for item in entities if item["type"] == entity_type]) for entity_type in types) - 0.35)]},
        yaxis={"visible": False, "range": [-0.6, max(0.6, len(types) - 0.4)], "autorange": False},
        showlegend=True,
        legend={"orientation": "h", "y": -0.08, "x": 0, "font": {"family": "Georgia, serif", "size": 11, "color": "#4B5563"}},
        margin={"l": 24, "r": 24, "t": 72, "b": 52},
        paper_bgcolor="#FEFDFB", plot_bgcolor="#FEFDFB",
        font={"family": "Georgia, serif", "color": "#1F2937"}
    )
    st.plotly_chart(graph, use_container_width=True, key="guided_candidate_graph")

    st.divider()
    st.markdown('<div class="sim-step"><span class="sim-step-num">2</span><span class="sim-step-title">Choose an Information Need (Search Query)</span></div><p class="sim-help">Search the candidate entities with a word or phrase. Try an exact name, a partial name, or a multi-word query.</p>', unsafe_allow_html=True)
    examples = _template_quick_queries(entities)
    example_columns = st.columns(max(1, len(examples)))
    for index, example in enumerate(examples):
        with example_columns[index]:
            st.button(example, key=f"guided_example_{index}", use_container_width=True, on_click=_select_guided_query, args=(example,))
    if not st.session_state.get("guided_query"):
        st.session_state["guided_query"] = examples[0]
    query = st.text_input("Query:", key="guided_query").strip()
    if not query:
        st.info("Enter a query to continue.")
        return

    st.markdown("**Configure the Retrieval Model Parameters:**")
    control_columns = st.columns(3)
    with control_columns[0]:
        k = st.slider("Top-K Evaluation Depth (K)", 1, len(entities), min(3, len(entities)), key="guided_k")
    with control_columns[1]:
        k1 = st.slider("BM25 k1 (Term-Frequency Saturation)", 0.5, 3.0, 1.5, 0.1, key="guided_k1")
    with control_columns[2]:
        b = st.slider("BM25 b (Document-Length Normalization)", 0.0, 1.0, 0.75, 0.05, key="guided_b")
    st.caption("K sets the number of ranked results evaluated. BM25 k1 controls term-frequency saturation, while BM25 b controls document-length normalization.")
    if st.button("Run Retrieval", type="primary", key="guided_run"):
        st.session_state["guided_retrieval_run"] = True
    if not st.session_state.get("guided_retrieval_run", False):
        st.info("Choose the settings above, then click **Run Retrieval** to calculate both methods.")
        return

    st.divider()
    st.markdown('<div class="sim-step"><span class="sim-step-num">3</span><span class="sim-step-title">Compare Retrieval Models (Boolean Matching vs. Probabilistic BM25)</span></div><p class="sim-help">Both methods search the same named items. Exact Retrieval checks for a direct match; Okapi BM25 scores useful query terms while accounting for candidate length.</p>', unsafe_allow_html=True)
    with st.status("Running both searches...", expanded=True) as status:
        st.write("Checking direct matches first, then calculating BM25 scores for every identified item.")
        retrieval = _template_retrieval(entities, query, k1, b)
        status.update(label="Both searches complete", state="complete")
    exact_results, bm25_results = retrieval["exact"], retrieval["bm25"]
    exact_names = [item["entity"] for item in exact_results]
    bm25_names = [item["entity"] for item in bm25_results]
    retrieval_columns = st.columns(2)
    with retrieval_columns[0]:
        st.markdown("**Exact / Keyword Retrieval (Boolean Matching)**")
        st.caption("This method asks: does the typed phrase directly appear in this item’s name? Yes gives 1; no gives 0.")
        st.dataframe(pd.DataFrame([{ "Rank": rank, "Entity": item["entity"], "What it is": friendly_types.get(item["type"], item["type"]), "Direct match": item["match"], "Score": item["score"]} for rank, item in enumerate(exact_results, 1)]), use_container_width=True, hide_index=True)
        if any(item["score"] > 0 for item in exact_results):
            st.success("Exact search found at least one item whose name directly matches your query.")
        else:
            st.warning("Exact search found no direct name match. That does not mean the passage is irrelevant; it only means the full phrase was not found in one name.")
    with retrieval_columns[1]:
        st.markdown("**Probabilistic Retrieval (Okapi BM25)**")
        st.caption("Okapi BM25 gives each item a relevance score: a larger value means the candidate shares more useful query terms with the search.")
        st.dataframe(pd.DataFrame([{ "Rank": rank, "Entity": item["entity"], "What it is": friendly_types.get(item["type"], item["type"]), "BM25 score": item["score"]} for rank, item in enumerate(bm25_results, 1)]), use_container_width=True, hide_index=True)
        if bm25_results:
            st.success(f"BM25 placed **{bm25_results[0]['entity']}** first because it received the highest score for this query.")

    st.divider()
    st.markdown('<div class="sim-step"><span class="sim-step-num">4</span><span class="sim-step-title">Evaluate Retrieval Quality (Information Retrieval Metrics)</span></div><p class="sim-help">Measure whether each method returns relevant entities, retrieves enough of them, and places the first useful result near the top.</p>', unsafe_allow_html=True)
    relevant = _template_entity_relevance(entities, query)
    st.markdown(
        f'<div class="sim-callout"><strong>Evaluation basis:</strong> the expected relevant entities for this query are '
        f'<strong>{", ".join(sorted(relevant)) or "None Identified"}</strong>. '
        'The same relevance set is used for Exact Retrieval and BM25 so the comparison is fair.</div>',
        unsafe_allow_html=True
    )
    exact_metrics = _retrieval_metrics(exact_names, relevant, k)
    bm25_metrics = _retrieval_metrics(bm25_names, relevant, k)
    metric_rows = [
        {"Metric": f"Precision@{k} (Top-K Accuracy)", "Exact Retrieval": _format_metric(exact_metrics["precision"]), "BM25 Retrieval": _format_metric(bm25_metrics["precision"]), "Interpretation": "Relevant results among the top K."},
        {"Metric": f"Recall@{k} (Result Coverage)", "Exact Retrieval": _format_metric(exact_metrics["recall"]), "BM25 Retrieval": _format_metric(bm25_metrics["recall"]), "Interpretation": "Known relevant entities found in the top K."},
        {"Metric": f"F1@{k} (Precision-Recall Balance)", "Exact Retrieval": _format_metric(exact_metrics["f1"]), "BM25 Retrieval": _format_metric(bm25_metrics["f1"]), "Interpretation": "Balance between precision and recall."},
        {"Metric": "MRR (Mean Reciprocal Rank)", "Exact Retrieval": _format_metric(exact_metrics["mrr"]), "BM25 Retrieval": _format_metric(bm25_metrics["mrr"]), "Interpretation": "How early the first relevant result appears."}
    ]
    metric_cards = "".join(
        f'<div class="sim-eval-card"><p class="sim-eval-name">{row["Metric"]}</p>'
        f'<div class="sim-eval-values"><span>Exact <b class="sim-eval-exact">{row["Exact Retrieval"]}</b></span>'
        f'<span>BM25 <b class="sim-eval-bm25">{row["BM25 Retrieval"]}</b></span></div></div>'
        for row in metric_rows
    )
    st.markdown(
        f'<div class="sim-eval-shell"><p class="sim-eval-label">Comparative Evidence</p>'
        f'<p class="sim-eval-title">How the two ranking methods performed</p>'
        f'<p class="sim-eval-copy">Precision measures correctness in the first K results. Recall measures coverage of the relevant set. F1 balances both measures, while MRR rewards an early first relevant result.</p>'
        f'<div class="sim-eval-grid">{metric_cards}</div>'
        f'<p class="sim-eval-note"><span class="sim-eval-exact">Exact Retrieval</span> is binary; '
        f'<span class="sim-eval-bm25">BM25 Retrieval</span> produces a ranked score for every candidate.</p></div>',
        unsafe_allow_html=True
    )
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    labels = [row["Metric"] for row in metric_rows]
    exact_values = [exact_metrics["precision"] or 0.0, exact_metrics["recall"] or 0.0, exact_metrics["f1"] or 0.0, exact_metrics["mrr"] or 0.0]
    bm25_values = [bm25_metrics["precision"] or 0.0, bm25_metrics["recall"] or 0.0, bm25_metrics["f1"] or 0.0, bm25_metrics["mrr"] or 0.0]
    metric_chart = go.Figure([go.Bar(name="Exact Retrieval", x=labels, y=exact_values), go.Bar(name="BM25 Retrieval", x=labels, y=bm25_values)])
    metric_chart.update_layout(
        barmode="group", title="Retrieval Quality Comparison",
        yaxis={"title": "Normalized Metric Value", "range": [0, 1.05], "gridcolor": "#E7E2D3"},
        height=360, paper_bgcolor="#FEFDFB", plot_bgcolor="#FEFDFB",
        font={"family": "Georgia, serif", "color": "#1F2937"},
        legend={"font": {"color": "#1F2937"}}
    )
    st.plotly_chart(metric_chart, use_container_width=True, key="guided_evaluation_chart")

    st.markdown('<div class="sim-step"><span class="sim-step-num">5</span><span class="sim-step-title">Interpret the result</span></div>', unsafe_allow_html=True)
    precision_winner = "Exact Retrieval" if exact_values[0] > bm25_values[0] else "BM25 Retrieval" if bm25_values[0] > exact_values[0] else "Both methods"
    recall_winner = "Exact Retrieval" if exact_values[1] > bm25_values[1] else "BM25 Retrieval" if bm25_values[1] > exact_values[1] else "Both methods"
    st.markdown(f'<div class="sim-callout"><strong>Query finding:</strong> {precision_winner} has the higher Precision@K, while {recall_winner} has the higher Recall@K. F1 summarizes their balance, and MRR shows how early the first relevant entity appears. This conclusion applies to the current query and candidate set only.</div>', unsafe_allow_html=True)
    st.markdown("**Key Learning**\n\n- Text was converted into structured graph entities.\n- The entities became the candidate search space.\n- Exact Retrieval performs direct matching.\n- BM25 Retrieval provides ranked results.\n- Precision, Recall, F1, and MRR evaluate retrieval quality.")

    st.divider()
    st.markdown('<div class="sim-step"><span class="sim-step-num">6</span><span class="sim-step-title">Trial log</span></div><p class="sim-help">Save this run so it can be included in the final lab report.</p>', unsafe_allow_html=True)
    if st.button("Record Current Trial", type="primary", key="guided_record_trial"):
        trial = {"Trial #": len(st.session_state["trials"]) + 1, "Query": query, "Entities": len(entities), "K": k, "Exact Results": ", ".join(exact_names), "BM25 Results": ", ".join(bm25_names), "Precision@K Exact": exact_metrics["precision"], "Precision@K BM25": bm25_metrics["precision"], "Recall@K Exact": exact_metrics["recall"], "Recall@K BM25": bm25_metrics["recall"], "F1@K Exact": exact_metrics["f1"], "F1@K BM25": bm25_metrics["f1"], "MRR Exact": exact_metrics["mrr"], "MRR BM25": bm25_metrics["mrr"], "Timestamp": datetime.now().strftime("%H:%M:%S")}
        st.session_state["trials"].append(trial)
        st.toast(f"Trial #{trial['Trial #']} recorded.")
    if st.session_state["trials"]:
        trial_df = pd.DataFrame(st.session_state["trials"])
        st.dataframe(trial_df, use_container_width=True, hide_index=True)
        st.download_button("Download Trial Log", trial_df.to_csv(index=False).encode("utf-8"), "experiment_6_trials.csv", "text/csv", key="guided_download_trials")
    else:
        st.info("Record this completed experiment to add it to the trial log.")


def _select_guided_query(query: str) -> None:
    st.session_state["guided_query"] = query


def _select_guided_preset() -> None:
    st.session_state["guided_preset_text"] = SAMPLE_PRESETS[st.session_state["guided_preset"]]


def render_quiz_section():
    """Renders Section 4: Quiz Assessment (10 questions randomly sampled from the question bank)."""
    st.markdown("### Concept Assessment Quiz")
    st.write("Answer the 10 conceptual questions below to evaluate your understanding of Graph Entities and Probabilistic Retrieval.")

    if st.button("🔄 Get New Question Set"):
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
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{version}_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)
            st.markdown("---")

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Detailed Feedback")
        for display_idx, q in enumerate(active_questions, start=1):
            user_ans = user_responses.get(q["id"])
            corr_ans = q["answer_index"]
            if user_ans == corr_ans:
                score += 1
                st.success(f"**Question {display_idx}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(
                    f"**Question {display_idx}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                    f"**Correct Answer:** {q['options'][corr_ans]}\n\n"
                    f"**Reasoning:** _{q['explanation']}_"
                )

        st.session_state["quiz_score"] = score
        perc = (score / len(active_questions)) * 100
        st.info(f"Final Quiz Score: **{score} / {len(active_questions)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already completed. Current score: **{st.session_state.get('quiz_score', 0)} / {len(active_questions)}**")


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
    st.markdown("### Official Lab Report Generation")
    st.write("Compile your student details, diagnostic scores, recorded simulation trials, and observations into a downloadable PDF report.")

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

    st.subheader("Observations & Analysis Notes")
    student_notes = st.text_area(
        "Enter your interpretation of results, observations, and conclusions:",
        value=st.session_state.get("student_notes", (
            "The experimental trials demonstrated that entity-aware probabilistic ranking consistently outperformed "
            "the standard Okapi BM25 keyword baseline across Precision@K, MAP, and NDCG@K metrics by capturing "
            "disambiguated semantic relationships between persons, organizations, locations, and concepts."
        )),
        height=120
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['exp_number']}. {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Discipline:** {EXPERIMENT_CONFIG['discipline']} | **Subject:** {EXPERIMENT_CONFIG['subject']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(
        f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(st.session_state.get('quiz_question_ids', []))}"
    )

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Note: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 trials.")

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
            file_name="kgirs_exp6_lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            use_container_width=True
        )


def render_certificate_section():
    """Renders Section 6: Certificate of Completion."""
    st.markdown("### 🎓 Certificate of Completion")
    st.write("Download a personalized certificate for completing this KGIRS virtual lab experiment.")

    col1, col2, col3 = st.columns(3)
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
    if st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz Score on Record: **{quiz_score} / {quiz_total}**")
    else:
        st.info(
            "You haven't submitted the Quiz yet — your certificate will show a score of 0 until you do. "
            "(The certificate itself is available regardless.)"
        )

    cert_bytes = generate_certificate_pdf(
        student_name=cert_name,
        student_id=cert_id,
        date_str=str(cert_date),
        quiz_score=quiz_score,
        quiz_total=quiz_total
    )

    st.divider()
    st.download_button(
        label="🎓 Download Certificate (.pdf)",
        data=cert_bytes,
        file_name="kgirs_exp6_certificate.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
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
