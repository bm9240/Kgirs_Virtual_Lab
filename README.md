Walkthrough
Walkthrough - Experiment 6: Identify Graph Entities (KGIRS Virtual Lab)
We have successfully implemented Experiment 6: Identify Graph Entities for the Knowledge Graph Information Retrieval System in template.py. The implementation strictly preserves the 4-section modular virtual lab architecture and adheres to all educational, algorithmic, and visual specifications.

What Was Accomplished
1. Architectural Structure Preserved
The application is structured into the 4 standard sections using Streamlit native components without custom CSS, ensuring compatibility with light and dark themes:

Theory: Theoretical background, learning objectives, lab procedure, and terminology reference.
Simulation: Interactive Entity Identification and Dual Retrieval (Exact vs. Probabilistic Ranking) sandbox.
Quiz: 10-question conceptual self-grading assessment with instant explanations and score tracking.
Report Generation: Lab report compilation with live preview and downloadable PDF document.
2. Pipeline Implementation
Text
⟶
Entity Identification (NER)
⟶
Candidate Entities
⟶
Query
⟶
Dual Retrieval
⟶
Probabilistic Ranking
⟶
Comparative Visualization
Text⟶Entity Identification (NER)⟶Candidate Entities⟶Query⟶Dual Retrieval⟶Probabilistic Ranking⟶Comparative Visualization
A. Entity Identification Engine
Extracts named entities from user-selected presets (e.g., Sundar Pichai at Google/Stanford, Alan Turing at Bletchley Park, Satya Nadella at Microsoft/OpenAI) or custom user input.
Classifies entities into standard ontological types:
PERSON (e.g., Sundar Pichai, Alan Turing, Marie Curie)
ORGANIZATION (e.g., Google, Stanford University, CERN, OpenAI)
LOCATION (e.g., California, San Francisco, Geneva, Stockholm)
ROLE_TITLE (e.g., CEO, Inventor, Scientist)
TECH_CONCEPT (e.g., Artificial Intelligence, World Wide Web, Enigma machine)
Displays identified entities with start/end character offsets and textual context windows as candidate nodes for the Knowledge Graph.
B. Dual Retrieval System
Method 1: Exact / Keyword Retrieval:
Deterministic boolean string and token containment.
Generates binary match status (MATCH FOUND vs. NO MATCH) and binary scores (
1.0
1.0 or 
0.0
0.0).
Demonstrates high precision on literal canonical names, but zero recall on multi-word compound queries (e.g., "Google CEO" returns 0 matches).
Method 2: Probabilistic Ranked Retrieval:
Implements Robertson's Probability Ranking Principle (PRP) by calculating continuous relevance probability: 
P
(
R
=
1
∣
Q
,
E
)
=
min
⁡
(
1.0
,
0.35
⋅
Coverage
+
0.30
⋅
J
(
Q
,
E
)
+
0.25
⋅
Sim
ngram
+
0.10
⋅
ContextBoost
)
P(R=1∣Q,E)=min(1.0,0.35⋅Coverage+0.30⋅J(Q,E)+0.25⋅Sim 
ngram
​
 +0.10⋅ContextBoost)
Ranks candidate entities in descending order of relevance score.
Gracefully retrieves multiple relevant entities for compound queries (e.g. for "Google CEO", ranks both Google and CEO with high scores) and partial terms (e.g. "Sundar" matches Sundar Pichai).
C. Visual Score Comparison & Trial Logging
Plotly Comparison Chart: Grouped bar chart plotting Exact Retrieval (0 or 1) alongside Probabilistic PRP Relevance Scores ($0.0 - 1.0$) for each candidate entity.
Analytical Insights Callout: Automatically explains why probabilistic ranking succeeded where exact match failed.
Trial Logger: Logs Trial #, Query, Entities Identified, Exact Matches, Top Ranked Entity, Relevance Score, and Timestamp. Includes CSV download and clear log functionality.
D. Concept Assessment Quiz (10 Questions)
Form with interactive radio buttons covering:
Definition of Graph Entities in Knowledge Graphs.
Sub-tasks of Named Entity Recognition (NER).
Role of Candidate Entities prior to graph integration.
Operational limitations of Exact/Keyword retrieval.
Robertson's Probability Ranking Principle (PRP).
Token Jaccard similarity and query term coverage formulation.
Compound query behavior (e.g., "Google CEO").
Sub-word n-gram similarity and fuzzy typo resilience.
Precision vs. Recall trade-offs.
Advantages of graded relevance over binary scoring.
Live grading with instant score calculation, percentage display, and detailed rationale for every question.
E. Lab Report Generator (.pdf)
Clean, professional PDF generation using fpdf2 with text sanitization to prevent unicode encoding issues.
Includes student metadata, experiment objectives, complete recorded trials table, student observation notes, and quiz score evaluation with an evaluator signature block.
Verification Results
1. Automated Test Suite Execution
Executed scratch/verify_app.py via py:

Entity Extraction: Verified across all 5 benchmark domain presets (extracted 5–6 entities per preset).
Exact vs. Probabilistic Retrieval:
Query "Google": Exact match found; Probabilistic score: 
1.0000
1.0000 (Rank #1).
Query "Google CEO": Exact matches: 
0
0; Probabilistic top 2: Google (
0.55
0.55) and CEO (
0.475
0.475).
Query "Sundar": Probabilistic ranked Sundar Pichai #1 (
0.8804
0.8804).
Quiz Structure: 10 questions validated with valid option indices and answer keys.
PDF Generation: Successfully compiled a complete PDF report of size 2,692 bytes.
Result: ALL TESTS PASSED SUCCESSFULLY.
2. Live Application Startup
Executed py -m streamlit run template.py --server.headless=true --server.port=8501.
Streamlit application is running and accessible on http://localhost:8501.
