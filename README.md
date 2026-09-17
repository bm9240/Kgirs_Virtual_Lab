# Walkthrough — Experiment 6: Identify Graph Entities

**Knowledge Graph Information Retrieval System (KGIRS) — Virtual Lab**

We have successfully implemented **Experiment 6: Identify Graph Entities** for the **Knowledge Graph Information Retrieval System** in `template.py`.

The implementation preserves the **4-section modular Virtual Lab architecture** and covers the required entity identification, retrieval, probabilistic ranking, comparative visualization, assessment, and report generation components.

---

## 1. Architectural Structure

The application is organized into four standard sections using **Streamlit native components** without custom CSS, ensuring compatibility with both light and dark themes.

1. **Theory**

   * Theoretical background
   * Learning objectives
   * Lab procedure
   * Key terminology

2. **Simulation**

   * Interactive entity identification
   * Candidate graph entities
   * Exact/keyword retrieval
   * Probabilistic ranked retrieval
   * Comparative visualization
   * Trial logging

3. **Quiz**

   * 10-question conceptual assessment
   * Self-grading
   * Instant feedback
   * Detailed explanations
   * Score tracking

4. **Report Generation**

   * Student information
   * Recorded trials
   * Observations
   * Quiz evaluation
   * Downloadable PDF report

---

## 2. Experiment Pipeline

The implemented workflow is:

```text
Text
  ↓
Entity Identification (NER)
  ↓
Candidate Entities
  ↓
Query
  ↓
Dual Retrieval
  ├── Exact / Keyword Retrieval
  └── Probabilistic Ranked Retrieval
  ↓
Probabilistic Ranking
  ↓
Comparative Visualization
```

---

## 3. Entity Identification Engine

The entity identification engine extracts named entities from predefined sample texts or custom user input.

### Supported Entity Types

| Entity Type    | Examples                                                |
| -------------- | ------------------------------------------------------- |
| `PERSON`       | Sundar Pichai, Alan Turing, Marie Curie                 |
| `ORGANIZATION` | Google, Stanford University, CERN, OpenAI               |
| `LOCATION`     | California, San Francisco, Geneva, Stockholm            |
| `ROLE_TITLE`   | CEO, Inventor, Scientist                                |
| `TECH_CONCEPT` | Artificial Intelligence, World Wide Web, Enigma machine |

The identified entities are displayed with:

* Entity name
* Entity type
* Start/end character offsets
* Relevant textual context

These identified entities serve as **candidate entities/nodes for the Knowledge Graph**.

---

## 4. Dual Retrieval System

The experiment implements two retrieval approaches so that students can compare deterministic matching with probabilistic ranking.

### 4.1 Exact / Keyword Retrieval

The exact retrieval method uses deterministic:

* String matching
* Token containment
* Boolean matching

The method produces a binary result:

```text
MATCH FOUND → 1.0
NO MATCH    → 0.0
```

This demonstrates the behavior of literal keyword matching.

For example, a query such as:

```text
Google
```

can directly match the entity:

```text
Google
```

However, a compound query such as:

```text
Google CEO
```

may fail to produce an exact entity match when no single candidate entity contains the complete phrase.

---

### 4.2 Probabilistic Ranked Retrieval

The probabilistic retrieval method assigns a continuous relevance score to candidate entities and ranks them in descending order.

The implementation uses the following relevance formulation:

```text
P(R=1 | Q,E) =
min(1.0,
    0.35 × Coverage
  + 0.30 × J(Q,E)
  + 0.25 × Sim_ngram
  + 0.10 × ContextBoost)
```

where the score considers factors such as:

* Query-term coverage
* Token Jaccard similarity
* Sub-word n-gram similarity
* Context relevance

The resulting candidates are sorted by their relevance score.

For example:

```text
Query: Sundar

1. Sundar Pichai    → 0.8804
```

This allows partial queries to retrieve relevant candidate entities even when the query is not an exact entity name.

---

## 5. Comparative Retrieval

The main feature of the experiment is the comparison between:

**Exact / Keyword Retrieval**

and

**Probabilistic Ranked Retrieval**

For every candidate entity, the system can display:

| Entity          | Exact Score | Probabilistic Score |
| --------------- | ----------: | ------------------: |
| Google          |         1.0 |              1.0000 |
| CEO             |         0.0 |               0.475 |
| Other candidate |         0.0 |    Calculated score |

This demonstrates the difference between:

* **Binary matching**, where an entity either matches or does not match.
* **Ranked retrieval**, where candidates can receive different degrees of relevance.

---

## 6. Comparative Visualization

A **Plotly grouped bar chart** is used to visualize retrieval scores.

The chart compares:

* Exact Retrieval Score (`0` or `1`)
* Probabilistic Relevance Score (`0.0–1.0`)

This provides a visual representation of how the two retrieval methods rank candidate entities.

The system also provides an **analytical insight callout** explaining the observed difference between the retrieval methods.

---

## 7. Trial Logging

The Simulation section includes an experimental data logger.

Each recorded trial contains:

| Field               | Description                         |
| ------------------- | ----------------------------------- |
| Trial #             | Experimental trial number           |
| Query               | Search query used                   |
| Entities Identified | Number of identified entities       |
| Exact Matches       | Number of exact matches             |
| Top Ranked Entity   | Highest-ranked probabilistic result |
| Relevance Score     | Score of the top-ranked entity      |
| Timestamp           | Trial execution time                |

The recorded trials can be:

* Viewed inside the application
* Cleared when required
* Downloaded as a CSV file

---

## 8. Concept Assessment Quiz

The application contains **10 conceptual questions** covering the main concepts of the experiment.

The quiz covers:

* Graph entities in Knowledge Graphs
* Named Entity Recognition (NER)
* Candidate entities
* Exact/keyword retrieval
* Probability Ranking Principle (PRP)
* Token Jaccard similarity
* Query-term coverage
* Compound query behavior
* Sub-word n-gram similarity
* Precision and recall
* Binary vs. graded relevance

The quiz provides:

* Interactive multiple-choice questions
* Automatic grading
* Final score
* Percentage score
* Correct/incorrect feedback
* Explanation for each answer

---

## 9. Lab Report Generator

The application includes a PDF-based lab report generator using `fpdf2`.

The generated report contains:

* Experiment title
* Student name
* Student ID/Roll number
* Experiment date
* Learning objectives
* Recorded experimental trials
* Observations and analysis
* Quiz evaluation
* Evaluator/student signature section

The report can be downloaded directly from the application.

---

## 10. Verification

### Automated Test Suite

The implementation was tested using:

```bash
py scratch/verify_app.py
```

The following components were verified:

* Entity extraction across all 5 benchmark domain presets
* Exact retrieval
* Probabilistic retrieval
* Ranking behavior
* Quiz structure
* Quiz answer keys
* PDF generation

### Benchmark Results

| Test Case      | Result                                                           |
| -------------- | ---------------------------------------------------------------- |
| `Google`       | Exact match found; probabilistic score `1.0000`, Rank #1         |
| `Google CEO`   | Exact matches `0`; probabilistic retrieval ranked Google and CEO |
| `Sundar`       | Sundar Pichai ranked #1 with score `0.8804`                      |
| Quiz           | 10 questions validated                                           |
| PDF Generation | Successfully generated (`2,692 bytes`)                           |

**Result: ALL TESTS PASSED SUCCESSFULLY.**

---

## 11. Application Startup

The application was started using:

```bash
py -m streamlit run template.py --server.headless=true --server.port=8501
```

The Streamlit application runs on:

```text
http://localhost:8501
```

---

## 12. Final Experiment Flow

The completed Virtual Lab follows this workflow:

```text
┌───────────────────────────────┐
│ Experiment 6                  │
│ Identify Graph Entities       │
└───────────────┬───────────────┘
                ↓
          Enter / Select Text
                ↓
        Identify Graph Entities
                ↓
       Candidate Entity Set
                ↓
           Enter Query
                ↓
       ┌────────┴─────────┐
       ↓                  ↓
Exact / Keyword     Probabilistic
   Retrieval           Retrieval
       ↓                  ↓
       └────────┬─────────┘
                ↓
       Compare Rankings
                ↓
      Visualize Relevance
                ↓
         Record Trial
                ↓
             Quiz
                ↓
        Generate Report
```

---

## Conclusion

**Experiment 6: Identify Graph Entities** has been implemented as an interactive Virtual Lab for the **Knowledge Graph Information Retrieval System**.

The implementation demonstrates the complete process of identifying entities from text and retrieving them using both **exact keyword matching** and **probabilistic relevance ranking**. The comparison allows students to observe how ranked retrieval can handle partial and compound queries differently from deterministic matching.

The application retains the original Virtual Lab architecture while replacing the generic simulation with an experiment-specific **entity identification and retrieval workflow**.
