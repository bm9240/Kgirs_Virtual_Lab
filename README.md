# Walkthrough — Experiment 6: Identify Graph Entities

**Knowledge Graph Information Retrieval System (KGIRS) — Virtual Lab**

Experiment 6 has been implemented as an interactive **Streamlit-based Virtual Lab** following the provided Virtual Lab template and IIT Kharagpur-style experimental structure.

The experiment demonstrates the complete process of **identifying graph entities from text and retrieving relevant entities using exact and probabilistic retrieval methods**.

---

## 1. Virtual Lab Structure

The application is divided into four sections:

1. **Theory**

   * Aim and objectives
   * Conceptual background
   * Entity identification
   * Graph entities
   * Information retrieval
   * Retrieval methods
   * Procedure and expected outcome

2. **Simulation**

   * Text input
   * Entity identification
   * Entity type identification
   * Candidate graph
   * Query-based retrieval
   * Exact retrieval
   * Probabilistic ranked retrieval
   * Comparative analysis
   * Trial logging

3. **Quiz**

   * 10 conceptual questions
   * Automatic evaluation
   * Explanations and score

4. **Report Generation**

   * Student details
   * Experimental trials
   * Observations
   * Quiz performance
   * Downloadable PDF report

---

# 2. Experiment Objective

The simulation demonstrates:

```text
Text
  ↓
Entity Identification
  ↓
Entity Types
  ↓
Candidate Graph Entities
  ↓
Query
  ↓
Exact / Keyword Retrieval
  ↓
Probabilistic Ranked Retrieval
  ↓
Comparison
  ↓
Observation
```

The objective is to allow students to **perform the retrieval experiment and observe the difference between exact matching and ranked retrieval**.

---

# 3. Simulation

The Simulation section is designed as a step-by-step laboratory activity.

## Step 1 — Text Input

The student can either:

* Select a predefined benchmark text, or
* Enter custom unstructured text.

For example:

> "Ananya Verma is a Manager at Amazon."

The student then selects **Identify Graph Entities** to begin the experiment.

The system processes the text and identifies candidate entities.

---

## Step 2 — Entity Identification

The identified entities are extracted from the input text.

Each entity is presented with information such as:

* Entity mention
* Entity type
* Character span
* Context from the original text

For example:

| Entity       | Type         |
| ------------ | ------------ |
| Ananya Verma | PERSON       |
| Manager      | ROLE         |
| Amazon       | ORGANIZATION |

The simulation therefore demonstrates how relevant concepts in unstructured text can be identified as potential graph entities.

---

## Step 3 — Candidate Graph

The identified entities are treated as **candidate graph nodes**.

The simulation intentionally uses a small number of entities so that the graph remains easy to understand and follows the instruction that Neo4j is not required.

The candidate graph represents the entities identified from the current text.

Selecting/examining an entity allows the student to understand:

* What the entity is
* What type it belongs to
* Where it appeared in the text
* Its surrounding context

No additional relationship extraction is performed because this experiment focuses specifically on **entity identification**.

---

# 4. Query-Based Retrieval

After identifying the candidate entities, the student provides a retrieval query.

Example:

```text
Amazon
```

or

```text
Manager Amazon
```

The same query is evaluated using two retrieval approaches.

---

# 5. Exact / Keyword Retrieval

The first retrieval method performs direct keyword-based matching against the candidate entities.

The result indicates whether a candidate entity directly matches the query.

For example:

```text
Query: Amazon

Amazon → Match
Ananya Verma → No Match
Manager → No Match
```

This demonstrates the limitation of strict keyword matching: candidates that are only partially related to the query may not be retrieved.

---

# 6. Probabilistic Ranked Retrieval

The second method calculates a **Probabilistic Relevance Score** for each candidate.

The score considers factors such as:

* Query-term coverage
* Token similarity
* Jaccard similarity
* Character n-gram similarity
* Relevant textual context

The candidates are then sorted according to their relevance score.

Example:

| Rank | Candidate    | Relevance Score |
| ---: | ------------ | --------------: |
|    1 | Amazon       |            0.82 |
|    2 | Manager      |            0.46 |
|    3 | Ananya Verma |            0.12 |

The score is used as a **relevance score**, not as a calibrated probability.

This demonstrates how probabilistic retrieval can rank candidates even when the query is not an exact match.

---

# 7. Comparative Retrieval

The main practical component of the experiment is the comparison between the two retrieval approaches.

| Aspect          | Exact / Keyword Retrieval | Probabilistic Ranked Retrieval |
| --------------- | ------------------------- | ------------------------------ |
| Matching        | Direct keyword matching   | Similarity-based matching      |
| Output          | Match / No Match          | Ranked candidates              |
| Score           | Binary                    | Continuous relevance score     |
| Partial matches | Limited                   | Can be ranked                  |
| Result ordering | No relevance ranking      | Ranked by relevance            |

The simulation presents both results together so that students can directly observe the difference.

---

# 8. Comparative Visualization

A visual comparison is provided for the retrieval results.

The visualization compares:

* Exact retrieval score
* Probabilistic relevance score

This allows students to see how the same candidate entities can behave differently under the two retrieval methods.

An interpretation section explains the result of the current query in simple terms.

For example:

> Exact retrieval depends on direct matching, whereas probabilistic retrieval assigns different relevance scores and ranks candidates according to their similarity to the query.

---

# 9. Trial Logging

Each completed retrieval experiment can be recorded.

The trial log stores information such as:

| Field               | Description                  |
| ------------------- | ---------------------------- |
| Trial Number        | Current experiment trial     |
| Query               | Query entered by the student |
| Entities Identified | Number of detected entities  |
| Exact Matches       | Number of exact matches      |
| Top Ranked Entity   | Highest-ranked candidate     |
| Relevance Score     | Score of the top candidate   |
| Timestamp           | Time of experiment           |

Students can review their trials and download the experimental data as a CSV file.

---

# 10. Quiz

The Quiz section contains **10 questions** covering the concepts demonstrated in the experiment.

Topics include:

* Graph entities
* Entity identification
* Entity types
* Candidate entities
* Exact retrieval
* Probabilistic retrieval
* Probability Ranking Principle
* Similarity measures
* Query coverage
* Ranked retrieval
* Binary vs. graded relevance

The quiz provides:

* Multiple-choice questions
* Automatic evaluation
* Score and percentage
* Correct/incorrect feedback
* Explanations

---

# 11. Report Generation

The Report Generation section creates a downloadable PDF containing:

* Experiment title
* Student name
* Roll number/ID
* Experiment date
* Objectives
* Recorded trials
* Experimental observations
* Quiz performance
* Signature section

This provides a record of the student's completed experiment.

---

# 12. Final Simulation Flow

The complete simulation follows:

```text
┌──────────────────────────────┐
│ Experiment 6                 │
│ Identify Graph Entities      │
└──────────────┬───────────────┘
               ↓
        Step 1: Text Input
               ↓
    Step 2: Identify Entities
               ↓
     Step 3: Entity Types &
        Candidate Graph
               ↓
       Step 4: Enter Query
               ↓
      ┌────────┴────────┐
      ↓                 ↓
 Exact / Keyword    Probabilistic
    Retrieval          Retrieval
      ↓                 ↓
      └────────┬────────┘
               ↓
        Compare Results
               ↓
       Visualize Ranking
               ↓
        Record Trial
               ↓
             Quiz
               ↓
       Generate Report
```

---

# Conclusion

**Experiment 6: Identify Graph Entities** is implemented as an interactive KGIRS Virtual Lab experiment.

The simulation takes the student from **unstructured text to identified graph entities and finally to query-based retrieval and ranking**. The comparison between exact retrieval and probabilistic ranked retrieval allows students to observe how different retrieval strategies handle direct and partial matches.

The implementation follows the provided Streamlit template, uses a small candidate graph without requiring Neo4j, and provides the complete **Theory → Simulation → Quiz → Report Generation** workflow.
