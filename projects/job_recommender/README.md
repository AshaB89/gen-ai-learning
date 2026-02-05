Scalable Job Recommendation System
==================================

### Overview

This project implements and evaluates a **job recommendation system** using real‑world–style job postings and **simulated user interaction data**.  
The focus is on:

- **Understanding** evaluation for recommenders (time‑based, future behavior as ground truth)
- **Comparing** simple baselines vs more complex models
- **Designing** the system with clear separation between:
  - data processing (`pipelines/`)
  - models (`models/`)
  - evaluation (`evaluation/`)

### Project Structure

```text
job_recommender/
├── data/
│   ├── raw/
│   │   └── jobs.csv
│   └── processed/
│       ├── jobs.csv          # cleaned job postings
│       ├── interactions.csv  # simulated user–job events
│       └── top_jobs.csv      # precomputed popular jobs
├── pipelines/
│   ├── ingest.py             # raw → processed jobs
│   └── simulate_events.py    # jobs → simulated interactions
├── models/
│   ├── baseline.py           # popularity baseline (Phase 0)
│   └── content_based.py      # TF‑IDF content model (Phase 1)
└── evaluation/
    ├── data_loader.py
    ├── split.py
    ├── metrics.py
    ├── run_baseline_eval.py
    └── run_content_eval.py
```

### Dataset

**Job data**

- ~22,000 job postings
- Fields (after processing):
  - `job_id`
  - `title`
  - `description`
  - `category`
  - `location`
  - `company`
  - `uniq_id`

**Interaction data**

- ~109,000 implicit interactions
- ~3,000 simulated users
- Event types:
  - `view`  → weak signal
  - `click` → medium‑strength signal
  - `apply` → strong signal
- Each interaction has a **timestamp**, which is critical for time‑based evaluation.

---

### Evaluation Strategy

**Ground truth definition**

- Recommenders don’t have fixed labels like classification.
- **Ground truth = what the user does in the future.**

Evaluation steps:

1. Sort each user’s interactions by `timestamp`.
2. Use earlier interactions as **history** (what the model “knows”).
3. Use later interactions as **ground truth** (what actually happened).
4. Check if recommended jobs overlap with those future interactions.

This answers:  
**“Given what we knew at time _t_, would we have recommended what the user did at time _t+Δ_?”**

**Metrics**

- **Recall@10**
  - Did at least one relevant (future) job appear in the top‑10?
- **MRR@10 (Mean Reciprocal Rank)**
  - How early does the first relevant job show up in the ranked list?

Implementation details:

- `evaluation/split.py` – time‑based train/test split per user
- `evaluation/metrics.py` – `recall_at_k`, `mrr_at_k`, `evaluate_model(...)`

---

### Phase 0 – Popularity‑Based Baseline

**What it is**

- A **non‑personalized** recommender:
  - Ranks jobs by weighted interaction count:
    - view = 1, click = 3, apply = 5
  - Returns the **same top‑K** jobs for every user (except for exclusions).
- Implemented in `models/baseline.py` and used by `run_baseline_eval.py`.

**Why it matters**

- Acts as:
  - **Cold‑start fallback**
  - **System safety net**
  - **Performance benchmark** for all other models

**Result (simulated data)**

- Recall@10 ≈ **0.0063**
- MRR@10    ≈ **0.0023**

**Key takeaways**

- Popularity is a **very strong signal** in job recommendation.
- Even a “dumb” baseline can be surprisingly hard to beat.
- Every production system should have a robust popularity layer.

Run evaluation from the `genai_learning` root:

```bash
python projects/job_recommender/evaluation/run_baseline_eval.py
```

---

### Phase 1 – Content‑Based Recommendation

**Model description**

- A **personalized** recommender using job content similarity:
  - Text fields: `title`, `description`, `category`, `location`
  - Represent each job with **TF‑IDF** vectors over concatenated text
  - Build a **user profile** as the average of vectors of jobs they interacted with
  - Score jobs for a user using **cosine similarity** to the user profile
- Implemented in `models/content_based.py` and evaluated via `run_content_eval.py`.

**Initial result (using all interaction types)**

- Recall@10 ≈ **0.0043**
- MRR@10    ≈ **0.0008**

This was **worse than the popularity baseline**, which is common when:

- Signals are noisy (many low‑intent views).
- The model sees too much weak signal relative to strong.

---

### Improving Content‑Based: Signal Filtering

**Motivation**

- Not all events are equally informative:
  - `view`  → often accidental or exploratory
  - `click` → clearer interest
  - `apply` → strong relevance

**Change**

- Build user profiles using **only `click` and `apply`** interactions.
- Optionally weight events, e.g. `apply` > `click`.

**Result**

- Recall@10 ≈ **0.0043** (unchanged)
- MRR@10    ≈ **0.0014** (improved)

**Interpretation**

- **Recall** stayed the same → no increase in overall coverage.
- **MRR** increased → relevant jobs moved **higher** in the ranking.
- Shows that improving **signal quality** can help more than changing the model architecture.

Run evaluation:

```bash
python projects/job_recommender/evaluation/run_content_eval.py
```

---

### Key Learnings

1. **Evaluation is about time**
   - Ground truth in recommender systems is **future behavior**, not static labels.

2. **Popularity is a strong baseline**
   - Any advanced model must **justify its complexity** by beating a simple popularity ranker on Recall@K / MRR@K.

3. **Signal quality > model complexity**
   - Filtering out weak interactions (e.g. views) improved ranking quality more than changing the model class.

4. **Content‑based models help ranking more than coverage**
   - They refine ordering of candidates but don’t necessarily “discover” many new items beyond what popularity and behavior already surface.

5. **No single model is sufficient**
   - Real systems typically combine:
     - **Popularity** (coverage, robustness, safety net)
     - **Content‑based** (cold‑start, explainability, matching on skills/domain)
     - **Collaborative filtering** (behavioral patterns across users)

This project implements **Phase 0 (popularity)** and **Phase 1 (content‑based)** and sets the foundation for future **hybrid** and **collaborative** approaches.