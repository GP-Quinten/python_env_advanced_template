# Pairwise Features for Duplicate Detection (Tier B)

This page lists recommended features for training a binary classifier that predicts whether two registry names refer to the same entity. Each feature group includes a short rationale and suggested implementation notes.


# Roadmap

- Refactor code so Gaetan can start working on it: 
  - Bucketing
    - Implement class with fit / transform methods so wee can use it pipeline
  - DONE Do train/test split separtly from feature extraction
  - Feature extraction:
    - CANCELED Do normalization inside FeatureExtractor and specify which feature uses normalized or not inputs
    - DONE  Make FeatureExtractor class design simple so it could be easy to integrate new features
      - DONE  it must have fit and transofrm, and all specifique feature params must be provided in class constructor so we can execute GridSearch on it
    - DONE  Plot feature discribution on train/test
    - DONE  move code into features_extracton/extractor.py - main class / files for specifique feature extracions
  - Training:
    - Visualise Grid search results
    - Implement hyperparms importance plot*
    - Plot loss function
    - Export as an object which gets features and implements predicts_proba
    - Move common code our of the gbm script, so we could try other models
  - DO FEATURE EXTRACTION AND MODEL TRAINING in the same script so we csn leverage grid search  
  - Inference (same as in production, without bucketing?)
    - On Test
    - on Unlabeled dataset
    - Two sepratate calls
  - Eval
    - Show top, middle, bottom examples from unlabeled dataset
    - Plot for test set X: unique clusters number (>3) Y: Precision(threshold)
    - Plot for unlabeled set X: unique clusters number (>3) Y: Threshold??

- Implement production version(MUST)
  - how it will work in production?
    - Registry Name -> Bucketing -> Pairs -> Feature Extraction -> Model -> Cluster Assigment
        - sample pairs candidates using LSH
        - inference with proba model
        - for positive prediciton get according clusters and decide to which cluster assign new member
          - Use strategy - Most popular positive cluster
  - Re-train every week? Validate manually new changes?
- Implement more features, especially include mistral embeddings, check README.md
  - add features based on abrevation, medical condition, geo area
- Run massive Grid search over Bucketing, FeatureExtraction and Model params
  - first study Model params, then FeatureExtraction, then Bucketing
- Add SHap values in Eval(could)
- Try other sklearn models RandomForest, other ?
- Try other models like XGboost, LightGBM (should)
- Improve Special Vocabs and normalization
- Refine dataset based on Error Analysis Report, improve LLM-as-a-judge
- Use Active Learning on unlabled dataset to imptove


TAEGET PERF: Percision 0.95, Recall 0.85

---

## A. String Similarity Metrics

**1) Jaro–Winkler similarity**

* **What it captures:** Character transpositions/near-duplicates; strong signal for minor typos and small edits.
* **Range:** \[0, 1] (1 = identical).
* **Notes:** Use on **normalized** strings; fast to compute and robust for short names.

**2) Normalized Levenshtein similarity**

* **What it captures:** Edit distance scaled by max length; penalizes insertions/deletions/substitutions.
* **Definition:** `1 − (levenshtein_distance / max_len)`
* **Notes:** Complements JW by handling longer gaps and rewordings.

**3) Token Jaccard similarity**

* **What it captures:** Set overlap of tokens, order-agnostic.
* **Definition:** `|tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|`
* **Notes:** Compute on normalized tokens with domain stopwords removed for better signal.

**4) Cosine similarity over TF–IDF vectors**

* **What it captures:** Weighted token or character n-gram overlap.
* **Variants:**

  * **BoW tokens:** TF–IDF on tokens.
  * **Char n-grams (3–5):** More tolerant to typos and affixes.
* **Notes:** Fit the TF–IDF vectorizer on **all normalized names**; keep the same vocabulary at train/test.

**5) Soft TF–IDF**

* **What it captures:** Like cosine on tokens, but lets *similar* tokens match (e.g., “tumour”≈“tumor”).
* **Mechanics:** Greedy one-to-one matching of tokens with similarity ≥ τ (e.g., JW ≥ 0.9), weighted by TF–IDF.
* **Notes:** Use as a high-precision matcher when token variants/typos are common.

**6) Overlap coefficient**

* **What it captures:** Intersection relative to the **smaller** set.
* **Definition:** `|A ∩ B| / min(|A|, |B|)`
* **Notes:** Useful when names have very different lengths; guards against short aliases.

---

## B. Token-Level Features

**1) Token count difference**

* **Definition:** `abs(len(tokens(A)) − len(tokens(B)))`.
* **Why:** Large differences often indicate non-duplicates.

**2) Common token fraction**

* **Definition:** `|A ∩ B| / min(|A|, |B|)` (same as overlap coefficient, but keep both for ablation clarity).
* **Why:** Normalizes by the shorter name to avoid length bias.

**3) Longest common subsequence (LCS) — normalized**

* **Definition:** `LCS(chars(A), chars(B)) / max_len`.
* **Why:** Captures order-sensitive similarity without being as strict as exact match.

**4) Shared rare tokens**

* **Definition:** Fraction of tokens present in both names whose IDF ≥ τ (e.g., top 20% rare).
* **Why:** Rare, domain-specific tokens are strong duplicate signals (e.g., “glioblastoma”, “Utrecht”).

---

## C. Character N-gram Features

**1) Char n-gram Jaccard**

* **Definition:** Jaccard over sets of 3-grams and 4-grams (compute both).
* **Why:** Tolerant to small typos; language-agnostic; strong for short strings.

**2) Char n-gram cosine**

* **Definition:** Cosine similarity over TF–IDF vectors built on character 3–5 grams.
* **Why:** Smooth measure; complements token-based similarities.

---

## D. Semantic Embedding Features

**1) Sentence embedding cosine similarity**

* **Encoders:** MistralAI.
* **Why:** Captures meaning beyond surface tokens (e.g., “myocardial infarction” ≈ “heart attack”).
* **Notes:** Precompute embeddings for all names; use ANN index for scalable nearest-neighbor recalls.

**2) Embedding distance margins (optional)**

* **Definition:** `cos(left, right) − max(cos(left, negative_k))` over a few hard negatives.
* **Why:** Measures how uniquely close the pair is compared to confounders.

---

## E. Boolean / Heuristic Indicators

**1) Exact normalized match**

* **Why:** Deterministic strong positive signal.

**2) Same token bag (ignoring order)**

* **Why:** Strong indicator that names differ only by punctuation/order.

**3) Startswith / endswith**

* **Why:** Short aliases or truncated forms; useful with other signals.

**4) Contains same acronym**

* **Why:** Acronym alignment with name expansion is a strong hint (e.g., “NCRI” ↔ “National Cancer Registry of Ireland”)

---

## F. Domain-Specific Features

**1) Geographical area distance**

* **What it captures:** Measures the distance between geographical areas associated with the two names.
* **Why:** Geographical proximity is often a strong predictor of duplicates in registry data.

**2) Medical condition distance**

* **What it captures:** Measures the similarity or distance between medical conditions extracted from the names.
* **Variants:**
  * Use pre-extracted medical condition features.
  * Use an embedding model specialized in medical tasks to compute the distance between the two names.
* **Why:** Medical conditions are key predictors for duplicates in healthcare-related registries.

* **Notes:** These features align with human intuition and are expected to be strong predictors for the model.



