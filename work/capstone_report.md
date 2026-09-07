# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Araan Sheikh
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Araan-Sheikh/flyrank
- **Date:** September 2026
- **Base Rate (Majority Class):** 54.21% (16,262 / 30,000 items in decay)

---

## 0. Abstract

Enterprise publishing repositories frequently suffer from organic search visibility decay, yet editorial teams lack automated, high-precision methods to triage which pages warrant content revisions. We investigate whether machine learning models can accurately detect search performance decay and outperform conventional heuristic scoring rules on real search intelligence data. Using the FlyRank Search Intelligence dataset—encompassing 30,000 anonymized content items across 32 enterprise clients observed over a trailing 90-day window—we engineer 52 leakage-free behavioral, position, and engagement features under a strict client-holdout validation split. A Random Forest classifier achieves **0.680 Precision@50** and **0.747 ROC-AUC** on completely held-out client domains, delivering a **2.83× precision lift** over the standard heuristic baseline (0.240 Precision@50). The resulting system compiles an automated, prioritized refresh queue with interpretable reason codes to direct editorial bandwidth toward high-ROI revisions while safeguarding non-decaying assets.

---

## 1. Problem framing

Enterprise content organizations maintain thousands to tens of thousands of published URLs. Over time, algorithmic adjustments, search intent shifts, competitive content launches, and factual obsolescence cause significant portions of this inventory to lose organic visibility. However, manually auditing every page is operationally infeasible: an in-depth editorial audit and content refresh typically demands 3 to 6 hours of writer and subject-matter expert time. 

- **Decision Supported:** Guiding weekly and monthly editorial refresh resource allocation, answering: *"Which 20 to 50 URLs across our enterprise catalog should editorial staff update this sprint to stem search visibility loss?"*
- **Unit of Analysis:** A single published content item (`content_id`) from an enterprise client (`client_id`) evaluated over a trailing 90-day performance window.
- **System Output:** A calibrated decay probability score ($[0, 1]$), a composite priority rank, an assigned categorical action (`refresh_and_review_ctr`, `refresh`, `refresh_and_review_engagement`, `expand_and_refresh`, `monitor`), and primary trigger reason codes.
- **Cost of Wrong Calls:**
  - *False Positives (High Score, Not Decaying):* Scarce editorial hours are wasted revising content that is already performing stably or naturally recovering, incurring substantial opportunity cost.
  - *False Negatives (Low Score, Decaying):* High-value content quietly loses page-one rankings and qualified organic search traffic to competing domains without editorial awareness.
- **Why Data & ML Help:** Heuristic rules (e.g., flagging any page with `impressions > 500` and `rank > 15`) suffer from severe impression bias and generate a 76% false discovery rate in top tiers. Machine learning balances multi-dimensional interaction effects—jointly assessing exposure scale, rank trajectory, CTR efficiency relative to position, and engagement depth—to isolate true decay from expected volatility.

---

## 2. Data safety & Privacy

This research strictly adheres to enterprise data privacy, public release safety, and data leakage protocols.

- **Data Source:** FlyRank Search Intelligence Internship Dataset (`data/raw/content_refresh_anonymized.csv`), containing 30,000 records across 32 enterprise client domains.
- **Excluded Columns (Leakage Guards):**
  - `trend_direction` and `trend_pct`: The ground-truth evaluation labels are derived directly from these trajectory metrics. Including them or any direct velocity derivatives in the feature matrix would constitute catastrophic label leakage.
  - 30-day velocity metrics (`impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, etc.) are held strictly out of feature engineering to prevent retrospective lookahead leakage.
- **Pseudonymous Identifiers:**
  - `client_id` is used exclusively for group splitting (`client_holdout`) and client-level stratification. It is never supplied as a training feature or one-hot encoded, preventing models from memorizing client-specific baseline scale.
  - `content_id` serves solely as a record key for join integrity and queue generation.
- **Public Safety Verification:** All client domain names, URLs, specific target keywords, brand names, and user queries have been pseudonymized or omitted. No proprietary search engine algorithm reverse-engineering is claimed; all analyses are framed as observational decision support on empirical search console data.

---

## 3. Baseline

Before training machine learning models, we established a transparent, deterministic heuristic baseline rule reflecting standard industry SEO practices.

### Heuristic Rule Architecture
The baseline computes a weighted composite risk index ($S_{\text{baseline}} \in [0, 1]$) combining four normalized percentile ranks:
$$S_{\text{baseline}} = 0.35 \cdot P_{\text{visibility}} + 0.30 \cdot P_{\text{freshness\_risk}} + 0.25 \cdot P_{\text{position\_opp}} + 0.10 \cdot P_{\text{depth\_gap}}$$
where:
- $P_{\text{visibility}} = \text{percentile\_rank}(\log(1 + \text{impressions\_90d}))$
- $P_{\text{freshness\_risk}} = \text{percentile\_rank}(\text{days\_since\_last\_update})$
- $P_{\text{position\_opp}} = 1 - |\text{avg\_position} - 15| / 15$ (favoring striking-distance pages on page 2)
- $P_{\text{depth\_gap}} = 1 - \text{percentile\_rank}(\text{word\_count})$

### Baseline Performance on Held-Out Client Test Set (2,325 rows)
- **Precision@20:** 0.150 (3 true decaying pages out of 20)
- **Precision@50:** 0.240 (12 true decaying pages out of 50)
- **Precision@100:** 0.360 (36 true decaying pages out of 100)
- **ROC-AUC:** 0.627
- **Average Precision (PR-AUC):** 0.468

**Why the Baseline Fails:** Heuristic thresholds disproportionately favor high-impression evergreen articles that are inherently stable, while failing to detect nuanced engagement decay or CTR drops on mid-tier content. Over 75% of the baseline's top-recommended pages do not actually require remediation.

---

## 4. Model / analysis

### Target Definition
The target variable is binary search trajectory decay:
$$Y = \mathbb{I}(\text{trend\_direction} == \text{'down'})$$
Across the full 30,000-row corpus, the observed positive base rate is **54.21%** (16,262 decaying items).

### Feature Vector Design (52 Features)
All features are constructed strictly from data observable prior to the evaluation window:
1. **Search Demand & Visibility (Log-Transformed):** `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`.
2. **Search Console Signals:** `avg_position`, `ctr`, `days_with_impressions`, `days_with_sessions`.
3. **Engagement & Traffic Quality:** `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
4. **Content Freshness & Scope:** `content_age_days`, `days_since_last_update`, `word_count`, `char_count`, `search_volume`, `competition`, `cpc`.
5. **Categorical Encodings (One-Hot):** `content_type` (article, blog, guide, product, landing), `competition_level` (low, medium, high), `main_intent` (informational, transactional, commercial, navigational), and performance tiers (`age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`).

### Model Candidates
We trained and evaluated three distinct model families:
1. **Regularized Logistic Regression:** L2-regularized linear model with standard scaling, providing a linear additive benchmark.
2. **Decision Tree Classifier:** Single decision tree (`max_depth=6`, `min_samples_leaf=20`) to capture non-linear threshold effects without ensemble complexity.
3. **Random Forest Classifier:** Ensemble of 100 estimators (`max_depth=12`, `min_samples_leaf=10`, `random_state=42`) using bootstrap aggregation and feature sub-sampling to model high-order interactions.

---

## 5. Evaluation

### Split Design: Client Holdout Split
To guarantee rigorous real-world evaluation, we implemented a **client-holdout split**:
- **Test Set:** 4 complete client domains (2,325 items, 7.75% of data) held out entirely.
- **Training Set:** 28 client domains (27,675 items, 92.25% of data).
- **Rationale:** Random row splitting permits pages from the same website to appear in both training and test partitions. Because domains exhibit unique internal linking structures, domain authority, and niche topic velocities, row-level splitting enables models to memorize domain baselines, inflating Precision@50 to 0.760. The client-holdout split evaluates true cross-domain generalization.

### Comparative Evaluation on Unseen Client Test Set

| Model / Baseline | Precision@20 | Precision@50 | Precision@100 | ROC-AUC | PR-AUC | F1-Score | Accuracy | Lift vs Baseline |
|:---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Deterministic Baseline Rule** | 0.150 | 0.240 | 0.360 | 0.627 | 0.468 | 0.421 | 0.518 | 1.00× |
| **Logistic Regression (L2)** | 0.350 | 0.400 | 0.440 | 0.700 | 0.522 | 0.562 | 0.584 | 1.67× |
| **Decision Tree (Depth 6)** | 0.450 | 0.580 | 0.620 | 0.742 | 0.575 | 0.648 | 0.655 | 2.42× |
| **Random Forest (100 Trees)** | **0.700** | **0.680** | **0.700** | **0.747** | **0.610** | **0.682** | **0.688** | **2.83×** |

### Headline Metric & Discrimination Lift
The Random Forest model achieved **0.680 Precision@50** compared to **0.240** for the baseline rule—representing a **2.83× lift**. In operational terms, for every 50 articles assigned to an editorial sprint, the ML model correctly identifies **34 truly decaying articles**, compared to only **12** under the baseline rule.

### Error Analysis
- **False Positives (Predicted Decay, Actually Stable/Growing):** Primarily high-impression legacy hub pages that experienced a minor drop in secondary keyword rankings while primary query volume remained resilient. The model interprets position fluctuations as decay, whereas the page's core visibility remained intact.
- **False Negatives (Predicted Stable, Actually Decaying):** Concentrated in niche, low-impression articles with short historical lifespans (<90 days). Because baseline impression counts were already low, early decay signals did not produce large absolute behavioral shifts.

---

## 6. Interpretation & Findings

### Feature Importance Hierarchy (Random Forest)
Feature importances derived from Mean Decrease in Impurity reveal the dominant predictive signals:

| Rank | Feature Name | Relative Importance | Interpretability Summary |
|:---:|:---|---:|:---|
| 1 | `days_with_impressions` | 16.06% | Consistency of daily search visibility over 90 days. Sporadic impression appearance indicates indexation instability. |
| 2 | `log_impressions_90d` | 12.85% | Overall visibility scale; high-volume assets exhibit distinct decay dynamics compared to long-tail items. |
| 3 | `avg_position` | 10.84% | Mean SERP rank. Pages drifting into positions 11–25 represent prime decay candidates. |
| 4 | `content_age_days` | 9.50% | Time since initial publication; older articles naturally accumulate factual and competitive obsolescence. |
| 5 | `days_with_sessions` | 5.34% | Frequency of organic session arrivals over the observation window. |
| 6 | `word_count` | 4.12% | Content depth proxy. Thin articles demonstrate higher vulnerability to competing content updates. |
| 7 | `ctr` | 3.87% | Click efficiency relative to rank position. |
| 8 | `log_clicks_90d` | 3.65% | Total realized click volume. |
| 9 | `engagement_rate` | 3.22% | User satisfaction and content consumption depth. |
| 10 | `search_volume` | 2.94% | Macro keyword search volume. |

### Negative Results & Surprises
- **AI Referral Traffic Share (`ai_traffic_pct`):** Accounted for less than 0.5% of model importance. AI-referral sessions remain sparse across the corpus and do not currently correlate with organic Google search trajectory decay.
- **Author / Provider Metadata:** Features encoding AI-assisted generation (`model_used`, `provider_used`) showed negligible predictive value ($<0.8\%$). In observational data, content quality and topic positioning dominate over publication tooling.

---

## 7. Operational Recommendations & Playbook

The trained Random Forest model scored all 30,000 items in the catalog, producing a prioritized action queue segmented by specific remediation strategies:

### Action Playbook Distribution

| Recommended Action | Volume | Share | Targeted Editorial Playbook |
|:---|---:|---:|:---|
| **`refresh`** | 8,207 | 27.36% | Comprehensive content update: refresh outdated statistics, update publication year, expand competitive subtopics, and re-verify external citations. |
| **`refresh_and_review_ctr`** | 6,655 | 22.18% | Snippet optimization: rewrite title tags and meta descriptions to improve SERP click capture for high-impression, low-CTR queries. |
| **`refresh_and_review_engagement`** | 1,987 | 6.62% | On-page experience audit: improve visual hierarchy, add summary callouts, optimize media load speeds, and refine introduction hooks. |
| **`expand_and_refresh`** | 82 | 0.27% | Structural rewrite: short, thin content (<500 words) currently ranking on page 2 requires substantial depth expansion. |
| **`monitor`** | 13,069 | 43.56% | Passive monitoring: stable or upward-trending assets requiring zero editorial intervention. |

### Sprint Triage Protocol
1. **Weekly Triage:** Content leads review the top 50 entries in `outputs/refresh_queue.csv` filtered by `confidence == 'high'`.
2. **Action Assignment:** Copywriters receive assigned URLs with their specific `suggested_action` and `final_reason_codes` (e.g., `high_visibility_low_ctr`).
3. **No-Go Exclusions:**
   - Homepage, pricing, or checkout URLs are strictly excluded from automated refresh queues.
   - Pages updated within the trailing 30 days are locked to allow search engine re-indexing.
   - Seasonal content (e.g., annual holiday guides) is held until 60 days prior to peak season.

---

## 8. Reproducibility

The complete pipeline is deterministically reproducible from the repository root:

### Environment Setup
```bash
git clone https://github.com/Araan-Sheikh/flyrank.git
cd flyrank
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### End-to-End Pipeline Execution
```bash
# Execute the full 5-step ML pipeline (features, baseline, training, evaluation, PDF report)
python scripts/run_all.py

# Execute and validate all 10 Jupyter notebooks top-to-bottom
python scripts/build_and_run_notebooks.py
```

### Artifact Manifest
- **Raw Input:** `data/raw/content_refresh_anonymized.csv` (SHA256 verified)
- **Engineered Features:** `data/processed/refresh_feature_vector.csv` (30,000 rows × 52 columns)
- **Baseline Queue:** `data/processed/baseline_refresh_queue.csv`
- **Model Evaluation:** `outputs/model_results.json` (sealed client-holdout metrics)
- **Final Action Queue:** `outputs/refresh_queue.csv` (30,000 scored recommendations)
- **Visualizations:** `outputs/charts/` (5 publication SVGs)
- **Executive PDF:** `outputs/flyrank_refresh_model_results.pdf`

---

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai)). 

We express gratitude to the FlyRank AI research team for curating and releasing this anonymized search intelligence dataset for academic and machine learning research. All analyses, findings, and recommendations presented in this paper reflect independent research and are intended solely for editorial decision support.
