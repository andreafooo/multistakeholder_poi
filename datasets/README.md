# Metadata POI Recommendations — Dataset Guide

This guide walks you through the structure and contents of the POI Recommendations datasets: **Foursquare Tokyo** (`foursquaretky`) and **Yelp**. Both datasets follow the same layout.

---

## 1. What's in each dataset

Each dataset folder contains three categories of files:

1. **Recommendation lists** — the actual ranked POI recommendations for each user as outputted by each recommendation method and re-ranker. 
2. **Original metadata (sampled)** — the underlying user, item, and interaction data, restricted to the 1,500-user sample used for training
3. **ID mappings** — a lookup table between the dataset's original IDs and the internal IDs used across all files
4. *(Optional)* **User popularity groups** — a label (high / medium / low) for each user based on their popularity profile

---

## 2. Recommendation lists

For every user, each method produces a ranked list of the **top-150 items**. Note that while 150 items are stored, **evaluation is only performed on the top-10**; the extra items are there in case you want to experiment with different cutoffs. 

The methods fall into three groups:

### 2.1 Baseline

| Method | Description |
|---|---|
| `baseline` | Standard BPR (Bayesian Personalized Ranking) recommendations for each user, with no re-ranking applied |

### 2.2 Simple agents

Each agent re-ranks the baseline list to optimize for a different stakeholder goal:

| Method | Agent role (CIKM paper) | Goal |
|---|---|---|
| `cp_min_js` | Platform Agent (Calibrated Popularity) | Align recommendations with the popularity distribution seen in the user's own profile |
| `mmr` | Provider Agent (Maximal Marginal Relevance) | Diversify the recommendation list |
| `geo` | Civic Agent | Minimize geographic distance between consecutive POIs, reducing travel cost as a proxy for emissions |

### 2.3 Social choice aggregation

These combine `baseline`, `cp_min_js`, `mmr`, and `geo` into a single ranked list per user, using a voting rule.

| Method | Aggregation rule |
|---|---|
| `borda` | Borda count — points assigned by rank position, summed across all four lists |
| `schulze` | Condorcet-winner based — the winner beats every other candidate in pairwise comparison |

**Equal weighting** is the default: all four stakeholders (baseline, cp_min_js, mmr, geo) contribute equally to the aggregated list.

### 2.4 Weighted aggregation experiments

Same aggregation rules as above, but with one stakeholder's vote doubled while the others stay at normal weight. This lets you study how much a single stakeholder can shift the outcome when given more influence.

**Naming convention:** `<aggregation><weight><stakeholder>`

Example: **`borda2baseline`** = Borda aggregation, with the `baseline` list weighted 2× relative to the other three.

The same pattern applies for any stakeholder (`cp_min_js`, `mmr`, `geo`) and either aggregation rule (`borda`, `schulze`).

---

## 3. Original metadata (sampled)

Alongside the recommendation lists, each dataset includes the underlying metadata for the **1,500-user sample** the models were trained on.

Depending on the dataset, this includes:

- **User metadata** — attributes of each of the 1,500 sampled users
- **Item / POI metadata** — attributes of the venues/businesses those users interacted with (e.g. location, name, category)
- **Interaction metadata** — the check-ins or reviews used as training/evaluation signal

**Note:** 
* **Foursquare** only includes item metadata, therefore the user/item interactions are in a single file `foursquaretky_sample_full_metadata.csv`
* **Yelp** includes a lot of metadata for users, businesses (items), and reviews: therefore they are in separate files (`yelp_sample_business_metadata.csv`, `yelp_sample_review_metadata.csv`, `yelp_sample_user_metadata.csv`)

---

## 4. ID mappings

The library used for recommendation generation worked with short IDs (e.g. `0_x, 1_x, 2_x, ...`) rather than the original IDs from the source dataset. To let you trace any internal ID back to the real-world entity it represents, each dataset includes an `id_mappings.json` file with:

- A mapping from **internal user IDs → original user IDs**
- A mapping from **internal item IDs → original item/business IDs**

---

## 5. User popularity groups (optional)

Where included, this file assigns each user to a popularity group:

- **High** — users who predominantly interact with popular/mainstream POIs
- **Medium** — users with a mixed popularity profile
- **Low** — users who predominantly interact with niche/unpopular POIs

This grouping is useful for slicing evaluation results by user type — for example, checking whether the `cp_min_js` agent behaves differently for high- vs. low-popularity users.

---

## 6. Quick reference

```
<dataset>_dataset/
├── id_mappings.json                   # original ID <-> internal ID
├── <recommendation files per method>  # baseline, cp_min_js, mmr, geo,
│                                      # borda, schulze, borda2<x>, schulze2<x>, ...
├── <sampled metadata files>.csv       # user / item / interaction data
│                                      # for the 1,500-user sample
└── <dataset>_user_id_popularity.json  # optional: high / medium / low per user
```

**Method cheat sheet:**

| Prefix/Name | Type |
|---|---|
| `baseline` | Unweighted BPR |
| `cp_min_js` | Simple agent — popularity calibration |
| `mmr` | Simple agent — diversity |
| `geo` | Simple agent — geographic distance |
| `borda` / `schulze` | Equal-weight aggregation |
| `borda<N><prefix>` / `schulze<N><prefix>` | Weighted aggregation, `<prefix>` weighted `<N>`× |

## 7. Paper References: 

For further info on the preprocessing etc., check out: 

* RecSys Short Paper: 
https://dl.acm.org/doi/abs/10.1145/3705328.3748017

* CIKM Short Paper:
(Added in this directory, currently under peer review 🤞) \
Paper Repo: https://github.com/andreafooo/multistakeholder_poi

