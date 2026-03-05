# Behavioral Segmentation

A full-stack web application that clusters website users by behavioral metrics using K-Means, profiles each segment, and generates targeted campaign recommendations.

---

## Features

- **Multi-method clustering** — K-Means (primary), Agglomerative (validation), DBSCAN (concept)
- **Automatic K selection** — Silhouette Score + Davies-Bouldin Score over K = 2–8
- **PCA scatter plot** — 2-component dimensionality reduction for visual cluster separation
- **4 segment profiles** — Power Users, At-Risk Users, Window Shoppers, Casual Browsers
- **Targeting table** — Channel, message strategy, CTA, priority and expected lift per segment
- **Clean sidebar dashboard** — 4 tabs: Overview, Cluster Profiles, Analysis, Targeting

---

## Project Structure

```
behavioriq/
├── app.py            # Flask backend — data generation, clustering, API routes
├── index.html        # Frontend — single-file HTML with inline CSS and JS
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Run the server

```bash
python app.py
```

You should see:

```
==================================================
  BehaviorIQ  |  User Segmentation Engine
==================================================
  Total users     : 500
  Optimal K       : 4
  Silhouette score: 0.8512
  PCA variance    : 92.4%
==================================================
  [Power Users    ]  n=125  conv_avg=8.80
  [At-Risk Users  ]  n=125  conv_avg=4.05
  [Window Shoppers]  n=125  conv_avg=0.47
  [Casual Browsers]  n=125  conv_avg=0.12
==================================================
  Dashboard -> http://127.0.0.1:5000
```

### 3 · Open the dashboard

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

> **Note:** `index.html` must be in the same directory as `app.py` so Flask can serve it.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves `index.html` |
| `/favicon.ico` | GET | Returns 204 No Content (suppresses browser 404) |
| `/api/overview` | GET | Total users, optimal K, silhouette score, PCA variance |
| `/api/elbow` | GET | Inertia, silhouette and DB scores for K = 2–8 |
| `/api/scatter` | GET | PCA coordinates + cluster label for every user |
| `/api/profiles` | GET | Mean feature values and metadata per cluster |
| `/api/radar` | GET | Normalised (0–100) feature means for radar chart |
| `/api/targeting` | GET | Campaign strategy table per segment |

---

## Clustering Details

### Features used

| Feature | Description |
|---|---|
| `session_frequency` | Number of sessions per month |
| `avg_time_spent` | Average session duration (minutes) |
| `scroll_depth` | Average page scroll depth (%) |
| `repeat_visits` | Number of return visits |
| `conversion_history` | Number of past conversions |

### Pipeline

```
Raw data (500 users)
  → StandardScaler (zero mean, unit variance)
  → KMeans(k=2..8) for each K
  → Silhouette + Davies-Bouldin to pick optimal K
  → Final KMeans(k=optimal) + AgglomerativeClustering(k=optimal)
  → PCA(n=2) for scatter plot
  → Profiles sorted by avg conversion_history (desc)
```

### Why K = 4?

Silhouette score peaks at **0.85** for K = 4, meaning clusters are dense and well-separated. Davies-Bouldin score is also lowest at K = 4, confirming this choice from a second independent metric.

---

## Segment Profiles

| Segment | Behaviour | Conversion | Strategy |
|---|---|---|---|
| 🔵 Power Users | High on every metric | Highest | Upsell / loyalty rewards |
| 🟠 At-Risk Users | Frequent short sessions | Medium | Win-back re-engagement |
| 🟢 Window Shoppers | Long sessions, no conversion | Low | Discount nudge |
| 🔴 Casual Browsers | Low engagement overall | Minimal | Nurture / SEO content |

---

## Notes

- The `/favicon.ico` route returns **204 No Content**. This is intentional — it silences the automatic browser request that previously generated a 404 log line. No icon file is needed.
- Synthetic data is generated with `numpy.random.seed(0)` for full reproducibility. Replace `generate_data()` in `app.py` with your own data source (CSV, database, etc.) — the rest of the pipeline is unchanged.
- The frontend makes API calls to `http://localhost:5000`. If you change the port, update the `API` constant at the top of the `<script>` block in `index.html`.
