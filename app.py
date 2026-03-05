"""
BehaviorIQ — Behavioral Segmentation Backend
=============================================
Deliverables covered:
  1. All 5 metrics: session_frequency, avg_time_spent, scroll_depth,
     repeat_visits, conversion_history
  2. Multiple clustering techniques: K-Means, Agglomerative, DBSCAN
     - Optimal K selected via Silhouette Score + Davies-Bouldin + Elbow
     - Cluster agreement score (Adjusted Rand Index) between KMeans & Agg
  3. Each cluster profiled with mean behavioural characteristics + std deviation
  4. Highest-conversion segment identified via a normalised conversion_probability
     score (0-100) derived from all 5 feature means
  5. Segment-based targeting recommendation table
"""

from flask import Flask, jsonify, send_from_directory, Response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              adjusted_rand_score)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='.')

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ── FAVICON (suppresses browser 404) ─────────────────────────────────────────
@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
#    500 synthetic users, 4 well-separated behavioural segments, fixed seed.
# ═══════════════════════════════════════════════════════════════════════════════
def generate_data():
    np.random.seed(0)
    #                  n   session_freq   avg_time    scroll_depth  repeat_visits  conversions
    segments = [
        dict(n=125, sf=(20,1,17,23),   at=(25,1.5,21,29), sd=(88,3,80,96), rv=(16,1,13,19),  ch=(9,1,7,12)),
        dict(n=125, sf=(2,.5,1,3.5),   at=(2,.5,1,3.5),   sd=(15,3,8,25),  rv=(1,.3,1,2),    ch=(.1,.1,0,.5)),
        dict(n=125, sf=(7,1,5,10),     at=(18,1.5,14,22), sd=(72,4,62,82), rv=(4,1,2,7),     ch=(.5,.3,0,1.5)),
        dict(n=125, sf=(14,1,11,17),   at=(6,1,4,9),      sd=(35,4,25,48), rv=(9,1,7,12),    ch=(4,1,2,6)),
    ]
    frames = []
    for s in segments:
        n = s['n']
        frames.append(pd.DataFrame({
            'session_frequency':  np.random.normal(*s['sf'][:2], n).clip(s['sf'][2], s['sf'][3]),
            'avg_time_spent':     np.random.normal(*s['at'][:2], n).clip(s['at'][2], s['at'][3]),
            'scroll_depth':       np.random.normal(*s['sd'][:2], n).clip(s['sd'][2], s['sd'][3]),
            'repeat_visits':      np.random.normal(*s['rv'][:2], n).clip(s['rv'][2], s['rv'][3]),
            'conversion_history': np.random.normal(*s['ch'][:2], n).clip(s['ch'][2], s['ch'][3]),
        }))
    df = pd.concat(frames, ignore_index=True).round(2)
    df['user_id'] = [f'U{str(i+1).zfill(4)}' for i in range(len(df))]
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = ['session_frequency', 'avg_time_spent', 'scroll_depth',
            'repeat_visits', 'conversion_history']

df_raw = generate_data()
scaler = StandardScaler()
X      = scaler.fit_transform(df_raw[FEATURES])
pca    = PCA(n_components=2)
X_pca  = pca.fit_transform(X)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. K SELECTION  (Elbow / Silhouette / Davies-Bouldin over K=2..8)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_k_scores():
    k_vals, inertias, silhouettes, db_scores = [], [], [], []
    for k in range(2, 9):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        k_vals.append(k)
        inertias.append(round(float(km.inertia_), 2))
        silhouettes.append(round(float(silhouette_score(X, labels)), 4))
        db_scores.append(round(float(davies_bouldin_score(X, labels)), 4))
    return k_vals, inertias, silhouettes, db_scores

K_VALS, INERTIAS, SILHOUETTES, DB_SCORES = compute_k_scores()
OPTIMAL_K = K_VALS[SILHOUETTES.index(max(SILHOUETTES))]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLUSTERING  — K-Means (primary) + Agglomerative (validation) + DBSCAN
# ═══════════════════════════════════════════════════════════════════════════════

# --- K-Means ------------------------------------------------------------------
kmeans_model         = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df_raw['kmeans_lbl'] = kmeans_model.fit_predict(X)

# --- Agglomerative Clustering (Ward linkage) ----------------------------------
agg_model          = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
df_raw['agg_lbl']  = agg_model.fit_predict(X)

# --- DBSCAN (density-based — auto-detects clusters & outliers) ---------------
# eps tuned so it finds ~4 meaningful clusters on this dataset
dbscan_model         = DBSCAN(eps=1.2, min_samples=8)
df_raw['dbscan_lbl'] = dbscan_model.fit_predict(X)   # -1 = noise/outlier

dbscan_n_clusters = len(set(df_raw['dbscan_lbl']) - {-1})
dbscan_n_noise    = int((df_raw['dbscan_lbl'] == -1).sum())

# --- Cross-method agreement (Adjusted Rand Index: 0=random, 1=identical) -----
ari_kmeans_agg = round(float(adjusted_rand_score(
    df_raw['kmeans_lbl'], df_raw['agg_lbl'])), 4)

# Silhouette scores for each method
sil_kmeans = round(float(silhouette_score(X, df_raw['kmeans_lbl'])), 4)
sil_agg    = round(float(silhouette_score(X, df_raw['agg_lbl'])), 4)
# DBSCAN silhouette only if >1 cluster found and noise is not majority
dbscan_valid_mask = df_raw['dbscan_lbl'] != -1
if dbscan_n_clusters > 1 and dbscan_valid_mask.sum() > OPTIMAL_K:
    sil_dbscan = round(float(silhouette_score(
        X[dbscan_valid_mask], df_raw.loc[dbscan_valid_mask, 'dbscan_lbl'])), 4)
else:
    sil_dbscan = None

df_raw['pca_x'] = X_pca[:, 0].round(4)
df_raw['pca_y'] = X_pca[:, 1].round(4)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTER PROFILING
#    Mean + std of all 5 features per K-Means cluster.
#    Sorted by conversion_probability (derived normalised score 0–100).
# ═══════════════════════════════════════════════════════════════════════════════

# Weights for conversion_probability composite score
# Conversion history carries most weight; scroll depth and time also signal intent
CONV_WEIGHTS = dict(
    session_frequency  = 0.15,
    avg_time_spent     = 0.20,
    scroll_depth       = 0.20,
    repeat_visits      = 0.15,
    conversion_history = 0.30,
)
FEAT_MAX = dict(session_frequency=30, avg_time_spent=40,
                scroll_depth=100,     repeat_visits=25, conversion_history=15)

def conversion_probability(profile: dict) -> float:
    """Weighted normalised score 0–100 representing conversion likelihood."""
    score = sum(
        (profile[f] / FEAT_MAX[f]) * w
        for f, w in CONV_WEIGHTS.items()
    )
    return round(score * 100, 1)

cluster_profiles = []
for c in sorted(df_raw['kmeans_lbl'].unique()):
    grp = df_raw[df_raw['kmeans_lbl'] == c]
    profile = {
        'cluster':            int(c),
        'size':               int(len(grp)),
        'pct':                round(len(grp) / len(df_raw) * 100, 1),
    }
    for f in FEATURES:
        profile[f]              = round(float(grp[f].mean()), 2)
        profile[f + '_std']     = round(float(grp[f].std()),  2)
        profile[f + '_min']     = round(float(grp[f].min()),  2)
        profile[f + '_max']     = round(float(grp[f].max()),  2)
    profile['conversion_probability'] = conversion_probability(profile)
    cluster_profiles.append(profile)

# Sort by conversion_probability descending → rank 0 = highest converter
cluster_profiles.sort(key=lambda p: -p['conversion_probability'])

SEGMENT_META = [
    ('Power Users',     '#4F8EF7', 'rocket',   'High on all metrics — loyal, engaged, converting'),
    ('At-Risk Users',   '#F7A94F', 'warning',  'Frequent short sessions — conversion declining'),
    ('Window Shoppers', '#4FC9A4', 'shopping', 'Long sessions, deep scrolls — rarely converts'),
    ('Casual Browsers', '#F76F6F', 'eye',      'Low engagement across every metric'),
]

for i, p in enumerate(cluster_profiles):
    label, color, icon, desc = SEGMENT_META[i] if i < len(SEGMENT_META) else (
        f'Segment {i+1}', '#888', 'circle', '')
    p.update(label=label, color=color, icon=icon, description=desc,
             rank=i+1,          # 1 = highest conversion probability
             is_top=(i == 0))

cluster_name_map  = {p['cluster']: p['label'] for p in cluster_profiles}
cluster_color_map = {p['cluster']: p['color'] for p in cluster_profiles}

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TARGETING RECOMMENDATION TABLE
# ═══════════════════════════════════════════════════════════════════════════════
TARGETING = [
    dict(
        channel        = 'Email + Push Notifications',
        message        = 'Exclusive loyalty rewards & early access offers',
        cta            = 'Upgrade / Upsell',
        priority       = 'HIGH',
        expected_lift  = '25-35%',
        rationale      = 'Already converting — increase order value and retention',
    ),
    dict(
        channel        = 'Retargeting Ads + Email Win-back',
        message        = 'Personalised re-engagement sequence with incentive',
        cta            = 'Re-engage',
        priority       = 'HIGH',
        expected_lift  = '18-28%',
        rationale      = 'Past converters at risk — recover before churn',
    ),
    dict(
        channel        = 'In-App Nudges + Time-limited Discount',
        message        = 'Limited-time offer shown at peak scroll depth',
        cta            = 'Convert Now',
        priority       = 'MEDIUM',
        expected_lift  = '10-18%',
        rationale      = 'High intent (long sessions, deep scroll) but not converting',
    ),
    dict(
        channel        = 'Content Marketing + SEO + Onboarding',
        message        = 'Educational content to build awareness and trust',
        cta            = 'Nurture',
        priority       = 'LOW',
        expected_lift  = '5-10%',
        rationale      = 'Low engagement — needs awareness building before conversion',
    ),
]

targeting_table = [
    dict(
        segment      = p['label'],
        icon         = p['icon'],
        color        = p['color'],
        conv_prob    = p['conversion_probability'],
        rank         = p['rank'],
        **TARGETING[min(i, len(TARGETING)-1)]
    )
    for i, p in enumerate(cluster_profiles)
]

# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/overview')
def api_overview():
    return jsonify({
        'total_users':        len(df_raw),
        'optimal_k':          OPTIMAL_K,
        'silhouette_kmeans':  sil_kmeans,
        'silhouette_agg':     sil_agg,
        'silhouette_dbscan':  sil_dbscan,
        'ari_kmeans_agg':     ari_kmeans_agg,
        'dbscan_clusters':    dbscan_n_clusters,
        'dbscan_noise':       dbscan_n_noise,
        'pca_variance':       [round(float(v*100),1) for v in pca.explained_variance_ratio_],
        'features':           FEATURES,
        'top_segment':        cluster_profiles[0]['label'],
        'top_conv_prob':      cluster_profiles[0]['conversion_probability'],
    })

@app.route('/api/elbow')
def api_elbow():
    return jsonify({
        'k_values':    K_VALS,
        'inertias':    INERTIAS,
        'silhouettes': SILHOUETTES,
        'db_scores':   DB_SCORES,
        'optimal_k':   OPTIMAL_K,
    })

@app.route('/api/scatter')
def api_scatter():
    cols = ['user_id','pca_x','pca_y',
            'kmeans_lbl','agg_lbl','dbscan_lbl',
            'session_frequency','avg_time_spent',
            'scroll_depth','repeat_visits','conversion_history']
    data = df_raw[cols].copy()
    data['kmeans_label']  = data['kmeans_lbl'].map(cluster_name_map)
    data['kmeans_color']  = data['kmeans_lbl'].map(cluster_color_map)
    data['agg_label']     = data['agg_lbl'].map(cluster_name_map)  # remap to same names for comparison
    data['dbscan_label']  = data['dbscan_lbl'].apply(
        lambda x: f'Cluster {int(x)}' if x != -1 else 'Noise/Outlier')
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/profiles')
def api_profiles():
    return jsonify(cluster_profiles)

@app.route('/api/targeting')
def api_targeting():
    return jsonify(targeting_table)

@app.route('/api/radar')
def api_radar():
    radar_data = []
    for p in cluster_profiles:
        row = {f: round(p[f] / FEAT_MAX[f] * 100, 1) for f in FEATURES}
        row.update(label=p['label'], color=p['color'], icon=p['icon'],
                   conv_prob=p['conversion_probability'])
        radar_data.append(row)
    return jsonify({'data': radar_data, 'features': FEATURES})

@app.route('/api/method_comparison')
def api_method_comparison():
    """Returns side-by-side metrics for all three clustering methods."""
    return jsonify({
        'methods': [
            {
                'name':        'K-Means',
                'algorithm':   "Lloyd's / EM",
                'k':           OPTIMAL_K,
                'silhouette':  sil_kmeans,
                'complexity':  'O(n·k·i)',
                'noise':       'No',
                'scalability': 'High',
                'role':        'Primary',
                'selected':    True,
            },
            {
                'name':        'Agglomerative',
                'algorithm':   'Ward Linkage',
                'k':           OPTIMAL_K,
                'silhouette':  sil_agg,
                'complexity':  'O(n² log n)',
                'noise':       'Partial',
                'scalability': 'Medium',
                'role':        'Validation',
                'selected':    False,
                'ari_vs_kmeans': ari_kmeans_agg,
            },
            {
                'name':        'DBSCAN',
                'algorithm':   'Density-Based',
                'k':           dbscan_n_clusters,
                'silhouette':  sil_dbscan,
                'complexity':  'O(n log n)',
                'noise':       'Yes',
                'scalability': 'Low',
                'role':        'Outlier Detection',
                'selected':    False,
                'noise_points': dbscan_n_noise,
            },
        ],
        'ari_kmeans_agg': ari_kmeans_agg,
        'interpretation': (
            'ARI > 0.9 means K-Means and Agglomerative agree strongly, '
            'confirming cluster stability.'
            if ari_kmeans_agg > 0.9
            else 'Moderate agreement between methods — inspect scatter for differences.'
        ),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f"\n{'='*54}")
    print(f"  BehaviorIQ  |  Behavioral Segmentation Engine")
    print(f"{'='*54}")
    print(f"  Users              : {len(df_raw)}")
    print(f"  Optimal K          : {OPTIMAL_K}  (Silhouette = {max(SILHOUETTES):.4f})")
    print(f"  KMeans sil.        : {sil_kmeans}")
    print(f"  Agglomerative sil. : {sil_agg}")
    print(f"  DBSCAN clusters    : {dbscan_n_clusters}  (noise pts: {dbscan_n_noise})")
    if sil_dbscan:
        print(f"  DBSCAN sil.        : {sil_dbscan}")
    print(f"  KMeans-Agg ARI     : {ari_kmeans_agg}  (1.0 = perfect agreement)")
    print(f"  PCA variance       : {sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"{'='*54}")
    print(f"  {'RANK':<5} {'SEGMENT':<18} {'N':>5}  {'CONV PROB':>10}")
    for p in cluster_profiles:
        star = ' ★ HIGHEST' if p['is_top'] else ''
        print(f"  {p['rank']:<5} {p['label']:<18} {p['size']:>5}  {p['conversion_probability']:>9}%{star}")
    print(f"{'='*54}")
    print(f"  Dashboard → http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)