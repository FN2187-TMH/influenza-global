"""
Influenza Global Surveillance Dashboard
========================================
Demo cho dự án MoMD: Climate → LSH → Mobility → Prediction

Cách chạy:
  1. Đặt các file parquet + flu_model.pkl vào thư mục data/
  2. pip install -r requirements.txt
  3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Influenza Surveillance",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
code, .stCode {
    font-family: 'JetBrains Mono', monospace;
}
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
}
.metric-card {
    background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 100%);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #2a3a4a;
    color: white;
}
.metric-card h3 {
    font-size: 0.85rem;
    color: #8899aa;
    margin: 0 0 0.3rem 0;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #e8f0fe;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #6688aa;
    margin-top: 0.3rem;
}
.tab-header {
    font-size: 1.1rem;
    color: #ccc;
    margin-bottom: 1rem;
    border-left: 3px solid #4ecdc4;
    padding-left: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    data = {}
    files = {
        'final':     'final_results.parquet',
        'models':    'model_results.parquet',
        'graph':     'graph_metrics.parquet',
        'lsh':       'climate_similarity_pairs.parquet',
        'risk_ts':   'risk_scores_timeseries.parquet',
        'lsh_val':   'lsh_validation.parquet',
    }
    for key, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data[key] = pd.read_parquet(path)
        else:
            data[key] = None
    return data

data = load_data()

# Check required files
if data['final'] is None:
    st.error(f"⚠️ Không tìm thấy `{DATA_DIR}/final_results.parquet`. "
             f"Hãy đặt các file output vào thư mục `{DATA_DIR}/`.")
    st.stop()

df = data['final'].copy()
df['year'] = pd.to_numeric(df.get('year'), errors='coerce')
df['week'] = pd.to_numeric(df.get('week'), errors='coerce')

# Load model if available
model_bundle = None
model_path = os.path.join(DATA_DIR, 'flu_model.pkl')
if os.path.exists(model_path):
    import joblib
    model_bundle = joblib.load(model_path)


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
def metric_card(title, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def find_col(cols, keys):
    """Tìm tên cột khớp (ưu tiên khớp chính xác, sau đó khớp chứa) — không phân biệt hoa thường."""
    low = {c.lower(): c for c in cols}
    for k in keys:
        if k in low:
            return low[k]
    for c in cols:
        if any(k in c.lower() for k in keys):
            return c
    return None


# --- Model family classification (dùng cho trang Model Performance) ---
FAMILY_COLORS = {
    'A3T-GCN':  '#a78bfa',   # tím — mô hình GNN mới
    'Ridge':    '#4ecdc4',
    'GBM':      '#ffa94d',
    'JL':       '#74c0fc',
    'Baseline': '#556677',
    'Other':    '#868e96',
}
FAMILY_LABEL = {
    'A3T-GCN':  'A3T-GCN (spatio-temporal GNN)',
    'Ridge':    'Ridge (linear)',
    'GBM':      'GBM / XGBoost (tree)',
    'JL':       'JL projection',
    'Baseline': 'Baseline (climate only)',
    'Other':    'Other',
}

def model_family(name):
    n = str(name).lower()
    if any(k in n for k in ('a3t', 'gnn', 'gcn')): return 'A3T-GCN'
    if 'baseline' in n:                            return 'Baseline'
    if 'ridge' in n:                               return 'Ridge'
    if any(k in n for k in ('gbm', 'xgb', 'boost')): return 'GBM'
    if 'jl' in n:                                  return 'JL'
    return 'Other'

def pred_label(col):
    cl = str(col).lower()
    if col == 'predicted_cases':                   return 'Predicted (deployed)'
    if any(k in cl for k in ('a3t', 'gnn', 'gcn')): return 'A3T-GCN'
    if any(k in cl for k in ('gbm', 'xgb', 'boost')): return 'GBM / XGBoost'
    if 'ridge' in cl:                              return 'Ridge'
    return col

def pred_color(col):
    cl = str(col).lower()
    if any(k in cl for k in ('a3t', 'gnn', 'gcn')): return '#a78bfa'
    if any(k in cl for k in ('gbm', 'xgb', 'boost')): return '#ffa94d'
    if 'ridge' in cl:                              return '#4ecdc4'
    return '#ff6b6b'


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌍 FluScope")
    st.caption("Global Influenza Surveillance Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🌡️ Climate → Flu Lag",
            "✈️ Epidemic Hub Ranking",
            "🗺️ Risk Score Map",
            "🔗 Regional Communities",
            "🤖 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    years = sorted(df['year'].dropna().unique().astype(int))
    sel_years = st.multiselect("Filter years", years, default=years)
    df_filtered = df[df['year'].isin(sel_years)]

    st.divider()
    st.caption(f"Data: {len(df_filtered):,} rows · {df_filtered['country_iso3'].nunique()} countries")


# ══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 🌍 FluScope — Global Influenza Surveillance")
    st.markdown('<p class="tab-header">Climate anomaly → LSH similarity → Mobility network → Prediction model</p>',
                unsafe_allow_html=True)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Countries", f"{df['country_iso3'].nunique()}", "monitored globally")
    with c2:
        metric_card("Years", f"{int(df['year'].min())}–{int(df['year'].max())}",
                     f"{len(years)} years of data")
    with c3:
        if data['models'] is not None:
            _r2 = find_col(data['models'].columns, ['r2', 'r²', 'rsquared'])
            _mc = find_col(data['models'].columns, ['model', 'model_name', 'name']) or data['models'].columns[0]
            if _r2:
                best = data['models'].loc[data['models'][_r2].idxmax()]
                metric_card("Best R²", f"{best[_r2]:.3f}", f"{best[_mc]}")
            else:
                metric_card("Best R²", "N/A", "")
        else:
            metric_card("Best R²", "N/A", "")
    with c4:
        if 'risk_level' in df.columns:
            n_high = (df_filtered['risk_level'] == 'High').sum()
            metric_card("High Risk", f"{n_high:,}", "country-week observations")
        else:
            metric_card("Records", f"{len(df):,}", "total observations")

    st.markdown("---")

    # Global flu trend
    weekly = (df_filtered.groupby(['year','week'])
              .agg(total_actual=('actual_cases','sum'),
                   total_predicted=('predicted_cases','sum'))
              .reset_index())
    weekly['week_label'] = weekly['year'].astype(int).astype(str) + '-W' + weekly['week'].astype(int).astype(str).str.zfill(2)
    weekly = weekly.sort_values(['year','week'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly['week_label'], y=weekly['total_actual'],
                             mode='lines', name='Actual', line=dict(color='#4ecdc4', width=2),
                             fill='tozeroy', fillcolor='rgba(78,205,196,0.1)'))
    fig.add_trace(go.Scatter(x=weekly['week_label'], y=weekly['total_predicted'],
                             mode='lines', name='Predicted (t+2)', line=dict(color='#ff6b6b', width=2, dash='dot')))
    fig.update_layout(
        title='Global Influenza Trend — Actual vs Predicted',
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,25,35,0.8)',
        height=400, margin=dict(t=50, b=40),
        xaxis=dict(showgrid=False, dtick=26),
        yaxis=dict(title='Total flu cases', gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation='h', y=1.12)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pipeline diagram
    st.markdown("### Pipeline")
    cols = st.columns(5)
    steps = [
        ("🌡️", "ERA5 Climate", "MapReduce + cos(lat)"),
        ("🔍", "MinHash LSH", "Weather similarity"),
        ("✈️", "Mobility Network", "PageRank + Risk"),
        ("📈", "Prediction", "Ridge · GBM · A3T-GCN"),
        ("🗺️", "Dashboard", "You are here"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; background:rgba(15,25,35,0.8);
                        border-radius:10px; border:1px solid #2a3a4a;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:700; margin:0.5rem 0; color:#e8f0fe;">{title}</div>
                <div style="font-size:0.8rem; color:#6688aa;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: CLIMATE → FLU LAG
# ══════════════════════════════════════════════════════════════════
elif page == "🌡️ Climate → Flu Lag":
    st.markdown("# 🌡️ Quantified Lag Effect")
    st.markdown('<p class="tab-header">Cross-correlation giữa Climate Anomaly và Flu Outbreak ở các lag khác nhau</p>',
                unsafe_allow_html=True)

    anomaly_cols = [c for c in df_filtered.columns if '_anomaly' in c and '_lag' not in c]
    flu_col = 'flu_cases_total' if 'flu_cases_total' in df_filtered.columns else 'actual_cases'

    if anomaly_cols:
        lags = [0, 1, 2, 3, 4, 6, 8]
        corr_data = []
        for var in anomaly_cols:
            for lag in lags:
                shifted = df_filtered.groupby('country_iso3').apply(
                    lambda g: g[[var, flu_col]].assign(flu_s=g[flu_col].shift(-lag)),
                ).reset_index(drop=True).dropna()
                r = shifted[var].corr(shifted['flu_s']) if len(shifted) > 30 else np.nan
                corr_data.append({'variable': var.replace('_anomaly',''), 'lag': lag, 'corr': r})

        corr_df = pd.DataFrame(corr_data)
        pivot = corr_df.pivot(index='variable', columns='lag', values='corr')

        # Heatmap
        fig = px.imshow(pivot, text_auto='.3f', color_continuous_scale='RdBu_r',
                        zmin=-0.3, zmax=0.3, aspect='auto',
                        labels=dict(x='Lag (weeks)', y='Climate Variable', color='Correlation'))
        fig.update_layout(title='Cross-Correlation Heatmap: Climate Anomaly → Flu Cases',
                          template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(15,25,35,0.8)', height=350)
        # Highlight lag=2
        fig.add_shape(type='rect', x0=1.5, x1=2.5, y0=-0.5, y1=len(pivot)-0.5,
                      line=dict(color='lime', width=3))
        st.plotly_chart(fig, use_container_width=True)

        # Line chart
        fig2 = px.line(corr_df, x='lag', y='corr', color='variable', markers=True,
                       labels={'lag':'Lag (weeks)', 'corr':'Pearson Correlation', 'variable':'Climate Variable'})
        fig2.add_vline(x=2, line_dash='dash', line_color='lime', annotation_text='lag=2')
        fig2.add_hline(y=0, line_dash='dot', line_color='gray')
        fig2.update_layout(title='Correlation vs Lag — Why lag=2?',
                           template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(15,25,35,0.8)', height=350)
        st.plotly_chart(fig2, use_container_width=True)

        # Best lag per variable
        st.markdown("### Optimal Lag per Variable")
        for var in pivot.index:
            vals = pivot.loc[var].values.astype(float)
            best = int(pivot.columns[np.nanargmax(np.abs(vals))])
            r = vals[np.nanargmax(np.abs(vals))]
            st.markdown(f"- **{var}**: best lag = `t+{best}` (r = {r:.4f})")
    else:
        st.warning("No anomaly columns found in data")


# ══════════════════════════════════════════════════════════════════
# PAGE: EPIDEMIC HUB RANKING
# ══════════════════════════════════════════════════════════════════
elif page == "✈️ Epidemic Hub Ranking":
    st.markdown("# ✈️ Epidemic Hub Ranking")
    st.markdown('<p class="tab-header">PageRank + Topic-Sensitive PageRank trên mobility network</p>',
                unsafe_allow_html=True)

    if data['graph'] is not None:
        gm = data['graph'].copy()

        c1, c2 = st.columns(2)

        # Standard PageRank
        with c1:
            if 'pagerank_score' in gm.columns:
                top = gm.nlargest(25, 'pagerank_score')
                fig = px.bar(top, y='country_iso3', x='pagerank_score', orientation='h',
                             color='pagerank_score', color_continuous_scale='Reds',
                             title='Standard PageRank — Hub Connectivity')
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(15,25,35,0.8)', height=600,
                                  yaxis=dict(autorange='reversed'), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Topic-Sensitive PageRank
        with c2:
            if 'tspr_score' in gm.columns:
                top = gm.nlargest(25, 'tspr_score')
                fig = px.bar(top, y='country_iso3', x='tspr_score', orientation='h',
                             color='tspr_score', color_continuous_scale='Oranges',
                             title='Topic-Sensitive PageRank — Flu-Biased')
                fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(15,25,35,0.8)', height=600,
                                  yaxis=dict(autorange='reversed'), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        if 'pagerank_score' in gm.columns and 'tspr_score' in gm.columns:
            st.markdown("### Rank Comparison")
            comp = gm[['country_iso3','pagerank_score','tspr_score']].dropna().copy()
            comp['pr_rank'] = comp['pagerank_score'].rank(ascending=False).astype(int)
            comp['tspr_rank'] = comp['tspr_score'].rank(ascending=False).astype(int)
            comp['rank_shift'] = comp['pr_rank'] - comp['tspr_rank']
            st.dataframe(comp.nlargest(20, 'pagerank_score')
                         [['country_iso3','pr_rank','tspr_rank','rank_shift','pagerank_score','tspr_score']]
                         .reset_index(drop=True), use_container_width=True)
    else:
        st.warning("graph_metrics.parquet not found")


# ══════════════════════════════════════════════════════════════════
# PAGE: RISK SCORE MAP
# ══════════════════════════════════════════════════════════════════
elif page == "🗺️ Risk Score Map":
    st.markdown("# 🗺️ Mobility-based Risk Score")
    st.markdown('<p class="tab-header">Risk = Σ(flu_neighbor × route_weight), scaled 0-100</p>',
                unsafe_allow_html=True)

    has_risk = 'risk_score' in df_filtered.columns

    if has_risk:
        # Week selector
        week_options = sorted(df_filtered['week_id'].dropna().unique()) if 'week_id' in df_filtered.columns else []
        if week_options:
            sel_week = st.select_slider("Select week", options=week_options, value=week_options[-1])
            week_data = df_filtered[df_filtered['week_id'] == sel_week]
        else:
            week_data = df_filtered.groupby('country_iso3').last().reset_index()

        # World map
        fig = px.choropleth(
            week_data, locations='country_iso3',
            color='risk_score', hover_name='country_iso3',
            hover_data=['actual_cases', 'predicted_cases', 'risk_score'],
            color_continuous_scale='YlOrRd',
            range_color=[0, 100],
            title=f'Global Risk Score — {sel_week if week_options else "Latest"}',
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(15,25,35,0.8)', lakecolor='rgba(15,25,35,0.8)',
                     landcolor='rgba(30,40,55,1)', showframe=False),
            height=500, margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top risk countries for selected week
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Top 15 High Risk")
            top15 = week_data.nlargest(15, 'risk_score')[['country_iso3','risk_score','actual_cases','predicted_cases']]
            st.dataframe(top15.reset_index(drop=True), use_container_width=True)

        with c2:
            st.markdown("### Risk Distribution")
            fig2 = px.histogram(week_data, x='risk_score', nbins=30,
                                color_discrete_sequence=['#ff6b6b'])
            fig2.add_vline(x=33, line_dash='dash', line_color='#f39c12', annotation_text='Medium')
            fig2.add_vline(x=66, line_dash='dash', line_color='#e74c3c', annotation_text='High')
            fig2.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(15,25,35,0.8)', height=350)
            st.plotly_chart(fig2, use_container_width=True)

    elif 'predicted_cases' in df_filtered.columns:
        # Fallback: show predicted cases on map
        latest = df_filtered.groupby('country_iso3').last().reset_index()
        fig = px.choropleth(latest, locations='country_iso3',
                            color='predicted_cases', hover_name='country_iso3',
                            color_continuous_scale='YlOrRd',
                            title='Predicted Flu Cases (latest week)')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          geo=dict(bgcolor='rgba(15,25,35,0.8)'), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No risk_score or predicted_cases in data")


# ══════════════════════════════════════════════════════════════════
# PAGE: REGIONAL COMMUNITIES
# ══════════════════════════════════════════════════════════════════
elif page == "🔗 Regional Communities":
    st.markdown("# 🔗 Regional Outbreak Communities")
    st.markdown('<p class="tab-header">Louvain community detection + LSH weather similarity validation</p>',
                unsafe_allow_html=True)

    if data['graph'] is not None and 'community_id' in data['graph'].columns:
        gm = data['graph'].copy()

        # Community map
        fig = px.choropleth(
            gm, locations='country_iso3', color='community_id',
            hover_name='country_iso3',
            color_continuous_scale='Rainbow',
            title='Mobility Network Communities (Louvain)',
        )
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            geo=dict(bgcolor='rgba(15,25,35,0.8)', landcolor='rgba(30,40,55,1)', showframe=False),
            height=450, margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Community breakdown
        st.markdown("### Communities")
        for cid in sorted(gm['community_id'].unique()):
            members = gm[gm['community_id'] == cid]['country_iso3'].tolist()
            with st.expander(f"Community {int(cid)} — {len(members)} countries"):
                st.write(", ".join(sorted(members)))
    else:
        st.warning("No community data found")

    # LSH Validation
    st.markdown("---")
    st.markdown("### LSH Validation: Weather Similarity → Flu Pattern?")

    if data['lsh_val'] is not None:
        lsh = data['lsh_val']

        c1, c2 = st.columns(2)
        with c1:
            mean_corr = lsh['flu_correlation'].mean()
            metric_card("Mean Flu Correlation", f"{mean_corr:.3f}", "LSH-similar country pairs")
        with c2:
            pct_pos = (lsh['flu_correlation'] > 0).mean()
            metric_card("% Positive Correlation", f"{pct_pos:.1%}", "pairs with similar flu patterns")

        fig = px.histogram(lsh, x='flu_correlation', nbins=40,
                           color_discrete_sequence=['#4ecdc4'],
                           title='Flu Correlation Distribution — LSH-Similar Country Pairs')
        fig.add_vline(x=0, line_dash='dash', line_color='white')
        fig.add_vline(x=mean_corr, line_dash='solid', line_color='#ff6b6b',
                      annotation_text=f'Mean={mean_corr:.3f}')
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(15,25,35,0.8)', height=350)
        st.plotly_chart(fig, use_container_width=True)

    if data['lsh'] is not None:
        lsh_pairs = data['lsh']
        st.markdown(f"**{len(lsh_pairs):,} country pairs** identified as weather-similar by MinHash LSH")
        st.dataframe(lsh_pairs.nlargest(20, 'jaccard_sim').reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE  (đã bổ sung A3T-GCN)
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Prediction & Model Performance")
    st.markdown('<p class="tab-header">Baseline (climate only) · Enhanced (climate + mobility + risk) · '
                'A3T-GCN (spatio-temporal GNN)</p>', unsafe_allow_html=True)

    # ───────────────────────── Model comparison ─────────────────────────
    if data['models'] is not None:
        mr = data['models'].copy()
        model_col = find_col(mr.columns, ['model', 'model_name', 'name']) or mr.columns[0]
        mr['family'] = mr[model_col].map(model_family)

        # sort by R² desc nếu có
        r2_col = find_col(mr.columns, ['r2', 'r²', 'rsquared'])
        if r2_col:
            mr = mr.sort_values(r2_col, ascending=False).reset_index(drop=True)

        # Banner cho A3T-GCN
        has_gnn = (mr['family'] == 'A3T-GCN').any()
        if has_gnn:
            grow = mr[mr['family'] == 'A3T-GCN'].iloc[0]
            extra = f" · R² = {grow[r2_col]:.3f}" if r2_col else ""
            st.markdown(
                f"<div style='background:rgba(167,139,250,0.12);border:1px solid #a78bfa;"
                f"border-radius:10px;padding:0.7rem 1rem;color:#cbb3ff;'>"
                f"🧠 <b>A3T-GCN</b> đã được đưa vào so sánh — mạng nơ-ron đồ thị thời gian "
                f"chạy trên mobility network{extra}.</div>", unsafe_allow_html=True)
        else:
            st.info("Chưa thấy model **A3T-GCN** trong `model_results.parquet`. "
                    "Thêm một dòng có `model='gnn_a3tgcn'` kèm các metric (R2, MAE, F1, RMSE) "
                    "để nó hiện ở đây — phần code đã sẵn sàng nhận diện.")

        # Metric bar charts (tự dò metric có sẵn, tô màu theo nhóm model)
        st.markdown("### Model Comparison")
        metric_specs, seen = [], set()
        for keys, higher, label in [(['r2', 'r²'], True, 'R²'),
                                     (['test rmse', 'test_rmse', 'rmse'], False, 'RMSE'),
                                     (['mae'], False, 'MAE'),
                                     (['f1'], True, 'F1')]:
            col = find_col(mr.columns, keys)
            if col and col not in seen:
                seen.add(col)
                metric_specs.append((col, higher, label))

        if metric_specs:
            chart_cols = st.columns(len(metric_specs))
            for cwidget, (mcol, higher, label) in zip(chart_cols, metric_specs):
                with cwidget:
                    fig = px.bar(mr, x=model_col, y=mcol, color='family',
                                 color_discrete_map=FAMILY_COLORS,
                                 category_orders={model_col: mr[model_col].tolist()},
                                 title=f'{label} ({"↑" if higher else "↓"} better)')
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(15,25,35,0.8)', height=320,
                                      showlegend=False, xaxis_tickangle=-40, xaxis_title=None,
                                      margin=dict(t=46, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        # Improvement highlight (robust)
        if r2_col:
            base = mr.loc[mr['family'] == 'Baseline', r2_col]
            enh = mr.loc[mr[model_col].astype(str).str.contains('enhanced', case=False)
                         & (mr['family'] != 'Baseline'), r2_col]
            if len(base) and len(enh):
                improvement = enh.max() - base.max()
                st.success(f"🚀 Mobility network **improves R² by {improvement:+.4f}** "
                           f"(Baseline {base.max():.4f} → Enhanced {enh.max():.4f})")

        # Leaderboard (bảng có tô đậm ô tốt nhất mỗi metric)
        st.markdown("### Leaderboard")
        metric_cols = [c for c in mr.columns
                       if c not in (model_col, 'family') and pd.api.types.is_numeric_dtype(mr[c])]
        show = mr[[model_col, 'family'] + metric_cols].copy()
        show['family'] = show['family'].map(FAMILY_LABEL).fillna(show['family'])
        show = show.rename(columns={model_col: 'model', 'family': 'type'})

        higher_cols = [c for c in metric_cols if any(k in c.lower() for k in ('r2', 'r²', 'f1', 'auc', 'acc'))]
        lower_cols  = [c for c in metric_cols if any(k in c.lower() for k in ('mae', 'rmse', 'mse', 'error'))]
        fmt = {c: ('{:.1f}' if c in lower_cols else '{:.3f}') for c in metric_cols}

        try:
            sty = show.style.format(fmt)
            if higher_cols:
                sty = sty.highlight_max(subset=higher_cols, color='#2f9e7d')
            if lower_cols:
                sty = sty.highlight_min(subset=lower_cols, color='#2f9e7d')
            st.dataframe(sty, use_container_width=True)
        except Exception:
            st.dataframe(show, use_container_width=True)
        st.caption("Ô xanh = tốt nhất ở mỗi cột. Mô hình triển khai (deployed) thường chọn theo Val RMSE thấp nhất.")

        # A3T-GCN spotlight
        with st.expander("ℹ️ Về mô hình A3T-GCN (Attention Temporal Graph Convolutional Network)"):
            st.markdown("""
- **Kiến trúc:** GNN không gian–thời gian chạy trên *mobility network* — nút là quốc gia, cạnh là tuyến bay với trọng số `log1p(route_count)`.
- **Đầu vào:** chuỗi tensor 3D `(T, N, F)` gồm đặc trưng động (climate lags, flu cases, mobility) + đặc trưng tĩnh (PageRank, centrality) + sin/cos tuần-trong-năm; thêm **observed-mask channel** cho các tuần thiếu dữ liệu.
- **Huấn luyện:** cửa sổ look-back `L = 8` tuần, dự báo `t + 2`; **Huber loss** (δ = 1.0), Adam (lr = 1e-2), PyTorch Geometric Temporal.
- **Vai trò:** baseline spatio-temporal có cơ sở. Trên bộ dữ liệu hiện tại nó xếp sau các mô hình bảng do dữ liệu thưa và đồ thị tĩnh không bám kịp cú sụp đổ di chuyển hàng không 2020 — nhưng đặt nền cho hướng mở rộng khi có đồ thị động và dữ liệu báo cáo dày hơn.
            """)
    else:
        st.warning("model_results.parquet not found")

    # ───────────────────────── Per-country prediction ─────────────────────────
    st.markdown("---")
    st.markdown("### Per-Country Prediction")

    countries = sorted(df_filtered['country_iso3'].unique())
    sel_country = st.selectbox("Select country", countries,
                                index=countries.index('USA') if 'USA' in countries else 0)

    cdf = df_filtered[df_filtered['country_iso3'] == sel_country].sort_values(['year','week'])

    # dò các cột prediction (hỗ trợ nhiều model nếu parquet có, vd: pred_gnn_a3tgcn, pred_ridge…)
    pred_candidates = [c for c in cdf.columns if 'pred' in c.lower() and 'actual' not in c.lower()]

    if len(cdf) > 0 and 'actual_cases' in cdf.columns and pred_candidates:
        cdf_plot = cdf.copy()
        if 'week_id' not in cdf_plot.columns:
            cdf_plot['week_id'] = (cdf_plot['year'].astype(int).astype(str) + '-W'
                                   + cdf_plot['week'].astype(int).astype(str).str.zfill(2))

        default_pred = 'predicted_cases' if 'predicted_cases' in pred_candidates else pred_candidates[0]
        if len(pred_candidates) > 1:
            chosen = st.multiselect("Prediction series to overlay", pred_candidates,
                                    default=[default_pred], format_func=pred_label)
        else:
            chosen = [default_pred]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cdf_plot['week_id'], y=cdf_plot['actual_cases'],
                                 mode='lines', name='Actual', line=dict(color='#4ecdc4', width=2),
                                 fill='tozeroy', fillcolor='rgba(78,205,196,0.1)'))
        for pc in chosen:
            fig.add_trace(go.Scatter(x=cdf_plot['week_id'], y=cdf_plot[pc],
                                     mode='lines', name=pred_label(pc),
                                     line=dict(color=pred_color(pc), width=2, dash='dot')))
        fig.update_layout(
            title=f'{sel_country} — Actual vs Predicted Flu Cases',
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,25,35,0.8)', height=400,
            xaxis=dict(showgrid=False, dtick=13),
            yaxis=dict(title='Flu cases', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Country metrics — tính trên series prediction mặc định
        metric_pred = 'predicted_cases' if 'predicted_cases' in cdf.columns else default_pred
        cdf_valid = cdf.dropna(subset=['actual_cases', metric_pred])
        if len(cdf_valid) > 1:
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(cdf_valid['actual_cases'], cdf_valid[metric_pred])
            mae = mean_absolute_error(cdf_valid['actual_cases'], cdf_valid[metric_pred])
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("R²", f"{r2:.4f}", f"{sel_country} · {pred_label(metric_pred)}")
            with c2: metric_card("MAE", f"{mae:.0f}", "cases")
            with c3: metric_card("Weeks", f"{len(cdf_valid)}", "predicted")
        elif len(cdf_valid) == 0:
            st.info(f"Không có prediction cho {sel_country} (chỉ có data train 2001-2020)")


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#556677; font-size:0.8rem; padding:1rem 0;">
    MoMD Project — Influenza Surveillance Dashboard<br>
    Algorithms: MapReduce · MinHash LSH · PageRank · Streaming · Gradient Boosting · A3T-GCN
</div>
""", unsafe_allow_html=True)