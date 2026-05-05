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
# HELPER: metric card
# ══════════════════════════════════════════════════════════════════
def metric_card(title, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


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
            best = data['models'].loc[data['models']['R2'].idxmax()]
            metric_card("Best R²", f"{best['R2']:.3f}", f"{best['model']}")
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
        ("📈", "Prediction", "XGBoost / GBM"),
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
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Prediction & Model Performance")
    st.markdown('<p class="tab-header">Baseline (climate only) vs Enhanced (climate + mobility + risk)</p>',
                unsafe_allow_html=True)

    # Model comparison
    if data['models'] is not None:
        mr = data['models'].copy()

        # Metrics bar charts
        st.markdown("### Model Comparison")
        c1, c2, c3 = st.columns(3)

        for col_widget, metric, higher_better in [(c1, 'R2', True), (c2, 'MAE', False), (c3, 'F1', True)]:
            with col_widget:
                if metric in mr.columns:
                    colors = ['#4ecdc4' if 'enhanced' in m else '#556677' for m in mr['model']]
                    fig = px.bar(mr, x='model', y=metric, color='model',
                                 color_discrete_sequence=colors,
                                 title=f'{metric} ({"↑" if higher_better else "↓"} better)')
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(15,25,35,0.8)', height=300,
                                      showlegend=False, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

        # Improvement highlight
        if 'R2' in mr.columns:
            baseline_r2 = mr[mr['model'].str.contains('baseline')]['R2'].max()
            enhanced_r2 = mr[mr['model'].str.contains('enhanced') & ~mr['model'].str.contains('jl')]['R2'].max()
            if pd.notna(baseline_r2) and pd.notna(enhanced_r2):
                improvement = enhanced_r2 - baseline_r2
                st.success(f"🚀 Mobility network **improves R² by {improvement:+.4f}** "
                          f"(Baseline: {baseline_r2:.4f} → Enhanced: {enhanced_r2:.4f})")

        # Full table
        st.dataframe(mr, use_container_width=True)

    # Per-country predictions
    st.markdown("---")
    st.markdown("### Per-Country Prediction")

    countries = sorted(df_filtered['country_iso3'].unique())
    sel_country = st.selectbox("Select country", countries,
                                index=countries.index('USA') if 'USA' in countries else 0)

    cdf = df_filtered[df_filtered['country_iso3'] == sel_country].sort_values(['year','week'])

    if len(cdf) > 0 and 'actual_cases' in cdf.columns and 'predicted_cases' in cdf.columns:
        cdf_plot = cdf.copy()
        if 'week_id' not in cdf_plot.columns:
            cdf_plot['week_id'] = cdf_plot['year'].astype(int).astype(str) + '-W' + cdf_plot['week'].astype(int).astype(str).str.zfill(2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cdf_plot['week_id'], y=cdf_plot['actual_cases'],
                                 mode='lines', name='Actual', line=dict(color='#4ecdc4', width=2),
                                 fill='tozeroy', fillcolor='rgba(78,205,196,0.1)'))
        fig.add_trace(go.Scatter(x=cdf_plot['week_id'], y=cdf_plot['predicted_cases'],
                                 mode='lines', name='Predicted (t+2)',
                                 line=dict(color='#ff6b6b', width=2, dash='dot')))
        fig.update_layout(
            title=f'{sel_country} — Actual vs Predicted Flu Cases',
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,25,35,0.8)', height=400,
            xaxis=dict(showgrid=False, dtick=13),
            yaxis=dict(title='Flu cases', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Country metrics
        cdf_valid = cdf.dropna(subset=['actual_cases', 'predicted_cases'])
        if len(cdf_valid) > 1:
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(cdf_valid['actual_cases'], cdf_valid['predicted_cases'])
            mae = mean_absolute_error(cdf_valid['actual_cases'], cdf_valid['predicted_cases'])
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("R²", f"{r2:.4f}", sel_country)
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
    Algorithms: MapReduce · MinHash LSH · PageRank · Streaming · Gradient Boosting
</div>
""", unsafe_allow_html=True)
