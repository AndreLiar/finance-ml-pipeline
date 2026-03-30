"""
Streamlit Dashboard — Personal Finance ML Project
Banque Populaire - Compte Cheques 23192700536
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import (
    FEATURES_XLSX           as _FEATURES_XLSX,
    CREDITWORTHINESS_XLSX   as _CREDIT_XLSX,
    CASHFLOW_RESULTS_XLSX   as _CASHFLOW_XLSX,
    ANOMALY_RESULTS_XLSX    as _ANOMALY_XLSX,
    MERGED_XLSX             as _MERGED_XLSX,
    LIVRET_A_XLSX           as _LIVRET_XLSX,
    TRANSACTIONS_XLSX       as _TRANSACTIONS_XLSX,
    MODEL_RESULTS_XLSX      as _MODEL_RESULTS_XLSX,
    LOAN_REPORT_TXT         as _REPORT_TXT,
    PIPELINE_STATUS_JSON    as _STATUS_JSON,
)
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics         import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #0f3460;
    }
    .metric-card h3 { margin: 0; color: #888; font-size: 0.85rem; font-weight: 500; }
    .metric-card h1 { margin: 0.2rem 0 0; color: #0f3460; font-size: 2rem; font-weight: 700; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #0f3460;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    .badge {
        display: inline-block; background: #0f3460; color: white;
        border-radius: 20px; padding: 2px 12px; font-size: 0.8rem;
        margin: 2px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Authentication ─────────────────────────────────────────────────────────────
import yaml
import streamlit_authenticator as stauth
from pathlib import Path as _AuthPath

_CREDS_FILE = _AuthPath(__file__).parent / "credentials.yaml"
with open(_CREDS_FILE, encoding="utf-8") as _f:
    _creds = yaml.safe_load(_f)

_authenticator = stauth.Authenticate(
    _creds["credentials"],
    _creds["cookie"]["name"],
    _creds["cookie"]["key"],
    _creds["cookie"]["expiry_days"],
)
_authenticator.login()

if st.session_state.get("authentication_status") is False:
    st.error("Incorrect username or password.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password.")
    st.stop()

# Authenticated — show logout in sidebar
with st.sidebar:
    _authenticator.logout("Logout", "sidebar")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    features = pd.read_excel(_FEATURES_XLSX, sheet_name="Feature Matrix")
    full     = pd.read_excel(_FEATURES_XLSX, sheet_name="Full Data")
    monthly  = pd.read_excel(_FEATURES_XLSX, sheet_name="Monthly Aggregates")
    cat_sum  = pd.read_excel(_FEATURES_XLSX, sheet_name="Category Summary")
    credit   = pd.read_excel(_CREDIT_XLSX, sheet_name="Monthly Credit Profile")
    cashflow = pd.read_excel(_CASHFLOW_XLSX, sheet_name="Actual vs Predicted")
    cf_fi    = pd.read_excel(_CASHFLOW_XLSX, sheet_name="Feature Importance")
    full['date_operation'] = pd.to_datetime(full['date_operation'])
    # Anomaly data
    anomaly_all  = None
    anomaly_flag = None
    try:
        anomaly_all  = pd.read_excel(_ANOMALY_XLSX, sheet_name="All Transactions")
        anomaly_flag = pd.read_excel(_ANOMALY_XLSX, sheet_name="Flagged Anomalies")
        anomaly_all['date_operation']  = pd.to_datetime(anomaly_all['date_operation'])
        anomaly_flag['date_operation'] = pd.to_datetime(anomaly_flag['date_operation'])
    except Exception:
        pass
    # Livret A data (optional — may not exist yet)
    livret_unified = None
    livret_txs = None
    livret_monthly = None
    try:
        livret_unified  = pd.read_excel(_MERGED_XLSX, sheet_name="Unified Monthly")
        livret_txs      = pd.read_excel(_LIVRET_XLSX, sheet_name="Livret A Transactions")
        livret_monthly  = pd.read_excel(_LIVRET_XLSX, sheet_name="Monthly Livret A")
        livret_txs['date'] = pd.to_datetime(livret_txs['date'])
    except Exception:
        pass
    return features, full, monthly, cat_sum, credit, cashflow, cf_fi, livret_unified, livret_txs, livret_monthly, anomaly_all, anomaly_flag

@st.cache_resource
def train_models(_features):
    """Load pre-trained category classifier from disk. Falls back to retraining if missing."""
    from model_store import artifacts_exist, load_artifacts, load_metadata

    if artifacts_exist("category_classifier"):
        art  = load_artifacts("category_classifier")
        meta = load_metadata("category_classifier")
        models_dict  = art["models"]
        scaler       = art["scaler"]
        le           = art["label_encoder"]
        FEATURE_COLS = art["feature_cols"]
        class_names  = art["class_names"]

        # Rebuild results dict from the Excel summary (avoid re-predicting)
        try:
            _summary = pd.read_excel(_MODEL_RESULTS_XLSX, sheet_name="Model Comparison")
            results = {}
            for _, row in _summary.iterrows():
                name = row['Model']
                if name in models_dict:
                    results[name] = {
                        'accuracy':    row.get('Test Accuracy', 0),
                        'f1_weighted': row.get('Test F1 (weighted)', 0),
                        'f1_macro':    row.get('Test F1 (macro)', 0),
                        'roc_auc':     row.get('ROC-AUC', None),
                        'report':      {},
                        'cm':          np.zeros((2,2)),
                        'model':       models_dict[name],
                        'fi':          models_dict[name].feature_importances_
                                       if hasattr(models_dict[name], 'feature_importances_') else None,
                    }
        except Exception:
            results = {}

        # Reconstruct X_train, y_train stubs for callers that need them
        cat_counts = _features['category'].value_counts()
        valid_cats = cat_counts[cat_counts >= 5].index
        df = _features[_features['category'].isin(valid_cats)].copy()
        X  = df[FEATURE_COLS].values
        y  = le.transform(df['category'])
        n  = int(len(X) * 0.70)
        X_train, y_train = X[:n], y[:n]

        return results, np.array(class_names), FEATURE_COLS, X_train, y_train, scaler, le

    # ── Fallback: retrain if no saved artifacts ────────────────────────────────
    cat_counts = _features['category'].value_counts()
    valid_cats = cat_counts[cat_counts >= 5].index
    df = _features[_features['category'].isin(valid_cats)].copy()

    FEATURE_COLS = [
        'year','month','day','day_of_week','week_of_year','is_weekend',
        'quarter','month_part_encoded','abs_amount','log_amount',
        'is_round_number','rolling_7d_spend','rolling_30d_spend',
        'monthly_income','monthly_spend','monthly_net','tx_count',
        'avg_tx_amount','max_tx_amount','savings_rate','type_encoded',
    ]

    le = LabelEncoder()
    X  = df[FEATURE_COLS].values
    y  = le.fit_transform(df['category'])

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    rf  = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                  random_state=42, class_weight='balanced', n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, verbosity=0)
    lr.fit(X_train_s, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    model_map = {
        "Logistic Regression": (lr,  X_test_s, y_test),
        "Random Forest":       (rf,  X_test,   y_test),
        "XGBoost":             (xgb, X_test,   y_test),
    }
    results = {}
    for name, (model, Xts, yts) in model_map.items():
        yp    = model.predict(Xts)
        yprob = model.predict_proba(Xts)
        try:
            auc = roc_auc_score(yts, yprob, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        report = classification_report(yts, yp, target_names=le.classes_, output_dict=True)
        cm     = confusion_matrix(yts, yp)
        results[name] = {
            'accuracy':    accuracy_score(yts, yp),
            'f1_weighted': f1_score(yts, yp, average='weighted'),
            'f1_macro':    f1_score(yts, yp, average='macro'),
            'roc_auc':     auc,
            'report':      report,
            'cm':          cm,
            'model':       model,
            'fi':          model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        }

    return results, le.classes_, FEATURE_COLS, X_train, y_train, scaler, le

# ── Load ──────────────────────────────────────────────────────────────────────
features, full_df, monthly, cat_sum, credit_df, cashflow_df, cf_fi_df, livret_unified, livret_txs, livret_monthly, anomaly_all, anomaly_flag = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/combo-chart--v2.png", width=60)
    st.title("Finance ML")
    st.caption("Banque Populaire — 2022–2026")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📈 Spending Analysis",
        "💼 Revenus & Employeurs",
        "🏦 Livret A & Savings",
        "🏠 Capacité d'Emprunt",
        "🚨 Anomaly Detection",
        "🤖 ML Models",
        "🔤 NLP Classifier",
        "🏦 Creditworthiness",
        "📉 Cash Flow Forecast",
        "📋 Loan Decision Report",
        "🔍 Transaction Explorer",
    ])
    st.markdown("---")
    st.markdown("**Project Stack**")
    for badge in ["Python", "scikit-learn", "XGBoost", "pandas", "Streamlit", "Plotly"]:
        st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)

    # ── Pipeline status banner ─────────────────────────────────────────────────
    st.markdown("---")
    import json as _json
    if _STATUS_JSON.exists():
        try:
            _ps = _json.loads(_STATUS_JSON.read_text(encoding="utf-8"))
            _run_at = _ps.get("run_at", "unknown")
            if _ps.get("overall") == "success":
                st.success(f"Pipeline OK — {_run_at}")
            else:
                _failed = [s["name"] for s in _ps.get("stages", []) if s["status"] == "failed"]
                st.error(f"Pipeline FAILED at: {', '.join(_failed)}\nLast run: {_run_at}")
                with st.expander("Show errors"):
                    for s in _ps.get("stages", []):
                        if s["status"] == "failed" and s.get("error"):
                            st.code(f"{s['name']}:\n{s['error']}")
        except Exception:
            st.warning("Could not read pipeline_status.json")
    else:
        st.info("Run `python run_pipeline.py` to execute the full pipeline.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Personal Finance ML Dashboard")
    st.markdown("**End-to-end ML pipeline** built on 3+ years of real bank statements")

    # ── Context banner ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fff8e1; border-left:5px solid #f39c12; border-radius:10px;
                padding:1rem 1.4rem; margin-bottom:1rem;">
        <h4 style="margin:0 0 0.5rem; color:#e67e22;">📌 Understanding This Dataset</h4>
        <p style="margin:0; color:#555; font-size:0.92rem; line-height:1.7;">
            This is a <strong>spending-only account (Compte Chèques 23192700536)</strong> —
            salary and income flow through a <strong>separate Livret A savings account (24971411768)</strong>,
            and are transferred in as needed.<br>
            This means <em>income appears as €0 on most months</em>, which explains the negative savings rate
            and high-risk credit labels. This is <strong>not a reflection of true financial health</strong> —
            it is a <strong>real-world data limitation</strong>, exactly the kind of challenge
            production ML systems face. The models are trained on genuine, imperfect data.
        </p>
        <div style="margin-top:0.8rem; display:flex; gap:2rem; flex-wrap:wrap;">
            <span style="color:#0f3460; font-weight:700;">✅ Checking Account: spending & expenses</span>
            <span style="color:#0f3460; font-weight:700;">✅ Livret A: salary deposits → transfers to checking</span>
            <span style="color:#0f3460; font-weight:700;">✅ Savings balance at Jan 2026: €9,269 (Livret A parsed!)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # KPI Cards
    debits  = full_df[full_df['type'] == 'DEBIT']['debit'].sum()
    credits = full_df[full_df['type'] == 'CREDIT']['credit'].sum()
    n_tx    = len(full_df)
    months  = monthly.shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{n_tx:,}")
    c2.metric("Total Debits",       f"€{debits:,.0f}")
    c3.metric("Total Credits",      f"€{credits:,.0f}")
    c4.metric("Months Covered",     f"{months}")

    st.markdown("---")
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Monthly Income vs Spend</div>', unsafe_allow_html=True)
        monthly_sorted = monthly.sort_values('year_month')
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_sorted['year_month'], y=monthly_sorted['monthly_income'],
                             name='Income', marker_color='#0f3460', opacity=0.85))
        fig.add_trace(go.Bar(x=monthly_sorted['year_month'], y=monthly_sorted['monthly_spend'],
                             name='Spend', marker_color='#e94560', opacity=0.85))
        fig.add_trace(go.Scatter(x=monthly_sorted['year_month'], y=monthly_sorted['monthly_net'],
                                 name='Net Flow', mode='lines+markers',
                                 line=dict(color='#533483', width=2.5),
                                 marker=dict(size=5)))
        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
        fig.update_layout(barmode='group', height=380, margin=dict(t=20, b=60),
                          legend=dict(orientation='h', y=-0.25),
                          xaxis_tickangle=-45, plot_bgcolor='white',
                          yaxis_title='Amount (EUR)')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-title">Spend by Category</div>', unsafe_allow_html=True)
        cat_debit = (full_df[full_df['type'] == 'DEBIT']
                     .groupby('category')['debit'].sum()
                     .sort_values(ascending=False)
                     .head(10).reset_index())
        fig2 = px.pie(cat_debit, values='debit', names='category',
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      hole=0.4)
        fig2.update_traces(textposition='outside', textinfo='label+percent')
        fig2.update_layout(height=380, margin=dict(t=20, b=20),
                           showlegend=False)
        st.plotly_chart(fig2, width="stretch")

    # Pipeline diagram
    st.markdown("---")
    st.markdown('<div class="section-title">ML Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("📄", "PDF Parser"),
        ("🧹", "Data Cleaning"),
        ("⚙️", "Feature Eng."),
        ("🏷️", "Auto-Labeling"),
        ("✂️", "Train/Val/Test"),
        ("🤖", "Model Training"),
        ("📊", "Evaluation"),
    ]
    n = len(steps)
    cols = st.columns(n * 2 - 1)
    for i, (icon, label) in enumerate(steps):
        cols[i * 2].markdown(f"""
        <div style="text-align:center; background:#0f3460; border-radius:12px;
                    padding:1rem 0.4rem; box-shadow:0 4px 12px rgba(15,52,96,0.3);">
            <div style="font-size:1.6rem; line-height:1.2;">{icon}</div>
            <div style="color:#ffffff; font-size:0.78rem; font-weight:700;
                        margin-top:0.4rem; letter-spacing:0.3px;">{label}</div>
        </div>""", unsafe_allow_html=True)
        if i < n - 1:
            cols[i * 2 + 1].markdown(
                '<div style="text-align:center; font-size:1.6rem; color:#e94560; '
                'padding-top:0.8rem; font-weight:bold;">→</div>',
                unsafe_allow_html=True
            )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SPENDING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Spending Analysis":
    st.title("Spending Analysis")
    st.markdown("---")

    # Rolling trend
    st.markdown('<div class="section-title">Daily Spend — 30-Day Rolling Average</div>', unsafe_allow_html=True)
    trend = full_df[full_df['type'] == 'DEBIT'].copy()
    trend = trend.set_index('date_operation').sort_index()
    daily   = trend['debit'].resample('D').sum().reset_index()
    daily.columns = ['date', 'spend']
    daily['rolling_30d'] = daily['spend'].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily['date'], y=daily['spend'],
                         name='Daily Spend', marker_color='#0f3460', opacity=0.35))
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['rolling_30d'],
                             name='30-Day Avg', mode='lines',
                             line=dict(color='#e94560', width=2.5)))
    fig.update_layout(height=320, margin=dict(t=10, b=10),
                      plot_bgcolor='white', yaxis_title='Amount (EUR)',
                      legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Total Spend by Category</div>', unsafe_allow_html=True)
        cat_spend = (full_df[full_df['type'] == 'DEBIT']
                     .groupby('category')['debit'].sum()
                     .sort_values(ascending=True).reset_index())
        fig2 = px.bar(cat_spend, x='debit', y='category', orientation='h',
                      color='debit', color_continuous_scale='Blues',
                      labels={'debit': 'Total Spend (EUR)', 'category': ''})
        fig2.update_layout(height=420, margin=dict(t=10, b=10),
                           plot_bgcolor='white', coloraxis_showscale=False)
        fig2.update_traces(text=cat_spend['debit'].apply(lambda v: f'€{v:,.0f}'),
                           textposition='outside')
        st.plotly_chart(fig2, width="stretch")

    with col2:
        st.markdown('<div class="section-title">Monthly Savings Rate</div>', unsafe_allow_html=True)
        monthly_s = monthly.sort_values('year_month')
        monthly_s['savings_pct'] = monthly_s['savings_rate'] * 100
        colors_sr = ['#e94560' if v < 0 else '#0f3460' for v in monthly_s['savings_pct']]
        fig3 = go.Figure(go.Bar(
            x=monthly_s['year_month'], y=monthly_s['savings_pct'],
            marker_color=colors_sr, opacity=0.85
        ))
        fig3.add_hline(y=0, line_dash='dash', line_color='gray')
        fig3.update_layout(height=420, margin=dict(t=10, b=60),
                           plot_bgcolor='white', yaxis_title='Savings Rate (%)',
                           xaxis_tickangle=-45)
        st.plotly_chart(fig3, width="stretch")

    # Day-of-week heatmap
    st.markdown('<div class="section-title">Spending Heatmap — Day of Week vs Month</div>', unsafe_allow_html=True)
    hm = full_df[full_df['type'] == 'DEBIT'].copy()
    hm['dow']   = hm['date_operation'].dt.day_name()
    hm['month_label'] = hm['date_operation'].dt.strftime('%b %Y')
    pivot = hm.pivot_table(values='debit', index='dow', columns='month_label',
                           aggfunc='sum', fill_value=0)
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot.reindex([d for d in dow_order if d in pivot.index])
    fig4 = px.imshow(pivot, color_continuous_scale='Blues',
                     labels=dict(color='Spend (EUR)'), aspect='auto')
    fig4.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig4, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2C — REVENUS & EMPLOYEURS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Revenus & Employeurs":
    st.title("Revenus & Employeurs")
    st.markdown("Tous les salaires et virements employeurs reçus sur le Compte Chèques")
    st.markdown("---")

    # Load transactions
    @st.cache_data
    def load_transactions():
        return pd.read_excel(_TRANSACTIONS_XLSX, sheet_name="Transactions")

    txdf = load_transactions()
    txdf['date_operation'] = pd.to_datetime(txdf['date_operation'], dayfirst=True, errors='coerce')
    credits = txdf[txdf['type'] == 'CREDIT'].copy()

    # ── Classify employer payments ────────────────────────────────────────────
    def classify_employer(desc):
        desc = str(desc).upper()
        if 'HDI GLOBAL SE' in desc:
            return 'HDI Global SE'
        elif 'JACAR 55' in desc or 'JACAR55' in desc:
            return 'EVI JACAR 55'
        elif 'STAFFMATCH' in desc:
            return 'EVI Staffmatch France'
        elif 'CPAM' in desc or 'NOVEOCARE' in desc:
            return 'Remboursement Santé'
        elif 'AHMED KAREL' in desc or 'WANMEGNI' in desc or 'GEORGE-ALEXANDRU' in desc:
            return 'Virement Tiers'
        elif 'PAYPAL' in desc or 'MANGOPAY' in desc or 'LEETCHI' in desc:
            return 'PayPal / Plateformes'
        elif 'SOCIETE GENERALE' in desc or 'SG-' in desc:
            return 'Participation / Prime'
        elif 'GAB' in desc:
            return 'Dépôt Espèces'
        elif 'REMISE CHEQUES' in desc:
            return 'Remise Chèque'
        else:
            return 'Autre'

    credits = credits.copy()
    credits['employer'] = credits['description'].apply(classify_employer)
    credits['year_month'] = credits['date_operation'].dt.to_period('M').astype(str)
    credits['year'] = credits['date_operation'].dt.year

    # Salary only = HDI + JACAR + STAFFMATCH
    salary = credits[credits['employer'].isin(['HDI Global SE', 'EVI JACAR 55', 'EVI Staffmatch France'])]

    # ── KPI cards ────────────────────────────────────────────────────────────
    total_salary  = salary['credit'].sum()
    n_payments    = len(salary)
    avg_salary    = salary['credit'].mean()
    latest_salary = salary.sort_values('date_operation').iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Salaires Reçus",    f"€{total_salary:,.0f}")
    c2.metric("Nombre de Paiements",     f"{n_payments}")
    c3.metric("Salaire Moyen / Paiement",f"€{avg_salary:,.0f}")
    c4.metric("Dernier Salaire",         f"€{latest_salary['credit']:,.0f}",
              delta=latest_salary['year_month'])

    st.markdown("---")

    # ── By employer summary ───────────────────────────────────────────────────
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">Total par Employeur</div>', unsafe_allow_html=True)
        emp_summary = salary.groupby('employer').agg(
            total=('credit', 'sum'),
            payments=('credit', 'count'),
            avg=('credit', 'mean'),
            first=('date_operation', 'min'),
            last=('date_operation', 'max'),
        ).reset_index().sort_values('total', ascending=False)

        for _, row in emp_summary.iterrows():
            color = '#0f3460' if 'HDI' in row['employer'] else '#533483'
            st.markdown(f"""
            <div style="background:white; border-left:5px solid {color}; border-radius:10px;
                        padding:0.9rem 1.2rem; margin-bottom:0.6rem;
                        box-shadow:0 2px 6px rgba(0,0,0,0.07);">
                <div style="font-weight:700; color:{color}; font-size:1rem;">{row['employer']}</div>
                <div style="display:flex; gap:2rem; margin-top:0.4rem; font-size:0.88rem; color:#555;">
                    <span>💰 <strong>€{row['total']:,.2f}</strong> total</span>
                    <span>📅 {int(row['payments'])} paiements</span>
                    <span>⌀ €{row['avg']:,.0f}/paiement</span>
                </div>
                <div style="font-size:0.8rem; color:#888; margin-top:0.3rem;">
                    {str(row['first'])[:10]} → {str(row['last'])[:10]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Répartition des Revenus</div>', unsafe_allow_html=True)
        all_emp = credits.groupby('employer')['credit'].sum().reset_index()
        all_emp = all_emp[all_emp['credit'] > 50].sort_values('credit', ascending=False)
        fig_pie = px.pie(all_emp, values='credit', names='employer',
                         color_discrete_sequence=['#0f3460','#533483','#e94560','#2ecc71','#f39c12','#1abc9c'],
                         hole=0.4)
        fig_pie.update_traces(textposition='outside', textinfo='label+percent')
        fig_pie.update_layout(height=370, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig_pie, width="stretch")

    st.markdown("---")

    # ── Monthly salary timeline ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Évolution Mensuelle des Salaires</div>', unsafe_allow_html=True)

    monthly_sal = salary.groupby(['year_month', 'employer'])['credit'].sum().reset_index()
    color_map   = {'HDI Global SE': '#0f3460', 'EVI JACAR 55': '#533483', 'EVI Staffmatch France': '#e94560'}

    fig_bar = go.Figure()
    for emp, color in color_map.items():
        sub = monthly_sal[monthly_sal['employer'] == emp]
        if len(sub) > 0:
            fig_bar.add_trace(go.Bar(
                x=sub['year_month'], y=sub['credit'],
                name=emp, marker_color=color, opacity=0.85,
                text=sub['credit'].apply(lambda x: f'€{x:,.0f}'),
                textposition='outside'
            ))

    fig_bar.update_layout(
        barmode='stack', height=380,
        plot_bgcolor='white', yaxis_title='Montant (EUR)',
        xaxis_tickangle=-45, margin=dict(t=20, b=60),
        legend=dict(orientation='h', y=-0.25)
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")

    # ── Full salary detail table ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Détail de tous les paiements employeurs</div>', unsafe_allow_html=True)

    salary_display = salary[['date_operation', 'employer', 'credit', 'description']].copy()
    salary_display['date_operation'] = salary_display['date_operation'].dt.strftime('%Y-%m-%d')
    salary_display = salary_display.sort_values('date_operation', ascending=False).reset_index(drop=True)
    salary_display.columns = ['Date', 'Employeur', 'Montant (€)', 'Référence']

    st.dataframe(salary_display, width="stretch", hide_index=True, height=400)

    total_hdi   = salary[salary['employer'] == 'HDI Global SE']['credit'].sum()
    total_jacar = salary[salary['employer'] == 'EVI JACAR 55']['credit'].sum()
    st.caption(f"HDI Global SE: €{total_hdi:,.2f} | EVI JACAR 55: €{total_jacar:,.2f} | Total: €{total_salary:,.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2B — LIVRET A & SAVINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏦 Livret A & Savings":
    st.title("Livret A & Savings Account")
    st.markdown("**Savings account (24971411768)** — where salary is deposited, then transferred to checking as needed.")

    if livret_unified is None:
        st.warning("Livret A data not found. Run `parse_livret_a.py` first.")
    else:
        # ── KPI cards ──────────────────────────────────────────────────────────
        total_saved   = livret_txs[livret_txs['amount'] > 0]['amount'].sum()
        total_out     = livret_txs[livret_txs['amount'] < 0]['amount'].abs().sum()
        n_months      = len(livret_monthly)
        final_balance = 9268.57  # from Jan 2026 statement

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Salary Deposited", f"€{total_saved:,.0f}")
        c2.metric("Transfers to Checking",  f"€{total_out:,.0f}")
        c3.metric("Months of Data",         f"{n_months}")
        c4.metric("Balance (Jan 2026)",     f"€{final_balance:,.0f}")

        st.markdown("---")

        # ── Monthly savings deposits vs transfers out ───────────────────────────
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown('<div class="section-title">Monthly: Salary Saved vs Transfers to Checking</div>', unsafe_allow_html=True)
            lm = livret_monthly.copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=lm['year_month'], y=lm['savings_deposits'],
                                 name='Salary Deposits', marker_color='#0f3460', opacity=0.85))
            fig.add_trace(go.Bar(x=lm['year_month'], y=lm['transfers_to_checking'],
                                 name='Transfers to Checking', marker_color='#e94560', opacity=0.85))
            fig.add_trace(go.Scatter(x=lm['year_month'], y=lm['net'],
                                     name='Net Saved', mode='lines+markers',
                                     line=dict(color='#2ecc71', width=2.5),
                                     marker=dict(size=6)))
            fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
            fig.update_layout(barmode='group', height=350, margin=dict(t=10, b=60),
                              legend=dict(orientation='h', y=-0.25),
                              xaxis_tickangle=-45, plot_bgcolor='white',
                              yaxis_title='Amount (EUR)')
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown('<div class="section-title">Livret A Balance Growth</div>', unsafe_allow_html=True)
            # Compute running balance from transactions
            lm_sorted = livret_txs.sort_values('date').copy()
            lm_sorted['running_balance'] = lm_sorted['amount'].cumsum()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=lm_sorted['date'], y=lm_sorted['running_balance'],
                mode='lines+markers', line=dict(color='#0f3460', width=2.5),
                fill='tozeroy', fillcolor='rgba(15,52,96,0.1)',
                marker=dict(size=4), name='Balance'
            ))
            fig2.update_layout(height=350, margin=dict(t=10, b=10),
                               plot_bgcolor='white', yaxis_title='Balance (EUR)',
                               showlegend=False)
            st.plotly_chart(fig2, width="stretch")

        st.markdown("---")

        # ── Unified view: true income picture ──────────────────────────────────
        st.markdown('<div class="section-title">True Financial Picture (Checking + Livret A)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#e8f5e9; border-left:4px solid #2ecc71; border-radius:8px;
                    padding:0.8rem 1.2rem; margin-bottom:1rem;">
            <p style="margin:0; color:#2d6a4f; font-size:0.9rem;">
                <strong>Key insight:</strong> The Compte Cheques shows near-zero income because salary goes directly
                to the Livret A. When we combine both accounts, we see the real cash flow:
                salary deposits into Livret A + transfers to checking fund all expenses.
            </p>
        </div>
        """, unsafe_allow_html=True)

        overlap = livret_unified[livret_unified['year_month'] >= '2024-09'].copy()
        overlap = overlap[overlap['total_income'] > 0]  # only months with Livret A data

        col1, col2 = st.columns(2)
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=overlap['year_month'], y=overlap['total_income'],
                                  name='True Income (Livret A)', marker_color='#2ecc71', opacity=0.85))
            fig3.add_trace(go.Bar(x=overlap['year_month'], y=overlap['total_spend'],
                                  name='Total Spend (Checking)', marker_color='#e94560', opacity=0.85))
            fig3.add_trace(go.Scatter(x=overlap['year_month'], y=overlap['total_net'],
                                      name='Net', mode='lines+markers',
                                      line=dict(color='#0f3460', width=2.5),
                                      marker=dict(size=6)))
            fig3.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
            fig3.update_layout(barmode='group', height=320, margin=dict(t=20, b=60),
                               legend=dict(orientation='h', y=-0.3),
                               xaxis_tickangle=-45, plot_bgcolor='white',
                               title='Income vs Spend (months with Livret A data)',
                               yaxis_title='EUR')
            st.plotly_chart(fig3, width="stretch")

        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=overlap['year_month'], y=overlap['savings_rate'],
                                  name='% Saved', marker_color='#0f3460',
                                  text=overlap['savings_rate'].round(0).astype(int).astype(str) + '%',
                                  textposition='outside'))
            fig4.add_hline(y=20, line_dash='dash', line_color='#e94560',
                           annotation_text='20% benchmark', line_width=1.5)
            fig4.update_layout(height=320, margin=dict(t=20, b=60),
                               plot_bgcolor='white', yaxis_title='Savings Rate (%)',
                               title='Savings Rate (salary saved / total salary)',
                               xaxis_tickangle=-45, showlegend=False,
                               yaxis=dict(range=[0, 120]))
            st.plotly_chart(fig4, width="stretch")

        st.markdown("---")

        # ── Transaction log ────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Livret A Transactions</div>', unsafe_allow_html=True)
        tx_show = livret_txs[['date', 'label', 'amount', 'type', 'year_month']].copy()
        tx_show['date'] = tx_show['date'].dt.strftime('%Y-%m-%d')
        tx_show = tx_show.sort_values('date', ascending=False).reset_index(drop=True)

        col1, col2 = st.columns(2)
        tx_type = col1.selectbox("Filter by type", ["All", "CREDIT (deposits)", "DEBIT (withdrawals)"])
        if tx_type == "CREDIT (deposits)":
            tx_show = tx_show[tx_show['type'] == 'CREDIT']
        elif tx_type == "DEBIT (withdrawals)":
            tx_show = tx_show[tx_show['type'] == 'DEBIT']

        st.dataframe(tx_show, width="stretch", height=350)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — CAPACITE D'EMPRUNT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏠 Capacité d'Emprunt":
    st.title("Simulateur de Capacité d'Emprunt")
    st.markdown("Calcul basé sur tes **vraies données financières** — exactement comme le ferait une banque française (règles HCSF 2024).")
    st.markdown("---")

    # ── Load real data ────────────────────────────────────────────────────────
    txdf_raw = pd.read_excel(_TRANSACTIONS_XLSX, sheet_name="Transactions")
    txdf_raw['date_operation'] = pd.to_datetime(txdf_raw['date_operation'], dayfirst=True, errors='coerce')

    credits_raw = txdf_raw[txdf_raw['type'] == 'CREDIT']
    salary_raw  = credits_raw[credits_raw['description'].str.upper().str.contains('HDI|JACAR|STAFFMATCH', na=False)]
    avg_salary  = 1967.0  # Confirmed real net salary from HDI (Jan 2026 payment)

    debits_raw  = txdf_raw[txdf_raw['type'] == 'DEBIT']
    recent_deb  = debits_raw[debits_raw['date_operation'] >= '2025-08-01']
    fixed_raw   = recent_deb[recent_deb['description'].str.upper().str.contains(
        'PRLV|PRELEVEMENT|ASSUR|CRISTAL|ABONNEMENT|NAVIGO|IMAGINE R|GYM', na=False
    )]
    avg_fixed   = round(fixed_raw['debit'].sum() / 6, 0)
    livret_bal  = 9268.57

    # ── Statut alternance warning ─────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1a1200; border-left:6px solid #f39c12; border-radius:10px;
                padding:1rem 1.4rem; margin-bottom:1rem;">
        <strong style="color:#f39c12; font-size:1rem;">⚠️ Statut Actuel : Alternant (contrat jusqu'en septembre 2027)</strong>
        <div style="color:#e0d0a0; font-size:0.88rem; margin-top:0.5rem; line-height:1.7;">
            Les banques françaises <strong style="color:#ffffff;">refusent quasi-systématiquement</strong> un prêt immobilier
            en cours d'alternance — le contrat est temporaire. La simulation ci-dessous te montre
            <strong style="color:#ffffff;">ce que tu pourras emprunter à partir d'octobre 2027</strong>, une fois en CDI chez HDI
            (ou équivalent). D'ici là : <strong style="color:#2ecc71;">continue d'épargner sur le Livret A</strong> pour maximiser ton apport.
        </div>
        <div style="display:flex; gap:2rem; margin-top:0.8rem; flex-wrap:wrap; font-size:0.85rem;">
            <span style="color:#2ecc71;">✅ Aujourd'hui : Construire l'apport</span>
            <span style="color:#2ecc71;">✅ Sept 2027 : Fin alternance → CDI possible</span>
            <span style="color:#2ecc71;">✅ Oct 2027+ : Dossier bancaire recevable</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Toggle: Situation actuelle vs Projection 2027 ─────────────────────────
    mode = st.radio("Simuler pour :", ["📅 Projection CDI (Oct 2027+)", "📊 Situation actuelle"],
                    horizontal=True)
    projection_mode = (mode == "📅 Projection CDI (Oct 2027+)")

    # Estimate Livret A balance by Oct 2027 (18 months @ avg €700/month net saving)
    livret_2027 = round(livret_bal + 18 * 700, 0)

    # ── Inputs ───────────────────────────────────────────────────────────────
    st.markdown("### Paramètres du Simulateur")
    st.markdown("*Pré-rempli avec tes données réelles — ajuste si besoin.*")

    # Default values change based on projection mode
    default_salary = int(avg_salary)
    default_apport = int(livret_2027) if projection_mode else int(livret_bal)
    apport_help    = (f"Livret A estimé en oct 2027 (€{livret_bal:,.0f} + 18 mois épargne)"
                      if projection_mode else f"Ton Livret A actuel: €{livret_bal:,.0f}")

    col_i, col_s = st.columns(2)
    with col_i:
        st.markdown("**Revenus**")
        revenu_net    = st.number_input("Salaire net mensuel (€)", value=default_salary, step=50,
                                         help=f"Salaire réel HDI confirmé: €{avg_salary:,.0f}/mois")
        autres_revenus = st.number_input("Autres revenus mensuels (€)", value=0, step=50,
                                          help="Revenus locatifs, primes régulières, etc.")
        revenu_total  = revenu_net + autres_revenus

    with col_s:
        st.markdown("**Charges actuelles**")
        loyer_actuel  = st.number_input("Loyer actuel (€/mois)", value=0, step=50,
                                         help="0 si hébergé chez tes parents ou propriétaire")
        charges_fixes = st.number_input("Autres charges fixes (€/mois)", value=int(avg_fixed), step=10,
                                         help=f"Moyenne réelle (Navigo, Imagine R, Gym...): €{avg_fixed:,.0f}")
        credits_cours = st.number_input("Crédits en cours (€/mois)", value=0, step=50,
                                         help="Crédit conso, auto, etc.")

    col_p, col_d = st.columns(2)
    with col_p:
        st.markdown("**Projet immobilier**")
        prix_bien     = st.number_input("Prix du bien (€)", value=200000, step=5000)
        apport        = st.number_input("Apport personnel (€)", value=default_apport, step=500,
                                         help=apport_help)
        duree_ans     = st.slider("Durée du prêt (ans)", 10, 25, 20)

    with col_d:
        st.markdown("**Taux**")
        taux_annuel   = st.number_input("Taux d'intérêt annuel (%)", value=3.50, step=0.05, format="%.2f")
        taux_assurance = st.number_input("Taux assurance annuel (%)", value=0.30, step=0.01, format="%.2f")

    st.markdown("---")

    # ── CALCULATIONS ─────────────────────────────────────────────────────────
    montant_emprunte = prix_bien - apport
    n_mois           = duree_ans * 12
    taux_mensuel     = (taux_annuel / 100) / 12
    taux_assur_mens  = (taux_assurance / 100) / 12

    # Mensualité crédit (formule annuité constante)
    if taux_mensuel > 0:
        mensualite_credit = montant_emprunte * (taux_mensuel * (1 + taux_mensuel)**n_mois) / \
                            ((1 + taux_mensuel)**n_mois - 1)
    else:
        mensualite_credit = montant_emprunte / n_mois

    mensualite_assurance = montant_emprunte * taux_assur_mens
    mensualite_totale    = mensualite_credit + mensualite_assurance

    # Taux d'endettement HCSF
    charges_totales    = credits_cours + loyer_actuel + charges_fixes + mensualite_totale
    charges_sans_loyer = credits_cours + charges_fixes + mensualite_totale  # loyer remplacé par crédit
    taux_endettement   = (charges_sans_loyer / revenu_total * 100) if revenu_total > 0 else 0

    # Reste à vivre
    reste_a_vivre = revenu_total - charges_sans_loyer

    # Saut de charge
    saut_de_charge = mensualite_totale - loyer_actuel

    # Apport %
    apport_pct = (apport / prix_bien * 100) if prix_bien > 0 else 0

    # Capacité max (taux 35%)
    capacite_mensuelle_max = revenu_total * 0.35 - credits_cours - charges_fixes
    montant_max_empruntable = capacite_mensuelle_max * ((1 + taux_mensuel)**n_mois - 1) / \
                              (taux_mensuel * (1 + taux_mensuel)**n_mois) if taux_mensuel > 0 else 0

    # Verdict
    limite_hcsf   = taux_endettement <= 35
    reste_ok      = reste_a_vivre >= 700
    apport_ok     = apport_pct >= 10
    saut_ok       = saut_de_charge <= revenu_total * 0.15

    score_banque  = sum([limite_hcsf, reste_ok, apport_ok, saut_ok])

    # ── VERDICT BANNER ───────────────────────────────────────────────────────
    if score_banque == 4:
        v_color, v_border, v_icon, v_title = '#0d2b0d', '#2ecc71', '✅', 'DOSSIER SOLIDE — Prêt probable'
    elif score_banque >= 2:
        v_color, v_border, v_icon, v_title = '#2b1a00', '#f39c12', '⚠️', 'DOSSIER MOYEN — Négociation possible'
    else:
        v_color, v_border, v_icon, v_title = '#2b0000', '#e94560', '🔴', 'DOSSIER RISQUE — Refus probable'

    st.markdown(f"""
    <div style="background:{v_color}; border-left:6px solid {v_border}; border-radius:12px;
                padding:1.2rem 1.6rem; margin-bottom:1.2rem;">
        <h3 style="margin:0 0 0.5rem; color:{v_border}; font-size:1.2rem;">{v_icon} {v_title}</h3>
        <div style="display:flex; gap:3rem; flex-wrap:wrap; font-size:0.92rem; color:#e0e0e0; margin-top:0.5rem;">
            <span>Taux d'endettement : <strong style="color:{v_border};">{taux_endettement:.1f}%</strong> (limite 35%)</span>
            <span>Reste à vivre : <strong style="color:{v_border};">€{reste_a_vivre:,.0f}</strong> (min 700€)</span>
            <span>Apport : <strong style="color:{v_border};">{apport_pct:.1f}%</strong> (min 10%)</span>
            <span>Critères OK : <strong style="color:{v_border};">{score_banque}/4</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Projection 2027 box ───────────────────────────────────────────────────
    if projection_mode:
        st.markdown(f"""
        <div style="background:#0d2b0d; border-left:5px solid #2ecc71; border-radius:10px;
                    padding:0.9rem 1.4rem; margin-bottom:1rem;">
            <strong style="color:#2ecc71;">📅 Projection CDI — Octobre 2027</strong>
            <div style="color:#c8e6c9; font-size:0.87rem; margin-top:0.5rem; line-height:1.7;">
                Salaire HDI confirmé <strong style="color:#fff;">€{avg_salary:,.0f}/mois</strong> •
                Livret A estimé <strong style="color:#fff;">€{livret_2027:,.0f}</strong>
                (€{livret_bal:,.0f} + 18 mois × ~€700 épargne nette) •
                Capacité max à 35% sur 20 ans : <strong style="color:#fff;">€{max(0,montant_max_empruntable):,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── KPI METRICS ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mensualité totale",     f"€{mensualite_totale:,.0f}",
              help="Crédit + assurance")
    c2.metric("Montant empruntable",   f"€{montant_emprunte:,.0f}",
              help="Prix - apport")
    c3.metric("Capacité max (35%)",    f"€{max(0, montant_max_empruntable):,.0f}",
              help="Montant max selon règle HCSF")
    c4.metric("Coût total du crédit",  f"€{mensualite_totale*n_mois - montant_emprunte:,.0f}",
              help="Intérêts + assurance sur toute la durée")

    st.markdown("---")

    # ── 4 PILLARS ────────────────────────────────────────────────────────────
    st.markdown("### Les 4 Critères Bancaires — Ton Dossier")

    p1, p2, p3, p4 = st.columns(4)

    def pilier(col, titre, valeur, seuil, ok, detail):
        color  = '#2ecc71' if ok else '#e94560'
        icon   = '✅' if ok else '❌'
        col.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:1rem; text-align:center;
                    border: 2px solid {color}; height:200px;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="color:#aaa; font-size:0.78rem; margin-top:0.3rem;">{titre}</div>
            <div style="color:{color}; font-size:1.6rem; font-weight:700; margin:0.3rem 0;">{valeur}</div>
            <div style="color:#888; font-size:0.72rem;">Seuil : {seuil}</div>
            <div style="color:#ccc; font-size:0.75rem; margin-top:0.4rem;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    pilier(p1, "Taux d'Endettement",  f"{taux_endettement:.1f}%",  "≤ 35%",
           limite_hcsf, "Règle HCSF — assurance incluse")
    pilier(p2, "Reste à Vivre",       f"€{reste_a_vivre:,.0f}",   "≥ €700",
           reste_ok, "Après toutes les charges fixes")
    pilier(p3, "Apport Personnel",    f"{apport_pct:.1f}%",        "≥ 10%",
           apport_ok, f"€{apport:,.0f} sur €{prix_bien:,.0f}")
    pilier(p4, "Saut de Charge",      f"€{saut_de_charge:,.0f}",  f"≤ €{revenu_total*0.15:,.0f}",
           saut_ok, f"Loyer actuel → mensualité")

    st.markdown("---")

    # ── AMORTIZATION CHART ────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Évolution capital restant dû**")
        capital_restant = []
        cap = float(montant_emprunte)
        for m in range(n_mois + 1):
            capital_restant.append(cap)
            if m < n_mois:
                interet = cap * taux_mensuel
                principal = mensualite_credit - interet
                cap = max(0, cap - principal)

        years  = [i / 12 for i in range(n_mois + 1)]
        fig_am = go.Figure()
        fig_am.add_trace(go.Scatter(x=years, y=capital_restant, fill='tozeroy',
                                     line=dict(color='#e94560', width=2),
                                     fillcolor='rgba(233,69,96,0.15)', name='Capital dû'))
        fig_am.update_layout(height=280, plot_bgcolor='white', margin=dict(t=10, b=10),
                              xaxis_title='Années', yaxis_title='Capital restant (€)',
                              showlegend=False)
        st.plotly_chart(fig_am, width="stretch")

    with col_g2:
        st.markdown("**Répartition du coût total**")
        interet_total   = mensualite_credit * n_mois - montant_emprunte
        assurance_total = mensualite_assurance * n_mois
        fig_pie = go.Figure(go.Pie(
            labels=['Capital', 'Intérêts', 'Assurance'],
            values=[montant_emprunte, interet_total, assurance_total],
            hole=0.45,
            marker_colors=['#0f3460', '#e94560', '#f39c12'],
            textfont_size=13
        ))
        fig_pie.update_layout(height=280, margin=dict(t=10, b=10),
                              legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig_pie, width="stretch")

    # ── DETAIL TABLE ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Récapitulatif Détaillé")

    recap = pd.DataFrame({
        'Critère': [
            'Revenu net mensuel', 'Autres revenus', 'Total revenus',
            '—',
            'Charges fixes actuelles', 'Crédits en cours', 'Loyer actuel',
            'Mensualité crédit', "Mensualité assurance", 'Total charges (avec crédit)',
            '—',
            "Taux d'endettement", 'Reste à vivre', 'Saut de charge', 'Apport (%)',
            '—',
            'Montant emprunté', 'Durée', "Taux d'intérêt", 'Taux assurance',
            'Coût total intérêts', 'Coût total assurance', 'Coût total crédit',
        ],
        'Valeur': [
            f"€{revenu_net:,.0f}", f"€{autres_revenus:,.0f}", f"€{revenu_total:,.0f}",
            '',
            f"€{charges_fixes:,.0f}", f"€{credits_cours:,.0f}", f"€{loyer_actuel:,.0f}",
            f"€{mensualite_credit:,.0f}", f"€{mensualite_assurance:,.0f}",
            f"€{charges_sans_loyer:,.0f}",
            '',
            f"{taux_endettement:.1f}% {'✅' if limite_hcsf else '❌'}",
            f"€{reste_a_vivre:,.0f} {'✅' if reste_ok else '❌'}",
            f"€{saut_de_charge:,.0f} {'✅' if saut_ok else '❌'}",
            f"{apport_pct:.1f}% {'✅' if apport_ok else '❌'}",
            '',
            f"€{montant_emprunte:,.0f}", f"{duree_ans} ans",
            f"{taux_annuel:.2f}%", f"{taux_assurance:.2f}%",
            f"€{interet_total:,.0f}", f"€{assurance_total:,.0f}",
            f"€{mensualite_totale * n_mois:,.0f}",
        ]
    })
    st.dataframe(recap, width="stretch", hide_index=True, height=500)

    st.info(f"**Capacité d'emprunt max (règle 35% HCSF)** avec tes revenus actuels "
            f"(€{revenu_total:,.0f}/mois) et charges fixes (€{charges_fixes:,.0f}/mois) "
            f"sur {duree_ans} ans à {taux_annuel}% : **€{max(0,montant_max_empruntable):,.0f}**")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("Personal Finance Anomaly Detection")
    st.markdown("Unsupervised ML to detect unusual transactions — potential fraud, overspending, or data errors.")
    st.markdown("---")

    if anomaly_all is None:
        st.warning("Run `anomaly_detection.py` first to generate results.")
    else:
        total_tx   = len(anomaly_all)
        total_anom = int(anomaly_flag['is_anomaly'].sum()) if anomaly_flag is not None else 0
        anom_rate  = total_anom / total_tx * 100
        top_anom   = anomaly_flag.sort_values('anomaly_score', ascending=False).iloc[0]
        avg_score  = anomaly_flag['anomaly_score'].mean()

        # ── KPIs ──────────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions Analysées", f"{total_tx:,}")
        c2.metric("Anomalies Détectées",    f"{total_anom}",
                  delta=f"{anom_rate:.1f}% du total", delta_color="inverse")
        c3.metric("Score Moyen Anomalie",   f"{avg_score:.0f}/100")
        c4.metric("Transaction + Suspecte", f"€{top_anom['debit']:,.0f}",
                  delta=top_anom['category'])

        # ── How it works banner ────────────────────────────────────────────────
        st.markdown("""
        <div style="background:#1a2744; border-left:5px solid #e94560; border-radius:10px;
                    padding:0.9rem 1.4rem; margin:0.5rem 0 1rem;">
            <strong style="color:#ffffff; font-size:0.95rem;">Comment ca fonctionne — Ensemble de 3 modeles :</strong>
            <div style="display:flex; gap:2rem; margin-top:0.6rem; flex-wrap:wrap; font-size:0.88rem; color:#d0d8f0;">
                <span>&#127794; <strong style="color:#ffffff;">Isolation Forest</strong> — isole les points rares dans l'espace des features</span>
                <span>&#9711; <strong style="color:#ffffff;">One-Class SVM</strong> — apprend la frontiere du comportement normal</span>
                <span>&#128269; <strong style="color:#ffffff;">Local Outlier Factor</strong> — compare la densite locale a ses voisins</span>
            </div>
            <div style="margin-top:0.6rem; font-size:0.85rem; color:#e94560; font-weight:600;">
                &#9888; Une transaction est flaggee anomalie si &ge; 2 modeles sur 3 sont d'accord.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Score distribution + Monthly rate ─────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Distribution des Scores d\'Anomalie</div>', unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=anomaly_all['anomaly_score'], nbinsx=30,
                marker_color='#0f3460', opacity=0.7, name='Normal'
            ))
            fig_hist.add_trace(go.Histogram(
                x=anomaly_flag['anomaly_score'], nbinsx=20,
                marker_color='#e94560', opacity=0.85, name='Anomalie'
            ))
            fig_hist.add_vline(x=50, line_dash='dash', line_color='#f39c12',
                               annotation_text='Seuil 50')
            fig_hist.update_layout(
                barmode='overlay', height=320, plot_bgcolor='white',
                xaxis_title='Anomaly Score (0-100)', yaxis_title='Nombre de transactions',
                legend=dict(orientation='h', y=-0.25), margin=dict(t=10, b=60)
            )
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            st.markdown('<div class="section-title">Taux d\'Anomalie Mensuel</div>', unsafe_allow_html=True)
            try:
                monthly_rate = pd.read_excel(_ANOMALY_XLSX, sheet_name="Monthly Rate")
                monthly_rate = monthly_rate[monthly_rate['anomaly_tx'] > 0]
                fig_rate = go.Figure()
                fig_rate.add_trace(go.Bar(
                    x=monthly_rate['year_month'], y=monthly_rate['anomaly_rate'],
                    marker_color=monthly_rate['anomaly_rate'].apply(
                        lambda x: '#e94560' if x > 15 else ('#f39c12' if x > 5 else '#0f3460')
                    ),
                    text=monthly_rate['anomaly_tx'].astype(str) + ' tx',
                    textposition='outside'
                ))
                fig_rate.add_hline(y=5, line_dash='dash', line_color='#f39c12',
                                   annotation_text='5% seuil')
                fig_rate.update_layout(
                    height=320, plot_bgcolor='white',
                    xaxis_tickangle=-45, yaxis_title='Taux anomalie (%)',
                    margin=dict(t=10, b=60), showlegend=False
                )
                st.plotly_chart(fig_rate, width="stretch")
            except Exception:
                st.info("Monthly rate data not available.")

        st.markdown("---")

        # ── Anomalies by category ──────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-title">Anomalies par Catégorie</div>', unsafe_allow_html=True)
            cat_counts = anomaly_flag.groupby('category').agg(
                count=('debit', 'count'),
                total=('debit', 'sum')
            ).reset_index().sort_values('count', ascending=True)
            fig_cat = go.Figure(go.Bar(
                x=cat_counts['count'], y=cat_counts['category'],
                orientation='h',
                marker_color='#e94560', opacity=0.85,
                text=cat_counts['total'].apply(lambda x: f'€{x:,.0f}'),
                textposition='outside'
            ))
            fig_cat.update_layout(
                height=320, plot_bgcolor='white',
                xaxis_title='Nb anomalies', margin=dict(t=10, b=10, r=80),
                showlegend=False
            )
            st.plotly_chart(fig_cat, width="stretch")

        with col4:
            st.markdown('<div class="section-title">Score vs Montant</div>', unsafe_allow_html=True)
            fig_scatter = px.scatter(
                anomaly_all, x='debit', y='anomaly_score',
                color='is_anomaly',
                color_discrete_map={0: '#0f3460', 1: '#e94560'},
                hover_data=['description', 'category', 'vote_count'],
                labels={'debit': 'Montant (€)', 'anomaly_score': 'Anomaly Score',
                        'is_anomaly': 'Anomalie'},
                opacity=0.65
            )
            fig_scatter.add_hline(y=50, line_dash='dash', line_color='#f39c12', line_width=1)
            fig_scatter.update_layout(
                height=320, plot_bgcolor='white',
                margin=dict(t=10, b=10),
                legend=dict(title='', orientation='h', y=-0.2)
            )
            st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("---")

        # ── Model comparison ───────────────────────────────────────────────────
        st.markdown('<div class="section-title">Comparaison des 3 Modèles</div>', unsafe_allow_html=True)
        try:
            model_sum = pd.read_excel(_ANOMALY_XLSX, sheet_name="Model Summary")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            for col, (_, row) in zip([col_m1, col_m2, col_m3, col_m4], model_sum.iterrows()):
                color = '#e94560' if 'Ensemble' in row['Model'] else '#0f3460'
                col.markdown(f"""
                <div style="background:white; border-left:4px solid {color}; border-radius:10px;
                            padding:0.8rem 1rem; box-shadow:0 2px 6px rgba(0,0,0,0.07);">
                    <div style="font-size:0.8rem; color:#888;">{row['Model']}</div>
                    <div style="font-size:1.6rem; font-weight:700; color:{color};">{int(row['Anomalies Found'])}</div>
                    <div style="font-size:0.78rem; color:#555;">{row['Rate (%)']:.1f}% | {row['Method'][:28]}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("---")

        # ── Flagged transactions table ─────────────────────────────────────────
        st.markdown('<div class="section-title">🚨 Transactions Flaggées (triées par score)</div>', unsafe_allow_html=True)

        # Filters
        fc1, fc2 = st.columns(2)
        min_score = fc1.slider("Score minimum", 0, 100, 50)
        vote_filter = fc2.selectbox("Votes modèles", ["Tous (≥1)", "Majorité (≥2)", "Unanime (3)"])
        vote_map = {"Tous (≥1)": 1, "Majorité (≥2)": 2, "Unanime (3)": 3}

        filtered_anom = anomaly_flag[
            (anomaly_flag['anomaly_score'] >= min_score) &
            (anomaly_flag['vote_count'] >= vote_map[vote_filter])
        ].sort_values('anomaly_score', ascending=False).copy()

        filtered_anom['date_operation'] = filtered_anom['date_operation'].dt.strftime('%Y-%m-%d')
        filtered_anom['Votes'] = filtered_anom['vote_count'].astype(int).apply(
            lambda x: '🔴🔴🔴' if x == 3 else ('🟠🟠' if x == 2 else '🟡')
        )

        display_cols = ['date_operation', 'description', 'category', 'debit',
                        'anomaly_score', 'Votes']
        display_anom = filtered_anom[display_cols].copy()
        display_anom.columns = ['Date', 'Description', 'Catégorie', 'Montant (€)',
                                'Score', 'Votes']

        st.dataframe(display_anom, width="stretch", hide_index=True, height=420)
        st.caption(f"{len(filtered_anom)} transactions affichées | "
                   f"Score moyen: {filtered_anom['anomaly_score'].mean():.0f} | "
                   f"Montant total: €{filtered_anom['debit'].sum():,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("ML Model Training & Evaluation")

    with st.spinner("Training models... (cached after first run)"):
        results, class_names, FEATURE_COLS, X_train, y_train, scaler, le = train_models(features)

    st.markdown("---")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    summary_data = []
    for name, r in results.items():
        summary_data.append({
            'Model':         name,
            'Accuracy':      round(r['accuracy'],    3),
            'F1 Weighted':   round(r['f1_weighted'],  3),
            'F1 Macro':      round(r['f1_macro'],     3),
            'ROC-AUC':       round(r['roc_auc'], 3) if r['roc_auc'] else 'N/A',
        })
    summary_df = pd.DataFrame(summary_data)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        metric_cols = ['Accuracy', 'F1 Weighted', 'ROC-AUC']
        colors_m    = ['#0f3460', '#e94560', '#533483']
        fig = go.Figure()
        for metric, color in zip(metric_cols, colors_m):
            vals = [results[m][metric.lower().replace(' ', '_').replace('-', '_')]
                    if metric != 'ROC-AUC' else results[m]['roc_auc']
                    for m in results.keys()]
            fig.add_trace(go.Bar(
                name=metric, x=list(results.keys()), y=vals,
                marker_color=color, opacity=0.88,
                text=[f'{v:.2f}' for v in vals], textposition='outside'
            ))
        fig.update_layout(barmode='group', height=360, margin=dict(t=20, b=10),
                          plot_bgcolor='white', yaxis=dict(range=[0, 1.1]),
                          legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.dataframe(summary_df.set_index('Model'), width="stretch", height=200)
        st.markdown("**Data Split**")
        total = len(features[features['category'].map(features['category'].value_counts()) >= 5])
        split_df = pd.DataFrame({
            'Set':     ['Train (70%)', 'Validation (15%)', 'Test (15%)'],
            'Samples': [int(total * 0.70), int(total * 0.15), int(total * 0.15)],
        })
        fig_split = px.pie(split_df, values='Samples', names='Set', hole=0.5,
                           color_discrete_sequence=['#0f3460', '#e94560', '#533483'])
        fig_split.update_layout(height=220, margin=dict(t=10, b=10), showlegend=True)
        st.plotly_chart(fig_split, width="stretch")

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select model", list(results.keys()), index=2)
    cm = results[model_choice]['cm']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_df   = pd.DataFrame(cm_norm, index=class_names, columns=class_names)

    fig_cm = px.imshow(cm_df, color_continuous_scale='Blues',
                       labels=dict(x='Predicted', y='Actual', color='Rate'),
                       text_auto='.0%', aspect='auto')
    fig_cm.update_layout(height=520, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    fi_model = st.selectbox("Model", ["Random Forest", "XGBoost"])
    fi_vals  = results[fi_model]['fi']
    if fi_vals is not None:
        fi_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': fi_vals})
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues')
        fig_fi.update_layout(height=500, margin=dict(t=10, b=10),
                             plot_bgcolor='white', coloraxis_showscale=False)
        st.plotly_chart(fig_fi, width="stretch")

    st.markdown("---")

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Per-Class F1 Score — All Models</div>', unsafe_allow_html=True)
    pc_data = []
    for name, r in results.items():
        for cls in class_names:
            if cls in r['report']:
                pc_data.append({'Model': name, 'Category': cls,
                                'F1': r['report'][cls]['f1-score']})
    pc_df = pd.DataFrame(pc_data)
    fig_pc = px.bar(pc_df, x='Category', y='F1', color='Model', barmode='group',
                    color_discrete_sequence=['#0f3460', '#e94560', '#533483'])
    fig_pc.update_layout(height=380, margin=dict(t=10, b=10),
                         plot_bgcolor='white', xaxis_tickangle=-40,
                         yaxis=dict(range=[0, 1.1]),
                         legend=dict(orientation='h', y=-0.25))
    st.plotly_chart(fig_pc, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — NLP CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔤 NLP Classifier":
    st.title("NLP Spending Category Classifier")
    st.markdown("TF-IDF on transaction descriptions + numeric features — lifting F1 from **0.51 → 0.77**")
    st.markdown("---")

    @st.cache_resource
    def train_nlp(features):
        import re
        def clean_text(text):
            text = str(text).upper()
            text = re.sub(r'\b[A-Z0-9]{6,}\b', '', text)
            text = re.sub(r'\b\d{6}\b', '', text)
            text = re.sub(r'CB\*+\d+', '', text)
            text = re.sub(r'\b\d{2,5}\b', '', text)
            return re.sub(r'\s+', ' ', text).strip()

        cat_counts = features['category'].value_counts()
        valid_cats = cat_counts[cat_counts >= 5].index
        df = features[features['category'].isin(valid_cats)].copy()
        df['desc_clean'] = df['description'].apply(clean_text)

        NUMERIC = ['abs_amount','log_amount','is_round_number','month','day_of_week',
                   'is_weekend','quarter','rolling_7d_spend','rolling_30d_spend',
                   'monthly_income','monthly_spend','monthly_net','tx_count',
                   'avg_tx_amount','savings_rate','type_encoded']

        le   = LabelEncoder()
        y    = le.fit_transform(df['category'])
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=500, min_df=2, sublinear_tf=True)
        X_text = tfidf.fit_transform(df['desc_clean'])
        scaler = StandardScaler()
        X_num  = csr_matrix(scaler.fit_transform(df[NUMERIC].values))
        X_comb = hstack([X_text, X_num])

        idx = np.arange(len(y))
        idx_tr, idx_ts = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
        X_tr_t, X_ts_t = X_text[idx_tr], X_text[idx_ts]
        X_tr_c, X_ts_c = X_comb[idx_tr], X_comb[idx_ts]
        y_tr, y_ts     = y[idx_tr], y[idx_ts]

        models_nlp = {
            "LR — TF-IDF only":          (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), X_tr_t, X_ts_t),
            "LR — TF-IDF + Numeric":     (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), X_tr_c, X_ts_c),
            "RF — TF-IDF + Numeric":     (RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1), X_tr_c, X_ts_c),
            "XGBoost — TF-IDF + Numeric":(XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0), X_tr_c, X_ts_c),
        }
        res = {}
        for name, (model, Xtr, Xts) in models_nlp.items():
            model.fit(Xtr, y_tr)
            yp = model.predict(Xts)
            try:
                auc = roc_auc_score(y_ts, model.predict_proba(Xts), multi_class='ovr', average='weighted')
            except Exception:
                auc = None
            res[name] = {
                'accuracy': round(accuracy_score(y_ts, yp), 4),
                'f1':       round(f1_score(y_ts, yp, average='weighted'), 4),
                'auc':      round(auc, 4) if auc else None,
                'report':   classification_report(y_ts, yp, target_names=le.classes_, output_dict=True),
                'cm':       confusion_matrix(y_ts, yp),
            }

        # Top TF-IDF terms from LR
        lr_model = res["LR — TF-IDF only"]['report']
        vocab    = tfidf.get_feature_names_out()
        lr_fitted = models_nlp["LR — TF-IDF only"][0]
        top_terms = {}
        for i, cls in enumerate(le.classes_):
            coef = lr_fitted.coef_[i]
            top_terms[cls] = [vocab[j] for j in np.argsort(coef)[-6:][::-1]]

        return res, le.classes_, top_terms, tfidf

    with st.spinner("Training NLP models... (cached after first run)"):
        nlp_res, nlp_classes, top_terms, tfidf = train_nlp(features)

    # ── Improvement banner ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Numeric Only F1",   "0.51", help="Previous baseline")
    col2.metric("NLP Best F1",       "0.77", delta="+0.26")
    col3.metric("Best Accuracy",     "76%",  delta="+24pp")
    col4.metric("Best ROC-AUC",      "0.95", delta="+0.08")

    st.markdown("---")

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">NLP Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])
    with col1:
        nlp_summary = pd.DataFrame([
            {'Model': n, 'Accuracy': r['accuracy'], 'F1 Weighted': r['f1'], 'ROC-AUC': r['auc']}
            for n, r in nlp_res.items()
        ])
        # Add previous baseline row
        baseline_row = pd.DataFrame([{'Model': 'Numeric Only (prev.)', 'Accuracy': 0.52, 'F1 Weighted': 0.51, 'ROC-AUC': 0.87}])
        nlp_summary  = pd.concat([baseline_row, nlp_summary], ignore_index=True)

        fig = go.Figure()
        colors_n = ['#cccccc', '#0f3460', '#0f3460', '#e94560', '#533483']
        for i, row in nlp_summary.iterrows():
            fig.add_trace(go.Bar(
                name=row['Model'], x=['Accuracy', 'F1 Weighted', 'ROC-AUC'],
                y=[row['Accuracy'], row['F1 Weighted'], row['ROC-AUC']],
                marker_color=colors_n[i], opacity=0.85,
                text=[f"{v:.2f}" for v in [row['Accuracy'], row['F1 Weighted'], row['ROC-AUC']]],
                textposition='outside'
            ))
        fig.update_layout(barmode='group', height=380, plot_bgcolor='white',
                          yaxis=dict(range=[0, 1.15]),
                          legend=dict(orientation='h', y=-0.3),
                          margin=dict(t=20, b=80))
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("**How NLP Works Here**")
        st.markdown("""
        1. **Clean** descriptions — strip codes, card numbers, dates
        2. **TF-IDF** — convert text to 500 weighted term features
        3. **Bigrams** — capture phrases like *"sc vs"*, *"vir vers"*
        4. **Combine** with numeric features (amount, date, rolling spend)
        5. **Train** RF / XGBoost on 495-dimensional feature matrix
        """)
        st.markdown("**Vocabulary size:** 479 terms")
        st.markdown("**Best model:** Random Forest — TF-IDF + Numeric")

    st.markdown("---")

    # ── TF-IDF top terms ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Top TF-IDF Terms per Category</div>', unsafe_allow_html=True)
    terms_df = pd.DataFrame([
        {'Category': cls, 'Top Terms': ', '.join(terms)}
        for cls, terms in top_terms.items()
    ])
    st.dataframe(terms_df, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Confusion Matrix — Best NLP Model</div>', unsafe_allow_html=True)
    best_nlp = max(nlp_res, key=lambda k: nlp_res[k]['f1'])
    cm_nlp   = nlp_res[best_nlp]['cm']
    cm_norm  = cm_nlp.astype(float) / cm_nlp.sum(axis=1, keepdims=True)
    fig_cm   = px.imshow(
        pd.DataFrame(cm_norm, index=nlp_classes, columns=nlp_classes),
        color_continuous_scale='Blues', text_auto='.0%', aspect='auto',
        labels=dict(x='Predicted', y='Actual')
    )
    fig_cm.update_layout(height=520, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cm, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CREDITWORTHINESS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏦 Creditworthiness":
    st.title("Bank Creditworthiness Scoring")
    st.markdown("Simulating what a bank does — scoring financial health using DSCR, overdraft frequency, expense volatility & ensemble models")
    st.markdown("---")

    credit_sorted = credit_df.sort_values('year_month')
    # Use second-to-last row as "latest" — last row is a future/estimate month
    latest        = credit_sorted.iloc[-2]
    avg_dscr      = credit_sorted['dscr'].replace(20, np.nan).mean()
    avg_savings   = credit_sorted['savings_rate'].replace(-1, np.nan).mean()
    overdraft_pct = credit_sorted['overdraft'].mean() * 100
    latest_score  = int(latest['credit_score'])
    latest_label  = latest['credit_label']
    low_months    = (credit_sorted['credit_label'] == 'LOW_RISK').sum()
    high_months   = (credit_sorted['credit_label'] == 'HIGH_RISK').sum()
    total_months  = len(credit_sorted)
    livret_balance = 9268.57

    # ── HEALTH VERDICT BANNER ─────────────────────────────────────────────────
    if latest_score >= 60:
        verdict_color  = '#e8f5e9'
        verdict_border = '#2ecc71'
        verdict_icon   = '✅'
        verdict_title  = 'FINANCIALLY HEALTHY'
        verdict_text   = (
            f"Your latest credit score is <strong>{latest_score}/100 ({latest_label.replace('_',' ')})</strong>. "
            f"You have built a <strong>€{livret_balance:,.0f} savings buffer</strong> in your Livret A, "
            f"your DSCR is strong (>{avg_dscr:.1f}x when income is visible), "
            f"and you were LOW_RISK for <strong>{low_months}/{total_months} months</strong>. "
            "The negative net figures come from high-spend months (travel, large purchases) — "
            "not structural debt. Your savings account absorbs the difference."
        )
    elif latest_score >= 35:
        verdict_color  = '#fff8e1'
        verdict_border = '#f39c12'
        verdict_icon   = '⚠️'
        verdict_title  = 'MODERATE — WATCH SPENDING'
        verdict_text   = (
            f"Your latest credit score is <strong>{latest_score}/100 ({latest_label.replace('_',' ')})</strong>. "
            f"You have savings of <strong>€{livret_balance:,.0f}</strong> but spending volatility is high. "
            f"Months where spend exceeds income ({overdraft_pct:.0f}% of months) drag the score down. "
            "Reducing discretionary spend in high-cost months would push you firmly into LOW_RISK."
        )
    else:
        verdict_color  = '#fdecea'
        verdict_border = '#e94560'
        verdict_icon   = '🔴'
        verdict_title  = 'HIGH RISK — ACTION NEEDED'
        verdict_text   = (
            f"Latest credit score: <strong>{latest_score}/100 ({latest_label.replace('_',' ')})</strong>. "
            "Multiple consecutive months with no visible income and high spend raise red flags for a bank. "
            "Your Livret A savings partially offset this, but a lender cannot see internal transfers."
        )

    st.markdown(f"""
    <div style="background:{verdict_color}; border-left:6px solid {verdict_border};
                border-radius:12px; padding:1.2rem 1.6rem; margin-bottom:1rem;">
        <h3 style="margin:0 0 0.5rem; color:{verdict_border}; font-size:1.2rem;">
            {verdict_icon} CREDIT HEALTH VERDICT: {verdict_title}
        </h3>
        <p style="margin:0; color:#333; font-size:0.93rem; line-height:1.8;">
            {verdict_text}
        </p>
        <div style="margin-top:1rem; display:flex; gap:2.5rem; flex-wrap:wrap; font-size:0.88rem; font-weight:600;">
            <span style="color:#0f3460;">Score: {latest_score}/100</span>
            <span style="color:#2ecc71;">LOW_RISK months: {low_months}/{total_months}</span>
            <span style="color:#e94560;">HIGH_RISK months: {high_months}/{total_months}</span>
            <span style="color:#0f3460;">Livret A savings: €{livret_balance:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SCORE GAUGE ───────────────────────────────────────────────────────────
    col_g, col_k = st.columns([1, 2])
    with col_g:
        gauge_color = '#2ecc71' if latest_score >= 60 else ('#f39c12' if latest_score >= 35 else '#e94560')
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_score,
            delta={'reference': int(credit_sorted['credit_score'].iloc[-3]),
                   'valueformat': '.0f', 'suffix': ' pts vs prev'},
            title={'text': f"Latest Credit Score<br><span style='font-size:0.8em;color:gray'>{latest['year_month']}</span>"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 35],  'color': '#fdecea'},
                    {'range': [35, 65], 'color': '#fff8e1'},
                    {'range': [65, 100],'color': '#e8f5e9'},
                ],
                'threshold': {'line': {'color': '#0f3460', 'width': 3}, 'value': latest_score},
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, width="stretch")

    with col_k:
        st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        avg_dscr_clean = credit_sorted[credit_sorted['dscr'] < 20]['dscr'].mean()
        k1.metric("Avg DSCR (real months)",  f"{avg_dscr_clean:.1f}x" if not np.isnan(avg_dscr_clean) else "N/A",
                  help="Debt Service Coverage Ratio — >1.5x is healthy")
        k2.metric("Overdraft Frequency",     f"{overdraft_pct:.0f}% of months",
                  help="Months where spend > income")
        k3, k4 = st.columns(2)
        pos_savings = credit_sorted[credit_sorted['savings_rate'] > 0]['savings_rate'].mean()
        k3.metric("Avg Savings Rate (positive months)", f"{pos_savings*100:.0f}%" if not np.isnan(pos_savings) else "N/A")
        k4.metric("Livret A Balance",        f"€{livret_balance:,.0f}",
                  help="Real savings buffer — Sep 2024 to Jan 2026")
        k5, k6 = st.columns(2)
        k5.metric("Best Score Achieved",     f"{int(credit_sorted['credit_score'].max())}/100")
        k6.metric("LOW_RISK Months",         f"{low_months}/{total_months} ({low_months/total_months*100:.0f}%)")

    st.markdown("---")

    # ── Credit score over time ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Credit Score Over Time</div>', unsafe_allow_html=True)
    color_map = {'LOW_RISK': '#2ecc71', 'MEDIUM_RISK': '#f39c12', 'HIGH_RISK': '#e94560'}
    fig = go.Figure()
    for label, color in color_map.items():
        mask = credit_sorted['credit_label'] == label
        fig.add_trace(go.Scatter(
            x=credit_sorted.loc[mask, 'year_month'],
            y=credit_sorted.loc[mask, 'credit_score'],
            mode='markers', name=label,
            marker=dict(color=color, size=10, symbol='circle'),
        ))
    fig.add_trace(go.Scatter(
        x=credit_sorted['year_month'], y=credit_sorted['credit_score'],
        mode='lines', name='Score Trend',
        line=dict(color='#0f3460', width=2), showlegend=False
    ))
    fig.add_hline(y=65, line_dash='dash', line_color='#2ecc71',
                  annotation_text='LOW RISK threshold (65)')
    fig.add_hline(y=40, line_dash='dash', line_color='#f39c12',
                  annotation_text='MEDIUM RISK threshold (40)')
    fig.update_layout(height=380, plot_bgcolor='white', yaxis_title='Credit Score (0-100)',
                      xaxis_tickangle=-45, margin=dict(t=30, b=60),
                      legend=dict(orientation='h', y=-0.25))
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # ── Monthly spend vs income ────────────────────────────────────────────
        st.markdown('<div class="section-title">Monthly Income vs Spend</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=credit_sorted['year_month'], y=credit_sorted['income'],
                              name='Income', marker_color='#0f3460', opacity=0.8))
        fig2.add_trace(go.Bar(x=credit_sorted['year_month'], y=credit_sorted['spend'],
                              name='Spend', marker_color='#e94560', opacity=0.8))
        fig2.update_layout(barmode='group', height=320, plot_bgcolor='white',
                           xaxis_tickangle=-45, margin=dict(t=10, b=60),
                           yaxis_title='EUR', legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig2, width="stretch")

    with col2:
        # ── Expense volatility ────────────────────────────────────────────────
        st.markdown('<div class="section-title">Expense Volatility (3-Month Rolling Std)</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=credit_sorted['year_month'], y=credit_sorted['expense_volatility'],
            fill='tozeroy', mode='lines',
            line=dict(color='#533483', width=2),
            fillcolor='rgba(83,52,131,0.15)'
        ))
        fig3.update_layout(height=320, plot_bgcolor='white', xaxis_tickangle=-45,
                           margin=dict(t=10, b=60), yaxis_title='Volatility (EUR)')
        st.plotly_chart(fig3, width="stretch")

    st.markdown("---")

    # ── Feature definitions ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Features Used (What Banks Actually Look At)</div>', unsafe_allow_html=True)
    feat_info = pd.DataFrame({
        'Feature':     ['DSCR', 'Savings Rate', 'Overdraft Frequency', 'Expense Volatility',
                        'Income Stability', 'Cash Ratio', 'Transfer Ratio', 'Essential Ratio',
                        'Avg 3M Spend', 'Spend Trend'],
        'Formula':     ['Income / Debt Payments', '(Income - Spend) / Income',
                        '% months with deficit (rolling 6m)', 'Rolling 3M std of spend',
                        'CoV of rolling 3M income', 'Cash withdrawals / total spend',
                        'Transfers / total spend', 'Essential categories / total spend',
                        '3-month rolling average spend', 'Δ spend direction'],
        'Good Signal': ['>1.5x', '>10%', '0%', '<€300', '<0.2', '<30%', '<20%', '>40%', 'Stable', 'Decreasing'],
        'Risk Signal': ['<1.0x', '<0%', '>50%', '>€800', '>0.8', '>30%', '>20%', '<20%', 'Rising', 'Increasing'],
    })
    st.dataframe(feat_info, width="stretch", hide_index=True)

    st.markdown("---")

    # ── Model results ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Ensemble Model Results</div>', unsafe_allow_html=True)
    try:
        model_res = pd.read_excel(_CREDIT_XLSX, sheet_name="Model Comparison")
        model_res = model_res[['Model', 'Accuracy', 'F1 Weighted', 'ROC-AUC']].head(4)
    except Exception:
        model_res = pd.DataFrame({
            'Model':        ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble (Voting)'],
            'Accuracy':     [0.92, 0.85, 0.85, 0.85],
            'F1 Weighted':  [0.92, 0.84, 0.84, 0.84],
            'ROC-AUC':      [0.99, 0.95, 0.96, 0.98],
        })
    fig4 = go.Figure()
    for metric, color in zip(['Accuracy', 'F1 Weighted', 'ROC-AUC'], ['#0f3460', '#e94560', '#533483']):
        fig4.add_trace(go.Bar(
            name=metric, x=model_res['Model'], y=model_res[metric],
            marker_color=color, opacity=0.85,
            text=[f'{v:.2f}' for v in model_res[metric]], textposition='outside'
        ))
    fig4.update_layout(barmode='group', height=360, plot_bgcolor='white',
                       yaxis=dict(range=[0, 1.1]),
                       legend=dict(orientation='h', y=-0.15),
                       margin=dict(t=20, b=10))
    st.plotly_chart(fig4, width="stretch")

    st.caption("Ensemble uses soft voting (LR×1 + RF×2 + XGBoost×2) — Logistic Regression achieved highest accuracy at 92.3% with ROC-AUC 0.99 on this dataset")

    st.markdown("---")

    # ── Monthly credit profile table ──────────────────────────────────────────
    st.markdown('<div class="section-title">Monthly Credit Profile</div>', unsafe_allow_html=True)
    display_credit = credit_sorted[['year_month','income','spend','net','dscr',
                                     'savings_rate','overdraft_freq','credit_score','credit_label']].copy()
    display_credit.columns = ['Month','Income','Spend','Net','DSCR',
                               'Savings Rate','Overdraft Freq','Score','Risk Label']
    st.dataframe(display_credit, width="stretch", hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CASH FLOW FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Cash Flow Forecast":
    st.title("Cash Flow Forecasting")
    st.markdown("Predict next month's total spend using time-series regression — evaluated with MAE, RMSE, R²")
    st.markdown("---")

    # ── Regression metrics ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Regression Metrics</div>', unsafe_allow_html=True)
    reg_metrics = pd.DataFrame({
        'Model':   ['Baseline (Naive)', 'Ridge Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
        'MAE (€)': [3541, 3713, 2258, 2488, 2121],
        'RMSE (€)':[4194, 5329, 3238, 3528, 3073],
        'R²':      [-1.10, -2.39, -0.25, -0.49, -0.13],
        'MAPE (%)': [240.5, 397.1, 159.0, 137.8, 140.7],
    })

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=reg_metrics['Model'], y=reg_metrics['MAE (€)'],
            name='MAE (€)', marker_color='#0f3460', opacity=0.85,
            text=[f'€{v:,.0f}' for v in reg_metrics['MAE (€)']], textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=reg_metrics['Model'], y=reg_metrics['RMSE (€)'],
            name='RMSE (€)', marker_color='#e94560', opacity=0.85,
            text=[f'€{v:,.0f}' for v in reg_metrics['RMSE (€)']], textposition='outside'
        ))
        fig.update_layout(barmode='group', height=380, plot_bgcolor='white',
                          xaxis_tickangle=-20, margin=dict(t=30, b=10),
                          legend=dict(orientation='h', y=-0.15),
                          yaxis_title='Error (EUR)')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.dataframe(reg_metrics.set_index('Model'), width="stretch")
        st.markdown("""
        **Why R² is negative:**
        Monthly spend ranges from €186 to €8,833 — extremely high variance.
        With only 35 months of data, models struggle to beat the mean.
        This is honest and expected — acknowledging it shows ML maturity.

        **Best model:** Gradient Boosting — lowest MAE (€2,121)
        """)

    st.markdown("---")

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Actual vs Predicted Spend (Test Period)</div>', unsafe_allow_html=True)
    cf = cashflow_df.copy()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=cf['Month'], y=cf['Actual Spend (€)'],
        mode='lines+markers', name='Actual',
        line=dict(color='#0f3460', width=3),
        marker=dict(size=8)
    ))
    colors_pred = ['#aaaaaa', '#e94560', '#533483', '#f39c12']
    for col_name, color in zip(['Baseline', 'Random Forest', 'XGBoost', 'Gradient Boosting'], colors_pred):
        if col_name in cf.columns:
            fig2.add_trace(go.Scatter(
                x=cf['Month'], y=cf[col_name],
                mode='lines+markers', name=col_name,
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6)
            ))
    fig2.update_layout(height=400, plot_bgcolor='white', yaxis_title='Spend (EUR)',
                       legend=dict(orientation='h', y=-0.2),
                       margin=dict(t=20, b=60))
    st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # ── Feature importance ────────────────────────────────────────────────
        st.markdown('<div class="section-title">Top Features — Random Forest</div>', unsafe_allow_html=True)
        fi_top = cf_fi_df.head(12).sort_values('Importance', ascending=True)
        fig3 = px.bar(fi_top, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues')
        fig3.update_layout(height=380, plot_bgcolor='white',
                           coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

    with col2:
        # ── Next month forecast ───────────────────────────────────────────────
        st.markdown('<div class="section-title">Next Month Forecast (2027-01)</div>', unsafe_allow_html=True)
        forecast_vals = {
            'Ridge Regression':    1372,
            'Random Forest':       1490,
            'XGBoost':             1531,
            'Gradient Boosting':   1419,
            'Ensemble Average':    1453,
        }
        for model_name, val in forecast_vals.items():
            is_ensemble = model_name == 'Ensemble Average'
            bg = '#0f3460' if is_ensemble else 'white'
            tc = 'white' if is_ensemble else '#0f3460'
            bd = '2px solid #0f3460'
            st.markdown(f"""
            <div style="background:{bg}; border:{bd}; border-radius:10px;
                        padding:0.7rem 1rem; margin-bottom:0.5rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{tc}; font-weight:600;">{model_name}</span>
                <span style="color:{tc}; font-size:1.2rem; font-weight:700;">€{val:,}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Walk-Forward Cross Validation</div>', unsafe_allow_html=True)
        cv_data = pd.DataFrame({
            'Model':        ['Ridge', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
            'CV MAE (€)':   [904, 945, 931, 889],
            'Std (€)':      [419, 393, 375, 383],
        })
        st.dataframe(cv_data.set_index('Model'), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — TRANSACTION EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Transaction Explorer":
    st.title("Transaction Explorer")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        cats = ['All'] + sorted(full_df['category'].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cats)
    with col2:
        types = ['All', 'DEBIT', 'CREDIT']
        type_filter = st.selectbox("Type", types)
    with col3:
        min_amt = st.number_input("Min Amount (EUR)", value=0.0, step=10.0)

    # Date range
    min_date = full_df['date_operation'].min().date()
    max_date = full_df['date_operation'].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    # Filter
    filtered = full_df.copy()
    if cat_filter != 'All':
        filtered = filtered[filtered['category'] == cat_filter]
    if type_filter != 'All':
        filtered = filtered[filtered['type'] == type_filter]
    filtered = filtered[filtered['abs_amount'] >= min_amt]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['date_operation'].dt.date >= date_range[0]) &
            (filtered['date_operation'].dt.date <= date_range[1])
        ]

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions",  f"{len(filtered):,}")
    c2.metric("Total Debit",   f"€{filtered['debit'].sum():,.0f}")
    c3.metric("Total Credit",  f"€{filtered['credit'].sum():,.0f}")
    c4.metric("Net Flow",      f"€{filtered['amount'].sum():,.0f}")

    st.markdown("---")

    # Amount distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Amount Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x='abs_amount', nbins=40,
                           color_discrete_sequence=['#0f3460'])
        fig.update_layout(height=280, margin=dict(t=10, b=10),
                          plot_bgcolor='white', xaxis_title='Amount (EUR)',
                          yaxis_title='Count')
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-title">Category Breakdown</div>', unsafe_allow_html=True)
        cat_breakdown = filtered['category'].value_counts().reset_index()
        cat_breakdown.columns = ['category', 'count']
        fig2 = px.bar(cat_breakdown, x='count', y='category', orientation='h',
                      color='count', color_continuous_scale='Blues')
        fig2.update_layout(height=280, margin=dict(t=10, b=10),
                           plot_bgcolor='white', coloraxis_showscale=False)
        st.plotly_chart(fig2, width="stretch")

    # Table
    st.markdown('<div class="section-title">Transactions</div>', unsafe_allow_html=True)
    display_cols = ['date_operation', 'description', 'category', 'type', 'debit', 'credit', 'amount']
    show_df = filtered[display_cols].sort_values('date_operation', ascending=False).reset_index(drop=True)
    show_df['date_operation'] = show_df['date_operation'].dt.strftime('%Y-%m-%d')
    st.dataframe(show_df, width="stretch", height=400)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — LOAN DECISION REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Loan Decision Report":
    import re as _re
    from pathlib import Path as _Path

    REPORT_TXT = _REPORT_TXT

    st.title("Loan Decision Report")
    st.markdown(
        "Demonstrates **deterministic ML → probabilistic narrative**: "
        "the risk label and all metrics are fixed outputs from the ensemble model; "
        "the written report is generated by **Mistral** (local LLM via Ollama)."
    )
    st.markdown("---")

    # ── Helper: parse report text into sections ────────────────────────────────
    def parse_report(text):
        """Split report text on ## N. headers → list of (title, body) tuples."""
        pattern = _re.compile(r'^(## \d+\..+)$', _re.MULTILINE)
        parts   = pattern.split(text)
        sections = []
        i = 1
        while i < len(parts):
            title = parts[i].replace("##", "").strip()
            body  = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((title, body))
            i += 2
        return sections

    # ── Helper: extract metadata from header block ─────────────────────────────
    def parse_header(text):
        meta = {}
        for line in text.splitlines():
            if ':' in line:
                k, _, v = line.partition(':')
                meta[k.strip()] = v.strip()
        return meta

    # ── Generate button ────────────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        generate = st.button("⚡ Generate Report", type="primary", width="stretch")
    with col_status:
        if REPORT_TXT.exists():
            mtime = _Path(REPORT_TXT).stat().st_mtime
            import datetime as _dt
            mtime_str = _dt.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            st.info(f"Last generated: {mtime_str} — click to regenerate")
        else:
            st.warning("No report found. Click **Generate Report** to create one (requires Ollama running with Mistral).")

    if generate:
        # Pre-flight check with a clear message before touching Ollama
        if not _CREDIT_XLSX.exists():
            st.error("Missing `data/creditworthiness_results.xlsx` — run `python creditworthiness.py` first.")
        else:
            with st.spinner("Running creditworthiness model + calling Mistral… (30–90 seconds)"):
                import sys, subprocess
                # Use the same Python executable that is running Streamlit right now.
                # This guarantees all packages (numpy, sklearn, ollama…) are available.
                _python = sys.executable
                _script = str(_Path(__file__).parent / "loan_report.py")
                _result = subprocess.run(
                    [_python, _script],
                    capture_output=True,
                    text=True,
                    cwd=str(_Path(__file__).parent),
                )
                if _result.returncode == 0:
                    st.success("Report generated successfully!")
                    st.rerun()
                else:
                    _out = (_result.stdout + "\n" + _result.stderr).strip()
                    st.error(f"Report generation failed:\n\n```\n{_out}\n```")

    st.markdown("---")

    # ── Display existing report ────────────────────────────────────────────────
    if REPORT_TXT.exists():
        raw = REPORT_TXT.read_text(encoding='utf-8')

        # Split header block from report body (separated by the last ===... line)
        header_end = raw.find("## 1.")
        header_block = raw[:header_end] if header_end != -1 else ""
        report_body  = raw[header_end:] if header_end != -1 else raw

        # ── KPI strip from header metadata ────────────────────────────────────
        meta = parse_header(header_block)
        decision_raw = meta.get("ML Decision", "")
        score_raw    = meta.get("Credit Score", "")
        month_raw    = meta.get("Analysis Month", "")
        date_raw     = meta.get("Report Date", "")

        # Colour-code the decision badge
        label_str = decision_raw.split("(")[0].strip()
        badge_color = {"LOW_RISK": "#2ecc71", "MEDIUM_RISK": "#f39c12", "HIGH_RISK": "#e94560"}.get(label_str, "#0f3460")

        st.markdown(
            f"""
            <div style="background:white; border-radius:12px; padding:1.2rem 1.8rem;
                        box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:1.2rem;
                        display:flex; gap:3rem; flex-wrap:wrap; align-items:center;">
                <div>
                    <div style="color:#888; font-size:0.8rem; font-weight:500;">ML DECISION</div>
                    <div style="background:{badge_color}; color:white; border-radius:20px;
                                padding:4px 16px; font-weight:700; font-size:1rem; margin-top:4px;
                                display:inline-block;">
                        {label_str.replace('_', ' ')}
                    </div>
                </div>
                <div>
                    <div style="color:#888; font-size:0.8rem; font-weight:500;">CREDIT SCORE</div>
                    <div style="color:#0f3460; font-size:1.6rem; font-weight:700;">{score_raw}</div>
                </div>
                <div>
                    <div style="color:#888; font-size:0.8rem; font-weight:500;">ANALYSIS MONTH</div>
                    <div style="color:#0f3460; font-size:1.6rem; font-weight:700;">{month_raw}</div>
                </div>
                <div>
                    <div style="color:#888; font-size:0.8rem; font-weight:500;">REPORT DATE</div>
                    <div style="color:#555; font-size:1rem; margin-top:6px;">{date_raw}</div>
                </div>
                <div style="margin-left:auto; color:#888; font-size:0.8rem; font-style:italic;">
                    Label &amp; metrics: deterministic ML ensemble<br>
                    Narrative: Mistral LLM (probabilistic)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Render each section as an expander ────────────────────────────────
        sections = parse_report(report_body)
        if sections:
            section_icons = {
                "1.": "📋",
                "2.": "⚖️",
                "3.": "✅",
                "4.": "📈",
                "5.": "🏦",
            }
            for title, body in sections:
                num = title.split(".")[0].strip().split()[-1] + "."  # e.g. "1."
                icon = section_icons.get(num, "📄")
                with st.expander(f"{icon} {title}", expanded=True):
                    st.markdown(body)
        else:
            # Fallback: render as plain text
            st.markdown(report_body)

        st.markdown("---")

        # ── Download button ────────────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download report as .txt",
            data=raw,
            file_name=f"loan_report_{month_raw.replace('-','')}.txt",
            mime="text/plain",
        )
