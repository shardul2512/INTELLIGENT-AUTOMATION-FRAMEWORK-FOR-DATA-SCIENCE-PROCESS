"""
⚗️ Intelligent Automation Framework for Data Science Process Automation
Streamlit app · AutoML · MLflow · SHAP · XAI Business Advisor
Pure AutoML Engine.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Automation Framework",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS + agents ────────────────────────────────────────────────────────
from styles import CSS
from agents import AGENTS, call_openrouter
from ml_engine import HAS_H2O, HAS_SHAP

st.markdown(CSS, unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None, "df_clean": None, "filename": None,
        "pipeline_steps": [],
        "openrouter_key": os.environ.get("OPENROUTER_API_KEY", ""), "model": "anthropic/claude-3-haiku",
        "chat_history": {},
        "h2o_result": None,
        "shap_data": None,
        "xai_report": None,
        "viz_code": None,
        "logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helpers ─────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%H:%M:%S")

def add_log(msg, kind="i"):
    st.session_state.logs.append({"msg": msg, "kind": kind, "ts": ts()})

def add_step(sid, name, icon, result=None):
    st.session_state.pipeline_steps = [s for s in st.session_state.pipeline_steps if s["id"] != sid]
    st.session_state.pipeline_steps.append(
        {"id": sid, "name": name, "icon": icon, "result": result or {}, "ts": ts()}
    )

def render_pipeline():
    steps = [
        ("📂", "Load Data",  "load"),
        ("🧹", "Clean",      "cleaner"),
        ("📊", "Visualize",  "viz"),
        ("🤖", "AutoML",     "ml"),
        ("🔍", "XAI",        "xai"),
        ("📋", "MLflow",     "mlflow"),
    ]
    done = {s["id"] for s in st.session_state.pipeline_steps}
    html = ""
    for i, (icon, lbl, sid) in enumerate(steps):
        cls  = "done" if sid in done else "idle"
        lcls = "done" if sid in done else ""
        if i > 0:
            ccon = "done" if steps[i-1][2] in done else "idle"
            html += f'<div class="pipe-conn {ccon}"></div>'
        html += (f'<div class="pipe-step"><div class="pipe-node {cls}">{icon}</div>'
                 f'<div class="pipe-lbl {lcls}">{lbl}</div></div>')
    st.markdown(
        f'<div class="pipeline-wrap"><div class="pipeline-label">▸ LIVE PIPELINE</div>'
        f'<div class="pipe-track">{html}</div></div>',
        unsafe_allow_html=True,
    )

def render_chat(chat_id, agent_label="Agent"):
    history = st.session_state.chat_history.get(chat_id, [])
    for msg in history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-u"><div class="chat-lbl">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            content = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-a"><div class="chat-lbl">{agent_label}</div>{content}</div>',
                unsafe_allow_html=True,
            )

def chat_send(chat_id, user_input, system_prompt, extra_context=""):
    full_system = system_prompt + (f"\n\n{extra_context}" if extra_context else "")
    history     = st.session_state.chat_history.get(chat_id, [])
    response    = call_openrouter(
        messages=history + [{"role": "user", "content": user_input}],
        system=full_system,
    )
    st.session_state.chat_history.setdefault(chat_id, [])
    st.session_state.chat_history[chat_id].append({"role": "user",      "content": user_input})
    st.session_state.chat_history[chat_id].append({"role": "assistant", "content": response})
    return response

def df_ctx(df):
    nc = list(df.select_dtypes(include="number").columns)
    cc = list(df.select_dtypes(include="object").columns)
    return (f"Dataset: {st.session_state.filename}  Shape: {df.shape}\n"
            f"Numeric: {nc}\nCategorical: {cc}\n"
            f"Missing: {dict(df.isnull().sum())}\n"
            f"Sample:\n{df.head(3).to_string()}")

def render_shap_bars(series, top_n=15, title="SHAP Feature Importance"):
    top = series.head(top_n)
    mx  = top.max() if top.max() > 0 else 1
    bars = ""
    for feat, val in top.items():
        pct  = int(val / mx * 100)
        bars += (f'<div class="shap-bar-wrap">'
                 f'<div class="shap-feature">{feat}</div>'
                 f'<div style="display:flex;align-items:center;gap:8px;">'
                 f'<div class="shap-bar-bg" style="flex:1;">'
                 f'<div class="shap-bar-pos" style="width:{pct}%;"></div></div>'
                 f'<div class="shap-val">{val:.4f}</div></div></div>')
    st.markdown(
        f'<div class="card"><div class="sh">{title}</div>{bars}</div>',
        unsafe_allow_html=True,
    )

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:6px 0 18px;">
      <div style="font-family:'Space Mono',monospace;font-size:17px;font-weight:700;
           background:linear-gradient(135deg,#00d4ff,#00ff88);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚗️ Intelligent Automation Framework</div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;">AutoML · MLflow · SHAP · OpenRouter</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">🔑 OPENROUTER</div>', unsafe_allow_html=True)
    if not st.session_state.openrouter_key:
        st.warning("API Key not found in .env (OPENROUTER_API_KEY).", icon="⚠️")
    else:
        st.success("API Key loaded securely from .env", icon="🔒")

    MODEL_OPTIONS = {
        "Claude 3 Haiku (Fast)": "anthropic/claude-3-haiku",
        "Claude 3.5 Sonnet":     "anthropic/claude-3-5-sonnet",
        "GPT-4o Mini":           "openai/gpt-4o-mini",
        "GPT-4o":                "openai/gpt-4o",
        "Gemini Flash 1.5":      "google/gemini-flash-1.5",
        "Llama 3.1 70B":         "meta-llama/llama-3.1-70b-instruct",
        "Mistral Large":         "mistralai/mistral-large",
        "DeepSeek Chat":         "deepseek/deepseek-chat",
    }
    sel = st.selectbox("Model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")
    st.session_state.model = MODEL_OPTIONS[sel]
    pill = ('<span class="pill ok">● CONNECTED</span>'
            if st.session_state.openrouter_key
            else '<span class="pill idle">○ NO API KEY</span>')
    st.markdown(pill, unsafe_allow_html=True)

    st.markdown('<div class="sh">📁 DATASET</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"],
                                label_visibility="collapsed")
    if uploaded:
        if st.session_state.get("last_uploaded_id") != uploaded.file_id:
            try:
                df_raw = (pd.read_csv(uploaded)
                          if uploaded.name.endswith(".csv")
                          else pd.read_excel(uploaded))
                st.session_state.df        = df_raw
                st.session_state.df_clean  = df_raw.copy()
                st.session_state.filename  = uploaded.name
                st.session_state.pipeline_steps = []
                st.session_state.h2o_result     = None
                st.session_state.shap_data      = None
                st.session_state.last_uploaded_id = uploaded.file_id
                add_step("load", "Load Data", "📂", {"rows": len(df_raw), "cols": len(df_raw.columns)})
                add_log(f"Loaded {uploaded.name} ({len(df_raw)} rows × {len(df_raw.columns)} cols)", "s")
            except Exception as e:
                st.error(f"Load error: {e}")

    st.markdown('<div class="sh">📦 SAMPLE DATASETS</div>', unsafe_allow_html=True)
    sample_choice = st.selectbox(
        "Sample dataset",
        ["— select —", "Telco Churn", "House Prices", "Iris Classification"],
        label_visibility="collapsed",
    )
    if st.button("Load Sample", use_container_width=True) and sample_choice != "— select —":
        rng = np.random.default_rng(42)
        n   = 1000
        if sample_choice == "Telco Churn":
            df_raw = pd.DataFrame({
                "tenure":          rng.integers(0, 72, n),
                "MonthlyCharges":  rng.uniform(20, 120, n).round(2),
                "TotalCharges":    rng.uniform(100, 8000, n).round(2),
                "SeniorCitizen":   rng.integers(0, 2, n),
                "NumSupportCalls": rng.integers(0, 10, n),
                "Contract":        rng.choice(["Month-to-month", "One year", "Two year"], n),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
                "PaymentMethod":   rng.choice(["Electronic check", "Mailed check",
                                               "Bank transfer", "Credit card"], n),
                "TechSupport":     rng.choice(["Yes", "No"], n),
                "OnlineBackup":    rng.choice(["Yes", "No"], n),
            })
            churn_prob = (
                0.4 * (df_raw["Contract"] == "Month-to-month").astype(float)
                + 0.3 * (df_raw["MonthlyCharges"] > 70).astype(float)
                - 0.2 * (df_raw["tenure"] > 36).astype(float)
                + 0.1 * rng.uniform(0, 1, n)
            )
            df_raw["Churn"] = (churn_prob > 0.45).map({True: "Yes", False: "No"})
            fname = "sample_telco_churn.csv"
        elif sample_choice == "House Prices":
            df_raw = pd.DataFrame({
                "GrLivArea":    rng.integers(600, 4000, n),
                "OverallQual":  rng.integers(1, 10, n),
                "YearBuilt":    rng.integers(1900, 2023, n),
                "TotalBsmtSF":  rng.integers(0, 3000, n),
                "GarageCars":   rng.integers(0, 4, n),
                "FullBath":     rng.integers(1, 4, n),
                "BedroomAbvGr": rng.integers(1, 6, n),
                "Neighborhood": rng.choice(["OldTown", "NridgHt", "CollgCr",
                                            "Somerst", "Gilbert"], n),
                "HouseStyle":   rng.choice(["1Story", "2Story", "1.5Fin"], n),
            })
            df_raw["SalePrice"] = (
                df_raw["GrLivArea"] * 65
                + df_raw["OverallQual"] * 8000
                + rng.normal(0, 15000, n)
            ).clip(50000, 800_000).round(-2)
            fname = "sample_house_prices.csv"
        else:  # Iris
            from sklearn.datasets import load_iris
            iris   = load_iris(as_frame=True)
            df_raw = iris.frame.copy()
            df_raw["target"] = df_raw["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
            df_raw.columns   = [c.replace(" (cm)", "").replace(" ", "_") for c in df_raw.columns]
            fname  = "sample_iris.csv"
        st.session_state.df        = df_raw
        st.session_state.df_clean  = df_raw.copy()
        st.session_state.filename  = fname
        st.session_state.pipeline_steps = []
        st.session_state.h2o_result     = None
        st.session_state.shap_data      = None
        add_step("load", "Load Data", "📂", {"rows": len(df_raw), "cols": len(df_raw.columns)})
        add_log(f"Loaded sample: {fname} ({len(df_raw)} rows × {len(df_raw.columns)} cols)", "s")
        st.rerun()

    if st.session_state.df is not None:
        dc = st.session_state.df_clean
        st.markdown(f"""<div class="card" style="padding:12px;cursor:default;">
          <div style="font-size:11px;color:#64748b;font-family:'Space Mono',monospace;">ACTIVE DATASET</div>
          <div style="font-size:13px;font-weight:600;margin:4px 0;">{st.session_state.filename}</div>
          <div style="display:flex;gap:14px;margin-top:6px;">
            <div><span style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:16px;">{len(dc):,}</span>
                 <br><span style="font-size:10px;color:#64748b;">rows</span></div>
            <div><span style="font-family:'Space Mono',monospace;color:#00ff88;font-size:16px;">{len(dc.columns)}</span>
                 <br><span style="font-size:10px;color:#64748b;">cols</span></div>
            <div><span style="font-family:'Space Mono',monospace;color:#ff6b35;font-size:16px;">{int(dc.isnull().sum().sum())}</span>
                 <br><span style="font-size:10px;color:#64748b;">missing</span></div>
          </div></div>""", unsafe_allow_html=True)

    if st.session_state.pipeline_steps:
        st.markdown('<div class="sh">⚡ PIPELINE</div>', unsafe_allow_html=True)
        for step in st.session_state.pipeline_steps:
            st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;
                 background:rgba(0,255,136,.04);border-radius:6px;margin-bottom:3px;">
              <span style="font-size:13px;">{step['icon']}</span>
              <div style="flex:1;">
                <div style="font-size:11px;font-weight:600;">{step['name']}</div>
                <div style="font-size:9px;color:#64748b;">{step['ts']}</div>
              </div><span style="color:#00ff88;font-size:11px;">✓</span>
            </div>""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""<div class="wb-header">
  <div class="wb-title">⚗️ INTELLIGENT AUTOMATION FRAMEWORK</div>
  <span class="wb-badge">H2O AUTOML · MLFLOW · SHAP · XAI</span>
</div>""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown("""<div style="text-align:center;padding:50px 20px 30px;">
      <div style="font-size:58px;margin-bottom:14px;">⚗️</div>
      <div style="font-family:'Space Mono',monospace;font-size:22px;font-weight:700;
           background:linear-gradient(135deg,#00d4ff,#00ff88);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px;">
        Intelligent Automation Framework<br>for Data Science Process Automation</div>
      <div style="color:#64748b;font-size:14px;max-width:560px;margin:0 auto 36px;line-height:1.7;">
        Upload a dataset from the sidebar. Then deploy AI agents to clean, visualise, model and explain your data.<br>
        Powered by <strong style="color:#00d4ff;">AutoML</strong> · <strong style="color:#00ff88;">MLflow</strong> ·
        <strong style="color:#c084fc;">SHAP</strong> · <strong style="color:#ff6b35;">OpenRouter LLMs</strong>
      </div></div>""", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (icon, name, cls, desc) in zip(cols, [
        ("🧹", "Data Wrangler",  "cleaner",  "AI-guided cleaning, missing values, type casting"),
        ("📊", "Viz Architect",  "viz",      "Plotly EDA charts with AI-generated code execution"),
        ("🤖", "AutoML",    "ml",       "AutoML leaderboard · GBM · XGBoost · Ensemble · MLflow"),
        ("🔍", "XAI + SHAP",    "explainer","SHAP Shapley values · beeswarm · business recommendations"),
    ]):
        with col:
            st.markdown(
                f'<div class="card {cls}"><div class="agent-row">'
                f'<span class="agent-icon">{icon}</span>'
                f'<span class="agent-name">{name}</span></div>'
                f'<div class="agent-desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )
    st.stop()

# ─── Main tabs ────────────────────────────────────────────────────────────────
render_pipeline()
df = st.session_state.df_clean

tabs = st.tabs(["🗄️ Explorer", "🧹 Wrangler", "📊 Viz", "🤖 AutoML", "🔍 XAI Advisor", "📋 MLflow", "📜 Log"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 · Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    n_miss = int(df.isnull().sum().sum())
    n_num  = len(df.select_dtypes(include="number").columns)
    n_cat  = len(df.select_dtypes(include="object").columns)
    n_dup  = int(df.duplicated().sum())
    st.markdown(f"""<div class="metric-row">
      <div class="metric-card"><div class="mv">{len(df):,}</div><div class="ml">Rows</div></div>
      <div class="metric-card"><div class="mv">{len(df.columns)}</div><div class="ml">Columns</div></div>
      <div class="metric-card"><div class="mv" style="color:#ff6b35">{n_miss:,}</div><div class="ml">Missing</div></div>
      <div class="metric-card"><div class="mv" style="color:#00ff88">{n_num}</div><div class="ml">Numeric</div></div>
      <div class="metric-card"><div class="mv" style="color:#c084fc">{n_cat}</div><div class="ml">Categorical</div></div>
      <div class="metric-card"><div class="mv" style="color:#ffd60a">{n_dup}</div><div class="ml">Duplicates</div></div>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="sh">TABLE PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(df.head(200), use_container_width=True, height=360)
    with c2:
        st.markdown('<div class="sh">COLUMN PROFILE</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "Column":  df.columns,
                "Type":    df.dtypes.astype(str).values,
                "Missing": df.isnull().sum().values,
                "Unique":  df.nunique().values,
            }),
            use_container_width=True, height=360,
        )
    st.markdown('<div class="sh">DESCRIPTIVE STATISTICS</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T.round(3), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Wrangler
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    a    = AGENTS[0]
    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown(
            f'<div class="card cleaner"><div class="agent-row">'
            f'<span class="agent-icon">{a["icon"]}</span>'
            f'<span class="agent-name">{a["name"]}</span>'
            f'<span class="pill ok" style="margin-left:auto;">● READY</span></div>'
            f'<div class="agent-desc">{a["desc"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sh">QUICK ACTIONS</div>', unsafe_allow_html=True)
        q1, q2 = st.columns(2)

        with q1:
            # ── Drop Duplicates ───────────────────────────────────────────────
            if st.button("🗑️ Drop Duplicates", use_container_width=True):
                before = len(st.session_state.df_clean)
                st.session_state.df_clean = (
                    st.session_state.df_clean
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                dropped = before - len(st.session_state.df_clean)
                add_log(f"Dropped {dropped} duplicate rows", "s")
                add_step("cleaner", "Clean: Duplicates", "🧹")
                st.rerun()

            # ── Fill Numeric (median) ─────────────────────────────────────────
            if st.button("📊 Fill Numeric (median)", use_container_width=True):
                df_tmp   = st.session_state.df_clean.copy()
                num_cols = df_tmp.select_dtypes(include="number").columns
                medians  = df_tmp[num_cols].median()
                df_tmp[num_cols] = df_tmp[num_cols].fillna(medians)
                filled = int(st.session_state.df_clean[num_cols].isnull().sum().sum())
                st.session_state.df_clean = df_tmp
                add_log(f"Filled {filled} numeric NaN cells with median", "s")
                add_step("cleaner", "Clean: Fill Median", "🧹")
                st.rerun()

            # ── Fill Categorical (mode) ───────────────────────────────────────
            if st.button("🔤 Fill Categorical (mode)", use_container_width=True):
                df_tmp = st.session_state.df_clean.copy()
                filled = 0
                for c in df_tmp.select_dtypes(include="object").columns:
                    n_miss_c = df_tmp[c].isnull().sum()
                    if n_miss_c > 0:
                        mode_val = df_tmp[c].mode()
                        if len(mode_val):
                            df_tmp[c] = df_tmp[c].fillna(mode_val.iloc[0])
                            filled   += n_miss_c
                st.session_state.df_clean = df_tmp
                add_log(f"Filled {filled} categorical NaN cells with mode", "s")
                add_step("cleaner", "Clean: Fill Mode", "🧹")
                st.rerun()

        with q2:
            # ── Drop Rows w/ NaN ─────────────────────────────────────────────
            if st.button("🚮 Drop Rows w/ NaN", use_container_width=True):
                before = len(st.session_state.df_clean)
                st.session_state.df_clean = (
                    st.session_state.df_clean
                    .dropna()
                    .reset_index(drop=True)
                )
                dropped = before - len(st.session_state.df_clean)
                add_log(f"Dropped {dropped} rows with NaN", "s")
                add_step("cleaner", "Clean: Drop NaN", "🧹")
                st.rerun()

            # ── Object → Numeric ─────────────────────────────────────────────
            if st.button("🔢 Object → Numeric", use_container_width=True):
                df_tmp    = st.session_state.df_clean.copy()
                converted = 0
                for c in df_tmp.select_dtypes(include="object").columns:
                    candidate = pd.to_numeric(df_tmp[c], errors="coerce")
                    # Only convert if ≥ 80 % of non-null values parse as numbers
                    non_null  = df_tmp[c].notna().sum()
                    parsed    = candidate.notna().sum()
                    if non_null > 0 and parsed / non_null >= 0.8:
                        df_tmp[c] = candidate
                        converted += 1
                st.session_state.df_clean = df_tmp
                add_log(f"Converted {converted} object cols to numeric", "s")
                add_step("cleaner", "Clean: Type Cast", "🧹")
                st.rerun()

            # ── Reset to Original ─────────────────────────────────────────────
            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.df_clean = st.session_state.df.copy()
                add_log("Reset to original dataset", "w")
                st.rerun()

        # ── Drop Columns ─────────────────────────────────────────────────────
        st.markdown('<div class="sh">DROP COLUMNS</div>', unsafe_allow_html=True)
        cols_drop = st.multiselect(
            "Columns to drop",
            list(st.session_state.df_clean.columns),
            label_visibility="collapsed",
        )
        if cols_drop and st.button("Drop Selected", use_container_width=True):
            st.session_state.df_clean = st.session_state.df_clean.drop(columns=cols_drop)
            add_log(f"Dropped columns: {cols_drop}", "s")
            add_step("cleaner", "Clean: Drop Cols", "🧹")
            st.rerun()

        # ── Rename / Re-type ─────────────────────────────────────────────────
        st.markdown('<div class="sh">TRIM STRING WHITESPACE</div>', unsafe_allow_html=True)
        if st.button("✂️ Strip Whitespace (all string cols)", use_container_width=True):
            df_tmp = st.session_state.df_clean.copy()
            for c in df_tmp.select_dtypes(include="object").columns:
                df_tmp[c] = df_tmp[c].str.strip()
            st.session_state.df_clean = df_tmp
            add_log("Stripped leading/trailing whitespace from string cols", "s")
            add_step("cleaner", "Clean: Strip WS", "🧹")
            st.rerun()

    with cr:
        st.markdown('<div class="sh">🤖 AI WRANGLER CHAT</div>', unsafe_allow_html=True)
        render_chat("cleaner", "🧹 Wrangler")
        ui = st.text_area(
            "Wrangler message", "",
            placeholder="Ask: What cleaning steps should I take?",
            height=90, key="win", label_visibility="collapsed",
        )
        if st.button("Send to Wrangler 🧹", use_container_width=True, key="wsend"):
            if ui.strip():
                with st.spinner("Wrangler thinking..."):
                    chat_send("cleaner", ui, a["system"], df_ctx(df))
                add_log("Wrangler responded", "a")
                add_step("cleaner", "AI: Wrangler", "🧹")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Viz
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    # px and go are imported at top level

    a        = AGENTS[1]
    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include="object").columns)
    all_cols = list(df.columns)
    
    # Fallback if no columns were parsed as numeric (e.g. dirty data)
    if not num_cols:
        num_cols = all_cols
    cl, cr   = st.columns([1, 1])

    with cl:
        st.markdown(
            f'<div class="card viz"><div class="agent-row">'
            f'<span class="agent-icon">{a["icon"]}</span>'
            f'<span class="agent-name">{a["name"]}</span></div>'
            f'<div class="agent-desc">{a["desc"]}</div></div>',
            unsafe_allow_html=True,
        )
        ct = st.selectbox("Chart type",
                          ["Histogram", "Scatter", "Box Plot",
                           "Correlation Heatmap", "Bar Count", "Violin", "Pair Plot"])
        _layout = dict(paper_bgcolor="#13161f", plot_bgcolor="#13161f")
        if ct == "Histogram":
            col = st.selectbox("Column", num_cols, key="hc")
            nb  = st.slider("Bins", 5, 100, 30)
            if st.button("Generate", use_container_width=True, key="gh"):
                fig = px.histogram(df, x=col, nbins=nb, template="plotly_dark",
                                   color_discrete_sequence=["#00d4ff"])
                fig.update_layout(**_layout)
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Histogram", "📊")
        elif ct == "Scatter":
            xc = st.selectbox("X", num_cols, key="sx")
            yc = st.selectbox("Y", num_cols, index=min(1, max(0, len(num_cols)-1)), key="sy")
            cc = st.selectbox("Color", ["None"] + cat_cols + num_cols, key="sc")
            if st.button("Generate", use_container_width=True, key="gs"):
                fig = px.scatter(df, x=xc, y=yc, color=None if cc=="None" else cc,
                                 template="plotly_dark", opacity=0.7)
                fig.update_layout(**_layout)
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Scatter", "📊")
        elif ct == "Box Plot":
            yc = st.selectbox("Value", num_cols, key="bxy")
            xc = st.selectbox("Group", ["None"] + cat_cols, key="bxx")
            if st.button("Generate", use_container_width=True, key="gbx"):
                fig = px.box(df, x=None if xc=="None" else xc, y=yc,
                             template="plotly_dark", color_discrete_sequence=["#9b5de5"])
                fig.update_layout(**_layout)
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Box", "📊")
        elif ct == "Correlation Heatmap":
            if st.button("Generate", use_container_width=True, key="gcorr"):
                corr = df[num_cols].corr()
                fig  = px.imshow(corr, text_auto=".2f", template="plotly_dark",
                                 color_continuous_scale="RdBu_r", aspect="auto")
                fig.update_layout(paper_bgcolor="#13161f")
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Heatmap", "📊")
        elif ct == "Bar Count":
            if cat_cols:
                col = st.selectbox("Column", cat_cols, key="barc")
                if st.button("Generate", use_container_width=True, key="gbar"):
                    vc          = df[col].value_counts().reset_index()
                    vc.columns  = [col, "count"]
                    fig = px.bar(vc, x=col, y="count", template="plotly_dark",
                                 color_discrete_sequence=["#c084fc"])
                    fig.update_layout(**_layout)
                    st.plotly_chart(fig, use_container_width=True)
                    add_step("viz", "Viz: Bar Count", "📊")
        elif ct == "Violin":
            yc = st.selectbox("Value", num_cols, key="vy")
            xc = st.selectbox("Group", ["None"] + cat_cols, key="vx")
            if st.button("Generate", use_container_width=True, key="gvio"):
                fig = px.violin(df, x=None if xc=="None" else xc, y=yc,
                                box=True, template="plotly_dark",
                                color_discrete_sequence=["#00ff88"])
                fig.update_layout(**_layout)
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Violin", "📊")
        elif ct == "Pair Plot":
            sel = st.multiselect("Dimensions", num_cols, default=num_cols[:4], key="spd")
            cc  = st.selectbox("Color", ["None"] + cat_cols, key="spc")
            if sel and st.button("Generate", use_container_width=True, key="gsp"):
                fig = px.scatter_matrix(df, dimensions=sel,
                                        color=None if cc=="None" else cc,
                                        template="plotly_dark", opacity=0.6)
                fig.update_layout(paper_bgcolor="#13161f")
                st.plotly_chart(fig, use_container_width=True)
                add_step("viz", "Viz: Pair Plot", "📊")

    with cr:
        st.markdown('<div class="sh">🤖 AI VIZ CHAT + CODE EXEC</div>', unsafe_allow_html=True)
        render_chat("viz", "📊 Viz Architect")
        ui = st.text_area(
            "Viz message", "",
            placeholder="Ask: Plot sales by region. Create a churn distribution chart.",
            height=90, key="vin", label_visibility="collapsed",
        )
        if st.button("Send to Viz Agent 📊", use_container_width=True, key="vsend"):
            if ui.strip():
                with st.spinner("Viz agent designing..."):
                    resp = chat_send("viz", ui, AGENTS[1]["system"], df_ctx(df))
                m = re.search(r"```python\n(.*?)```", resp, re.DOTALL)
                if m:
                    st.session_state.viz_code = m.group(1)
                add_log("Viz agent responded", "a")
                add_step("viz", "AI: Viz", "📊")
                st.rerun()

        # ── Code editor + executor ────────────────────────────────────────────
        if st.session_state.viz_code:
            st.markdown('<div class="sh">EXECUTE AI CODE</div>', unsafe_allow_html=True)
            edited = st.text_area(
                "Viz code editor",
                value=st.session_state.viz_code,
                height=180, key="vce", label_visibility="collapsed",
            )
            if st.button("▶ Run Code", use_container_width=True, key="rviz"):
                try:
                    # Strip file-load and display calls — we inject df ourselves
                    sanitized = []
                    for _line in edited.splitlines():
                        _s = _line.strip()
                        if re.search(r'\bpd\.(read_csv|read_excel|read_json|read_parquet|read_table)\s*\(', _s):
                            sanitized.append("# (file load removed — df already injected)")
                        elif re.search(r'\bplt\.show\s*\(', _s):
                            sanitized.append("# (plt.show removed)")
                        elif re.search(r'^st\.', _s):
                            sanitized.append("# (st.* call removed)")
                        else:
                            sanitized.append(_line)
                    clean_code = "\n".join(sanitized)

                    # Build exec namespace with everything the AI code might need
                    exec_ns = {
                        "__builtins__": __builtins__,
                        "df": df.copy(),
                        "pd": pd, "np": np,
                        "px": px, "go": go,
                    }
                    try:
                        import matplotlib.pyplot as _plt
                        import matplotlib as _matplotlib
                        exec_ns["plt"]        = _plt
                        exec_ns["matplotlib"] = _matplotlib
                    except Exception:
                        pass

                    exec(clean_code, exec_ns)

                    # Render whatever the code produced
                    if "fig" in exec_ns and exec_ns["fig"] is not None:
                        exec_ns["fig"].update_layout(
                            paper_bgcolor="#13161f",
                            plot_bgcolor="#13161f",
                            template="plotly_dark",
                        )
                        st.plotly_chart(exec_ns["fig"], use_container_width=True)
                        add_step("viz", "AI: Exec Viz", "📊")
                    elif "plt" in exec_ns:
                        try:
                            st.pyplot(exec_ns["plt"].gcf())
                            exec_ns["plt"].clf()
                            add_step("viz", "AI: Exec Viz (matplotlib)", "📊")
                        except Exception:
                            pass
                    else:
                        st.info("Code ran but produced no figure. Check the code above.")
                except Exception as e:
                    st.error(f"❌ Code error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · AutoML Engine
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    from ml_engine import train_h2o, get_mlflow_runs

    a        = AGENTS[2]
    all_cols = list(df.columns)

    # ── Smart ID-column exclusion ─────────────────────────────────────────────
    def is_id_like(col, series):
        n = len(series.dropna())
        if n == 0:
            return True
        frac = series.nunique() / n
        if series.dtype == object and frac > 0.9:
            return True
        if frac == 1.0:
            return True
        low = col.lower()
        if any(low == p or low.endswith(p)
               for p in ["id", "_id", "uuid", "key", "index",
                         "no", "num", "number", "code"]):
            return True
        return False

    id_cols = [c for c in all_cols if is_id_like(c, df[c])]

    # ── Smart default target ──────────────────────────────────────────────────
    def guess_target(cols, id_c):
        pref = ["churn", "target", "label", "class", "y", "outcome",
                "survived", "default", "fraud", "converted", "purchased",
                "cancelled", "attrition", "response", "saleprice", "price",
                "salary", "revenue", "score"]
        lower_map = {c.lower(): c for c in cols}
        for p in pref:
            if p in lower_map:
                return lower_map[p]
        non_id = [c for c in cols if c not in id_c]
        return non_id[-1] if non_id else cols[-1]

    default_target     = guess_target(all_cols, id_cols)
    default_target_idx = all_cols.index(default_target)

    cl, cr = st.columns([1, 1])

    with cl:
        h2o_badge  = ('<span class="pill ok" style="margin-left:6px;">● H2O</span>'
                      if HAS_H2O else
                      '<span class="pill idle" style="margin-left:6px;">○ H2O MISSING</span>')
        shap_badge_hdr = ('<span class="pill ok" style="margin-left:4px;">● SHAP</span>'
                          if HAS_SHAP else
                          '<span class="pill idle" style="margin-left:4px;">○ SHAP MISSING</span>')
        st.markdown(
            f'<div class="card ml"><div class="agent-row">'
            f'<span class="agent-icon">{a["icon"]}</span>'
            f'<span class="agent-name">{a["name"]}</span>{h2o_badge}{shap_badge_hdr}</div>'
            f'<div class="agent-desc">{a["desc"]}</div></div>',
            unsafe_allow_html=True,
        )

        if not HAS_H2O:
            st.error("H2O is not installed. Run: `pip install h2o`")
            st.stop()

        target_col  = st.selectbox("🎯 Target", all_cols, index=default_target_idx)
        feat_opts   = [c for c in all_cols if c != target_col and c not in id_cols]
        feat_default = feat_opts[:10]
        if id_cols:
            st.caption(f"ℹ️ Auto-excluded ID-like columns: {', '.join(id_cols)}")
        feature_cols = st.multiselect("📐 Features", feat_opts, default=feat_default)
        task_type    = st.selectbox("Task", ["Auto-detect", "Classification", "Regression"])
        task_map     = {"Auto-detect": "auto", "Classification": "classification",
                        "Regression": "regression"}
        max_m = st.slider("Max models",   3, 20,  8)
        max_s = st.slider("Max seconds", 30, 300, 90)

        if st.button("🚀 Run AutoML + SHAP + MLflow", use_container_width=True):
            if not feature_cols:
                st.warning("Select at least one feature column.")
            elif target_col in feature_cols:
                st.warning("Target column cannot be a feature.")
            else:
                with st.spinner(f"🤖 AutoML — {max_m} models · {max_s}s budget · computing SHAP…"):
                    try:
                        result = train_h2o(
                            df=df,
                            target=target_col,
                            features=feature_cols,
                            max_models=max_m,
                            max_secs=max_s,
                            task=task_map[task_type],
                        )
                        if result.get("ok"):
                            st.session_state.h2o_result = result
                            st.session_state.shap_data  = result.get("shap")
                            add_log(f"H2O done. Best: {result['model_id']} | "
                                    f"MLflow: {result.get('run_id','')[:8]}", "s")
                            add_step("ml", f"AutoML: {result['model_type']}", "🤖")
                        else:
                            st.error(f"❌ H2O error: {result.get('error')}")
                            add_log(f"H2O failed: {result.get('error')}", "e")
                    except Exception as ex:
                        st.error(f"❌ Unexpected error: {ex}")
                        add_log(f"H2O error: {ex}", "e")
                st.rerun()

        # ── Results ───────────────────────────────────────────────────────────
        hr = st.session_state.h2o_result
        if hr:
            st.markdown('<div class="sh">AUTOML RESULTS</div>', unsafe_allow_html=True)
            shap_src = hr.get("shap", {}).get("source", "")
            _shap_labels = {
                "shapley_contributions": "True Shapley ✓",
                "shap_surrogate":        "SHAP Surrogate ✓",
                "varimp_proxy":          "VarImp Proxy ✓",
            }
            shap_tag   = _shap_labels.get(shap_src, "")
            shap_badge = (f'<span class="pill ok" style="font-size:9px;">{shap_tag}</span>'
                          if shap_tag else "")

            if hr["task"] == "classification":
                auc_val  = f'{hr["auc"]:.4f}'   if hr.get("auc")      else "N/A"
                acc_val  = f'{hr["accuracy"]:.1%}' if hr.get("accuracy") else "N/A"
                loss_val = f'{hr["logloss"]:.4f}'  if hr.get("logloss")  else "N/A"
                st.markdown(f"""<div class="metric-row">
                  <div class="metric-card"><div class="mv">{auc_val}</div><div class="ml">ROC-AUC</div></div>
                  <div class="metric-card"><div class="mv" style="color:#00ff88">{acc_val}</div><div class="ml">Accuracy</div></div>
                  <div class="metric-card"><div class="mv" style="color:#ff6b35">{loss_val}</div><div class="ml">Logloss</div></div>
                </div>""", unsafe_allow_html=True)
            else:
                r2_val   = f'{hr["r2"]:.4f}'   if hr.get("r2")   else "N/A"
                rmse_val = f'{hr["rmse"]:.4f}'  if hr.get("rmse") else "N/A"
                mae_val  = f'{hr["mae"]:.4f}'   if hr.get("mae")  else "N/A"
                st.markdown(f"""<div class="metric-row">
                  <div class="metric-card"><div class="mv">{r2_val}</div><div class="ml">R²</div></div>
                  <div class="metric-card"><div class="mv" style="color:#ff6b35">{rmse_val}</div><div class="ml">RMSE</div></div>
                  <div class="metric-card"><div class="mv" style="color:#00ff88">{mae_val}</div><div class="ml">MAE</div></div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
                f'<span style="font-size:11px;color:#c084fc;font-family:\'Space Mono\',monospace;">'
                f'Best: {hr.get("model_id","")}</span>{shap_badge}</div>',
                unsafe_allow_html=True,
            )
            if hr.get("run_id"):
                st.markdown(
                    f'<div style="font-size:10px;color:#64748b;font-family:\'Space Mono\',monospace;">'
                    f'MLflow: {hr["run_id"][:16]}…</div>',
                    unsafe_allow_html=True,
                )

            if "leaderboard" in hr:
                st.markdown('<div class="sh">LEADERBOARD (TOP 10)</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(hr["leaderboard"]), use_container_width=True, height=220)

    with cr:
        st.markdown('<div class="sh">🤖 ML AGENT CHAT</div>', unsafe_allow_html=True)
        hr = st.session_state.h2o_result
        sd = st.session_state.shap_data
        ml_ctx = df_ctx(df)
        if hr:
            ml_ctx += f"\nBest H2O model: {hr.get('model_id')} | algo: {hr.get('model_type')} | task: {hr['task']}"
            for k in ("auc", "accuracy", "logloss", "r2", "rmse", "mae"):
                if hr.get(k) is not None:
                    ml_ctx += f" | {k}={hr[k]:.4f}"
        if sd and sd.get("ok"):
            ml_ctx += f"\nTop SHAP features: {sd['mean_abs_shap'].head(5).to_dict()}"

        render_chat("ml", "🤖 ML Engineer")
        ui = st.text_area(
            "ML message", "",
            placeholder="Ask: Best model for churn? How to improve AUC? Interpret SHAP values.",
            height=90, key="mlin", label_visibility="collapsed",
        )
        if st.button("Send to ML Agent 🤖", use_container_width=True, key="mlsend"):
            if ui.strip():
                with st.spinner("ML agent thinking..."):
                    chat_send("ml", ui, AGENTS[2]["system"], ml_ctx)
                add_log("ML agent responded", "a")
                st.rerun()

        # SHAP preview panel
        sd = st.session_state.shap_data
        if sd and sd.get("ok"):
            _src_icons = {
                "shapley_contributions": "🔵 True Shapley (H2O native)",
                "shap_surrogate":        "🟢 SHAP Surrogate (shap lib)",
                "varimp_proxy":          "🟡 Variable Importance Proxy",
            }
            src_label = _src_icons.get(sd.get("source", ""), "📊 Feature Importance")
            st.markdown(
                f'<div style="font-size:10px;color:#64748b;font-family:\'Space Mono\','
                f'monospace;margin:6px 0;">{src_label}</div>',
                unsafe_allow_html=True,
            )
            render_shap_bars(sd["mean_abs_shap"], top_n=12)
        elif hr and "feature_importance" in hr:
            fi = pd.Series(hr["feature_importance"]).sort_values(ascending=False)
            render_shap_bars(fi, top_n=12, title="H2O Variable Importance")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 · XAI Advisor
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    a  = AGENTS[3]
    st.markdown(
        f'<div class="card explainer" style="margin-bottom:18px;">'
        f'<div class="agent-row"><span class="agent-icon">{a["icon"]}</span>'
        f'<span class="agent-name">{a["name"]} — SHAP Explainability + Business Advisor</span></div>'
        f'<div class="agent-desc">{a["desc"]}</div></div>',
        unsafe_allow_html=True,
    )

    sd = st.session_state.shap_data
    hr = st.session_state.h2o_result

    if not hr:
        st.info("🔍 Run **AutoML** in the AutoML tab first to unlock XAI insights.")
    else:
        cl, cr = st.columns([1, 1])

        def build_xai_ctx():
            ctx = df_ctx(df) + "\n"
            ctx += (f"\nH2O Best Model: {hr.get('model_id')} | "
                    f"Algo: {hr.get('model_type')} | Task: {hr['task']} | Target: {hr['target']}")
            for k, label in [("auc","AUC"), ("accuracy","Accuracy"),
                              ("logloss","Logloss"), ("r2","R²"),
                              ("rmse","RMSE"), ("mae","MAE")]:
                if hr.get(k) is not None:
                    ctx += f"\n{label}={hr[k]:.4f}"
            if sd and sd.get("ok"):
                top10   = sd["mean_abs_shap"].head(10)
                ctx += f"\n\nTop SHAP feature importances (mean |value|):\n{top10.to_string()}"
                sv0     = sd["shap_values"][0]
                contrib = (pd.Series(sv0, index=sd["feature_names"])
                             .sort_values(key=abs, ascending=False)
                             .head(8))
                ctx += f"\n\nFirst-row SHAP contributions:\n{contrib.to_string()}"
            elif "feature_importance" in hr:
                ctx += f"\n\nVariable importance: {dict(list(hr['feature_importance'].items())[:8])}"
            return ctx

        with cl:
            st.markdown('<div class="sh">SHAP GLOBAL IMPORTANCE</div>', unsafe_allow_html=True)
            if sd and sd.get("ok"):
                _src_full = {
                    "shapley_contributions": "True Shapley contributions (H2O native)",
                    "shap_surrogate":        "SHAP surrogate via shap library",
                    "varimp_proxy":          "Variable importance proxy",
                }
                src_lbl = _src_full.get(sd.get("source", ""), "Feature importance")
                st.caption(f"Source: {src_lbl} · {len(sd['shap_values'])} samples")
                render_shap_bars(sd["mean_abs_shap"], top_n=15)

                st.markdown('<div class="sh">SINGLE PREDICTION BREAKDOWN</div>', unsafe_allow_html=True)
                sv0  = sd["shap_values"][0]
                feat = sd["feature_names"]
                ev   = sd["expected_value"]
                contrib = (pd.Series(sv0, index=feat)
                             .sort_values(key=abs, ascending=False)
                             .head(12))
                bh  = ""
                mx2 = max(abs(contrib).max(), 1e-9)
                for fn, fv in contrib.items():
                    pct2 = int(abs(fv) / mx2 * 100)
                    cls2 = "shap-bar-pos" if fv >= 0 else "shap-bar-neg"
                    sgn  = "↑" if fv >= 0 else "↓"
                    bh  += (f'<div class="shap-bar-wrap">'
                            f'<div class="shap-feature">{sgn} {fn}</div>'
                            f'<div style="display:flex;align-items:center;gap:8px;">'
                            f'<div class="shap-bar-bg" style="flex:1;">'
                            f'<div class="{cls2}" style="width:{pct2}%;"></div></div>'
                            f'<div class="shap-val">{fv:+.4f}</div></div></div>')
                st.markdown(
                    f'<div class="card"><div style="font-size:11px;color:#64748b;'
                    f'font-family:\'Space Mono\',monospace;margin-bottom:10px;">'
                    f'Base value: {ev:.4f} | ↑ pushes prediction up ↓ pushes down</div>'
                    f'{bh}</div>',
                    unsafe_allow_html=True,
                )

                # Beeswarm-style scatter (only when we have real Shapley values)
                if sd.get("source") == "shapley_contributions":
                    top_feats = sd["mean_abs_shap"].head(10).index.tolist()
                    fi_idx    = [sd["feature_names"].index(f) for f in top_feats
                                 if f in sd["feature_names"]]
                    sv_top    = sd["shap_values"][:, fi_idx]
                    fig_sw    = go.Figure()
                    for i, fn in enumerate(reversed(top_feats)):
                        col_idx = top_feats.index(fn)
                        if col_idx >= sv_top.shape[1]:
                            continue
                        vals   = sv_top[:, col_idx]
                        jitter = np.random.uniform(-0.3, 0.3, len(vals))
                        fig_sw.add_trace(go.Scatter(
                            x=vals, y=[i + j for j in jitter],
                            mode="markers",
                            marker=dict(
                                size=4, opacity=0.6, color=vals,
                                colorscale=[[0, "#f72585"], [0.5, "#333"], [1, "#00d4ff"]],
                                cmin=-abs(vals).max(), cmax=abs(vals).max(),
                            ),
                            showlegend=False, name=fn,
                        ))
                    fig_sw.update_layout(
                        template="plotly_dark", paper_bgcolor="#13161f",
                        plot_bgcolor="#13161f", height=320,
                        margin=dict(l=130, r=20, t=10, b=30),
                        xaxis_title="SHAP value",
                        yaxis=dict(
                            tickvals=list(range(len(top_feats))),
                            ticktext=list(reversed(top_feats)),
                            tickfont=dict(size=9),
                        ),
                    )
                    st.plotly_chart(fig_sw, use_container_width=True)

            elif "feature_importance" in hr:
                fi2 = pd.Series(hr["feature_importance"]).sort_values(ascending=False)
                render_shap_bars(fi2, top_n=15, title="H2O Variable Importance")
            else:
                st.info("No SHAP data yet — train a model first.")

        with cr:
            st.markdown('<div class="sh">🤖 AI BUSINESS ADVISOR</div>', unsafe_allow_html=True)
            scenarios = {
                "🔄 Churn Analysis":  "The model predicts customer churn. Based on the SHAP feature importances and model results, what are the key churn drivers and what concrete business actions should we take to reduce churn rate? Include expected impact estimates.",
                "💰 Revenue Growth":  "The model predicts revenue/sales. Based on the SHAP values, which factors drive revenue most and what strategies should the business prioritise to increase revenue?",
                "⚠️ Risk / Fraud":   "The model detects risk or fraud. Explain the key risk drivers from SHAP analysis and recommend prioritised mitigation strategies.",
                "😊 Satisfaction":    "The model predicts customer satisfaction or NPS. What are the biggest factors impacting satisfaction and how can we improve each one?",
                "📦 Demand":          "The model forecasts demand. Based on feature importances, what drives demand patterns and how should we optimise supply chain and inventory?",
            }
            sc_cols = st.columns(3)
            for i, (label, prompt) in enumerate(scenarios.items()):
                with sc_cols[i % 3]:
                    if st.button(label, use_container_width=True, key=f"sc{i}"):
                        with st.spinner("XAI Advisor generating insights..."):
                            resp = chat_send("explainer", prompt, a["system"], build_xai_ctx())
                        st.session_state.xai_report = resp
                        add_log("XAI Advisor: business report generated", "a")
                        add_step("xai", "XAI: Business Report", "🔍")
                        st.rerun()

            render_chat("explainer", "🔍 XAI Advisor")
            ui = st.text_area(
                "XAI message", "",
                placeholder="Ask about predictions, root causes, or business impact...",
                height=90, key="xain", label_visibility="collapsed",
            )
            if st.button("Ask XAI Advisor 🔍", use_container_width=True, key="xaisend"):
                if ui.strip():
                    with st.spinner("XAI Advisor thinking..."):
                        resp = chat_send("explainer", ui, a["system"], build_xai_ctx())
                    st.session_state.xai_report = resp
                    add_log("XAI Advisor responded", "a")
                    add_step("xai", "XAI: Insight", "🔍")
                    st.rerun()

            if st.session_state.xai_report:
                report = st.session_state.xai_report
                st.markdown('<div class="sh">BUSINESS REPORT</div>', unsafe_allow_html=True)
                lines    = report.split("\n")
                rec_lines = [l for l in lines if re.match(r"^\s*\d+[\.\)]\s+", l)]
                if rec_lines:
                    for i, rec in enumerate(rec_lines[:8], 1):
                        clean_r = re.sub(r"^\s*\d+[\.\)]\s*", "", rec).strip()
                        st.markdown(
                            f'<div style="background:#181c28;border:1px solid #1a2035;'
                            f'border-left:3px solid #00ff88;border-radius:8px;'
                            f'padding:12px 16px;margin-bottom:8px;">'
                            f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                            f'color:#00ff88;margin-bottom:4px;">REC {i:02d}</div>'
                            f'<div style="font-size:13px;line-height:1.6;">{clean_r}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    for chunk in re.split(r"(##\s+.+)", report):
                        if chunk.strip():
                            if chunk.startswith("##"):
                                head = chunk.replace("##", "").strip()
                                st.markdown(f'<div class="sh">{head}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(chunk.strip())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · MLflow
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    from ml_engine import get_mlflow_runs

    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:16px;font-weight:700;'
        'color:#00d4ff;margin-bottom:16px;">📋 MLflow Experiment Tracker</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh"):
        st.rerun()

    runs = get_mlflow_runs(30)
    if not runs:
        st.info("No MLflow runs yet. Run AutoML to populate experiment history.")
    else:
        add_step("mlflow", "MLflow Tracking", "📋")
        rows = []
        for r2 in runs:
            rows.append({
                "Run":      r2.get("tags.mlflow.runName", r2.get("run_id", "")[:8]),
                "Engine":   r2.get("params.engine", "AutoML"),
                "Model":    r2.get("params.best_algo", r2.get("params.best_model", "–"))[:30],
                "Target":   r2.get("params.target",    "–"),
                "Task":     r2.get("params.task",      "–"),
                "Accuracy": r2.get("metrics.accuracy", ""),
                "AUC":      r2.get("metrics.auc",      ""),
                "R²":       r2.get("metrics.r2",       ""),
                "RMSE":     r2.get("metrics.rmse",     ""),
            })
        sdf = pd.DataFrame(rows)
        for mc in ["Accuracy", "AUC", "R²", "RMSE"]:
            sdf[mc] = pd.to_numeric(sdf[mc], errors="coerce").round(4)
        st.dataframe(sdf, use_container_width=True, height=280)

        st.markdown('<div class="sh">RUN CARDS</div>', unsafe_allow_html=True)
        for r2 in runs[:8]:
            rn  = r2.get("tags.mlflow.runName", r2.get("run_id", "")[:12])
            md  = r2.get("params.best_algo",    r2.get("params.best_model", "–"))
            tk  = r2.get("params.task",          "–")
            auc = f'AUC={float(r2["metrics.auc"]):.3f}' if r2.get("metrics.auc") else ""
            r2v = f'R²={float(r2["metrics.r2"]):.4f}'   if r2.get("metrics.r2")  else ""
            ms  = auc or r2v or "–"
            st.markdown(
                f'<div style="background:#181c28;border:1px solid #1a2035;border-radius:8px;'
                f'padding:12px 14px;margin-bottom:7px;display:flex;align-items:center;gap:14px;">'
                f'<div style="font-size:20px;">📊</div>'
                f'<div style="flex:1;">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:12px;font-weight:700;">{rn}</div>'
                f'<div style="font-size:11px;color:#64748b;">{md} · {tk} · {ms}</div></div>'
                f'<span style="background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);'
                f'color:#00d4ff;font-family:\'Space Mono\',monospace;font-size:10px;'
                f'padding:2px 8px;border-radius:4px;">{r2.get("run_id","")[:8]}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sh">METRIC COMPARISON CHART</div>', unsafe_allow_html=True)
        mc_sel = st.selectbox("Metric", ["Accuracy", "AUC", "R²", "RMSE"],
                              label_visibility="collapsed")
        chart_data = sdf[["Run", mc_sel]].dropna()
        if not chart_data.empty:
            fig = px.bar(chart_data, x="Run", y=mc_sel, template="plotly_dark",
                         color=mc_sel, color_continuous_scale=["#1a2035", "#00d4ff"])
            fig.update_layout(paper_bgcolor="#13161f", plot_bgcolor="#13161f",
                              height=240, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇ Export Runs CSV", data=sdf.to_csv(index=False),
                           file_name="mlflow_runs.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 · Log
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="sh">⚡ PIPELINE STEPS</div>', unsafe_allow_html=True)
        if not st.session_state.pipeline_steps:
            st.info("No steps yet.")
        for i, step in enumerate(st.session_state.pipeline_steps):
            st.markdown(
                f'<div style="display:flex;align-items:start;gap:12px;padding:11px 13px;'
                f'background:#13161f;border:1px solid #1a2035;border-left:3px solid #00ff88;'
                f'border-radius:8px;margin-bottom:6px;">'
                f'<div style="font-size:20px;line-height:1;">{step["icon"]}</div>'
                f'<div style="flex:1;">'
                f'<div style="font-family:\'Space Mono\',monospace;font-size:12px;font-weight:700;">'
                f'Step {i+1}: {step["name"]}</div>'
                f'<div style="font-size:10px;color:#64748b;">{step["ts"]}</div></div>'
                f'<span style="color:#00ff88;">✓</span></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.pipeline_steps:
            st.download_button(
                "⬇ Export Pipeline JSON",
                data=json.dumps(st.session_state.pipeline_steps, default=str, indent=2),
                file_name="pipeline.json", mime="application/json",
                use_container_width=True,
            )
            if st.session_state.df_clean is not None:
                st.download_button(
                    "⬇ Export Cleaned CSV",
                    data=st.session_state.df_clean.to_csv(index=False),
                    file_name="cleaned_data.csv", mime="text/csv",
                    use_container_width=True,
                )
    with c2:
        st.markdown('<div class="sh">🖥 ACTIVITY LOG</div>', unsafe_allow_html=True)
        kind_css = {"i": "li", "s": "ls", "w": "lw", "a": "la", "e": "le"}
        log_html = "".join(
            f'<div class="{kind_css.get(l["kind"], "li")}">[{l["ts"]}] {l["msg"]}</div>'
            for l in reversed(st.session_state.logs[-60:])
        ) or '<div class="li">No activity yet…</div>'
        st.markdown(f'<div class="logbox">{log_html}</div>', unsafe_allow_html=True)
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.logs = []
            st.rerun()
