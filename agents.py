"""
Agent definitions and OpenRouter client for DS Workbench.
Stack: AutoML (Sklearn) · MLflow · SHAP · OpenRouter LLMs
"""
import requests
import streamlit as st

AGENTS = [
    {
        "id": "cleaner",
        "name": "Data Wrangler",
        "icon": "🧹",
        "class": "cleaner",
        "desc": "Inspects, cleans and transforms your dataset. Handles missing values, outliers, type casting and feature engineering.",
        "tags": ["MISSING VALUES", "OUTLIERS", "TYPE CASTING", "ENCODING"],
        "color": "#00d4ff",
        "system": """You are an expert data cleaning and wrangling AI agent.
Your job:
1. Analyse data quality issues concisely
2. Suggest specific cleaning operations with pandas code snippets
3. Explain WHY each step matters
Format: ## Analysis, ## Recommended Steps, ## Code.
Always provide working Python/pandas code in code blocks.""",
    },
    {
        "id": "viz",
        "name": "Viz Architect",
        "icon": "📊",
        "class": "viz",
        "desc": "Designs and generates insightful visualizations for EDA.",
        "tags": ["PLOTLY", "MATPLOTLIB", "EDA", "STATISTICS"],
        "color": "#7c3aed",
        "system": """You are an expert data visualization AI agent.
Rules:
- Always use plotly.express or plotly.graph_objects
- Provide complete, executable Python code
- The dataframe is available as `df`
- Use `fig` as the figure variable
- Include all imports
- Explain what insight the chart reveals""",
    },
    {
        "id": "ml",
        "name": "AutoML Engineer",
        "icon": "🤖",
        "class": "ml",
        "desc": "Orchestrates AutoML (Sklearn) across GBM, XGBoost, RF and stacked ensembles. Logs every run to MLflow and explains predictions with SHAP Shapley values.",
        "tags": ["AutoML AUTOML", "MLFLOW", "SHAP", "GBM", "ENSEMBLE"],
        "color": "#ff6b35",
        "system": """You are an expert AutoML AI agent with deep knowledge of AutoML (Sklearn), MLflow experiment tracking, and SHAP explainability.
When answering:
1. Recommend the best AutoML configuration given the data characteristics
2. Explain the winning model's algorithm and hyperparameters in plain language
3. Interpret SHAP values and feature importances in plain English
4. Translate model insights into concrete business actions
5. Suggest follow-up experiments (e.g., feature engineering, different time budgets)
6. Provide working Python code when asked (AutoML, MLflow, shap)
Be concise, technical, and actionable.""",
    },
    {
        "id": "explainer",
        "name": "XAI Advisor",
        "icon": "🔍",
        "class": "explainer",
        "desc": "Translates SHAP Shapley explanations into plain-English business recommendations. Identifies root causes and prescribes next steps.",
        "tags": ["SHAP", "BUSINESS RECS", "ROOT CAUSE", "PRESCRIPTIVE"],
        "color": "#00ff88",
        "system": """You are an expert AI explainability and business strategy advisor.
Given SHAP feature importance data, AutoML (Sklearn) model results, and dataset context, you:
1. Translate SHAP values into plain business language (no jargon)
2. Identify the TOP drivers of the predicted outcome
3. Recommend 5-8 concrete, prioritized business actions to improve outcomes
4. Explain the reasoning behind each recommendation
5. Quantify impact where possible (e.g., "reducing X by 10% could decrease churn by ~Y%")
Be specific, actionable, and business-focused. Always structure with:
## 🔍 Key Drivers
## 💡 Business Recommendations  
## ⚠️ Risk Factors
## 📊 Next Steps""",
    },
]


def call_openrouter(messages: list, system: str = "", model: str = None) -> str:
    key = st.session_state.get("openrouter_key", "")
    if not key:
        return "⚠️ No OpenRouter API key set. Please add OPENROUTER_API_KEY to your .env file."
    m = model or st.session_state.get("model", "anthropic/claude-3-haiku")
    payload = {
        "model": m,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://ds-workbench.app",
                "X-Title": "DS Workbench",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        return f"❌ API Error {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
