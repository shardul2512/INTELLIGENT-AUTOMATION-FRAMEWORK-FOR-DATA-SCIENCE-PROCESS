# ⚗️ AN INTELLIGENT AUTOMATION FRAMEWORK FOR DATA SCEIENCE PROCESS 

A visual, AI-powered data science pipeline studio built with Streamlit and OpenRouter.

## Features

- **Visual Pipeline Studio** — every action becomes a traceable step in a live pipeline DAG
- **🧹 Data Wrangler Agent** — AI-guided data cleaning with quick-action buttons
- **📊 Viz Architect Agent** — AI-generated Plotly visualizations with in-app code execution
- **🤖 ML Engineer Agent** — One-click model training (classification + regression) with feature importance
- **Pipeline Export** — export the full pipeline as JSON + cleaned data as CSV
- **OpenRouter Backend** — plug in any model: Claude, GPT-4o, Gemini, Llama, Mistral, DeepSeek

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

1. **OpenRouter API Key** — get yours at https://openrouter.ai/keys  
   Add it to the `.env` file in the project root: `OPENROUTER_API_KEY=sk-or-v1-...`

2. **Model Selection** — choose from 8 top models:
   - Claude 3 Haiku (fast & cheap — recommended)
   - Claude 3.5 Sonnet
   - GPT-4o / GPT-4o Mini
   - Gemini Flash 1.5
   - Llama 3.1 70B
   - Mistral Large
   - DeepSeek Chat

3. **Upload your dataset** — CSV or Excel files supported

## Agent Architecture

Each agent is a stateful chat interface backed by an OpenRouter LLM call with a specialized system prompt:

| Agent | System Prompt Focus | Output |
|---|---|---|
| Data Wrangler | Cleaning strategies, pandas code | Actionable code + chat |
| Viz Architect | Plotly chart design, EDA | Executable Python + rendered chart |
| ML Engineer | Sklearn model selection + metrics | Trained model + metrics + feature importance |

All agents share the dataset context (shape, dtypes, sample rows, missing value counts) in every API call for grounded, accurate responses.

## Pipeline Reproducibility

Every action — loading data, cleaning, visualizing, training — is recorded as a timestamped pipeline step. Export the full pipeline as JSON to reproduce or share your workflow.
