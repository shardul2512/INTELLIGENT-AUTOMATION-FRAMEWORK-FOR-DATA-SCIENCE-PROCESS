CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg-primary: #08090f;
    --bg-secondary: #0f111a;
    --bg-card: #13161f;
    --bg-card2: #181c28;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-purple: #9b5de5;
    --accent-orange: #ff6b35;
    --accent-yellow: #ffd60a;
    --accent-pink: #f72585;
    --text-primary: #dde4f0;
    --text-secondary: #64748b;
    --border: #1a2035;
    --border-bright: #1e3a5f;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
}
.stApp { background: var(--bg-primary); }

/* ── Hide Streamlit Default Header & Adjust Top Padding ── */
header[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Header ── */
.wb-header {
    display:flex; align-items:center; gap:14px;
    padding-bottom:20px; border-bottom:1px solid var(--border); margin-bottom:20px;
}
.wb-title {
    font-family:'Space Mono',monospace; font-size:20px; font-weight:700;
    background:linear-gradient(135deg,var(--accent-cyan),var(--accent-green));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.wb-badge {
    background:rgba(0,212,255,.08); border:1px solid rgba(0,212,255,.3);
    color:var(--accent-cyan) !important; font-family:'Space Mono',monospace;
    font-size:9px; padding:3px 9px; border-radius:4px; letter-spacing:1.5px;
}

/* ── Pipeline Bar ── */
.pipeline-wrap {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:12px; padding:18px 22px; margin:0 0 20px 0;
}
.pipeline-label {
    font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px;
    color:var(--text-secondary); text-transform:uppercase; margin-bottom:14px;
}
.pipe-track { display:flex; align-items:center; }
.pipe-step { display:flex; flex-direction:column; align-items:center; gap:6px; }
.pipe-node {
    width:50px; height:50px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:19px; border:2px solid; transition:all .3s;
}
.pipe-node.idle   { background:var(--bg-secondary); border-color:var(--border); opacity:.35; }
.pipe-node.done   { background:rgba(0,255,136,.1); border-color:var(--accent-green); box-shadow:0 0 14px rgba(0,255,136,.2); }
.pipe-node.run    { background:rgba(255,214,10,.1); border-color:var(--accent-yellow);
                    box-shadow:0 0 20px rgba(255,214,10,.3); animation:pulse-y 1.4s infinite; }
.pipe-node.active { background:rgba(0,212,255,.1); border-color:var(--accent-cyan);
                    box-shadow:0 0 18px rgba(0,212,255,.25); animation:pulse-c 2s infinite; }
@keyframes pulse-y{0%,100%{box-shadow:0 0 20px rgba(255,214,10,.3)}50%{box-shadow:0 0 36px rgba(255,214,10,.6)}}
@keyframes pulse-c{0%,100%{box-shadow:0 0 18px rgba(0,212,255,.25)}50%{box-shadow:0 0 32px rgba(0,212,255,.5)}}
.pipe-lbl { font-family:'Space Mono',monospace; font-size:9px; color:var(--text-secondary);
             text-align:center; max-width:68px; line-height:1.3; }
.pipe-lbl.done   { color:var(--accent-green); }
.pipe-lbl.active { color:var(--accent-cyan); }
.pipe-conn { flex:1; height:2px; min-width:30px; }
.pipe-conn.idle { background:var(--border); opacity:.4; }
.pipe-conn.done { background:linear-gradient(90deg,var(--accent-green),var(--accent-cyan)); }

/* ── Section Header ── */
.sh {
    font-family:'Space Mono',monospace; font-size:9px; letter-spacing:2px;
    text-transform:uppercase; color:var(--text-secondary);
    display:flex; align-items:center; gap:8px; margin:18px 0 10px;
}
.sh::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Cards ── */
.card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:10px; padding:16px; margin-bottom:12px; position:relative; overflow:hidden;
}
.card::before { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.card.cleaner::before { background:linear-gradient(90deg,var(--accent-cyan),#0099bb); }
.card.viz::before     { background:linear-gradient(90deg,var(--accent-purple),#c084fc); }
.card.ml::before      { background:linear-gradient(90deg,var(--accent-orange),#ffaa00); }
.card.explainer::before { background:linear-gradient(90deg,var(--accent-green),#00cc66); }

.agent-row { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.agent-icon { font-size:22px; }
.agent-name { font-family:'Space Mono',monospace; font-size:13px; font-weight:700; }
.agent-desc { font-size:12px; color:var(--text-secondary); line-height:1.5; }
.tag {
    display:inline-block; background:rgba(124,58,237,.12);
    border:1px solid rgba(124,58,237,.25); color:#c084fc;
    font-size:9px; padding:2px 7px; border-radius:3px;
    margin:3px 3px 0 0; font-family:'Space Mono',monospace; letter-spacing:.5px;
}

/* ── Metrics ── */
.metric-row { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0; }
.metric-card {
    background:var(--bg-card); border:1px solid var(--border);
    border-radius:8px; padding:12px 16px; flex:1; min-width:110px;
}
.mv { font-family:'Space Mono',monospace; font-size:20px; font-weight:700; color:var(--accent-cyan); }
.ml { font-size:11px; color:var(--text-secondary); margin-top:2px; }

/* ── Chat ── */
.chat-u {
    padding:11px 14px; border-radius:8px; margin-bottom:9px; font-size:13px; line-height:1.6;
    background:rgba(0,212,255,.07); border:1px solid rgba(0,212,255,.18);
}
.chat-a {
    padding:11px 14px; border-radius:8px; margin-bottom:9px; font-size:13px; line-height:1.6;
    background:rgba(124,58,237,.07); border:1px solid rgba(124,58,237,.18);
}
.chat-lbl { font-size:10px; opacity:.6; font-family:'Space Mono',monospace; margin-bottom:4px; }

/* ── Log ── */
.logbox {
    background:#050710; border:1px solid var(--border); border-radius:8px;
    padding:14px; font-family:'Space Mono',monospace; font-size:11px;
    max-height:280px; overflow-y:auto; line-height:1.9;
}
.li   { color:var(--accent-cyan); }
.ls   { color:var(--accent-green); }
.lw   { color:var(--accent-yellow); }
.la   { color:#c084fc; }
.le   { color:var(--accent-pink); }

/* ── Status pills ── */
.pill {
    display:inline-flex; align-items:center; gap:4px; padding:3px 9px;
    border-radius:20px; font-family:'Space Mono',monospace; font-size:9px; letter-spacing:.5px;
}
.pill.ok   { background:rgba(0,255,136,.08); color:var(--accent-green); border:1px solid rgba(0,255,136,.25); }
.pill.idle { background:rgba(100,116,139,.08); color:var(--text-secondary); border:1px solid var(--border); }
.pill.busy { background:rgba(255,214,10,.08); color:var(--accent-yellow); border:1px solid rgba(255,214,10,.25); }

/* ── SHAP ── */
.shap-bar-wrap { margin:6px 0; }
.shap-feature { font-family:'Space Mono',monospace; font-size:10px; color:var(--text-secondary); margin-bottom:2px; }
.shap-bar-bg { background:var(--border); border-radius:3px; height:10px; overflow:hidden; }
.shap-bar-pos { background:linear-gradient(90deg,#00d4ff,#00ff88); height:10px; border-radius:3px; transition:width .5s; }
.shap-bar-neg { background:linear-gradient(90deg,#f72585,#ff6b35); height:10px; border-radius:3px; transition:width .5s; float:right; }
.shap-val { font-family:'Space Mono',monospace; font-size:10px; text-align:right; color:var(--text-secondary); }

/* ── MLflow run cards ── */
.run-card {
    background:var(--bg-card2); border:1px solid var(--border);
    border-radius:8px; padding:12px 14px; margin-bottom:8px;
    display:flex; align-items:center; gap:14px;
}
.run-name { font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }
.run-meta { font-size:11px; color:var(--text-secondary); margin-top:2px; }
.run-badge {
    background:rgba(0,212,255,.1); border:1px solid rgba(0,212,255,.2);
    color:var(--accent-cyan); font-family:'Space Mono',monospace;
    font-size:10px; padding:2px 8px; border-radius:4px; white-space:nowrap;
}

/* ── Biz rec cards ── */
.biz-card {
    background:var(--bg-card2); border:1px solid var(--border);
    border-left:3px solid var(--accent-green); border-radius:8px;
    padding:14px 16px; margin-bottom:10px;
}
.biz-num { font-family:'Space Mono',monospace; font-size:11px; color:var(--accent-green); margin-bottom:4px; }
.biz-text { font-size:13px; line-height:1.6; }

/* ── Streamlit overrides ── */
.stButton button {
    background:transparent !important; border:1px solid var(--accent-cyan) !important;
    color:var(--accent-cyan) !important; font-family:'Space Mono',monospace !important;
    font-size:11px !important; border-radius:6px !important;
    letter-spacing:.5px !important; transition:all .2s !important;
}
.stButton button:hover {
    background:rgba(0,212,255,.1) !important; box-shadow:0 0 14px rgba(0,212,255,.2) !important;
}
.stTextInput [data-baseweb="input"], 
.stTextArea [data-baseweb="textarea"], 
.stTextArea [data-baseweb="base-input"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stTextInput [data-baseweb="input"] input, 
.stTextArea textarea {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
}
.stTextInput [data-baseweb="input"]:focus-within, 
.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextArea [data-baseweb="base-input"]:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px var(--accent-cyan) !important;
}
.stSelectbox > div > div {
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    color:var(--text-primary) !important; border-radius:6px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'Space Mono',monospace !important; font-size:10px !important; letter-spacing:1px !important;
}
.stExpander { border:1px solid var(--border) !important; border-radius:8px !important; }
div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:8px; overflow:hidden; }
hr { border-color:var(--border) !important; }

/* ── File Uploader Styling ── */
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-cyan) !important;
    background-color: var(--bg-card2) !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-secondary) !important;
}
/* Style the uploaded file box itself */
[data-testid="stFileUploader"] section, 
[data-testid="stFileUploader"] ul, 
[data-testid="stFileUploader"] li, 
[data-testid="stUploadedFile"],
div[data-testid="stUploadedFile"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section *, 
[data-testid="stFileUploader"] ul *, 
[data-testid="stFileUploader"] li *,
[data-testid="stUploadedFile"] * {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}

</style>
"""
