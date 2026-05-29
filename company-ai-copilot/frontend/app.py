"""
app.py  —  Enterprise AI Knowledge Copilot  —  Streamlit Frontend
=================================================================
Run:  streamlit run app.py
"""

import csv
import io
import json
import time
from datetime import datetime
import requests
import streamlit as st

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND = "http://127.0.0.1:8000"

SUGGESTED_PROMPTS = [
    "What products or services does this company offer?",
    "Who are the founders or leadership team?",
    "What is the company's mission or vision?",
    "What industries or markets does this company serve?",
    "What makes this company different from competitors?",
]

TECH_PILLS = [
    ("⚡","FastAPI"), ("🧠","Llama 3"), ("🗄️","ChromaDB"),
    ("🔍","RAG"), ("📄","pypdf"), ("🔄","Streaming"), ("💾","SQLite"),
]

# =============================================================================
# CSS — production-grade design system
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}

/* ── Base ── */
.stApp{background-color:#060c18;color:#e2e8f0}
.block-container{padding-top:0!important}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a1020 0%,#0d1528 100%);
  border-right:1px solid rgba(255,255,255,0.06)}
section[data-testid="stSidebar"]>div{padding:1.4rem 1.1rem}

.sb-brand{display:flex;align-items:center;gap:9px;margin-bottom:3px}
.sb-icon{font-size:20px;line-height:1}
.sb-name{font-size:14px;font-weight:700;color:#f1f5f9;letter-spacing:-.02em}
.sb-sub{font-size:10px;color:#334155;margin-bottom:16px;padding-left:29px;letter-spacing:.02em}
.sb-label{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
          color:#334155;margin-bottom:5px;margin-top:14px}
.sb-div{border:none;border-top:1px solid rgba(255,255,255,0.05);margin:10px 0}

/* ── Status badges — smaller and tighter ── */
.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;
       padding:3px 9px;border-radius:20px;font-weight:500;margin-bottom:3px}
.b-on {background:rgba(34,197,94,.1);color:#4ade80;border:1px solid rgba(34,197,94,.18)}
.b-off{background:rgba(239,68,68,.1);color:#f87171;border:1px solid rgba(239,68,68,.18)}
.b-warn{background:rgba(234,179,8,.1);color:#fbbf24;border:1px solid rgba(234,179,8,.18)}
.dot{width:5px;height:5px;border-radius:50%;display:inline-block}
.d-on{background:#4ade80}.d-off{background:#f87171}.d-warn{background:#fbbf24}

/* ── Inputs ── */
.stTextInput>div>div>input{
  background-color:#0d1528!important;border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:7px!important;color:#e2e8f0!important;font-size:13px!important;
  padding:7px 11px!important;transition:border-color .15s!important}
.stTextInput>div>div>input:focus{
  border-color:#3b82f6!important;box-shadow:0 0 0 3px rgba(59,130,246,.12)!important}
.stTextInput label,.stTextArea label,.stSelectbox label{
  color:#475569!important;font-size:11px!important;font-weight:600!important;
  letter-spacing:.04em!important;text-transform:uppercase!important}
.stSelectbox>div>div{
  background-color:#0d1528!important;border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:7px!important;color:#e2e8f0!important}
textarea{background-color:#0d1528!important;border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:7px!important;color:#e2e8f0!important;font-size:13px!important}

/* ── Sidebar ingest button — distinct from other buttons ── */
.stButton>button{
  background:#131f38!important;color:#93c5fd!important;
  border:1px solid rgba(59,130,246,.25)!important;border-radius:7px!important;
  font-size:12px!important;font-weight:600!important;
  padding:7px 14px!important;width:100%!important;
  transition:background .15s,border-color .15s!important;letter-spacing:.01em!important}
.stButton>button:hover{
  background:#1e3a5f!important;border-color:rgba(59,130,246,.5)!important}

/* ── File uploader ── */
[data-testid="stFileUploader"]{
  background:#0d1528!important;border:1px dashed rgba(255,255,255,0.08)!important;
  border-radius:7px!important;padding:6px!important}
[data-testid="stFileUploader"] label{color:#475569!important;font-size:11px!important}

/* ── Tabs — professional, not pill buttons ── */
.stTabs [data-baseweb="tab-list"]{
  background:transparent!important;
  border-bottom:1px solid rgba(255,255,255,0.06)!important;
  gap:0!important;padding:0!important}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;color:#475569!important;
  border-radius:0!important;font-size:13px!important;font-weight:500!important;
  padding:10px 18px!important;border-bottom:2px solid transparent!important;
  margin-bottom:-1px!important}
.stTabs [aria-selected="true"]{
  background:transparent!important;color:#f1f5f9!important;
  border-bottom:2px solid #3b82f6!important}
.stTabs [data-baseweb="tab"]:hover{color:#94a3b8!important}
.stTabs [data-baseweb="tab-panel"]{background:transparent!important;padding-top:24px!important}

/* ── Dashboard stat cards — asymmetric, professional ── */
.stat-card{
  background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:10px;padding:18px 20px;margin-bottom:8px;
  position:relative;overflow:hidden}
.stat-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6)}
.stat-accent{position:absolute;top:14px;right:16px;
  font-size:24px;opacity:.12}
.stat-num{font-size:32px;font-weight:800;color:#f1f5f9;
  line-height:1;letter-spacing:-.03em;margin-bottom:4px}
.stat-lbl{font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:#475569}
.stat-sub{font-size:11px;color:#334155;margin-top:3px}

/* ── Hero — tighter, more editorial ── */
.hero-wrap{padding:20px 0 16px 0}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:5px;
  font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:#3b82f6;margin-bottom:10px}
.hero-title{
  font-size:30px;font-weight:800;color:#f8fafc;
  line-height:1.15;letter-spacing:-.03em;margin-bottom:6px}
.hero-title .hl{
  background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text}
.hero-sub{font-size:13px;color:#475569;line-height:1.6;
  max-width:480px;margin-bottom:14px}
.pills{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:0}
.pill{
  display:inline-flex;align-items:center;gap:4px;
  background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
  border-radius:4px;padding:3px 8px;font-size:10px;color:#64748b;font-weight:500}

/* ── Prompt chips — small and subtle, not big buttons ── */
.prompt-chip-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}
.prompt-chip{
  display:inline-block;background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.08);border-radius:20px;
  padding:6px 14px;font-size:12px;color:#64748b;cursor:pointer;
  transition:border-color .15s,color .15s}
.prompt-chip:hover{border-color:#3b82f6;color:#93c5fd}

/* ── Section divider ── */
.c-div{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#1e2a45;margin:12px 0 18px;display:flex;align-items:center;gap:10px}
.c-line{flex:1;height:1px;background:rgba(255,255,255,0.05)}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"]{
  background:transparent!important;padding:4px 0!important;border:none!important}
.u-bub{
  background:#152340;border:1px solid #1e3a5f;
  border-radius:14px 14px 4px 14px;
  padding:10px 15px;font-size:14px;color:#e2e8f0;
  max-width:74%;margin-left:auto;line-height:1.55}
.a-bub{
  background:#0d1528;border:1px solid rgba(255,255,255,0.07);
  border-radius:4px 14px 14px 14px;
  padding:13px 16px;font-size:14px;color:#cbd5e1;
  max-width:86%;line-height:1.65}

/* ── Confidence badges ── */
.conf-hi{display:inline-block;font-size:9px;font-weight:700;padding:2px 7px;
  border-radius:10px;margin-top:5px;letter-spacing:.04em;
  background:rgba(34,197,94,.1);color:#4ade80;border:1px solid rgba(34,197,94,.18)}
.conf-md{display:inline-block;font-size:9px;font-weight:700;padding:2px 7px;
  border-radius:10px;margin-top:5px;letter-spacing:.04em;
  background:rgba(234,179,8,.1);color:#fbbf24;border:1px solid rgba(234,179,8,.18)}
.conf-lo{display:inline-block;font-size:9px;font-weight:700;padding:2px 7px;
  border-radius:10px;margin-top:5px;letter-spacing:.04em;
  background:rgba(239,68,68,.1);color:#f87171;border:1px solid rgba(239,68,68,.18)}

/* ── Star rating ── */
.star{font-size:16px;cursor:pointer;transition:transform .1s}
.star:hover{transform:scale(1.2)}

/* ── Source / expander ── */
[data-testid="stExpander"]{
  background:#0d1528!important;border:1px solid rgba(255,255,255,0.06)!important;
  border-radius:8px!important;margin-top:6px!important}
[data-testid="stExpander"] summary{
  color:#334155!important;font-size:11px!important;
  font-weight:600!important;padding:8px 12px!important}
.src-chunk{
  background:#060c18;border:1px solid rgba(255,255,255,0.05);
  border-radius:6px;padding:10px 12px;font-size:12px;
  color:#475569;line-height:1.5;margin-bottom:6px}

/* ── History items ── */
.hist-item{background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;padding:11px 15px;margin-bottom:7px}
.h-time{font-size:9px;color:#1e2a45;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:5px}
.h-q{font-size:13px;color:#93c5fd;font-weight:500;margin-bottom:3px}
.h-a{font-size:12px;color:#475569;line-height:1.5}

/* ── Analytics cards ── */
.a-card{background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:10px;padding:18px;margin-bottom:14px}
.a-title{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#334155;margin-bottom:14px}

/* ── KB rows ── */
.kb-row{display:flex;align-items:center;padding:12px 0;
  border-bottom:1px solid rgba(255,255,255,0.04)}
.kb-name{font-size:14px;font-weight:600;color:#f1f5f9}
.kb-sub{font-size:10px;color:#334155;margin-top:2px}

/* ── Architecture / about ── */
.arch-box{background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:10px;padding:18px;font-family:monospace;
  font-size:12px;color:#475569;line-height:1.9}
.tech-row{display:flex;justify-content:space-between;align-items:center;
  padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px}
.t-name{color:#f1f5f9;font-weight:500}
.t-role{color:#334155;font-size:12px}
.t-badge{font-size:9px;padding:2px 7px;border-radius:10px;
  background:rgba(59,130,246,.08);color:#64748b;
  border:1px solid rgba(59,130,246,.15);font-weight:600;letter-spacing:.04em}

/* ── Batch ingest results ── */
.batch-row-ok{background:rgba(34,197,94,.04);border:1px solid rgba(34,197,94,.12);
  border-radius:6px;padding:7px 11px;margin-bottom:4px;font-size:12px;color:#4ade80}
.batch-row-err{background:rgba(239,68,68,.04);border:1px solid rgba(239,68,68,.12);
  border-radius:6px;padding:7px 11px;margin-bottom:4px;font-size:12px;color:#f87171}

/* ── Search results ── */
.sim-card,.search-result{
  background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;padding:10px 14px;margin-bottom:6px}
.sim-card:hover,.search-result:hover{border-color:rgba(59,130,246,.3)}
.sim-q,.sr-q{font-size:12px;color:#93c5fd;font-weight:500}
.sim-a,.sr-a{font-size:11px;color:#475569;margin-top:3px}
.sr-meta{font-size:9px;color:#1e2a45;margin-top:4px;letter-spacing:.04em}

/* ── Test suite ── */
.test-row{display:flex;align-items:flex-start;gap:10px;padding:10px 14px;
  background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:8px;margin-bottom:5px}
.test-pass{color:#4ade80;font-size:12px;font-weight:700;min-width:16px;margin-top:1px}
.test-fail{color:#f87171;font-size:12px;font-weight:700;min-width:16px;margin-top:1px}
.test-skip{color:#fbbf24;font-size:12px;font-weight:700;min-width:16px;margin-top:1px}
.test-name{font-size:13px;color:#e2e8f0;font-weight:500}
.test-detail{font-size:11px;color:#475569;margin-top:2px}
.test-summary{
  background:#0d1528;border:1px solid rgba(255,255,255,0.06);
  border-radius:10px;padding:14px 18px;margin-bottom:14px;
  display:flex;gap:24px;align-items:center}

/* ── Chat input ── */
[data-testid="stChatInput"]{
  background:#0d1528!important;border:1px solid rgba(255,255,255,0.08)!important;
  border-radius:10px!important}
[data-testid="stChatInput"] textarea{color:#e2e8f0!important;font-size:14px!important}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#1e2a45;border-radius:10px}

/* ── Hide chrome ── */
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}

/* ── Search box ── *//* ── Search box ── */
.search-wrap{display:flex;gap:8px;align-items:center;
  background:#0d1224;border:1px solid #1e2a45;border-radius:10px;
  padding:8px 14px;margin-bottom:18px}
.search-icon{font-size:14px;color:#475569}
.search-result{background:#0d1224;border:1px solid #1e2a45;border-radius:8px;
  padding:10px 14px;margin-bottom:8px;cursor:pointer;transition:border-color .15s}
.search-result:hover{border-color:#3b82f6}
.sr-q{font-size:13px;color:#93c5fd;font-weight:500;margin-bottom:3px}
.sr-a{font-size:12px;color:#64748b;line-height:1.4}
.sr-meta{font-size:10px;color:#334155;margin-top:4px}

/* ── Test suite ── */
.test-row{display:flex;align-items:center;gap:10px;padding:10px 14px;
  background:#0d1224;border:1px solid #1e2a45;border-radius:8px;margin-bottom:6px}
.test-pass{color:#4ade80;font-size:13px;font-weight:600;min-width:18px}
.test-fail{color:#f87171;font-size:13px;font-weight:600;min-width:18px}
.test-skip{color:#fbbf24;font-size:13px;font-weight:600;min-width:18px}
.test-name{font-size:13px;color:#e2e8f0;font-weight:500}
.test-detail{font-size:11px;color:#475569;margin-top:2px}
.test-summary{background:#0d1224;border:1px solid #1e2a45;border-radius:10px;
  padding:14px 18px;margin-bottom:16px;display:flex;gap:24px;align-items:center}

/* ── Export report ── */
.export-btn{display:inline-flex;align-items:center;gap:6px;
  background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.3);
  border-radius:8px;padding:6px 14px;font-size:12px;
  color:#c4b5fd;font-weight:500;cursor:pointer}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def _init():
    for k, v in {
        "logged_in": False, "username": "", "messages": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# =============================================================================
# BACKEND HELPERS
# =============================================================================
def _post(ep, **kw):
    try:
        return requests.post(f"{BACKEND}{ep}", timeout=180, **kw).json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return {"success": False, "message": str(e)}

def _get(ep, timeout=8):
    try:
        return requests.get(f"{BACKEND}{ep}", timeout=timeout).json()
    except Exception:
        return None

def _delete(ep):
    try:
        return requests.delete(f"{BACKEND}{ep}", timeout=10).json()
    except Exception:
        return None

def _backend_ok():
    d = _get("/")
    return d is not None and d.get("status") == "ok"

def _confidence(answer, sources):
    if "could not find this information" in answer.lower():
        return "Low — not in knowledge base", "conf-lo"
    if sources and len(answer) > 200:
        return "High — grounded in sources", "conf-hi"
    return "Medium confidence", "conf-md"


# =============================================================================
# LOGIN
# =============================================================================
def _login():
    _, col, _ = st.columns([1.2, 1, 1.2])
    with col:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px">
          <div style="font-size:40px;margin-bottom:10px">⚡</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;margin-bottom:6px">
            Enterprise AI Copilot</div>
          <div style="font-size:13px;color:#475569">Sign in to access your knowledge base</div>
        </div>""", unsafe_allow_html=True)

        user = st.text_input("Username", placeholder="admin",    key="lu")
        pw   = st.text_input("Password", type="password",
                              placeholder="••••••••",             key="lp")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("Sign in →", key="lbtn"):
            if not user.strip() or not pw.strip():
                st.error("Enter username and password.")
                return
            with st.spinner("Authenticating..."):
                d = _post("/login", json={"username": user, "password": pw})
            if d is None:
                st.error("⚠️ Backend unreachable. Is uvicorn running on port 8000?")
            elif d.get("success"):
                st.session_state.logged_in = True
                st.session_state.username  = user.strip()
                st.rerun()
            else:
                st.error(d.get("message", "Invalid credentials."))

        st.markdown("""<div style="text-align:center;margin-top:14px;font-size:11px;color:#334155">
            Default: admin / admin123</div>""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
          <span class="sb-icon">⚡</span>
          <span class="sb-name">AI Copilot</span>
        </div>
        <div class="sb-sub">Enterprise Knowledge Assistant</div>""",
        unsafe_allow_html=True)

        # Status
        st.markdown('<div class="sb-label">System Status</div>', unsafe_allow_html=True)
        if _backend_ok():
            st.markdown('<div class="badge b-on"><span class="dot d-on"></span>Backend online</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge b-off"><span class="dot d-off"></span>Backend offline</div>',
                        unsafe_allow_html=True)

        od = _get("/ollama/status", timeout=5)
        if od and od.get("server_running"):
            if od.get("model_available"):
                st.markdown('<div class="badge b-on"><span class="dot d-on"></span>Llama 3 ready</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge b-warn"><span class="dot d-warn"></span>Model missing</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge b-off"><span class="dot d-off"></span>Ollama offline</div>',
                        unsafe_allow_html=True)

        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)

        # Company selector
        st.markdown('<div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
        cd = _get("/companies", timeout=4)
        existing = (cd.get("companies", []) if cd and cd.get("success") else [])
        if existing:
            chosen = st.selectbox("Switch KB", ["— new —"] + existing,
                                  key="kb_dd", label_visibility="collapsed")
        else:
            chosen = "— new —"
        default = chosen if chosen != "— new —" else "mycompany"
        company = st.text_input("Company name", value=default,
                                key="co_name",
                                help="Each company = isolated ChromaDB collection")

        # Chunk counter
        if company.strip():
            s = _get(f"/stats/{company.strip()}", timeout=4)
            if s and s.get("success"):
                n = s.get("chunks", 0)
                col = "#4ade80" if n > 0 else "#475569"
                st.markdown(
                    f'<div style="font-size:11px;color:{col};margin-top:-2px">'
                    f'📦 {n} chunks stored</div>', unsafe_allow_html=True)

        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)

        # Ingest
        st.markdown('<div class="sb-label">Ingest Website</div>', unsafe_allow_html=True)
        url = st.text_input("URL", placeholder="https://example.com",
                            key="url_in", label_visibility="collapsed")
        if st.button("🌐 Ingest website", key="ing_btn"):
            _do_ingest(company, url)

        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)

        # PDF
        st.markdown('<div class="sb-label">Upload PDF</div>', unsafe_allow_html=True)
        f = st.file_uploader("PDF", type=["pdf"], key="pdf_up",
                             label_visibility="collapsed")
        if f:
            _do_pdf(company, f)

        st.markdown('<hr class="sb-div">', unsafe_allow_html=True)

        # Clear + sign out
        if st.session_state.messages:
            if st.button("🗑️ Clear chat", key="clr"):
                st.session_state.messages = []
                st.rerun()

        st.markdown(
            f'<div style="font-size:11px;color:#334155;margin-top:6px;margin-bottom:6px">'
            f'Signed in as <strong style="color:#475569">'
            f'{st.session_state.username}</strong></div>', unsafe_allow_html=True)
        if st.button("Sign out", key="so"):
            st.session_state.logged_in = False
            st.session_state.messages  = []
            st.rerun()

    return company


def _do_ingest(company, url):
    if not url.strip():
        st.sidebar.warning("Enter a URL.")
        return
    if not url.strip().startswith(("http://","https://")):
        st.sidebar.error("URL must start with https://")
        return
    with st.sidebar:
        with st.spinner("Scraping..."):
            d = _post("/ingest_url", json={"company": company.strip(),
                                           "url": url.strip()})
    if d is None:
        st.sidebar.error("⚠️ Backend unreachable.")
    elif d.get("success"):
        st.sidebar.success(f"✓ {d.get('chunks',0)} chunks · {d.get('pages_scraped',0)} page(s)")
    else:
        st.sidebar.error(d.get("message","Failed."))


def _do_pdf(company, f):
    k = f"up_{f.name}_{f.size}"
    if st.session_state.get(k): return
    with st.sidebar:
        with st.spinner(f"Indexing {f.name}..."):
            d = _post(f"/upload_pdf?company={company.strip()}",
                      files={"file":(f.name, f.getvalue(), "application/pdf")})
    if d is None:
        st.sidebar.error("⚠️ Backend unreachable.")
    elif d.get("success"):
        st.sidebar.success(f"✓ {d.get('chunks',0)} chunks · {d.get('pages',0)} pages")
        st.session_state[k] = True
    else:
        st.sidebar.error(d.get("message","Failed."))


# =============================================================================
# HERO + DASHBOARD
# =============================================================================
def _hero():
    pills = "".join(f'<div class="pill">{i} {l}</div>' for i,l in TECH_PILLS)
    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-eyebrow">▸ &nbsp; Retrieval-Augmented Generation</div>
      <div class="hero-title">Enterprise <span class="hl">AI Knowledge</span> Copilot</div>
      <div class="hero-sub">
        Ask questions about any company. Ingest websites and PDFs —
        get grounded, citation-backed answers from your own private knowledge base.
      </div>
      <div class="pills">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def _dashboard():
    d = _get("/dashboard", timeout=6)
    if not d: return
    db  = d.get("database",{})
    vec = d.get("vector_store",{})
    llm = d.get("llm",{})
    tq  = db.get("total_queries",0)
    ct  = db.get("companies_tracked",0)
    tc  = vec.get("total_collections",0)
    lok = llm.get("model_available", False)
    llm_color = "#4ade80" if lok else "#f87171"
    llm_val = "Ready" if lok else "Offline"

    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1, str(tq),  "Total Queries",     "all time",            "🔍"),
        (c2, str(ct),  "Companies Queried", f"{tc} knowledge bases","🏢"),
        (c3, str(tc),  "Knowledge Bases",   "in ChromaDB",         "🗄️"),
        (c4, llm_val,  "LLM Status",        "Llama 3 via Ollama",  "🧠"),
    ]
    for col, num, lbl, sub, acc in cards:
        num_style = f"color:{llm_color}" if lbl == "LLM Status" else ""
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-accent">{acc}</div>'
                f'<div class="stat-num" style="{num_style}">{num}</div>'
                f'<div class="stat-lbl">{lbl}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# =============================================================================
# CHAT TAB
# =============================================================================
def _tab_chat(company):
    _hero()
    _dashboard()

    # Suggested prompts
    prompt_click = None
    if not st.session_state.messages:
        st.markdown('<div class="p-label">Suggested prompts</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, p in enumerate(SUGGESTED_PROMPTS):
            with cols[i % 3]:
                if st.button(p, key=f"sp{i}", use_container_width=True):
                    prompt_click = p

    # Messages
    if st.session_state.messages:
        st.markdown("""<div class="c-div">
          <div class="c-line"></div>Conversation<div class="c-line"></div>
        </div>""", unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.messages):
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        hist_id = msg.get("hist_id")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="u-bub">{content}</div>',
                            unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="a-bub">{content}</div>',
                            unsafe_allow_html=True)

                # Confidence + char count + latency
                conf_label, conf_cls = _confidence(content, sources)
                lat = msg.get("latency_ms")
                lat_str = f" · {lat:.0f} ms" if lat else ""
                st.markdown(
                    f'<div style="margin-top:4px">'
                    f'<span class="{conf_cls}">{conf_label}</span>'
                    f'<span style="font-size:10px;color:#334155;margin-left:10px">'
                    f'{len(content)} chars{lat_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

                # ── Star rating ────────────────────────────────────────────────
                if hist_id:
                    existing_rating = msg.get("rating", 0)
                    st.markdown(
                        '<div style="font-size:10px;color:#334155;margin-top:6px;'
                        'margin-bottom:2px">Rate this answer:</div>',
                        unsafe_allow_html=True)
                    r_cols = st.columns(5)
                    for star in range(1, 6):
                        with r_cols[star - 1]:
                            filled = "⭐" if star <= existing_rating else "☆"
                            if st.button(filled, key=f"star_{idx}_{star}"):
                                res = _post("/rate", json={
                                    "history_id": hist_id, "rating": star})
                                if res and res.get("success"):
                                    st.session_state.messages[idx]["rating"] = star
                                    st.rerun()

                # Sources
                if sources:
                    with st.expander(f"📎 {len(sources)} source chunk(s)"):
                        for i, src in enumerate(sources, 1):
                            preview = src[:400] + "..." if len(src) > 400 else src
                            st.markdown(
                                f'<div class="src-chunk">'
                                f'<span style="color:#334155;font-size:10px;'
                                f'font-weight:600">CHUNK {i}</span><br>{preview}</div>',
                                unsafe_allow_html=True)

                # ── Similar questions ──────────────────────────────────────────
                prev_q = msg.get("query", "")
                if prev_q:
                    sim_data = _get(
                        f"/similar/{company}?q={requests.utils.quote(prev_q)}",
                        timeout=4)
                    if sim_data and sim_data.get("similar"):
                        with st.expander("💡 Similar past questions"):
                            for s in sim_data["similar"]:
                                st.markdown(
                                    f'<div class="sim-card">'
                                    f'<div class="sim-q">{s["question"]}</div>'
                                    f'<div class="sim-a">{s["answer"][:120]}...</div>'
                                    f'</div>',
                                    unsafe_allow_html=True)

    # History panel
    if st.session_state.messages:
        _db_history(company)

    # Input
    user_input = st.chat_input(
        "Ask anything about the company knowledge base...", key="ci")
    if user_input:
        _query(company, user_input)
    elif prompt_click:
        _query(company, prompt_click)


# =============================================================================
# ANALYTICS TAB
# =============================================================================
def _tab_analytics():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#f1f5f9;
      letter-spacing:-.01em;margin-bottom:4px">📊 Query Analytics</div>
    <div style="font-size:13px;color:#475569;margin-bottom:22px">
      Live metrics from your SQLite database and ChromaDB.</div>
    """, unsafe_allow_html=True)

    dash = _get("/dashboard", timeout=6)
    if not dash:
        st.error("Cannot reach backend.")
        return

    companies = dash.get("vector_store", {}).get("companies", [])
    total_q   = dash.get("database", {}).get("total_queries", 0)

    if not companies or total_q == 0:
        st.info("No queries yet. Ask some questions first.")
        return

    # Top-level stats
    c1,c2,c3 = st.columns(3)
    avg = round(total_q / max(len(companies),1), 1)
    for col, num, lbl in [
        (c1,total_q,"Total Queries"),
        (c2,len(companies),"Knowledge Bases"),
        (c3,avg,"Avg per KB"),
    ]:
        with col:
            st.markdown(f"""<div class="card">
              <div class="c-num">{num}</div>
              <div class="c-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # Per-company detailed analytics
    for co in companies:
        an = _get(f"/analytics/{co}", timeout=5)
        if not an or not an.get("success"):
            continue

        with st.expander(f"📂 {co}  —  {an.get('total_queries',0)} queries",
                         expanded=(len(companies)==1)):

            a1,a2,a3,a4 = st.columns(4)
            rt = an.get("response_time_ms", {})
            for col, val, lbl in [
                (a1, an.get("total_queries",0),        "Queries"),
                (a2, f"{rt.get('avg','—')} ms" if rt.get('avg') else "—", "Avg Latency"),
                (a3, f"★ {an.get('avg_rating','—')}" if an.get('avg_rating') else "—", "Avg Rating"),
                (a4, an.get("avg_answer_length","—"),  "Avg Ans Len"),
            ]:
                with col:
                    st.markdown(f"""<div class="card" style="padding:12px">
                      <div class="c-num" style="font-size:20px">{val}</div>
                      <div class="c-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            # Daily volume bar chart
            daily = an.get("recent_daily_counts", {})
            if daily:
                st.markdown('<div class="a-title" style="margin-top:16px">Daily query volume (last 7 days)</div>',
                            unsafe_allow_html=True)
                max_d = max(daily.values()) if daily else 1
                for day, cnt in sorted(daily.items()):
                    pct = int((cnt / max_d) * 100)
                    st.markdown(
                        f'<div style="margin-bottom:8px">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'margin-bottom:3px">'
                        f'<span style="font-size:12px;color:#94a3b8">{day}</span>'
                        f'<span style="font-size:11px;color:#475569">{cnt}</span></div>'
                        f'<div style="background:#1e2a45;border-radius:4px;height:7px">'
                        f'<div style="background:linear-gradient(90deg,#3b82f6,#8b5cf6);'
                        f'width:{pct}%;height:7px;border-radius:4px"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True)

            # Top questions
            top_q = an.get("top_questions", [])
            if top_q:
                st.markdown('<div class="a-title" style="margin-top:16px">Recent unique questions</div>',
                            unsafe_allow_html=True)
                for q in top_q:
                    st.markdown(
                        f'<div style="font-size:12px;color:#94a3b8;padding:5px 0;'
                        f'border-bottom:1px solid #1e2a45">▸ {q}</div>',
                        unsafe_allow_html=True)


# =============================================================================
# KNOWLEDGE BASES TAB
# =============================================================================
def _tab_kb():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#f1f5f9;
      letter-spacing:-.01em;margin-bottom:4px">🗄️ Knowledge Bases</div>
    <div style="font-size:13px;color:#475569;margin-bottom:22px">
      All ChromaDB collections. Each company is fully isolated.</div>
    """, unsafe_allow_html=True)

    # ── Batch ingest tool ──────────────────────────────────────────────────────
    with st.expander("⚡ Batch ingest — multiple URLs at once", expanded=False):
        st.markdown(
            '<div style="font-size:12px;color:#475569;margin-bottom:8px">'
            'Enter one URL per line (max 10). All will be ingested for the '
            'company selected in the sidebar.</div>',
            unsafe_allow_html=True)

        batch_company = st.text_input("Company for batch", key="bc",
                                       placeholder="mycompany")
        batch_urls_raw = st.text_area("URLs (one per line)", key="burls",
                                       placeholder="https://example.com\nhttps://example.com/about",
                                       height=100)

        if st.button("🚀 Run batch ingest", key="batch_run"):
            urls = [u.strip() for u in batch_urls_raw.strip().split("\n") if u.strip()]
            if not urls:
                st.warning("Enter at least one URL.")
            elif not batch_company.strip():
                st.warning("Enter a company name.")
            else:
                with st.spinner(f"Ingesting {len(urls)} URL(s)..."):
                    d = _post("/ingest_batch",
                              json={"company": batch_company.strip(), "urls": urls})
                if d is None:
                    st.error("Backend unreachable.")
                elif d.get("success") or d.get("urls_ok", 0) > 0:
                    st.success(
                        f"✓ {d.get('urls_ok',0)}/{len(urls)} URLs · "
                        f"{d.get('total_chunks',0)} total chunks")
                    for r in d.get("results", []):
                        if r.get("success"):
                            st.markdown(
                                f'<div class="batch-row-ok">✓ {r["url"]} → '
                                f'{r.get("chunks",0)} chunks</div>',
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f'<div class="batch-row-err">✗ {r["url"]} → '
                                f'{r.get("message","error")}</div>',
                                unsafe_allow_html=True)
                else:
                    st.error(d.get("message","Batch failed."))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── KB list ────────────────────────────────────────────────────────────────
    cd = _get("/companies", timeout=6)
    if not cd:
        st.error("Cannot reach backend.")
        return
    companies = cd.get("companies", [])

    if not companies:
        st.info("No knowledge bases yet. Ingest a website or upload a PDF.")
        return

    st.markdown(
        f'<div style="font-size:12px;color:#475569;margin-bottom:14px">'
        f'{len(companies)} knowledge base(s)</div>',
        unsafe_allow_html=True)

    for co in companies:
        s = _get(f"/stats/{co}", timeout=4)
        chunks = s.get("chunks", 0) if s else "?"

        c_name, c_ch, c_exp, c_del = st.columns([3, 2, 2, 1])

        with c_name:
            st.markdown(
                f'<div style="padding:8px 0">'
                f'<div class="kb-name">🗄️ {co}</div>'
                f'<div class="kb-sub">ChromaDB collection</div></div>',
                unsafe_allow_html=True)

        with c_ch:
            clr = "#4ade80" if isinstance(chunks,int) and chunks>0 else "#475569"
            st.markdown(
                f'<div style="padding:8px 0;color:{clr};font-size:13px;font-weight:600">'
                f'📦 {chunks}</div>', unsafe_allow_html=True)

        with c_exp:
            # Export CSV button
            if st.button("📥 Export CSV", key=f"exp_{co}"):
                res = _get(f"/export/{co}", timeout=10)
                if res and res.get("success"):
                    csv_data = res.get("csv","")
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"{co}_history.csv",
                        mime="text/csv",
                        key=f"dl_{co}",
                    )
                else:
                    st.error("Export failed.")

        with c_del:
            if st.button("🗑️", key=f"del_{co}", help=f"Delete {co}"):
                r = _delete(f"/company/{co}")
                if r and r.get("success"):
                    st.success(f"Deleted: {co}")
                    st.rerun()
                else:
                    st.error("Delete failed.")

        st.markdown('<div style="border-bottom:1px solid #1e2a45;margin-bottom:4px"></div>',
                    unsafe_allow_html=True)


# =============================================================================
# ABOUT TAB
# =============================================================================
def _tab_about():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#f1f5f9;
      letter-spacing:-.01em;margin-bottom:4px">📖 About This Project</div>
    <div style="font-size:13px;color:#475569;margin-bottom:22px">
      Enterprise AI Knowledge Copilot · Final-Year B.Tech AI & Data Science</div>
    """, unsafe_allow_html=True)

    # Architecture
    st.markdown('<div class="a-title">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-box">
    ┌─────────────────────────────────────────────────────┐<br>
    │  STREAMLIT FRONTEND :8501                           │<br>
    │  Login · Chat · Analytics · KB Manager · About      │<br>
    └──────────────┬──────────────────────────────────────┘<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp; HTTP / SSE Streaming<br>
    ┌──────────────▼──────────────────────────────────────┐<br>
    │  FASTAPI BACKEND :8000  (17 endpoints)              │<br>
    │  /query/stream · /analytics · /export · /rate       │<br>
    │  /ingest_batch · /similar · /dashboard              │<br>
    └───┬──────────┬──────────────┬───────────────────────┘<br>
    &nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
    ┌───▼──────┐ ┌─▼──────────┐ ┌─▼──────────────┐<br>
    │ ChromaDB │ │   SQLite   │ │ Ollama :11434  │<br>
    │ Vectors  │ │  History   │ │ Llama 3 8B     │<br>
    └──────────┘ └────────────┘ └────────────────┘<br>
    </div>
    """, unsafe_allow_html=True)

    # RAG pipeline
    st.markdown('<div class="a-title" style="margin-top:22px">RAG Pipeline</div>',
                unsafe_allow_html=True)
    for num, title, desc in [
        ("1","Ingest",   "Website scraper (BS4) or pypdf → raw text"),
        ("2","Chunk",    "Sliding window · 700 chars · 120 overlap"),
        ("3","Embed",    "sentence-transformers all-MiniLM-L6-v2 → 384-dim"),
        ("4","Store",    "ChromaDB PersistentClient · HNSW index · per-company"),
        ("5","Retrieve", "Query embedding → cosine sim → top-4 chunks"),
        ("6","Generate", "Grounded prompt → Ollama Llama 3 → SSE stream"),
        ("7","Persist",  "Q&A + latency + chunks → SQLite chat_history"),
        ("8","Rate",     "User 1–5 star rating stored against history row"),
    ]:
        st.markdown(
            f'<div style="display:flex;gap:12px;align-items:flex-start;'
            f'margin-bottom:10px;padding:10px 14px;background:#0d1224;'
            f'border:1px solid #1e2a45;border-radius:8px">'
            f'<div style="min-width:24px;height:24px;background:rgba(59,130,246,.15);'
            f'border:1px solid rgba(59,130,246,.3);border-radius:50%;display:flex;'
            f'align-items:center;justify-content:center;font-size:11px;'
            f'font-weight:700;color:#93c5fd">{num}</div>'
            f'<div><div style="font-size:13px;font-weight:600;color:#f1f5f9">{title}</div>'
            f'<div style="font-size:12px;color:#475569;margin-top:2px">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="a-title" style="margin-top:22px">Tech Stack</div>',
                unsafe_allow_html=True)
    stack = [
        ("FastAPI",               "REST API + SSE streaming · 17 endpoints",  "Backend"),
        ("Streamlit",             "Multi-tab SaaS UI",                         "Frontend"),
        ("ChromaDB",              "Persistent HNSW vector store",              "Vector DB"),
        ("Ollama + Llama 3",      "Local LLM inference — no API key",          "LLM"),
        ("sentence-transformers", "all-MiniLM-L6-v2 · 384-dim embeddings",    "Embeddings"),
        ("SQLAlchemy + SQLite",   "Chat history · ratings · latency audit",    "Database"),
        ("requests + BS4",        "BFS / single-page website scraper",         "Scraper"),
        ("pypdf",                 "Pure-Python PDF text extraction",           "Ingestion"),
    ]
    st.markdown(
        '<div style="background:#0d1224;border:1px solid #1e2a45;'
        'border-radius:10px;padding:4px 16px">',
        unsafe_allow_html=True)
    for name, role, cat in stack:
        st.markdown(
            f'<div class="tech-row">'
            f'<span class="t-name">{name}</span>'
            f'<span class="t-role">{role}</span>'
            f'<span class="t-badge">{cat}</span>'
            f'</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:22px;padding:16px;background:#0d1224;
      border:1px solid #1e2a45;border-radius:10px;
      font-size:12px;color:#475569;line-height:1.9">
      <strong style="color:#64748b">Project:</strong> Enterprise AI Knowledge Copilot<br>
      <strong style="color:#64748b">Degree:</strong> B.Tech — AI &amp; Data Science<br>
      <strong style="color:#64748b">Key concepts:</strong>
        RAG · Vector Embeddings · LLM Grounding · Multi-tenant Architecture ·
        Semantic Search · SSE Streaming · Confidence Scoring · Latency Tracking<br>
      <strong style="color:#64748b">API Docs:</strong>
        <a href="http://localhost:8000/docs" target="_blank"
           style="color:#3b82f6">http://localhost:8000/docs</a>
        &nbsp;(17 endpoints)
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DB HISTORY PANEL
# =============================================================================
def _db_history(company):
    st.markdown('<hr style="border:none;border-top:1px solid #1e2a45;margin:14px 0">',
                unsafe_allow_html=True)
    with st.expander("📋 Saved conversation history", expanded=False):
        cl, cr = st.columns([2,1])
        with cl:
            st.markdown(
                f'<span style="font-size:12px;color:#475569">'
                f'Company: <strong style="color:#64748b">{company}</strong></span>',
                unsafe_allow_html=True)
        with cr:
            load = st.button("🔄 Load", key="lh", use_container_width=True)

        if load:
            d = _get(f"/history/{company.strip()}")
            if d is None:
                st.error("Backend unreachable.")
            elif not d.get("success"):
                st.error(d.get("message","Failed."))
            elif d.get("total",0) == 0:
                st.info("No history yet.")
            else:
                st.markdown(
                    f'<div style="font-size:12px;color:#475569;margin-bottom:10px">'
                    f'<strong style="color:#64748b">{d["total"]}</strong> conversation(s)</div>',
                    unsafe_allow_html=True)
                for item in d["history"]:
                    preview = item["answer"][:200]+"..." if len(item["answer"])>200 else item["answer"]
                    st.markdown(
                        f'<div class="hist-item">'
                        f'<div class="h-time">🕐 {item["time"][:19]}</div>'
                        f'<div class="h-q">Q: {item["question"]}</div>'
                        f'<div class="h-a">A: {preview}</div>'
                        f'</div>', unsafe_allow_html=True)


# =============================================================================
# STREAMING QUERY
# =============================================================================
def _query(company, query_text):
    if not query_text.strip(): return
    if not company.strip():
        st.error("Enter a company name in the sidebar.")
        return

    st.session_state.messages.append({
        "role": "user", "content": query_text.strip(),
        "sources": [], "query": query_text.strip(),
    })
    with st.chat_message("user"):
        st.markdown(f'<div class="u-bub">{query_text.strip()}</div>',
                    unsafe_allow_html=True)

    with st.chat_message("assistant"):
        ph = st.empty()

    accumulated = ""
    sources = []
    t_start = time.perf_counter()

    try:
        with requests.post(
            f"{BACKEND}/query/stream",
            json={"company": company.strip(), "query": query_text.strip()},
            stream=True, timeout=180,
        ) as resp:
            for raw in resp.iter_lines():
                if not raw: continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "): continue
                try:
                    payload = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "token" in payload:
                    accumulated += payload["token"]
                    ph.markdown(f'<div class="a-bub">{accumulated}▌</div>',
                                unsafe_allow_html=True)

                if payload.get("done"):
                    sources = payload.get("sources", [])
                    ph.markdown(f'<div class="a-bub">{accumulated}</div>',
                                unsafe_allow_html=True)
                    break

    except requests.exceptions.ConnectionError:
        accumulated = "⚠️ Cannot reach backend. Is uvicorn running on port 8000?"
        ph.markdown(f'<div class="a-bub">{accumulated}</div>', unsafe_allow_html=True)
    except Exception as e:
        accumulated = f"⚠️ Error: {e}"
        ph.markdown(f'<div class="a-bub">{accumulated}</div>', unsafe_allow_html=True)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # Fetch the history ID just saved so we can attach it for star rating
    hist_id = None
    try:
        hist_d = _get(f"/history/{company.strip()}", timeout=4)
        if hist_d and hist_d.get("success") and hist_d.get("history"):
            hist_id = hist_d["history"][0]["id"]
    except Exception:
        pass

    st.session_state.messages.append({
        "role": "assistant", "content": accumulated,
        "sources": sources, "query": query_text.strip(),
        "latency_ms": round(latency_ms, 1),
        "hist_id": hist_id,
        "rating": 0,
    })
    st.rerun()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not st.session_state.logged_in:
        _login()
        st.stop()

    company = _sidebar()

    tab_chat, tab_analytics, tab_kb, tab_test, tab_about = st.tabs([
        "💬 Chat",
        "📊 Analytics",
        "🗄️ Knowledge Bases",
        "🧪 Test Suite",
        "📖 About",
    ])

    with tab_chat:
        _tab_chat(company)

    with tab_analytics:
        _tab_analytics()

    with tab_kb:
        _tab_kb()

    with tab_test:
        _tab_test(company)

    with tab_about:
        _tab_about()


# =============================================================================
# ENHANCEMENT 1 — HISTORY SEARCH
# Added to Chat ta

# =============================================================================
# ENHANCEMENT 1 — HISTORY SEARCH
# =============================================================================

def _search_history(company: str):
    """Keyword search over saved conversations for a company."""
    st.markdown(
        '<div style="font-size:10px;font-weight:600;letter-spacing:.07em;'        'text-transform:uppercase;color:#475569;margin-bottom:8px">'        'Search past conversations</div>',
        unsafe_allow_html=True,
    )
    q = st.text_input(
        "Search", placeholder="Type a keyword to search past Q&As...",
        key="hist_search", label_visibility="collapsed",
    )
    if not q or len(q.strip()) < 2:
        return

    with st.spinner("Searching..."):
        data = _get(f"/history/{company.strip()}")
    if not data or not data.get("success"):
        return

    keyword = q.strip().lower()
    matches = [
        item for item in data.get("history", [])
        if keyword in item["question"].lower()
        or keyword in item["answer"].lower()
    ]
    if not matches:
        st.info(f'No saved conversations containing "{q}"')
        return

    st.markdown(
        f'<div style="font-size:12px;color:#475569;margin-bottom:10px">'        f'<strong style="color:#64748b">{len(matches)}</strong> result(s) for '        f'<strong style="color:#93c5fd">"{q}"</strong></div>',
        unsafe_allow_html=True,
    )
    for item in matches[:10]:
        ans_preview = item["answer"][:180] + "..." if len(item["answer"]) > 180 else item["answer"]
        st.markdown(
            f'<div class="search-result">'            f'<div class="sr-q">Q: {item["question"]}</div>'            f'<div class="sr-a">A: {ans_preview}</div>'            f'<div class="sr-meta">🕐 {item["time"][:19]}  ·  '            f'<span style="color:#3b82f6">{company}</span></div>'            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# ENHANCEMENT 2 — AUTOMATED TEST SUITE TAB
# =============================================================================

def _tab_test(company: str):
    """10 live automated system checks across every layer."""
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#f1f5f9;
      letter-spacing:-.01em;margin-bottom:4px">🧪 System Test Suite</div>
    <div style="font-size:13px;color:#475569;margin-bottom:22px">
      Live automated checks across every layer of the stack.</div>
    """, unsafe_allow_html=True)

    test_company = st.text_input(
        "Company to test against", value=company or "testco", key="test_co",
        help="Uses this company name for ingest/query tests.",
    )

    if not st.button("Run all tests", key="run_tests"):
        st.markdown(
            '<div style="font-size:12px;color:#475569;padding:14px;'            'background:#0d1224;border:1px solid #1e2a45;border-radius:8px;">'            'Click <strong style="color:#f1f5f9">Run all tests</strong> to check '            'every component. Takes about 30-60 seconds.</div>',
            unsafe_allow_html=True,
        )
        return

    results = []
    progress = st.progress(0)
    status_box = st.empty()

    def run_test(name, fn):
        status_box.markdown(
            f'<div style="font-size:12px;color:#64748b">Running: {name}...</div>',
            unsafe_allow_html=True,
        )
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"Exception: {str(e)}"
        results.append((name, passed, detail))
        progress.progress(len(results) / 10)

    def t1():
        d = _get("/")
        if d and d.get("status") == "ok":
            return True, f"Version {d.get('version','?')} running"
        return False, "GET / did not return status=ok"
    run_test("Backend health check", t1)

    def t2():
        d = _post("/login", json={"username": "admin", "password": "admin123"})
        if d and d.get("success"):
            return True, "Login successful with default credentials"
        return False, f"Login failed: {d.get('message','no response') if d else 'no response'}"
    run_test("Auth — correct credentials accepted", t2)

    def t3():
        d = _post("/login", json={"username": "admin", "password": "wrongpassword"})
        if d and not d.get("success"):
            return True, "Correctly rejected wrong password"
        return False, "Wrong credentials were accepted — auth is broken"
    run_test("Auth — wrong credentials rejected", t3)

    def t4():
        d = _get("/ollama/status", timeout=8)
        if d and d.get("server_running"):
            if d.get("model_available"):
                return True, "Ollama running · llama3 available"
            return None, "Server up but llama3 not installed — run: ollama pull llama3"
        return False, "Ollama not reachable — run: ollama serve"
    run_test("Ollama server + model", t4)

    def t5():
        d = _get("/companies")
        if d and d.get("success"):
            names = d.get("companies", [])
            return True, f"{len(names)} collection(s): {names or '(empty)'}"
        return False, "GET /companies failed"
    run_test("ChromaDB collection listing", t5)

    def t6():
        d = _post("/ingest_url", json={
            "company": test_company,
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        })
        if d and d.get("success"):
            return True, f"{d.get('chunks',0)} chunks · {d.get('pages_scraped',0)} page(s)"
        return False, f"Ingest failed: {d.get('message','no response') if d else 'no response'}"
    run_test("Website ingestion pipeline", t6)

    def t7():
        d = _get(f"/stats/{test_company}", timeout=5)
        if d and d.get("success"):
            n = d.get("chunks", 0)
            if n > 0:
                return True, f"{n} chunks stored in '{test_company}'"
            return False, "Ingest ran but 0 chunks stored"
        return False, "GET /stats failed"
    run_test("Vector store chunk count", t7)

    def t8():
        d = _post("/query", json={
            "company": test_company,
            "query": "What is artificial intelligence?",
        })
        if d and d.get("success") and d.get("answer"):
            ans = d["answer"]
            sources = d.get("sources", [])
            if "could not find" in ans.lower():
                return None, f"Fallback response — check chunk quality. Sources: {len(sources)}"
            return True, f"{ans[:100]}...  |  Sources: {len(sources)}"
        return False, f"Query failed: {d.get('message','no response') if d else 'no response'}"
    run_test("RAG query pipeline", t8)

    def t9():
        d = _get(f"/history/{test_company}", timeout=5)
        if d and d.get("success"):
            total = d.get("total", 0)
            if total > 0:
                return True, f"{total} conversation(s) saved in SQLite"
            return False, "History endpoint works but 0 rows — persistence failed"
        return False, "GET /history failed"
    run_test("SQLite history persistence", t9)

    def t10():
        d = _get(f"/analytics/{test_company}", timeout=5)
        if d and d.get("success"):
            tq = d.get("total_queries", 0)
            return True, f"{tq} queries · avg_len={d.get('avg_answer_length','?')}"
        return False, "GET /analytics failed"
    run_test("Analytics endpoint", t10)

    progress.empty()
    status_box.empty()

    passed_n = sum(1 for _, p, _ in results if p is True)
    warned_n = sum(1 for _, p, _ in results if p is None)
    failed_n = sum(1 for _, p, _ in results if p is False)

    st.markdown(
        f'<div class="test-summary">'        f'<span style="font-size:22px;font-weight:700;color:#f1f5f9">{len(results)}</span>'        f'<span style="font-size:11px;color:#475569">tests</span>'        f'&nbsp;&nbsp;'        f'<span style="font-size:18px;font-weight:700;color:#4ade80">{passed_n}</span>'        f'<span style="font-size:11px;color:#475569">passed</span>'        f'&nbsp;&nbsp;'        f'<span style="font-size:18px;font-weight:700;color:#fbbf24">{warned_n}</span>'        f'<span style="font-size:11px;color:#475569">warnings</span>'        f'&nbsp;&nbsp;'        f'<span style="font-size:18px;font-weight:700;color:#f87171">{failed_n}</span>'        f'<span style="font-size:11px;color:#475569">failed</span>'        f'</div>',
        unsafe_allow_html=True,
    )

    for name, passed_flag, detail in results:
        if passed_flag is True:
            icon = '<span class="test-pass">✓</span>'
        elif passed_flag is None:
            icon = '<span class="test-skip">⚠</span>'
        else:
            icon = '<span class="test-fail">✗</span>'
        st.markdown(
            f'<div class="test-row">{icon}<div>'            f'<div class="test-name">{name}</div>'            f'<div class="test-detail">{detail}</div>'            f'</div></div>',
            unsafe_allow_html=True,
        )

    if failed_n == 0 and warned_n == 0:
        st.success("All tests passed — take a screenshot for your project report.")
    elif failed_n == 0:
        st.warning(f"{warned_n} warning(s) — check before demo.")
    else:
        st.error(f"{failed_n} test(s) failed — fix before demo.")

    report_lines = [
        "Enterprise AI Knowledge Copilot — System Test Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Test company: {test_company}", "=" * 60, "",
    ]
    for name, passed_flag, detail in results:
        status = "PASS" if passed_flag is True else ("WARN" if passed_flag is None else "FAIL")
        report_lines += [f"[{status}]  {name}", f"       {detail}", ""]
    report_lines += ["=" * 60, f"Summary: {passed_n} passed · {warned_n} warnings · {failed_n} failed"]

    st.download_button(
        label="Download test report (.txt)",
        data="\n".join(report_lines),
        file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain", key="dl_test_report",
    )


# =============================================================================
# ENHANCEMENT 3 — OVERRIDE _tab_chat TO ADD SEARCH
# =============================================================================

def _tab_chat(company: str):
    """Chat tab with hero, dashboard, prompts, messages, search."""
    _hero()
    _dashboard()

    prompt_click = None
    if not st.session_state.messages:
        # Render prompts as styled chips, not full-width buttons
        st.markdown('<div class="p-label" style="margin-bottom:10px">Try asking</div>',
                    unsafe_allow_html=True)
        # Use columns so they sit side by side like chips
        cols = st.columns(len(SUGGESTED_PROMPTS))
        for i, p in enumerate(SUGGESTED_PROMPTS):
            with cols[i]:
                if st.button(
                    p, key=f"sp{i}",
                    use_container_width=True,
                    help=p,
                ):
                    prompt_click = p
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("""<div class="c-div">
          <div class="c-line"></div>Conversation<div class="c-line"></div>
        </div>""", unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.messages):
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="u-bub">{content}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="a-bub">{content}</div>', unsafe_allow_html=True)

                conf_label, conf_cls = _confidence(content, sources)
                lat = msg.get("latency_ms", "")
                lat_str = (
                    f'<span style="font-size:10px;color:#334155;margin-left:10px">'                    f'⏱ {lat:.0f} ms</span>'
                    if lat else ""
                )
                st.markdown(
                    f'<div style="margin-top:4px">'                    f'<span class="{conf_cls}">{conf_label}</span>{lat_str}'                    f'<span style="font-size:10px;color:#334155;margin-left:10px">'                    f'{len(content)} chars</span></div>',
                    unsafe_allow_html=True,
                )

                hist_id = msg.get("hist_id")
                if hist_id:
                    current_rating = msg.get("rating", 0)
                    star_cols = st.columns(5)
                    for s in range(1, 6):
                        with star_cols[s - 1]:
                            if st.button(
                                "★" if s <= current_rating else "☆",
                                key=f"star_{idx}_{s}",
                            ):
                                res = _post("/rate", json={
                                    "company": company,
                                    "history_id": hist_id,
                                    "rating": s,
                                })
                                if res and res.get("success"):
                                    st.session_state.messages[idx]["rating"] = s
                                    st.rerun()

                if sources:
                    with st.expander(f"📎 {len(sources)} source chunk(s)"):
                        for i, src in enumerate(sources, 1):
                            preview = src[:400] + "..." if len(src) > 400 else src
                            st.markdown(
                                f'<div class="src-chunk">'                                f'<span style="color:#334155;font-size:10px;font-weight:600">'                                f'CHUNK {i}</span><br>{preview}</div>',
                                unsafe_allow_html=True)

    if st.session_state.messages:
        _db_history(company)

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e2a45;margin:14px 0">',
        unsafe_allow_html=True,
    )
    _search_history(company)

    user_input = st.chat_input("Ask anything about the company knowledge base...", key="ci")
    if user_input:
        _query(company, user_input)
    elif prompt_click:
        _query(company, prompt_click)


# =============================================================================
# ENHANCEMENT 4 — PROJECT REPORT EXPORT added to KB tab
# =============================================================================

_tab_kb_orig = _tab_kb  # type: ignore[name-defined]

def _tab_kb():
    """KB tab with report export appended."""
    _tab_kb_orig()

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1e2a45;margin:20px 0">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:10px;font-weight:600;letter-spacing:.07em;'        'text-transform:uppercase;color:#475569;margin-bottom:10px">'        'Project Report Export</div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate project report (.md)", key="gen_report"):
        dash  = _get("/dashboard", timeout=6)
        db_d  = (dash or {}).get("database", {})
        vec_d = (dash or {}).get("vector_store", {})
        llm_d = (dash or {}).get("llm", {})

        total_q   = db_d.get("total_queries", 0)
        companies = vec_d.get("companies", [])
        llm_s     = "Ready" if llm_d.get("model_available") else "Offline"
        now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Enterprise AI Knowledge Copilot
## Project Report — {now}

**Degree:** B.Tech — Artificial Intelligence & Data Science  
**Stack:** FastAPI · Streamlit · ChromaDB · Ollama Llama 3 · SQLite · sentence-transformers

---

## Architecture

```
[Streamlit :8501]  →  [FastAPI :8000]  →  [ChromaDB | SQLite | Ollama :11434]
```

## RAG Pipeline

1. Ingest — requests+BS4 or pypdf → raw text
2. Chunk — 700 chars, 120 overlap
3. Embed — all-MiniLM-L6-v2 → 384-dim vectors
4. Store — ChromaDB HNSW per company
5. Retrieve — cosine sim → top-4
6. Generate — Ollama Llama 3 → SSE stream
7. Persist — SQLite chat_history

## Live Stats

| Metric | Value |
|--------|-------|
| Total queries | {total_q} |
| Knowledge bases | {len(companies)} |
| Companies | {", ".join(companies) if companies else "none"} |
| LLM status | {llm_s} |

## 17 API Endpoints

GET / · POST /login · POST /ingest_url · POST /ingest_batch  
POST /upload_pdf · POST /query · POST /query/stream  
GET /history/{{co}} · GET /analytics/{{co}} · GET /stats/{{co}}  
GET /companies · DELETE /company/{{co}} · GET /dashboard  
GET /export/{{co}} · GET /similar/{{co}} · GET /ollama/status · POST /rate

---
*Generated by Enterprise AI Copilot v1.0.0*
"""
        st.download_button(
            label="Download (.md)",
            data=report,
            file_name=f"project_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown", key="dl_report",
        )
        st.success("Report ready — click download above.")


# =============================================================================
# ENTRY POINT — must be last, after all functions are defined
# =============================================================================
main()