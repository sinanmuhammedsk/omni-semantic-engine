import sys
import os
import subprocess
import time

# 1. This tells Streamlit to look at the main folder so it can find the backend folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. START THE BACKEND SERVER VIA UVICORN EXPLICITLY
if "backend_started" not in os.environ:
    os.environ["backend_started"] = "true"
    
    # This fires up Uvicorn on localhost port 8000 pointing directly to your app instance
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.main:app", 
        "--host", "127.0.0.1", 
        "--port", "8000"
    ])
    time.sleep(5) # Give Uvicorn a full 5 seconds to bind to the port completely

import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session
from backend.db.models import DocumentMetadata
from backend.db.base import get_db
# ─────────────────────────────────────────────────────────────────────────────
# 0. ENVIRONMENT & PAGE CONFIG  (Must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

st.set_page_config(
    page_title="OMNI | Enterprise AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# 1. SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "pdf_uploaded": False,
    "pipeline_proceeded": False,
    "pdf_processed": False,
    "active_doc_id": None,
    "active_doc_name": None,
    "messages": [],
    "theme": "dark",          # "dark" | "light"
    "theme_radio": "Light Mode",
    "dots_open": False,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM  — reads key from .env
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    # 1. Safely pull from Streamlit Secrets first; fallback to local environment variables (.env)
    live_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    model    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    if not live_key:
        raise ValueError("Critical Error: GROQ_API_KEY could not be loaded from secrets or .env")
        
    return ChatGroq(
        groq_api_key=live_key,
        model_name=model,
        temperature=0.2,
        streaming=True,
    )

llm = get_llm()

# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKEND HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upload_file(file):
    files = {"file": (file.name, file, "application/pdf")}
    r = requests.post(f"{API_URL}/upload", files=files)
    return r.json()

def clear_backend_documents():
    try:
        r = requests.delete(f"{API_URL}/documents")
        return r.status_code == 200
    except Exception:
        return False

def get_document_status(doc_id: int):
    try:
        r = requests.get(f"{API_URL}/documents")
        if r.status_code == 200:
            docs = r.json()
            match = next((d for d in docs if d["id"] == doc_id), None)
            return match["status"] if match else None
    except Exception:
        return None

def query_llm_stream(prompt: str):
    try:
        r = requests.post(f"{API_URL}/retrieve", json={"query": prompt, "top_k": 5})
        chunks = r.json().get("chunks", []) if r.status_code == 200 else []
    except Exception:
        chunks = []

    if chunks:
        context = "\n\n---\n\n".join(
            f"Source: {h['filename']} (Page {h['page_number']})\nContent: {h['text']}"
            for h in chunks
        )
        full_prompt = (
            "You are a helpful enterprise AI assistant. Prioritize using the provided document context "
            "to answer the question. If the context does not contain the answer or is not relevant, "
            "answer the question to the best of your general knowledge, and clearly note that you are "
            "using general knowledge as a fallback.\n\n"
            f"Context:\n{context}\n\nQuestion: {prompt}"
        )
    else:
        full_prompt = (
            "You are a helpful enterprise AI assistant. "
            "Explain that no relevant documents were found, but answer to the best of your knowledge.\n\n"
            f"Question: {prompt}"
        )

    for chunk in llm.stream(full_prompt):
        if chunk.content:
            yield chunk.content

# ─────────────────────────────────────────────────────────────────────────────
# 4. THEME TOKENS
# ─────────────────────────────────────────────────────────────────────────────
def get_theme_tokens(mode: str) -> dict:
    if mode == "dark":
        return {
            "bg":              "#090D1A",
            "surface":         "#0F1629",
            "surface2":        "#141D35",
            "border":          "#1E2A45",
            "border_accent":   "#2D3A5C",
            "text":            "#E8EDF7",
            "text_muted":      "#6B7BA4",
            "text_subtle":     "#404E72",
            "accent":          "#818CF8",
            "accent2":         "#C084FC",
            "accent_glow":     "rgba(129,140,248,0.12)",
            "accent_glow2":    "rgba(192,132,252,0.08)",
            "success":         "#34D399",
            "warning":         "#FBBF24",
            "danger":          "#F87171",
            "input_bg":        "#0F1629",
            "input_border":    "#1E2A45",
            "shadow":          "rgba(0,0,0,0.4)",
            "shadow_accent":   "rgba(129,140,248,0.15)",
            "scrollbar":       "#1E2A45",
        }
    else:
        return {
            "bg":              "#F0F4FF",
            "surface":         "#FFFFFF",
            "surface2":        "#F8FAFF",
            "border":          "#DDE3F0",
            "border_accent":   "#C5CEEA",
            "text":            "#111827",
            "text_muted":      "#4B5A7A",
            "text_subtle":     "#9AA3BE",
            "accent":          "#4F46E5",
            "accent2":         "#7C3AED",
            "accent_glow":     "rgba(79,70,229,0.10)",
            "accent_glow2":    "rgba(124,58,237,0.07)",
            "success":         "#059669",
            "warning":         "#D97706",
            "danger":          "#DC2626",
            "input_bg":        "#FFFFFF",
            "input_border":    "#DDE3F0",
            "shadow":          "rgba(0,0,0,0.08)",
            "shadow_accent":   "rgba(79,70,229,0.12)",
            "scrollbar":       "#DDE3F0",
        }

T = get_theme_tokens(st.session_state.theme)
accent_svg = T['accent'].replace('#', '%23')


# ─────────────────────────────────────────────────────────────────────────────
# 5. CSS DESIGN SYSTEM INJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stApp"], .stApp {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: {T['bg']} !important;
    color: {T['text']} !important;
    transition: background-color 0.35s ease, color 0.35s ease;
}}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu,
footer,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
button[kind="header"],
.stDeployButton {{
    display: none !important;
    visibility: hidden !important;
}}

/* ── STICKY NATIVE HEADER BAR ── */
div[data-testid="stHorizontalBlock"]:first-of-type {{
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
    background: {T['bg']}CC !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    padding: 8px 16px !important;
    margin-top: -20px !important;
    border-bottom: 1px solid {T['border']}33 !important;
    align-items: center !important;
}}

.sidebar-toggle-btn-custom {{
    background: transparent !important;
    border: none !important;
    font-size: 1.35rem !important;
    color: {T['text']} !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 36px !important;
    height: 36px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    padding: 0 !important;
}}
.sidebar-toggle-btn-custom:hover {{
    background: {T['surface2']} !important;
    color: {T['accent']} !important;
}}

div.st-key-theme_toggle_btn button,
div[data-testid="stHorizontalBlock"] button.theme-toggle-btn-custom,
.theme-toggle-btn-custom {{
    background: {T['surface']} !important;
    background-color: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.05rem !important;
    color: {T['text']} !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 8px {T['shadow']} !important;
    padding: 0 !important;
}}
div.st-key-theme_toggle_btn button:hover,
div[data-testid="stHorizontalBlock"] button.theme-toggle-btn-custom:hover,
.theme-toggle-btn-custom:hover {{
    border-color: {T['accent']} !important;
    background: {T['surface2']} !important;
    transform: scale(1.08) !important;
}}

/* Note: Removed .block-container width overrides that break Streamlit's JS column calculations,
   but we CAN safely override vertical padding to prevent the page from scrolling. */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {T['scrollbar']}; border-radius: 99px; }}

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Inter', sans-serif !important;
    color: {T['text']} !important;
    letter-spacing: -0.025em;
    line-height: 1.2;
}}

/* ── HEADER BAR ── */
.omni-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0px 10px 0px;
    position: sticky;
    top: 0;
    z-index: 1000;
    background: {T['bg']};
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-top: -20px;
}}

.sidebar-toggle-btn {{
    background: transparent;
    border: none;
    font-size: 1.35rem;
    color: {T['text']};
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    transition: all 0.2s ease;
    margin-right: 2px;
}}
.sidebar-toggle-btn:hover {{
    background: {T['surface2']};
    color: {T['accent']};
}}

.theme-toggle-btn {{
    margin-left: auto;
    background: transparent;
    border: 1px solid {T['border']};
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    color: {T['text']};
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px {T['shadow']};
}}
.theme-toggle-btn:hover {{
    border-color: {T['accent']};
    background: {T['surface2']};
    transform: scale(1.08);
}}

.omni--icon {{
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent !important;
    box-shadow: none !important;
    flex-shrink: 0;
    position: relative;
}}

.omni--icon svg {{
    width: 24px;
    height: 24px;
}}

.omni-wordmark {{
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: 0.18em;
    background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}}

.omni-tagline {{
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: {T['text_subtle']};
    text-transform: uppercase;
    margin-left: 4px;
    margin-top: 2px;
    align-self: flex-end;
}}

.omni-header-divider {{
    height: 1px;
    background: linear-gradient(90deg, {T['accent']}33 0%, {T['border']} 40%, transparent 100%);
    margin: 0 0px 10px 0px;
}}

/* ── SIDEBAR SYSTEM ── */
section[data-testid="stSidebar"] {{
    background-color: {T['surface']} !important;
    border-right: 1px solid {T['border']} !important;
    box-shadow: 8px 0 30px {T['shadow']} !important;
    z-index: 9999 !important;
}}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {{
    color: {T['text']} !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    padding: 24px 16px !important;
}}

.sidebar-history-scroll {{
    flex: 1;
    overflow-y: auto;
    margin-top: 10px;
    padding-right: 2px;
}}

.sidebar-section-title {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {T['text_subtle']};
    margin-bottom: 10px;
    padding-left: 4px;
}}

.sidebar-history-item button {{
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: {T['text_muted']} !important;
    width: 100% !important;
    padding: 8px 12px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease !important;
}}
.sidebar-history-item button:hover {{
    background: {T['surface2']} !important;
    color: {T['text']} !important;
    border-color: {T['border']} !important;
}}
.sidebar-history-item-active button {{
    background: {T['accent_glow']} !important;
    color: {T['accent']} !important;
    border-color: {T['accent']}33 !important;
    font-weight: 600 !important;
}}

.sidebar-footer {{
    border-top: 1px solid {T['border']};
    padding-top: 15px;
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}}

/* ── INITIAL UPLOAD STATE ── */
.omni-hero-title {{
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.035em;
    color: {T['text']};
    margin-bottom: 10px;
    line-height: 1.15;
}}

.omni-hero-title span {{
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.omni-hero-subtitle {{
    font-size: 0.92rem;
    color: {T['text_muted']};
    text-align: center;
    margin-bottom: 24px;
    font-weight: 400;
    letter-spacing: 0.01em;
}}

/* ── FILE UPLOADER NATIVE OVERRIDE ── */
div[data-testid="stFileUploader"] {{
    width: 100% !important;
    max-width: 520px !important;
    margin: 0 auto !important;
    background-color: transparent !important;
}}

/* Target the dropzone container section */
div[data-testid="stFileUploader"] section,
[data-testid="stFileUploadDropzone"] {{
    background: {T['surface']} !important;
    border: 2px dashed {T['border_accent']} !important;
    border-radius: 20px !important;
    padding: 28px 20px !important;
    text-align: center !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 8px 40px {T['shadow']}, inset 0 1px 0 rgba(255,255,255,0.04) !important;
    min-height: 130px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    color: {T['text']} !important;
}}

div[data-testid="stFileUploader"] section *,
[data-testid="stFileUploadDropzone"] * {{
    color: {T['text']} !important;
}}

div[data-testid="stFileUploader"] section:hover,
[data-testid="stFileUploadDropzone"]:hover {{
    border-color: {T['accent']} !important;
    background: {T['surface2']} !important;
    transform: translateY(-2px);
}}

/* Hide native default icon and all children of instructions to clear default text */
div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneIcon"],
div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] > *,
[data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneIcon"],
[data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] > * {{
    display: none !important;
}}

/* Premium style override for the native upload button */
div[data-testid="stFileUploader"] section button,
[data-testid="stFileUploadDropzone"] button {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: {T['surface2']} !important;
    color: {T['text']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 12px !important;
    box-shadow: 0 2px 8px {T['shadow']} !important;
}}

div[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploadDropzone"] button:hover {{
    background: {T['accent_glow']} !important;
    border-color: {T['accent']}66 !important;
    color: {T['accent']} !important;
    transform: translateY(-1px) !important;
}}

/* Color-correct the upload button's svg icon */
div[data-testid="stFileUploader"] section button svg,
[data-testid="stFileUploadDropzone"] button svg {{
    fill: currentColor !important;
    stroke: currentColor !important;
    color: inherit !important;
    margin-right: 8px !important;
    width: 14px !important;
    height: 14px !important;
}}

/* Muted override for the native file size hint small text */
div[data-testid="stFileUploader"] section small,
[data-testid="stFileUploadDropzone"] small {{
    display: block !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: {T['text_subtle']} !important;
    margin-top: 8px !important;
}}



/* Custom premium text rendering on the instructions wrapper */
div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    margin-top: 5px !important;
}}

div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"]::before,
[data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::before {{
    content: 'Initialize Intelligence Node' !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: {T['text']} !important;
    letter-spacing: -0.02em !important;
    line-height: 1.3 !important;
    display: block !important;
}}

div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"]::after,
[data-testid="stFileUploadDropzone"] [data-testid="stFileUploaderDropzoneInstructions"]::after {{
    content: 'Drag & drop your CV PDF here or click to browse' !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: {T['text_muted']} !important;
    line-height: 1.3 !important;
    display: block !important;
}}


/* Style the native uploaded file card when a file is selected */
[data-testid="stUploadedFile"],
[data-testid="stUploadedFile"] > div,
[data-testid="stUploadedFile"] div {{
    background-color: {T['surface2']} !important;
    background: {T['surface2']} !important;
    color: {T['text']} !important;
}}

[data-testid="stUploadedFile"] {{
    border: 1px solid {T['accent']} !important;
    border-radius: 12px !important;
    padding: 10px 15px !important;
    box-shadow: 0 8px 30px rgba(52,211,153,0.1) !important;
}}

[data-testid="stUploadedFile"] * {{
    color: {T['text']} !important;
}}

[data-testid="stUploadedFile"] svg {{
    fill: {T['text']} !important;
    color: {T['text']} !important;
}}

/* Style the file card delete button */
[data-testid="stUploadedFile"] button {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    color: {T['text']} !important;
}}

[data-testid="stUploadedFile"] button:hover {{
    background-color: {T['border']} !important;
    color: {T['danger']} !important;
}}

/* ── PROCEED BUTTON ── */
button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {{
    width: 100% !important;
    background: linear-gradient(135deg, {T['accent']} 0%, {T['accent2']} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px {T['shadow_accent']}, 0 2px 8px {T['shadow']} !important;
    position: relative !important;
    overflow: hidden !important;
}}
button[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px {T['shadow_accent']}, 0 4px 12px {T['shadow']} !important;
    filter: brightness(1.08) !important;
}}
button[data-testid="baseButton-primary"]:active,
.stButton > button[kind="primary"]:active {{
    transform: translateY(0px) !important;
}}

/* ── SECONDARY BUTTONS ── */
button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"] {{
    background: {T['surface2']} !important;
    color: {T['text']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}}
button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {{
    border-color: {T['accent']}66 !important;
    background: {T['accent_glow']} !important;
    color: {T['accent']} !important;
}}

/* ── ACTIVE STATE LAYOUT ── */
.omni-workspace-grid {{
    display: flex;
    height: calc(100vh - 90px);
    padding: 0;
    gap: 24px;
}}

/* ── CHAT AREA ── */
.omni-chat-area {{
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    padding: 24px 0;
}}

.chat-workspace-header {{
    margin-bottom: 20px;
    flex-shrink: 0;
}}

.chat-workspace-title {{
    font-size: 1.3rem;
    font-weight: 800;
    color: {T['text']};
    letter-spacing: -0.025em;
    margin-bottom: 6px;
}}

.active-file-pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px 4px 8px;
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    color: {T['text_muted']};
    letter-spacing: 0.01em;
}}

.active-file-dot {{
    width: 7px;
    height: 7px;
    background: {T['success']};
    border-radius: 50%;
    box-shadow: 0 0 6px {T['success']};
    animation: pulse-dot 2s infinite;
    flex-shrink: 0;
}}

@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.6; transform: scale(0.85); }}
}}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border-bottom: 1px solid {T['border']}66 !important;
    padding: 16px 4px !important;
    transition: background 0.15s ease;
}}
[data-testid="stChatMessage"]:hover {{
    background: {T['accent_glow']} !important;
    border-radius: 12px;
    border-bottom-color: transparent !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {{
    color: {T['text']} !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {{
    background: {T['surface2']} !important;
    color: {T['accent']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {{
    color: {T['accent']} !important;
    text-decoration: underline !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {{
    border-left: 3px solid {T['border_accent']} !important;
    padding-left: 10px !important;
    margin: 10px 0 !important;
    color: {T['text_muted']} !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {{
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 10px 0 !important;
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {{
    border: 1px solid {T['border']} !important;
    padding: 8px 12px !important;
    text-align: left !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {{
    background: {T['surface2']} !important;
    font-weight: 700 !important;
}}
[data-testid="stChatMessage"] p {{
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
}}
[data-testid="stChatMessageAvatarUser"] {{
    background: linear-gradient(135deg, {T['accent']}, {T['accent2']}) !important;
    border-radius: 9px !important;
}}
[data-testid="stChatMessageAvatarAssistant"] {{
    background: {T['surface2']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 9px !important;
}}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border_accent']} !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px {T['shadow']}, 0 0 0 0px {T['accent']} !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    margin-top: 12px;
}}
[data-testid="stChatInput"] div {{
    background-color: transparent !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: {T['accent']} !important;
    box-shadow: 0 4px 20px {T['shadow']}, 0 0 0 3px {T['accent_glow']} !important;
}}
[data-testid="stChatInput"] textarea {{
    color: {T['text']} !important;
    background-color: transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {T['text_subtle']} !important;
}}

/* ── RIGHT PANEL ── */
.omni-side-panel {{
    width: 260px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 24px 0;
    overflow-y: auto;
}}

/* ── PANEL CARDS ── */
.side-card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 14px;
    overflow: hidden;
    transition: border-color 0.2s ease;
}}
.side-card:hover {{
    border-color: {T['border_accent']};
}}

.side-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 15px;
    cursor: pointer;
    user-select: none;
}}

.side-card-label {{
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {T['text_subtle']};
}}

.side-card-body {{
    padding: 0 15px 15px;
    border-top: 1px solid {T['border']};
}}

/* ── 3-DOTS EXPANDER OVERRIDE ── */
div[data-testid="stExpander"] {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}}
div[data-testid="stExpander"] > details > summary {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: {T['text']} !important;
    padding: 12px 14px !important;
    border-radius: 14px !important;
}}
div[data-testid="stExpander"] > details > summary:hover {{
    background: {T['accent_glow']} !important;
}}
div[data-testid="stExpander"] > details[open] > summary {{
    border-radius: 14px 14px 0 0 !important;
    border-bottom: 1px solid {T['border']} !important;
}}
div[data-testid="stExpander"] .stRadio label {{
    font-size: 0.82rem !important;
    color: {T['text']} !important;
    font-weight: 500 !important;
    padding: 4px 0 !important;
}}

/* ── ACTIVE FILE CARD ── */
.active-file-card {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-left: 3px solid {T['accent']};
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}}
.active-file-card .afc-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {T['text_subtle']};
    margin-bottom: 6px;
}}
.active-file-card .afc-name {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {T['text']};
    word-break: break-all;
    margin-bottom: 8px;
    line-height: 1.4;
}}
.afc-status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.badge-processing {{ background: rgba(251,191,36,0.12); color: {T['warning']}; border: 1px solid {T['warning']}44; }}
.badge-completed  {{ background: rgba(52,211,153,0.12); color: {T['success']}; border: 1px solid {T['success']}44; }}
.badge-failed     {{ background: rgba(248,113,113,0.12); color: {T['danger']};  border: 1px solid {T['danger']}44; }}
.badge-pending    {{ background: {T['accent_glow']}; color: {T['accent']}; border: 1px solid {T['accent']}44; }}

/* ── HEALTH STATS CARD ── */
.health-card {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 14px;
    padding: 15px;
}}
.health-card-title {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {T['text_subtle']};
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.health-row {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 9px;
    font-size: 0.78rem;
}}
.health-row:last-child {{ margin-bottom: 0; }}
.health-key {{
    font-weight: 600;
    color: {T['text_muted']};
    min-width: 90px;
    flex-shrink: 0;
}}
.health-val {{
    font-weight: 500;
    color: {T['text']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}}
.health-online {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: {T['success']};
    font-weight: 600;
}}
.health-dot {{
    width: 6px; height: 6px;
    background: {T['success']};
    border-radius: 50%;
    box-shadow: 0 0 5px {T['success']};
    animation: pulse-dot 2s infinite;
}}

/* ── PROCESSING / STATUS CARDS ── */
.status-hero {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    text-align: center;
    padding: 60px 40px;
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-radius: 20px;
    box-shadow: 0 8px 40px {T['shadow']};
    position: relative;
    overflow: hidden;
}}

.status-hero::before {{
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, {T['accent_glow']} 0%, transparent 65%);
    pointer-events: none;
}}

.status-icon {{
    font-size: 3.5rem;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 12px {T['accent']});
}}

.status-title {{
    font-size: 1.3rem;
    font-weight: 800;
    color: {T['text']};
    margin-bottom: 10px;
    letter-spacing: -0.025em;
}}

.status-sub {{
    font-size: 0.85rem;
    color: {T['text_muted']};
    line-height: 1.6;
    max-width: 380px;
}}
.status-sub b,
.status-sub strong {{
    color: {T['text']} !important;
}}

.stage-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 24px;
    padding: 7px 18px;
    background: {T['accent_glow']};
    border: 1px solid {T['accent']}55;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 700;
    color: {T['accent']};
    letter-spacing: 0.05em;
    text-transform: uppercase;
    animation: glow-pulse 2s ease-in-out infinite;
}}

@keyframes glow-pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 {T['accent_glow']}; }}
    50% {{ box-shadow: 0 0 16px 4px {T['accent_glow']}; }}
}}

/* ── PANEL SECTION LABELS ── */
.panel-section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {T['text_subtle']};
    margin-bottom: 8px;
    padding-left: 2px;
}}

/* ── NO FILES PLACEHOLDER ── */
.no-files-placeholder {{
    font-size: 0.8rem;
    color: {T['text_subtle']};
    text-align: center;
    padding: 18px 10px;
    border: 1px dashed {T['border']};
    border-radius: 10px;
    background: {T['surface2']};
}}

/* ── SPINNER ── */
.stSpinner > div {{ color: {T['accent']} !important; }}

/* ── SUCCESS / WARNING / ERROR BANNERS ── */
[data-testid="stAlert"] {{
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    border: 1px solid !important;
}}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. POLL DOCUMENT STATUS (before render)
# ─────────────────────────────────────────────────────────────────────────────
active_doc_status = None
if st.session_state.active_doc_id is not None:
    active_doc_status = get_document_status(st.session_state.active_doc_id)
    if active_doc_status == "completed":
        st.session_state.pdf_processed = True
    elif active_doc_status == "failed":
        st.session_state.pdf_processed = False

# ─────────────────────────────────────────────────────────────────────────────
# 7. TOP BRANDING BAR  (always visible)
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_brand, col_spacer, col_home, col_theme = st.columns([0.05, 0.45, 2.0, 0.3, 0.2])

with col_logo:
    st.markdown(f"""
        <div class="omni-logo-icon" style="margin-top: 5px;">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="width:28px; height:28px;">
                <path d="M12 2L20.196 7V17L12 22L3.804 17V7L12 2Z"
                      fill="{T['accent']}" fill-opacity="0.95"/>
                <path d="M12 6L16.9 9V15L12 18L7.1 15V9L12 6Z"
                      fill="{T['accent2']}" fill-opacity="0.4"/>
                <circle cx="12" cy="12" r="2" fill="{T['text']}" fill-opacity="0.95"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)

with col_brand:
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; margin-top: 7px;">
            <div class="omni-wordmark" style="font-size: 1.45rem;">OMNI</div>
            <div class="omni-tagline" style="margin-left: 2px; font-size: 0.68rem; margin-top: 4px;">Enterprise AI Engine</div>
        </div>
    """, unsafe_allow_html=True)

with col_home:
    if st.session_state.pipeline_proceeded:
        st.markdown(f"""
            <style>
                div.st-key-home_btn button {{
                    background-color: transparent !important;
                    border: 1px solid {T['border']} !important;
                    color: {T['text']} !important;
                    font-size: 0.85rem !important;
                    font-weight: 600 !important;
                    height: 38px !important;
                    border-radius: 10px !important;
                    padding: 0 16px !important;
                    margin-left: auto !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    transition: all 0.25s ease !important;
                    box-shadow: 0 2px 8px {T['shadow']} !important;
                }}
                div.st-key-home_btn button:hover {{
                    border-color: {T['accent']} !important;
                    background-color: {T['surface2']} !important;
                    color: {T['accent']} !important;
                    transform: scale(1.03) !important;
                }}
            </style>
        """, unsafe_allow_html=True)
        if st.button("Home", key="home_btn"):
            clear_backend_documents()
            for k in ["pdf_uploaded", "pipeline_proceeded", "pdf_processed", 
                      "active_doc_id", "active_doc_name", "messages"]:
                st.session_state[k] = False if isinstance(st.session_state[k], bool) else (
                    [] if isinstance(st.session_state[k], list) else None
                )
            if "omni_portal_uploader" in st.session_state:
                del st.session_state["omni_portal_uploader"]
            st.rerun()

with col_theme:
    theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    st.markdown("""
        <style>
            div.st-key-theme_toggle_btn button {
                padding: 0 !important;
                height: 38px !important;
                width: 38px !important;
                min-width: 38px !important;
                border-radius: 50% !important;
                font-size: 1.15rem !important;
                margin-left: auto !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
        </style>
    """, unsafe_allow_html=True)
    if st.button(theme_icon, key="theme_toggle_btn"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.session_state.theme_radio = "Dark Mode" if st.session_state.theme == "dark" else "Light Mode"
        st.rerun()

st.markdown('<div class="omni-header-divider" style="margin-top: 10px;"></div>', unsafe_allow_html=True)



# Custom main workspace rendering function
def render_workspace_content(container, active_doc_status):
    with container:
        # ════════════════════════════════════════════════════════════════════
        #  INITIAL STATE  — file upload portal
        # ════════════════════════════════════════════════════════════════════
        if not st.session_state.pipeline_proceeded:
            # Vertical spacer to center vertically (reduced to fit entirely in page view)
            st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)

            # ── Centre column ──
            # Adjusting to perfectly symmetrical layout.
            _, center, _ = st.columns([1, 1.5, 1])
            with center:
                st.markdown("""
                <h1 class="omni-hero-title">
                    Enterprise <span>Semantic</span><br>Intelligence System
                </h1>
                <p class="omni-hero-subtitle">
                    Automated parsing and analysis engine for CV portfolios.
                </p>
                """, unsafe_allow_html=True)

                # NATIVE Streamlit uploader (Restyled purely via CSS to look like the big portal)
                uploaded_file = st.file_uploader(
                    label="",
                    type=["pdf"],
                    key="omni_portal_uploader",
                    label_visibility="collapsed",
                )

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                if st.button("⬡  Proceed", use_container_width=True, type="primary", key="proceed_btn"):
                    if uploaded_file:
                        with st.spinner("Connecting upload stream to data pipeline…"):
                            result = upload_file(uploaded_file)
                        if "id" in result:
                            st.session_state.active_doc_id   = result["id"]
                            st.session_state.active_doc_name = uploaded_file.name
                            st.session_state.pdf_uploaded    = True
                            st.session_state.pipeline_proceeded = True
                            st.rerun()
                        else:
                            st.error(f"Vector parsing failed: {result.get('detail', 'Unknown error')}")
                    else:
                        st.warning("⚠️  Please select a PDF before proceeding.")

        # ════════════════════════════════════════════════════════════════════
        #  ACTIVE STATE  — chat workspace
        # ════════════════════════════════════════════════════════════════════
        else:
            # ── Sub-state A: Processing ────────────────────────────────
            if not st.session_state.pdf_processed and active_doc_status != "failed":
                stage_label = (active_doc_status or "pending").upper()
                st.markdown(f"""
                <div style="display:flex;flex-direction:column;min-height:calc(100vh - 110px);
                            align-items:center;justify-content:center;padding:40px 0;">
                    <div class="status-hero">
                        <div class="status-icon">⚡</div>
                        <div class="status-title">Vectorizing Neural Memory</div>
                        <div class="status-sub">
                            Chunking and embedding <b>{st.session_state.active_doc_name}</b>
                            into high-dimensional Qdrant space. This takes a few seconds.
                        </div>
                        <div class="stage-badge">⬡ &nbsp; Stage: {stage_label}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(2)
                st.rerun()

            # ── Sub-state B: Failed ────────────────────────────────────
            elif active_doc_status == "failed":
                st.markdown(f"""
                <div style="display:flex;flex-direction:column;min-height:calc(100vh - 110px);
                            align-items:center;justify-content:center;padding:40px 0;">
                    <div class="status-hero" style="border-color:#F87171 !important;">
                        <div class="status-icon" style="filter:drop-shadow(0 0 12px #F87171);">✕</div>
                        <div class="status-title" style="color:#F87171;">Pipeline Vectorization Failed</div>
                        <div class="status-sub">
                            An error occurred while building context memory for
                            <b>{st.session_state.active_doc_name}</b>.
                            Please retry with a valid PDF.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("↩  Reset & Retry", type="secondary", use_container_width=True):
                    for k in ["pdf_uploaded","pipeline_proceeded","pdf_processed",
                              "active_doc_id","active_doc_name","messages"]:
                        st.session_state[k] = False if isinstance(st.session_state[k], bool) else (
                            [] if isinstance(st.session_state[k], list) else None
                        )
                    st.rerun()

            # ── Sub-state C: Active Chat ───────────────────────────────
            else:
                # Header
                st.markdown(f"""
                <div class="chat-workspace-header" style="padding-top:20px;">
                    <div class="chat-workspace-title">Cognitive Analysis Workspace</div>
                    <div class="active-file-pill">
                        <div class="active-file-dot"></div>
                        {st.session_state.active_doc_name}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Chat message history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                # Chat input
                if user_query := st.chat_input("Query the active memory…", key="chat_input"):
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(user_query)
                        with st.chat_message("assistant"):
                            try:
                                full_response = st.write_stream(query_llm_stream(user_query))
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": full_response}
                                )
                            except Exception as e:
                                err = f"Execution error: {e}"
                                st.error(err)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": err}
                                )

# ─────────────────────────────────────────────────────────────────────────────
# 8. LAYOUT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
render_workspace_content(st.container(), active_doc_status)
# ─────────────────────────────────────────────────────────────────────────────
# 9. BACKGROUND POLLING GUARD  (only while ingesting)
# ─────────────────────────────────────────────────────────────────────────────
if (st.session_state.active_doc_id is not None
        and not st.session_state.pdf_processed
        and active_doc_status not in ("failed", "completed")):
    time.sleep(2)
    st.rerun()
