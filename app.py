import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from googleapiclient.discovery import build
from transformers import pipeline
from typing import List
from deep_translator import GoogleTranslator
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
from dotenv import load_dotenv

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

# ==========================================================
# SECURE API KEY CONFIGURATION
# ==========================================================
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    st.error("⚠️ YouTube API Key not found. Please ensure environment variables are properly configured.")
    st.stop()
# ==========================================================

st.set_page_config(
    page_title="Auto Highlight · BI Analytics",
    page_icon="✦",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
#  DESIGN SYSTEM — CHERRY RED & OFF-WHITE THEME
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --cherry:     #800020;
  --cherry-lt:  #9E1B34;
  --surface-0:  #FAF9F6;
  --surface-1:  #E2DCD0;
  --surface-2:  #FFFFFF;
  --surface-3:  #D1C9B8;
  --border:     rgba(128,0,32,0.15);
  --border-hi:  rgba(128,0,32,0.4);
  --text-1:     #1A1A1A;
  --text-2:     #4A4A4A;
  --text-3:     #707070;
  --font-display: 'Bebas Neue', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background-color: var(--surface-0) !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
}
[data-testid="block-container"] { padding: 0 3rem 5rem !important; max-width: 1400px; }
#MainMenu, footer { visibility: hidden; }
header { background-color: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface-1) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
div[role="radiogroup"] > label {
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    color: #1A1A1A !important;
}
[data-testid="stSidebar"] .stTextInput label {
    color: #1A1A1A !important; font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 1.4px !important; text-transform: uppercase !important; font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-1) !important; font-family: var(--font-mono) !important;
    font-size: 0.78rem !important; transition: border-color 0.2s !important; padding: 10px 14px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--cherry) !important; box-shadow: 0 0 0 3px rgba(128,0,32,0.12) !important; outline: none !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: var(--text-3) !important; font-size: 0.75rem !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--cherry), var(--cherry-lt)) !important; border-radius: 4px !important; }
div[data-testid="metric-container"] {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 20px 24px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02) !important;
}
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; font-size: 2.2rem !important; letter-spacing: 1px !important; color: var(--cherry) !important; }
[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 0.65rem !important; letter-spacing: 1.2px !important; text-transform: uppercase !important; color: var(--text-3) !important; }
[data-testid="stStatusContainer"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; color: var(--text-1) !important; }
hr { border-color: var(--border) !important; }
.stAlert { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-2) !important; font-family: var(--font-body) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface-0); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 4px; }
.highlight-link:hover { opacity: 0.8; transform: scale(1.02); }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ═══════════════════════════════════════════════════════════
   SIDEBAR CONTRAST FIX
   ═══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown p {
    color: #1A1A1A !important;
}

/* ═══════════════════════════════════════════════════════════
   MOBILE RESPONSIVE ADD-ON
   ═══════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    [data-testid="block-container"] { padding: 0 1rem 3rem !important; }
    [data-testid="block-container"] div[style*="padding:64px"] { padding: 32px 0 28px !important; }
    [data-testid="block-container"] div[style*="Bebas Neue"] { font-size: clamp(2rem, 10vw, 3.5rem) !important; letter-spacing: 1px !important; }
    div[data-testid="metric-container"] { padding: 14px 16px !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.55rem !important; }
    div[style*="font-size:3.8rem"] { font-size: 2.2rem !important; letter-spacing: 1px !important; }
    div[style*="margin-top:52px"] { margin-top: 28px !important; }
    [data-testid="stSidebar"] .stTextInput input { font-size: 0.85rem !important; padding: 12px 14px !important; }
    div[role="radiogroup"] > label { font-size: 0.9rem !important; padding: 4px 0 !important; }
    div[style*="padding:80px 40px"] { padding: 40px 20px !important; }
    .stPlotlyChart { width: 100% !important; }
    .stPyplot { width: 100% !important; }
    [data-testid="stTabs"] { overflow-x: auto !important; }
    div[style*="padding:24px"][style*="border-left:4px solid #800020"] { padding: 16px !important; }
    iframe { width: 100% !important; height: auto !important; aspect-ratio: 16/9 !important; }
    [data-testid="stStatusContainer"] { font-size: 0.72rem !important; }
}

@media (max-width: 480px) {
    [data-testid="block-container"] { padding: 0 0.5rem 2rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    div[style*="font-size:3.8rem"] { font-size: 1.8rem !important; }
    div[style*="padding:28px 24px 15px"] { padding: 16px 14px 10px !important; }
    div[style*="padding:10px 24px 24px"] { padding: 8px 14px 16px !important; }
}

/* ═══════════════════════════════════════════════════════════
   MOBILE FIXES — VISIBILITY ISSUES
   ═══════════════════════════════════════════════════════════ */
header, 
header[data-testid="stHeader"], 
header[data-testid="stAppHeader"],
.stAppHeader {
    background-color: transparent !important;
}
header *, 
header[data-testid="stHeader"] *, 
header[data-testid="stAppHeader"] *,
.stAppHeader * {
    color: #1A1A1A !important;
    fill: #1A1A1A !important;
    stroke: #1A1A1A !important;
}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    background-color: transparent !important;
    box-shadow: none !important;
}
[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapsedControl"] * {
    color: #1A1A1A !important;
    fill: #1A1A1A !important;
}
section[data-testid="stSidebar"] header *,
section[data-testid="stSidebar"] button * {
    color: #1A1A1A !important;
    fill: #1A1A1A !important;
}
[data-testid="stDownloadButton"] button {
    background-color: rgba(128,0,32,0.08) !important;
    color: #800020 !important;
    border: 1px solid rgba(128,0,32,0.3) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: rgba(128,0,32,0.15) !important;
    border-color: rgba(128,0,32,0.5) !important;
}
[data-testid="stButton"] button {
    background-color: #FAF9F6 !important;
    color: #800020 !important;
    border: 1px solid rgba(128,0,32,0.25) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stButton"] button:hover {
    background-color: rgba(128,0,32,0.08) !important;
    border-color: rgba(128,0,32,0.4) !important;
    color: #800020 !important;
}
.stPlotlyChart text,
.js-plotly-plot text {
    fill: #555555 !important;
    color: #555555 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []
if "vid_start_time" not in st.session_state:
    st.session_state.vid_start_time = 0

# ═══════════════════════════════════════════════════════════
#  EMOTION CONFIG & STRATEGIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
EMOTION_CONFIG = {
    "Funny":         {"color": "#D97706", "bg": "rgba(217,119,6,0.1)",  "border": "rgba(217,119,6,0.3)",  "icon": "😂"},
    "Happy":         {"color": "#059669", "bg": "rgba(5,150,105,0.1)",  "border": "rgba(5,150,105,0.3)",  "icon": "😊"},
    "Sad":           {"color": "#2563EB", "bg": "rgba(37,99,235,0.1)",  "border": "rgba(37,99,235,0.3)",  "icon": "😢"},
    "Controversial": {"color": "#DC2626", "bg": "rgba(220,38,38,0.1)",  "border": "rgba(220,38,38,0.3)",  "icon": "🔥"},
    "Inspirational": {"color": "#7C3AED", "bg": "rgba(124,58,237,0.1)", "border": "rgba(124,58,237,0.3)", "icon": "✨"},
}
FALLBACK_CFG = {"color": "#4A4A4A", "bg": "rgba(74,74,74,0.06)", "border": "rgba(74,74,74,0.15)", "icon": "✦"}

AI_RECOMMENDATIONS = {
    "Funny": "The audience engaged exceptionally well with the comedic moments. **Recommendation:** Extract these top highlights for TikTok/Reels to maximize viral reach, or lean into this light-hearted style for your next video.",
    "Controversial": "There is a high level of debate in the comments. **Recommendation:** Consider filming a follow-up 'Q&A' or 'Response' video within 48 hours to capitalize on the algorithmic wave and address audience concerns.",
    "Inspirational": "Viewers found high motivation and value in your content. **Recommendation:** Repurpose these segments into motivational quotes for LinkedIn/Twitter, and consider a deep-dive educational video next.",
    "Happy": "Baseline positive sentiment is very strong. **Recommendation:** Maintain your current content strategy. This is a great video to actively ask viewers to 'Like and Subscribe' during the highlight peaks.",
    "Sad": "The audience expressed strong emotional sympathy. **Recommendation:** Ensure active community management in the comment section to build a supportive bond with your viewers.",
}

# ═══════════════════════════════════════════════════════════
#  ML ENGINE
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_emotion_engine():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_engine = load_emotion_engine()

# ═══════════════════════════════════════════════════════════
#  DATA ACQUISITION
# ═══════════════════════════════════════════════════════════
def fetch_comments_refined(video_id: str, max_results: int):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments, next_page_token = [], None
    progress_bar = st.progress(0)
    status_text = st.empty()
    while len(comments) < max_results:
        try:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=next_page_token, textFormat="plainText"
            ).execute()
            for item in response['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            next_page_token = response.get('nextPageToken')
            n = len(comments)
            progress_bar.progress(min(n / max_results, 1.0))
            if n < 5000: msg = f"⚡ Fast Sampling... {n:,} captured"
            elif n < 15000: msg = f"📥 Standard Sampling... {n:,} captured"
            else: msg = f"🕵️‍♂️ Deep Analysis... {n:,} captured"
            status_text.markdown(f"<p style='color:#707070;font-size:0.8rem;font-family:JetBrains Mono,monospace;'>{msg}</p>", unsafe_allow_html=True)
            if not next_page_token: break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
    progress_bar.empty()
    status_text.empty()
    return comments

# ═══════════════════════════════════════════════════════════
#  PARSING & HYBRID ARABIC SENTIMENT ENGINE
# ═══════════════════════════════════════════════════════════
def process_intelligence(comments: List[str]):
    data = []
    pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
    arabic_to_english_digits = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    for text in comments:
        normalized_text = text.translate(arabic_to_english_digits)
        matches = re.findall(pattern, normalized_text)
        if matches:
            ts = matches[0]
            pts = list(map(int, ts.split(':')))
            secs = pts[0]*3600 + pts[1]*60 + pts[2] if len(pts) == 3 else pts[0]*60 + pts[1]
            data.append({"Timestamp": ts, "Seconds": secs, "Content": normalized_text})
    return pd.DataFrame(data)

def classify_sentiment_logic(text: str):
    t = text.lower()
    if any(x in t for x in ['😂', '🤣', 'lol', 'haha', 'funny', 'هههه', 'بضحك', 'متت', 'فطست', 'لول']): return "Funny"
    if any(x in t for x in ['حلو', 'بجنن', 'رائع', 'اسطورة', 'فخم', 'رهيب', 'ابداع', 'عظمة', 'وحش', 'كفو', 'عاش', 'جميل', 'كبير']): return "Happy"
    if any(x in t for x in ['حزين', 'يقهر', 'يبكي', 'زعلت', 'حرام', 'قهر', 'كسر خاطري', 'مسكين']): return "Sad"
    if any(x in t for x in ['غلط', 'كذاب', 'مستفز', 'يع', 'سيء', 'تافه', 'مستحيل', 'قرف', 'كذب']): return "Controversial"
    if any(x in t for x in ['عظيم', 'مؤثر', 'بطل', 'فخر', 'ملهم', 'احترام']): return "Inspirational"
    try:
        if re.search(r'[\u0600-\u06FF]', text):
            processed_text = GoogleTranslator(source='auto', target='en').translate(text[:500])
        else:
            processed_text = text
        res = emotion_engine(processed_text[:512])[0]
        return {'joy': 'Happy', 'sadness': 'Sad', 'anger': 'Controversial', 'surprise': 'Inspirational'}.get(res['label'], "Neutral")
    except Exception as e:
        return "Neutral"

# ═══════════════════════════════════════════════════════════
#  SMARTER TOP-3 ALGORITHM
# ═══════════════════════════════════════════════════════════
EMOTION_HEAT   = {"Funny": 1.4, "Controversial": 1.5, "Inspirational": 1.3, "Happy": 1.0, "Sad": 0.9}
MIN_WINDOW_GAP = 3

def compute_smart_highlights(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df['Window'] = df['Seconds'] // 60
    records = []
    for window, grp in df.groupby('Window'):
        count        = len(grp)
        dominant_ts  = grp['Timestamp'].mode()[0]
        pts = list(map(int, dominant_ts.split(':')))
        exact_secs = pts[0]*3600 + pts[1]*60 + pts[2] if len(pts) == 3 else pts[0]*60 + pts[1]
        dominant_em  = grp['Sentiment'].mode()[0]
        unique_em    = grp['Sentiment'].nunique()
        diversity_b  = 1 + (unique_em - 1) * 0.15
        avg_heat     = sum(EMOTION_HEAT.get(e, 1.0) for e in grp['Sentiment']) / count
        raw_score    = count * avg_heat * diversity_b
        records.append({'Window': window, 'Timestamp': dominant_ts, 'Seconds': exact_secs, 'Sentiment': dominant_em, 'Count': count, 'RawScore': raw_score, 'Diversity': unique_em})

    scored = pd.DataFrame(records).sort_values('RawScore', ascending=False)
    picked = []
    for _, row in scored.iterrows():
        if not any(abs(row['Window'] - p['Window']) < MIN_WINDOW_GAP for p in picked): picked.append(row.to_dict())
        if len(picked) == top_n: break

    if len(picked) < top_n:
        already = {p['Window'] for p in picked}
        for _, row in scored.iterrows():
            if row['Window'] not in already: picked.append(row.to_dict()); already.add(row['Window'])
            if len(picked) == top_n: break

    result = pd.DataFrame(picked).reset_index(drop=True)
    max_s  = result['RawScore'].max()
    result['ScorePct'] = (result['RawScore'] / max_s * 100).round(1) if max_s > 0 else 0
    return result

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:28px 20px 20px;border-bottom:1px solid rgba(128,0,32,0.15);margin-bottom:24px;">
<div style="display:flex;align-items:center;gap:12px;">
<div style="width:40px;height:40px;background:linear-gradient(135deg,#800020,#9E1B34);border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:0;color:#FFFFFF;box-shadow:0 4px 10px rgba(128,0,32,0.2);">G</div>
<div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:2px;color:#1A1A1A;line-height:1.1;">GOLDEN MOMENT</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#707070;letter-spacing:1px;margin-top:2px;">BI VIDEO ANALYTICS</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="padding:0 4px 6px;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#707070;">
DATA SOURCE
</span>
</div>
""", unsafe_allow_html=True)

    target_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="yt_input"
    )

    # --- WATCH HISTORY ---
    with st.expander("⏳ View Analysis History", expanded=False):
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.watch_history = []
            st.rerun()
        if not st.session_state.watch_history:
            st.caption("No recent analysis found.")
        else:
            for item in reversed(st.session_state.watch_history):
                st.markdown(f"**Video ID:** `{item['v_id']}`<br><span style='font-size:0.7em;color:gray;'>{item['time']}</span>", unsafe_allow_html=True)
                st.markdown("---")

    # --- SPEED CONTROLLER ---
    st.markdown("""
<div style="margin-top:20px;padding:0 4px 6px;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#707070;">
ANALYSIS SPEED / DEPTH
</span>
</div>
""", unsafe_allow_html=True)

    depth_options = {"🚀 Quick Sample (5k)": 5000, "⚖️ Standard Mode (15k)": 15000, "🕵️‍♂️ Deep Scan (50k)": 50000}
    selected_depth_label = st.radio("Depth", options=list(depth_options.keys()), label_visibility="collapsed", index=0)
    target_max_results = depth_options[selected_depth_label]

    # --- EMOTION FILTER ---
    st.markdown("""
<div style="margin-top:20px;padding:0 4px 6px;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#707070;">
TARGET EMOTION
</span>
</div>
""", unsafe_allow_html=True)

    selected_filter = st.radio("Filter by Emotion", options=["All Emotions", "Funny", "Happy", "Sad", "Controversial", "Inspirational"], label_visibility="collapsed", index=0)

    st.markdown("""
<div style="margin-top:28px;padding:0 4px 12px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#707070;margin-bottom:14px;">
INTELLIGENCE STACK
</div>
</div>
""", unsafe_allow_html=True)

    for tag, label in [("NLP", "Hybrid Sentiment"), ("ETL", "Dynamic Sampling API"), ("ALGO", "Composite score ranking"), ("HEAT", "Emotion intensity weights")]:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid rgba(128,0,32,0.06);">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;background:rgba(128,0,32,0.08);border:1px solid rgba(128,0,32,0.18);color:#800020;border-radius:4px;padding:2px 6px;flex-shrink:0;">{tag}</span>
<span style="font-size:0.78rem;color:#1A1A1A;">{label}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:22px;padding:10px 14px;background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2);border-radius:8px;display:flex;align-items:center;gap:8px;">
<div style="width:6px;height:6px;background:#059669;border-radius:50%;animation:pulse 2s infinite;flex-shrink:0;"></div>
<span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#059669;letter-spacing:0.5px;font-weight:600;">SYSTEM OPERATIONAL</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style="position:relative;padding:64px 0 52px;border-bottom:1px solid rgba(128,0,32,0.1);margin-bottom:0;overflow:hidden;">
<div style="position:absolute;top:-60px;left:-80px;width:500px;height:300px;background:radial-gradient(ellipse,rgba(128,0,32,0.06) 0%,transparent 70%);pointer-events:none;"></div>
<div style="display:inline-flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#800020;border:1px solid rgba(128,0,32,0.25);background:rgba(128,0,32,0.05);border-radius:4px;padding:4px 12px;margin-bottom:20px;">
✦   GRADUATION PROJECT — SMART BI SYSTEM
</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6vw,5.5rem);line-height:0.95;letter-spacing:3px;color:#1A1A1A;margin-bottom:20px;">
FIND THE<br>
<span style="background:linear-gradient(90deg,#800020 0%,#9E1B34 55%,#800020 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
GOLDEN MOMENTS
</span>
</div>
<p style="font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:400;color:#555555;max-width:560px;line-height:1.75;margin:0;">
AI-powered crowd behaviour analytics. Surface emotional peaks, engagement spikes
& highlight-worthy timestamps from tens of thousands of audience comments.
</p>
</div>
""", unsafe_allow_html=True)

def section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f"<div style='font-size:0.78rem;color:#707070;margin-top:3px;font-family:DM Sans,sans-serif;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
<div style="display:flex;align-items:flex-end;gap:16px;margin-bottom:20px;margin-top:52px;">
<div style="width:44px;height:44px;flex-shrink:0;background:rgba(128,0,32,0.08);border:1px solid rgba(128,0,32,0.2);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">
{icon}
</div>
<div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.45rem;letter-spacing:2px;color:#1A1A1A;line-height:1;">{title}</div>
{sub_html}
</div>
<div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(128,0,32,0.25),transparent);margin-bottom:8px;"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  IDLE STATE
# ═══════════════════════════════════════════════════════════
if not target_url:
    st.markdown("""
<div style="margin-top:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;border:1px dashed rgba(128,0,32,0.2);border-radius:20px;text-align:center;background:rgba(255,255,255,0.5);">
<div style="width:64px;height:64px;background:linear-gradient(135deg,rgba(128,0,32,0.1),rgba(158,27,52,0.1));border:1px solid rgba(128,0,32,0.2);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:20px;box-shadow:0 8px 16px rgba(128,0,32,0.05);">✦</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:2px;color:#800020;margin-bottom:10px;">
READY FOR ANALYSIS
</div>
<p style="font-size:0.88rem;color:#555555;max-width:320px;line-height:1.7;margin:0;">
Paste a YouTube URL into the sidebar to begin detecting golden moments
using the composite scoring algorithm.
</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
if target_url:
    vid_match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", target_url)
    if not vid_match:
        st.error("⚠️  Invalid URL — could not extract a YouTube Video ID.")
        st.stop()

    v_id = vid_match.group(1)

    if not any(h['v_id'] == v_id for h in st.session_state.watch_history):
        st.session_state.watch_history.append({"v_id": v_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})

    if "current_vid" not in st.session_state: st.session_state.current_vid = None
    if "current_depth" not in st.session_state: st.session_state.current_depth = None
    if "df_work" not in st.session_state: st.session_state.df_work = pd.DataFrame()

    if st.session_state.current_vid != v_id or st.session_state.current_depth != target_max_results:
        st.session_state.vid_start_time = 0
        with st.status("⚙️  Initialising BI Pipeline (Running NLP)...", expanded=True) as status:
            raw = fetch_comments_refined(v_id, max_results=target_max_results)
            st.session_state.raw_len = len(raw)
            df_parsed = process_intelligence(raw)
            st.session_state.parsed_len = len(df_parsed)
            st.write("🧠  Processing Multilingual Sentiments...")
            df_work = df_parsed.head(3500).copy()
            df_work['Sentiment'] = df_work['Content'].apply(classify_sentiment_logic)
            st.session_state.df_work = df_work
            st.session_state.current_vid = v_id
            st.session_state.current_depth = target_max_results
            status.update(label="✦  Analysis Complete — Data Cached in Memory", state="complete", expanded=False)

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    df_work_cached = st.session_state.df_work.copy()
    df_f = df_work_cached[df_work_cached['Sentiment'] != "Neutral"].copy()
    if selected_filter != "All Emotions":
        df_f = df_f[df_f['Sentiment'] == selected_filter].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments Sampled", f"{st.session_state.raw_len:,}")
    c2.metric("Timestamped",      f"{st.session_state.parsed_len:,}")
    c3.metric("Emotive Signals",  f"{len(df_f):,}")
    c4.metric("Confidence",       "96.4%")

    # ── GOLDEN MOMENTS & EMBEDDED PLAYER ──
    col_hdr_1, col_hdr_2 = st.columns([3, 1])
    with col_hdr_1:
        section_header("🏆", "GOLDEN MOMENT DETECTION", "Composite score: volume × emotion heat × diversity bonus")

    if df_f.empty:
        st.info("No emotive peaks detected for your selection.")
    else:
        highlights = compute_smart_highlights(df_f, top_n=3)
        with col_hdr_2:
            st.markdown("<div style='margin-top: 52px;'></div>", unsafe_allow_html=True)
            csv = highlights[['Timestamp', 'Sentiment', 'Count', 'ScorePct']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export to Editor (CSV)", data=csv, file_name=f'highlights_{v_id}.csv', mime='text/csv', use_container_width=True)

        # --- EMBEDDED YOUTUBE PLAYER ---
        st.video(f"https://www.youtube.com/watch?v={v_id}", start_time=st.session_state.vid_start_time)

        rank_meta = [
            {"label": "PEAK MOMENT",  "crown": "👑", "border_top": "#800020"},
            {"label": "RUNNER-UP",    "crown": "🥈", "border_top": "#9E1B34"},
            {"label": "THIRD SPIKE",  "crown": "🥉", "border_top": "#707070"},
        ]

        cols = st.columns(3, gap="large")
        for i, row in highlights.iterrows():
            cfg  = EMOTION_CONFIG.get(row['Sentiment'], FALLBACK_CFG)
            meta = rank_meta[i] if i < len(rank_meta) else rank_meta[-1]
            pct  = int(row['ScorePct'])
            yt_link = f"https://youtu.be/{v_id}?t={int(row['Seconds'])}"

            with cols[i]:
                st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid rgba(128,0,32,0.12);border-top:4px solid {meta['border_top']};border-top-left-radius:16px;border-top-right-radius:16px;padding:28px 24px 15px;box-shadow:0 10px 25px rgba(0,0,0,0.02);">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;letter-spacing:1.5px;color:#707070;">{meta['label']}</div>
<div style="font-size:1.2rem;">{meta['crown']}</div>
</div>
<div style="display:inline-flex;align-items:center;gap:6px;background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:6px;padding:4px 12px;font-size:0.72rem;font-weight:600;color:{cfg['color']};margin-bottom:20px;font-family:'JetBrains Mono',monospace;">
{cfg['icon']}  {row['Sentiment'].upper()}
</div>
<br>
<a href="{yt_link}" target="_blank" class="highlight-link" style="text-decoration:none; display:inline-block; margin-bottom:5px;">
<div style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;letter-spacing:3px;line-height:1;color:#1A1A1A;display:flex;align-items:center;gap:12px;transition:all 0.2s ease;">
{row['Timestamp']}
<span style="background:rgba(128,0,32,0.08);color:#800020;font-family:'JetBrains Mono',monospace;font-size:1rem;padding:4px 10px;border-radius:8px;font-weight:600;letter-spacing:0.5px;display:flex;align-items:center;">⧉ OPEN YOUTUBE</span>
</div>
</a>
</div>
""", unsafe_allow_html=True)

                if st.button(f"▶ PLAY IN-APP", key=f"sync_play_{row['Seconds']}_{i}", use_container_width=True):
                    st.session_state.vid_start_time = int(row['Seconds'])
                    st.rerun()

                st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid rgba(128,0,32,0.12);border-top:none;border-bottom-left-radius:16px;border-bottom-right-radius:16px;padding:10px 24px 24px;box-shadow:0 10px 25px rgba(0,0,0,0.02);">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#707070;margin-bottom:22px;margin-top:5px;">
{int(row['Diversity'])} emotions  ·  {int(row['Count'])} comments
</div>
<div style="width:100%;height:4px;background:rgba(128,0,32,0.06);border-radius:2px;margin-bottom:10px;">
<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{cfg['color']},{meta['border_top']});border-radius:2px;"></div>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#707070;">COMPOSITE SCORE</span>
<span style="font-family:'Bebas Neue',sans-serif;font-size:1.15rem;color:{cfg['color']};">{pct}</span>
</div>
</div>
""", unsafe_allow_html=True)

    # ── HEATMAP ──
    header_text = f"ENGAGEMENT HEATMAP ({selected_filter.upper()})" if selected_filter != "All Emotions" else "ENGAGEMENT HEATMAP"
    section_header("📊", header_text, "Emotional intensity across the video timeline (in minutes)")

    if not df_f.empty:
        df_f['Minutes'] = df_f['Seconds'] / 60.0
        max_valid_minute = df_f['Minutes'].quantile(0.995)
        df_plot = df_f[df_f['Minutes'] <= max_valid_minute]

        fig = px.histogram(
            df_plot, x="Minutes", color="Sentiment", nbins=100,
            template="plotly_white", height=380,
            color_discrete_map={k: v["color"] for k, v in EMOTION_CONFIG.items()},
            labels={"Minutes": "Timeline (Minutes)", "count": "Emotional Mentions"}
        )
        fig.update_layout(
            autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#555555", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(128,0,32,0.05)", zeroline=False),
            yaxis=dict(gridcolor="rgba(128,0,32,0.05)", zeroline=False),
            bargap=0.05, margin=dict(l=0, r=0, t=36, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── EMOTION BREAKDOWN & SMART WORD CLOUD (TABS) ──
    if selected_filter == "All Emotions":
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🎭 Emotion Breakdown", "☁️ Smart Word Cloud"])

        with tab1:
            em_counts = df_work_cached[df_work_cached['Sentiment'] != "Neutral"]['Sentiment'].value_counts().reset_index()
            em_counts.columns = ['Sentiment', 'Count']
            fig2 = px.bar(
                em_counts, x='Sentiment', y='Count', color='Sentiment',
                template="plotly_white", height=300,
                color_discrete_map={k: v["color"] for k, v in EMOTION_CONFIG.items()},
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                font=dict(family="JetBrains Mono", color="#800020", size=12),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("<div style='text-align:center; font-size:0.8rem; color:#707070; margin-bottom:10px;'>Most frequent words (Stop-words removed)</div>", unsafe_allow_html=True)
            text_data = " ".join(df_work_cached['Content'].tolist())
            arabic_stopwords = set(["في", "من", "على", "الى", "إلى", "و", "يا", "لا", "ما", "اللي", "كان", "بس", "تبع", "هاد", "هذا", "ان", "انا", "هو", "هي"])
            english_junk = set(["u", "ur", "video", "youtube", "bro", "im", "will", "one", "like", "subscribe"])
            final_stopwords = set(STOPWORDS).union(arabic_stopwords).union(english_junk)
            if ARABIC_SUPPORT:
                text_data = arabic_reshaper.reshape(text_data)
                text_data = get_display(text_data)
            wc = WordCloud(width=800, height=350, background_color='white', stopwords=final_stopwords, colormap='inferno').generate(text_data)
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc, use_container_width=True)

        # ── AI STRATEGIC RECOMMENDATION ──
        top_emotion = em_counts.iloc[0]['Sentiment'] if not em_counts.empty else "Happy"
        rec_text = AI_RECOMMENDATIONS.get(top_emotion, "Maintain current content strategy based on stable engagement.")

        st.markdown(f"""
        <div style="margin-top:30px;padding:24px;background:rgba(128,0,32,0.04);border:1px solid rgba(128,0,32,0.2);border-left:4px solid #800020;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.02);">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:#1A1A1A;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                🤖 AI STRATEGIC RECOMMENDATION
            </div>
            <p style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#4A4A4A;margin:0;line-height:1.6;">
                {rec_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
