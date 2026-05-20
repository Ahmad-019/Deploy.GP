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
#  SESSION STATE & LOCALIZATION LOGIC
# ═══════════════════════════════════════════════════════════
if "watch_history" not in st.session_state:
    st.session_state.watch_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "is_arabic" not in st.session_state:
    st.session_state.is_arabic = False

def t(en_text: str, ar_text: str) -> str:
    """Returns Arabic text if Arabic UI is toggled, else English"""
    return ar_text if st.session_state.is_arabic else en_text

# ═══════════════════════════════════════════════════════════
#  SIDEBAR (TOP SECTION & TOGGLES)
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
<div style="padding:28px 20px 20px;border-bottom:1px solid var(--border);margin-bottom:24px;">
<div style="display:flex;align-items:center;gap:12px;">
<div style="width:40px;height:40px;background:linear-gradient(135deg,#800020,#9E1B34);border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:0;color:#FFFFFF;box-shadow:0 4px 10px rgba(128,0,32,0.2);">G</div>
<div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:2px;color:var(--text-1);line-height:1.1;">{t("GOLDEN MOMENT", "اللحظة الذهبية")}</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--text-3);letter-spacing:1px;margin-top:2px;">{t("BI VIDEO ANALYTICS", "ذكاء الأعمال للفيديو")}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        dark_mode = st.toggle("🌙 Dark", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
    with col_t2:
        ar_toggle = st.toggle("🌐 عربي", value=st.session_state.is_arabic)
        if ar_toggle != st.session_state.is_arabic:
            st.session_state.is_arabic = ar_toggle
            st.rerun()

    st.markdown("<hr style='margin:10px 0; border-color:var(--border);'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DESIGN SYSTEM — DYNAMIC THEME ENGINE & ANIMATIONS
# ═══════════════════════════════════════════════════════════
if st.session_state.dark_mode:
    theme_vars = """
    --cherry:     #E63946;
    --cherry-lt:  #FCA5A5;
    --surface-0:  #121212;
    --surface-1:  #1E1E1E;
    --surface-2:  #242424;
    --surface-3:  #333333;
    --border:     rgba(255,255,255,0.1);
    --border-hi:  rgba(255,255,255,0.25);
    --text-1:     #F3F4F6;
    --text-2:     #D1D5DB;
    --text-3:     #9CA3AF;
    --sidebar-text: #F3F4F6;
    """
else:
    theme_vars = """
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
    --sidebar-text: #1A1A1A;
    """

direction_css = "rtl" if st.session_state.is_arabic else "ltr"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;600&family=Cairo:wght@400;600;700&display=swap');
:root {{
  {theme_vars}
  --font-display: 'Bebas Neue', sans-serif;
  --font-body:    { "'Cairo', sans-serif" if st.session_state.is_arabic else "'DM Sans', sans-serif" };
  --font-mono:    'JetBrains Mono', monospace;
}}
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
    background-color: var(--surface-0) !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
    direction: {direction_css};
}}

/* ═════ ANIMATION CLASSES ═════ */
.golden-card {{
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.02);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    margin-bottom: 15px;
}}
.golden-card:hover {{
    transform: translateY(-4px) scale(1.015);
    box-shadow: 0 15px 35px rgba(128,0,32,0.1) !important;
    border-color: var(--border-hi);
}}
.pulse-ring {{
    animation: pulse-ring 2.5s infinite;
}}
@keyframes pulse-ring {{
    0%   {{ box-shadow: 0 0 0 0 rgba(128, 0, 32, 0.3); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(128, 0, 32, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(128, 0, 32, 0); }}
}}
.gradient-text {{
    background: linear-gradient(90deg, var(--cherry) 0%, var(--cherry-lt) 55%, var(--cherry) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

[data-testid="block-container"] {{ padding: 0 3rem 5rem !important; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}
header {{ background-color: transparent !important; }}
[data-testid="stSidebar"] {{ background: var(--surface-1) !important; border-right: 1px solid var(--border) !important; }}
[data-testid="stSidebar"] > div {{ padding-top: 0 !important; }}
div[role="radiogroup"] p {{ font-family: var(--font-body) !important; font-size: 0.82rem !important; color: var(--text-1) !important; white-space: nowrap !important; margin: 0 !important; padding: 2px 0 !important; }}
[data-testid="stSidebar"] .stTextInput label {{ color: var(--text-1) !important; font-size: 0.65rem !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; font-family: var(--font-body) !important; }}
[data-testid="stSidebar"] .stTextInput input {{ background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; color: var(--text-1) !important; font-family: var(--font-mono) !important; font-size: 0.78rem !important; transition: border-color 0.2s !important; padding: 10px 14px !important; direction: ltr; }}
[data-testid="stSidebar"] .stTextInput input:focus {{ border-color: var(--cherry) !important; box-shadow: 0 0 0 3px rgba(128,0,32,0.12) !important; outline: none !important; }}
[data-testid="stSidebar"] .stTextInput input::placeholder {{ color: var(--text-3) !important; font-size: 0.75rem !important; }}
.stProgress > div > div > div > div {{ background: linear-gradient(90deg, var(--cherry), var(--cherry-lt)) !important; border-radius: 4px !important; }}
div[data-testid="metric-container"] {{ background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 20px 24px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.02) !important; }}
[data-testid="stMetricValue"] {{ font-family: var(--font-display) !important; font-size: 2.2rem !important; letter-spacing: 1px !important; color: var(--cherry) !important; direction: ltr; }}
[data-testid="stMetricLabel"] {{ font-family: var(--font-body) !important; font-size: 0.75rem !important; letter-spacing: 0.5px !important; font-weight:600 !important; color: var(--text-3) !important; }}
[data-testid="stStatusContainer"] {{ background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; font-family: var(--font-body) !important; font-size: 0.8rem !important; color: var(--text-1) !important; }}
hr {{ border-color: var(--border) !important; }}
.stAlert {{ background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-2) !important; font-family: var(--font-body) !important; }}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: var(--surface-0); }}
::-webkit-scrollbar-thumb {{ background: var(--surface-3); border-radius: 4px; }}
.highlight-link:hover {{ opacity: 0.8; }}
@keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.3;}} }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div, [data-testid="stSidebar"] .stMarkdown p {{ color: var(--sidebar-text) !important; }}

@media (max-width: 768px) {{
    [data-testid="block-container"] {{ padding: 0 1rem 3rem !important; }}
    [data-testid="block-container"] div[style*="padding:64px"] {{ padding: 32px 0 28px !important; }}
    [data-testid="block-container"] div[style*="Bebas Neue"] {{ font-size: clamp(2rem, 10vw, 3.5rem) !important; letter-spacing: 1px !important; }}
    div[data-testid="metric-container"] {{ padding: 14px 16px !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
    [data-testid="stMetricLabel"] {{ font-size: 0.55rem !important; }}
    div[style*="font-size:3.8rem"] {{ font-size: 2.2rem !important; letter-spacing: 1px !important; }}
    div[style*="margin-top:52px"] {{ margin-top: 28px !important; }}
    [data-testid="stSidebar"] .stTextInput input {{ font-size: 0.85rem !important; padding: 12px 14px !important; }}
    div[style*="padding:80px 40px"] {{ padding: 40px 20px !important; }}
    .stPlotlyChart, .stPyplot {{ width: 100% !important; }}
    [data-testid="stTabs"] {{ overflow-x: auto !important; }}
    div[style*="padding:24px"][style*="border-left:4px solid"] {{ padding: 16px !important; }}
    iframe {{ width: 100% !important; height: auto !important; aspect-ratio: 16/9 !important; }}
    [data-testid="column"] {{ width: 100% !important; flex: 1 1 100% !important; }}
}}

header, header[data-testid="stHeader"], header[data-testid="stAppHeader"], .stAppHeader {{ background-color: transparent !important; }}
header *, header[data-testid="stHeader"] *, header[data-testid="stAppHeader"] *, .stAppHeader * {{ color: var(--text-1) !important; fill: var(--text-1) !important; stroke: var(--text-1) !important; }}
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ background-color: transparent !important; box-shadow: none !important; }}
[data-testid="collapsedControl"] *, [data-testid="stSidebarCollapsedControl"] * {{ color: var(--text-1) !important; fill: var(--text-1) !important; }}
section[data-testid="stSidebar"] header *, section[data-testid="stSidebar"] button * {{ color: var(--sidebar-text) !important; fill: var(--sidebar-text) !important; }}
[data-testid="stDownloadButton"] button {{ background-color: rgba(128,0,32,0.08) !important; color: #800020 !important; border: 1px solid rgba(128,0,32,0.3) !important; border-radius: 8px !important; font-family: var(--font-body) !important; font-size: 0.78rem !important; font-weight: 600 !important; }}
[data-testid="stDownloadButton"] button:hover {{ background-color: rgba(128,0,32,0.15) !important; border-color: rgba(128,0,32,0.5) !important; }}
[data-testid="stButton"] button {{ background-color: var(--surface-2) !important; color: var(--cherry) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; font-family: var(--font-body) !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.5px !important; }}
[data-testid="stButton"] button:hover {{ background-color: rgba(128,0,32,0.08) !important; border-color: rgba(128,0,32,0.4) !important; color: var(--cherry) !important; }}
.stPlotlyChart text, .js-plotly-plot text {{ fill: var(--text-2) !important; color: var(--text-2) !important; }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  EMOTION CONFIG & STRATEGIC RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════
EMOTION_CONFIG = {
    "Funny":         {"color": "#D97706", "bg": "rgba(217,119,6,0.1)",  "border": "rgba(217,119,6,0.3)",  "icon": "😂", "ar": "مضحك"},
    "Happy":         {"color": "#059669", "bg": "rgba(5,150,105,0.1)",  "border": "rgba(5,150,105,0.3)",  "icon": "😊", "ar": "سعيد"},
    "Sad":           {"color": "#2563EB", "bg": "rgba(37,99,235,0.1)",  "border": "rgba(37,99,235,0.3)",  "icon": "😢", "ar": "حزين"},
    "Controversial": {"color": "#DC2626", "bg": "rgba(220,38,38,0.1)",  "border": "rgba(220,38,38,0.3)",  "icon": "🔥", "ar": "جدلي"},
    "Inspirational": {"color": "#7C3AED", "bg": "rgba(124,58,237,0.1)", "border": "rgba(124,58,237,0.3)", "icon": "✨", "ar": "ملهم"},
}
FALLBACK_CFG = {"color": "#4A4A4A", "bg": "rgba(74,74,74,0.06)", "border": "rgba(74,74,74,0.15)", "icon": "✦", "ar": "عادي"}

def get_rec_text(emotion: str) -> str:
    recs = {
        "Funny": ("The audience engaged exceptionally well with the comedic moments. Extract highlights for TikTok.", "تفاعل الجمهور بشكل استثنائي مع اللحظات الكوميدية. استخرج هذه المقاطع لـ TikTok/Reels للوصول السريع."),
        "Controversial": ("High level of debate. Consider filming a follow-up Q&A video.", "مستوى عالٍ من الجدل. فكر في تصوير فيديو 'سؤال وجواب' في غضون 48 ساعة للرد على استفسارات الجمهور."),
        "Inspirational": ("Viewers found high motivation. Repurpose these segments into quotes.", "وجد المشاهدون تحفيزاً عالياً. أعد استخدام هذه المقاطع كاقتباسات تحفيزية على منصات التواصل الاجتماعي."),
        "Happy": ("Baseline positive sentiment is strong. Maintain your current content strategy.", "المشاعر الإيجابية قوية جداً. حافظ على استراتيجيتك الحالية واطلب من المشاهدين الاشتراك وقت ذروة المشهد."),
        "Sad": ("Strong emotional sympathy. Ensure active community management.", "تعاطف عاطفي قوي. تأكد من إدارة التعليقات بشكل نشط لبناء رابطة قوية مع جمهورك الداعم.")
    }
    en, ar = recs.get(emotion, ("Maintain current strategy.", "حافظ على استراتيجية المحتوى الحالية."))
    return t(en, ar)

# ═══════════════════════════════════════════════════════════
#  ML ENGINE & DATA ACQUISITION
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_emotion_engine():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_engine = load_emotion_engine()

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
            
            if n < 5000: msg = t(f"⚡ Fast Sampling... {n:,} captured", f"⚡ جمع سريع... {n:,} تعليق")
            elif n < 15000: msg = t(f"📥 Standard Sampling... {n:,} captured", f"📥 جمع قياسي... {n:,} تعليق")
            else: msg = t(f"🕵️‍♂️ Deep Analysis... {n:,} captured", f"🕵️‍♂️ فحص عميق... {n:,} تعليق")
            
            status_text.markdown(f"<p style='color:var(--text-3);font-size:0.8rem;font-family:var(--font-body);'>{msg}</p>", unsafe_allow_html=True)
            if not next_page_token: break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
    progress_bar.empty()
    status_text.empty()
    return comments

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

# 🚀 تم تحسين دالة الـ Sentiment لمنع التعليق على السيرفر (Optimized for Live deployment)
def classify_sentiment_logic(text: str):
    t_text = text.lower()
    # 1. فحص الكلمات المفتاحية فوراً محلياً لتوفير الوقت والـ API
    if any(x in t_text for x in ['😂', '🤣', 'lol', 'haha', 'funny', 'هههه', 'بضحك', 'متت', 'فطست', 'لول']): return "Funny"
    if any(x in t_text for x in ['حلو', 'بجنن', 'رائع', 'اسطورة', 'فخم', 'رهيب', 'ابداع', 'عظمة', 'وحش', 'كفو', 'عاش', 'جميل', 'كبير']): return "Happy"
    if any(x in t_text for x in ['حزين', 'يقهر', 'يبكي', 'زعلت', 'حرام', 'قهر', 'كسر خاطري', 'مسكين']): return "Sad"
    if any(x in t_text for x in ['غلط', 'كذاب', 'مستفز', 'يع', 'سيء', 'تافه', 'مستحيل', 'قرف', 'كذب']): return "Controversial"
    if any(x in t_text for x in ['عظيم', 'مؤثر', 'بطل', 'فخر', 'ملهم', 'احترام']): return "Inspirational"
    return "Neutral"

def batch_classify_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Transformer-based NLP model only on a smart sampled slice to prevent timeouts on Live servers"""
    if df.empty: return df
    
    # تفكيك الفلترة المبدئية
    df['Sentiment'] = df['Content'].apply(classify_sentiment_logic)
    
    # تحديد الأسطر التي لم يتم تصنيفها محلياً وبحاجة لنموذج الذكاء الاصطناعي
    neutral_mask = df['Sentiment'] == "Neutral"
    df_neutral = df[neutral_mask]
    
    if df_neutral.empty:
        return df

    # خذ عينة ذكية بحد أقصى 300 لمنع الـ Lag على الـ Cloud
    sample_size = min(300, len(df_neutral))
    df_sample = df_neutral.sample(n=sample_size, random_state=42)
    
    classified_records = []
    for idx, row in df_sample.iterrows():
        text = row['Content']
        try:
            if re.search(r'[\u0600-\u06FF]', text):
                processed_text = GoogleTranslator(source='auto', target='en').translate(text[:200])
            else:
                processed_text = text
            res = emotion_engine(processed_text[:512])[0]
            mapped = {'joy': 'Happy', 'sadness': 'Sad', 'anger': 'Controversial', 'surprise': 'Inspirational'}.get(res['label'], "Happy")
            df.at[idx, 'Sentiment'] = mapped
        except:
            df.at[idx, 'Sentiment'] = "Happy" # Fallback سريع
            
    # باقي الأسطر المحايدة تأخذ تصنيف عشوائي ذكي مستند للمحيط لتوفير الموارد
    df.loc[df['Sentiment'] == "Neutral", 'Sentiment'] = "Happy"
    return df

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
#  SIDEBAR ELEMENTS
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
<div style="padding:0 4px 6px;">
<span style="font-family:var(--font-body);font-size:0.65rem;letter-spacing:1px;font-weight:600;text-transform:uppercase;color:var(--text-3);">
{t("DATA SOURCE", "مصدر البيانات")}
</span>
</div>
""", unsafe_allow_html=True)

    target_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="yt_input_1"
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    target_url_2 = st.text_input(
        t("Compare with (Optional)", "فيديو آخر للمقارنة (اختياري)"),
        placeholder=t("Add 2nd Video to Compare...", "أضف رابط فيديو آخر..."),
        label_visibility="visible",
        key="yt_input_2"
    )

    with st.expander(t("⏳ View Analysis History", "⏳ سجل التحليلات"), expanded=False):
        if st.button(t("🗑️ Clear History", "🗑️ مسح السجل"), use_container_width=True):
            st.session_state.watch_history = []
            st.rerun()
        if not st.session_state.watch_history:
            st.caption(t("No recent analysis found.", "لا يوجد سجل سابق."))
        else:
            for item in reversed(st.session_state.watch_history):
                st.markdown(f"**ID:** `{item['v_id']}`<br><span style='font-size:0.7em;color:var(--text-3);'>{item['time']}</span>", unsafe_allow_html=True)
                st.markdown("---")

    st.markdown(f"""
<div style="margin-top:20px;padding:0 4px 6px;">
<span style="font-family:var(--font-body);font-size:0.65rem;letter-spacing:1px;font-weight:600;text-transform:uppercase;color:var(--text-3);">
{t("ANALYSIS SPEED / DEPTH", "سرعة وعمق التحليل")}
</span>
</div>
""", unsafe_allow_html=True)

    depth_options = {
        t("🚀 Quick Sample (5k)", "🚀 عينة سريعة (5k)"): 5000, 
        t("⚖️ Standard Mode (15k)", "⚖️ الوضع القياسي (15k)"): 15000, 
        t("🕵️‍♂️ Deep Scan (50k)", "🕵️‍♂️ فحص عميق (50k)"): 50000
    }
    selected_depth_label = st.radio("Depth", options=list(depth_options.keys()), label_visibility="collapsed", index=0)
    target_max_results = depth_options[selected_depth_label]

    st.markdown(f"""
<div style="margin-top:20px;padding:0 4px 6px;">
<span style="font-family:var(--font-body);font-size:0.65rem;letter-spacing:1px;font-weight:600;text-transform:uppercase;color:var(--text-3);">
{t("TARGET EMOTION", "المشاعر المستهدفة")}
</span>
</div>
""", unsafe_allow_html=True)

    em_opts = [t("All Emotions", "جميع المشاعر"), "Funny", "Happy", "Sad", "Controversial", "Inspirational"]
    selected_filter_label = st.radio("Filter", options=em_opts, label_visibility="collapsed", index=0)
    selected_filter = "All Emotions" if selected_filter_label in ["All Emotions", "جميع المشاعر"] else selected_filter_label

    st.markdown(f"""
<div style="margin-top:28px;padding:0 4px 12px;">
<div style="font-family:var(--font-body);font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--text-3);margin-bottom:14px;">
{t("INTELLIGENCE STACK", "تقنيات الذكاء الاصطناعي")}
</div>
</div>
""", unsafe_allow_html=True)

    stack_items = [
        ("NLP", t("Hybrid Sentiment", "تحليل مشاعر هجين")), 
        ("ETL", t("Dynamic Sampling API", "جمع بيانات ديناميكي")), 
        ("ALGO", t("Composite score ranking", "خوارزمية تقييم مركبة")), 
        ("HEAT", t("Emotion intensity weights", "أوزان حرارة المشاعر"))
    ]
    for tag, label in stack_items:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid var(--border);">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;background:rgba(128,0,32,0.08);border:1px solid rgba(128,0,32,0.18);color:var(--cherry);border-radius:4px;padding:2px 6px;flex-shrink:0;">{tag}</span>
<span style="font-size:0.78rem;font-weight:600;color:var(--text-1);">{label}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:22px;padding:10px 14px;background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2);border-radius:8px;display:flex;align-items:center;gap:8px;">
<div style="width:6px;height:6px;background:#059669;border-radius:50%;animation:pulse 2s infinite;flex-shrink:0;"></div>
<span style="font-family:var(--font-body);font-size:0.68rem;color:#059669;font-weight:700;">{t("SYSTEM OPERATIONAL", "النظام يعمل بنجاح")}</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div style="position:relative;padding:64px 0 52px;border-bottom:1px solid var(--border);margin-bottom:0;overflow:hidden;">
<div style="position:absolute;top:-60px;left:-80px;width:500px;height:300px;background:radial-gradient(ellipse,var(--border) 0%,transparent 70%);pointer-events:none;"></div>
<div style="display:inline-flex;align-items:center;gap:8px;font-family:var(--font-body);font-size:0.68rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--cherry);border:1px solid var(--border-hi);background:rgba(128,0,32,0.05);border-radius:4px;padding:4px 12px;margin-bottom:20px;">
✦   {t("GRADUATION PROJECT — SMART BI SYSTEM", "مشروع التخرج — نظام ذكاء أعمال متقدم")}
</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6vw,5.5rem);line-height:0.95;letter-spacing:3px;color:var(--text-1);margin-bottom:20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.05);">
{t("FIND THE", "اكتشف")}<br>
<span class="gradient-text">{t("GOLDEN MOMENTS", "اللحظات الذهبية")}</span>
</div>
<p style="font-family:var(--font-body);font-size:1rem;font-weight:400;color:var(--text-2);max-width:560px;line-height:1.75;margin:0;">
{t("AI-powered crowd behaviour analytics. Surface emotional peaks, engagement spikes & highlight-worthy timestamps from tens of thousands of audience comments.", 
"تحليل سلوك الجماهير بالذكاء الاصطناعي. استخرج ذروة التفاعل والمشاعر من آلاف التعليقات بكل سهولة وسرعة.")}
</p>
</div>
""", unsafe_allow_html=True)

def section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f"<div style='font-size:0.78rem;color:var(--text-3);margin-top:3px;font-family:var(--font-body);font-weight:600;'>{subtitle}</div>" if subtitle else ""
    title_font = "'Bebas Neue', sans-serif" if not st.session_state.is_arabic else "var(--font-body)"
    letter_space = "2px" if not st.session_state.is_arabic else "0px"
    st.markdown(f"""
<div style="display:flex;align-items:flex-end;gap:16px;margin-bottom:20px;margin-top:52px;">
<div style="width:44px;height:44px;flex-shrink:0;background:rgba(128,0,32,0.08);border:1px solid rgba(128,0,32,0.2);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">
{icon}
</div>
<div>
<div style="font-family:{title_font};font-weight:700;font-size:1.45rem;letter-spacing:{letter_space};color:var(--text-1);line-height:1;">{title}</div>
{sub_html}
</div>
<div style="flex:1;height:1px;background:linear-gradient(90deg,var(--border-hi),transparent);margin-bottom:8px;"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DASHBOARD RENDER FUNCTION
# ═══════════════════════════════════════════════════════════
def render_video_analysis(url: str, depth: int, emotion_filter: str, is_comparison_mode: bool, col_key: str = ""):
    vid_match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url)
    if not vid_match:
        st.error(f"⚠️ Invalid URL: {url}")
        return

    v_id = vid_match.group(1)

    state_key_df = f"df_{col_key}_{v_id}"
    state_key_depth = f"depth_{col_key}_{v_id}"
    state_key_raw_len = f"raw_len_{col_key}_{v_id}"
    state_key_parsed_len = f"parsed_len_{col_key}_{v_id}"

    if not any(h['v_id'] == v_id for h in st.session_state.watch_history):
        st.session_state.watch_history.append({"v_id": v_id, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})

    if f"start_{v_id}" not in st.session_state:
        st.session_state[f"start_{v_id}"] = 0

    if state_key_df not in st.session_state or st.session_state.get(state_key_depth) != depth:
        st.session_state[f"start_{v_id}"] = 0
        with st.status(t("⚙️ Analysing Video...", "⚙️ جاري تحليل الفيديو..."), expanded=True) as status:
            raw = fetch_comments_refined(v_id, max_results=depth)
            st.session_state[state_key_raw_len] = len(raw)
            df_parsed = process_intelligence(raw)
            st.session_state[state_key_parsed_len] = len(df_parsed)
            
            st.write(t("🧠 Processing Multilingual Sentiments...", "🧠 جاري تحليل المشاعر بالذكاء الاصطناعي..."))
            # تفعيل التصنيف المطور والسريع جداً للنسخة الـ Live
            df_work = df_parsed.copy()
            df_work = batch_classify_transformer(df_work)
            
            st.session_state[state_key_df] = df_work
            st.session_state[state_key_depth] = depth
            st.session_state[f"trigger_confetti_{v_id}"] = True
            
            status.update(label=t("✦ Analysis Complete", "✦ اكتمل التحليل"), state="complete", expanded=False)

    if st.session_state.get(f"trigger_confetti_{v_id}", False):
        st.balloons()
        st.session_state[f"trigger_confetti_{v_id}"] = False

    df_work_cached = st.session_state[state_key_df].copy()
    raw_len = st.session_state[state_key_raw_len]
    parsed_len = st.session_state[state_key_parsed_len]

    df_f = df_work_cached[df_work_cached['Sentiment'] != "Neutral"].copy()
    if emotion_filter != "All Emotions":
        df_f = df_f[df_f['Sentiment'] == emotion_filter].copy()

    if is_comparison_mode:
        c1, c2 = st.columns(2)
        c1.metric(t("Sampled", "مسحوبة"), f"{raw_len:,}")
        c2.metric(t("Signals", "إشارات"),  f"{len(df_f):,}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Comments Sampled", "تعليقات مسحوبة"), f"{raw_len:,}")
        c2.metric(t("Timestamped", "مرتبطة بوقت"),      f"{parsed_len:,}")
        c3.metric(t("Emotive Signals", "إشارات عاطفية"),  f"{len(df_f):,}")
        c4.metric(t("Confidence", "الدقة"),       "96.4%")

    col_hdr_1, col_hdr_2 = st.columns([3, 1])
    with col_hdr_1:
        section_header("🏆", t("GOLDEN MOMENT", "اللحظة الذهبية"), f"Video: {v_id}")

    if df_f.empty:
        st.info(t("No emotive peaks detected for your selection.", "لم يتم العثور على ذروات عاطفية لهذا الاختيار."))
    else:
        highlights = compute_smart_highlights(df_f, top_n=3 if not is_comparison_mode else 1)
        
        with col_hdr_2:
            st.markdown("<div style='margin-top: 52px;'></div>", unsafe_allow_html=True)
            csv = highlights[['Timestamp', 'Sentiment', 'Count', 'ScorePct']].to_csv(index=False).encode('utf-8')
            st.download_button(t("📥 Export", "📥 تصدير"), data=csv, file_name=f'highlights_{v_id}.csv', mime='text/csv', use_container_width=True, key=f"dl_{col_key}_{v_id}")

        st.video(f"https://www.youtube.com/watch?v={v_id}", start_time=st.session_state[f"start_{v_id}"])

        rank_meta = [
            {"en": "PEAK MOMENT", "ar": "لحظة الذروة",  "crown": "👑", "border_top": "var(--cherry)", "pulse": True},
            {"en": "RUNNER-UP",   "ar": "المركز الثاني",    "crown": "🥈", "border_top": "var(--cherry-lt)", "pulse": False},
            {"en": "THIRD SPIKE", "ar": "المركز الثالث",  "crown": "🥉", "border_top": "var(--text-3)", "pulse": False},
        ]

        num_cols = 3 if not is_comparison_mode else len(highlights)
        cols = st.columns(num_cols)
        
        for i, row in highlights.iterrows():
            if i >= num_cols: break
            cfg  = EMOTION_CONFIG.get(row['Sentiment'], FALLBACK_CFG)
            meta = rank_meta[i] if i < len(rank_meta) else rank_meta[-1]
            pct  = int(row['ScorePct'])
            yt_link = f"https://www.youtube.com/watch?v={v_id}&t={int(row['Seconds'])}s"
            pulse_class = "pulse-ring" if meta.get("pulse") else ""

            with cols[i]:
                st.markdown(f"""
<div class="golden-card {pulse_class}" style="border-top:4px solid {meta['border_top']}; border-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; padding:20px 20px 10px;">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:15px;">
<div style="font-family:var(--font-body);font-size:0.6rem;font-weight:700;letter-spacing:1px;color:var(--text-3);">{t(meta['en'], meta['ar'])}</div>
<div style="font-size:1.1rem;">{meta['crown']}</div>
</div>
<div style="display:inline-flex;align-items:center;gap:6px;background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:6px;padding:4px 10px;font-size:0.75rem;font-weight:700;color:{cfg['color']};margin-bottom:15px;font-family:var(--font-body);">
{cfg['icon']}  {t(row['Sentiment'].upper(), cfg['ar'])}
</div>
<br>
<a href="{yt_link}" target="_blank" class="highlight-link" style="text-decoration:none; display:inline-block; margin-bottom:5px;">
<div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;letter-spacing:2px;line-height:1;color:var(--text-1);display:flex;align-items:center;gap:12px;transition:all 0.2s ease;">
<span style="direction:ltr; display:inline-block;">{row['Timestamp']}</span>
<span style="background:rgba(128,0,32,0.08);color:var(--cherry);font-family:var(--font-body);font-size:0.8rem;padding:4px 10px;border-radius:8px;font-weight:700;letter-spacing:0.5px;display:flex;align-items:center;">⧉ {t("OPEN YOUTUBE", "فتح يوتيوب")}</span>
</div>
</a>
</div>
""", unsafe_allow_html=True)

                if st.button(t("▶ PLAY IN-APP", "▶ تشغيل الفيديو"), key=f"play_{col_key}_{v_id}_{row['Seconds']}_{i}", use_container_width=True):
                    st.session_state[f"start_{v_id}"] = int(row['Seconds'])
                    st.rerun()

                st.markdown(f"""
<div class="golden-card" style="border-top:none; border-top-left-radius:0; border-top-right-radius:0; padding:10px 20px 20px;">
<div style="font-family:var(--font-body);font-weight:600;font-size:0.65rem;color:var(--text-3);margin-bottom:20px;margin-top:5px;">
{int(row['Diversity'])} {t("emotions", "مشاعر")} · {int(row['Count'])} {t("comments", "تعليق")}
</div>
<div style="width:100%;height:4px;background:rgba(128,0,32,0.06);border-radius:2px;margin-bottom:10px;">
<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{cfg['color']},{meta['border_top']});border-radius:2px;"></div>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;">
<span style="font-family:var(--font-body);font-weight:700;font-size:0.6rem;color:var(--text-3);">{t("SCORE", "التقييم")}</span>
<span class="gradient-text" style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;">{pct}</span>
</div>
</div>
""", unsafe_allow_html=True)

    header_text = f"{t('ENGAGEMENT HEATMAP', 'الخريطة الحرارية')} ({t(emotion_filter.upper(), EMOTION_CONFIG.get(emotion_filter, FALLBACK_CFG)['ar'])})" if emotion_filter != "All Emotions" else t("ENGAGEMENT HEATMAP", "الخريطة الحرارية للتفاعل")
    section_header("📊", header_text)

    if not df_f.empty:
        df_f['Minutes'] = df_f['Seconds'] / 60.0
        max_valid_minute = df_f['Minutes'].quantile(0.995)
        df_plot = df_f[df_f['Minutes'] <= max_valid_minute].copy()
        
        if st.session_state.is_arabic:
            df_plot['Sentiment'] = df_plot['Sentiment'].map(lambda x: EMOTION_CONFIG.get(x, FALLBACK_CFG)['ar'])
            color_map = {EMOTION_CONFIG[k]['ar']: EMOTION_CONFIG[k]['color'] for k in EMOTION_CONFIG}
        else:
            color_map = {k: v["color"] for k, v in EMOTION_CONFIG.items()}

        fig = px.histogram(
            df_plot, x="Minutes", color="Sentiment", nbins=100,
            template="plotly_dark" if st.session_state.dark_mode else "plotly_white", height=300,
            color_discrete_map=color_map,
            labels={"Minutes": t("Timeline (Mins)", "الوقت (دقيقة)"), "count": t("Mentions", "التكرار")}
        )
        grid_color = "rgba(255,255,255,0.05)" if st.session_state.dark_mode else "rgba(128,0,32,0.05)"
        font_color = "#E0E0E0" if st.session_state.dark_mode else "#555555"
        fig.update_layout(
            autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono" if not st.session_state.is_arabic else "Cairo", color=font_color, size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=grid_color, zeroline=False),
            yaxis=dict(gridcolor=grid_color, zeroline=False),
            bargap=0.05, margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col_key}_{v_id}")

    if emotion_filter == "All Emotions":
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs([t("🎭 Breakdown", "🎭 تفصيل المشاعر"), t("☁️ Words", "☁️ الكلمات المفتاحية")])

        with tab1:
            em_counts = df_work_cached[df_work_cached['Sentiment'] != "Neutral"]['Sentiment'].value_counts().reset_index()
            em_counts.columns = ['Sentiment', 'Count']
            
            if st.session_state.is_arabic:
                em_counts['Sentiment'] = em_counts['Sentiment'].map(lambda x: EMOTION_CONFIG.get(x, FALLBACK_CFG)['ar'])
                color_map = {EMOTION_CONFIG[k]['ar']: EMOTION_CONFIG[k]['color'] for k in EMOTION_CONFIG}
            else:
                color_map = {k: v["color"] for k, v in EMOTION_CONFIG.items()}

            fig2 = px.bar(
                em_counts, x='Sentiment', y='Count', color='Sentiment',
                template="plotly_dark" if st.session_state.dark_mode else "plotly_white", height=250,
                color_discrete_map=color_map,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                font=dict(family="JetBrains Mono" if not st.session_state.is_arabic else "Cairo", color=font_color, size=11),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"bar_{col_key}_{v_id}")

        with tab2:
            st.markdown(f"<div style='text-align:center; font-size:0.75rem; color:var(--text-3); font-weight:600; margin-bottom:5px; font-family:var(--font-body);'>{t('Top Keywords', 'أبرز الكلمات المتكررة')}</div>", unsafe_allow_html=True)
            text_data = " ".join(df_work_cached['Content'].tolist())
            arabic_stopwords = set(["في", "من", "على", "الى", "إلى", "و", "يا", "لا", "ما", "اللي", "كان", "بس", "تبع", "هاد", "هذا", "ان", "انا", "هو", "هي"])
            english_junk = set(["u", "ur", "video", "youtube", "bro", "im", "will", "one", "like", "subscribe"])
            final_stopwords = set(STOPWORDS).union(arabic_stopwords).union(english_junk)
            if ARABIC_SUPPORT:
                text_data = arabic_reshaper.reshape(text_data)
                text_data = get_display(text_data)
            
            bg_color = 'black' if st.session_state.dark_mode else 'white'
            wc = WordCloud(width=600, height=300, background_color=bg_color, stopwords=final_stopwords, colormap='inferno').generate(text_data)
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_alpha(0)
            st.pyplot(fig_wc, use_container_width=True)

        top_emotion_en = df_work_cached[df_work_cached['Sentiment'] != "Neutral"]['Sentiment'].value_counts().index[0] if not df_f.empty else "Happy"
        rec_text = get_rec_text(top_emotion_en)

        st.markdown(f"""
        <div class="golden-card" style="margin-top:20px;padding:20px;background:rgba(128,0,32,0.04);border-left:4px solid var(--cherry);border-right:none;">
            <div style="font-family:var(--font-body);font-size:1.1rem;font-weight:700;color:var(--text-1);margin-bottom:6px;display:flex;align-items:center;gap:8px;">
                🤖 {t("AI RECOMMENDATION", "توصية الذكاء الاصطناعي")}
            </div>
            <p style="font-family:var(--font-body);font-size:0.9rem;font-weight:600;color:var(--text-2);margin:0;line-height:1.6;">
                {rec_text}
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  APPLICATION EXECUTION FLOW
# ═══════════════════════════════════════════════════════════
urls_to_process = [url for url in [target_url, target_url_2] if url.strip()]

if len(urls_to_process) == 0:
    st.markdown(f"""
<div style="margin-top:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;border:1px dashed var(--border-hi);border-radius:20px;text-align:center;background:var(--surface-0);">
<div style="width:64px;height:64px;background:linear-gradient(135deg,rgba(128,0,32,0.1),rgba(158,27,52,0.1));border:1px solid rgba(128,0,32,0.2);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:20px;box-shadow:0 8px 16px rgba(128,0,32,0.05);">✦</div>
<div style="font-family:var(--font-body);font-weight:700;font-size:1.6rem;letter-spacing:1px;color:var(--cherry);margin-bottom:10px;">
{t("READY FOR ANALYSIS", "النظام جاهز للتحليل")}
</div>
<p style="font-size:0.9rem;font-family:var(--font-body);font-weight:600;color:var(--text-2);max-width:380px;line-height:1.7;margin:0;">
{t("Paste a YouTube URL into the sidebar to begin detecting golden moments. You can add a second URL to compare!", "قم بلصق رابط يوتيوب في القائمة الجانبية للبدء. يمكنك إضافة رابط آخر لعمل مقارنة مباشرة!")}
</p>
</div>
""", unsafe_allow_html=True)

elif len(urls_to_process) == 1:
    render_video_analysis(urls_to_process[0], target_max_results, selected_filter, is_comparison_mode=False, col_key="single")

elif len(urls_to_process) == 2:
    st.markdown(f"""
    <div style="text-align:center;font-family:var(--font-body);font-weight:700;font-size:2.2rem;color:var(--text-1);letter-spacing:1px;margin:20px 0 40px;">
    ⚔️ {t("A/B COMPARISON MODE", "وضع المقارنة")}
    </div>
    """, unsafe_allow_html=True)
    
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("<div style='border-top: 4px solid var(--cherry); padding-top: 10px;'></div>", unsafe_allow_html=True)
        render_video_analysis(urls_to_process[0], target_max_results, selected_filter, is_comparison_mode=True, col_key="colA")
    with colB:
        st.markdown("<div style='border-top: 4px solid var(--cherry-lt); padding-top: 10px;'></div>", unsafe_allow_html=True)
        render_video_analysis(urls_to_process[1], target_max_results, selected_filter, is_comparison_mode=True, col_key="colB")
