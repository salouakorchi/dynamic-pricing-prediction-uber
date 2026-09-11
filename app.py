import streamlit as st
import numpy as np
import joblib
import requests
import tensorflow as tf
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import math
import random
import json

# ============================================================
st.set_page_config(
    page_title="Fare.io — Intelligence Tarifaire",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DARK MODE INIT ────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = False

dark = st.session_state.dark

# ============================================================
# DESIGN SYSTEM — DYNAMIC THEME
# ============================================================
if dark:
    theme = """
    --bg:        #0a0a0a;
    --bg2:       #111111;
    --bg3:       #1a1a1a;
    --border:    #2a2a2a;
    --border2:   #333333;
    --text:      #f5f5f5;
    --text2:     #a0a0a0;
    --text3:     #606060;
    --black:     #f5f5f5;
    --white:     #111111;
    --hero-bg:   #0f0f0f;
    --card-bg:   #141414;
    --input-bg:  #1e1e1e;
    --nav-bg:    rgba(10,10,10,0.95);
    """
else:
    theme = """
    --bg:        #ffffff;
    --bg2:       #f9f9f9;
    --bg3:       #f3f3f3;
    --border:    #e8e8e8;
    --border2:   #d4d4d4;
    --text:      #0a0a0a;
    --text2:     #545454;
    --text3:     #8a8a8a;
    --black:     #000000;
    --white:     #ffffff;
    --hero-bg:   #000000;
    --card-bg:   #ffffff;
    --input-bg:  #f9f9f9;
    --nav-bg:    rgba(255,255,255,0.95);
    """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
    {theme}
    --blue:      #276EF1;
    --blue-dark: #1957D2;
    --green:     #05944F;
    --amber:     #C8760A;
    --red:       #C2292A;
    --r:         12px;
    --r2:        16px;
    --r3:        20px;
    --shadow:    0 2px 12px rgba(0,0,0,{'0.3' if dark else '0.06'});
    --shadow-md: 0 8px 32px rgba(0,0,0,{'0.5' if dark else '0.12'});
    --shadow-lg: 0 16px 48px rgba(0,0,0,{'0.6' if dark else '0.16'});
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    transition: background 0.3s ease, color 0.3s ease;
}}

[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer, header {{ display: none !important; }}

.block-container {{ padding: 0 !important; max-width: 100% !important; }}

/* ── NAV ─────────────────────────────────────────────── */
.nav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2.5rem; height: 68px;
    background: var(--nav-bg);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 200;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}}
.nav-brand {{ display: flex; align-items: center; gap: 1rem; }}
.nav-logo {{
    font-size: 1.5rem; font-weight: 900; letter-spacing: -0.5px;
    color: var(--text);
    background: linear-gradient(135deg, {'#ffffff' if dark else '#000000'} 0%, {'#888' if dark else '#444'} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.nav-divider {{ width: 1px; height: 20px; background: var(--border); }}
.nav-subtitle {{ font-size: 0.78rem; font-weight: 500; color: var(--text3); }}
.nav-right {{ display: flex; align-items: center; gap: 0.75rem; }}
.nav-pill {{
    display: flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 50px;
    background: var(--bg3); border: 1px solid var(--border);
    font-size: 0.72rem; font-weight: 600; color: var(--text2);
    letter-spacing: 0.3px;
}}
.dot-live {{
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    animation: blink 2s ease infinite; flex-shrink: 0;
}}
@keyframes blink {{
    0%,100%{{ box-shadow: 0 0 0 0 rgba(5,148,79,0.4); }}
    50%{{ box-shadow: 0 0 0 5px rgba(5,148,79,0); }}
}}
.nav-time {{ font-size: 0.78rem; font-weight: 500; color: var(--text3); }}

/* ── SECTION LABELS ──────────────────────────────────── */
.lbl {{
    font-size: 0.68rem; font-weight: 700; color: var(--text3);
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: 0.8rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px;
}}

/* ── WEATHER CARD ────────────────────────────────────── */
.wx {{
    display: grid; grid-template-columns: auto 1fr auto; gap: 1.5rem; align-items: center;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r3); padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: background 0.3s, border-color 0.3s;
}}
.wx-city {{ font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 2px; }}
.wx-status {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.68rem; font-weight: 600; color: var(--green); letter-spacing: 0.3px;
}}
.wx-emoji-lg {{ font-size: 3rem; }}
.wx-temp-display {{ font-size: 3rem; font-weight: 800; color: var(--text); line-height: 1; letter-spacing: -1px; }}
.wx-temp-display sup {{ font-size: 1.2rem; vertical-align: super; font-weight: 600; }}
.wx-feels {{ font-size: 0.72rem; color: var(--text3); margin-top: 3px; font-weight: 500; }}
.wx-desc-txt {{ font-size: 0.82rem; color: var(--text2); margin-top: 2px; font-weight: 400; }}
.wx-metrics {{ display: flex; gap: 0.5rem; }}
.wx-m {{
    text-align: center; background: var(--bg);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.7rem 0.9rem; min-width: 64px;
    transition: background 0.3s;
}}
.wx-mv {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
.wx-mk {{ font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; font-weight: 600; }}

/* ── LOCATION CARD ───────────────────────────────────── */
.loc-card {{
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r3); padding: 1.75rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: background 0.3s, border-color 0.3s;
}}
.loc-route {{
    display: flex; align-items: stretch; gap: 0; margin-bottom: 1.2rem;
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--r2); overflow: hidden;
}}
.loc-point {{
    flex: 1; padding: 1rem 1.2rem;
}}
.loc-point-pick {{ border-right: 1px solid var(--border); }}
.loc-label {{
    font-size: 0.6rem; font-weight: 700; color: var(--text3);
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 5px;
}}
.loc-dot-green {{ width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; }}
.loc-dot-red   {{ width: 8px; height: 8px; border-radius: 50%; background: var(--red);   display: inline-block; }}
.loc-name {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
.loc-coords {{ font-size: 0.65rem; color: var(--text3); margin-top: 2px; font-weight: 400; }}

.loc-divider {{
    display: flex; align-items: center; justify-content: center;
    width: 48px; flex-shrink: 0;
    background: var(--bg2);
    font-size: 1rem; color: var(--text3);
    border-left: 1px solid var(--border);
    border-right: 1px solid var(--border);
}}

.dist-badge {{
    display: flex; align-items: center; justify-content: space-between;
    background: {'rgba(39,110,241,0.12)' if dark else '#EBF3FE'};
    border: 1px solid {'rgba(39,110,241,0.25)' if dark else '#C5D9FB'};
    border-radius: var(--r); padding: 0.9rem 1.2rem;
}}
.dist-main {{ font-size: 1.4rem; font-weight: 800; color: var(--blue); letter-spacing: -0.5px; }}
.dist-sub  {{ font-size: 0.68rem; color: var(--text3); margin-top: 2px; font-weight: 500; }}
.dist-eta  {{
    text-align: right; background: var(--bg);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.6rem 1rem;
}}
.dist-eta-val {{ font-size: 1rem; font-weight: 700; color: var(--text); }}
.dist-eta-lbl {{ font-size: 0.58rem; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; font-weight: 600; }}

/* ── CONTROLS ────────────────────────────────────────── */
.ctrl-panel {{
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r3); padding: 1.75rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: background 0.3s, border-color 0.3s;
}}
[data-testid="stSlider"] {{ padding: 0 !important; margin-bottom: 0.3rem !important; }}
[data-testid="stSlider"] label p {{
    font-family: 'Inter', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 600 !important; color: var(--text2) !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}}
[data-baseweb="slider"] div[role="progressbar"] {{ background: var(--blue) !important; }}
[data-baseweb="slider"] [role="slider"] {{
    background: var(--blue) !important; border: 3px solid var(--bg) !important;
    box-shadow: 0 0 0 2px var(--blue), var(--shadow) !important;
    width: 18px !important; height: 18px !important;
}}
[data-testid="stThumbValue"] {{
    font-family: 'Inter', sans-serif !important; font-size: 0.65rem !important;
    font-weight: 700 !important; color: var(--text) !important;
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    padding: 2px 8px !important; border-radius: 6px !important;
    box-shadow: var(--shadow) !important;
}}
[data-testid="stSelectbox"] label p {{
    font-family: 'Inter', sans-serif !important; font-size: 0.72rem !important;
    font-weight: 600 !important; color: var(--text2) !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}}
[data-baseweb="select"] > div {{
    background: var(--bg) !important; border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 500 !important;
}}
[data-baseweb="select"] > div:hover {{ border-color: var(--blue) !important; }}
[data-baseweb="menu"] {{
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; box-shadow: var(--shadow-md) !important;
}}

.wx-inject {{
    display: flex; align-items: center; gap: 0.75rem;
    background: {'rgba(39,110,241,0.15)' if dark else '#EBF3FE'};
    border: 1px solid {'rgba(39,110,241,0.3)' if dark else '#C5D9FB'};
    border-radius: 10px; padding: 0.8rem 1rem;
    font-size: 0.75rem; font-weight: 500; color: var(--blue); margin-top: 0.5rem;
}}
.wx-inject-data {{ color: var(--blue); font-size: 0.7rem; margin-top: 2px; font-weight: 600; opacity: 0.8; }}

/* ── BUTTON ──────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, var(--blue) 0%, var(--blue-dark) 100%) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 0.3px; text-transform: none;
    border: none !important; border-radius: 12px !important;
    padding: 0.95rem 2rem !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(39,110,241,0.35) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(39,110,241,0.5) !important;
}}
.stButton > button:active {{ transform: scale(0.99) !important; }}

/* ── RESULT CARD ─────────────────────────────────────── */
.result-wrap {{
    border-radius: var(--r3); overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-lg); margin-bottom: 1.5rem;
}}
.price-hero {{
    background: var(--hero-bg);
    padding: 3rem 2rem 2.5rem; text-align: center; position: relative;
    overflow: hidden;
}}
.price-hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(39,110,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}}
.ph-eyebrow {{
    font-size: 0.65rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: rgba(255,255,255,0.4); margin-bottom: 0.8rem;
}}
.ph-amount {{
    font-size: 6.5rem; font-weight: 800; color: #ffffff; line-height: 1;
    letter-spacing: -4px; position: relative;
}}
.ph-dollar {{ font-size: 2.5rem; vertical-align: super; font-weight: 700; letter-spacing: 0; }}
.ph-cents  {{ font-size: 2rem; letter-spacing: 0; font-weight: 700; }}
.surge-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 18px; border-radius: 50px; margin-top: 1.4rem;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
}}

/* ── RIDE COMPARISON TABLE ───────────────────────────── */
.comp-wrap {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--r3); overflow: hidden; margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: background 0.3s;
}}
.comp-head {{
    display: grid; grid-template-columns: 1fr 90px 120px;
    padding: 0.85rem 1.4rem; background: var(--bg2);
    border-bottom: 1px solid var(--border);
}}
.comp-hc {{ font-size: 0.62rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; }}
.comp-hc.right {{ text-align: right; }}
.comp-r {{
    display: grid; grid-template-columns: 1fr 90px 120px;
    align-items: center; padding: 1rem 1.4rem;
    border-bottom: 1px solid var(--border);
    transition: background 0.12s;
}}
.comp-r:last-child {{ border-bottom: none; }}
.comp-r:hover {{ background: var(--bg2); }}
.comp-r.is-active {{
    background: {'rgba(39,110,241,0.12)' if dark else '#EBF3FE'};
    border-left: 3px solid var(--blue);
    padding-left: calc(1.4rem - 3px);
}}
.comp-nm {{ display: flex; align-items: center; gap: 0.75rem; font-size: 0.9rem; font-weight: 600; color: var(--text); }}
.comp-ic {{
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; background: var(--bg3); flex-shrink: 0;
}}
.comp-mt {{ font-size: 0.7rem; color: var(--text3); margin-top: 1px; font-weight: 400; }}
.comp-pr {{ font-size: 1.05rem; font-weight: 800; color: var(--text); text-align: right; letter-spacing: -0.3px; }}
.comp-bg {{
    font-size: 0.6rem; padding: 4px 10px; border-radius: 6px;
    text-transform: uppercase; letter-spacing: 0.5px;
    text-align: right; display: block; font-weight: 700;
}}
.bg-best {{ background: {'rgba(5,148,79,0.2)' if dark else '#E6F4EE'}; color: var(--green); }}
.bg-mid  {{ background: {'rgba(200,118,10,0.2)' if dark else '#FEF3E2'}; color: var(--amber); }}
.bg-prem {{ background: var(--bg3); color: var(--text2); }}
.bg-eco  {{ background: {'rgba(39,110,241,0.15)' if dark else '#EBF3FE'}; color: var(--blue); }}

/* ── INSIGHTS ────────────────────────────────────────── */
.ins-card {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--r2); padding: 1.2rem 1.3rem; margin-bottom: 0.6rem;
    box-shadow: var(--shadow); transition: box-shadow 0.15s, background 0.3s;
}}
.ins-card:hover {{ box-shadow: var(--shadow-md); }}
.ins-ic {{
    width: 34px; height: 34px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; margin-bottom: 0.6rem;
}}
.ins-lbl {{ font-size: 0.65rem; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; }}
.ins-val {{ font-size: 1.3rem; font-weight: 800; color: var(--text); margin-top: 2px; letter-spacing: -0.5px; }}
.ins-sub {{ font-size: 0.68rem; color: var(--text3); margin-top: 2px; font-weight: 500; }}

/* ── DEMAND CHART ────────────────────────────────────── */
.surge-chart {{
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--r3); padding: 1.75rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow); transition: background 0.3s;
}}
.surge-bars {{ display: flex; align-items: flex-end; gap: 3px; height: 64px; margin: 1rem 0 0.5rem; }}
.s-bar {{
    flex: 1; border-radius: 4px 4px 0 0; min-height: 4px;
    background: var(--border2); transition: background 0.2s;
}}
.s-bar.peak {{ background: {'rgba(200,118,10,0.7)' if dark else '#FDE8C7'}; }}
.s-bar.high {{ background: {'rgba(194,41,42,0.7)' if dark else '#FBBFBF'}; }}
.s-bar.cur  {{ background: var(--blue); box-shadow: 0 0 12px rgba(39,110,241,0.6); }}
.surge-xlbls {{
    display: flex; justify-content: space-between;
    font-size: 0.6rem; color: var(--text3); font-weight: 500;
}}
.leg {{ display: flex; align-items: center; gap: 5px; font-size: 0.62rem; color: var(--text3); font-weight: 500; }}
.leg-dot {{ width: 10px; height: 8px; border-radius: 2px; display: inline-block; }}

/* ── HISTORY ─────────────────────────────────────────── */
.hist-item {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.3rem; border-radius: var(--r2);
    background: var(--card-bg); border: 1px solid var(--border);
    margin-bottom: 0.5rem; transition: border-color 0.15s, box-shadow 0.15s, background 0.3s;
    box-shadow: var(--shadow);
}}
.hist-item:hover {{ border-color: var(--border2); box-shadow: var(--shadow-md); }}
.hist-main {{ font-size: 0.85rem; color: var(--text); font-weight: 600; }}
.hist-sub {{ font-size: 0.68rem; color: var(--text3); margin-top: 1px; font-weight: 400; }}
.hist-pr {{ font-size: 1.05rem; font-weight: 800; color: var(--text); letter-spacing: -0.3px; }}

/* ── EMPTY STATE ─────────────────────────────────────── */
.empty {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 5rem 2rem; text-align: center;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--r3); position: relative;
    box-shadow: var(--shadow);
}}
.empty-ring {{
    width: 90px; height: 90px; border-radius: 50%;
    background: var(--bg3); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    position: relative;
}}
.empty-ring::before {{
    content: ''; position: absolute; inset: -10px; border-radius: 50%;
    border: 1.5px dashed var(--border2);
    animation: spin 14s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.empty-ring-inner {{
    width: 60px; height: 60px; border-radius: 50%;
    background: var(--bg); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
}}
.empty-h {{ font-size: 1.6rem; font-weight: 800; color: var(--text); margin-top: 1.4rem; letter-spacing: -0.5px; }}
.empty-p {{ font-size: 0.82rem; color: var(--text3); font-weight: 400; margin-top: 0.5rem; max-width: 280px; line-height: 1.6; }}

[data-testid="column"] {{ padding: 0 0.4rem !important; }}

/* scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text3); }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    def qloss(q):
        def loss(yt, yp):
            e = yt - yp
            return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
        return loss
    m  = tf.keras.models.load_model('uber_prix_dynamique_v5.keras', custom_objects={'loss': qloss(0.5)})
    sx = joblib.load('scaler_X.pkl')
    sy = joblib.load('scaler_Y.pkl')
    fc = joblib.load('feature_columns.pkl')
    return m, sx, sy, fc

model, scaler_X, scaler_Y, feature_columns = load_model()


# ============================================================
# HEURE NYC AUTOMATIQUE
# ============================================================
nyc_tz   = ZoneInfo("America/New_York")
now_nyc  = datetime.now(nyc_tz)
nyc_hour = now_nyc.hour


# ============================================================
# MÉTÉO
# ============================================================
WMO = {
    0:("☀️","Ciel dégagé"), 1:("🌤️","Principalement dégagé"), 2:("⛅","Partiellement nuageux"),
    3:("☁️","Couvert"), 45:("🌫️","Brouillard"), 48:("🌫️","Brouillard givrant"),
    51:("🌦️","Bruine légère"), 53:("🌦️","Bruine"), 55:("🌧️","Bruine forte"),
    61:("🌧️","Pluie légère"), 63:("🌧️","Pluie"), 65:("🌧️","Forte pluie"),
    71:("❄️","Neige légère"), 73:("❄️","Neige"), 75:("❄️","Forte neige"),
    80:("🌦️","Averses"), 81:("🌧️","Fortes averses"), 82:("⛈️","Averses violentes"),
    95:("⛈️","Orage"), 99:("⛈️","Orage + grêle"),
}

@st.cache_data(ttl=300)
def get_weather():
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=40.7128&longitude=-74.0060"
            "&current=temperature_2m,precipitation_probability,"
            "weather_code,wind_speed_10m,relative_humidity_2m,apparent_temperature"
            "&temperature_unit=fahrenheit&wind_speed_unit=mph"
            "&timezone=America%2FNew_York",
            timeout=8
        )
        r.raise_for_status()
        c = r.json()["current"]
        icon, desc = WMO.get(c.get("weather_code", 0), ("🌡️","Variable"))
        return dict(temp=c["temperature_2m"], feels=c["apparent_temperature"],
                    precip=c["precipitation_probability"]/100,
                    wind=c["wind_speed_10m"], humidity=c["relative_humidity_2m"],
                    desc=desc, icon=icon, ok=True)
    except:
        return dict(temp=58.0, feels=55.0, precip=0.1, wind=8.0,
                    humidity=60, desc="Indisponible", icon="🌡️", ok=False)

wx = get_weather()


# ============================================================
# PRÉDICTION
# ============================================================
def predict(distance, surge, temp, precip, hour, ride):
    row = {c: 0 for c in feature_columns}
    row.update({
        'distance': distance, 'surge_multiplier': surge,
        'temperature': temp, 'precipProbability': precip,
        'hour_sin': math.sin(2*math.pi*hour/24),
        'hour_cos': math.cos(2*math.pi*hour/24),
        'distance_x_surge': distance*surge,
        'surge_flag': int(surge>1.0),
        'distance_sq': distance**2,
        'surge_sq': surge**2,
        'rain_x_surge': precip*surge,
    })
    k = f'name_{ride}'
    if k in row: row[k] = 1
    X = np.array([[row[c] for c in feature_columns]])
    p = model.predict(scaler_X.transform(X), verbose=0)
    return round(float(scaler_Y.inverse_transform(p)[0][0]), 2)


RIDE_META = {
    "UberX":     {"icon": "🚗", "desc": "Course quotidienne",    "tier": "economy"},
    "UberXL":    {"icon": "🚙", "desc": "Jusqu'à 6 passagers",   "tier": "mid"},
    "Black":     {"icon": "🖤", "desc": "Berline premium",        "tier": "premium"},
    "Black SUV": {"icon": "🚘", "desc": "SUV de luxe",            "tier": "premium"},
    "UberPool":  {"icon": "🤝", "desc": "Course partagée",        "tier": "economy"},
    "WAV":       {"icon": "♿", "desc": "Accès fauteuil roulant", "tier": "mid"},
}

SURGE_PATTERN = [0.3,0.2,0.2,0.2,0.3,0.5,0.9,1.0,0.8,0.6,0.5,0.5,
                 0.6,0.6,0.5,0.5,0.7,0.9,1.0,0.9,0.8,0.7,0.6,0.4]

# NYC landmark coordinates
NYC_LANDMARKS = {
    "Times Square":          (40.7580, -73.9855),
    "Central Park":          (40.7829, -73.9654),
    "JFK Airport":           (40.6413, -73.7781),
    "LaGuardia Airport":     (40.7769, -73.8740),
    "Grand Central":         (40.7527, -73.9772),
    "Brooklyn Bridge":       (40.7061, -73.9969),
    "Empire State Building": (40.7484, -73.9967),
    "Midtown Manhattan":     (40.7549, -73.9840),
}

# Durée estimée moyenne NYC : ~3 min/mile en circulation normale
def estimate_eta(distance_miles: float) -> int:
    return max(3, round(distance_miles * 3.2))


# ============================================================
# SESSION STATE
# ============================================================
defaults = dict(
    result=None, done=False, all_prices={},
    dist=3.0, surge=1.0, hour=nyc_hour,
    ride="UberX", temp=wx["temp"], precip=wx["precip"],
    history=[], pickup="Times Square", dropoff="Central Park"
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# NAV
# ============================================================
toggle_label = "☀️" if dark else "🌙"
toggle_title = "Mode clair" if dark else "Mode sombre"

col_nav, col_toggle = st.columns([11, 1])
with col_nav:
    st.markdown(f"""
    <div class="nav">
      <div class="nav-brand">
        <div class="nav-logo">Fare.io</div>
        <div class="nav-divider"></div>
        <div class="nav-subtitle">Intelligence Tarifaire · New York City</div>
      </div>
      <div class="nav-right">
        <div class="nav-pill">
          <span class="dot-live"></span>
          {"Open-Meteo · En direct" if wx["ok"] else "Mode hors ligne"}
        </div>
        <div class="nav-time">{now_nyc.strftime("%H:%M")} EST</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    st.markdown("""<style>
    .toggle-btn > div > button {
        background: var(--bg3) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 50% !important;
        width: 36px !important; height: 36px !important;
        min-height: 36px !important; padding: 0 !important;
        font-size: 1rem !important; box-shadow: none !important;
        margin-top: 16px; line-height: 1 !important;
    }
    .toggle-btn > div > button:hover {
        background: var(--border) !important;
        transform: none !important; box-shadow: none !important;
    }
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="toggle-btn">', unsafe_allow_html=True)
    if st.button(toggle_label, key="dark_toggle", help=toggle_title):
        st.session_state.dark = not st.session_state.dark
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# LAYOUT
# ============================================================
_, center, _ = st.columns([1, 2.6, 1])

with center:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── MÉTÉO ────────────────────────────────────────────────
    st.markdown('<div class="lbl">🌤 Conditions météo · New York City</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="wx">
      <div style="text-align:center"><div class="wx-emoji-lg">{wx["icon"]}</div></div>
      <div>
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:4px">
          <span class="wx-city">New York City</span>
          <span class="wx-status">
            <span style="width:5px;height:5px;border-radius:50%;background:var(--green);display:inline-block;margin-right:3px"></span>
            {"En direct" if wx["ok"] else "Données de secours"}
          </span>
        </div>
        <div class="wx-desc-txt">{wx["desc"]}</div>
      </div>
      <div style="display:flex;align-items:center;gap:1.5rem">
        <div style="text-align:right">
          <div class="wx-temp-display">{wx["temp"]:.0f}<sup>°F</sup></div>
          <div class="wx-feels">Ressenti {wx["feels"]:.0f}°F</div>
        </div>
        <div class="wx-metrics">
          <div class="wx-m"><div class="wx-mv">{wx["precip"]*100:.0f}%</div><div class="wx-mk">Pluie</div></div>
          <div class="wx-m"><div class="wx-mv">{wx["wind"]:.0f}</div><div class="wx-mk">mph</div></div>
          <div class="wx-m"><div class="wx-mv">{wx["humidity"]:.0f}%</div><div class="wx-mk">Humidité</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

   



    # ── PARAMÈTRES ───────────────────────────────────────────
    st.markdown('<div class="lbl">⚙️ Configuration du trajet</div>', unsafe_allow_html=True)
    st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        surge = st.slider("⚡ Multiplicateur de surge ×", 1.0, 3.0, 1.0, 0.25, key="slider_surge")
        # Distance ajustable — initialisée à la distance calculée depuis les lieux
        distance_finale = st.slider(
            "📐 Distance du trajet (miles)",
            0.5, 30.0, 2.0, 0.5,
            key="slider_dist"
        )

    with c2:
        hour = st.slider(
            f"🕐 Heure de départ (NYC — {now_nyc.strftime('%H:%M')} actuellement)",
            0, 23, nyc_hour, key="slider_hour"
        )
        ride = st.selectbox("🚗 Type de course", list(RIDE_META.keys()), key="select_ride")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── CONDITIONS MÉTÉO ─────────────────────────────────────
    st.markdown('<div class="lbl">🌡 Conditions météorologiques</div>', unsafe_allow_html=True)
    if wx["ok"]:
        temperature = wx["temp"]
        precip_prob = wx["precip"]
        st.markdown(f"""
        <div class="wx-inject">
          <span style="font-size:1rem;flex-shrink:0">⚡</span>
          <div>
            <div>Données injectées automatiquement depuis Open-Meteo — aucune saisie requise</div>
            <div class="wx-inject-data">{temperature:.1f}°F · {wx["desc"]} · Pluie {precip_prob*100:.0f}% · Vent {wx["wind"]:.0f} mph</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        cw1, cw2 = st.columns(2)
        with cw1: temperature = st.slider("🌡 Température (°F)", 0, 110, 58)
        with cw2: precip_prob = st.slider("🌧 Probabilité de pluie", 0.0, 1.0, 0.1, 0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    go = st.button("✦  Calculer le tarif", use_container_width=True)
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── LOGIQUE ──────────────────────────────────────────────
    if go:
        with st.spinner("Calcul du réseau neuronal…"):
            time.sleep(0.3)
            price = predict(distance_finale, surge, temperature, precip_prob, hour, ride)
            all_p = {rn: predict(distance_finale, surge, temperature, precip_prob, hour, rn)
                     for rn in RIDE_META}

        st.session_state.update(
            result=price, done=True, all_prices=all_p,
            dist=distance_finale, surge=surge, hour=hour,
            ride=ride, temp=temperature, precip=precip_prob
        )
        st.session_state.history = (
            [{"ride": ride, "price": price, "dist": distance_finale,
              "surge": surge, "time": now_nyc.strftime("%H:%M")}]
            + st.session_state.history
        )[:5]

    # ── RÉSULTATS ────────────────────────────────────────────
    if st.session_state.done and st.session_state.result:
        p  = st.session_state.result
        sv = st.session_state.surge
        r  = st.session_state.ride

        if sv == 1.0:
            s_bg, s_co, s_icon, s_txt = "#E6F4EE","#05944F","✦","Tarif standard"
            s_bd = "#B8DFC9"
        elif sv <= 1.75:
            s_bg, s_co, s_icon, s_txt = "#FEF3E2","#C8760A","▲",f"Surge actif ×{sv}"
            s_bd = "#F7D5A1"
        else:
            s_bg, s_co, s_icon, s_txt = "#FDEAEA","#C2292A","⚡",f"Surge élevé ×{sv}"
            s_bd = "#F5B8B8"

        if dark:
            s_bg = s_bg.replace("#E6F4EE","rgba(5,148,79,0.2)")\
                       .replace("#FEF3E2","rgba(200,118,10,0.2)")\
                       .replace("#FDEAEA","rgba(194,41,42,0.2)")
            s_bd = "transparent"

        doll = int(p)
        cent = round((p - doll) * 100)

        st.markdown('<div class="lbl">💳 Estimation du tarif</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-wrap">
          <div class="price-hero">
            <div class="ph-eyebrow">◈ Tarif estimé · {r.upper()}</div>
            <div class="ph-amount"><span class="ph-dollar">$</span>{doll}<span class="ph-cents">.{cent:02d}</span></div>
            <div class="surge-pill" style="background:{s_bg};color:{s_co};border:1px solid {s_bd}">
              {s_icon} &nbsp; {s_txt}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TABLEAU COMPARATIF ────────────────────────────────
        st.markdown('<div class="lbl">🚖 Toutes les options de course</div>', unsafe_allow_html=True)

        all_p = st.session_state.all_prices
        sorted_rides = sorted(all_p.items(), key=lambda x: x[1])
        min_p = sorted_rides[0][1]
        max_p = sorted_rides[-1][1]

        st.markdown("""
        <div class="comp-wrap">
          <div class="comp-head">
            <span class="comp-hc">Type de course</span>
            <span class="comp-hc right">Tarif est.</span>
            <span class="comp-hc right">Catégorie</span>
          </div>
        """, unsafe_allow_html=True)

        for rname, rp in sorted_rides:
            meta = RIDE_META[rname]
            active_cls = " is-active" if rname == r else ""
            if rp == min_p:
                badge_cls, badge_txt = "bg-best", "Meilleur prix"
            elif rp == max_p:
                badge_cls, badge_txt = "bg-prem", "Premium"
            elif meta["tier"] == "mid":
                badge_cls, badge_txt = "bg-mid", "Intermédiaire"
            else:
                badge_cls, badge_txt = "bg-eco", "Économique"
            st.markdown(
                f'<div class="comp-r{active_cls}">'
                f'<div class="comp-nm">'
                f'<div class="comp-ic">{meta["icon"]}</div>'
                f'<div><div>{rname}</div><div class="comp-mt">{meta["desc"]}</div></div>'
                f'</div>'
                f'<div class="comp-pr">${rp:.2f}</div>'
                f'<span class="comp-bg {badge_cls}">{badge_txt}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── INSIGHTS ─────────────────────────────────────────
        st.markdown('<div class="lbl">📊 Analyse du trajet</div>', unsafe_allow_html=True)
        per_mile   = p / st.session_state.dist
        savings    = max_p - min_p
        h          = st.session_state.hour
        demand_pct = int(SURGE_PATTERN[h] * 100)
        surge_cost = round(p - p / sv, 2)

        ins_data = [
            ("📐", "#EBF3FE",  "Coût par mile",    f"${per_mile:.2f}/mi", f"Sur {st.session_state.dist:.1f} mi"),
            ("💰", "#E6F4EE",  "Économie max",      f"${savings:.2f}",    "vs. l'option la plus chère"),
            ("📊", "#FEF3E2",  "Niveau de demande", f"{demand_pct}%",     f"Typique à {h:02d}:00 à NYC"),
            ("⚡", "#F0F0F0",  "Impact du surge",   f"+${surge_cost:.2f}", f"Ajouté par le surge ×{sv}"),
        ]
        ic1, ic2 = st.columns(2)
        for idx, (icon, bg, lbl, val, sub) in enumerate(ins_data):
            col = ic1 if idx % 2 == 0 else ic2
            with col:
                st.markdown(
                    f'<div class="ins-card">'
                    f'<div class="ins-ic" style="background:{bg}">{icon}</div>'
                    f'<div class="ins-lbl">{lbl}</div>'
                    f'<div class="ins-val">{val}</div>'
                    f'<div class="ins-sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── GRAPHIQUE DEMANDE HORAIRE ─────────────────────────
        st.markdown('<div class="lbl">📈 Demande horaire à NYC</div>', unsafe_allow_html=True)
        bars_html = ""
        for i, v in enumerate(SURGE_PATTERN):
            pct = int(v * 100)
            if i == h:
                cls = "cur"
            elif v >= 0.85:
                cls = "high"
            elif v >= 0.7:
                cls = "peak"
            else:
                cls = ""
            bars_html += f'<div class="s-bar {cls}" style="height:{max(4,pct*0.6):.0f}px" title="{i:02d}:00 · {pct}%"></div>'

        st.markdown(
            f'<div class="surge-chart">'
            f'<div class="lbl" style="margin-bottom:0">Demande par heure · moyenne NYC</div>'
            f'<div class="surge-bars">{bars_html}</div>'
            f'<div class="surge-xlbls"><span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span></div>'
            f'<div style="display:flex;gap:1rem;margin-top:0.8rem">'
            f'<span class="leg"><span class="leg-dot" style="background:var(--red)"></span>Surge élevé</span>'
            f'<span class="leg"><span class="leg-dot" style="background:var(--amber)"></span>Heures de pointe</span>'
            f'<span class="leg"><span class="leg-dot" style="background:var(--border2)"></span>Standard</span>'
            f'<span class="leg"><span class="leg-dot" style="background:var(--blue)"></span>Heure sélectionnée</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # ── HISTORIQUE ───────────────────────────────────────
        if len(st.session_state.history) > 1:
            st.markdown('<div class="lbl">🕐 Prédictions récentes</div>', unsafe_allow_html=True)
            for h_entry in st.session_state.history[1:]:
                meta = RIDE_META.get(h_entry["ride"], {"icon": "🚗"})
                route_str = f'{h_entry.get("from","?")} → {h_entry.get("to","?")}'
                st.markdown(
                    f'<div class="hist-item">'
                    f'<div>'
                    f'<div class="hist-main">{meta["icon"]} &nbsp;{h_entry["ride"]} · {h_entry["dist"]:.1f} mi</div>'
                    f'<div class="hist-sub">{route_str} · Surge ×{h_entry["surge"]} · {h_entry["time"]}</div>'
                    f'</div>'
                    f'<div class="hist-pr">${h_entry["price"]:.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="empty">'
            '<div class="empty-ring"><div class="empty-ring-inner">🚗</div></div>'
            '<div class="empty-h">Prêt à estimer</div>'
            '<div class="empty-p">Choisissez votre départ et destination · Configurez votre trajet · Cliquez sur calculer</div>'
            '</div>'
            '<div style="height:2rem"></div>',
            unsafe_allow_html=True
        )