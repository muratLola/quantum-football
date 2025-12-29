import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from difflib import get_close_matches
import os

# -----------------------------------------------------------------------------
# 1. KONFIGÜRASYON & SABİTLER
# -----------------------------------------------------------------------------
CONFIG = {
    "SIM_COUNT": 15000,
    "HOME_ADV_BASE": 0.35,
    "AWAY_PENALTY": 0.90,
    "CACHE_TTL": 3600,
    "LOOKAHEAD_DAYS": 30,
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png" 
}

st.set_page_config(
    page_title="Quantum AI v36.2 Render",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Oturum Hafızası
if 'q_data' not in st.session_state:
    st.session_state.q_data = None

# -----------------------------------------------------------------------------
# 2. PREMIUM UI (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
    
    .quantum-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #fff;
        text-align: center;
        letter-spacing: 2px;
        margin-top: 10px;
        background: linear-gradient(90deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(0, 255, 136, 0.3);
    }
    
    .team-header { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 15px; }
    .team-logo { width: 60px; height: 60px; object-fit: contain; filter: drop-shadow(0 0 8px rgba(255,255,255,0.2)); }
    .team-name { font-size: 1.4rem; font-weight: 800; color: #f1f5f9; }
    
    .scoreboard { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; padding: 20px; border-radius: 16px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    
    .ticket-container { background: radial-gradient(circle at center, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; border-top: 4px solid #00ff88; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.6); margin-bottom: 25px; max-width: 800px; margin-left: auto; margin-right: auto; }
    .main-pred { font-size: 3rem; font-weight: 900; color: #facc15; margin: 10px 0; letter-spacing: -1px; text-shadow: 0 0 15px rgba(250, 204, 21, 0.4); }
    
    .market-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: all 0.2s; }
    .market-box:hover { border-color: #00ff88; transform: translateY(-3px); }
    .market-title { color: #00ff88; font-weight: bold; font-size: 0.9rem; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .market-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.95rem; color: #e5e7eb; }
    
    .prob-high { color: #4ade80; font-weight: 800; } 
    .prob-med { color: #facc15; font-weight: 600; } 
    .prob-low { color: #f87171; } 

    .form-badges { display: flex; gap: 4px; justify-content: center; margin-top: 5px; }
    .badge { width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #000; font-size: 0.75rem; }
    .badge-W { background-color: #4ade80; } .badge-D { background-color: #facc15; } .badge-L { background-color: #f87171; }
    
    .decision-play { background-color: #238636; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }
    .decision-pass { background-color: #da3633; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }

    .rtl { direction: rtl; text-align: right; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); } }
    .live-badge { background-color: #ff5252; color: white; padding: 3px 8px; border-radius: 10px; font-weight: bold; font-size: 0.7rem; animation: pulse-red 2s infinite; vertical-align: middle; }
    
    .disclaimer { font-size: 0.75rem; color: #64748b; text-align: center; margin-top: 50px; border-top: 1px solid #334155; padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DİL PAKETİ
# -----------------------------------------------------------------------------
LANGUAGES = {
    "tr": {"flag": "🇹🇷", "name": "Türkçe", "dir": "ltr"},
    "en": {"flag": "🇬🇧", "name": "English", "dir": "ltr"},
    "de": {"flag": "🇩🇪", "name": "Deutsch", "dir": "ltr"},
    "fr": {"flag": "🇫🇷", "name": "Français", "dir": "ltr"},
    "ar": {"flag": "🇸🇦", "name": "العربية", "dir": "rtl"}
}

TRANSLATIONS = {
    "tr": {
        "title": "QUANTUM AI v36", "sel_league": "LİG SEÇ", "sel_match": "MAÇ SEÇ", "live_mode": "🔴 CANLI SİMÜLASYON",
        "btn_analyze": "ANALİZİ BAŞLAT", "loading": "Quantum İşleniyor...", "ai_decision": "AI KARARI",
        "ticket_title": "ENSEMBLE TAHMİNİ", "score": "SKOR", "confidence": "GÜVEN",
        "win": "KAZANIR", "draw": "BERABERLİK", "play": "OYNA", "pass": "PAS GEÇ",
        "high": "YÜKSEK", "med": "ORTA", "low": "DÜŞÜK", "market_dc": "ÇİFTE ŞANS", "market_ou": "ALT / ÜST",
        "market_other": "DİĞER", "team_goals": "TAKIM GOLLERİ", "home_goal": "Ev Gol", "away_goal": "Dep Gol",
        "ai_reason": "AI Nedeni", "power_balance": "Güç Dengesi", "minute": "Dakika", 
        "disclaimer": "⚠️ YASAL UYARI: İstatistiksel veri analizidir. Bahis tavsiyesi değildir. 18+",
        "warn_key": "⚠️ API Anahtarı Eksik! Lütfen sol menüden giriniz.", "warn_no_data": "⚠️ Veri bulunamadı.",
        "r_power": "Güç farkı ve saha avantajı belirleyici.", "r_live": "Canlı skor ve dakikaya göre simüle edildi.",
        "r_var": "Yüksek belirsizlik tespit edildi.", "r_elo": "Kritik maç, ELO modeli önceliklendirildi."
    },
    "en": {
        "title": "QUANTUM AI v36", "sel_league": "SELECT LEAGUE", "sel_match": "SELECT MATCH", "live_mode": "🔴 LIVE SIMULATION",
        "btn_analyze": "START ANALYSIS", "loading": "Processing Quantum...", "ai_decision": "AI DECISION",
        "ticket_title": "ENSEMBLE PREDICTION", "score": "SCORE", "confidence": "CONFIDENCE",
        "win": "WINS", "draw": "DRAW", "play": "PLAY", "pass": "PASS",
        "high": "HIGH", "med": "MEDIUM", "low": "LOW", "market_dc": "DOUBLE CHANCE", "market_ou": "OVER / UNDER",
        "market_other": "OTHERS", "team_goals": "TEAM GOALS", "home_goal": "Home G.", "away_goal": "Away G.",
        "ai_reason": "AI Reason", "power_balance": "Power Balance", "minute": "Minute", 
        "disclaimer": "⚠️ DISCLAIMER: Statistical analysis only. Not financial advice. 18+",
        "warn_key": "⚠️ Missing API Key! Please enter it in the sidebar.", "warn_no_data": "⚠️ No data found.",
        "r_power": "Base power difference & home advantage.", "r_live": "Simulated based on live score & time.",
        "r_var": "High variance detected.", "r_elo": "Critical match, ELO model prioritized."
    }
}
# Fallback
for l in ["de", "fr", "ar"]: TRANSLATIONS[l] = TRANSLATIONS["en"]

LEAGUES = {
    "🇬🇧 Premier League": "PL", "🇹🇷 Süper Lig": "TR1", "🇪🇸 La Liga": "PD",
    "🇩🇪 Bundesliga": "BL1", "🇮🇹 Serie A": "SA", "🇫🇷 Ligue 1": "FL1",
    "🇳🇱 Eredivisie": "DED", "🇪🇺 Champions League": "CL"
}

# -----------------------------------------------------------------------------
# 4. DATA & ENGINE CLASSES
# -----------------------------------------------------------------------------
class SecureDataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
        self.base_url = "https://api.football-data.org/v4"

    @st.cache_data(ttl=CONFIG["CACHE_TTL"])
    def fetch_data(_self, code):
        try:
            r1 = requests.get(f"{_self.base_url}/competitions/{code}/standings", headers=_self.headers)
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=CONFIG["LOOKAHEAD_DAYS"])).strftime("%Y-%m-%d")
            r2 = requests.get(f"{_self.base_url}/competitions/{code}/matches", headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            if r1.status_code != 200: return None
            return {"standings": r1.json(), "matches": r2.json()}
        except: return None

    @staticmethod
    def match_team_name(target, names):
        if target in names: return target
        m = get_close_matches(target, names, n=1, cutoff=0.5)
        return m[0] if m else None

    @staticmethod
    def generate_smart_form(points, played):
        if played == 0: return "D,D,D,D,D"
        ppg = points / played
        w = [0.75, 0.2, 0.05] if ppg >= 2.0 else [0.3, 0.4, 0.3] if ppg >= 1.2 else [0.1, 0.3, 0.6]
        return ",".join(np.random.choice(["W","D","L"], 5, p=w))

class QuantumEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run(self, home, away, stats, avg_goals, live_data=None):
        h = SecureDataManager.match_team_name(home, list(stats.keys()))
        a = SecureDataManager.match_team_name(away, list(stats.keys()))
        if not h or not a: return None
        hs, as_ = stats[h], stats[a]

        h_xg = hs["att"] * as_["def"] * avg_goals + (CONFIG["HOME_ADV_BASE"] * hs["home_factor"])
        a_xg = as_["att"] * hs["def"] * avg_goals * as_["away_factor"] * CONFIG["AWAY_PENALTY"]
        h_xg *= hs["form_val"]; a_xg *= as_["form_val"]

        curr_h, curr_a, minute = 0, 0, 0
        if live_data:
            minute = live_data['minute']
            curr_h, curr_a = live_data['h_score'], live_data['a_score']
            rem = max(0, (90 - minute) / 90)
            h_xg *= rem; a_xg *= rem
            if curr_h < curr_a: h_xg *= 1.35
            if curr_a < curr_h: a_xg *= 1.35

        hg = self.rng.poisson(h_xg, CONFIG["SIM_COUNT"]) + curr_h
        ag = self.rng.poisson(a_xg, CONFIG["SIM_COUNT"]) + curr_a
        
        p1, px, p2 = np.mean(hg > ag)*100, np.mean(hg == ag)*100, np.mean(hg < ag)*100

        if not live_data:
            p_diff = hs["power"] - as_["power"]
            p1_e = 100 / (1 + np.exp(-(p_diff + 25) / 35))
            p2_e = 100 / (1 + np.exp((p_diff - 25) / 35))
            w = 0.6 if abs(p_diff) >= 12 else 0.4
            p1 = (p1*w) + (p1_e*(1-w)); p2 = (p2*w) + (p2_e*(1-w))
            tot = p1+px+p2; p1,px,p2 = p1/tot*100, px/tot*100, p2/tot*100

        probs_n = np.array([p1, px, p2]) / 100
        ent = -np.sum(probs_n * np.log(probs_n + 1e-9))
        u, c = np.unique(hg * 100 + ag, return_counts=True)
        best = u[np.argmax(c)]
        
        rc = "r_live" if live_data else "r_var" if ent > 1.05 else "r_elo" if abs(hs["power"] - as_["power"]) < 12 else "r_power"
        return {
            "p1": p1, "px": px, "p2": p2, "score_h": best // 100, "score_a": best % 100,
            "conf": max(p1, px, p2), "entropy": ent, "reason_code": rc,
            "dc": {"1X": p1+px, "X2": px+p2}, "goals": {"o25": np.mean((hg+ag)>2.5)*100, "btts": np.mean((hg>curr_h)&(ag>curr_a))*100},
            "team_goals": {"h": np.mean(hg>curr_h)*100, "a": np.mean(ag>curr_a)*100},
            "names": {"h": h, "a": a}, "logos": {"h": hs.get("crest") or CONFIG["DEFAULT_LOGO"], "a": as_.get("crest") or CONFIG["DEFAULT_LOGO"]},
            "forms": {"h": hs["form_str"], "a": as_["form_str"]}, "power": {"h": hs["power"], "a": as_["power"]}
        }

# -----------------------------------------------------------------------------
# 5. UI COMPONENTS
# -----------------------------------------------------------------------------
def render_badges(form):
    f = form.replace(',', '')[-5:]
    html = "<div class='form-badges'>"
    for c in f:
        html += f"<div class='badge badge-{c}'>{c.replace('W','G').replace('L','M')}</div>"
    return html + "</div>"

def main():
    # --- RENDER SAFE API KEY LOGIC ---
    api_key = os.environ.get("FOOTBALL_API_KEY") # Önce Render Environment
    if not api_key:
        try: api_key = st.secrets["FOOTBALL_API_KEY"] # Sonra Local Secrets
        except: pass

    with st.sidebar:
        st.title("⚙️ Config")
        lang = st.selectbox("Language", list(LANGUAGES.keys()), format_func=lambda x: f"{LANGUAGES[x]['flag']} {LANGUAGES[x]['name']}")
        t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
        if not api_key:
            api_key = st.text_input("🔑 API Key Required", type="password")
            if not api_key: st.warning(t['warn_key']); st.stop()

    if LANGUAGES[lang]['dir'] == 'rtl': st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)
    st.markdown(f"<div class='quantum-title'>{t['title']}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1: l_key = st.selectbox(t['sel_league'], list(LEAGUES.keys()))
    
    manager = SecureDataManager(api_key)
    with st.spinner(t['loading']): data = manager.fetch_data(LEAGUES[l_key])

    if not data or not data.get('matches'): st.warning(t['warn_no_data']); return

    table = data["standings"]["standings"][0]["table"]
    stats = {}
    tg = sum(r["goalsFor"] for r in table); tp = sum(r["playedGames"] for r in table)
    avg_g = tg/tp if tp else 1.5

    for row in table:
        n, p, pts = row["team"]["name"], row["playedGames"], row["points"]
        f = row.get("form") or SecureDataManager.generate_smart_form(pts, p)
        fv = np.mean([{"W":1.1,"D":1.0,"L":0.9}.get(x, 1.0) for x in f.replace(",","")])
        stats[n] = {
            "att": (row["goalsFor"]/p)/avg_g if p else 1, "def": (row["goalsAgainst"]/p)/avg_g if p else 1,
            "form_val": fv, "form_str": f, "power": 100 + (pts/p*20) + ((row["goalsFor"]-row["goalsAgainst"])/p*5) + (fv*10),
            "home_factor": 1.12, "away_factor": 0.88, "crest": row["team"].get("crest")
        }

    matches = {f'{m["homeTeam"]["name"]} - {m["awayTeam"]["name"]}': m for m in data["matches"]["matches"] if m["status"] in ["SCHEDULED", "TIMED", "UPCOMING"]}
    if not matches: st.warning(t['warn_no_data']); return
    with c2: game = st.selectbox(t['sel_match'], list(matches.keys()))

    live_on = st.toggle(t['live_mode'], value=False)
    live_data = None
    if live_on:
        lc1, lc2, lc3 = st.columns(3)
        with lc1: hs = st.number_input(t['home_goal'], 0, 10, 0)
        with lc2: mi = st.slider(t['minute'], 0, 90, 45)
        with lc3: as_ = st.number_input(t['away_goal'], 0, 10, 0)
        live_data = {"minute": mi, "h_score": hs, "a_score": as_}

    if st.button(t['btn_analyze'], use_container_width=True):
        m = matches[game]
        bar = st.progress(0)
        for i in range(20): time.sleep(0.01); bar.progress((i+1)*5)
        bar.empty()
        st.session_state.q_data = QuantumEngine().run(m["homeTeam"]["name"], m["awayTeam"]["name"], stats, avg_g, live_data)

    if st.session_state.q_data:
        res = st.session_state.q_data
        dt = f"{t['pass']} (Risk)" if res["entropy"] > 1.05 else f"{t['pass']} ({t['low']})" if res["conf"] < 38 else t['play']
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'><span class='{'decision-play' if t['play'] in dt else 'decision-pass'}'>{t['ai_decision']}: {dt}</span></div>", unsafe_allow_html=True)
        
        mp = f"{res['names']['h']} {t['win']}" if res['p1']>res['p2'] and res['p1']>res['px'] else f"{res['names']['a']} {t['win']}" if res['p2']>res['p1'] and res['p2']>res['px'] else t['draw']
        st.markdown(f"<div class='ticket-container'><div style='color:#aaa; font-size:0.9rem;'>{t['ticket_title']}</div><div class='main-pred'>{mp}</div><div class='ticket-stats'><div>{t['score']}: {res['score_h']}-{res['score_a']}</div><div style='color:#00ff88'>{t['confidence']}: %{res['conf']:.1f}</div></div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='team-header'><img src='{res['logos']['h']}' class='team-logo'><div class='team-name'>{res['names']['h']}</div></div>", unsafe_allow_html=True)
            st.markdown(render_badges(res['forms']['h']), unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='team-header'><div class='team-name'>{res['names']['a']}</div><img src='{res['logos']['a']}' class='team-logo'></div>", unsafe_allow_html=True)
            st.markdown(render_badges(res['forms']['a']), unsafe_allow_html=True)

        st.markdown("---")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='market-box'><div class='market-title'>🛡️ {t['market_dc']}</div><div class='market-row'><span>1X</span><span class='prob-high'>%{res['dc']['1X']:.1f}</span></div><div class='market-row'><span>X2</span><span class='prob-med'>%{res['dc']['X2']:.1f}</span></div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-box'><div class='market-title'>🥅 {t['market_ou']}</div><div class='market-row'><span>2.5 {t['high']}</span><span class='{'prob-high' if res['goals']['o25']>55 else 'prob-low'}'>%{res['goals']['o25']:.1f}</span></div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='market-box'><div class='market-title'>⚽ {t['team_goals']}</div><div class='market-row'><span>{t['home_goal']} > 0.5</span><span class='prob-high'>%{res['team_goals']['h']:.1f}</span></div><div class='market-row'><span>{t['away_goal']} > 0.5</span><span class='prob-med'>%{res['team_goals']['a']:.1f}</span></div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-box'><div class='market-title'>🔥 {t['market_other']}</div><div class='market-row'><span>KG VAR</span><span class='{'prob-high' if res['goals']['btts']>55 else 'prob-low'}'>%{res['goals']['btts']:.1f}</span></div></div>", unsafe_allow_html=True)

        st.info(f"🧠 **{t['ai_reason']}:** {t.get(res['reason_code'], '...')}")
        st.plotly_chart(create_radar(res, stats, t), use_container_width=True)

    st.markdown(f"<div class='disclaimer'>{t['disclaimer']}<br>Render Free Tier: Initial load may take 1-2 mins.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
