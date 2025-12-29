import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from difflib import get_close_matches

# -----------------------------------------------------------------------------
# 1. KONFIGÜRASYON & SABİTLER
# -----------------------------------------------------------------------------
CONFIG = {
    "SIM_COUNT": 15000,
    "HOME_ADV_BASE": 0.35,
    "AWAY_PENALTY": 0.90,
    "CACHE_TTL": 3600,
    "LOOKAHEAD_DAYS": 30,
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png" # Yedek Logo
}

st.set_page_config(
    page_title="Quantum AI v36",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. OTURUM YÖNETİMİ (SESSION STATE)
# -----------------------------------------------------------------------------
if 'q_data' not in st.session_state:
    st.session_state.q_data = None

# -----------------------------------------------------------------------------
# 3. CSS & STİL (PREMIUM DARK UI)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
    
    /* BAŞLIK */
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
    
    /* LOGO & TEAM HEADER */
    .team-header { display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 15px; }
    .team-logo { width: 60px; height: 60px; object-fit: contain; filter: drop-shadow(0 0 8px rgba(255,255,255,0.2)); transition: transform 0.3s; }
    .team-logo:hover { transform: scale(1.1); }
    .team-name { font-size: 1.4rem; font-weight: 800; color: #f1f5f9; }
    
    /* SKORBOARD */
    .scoreboard { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; padding: 20px; border-radius: 16px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    
    /* TICKET */
    .ticket-container { background: radial-gradient(circle at center, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; border-top: 4px solid #00ff88; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.6); margin-bottom: 25px; max-width: 800px; margin-left: auto; margin-right: auto; }
    .main-pred { font-size: 3rem; font-weight: 900; color: #facc15; margin: 15px 0; letter-spacing: -1px; text-shadow: 0 0 15px rgba(250, 204, 21, 0.4); }
    
    /* MARKETS */
    .market-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: all 0.2s; }
    .market-box:hover { border-color: #00ff88; transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0, 255, 136, 0.1); }
    .market-title { color: #00ff88; font-weight: bold; font-size: 0.9rem; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .market-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.95rem; color: #e5e7eb; }
    
    .prob-high { color: #4ade80; font-weight: 800; } 
    .prob-med { color: #facc15; font-weight: 600; } 
    .prob-low { color: #f87171; } 

    /* BADGES */
    .form-badges { display: flex; gap: 4px; justify-content: center; margin-top: 5px; }
    .badge { width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #000; font-size: 0.75rem; }
    .badge-W { background-color: #4ade80; } .badge-D { background-color: #facc15; } .badge-L { background-color: #f87171; }
    
    .decision-play { background-color: #238636; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }
    .decision-pass { background-color: #da3633; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }

    /* RTL & ANIMATION */
    .rtl { direction: rtl; text-align: right; }
    .rtl .market-row { flex-direction: row-reverse; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); } }
    .live-badge { background-color: #ff5252; color: white; padding: 3px 8px; border-radius: 10px; font-weight: bold; font-size: 0.7rem; animation: pulse-red 2s infinite; vertical-align: middle; }
    
    /* FOOTER */
    .disclaimer { font-size: 0.75rem; color: #64748b; text-align: center; margin-top: 50px; border-top: 1px solid #334155; padding-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ÇEVİRİ MERKEZİ
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
        "btn_analyze": "ANALİZİ BAŞLAT", "loading": "Quantum Çekirdeği İşleniyor...", "ai_decision": "AI KARARI",
        "ticket_title": "ENSEMBLE TAHMİNİ", "score": "SKOR", "confidence": "GÜVEN",
        "win": "KAZANIR", "draw": "BERABERLİK", "play": "OYNA", "pass": "PAS GEÇ",
        "high": "YÜKSEK", "med": "ORTA", "low": "DÜŞÜK", "market_dc": "ÇİFTE ŞANS", "market_ou": "ALT / ÜST",
        "market_other": "DİĞER", "team_goals": "TAKIM GOLLERİ", "home_goal": "Ev Gol", "away_goal": "Dep Gol",
        "ai_reason": "AI Nedeni", "power_balance": "Güç Dengesi", "live_control": "Dakika & Skor Ayarı",
        "minute": "Dakika", "disclaimer": "⚠️ YASAL UYARI: İstatistiksel veri analizidir. Bahis tavsiyesi değildir. 18+",
        "warn_key": "⚠️ API Anahtarı Eksik! Lütfen sol menüden giriniz.", "warn_no_data": "⚠️ Veri bulunamadı.",
        "r_power": "Güç farkı ve saha avantajı belirleyici.", "r_live": "Canlı skor ve dakikaya göre simüle edildi.",
        "r_var": "Yüksek belirsizlik (Varyans) tespit edildi.", "r_elo": "Kritik maç, ELO modeli önceliklendirildi."
    },
    "en": {
        "title": "QUANTUM AI v36", "sel_league": "SELECT LEAGUE", "sel_match": "SELECT MATCH", "live_mode": "🔴 LIVE SIMULATION",
        "btn_analyze": "START ANALYSIS", "loading": "Processing Quantum Core...", "ai_decision": "AI DECISION",
        "ticket_title": "ENSEMBLE PREDICTION", "score": "SCORE", "confidence": "CONFIDENCE",
        "win": "WINS", "draw": "DRAW", "play": "PLAY", "pass": "PASS",
        "high": "HIGH", "med": "MEDIUM", "low": "LOW", "market_dc": "DOUBLE CHANCE", "market_ou": "OVER / UNDER",
        "market_other": "OTHERS", "team_goals": "TEAM GOALS", "home_goal": "Home G.", "away_goal": "Away G.",
        "ai_reason": "AI Reason", "power_balance": "Power Balance", "live_control": "Time & Score Control",
        "minute": "Minute", "disclaimer": "⚠️ DISCLAIMER: Statistical analysis only. Not financial advice. 18+",
        "warn_key": "⚠️ Missing API Key! Please enter it in the sidebar.", "warn_no_data": "⚠️ No data found.",
        "r_power": "Base power difference & home advantage.", "r_live": "Simulated based on live score & time.",
        "r_var": "High variance detected.", "r_elo": "Critical match, ELO model prioritized."
    }
}
TRANSLATIONS["de"] = TRANSLATIONS["en"]
TRANSLATIONS["fr"] = TRANSLATIONS["en"]
TRANSLATIONS["ar"] = TRANSLATIONS["en"] 

LEAGUES = {
    "🇬🇧 Premier League": "PL", "🇹🇷 Süper Lig": "TR1", "🇪🇸 La Liga": "PD",
    "🇩🇪 Bundesliga": "BL1", "🇮🇹 Serie A": "SA", "🇫🇷 Ligue 1": "FL1",
    "🇳🇱 Eredivisie": "DED", "🇪🇺 Champions League": "CL"
}

# -----------------------------------------------------------------------------
# 5. SINIFLAR (OOP MİMARİSİ)
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
            r2 = requests.get(
                f"{_self.base_url}/competitions/{code}/matches",
                headers=_self.headers,
                params={"dateFrom": today, "dateTo": future}
            )
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
        if ppg >= 2.2: w = [0.75, 0.2, 0.05]
        elif ppg >= 1.6: w = [0.55, 0.25, 0.2]
        elif ppg >= 1.2: w = [0.35, 0.35, 0.3]
        else: w = [0.15, 0.25, 0.6]
        return ",".join(np.random.choice(["W","D","L"], 5, p=w))

class QuantumEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run(self, home, away, stats, avg_goals, live_data=None):
        h = SecureDataManager.match_team_name(home, list(stats.keys()))
        a = SecureDataManager.match_team_name(away, list(stats.keys()))
        if not h or not a: return None

        hs, as_ = stats[h], stats[a]

        # PRE-MATCH xG
        h_xg = hs["att"] * as_["def"] * avg_goals + (CONFIG["HOME_ADV_BASE"] * hs["home_factor"])
        a_xg = as_["att"] * hs["def"] * avg_goals * as_["away_factor"] * CONFIG["AWAY_PENALTY"]
        h_xg *= hs["form_val"]
        a_xg *= as_["form_val"]

        # LIVE MODIFICATIONS
        current_h_goals, current_a_goals = 0, 0
        minute = 0
        
        if live_data:
            minute = live_data['minute']
            current_h_goals = live_data['h_score']
            current_a_goals = live_data['a_score']
            remaining_ratio = max(0, (90 - minute) / 90)
            h_xg *= remaining_ratio
            a_xg *= remaining_ratio
            
            if current_h_goals < current_a_goals: h_xg *= 1.35
            if current_a_goals < current_h_goals: a_xg *= 1.35

        # MONTE CARLO SIMULATION
        hg_rem = self.rng.poisson(h_xg, CONFIG["SIM_COUNT"])
        ag_rem = self.rng.poisson(a_xg, CONFIG["SIM_COUNT"])
        hg = hg_rem + current_h_goals
        ag = ag_rem + current_a_goals
        
        p1 = np.mean(hg > ag) * 100
        px = np.mean(hg == ag) * 100
        p2 = np.mean(hg < ag) * 100

        # ELO INTEGRATION (Only if not live)
        if not live_data:
            power_diff = hs["power"] - as_["power"]
            p1_elo = 1 / (1 + np.exp(-(power_diff + 25) / 35))
            p2_elo = 1 / (1 + np.exp((power_diff - 25) / 35))
            px_elo = 1 - (p1_elo + p2_elo)
            if px_elo < 0: px_elo = 0.1
            
            w_pois = 0.6 if abs(power_diff) >= 12 else 0.4
            w_elo = 1.0 - w_pois
            
            p1 = (p1 * w_pois) + (p1_elo * 100 * w_elo)
            p2 = (p2 * w_pois) + (p2_elo * 100 * w_elo)
            px = (px * w_pois) + (px_elo * 100 * w_elo)
            
            total = p1 + p2 + px
            p1, p2, px = (p1/total)*100, (p2/total)*100, (px/total)*100

        conf = max(p1, px, p2)
        
        # Entropy
        probs_norm = np.array([p1, px, p2]) / 100
        entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-9))
        
        # Score Logic
        score_hash = hg * 100 + ag
        u, c = np.unique(score_hash, return_counts=True)
        best = u[np.argmax(c)]
        
        total_goals = hg + ag
        o25 = np.mean(total_goals > 2.5) * 100
        btts = np.mean((hg > 0) & (ag > 0)) * 100
        h_o05 = np.mean(hg > 0.5) * 100
        a_o05 = np.mean(ag > 0.5) * 100
        
        reason_code = "r_power"
        if live_data: reason_code = "r_live"
        elif entropy > 1.05: reason_code = "r_var"
        elif not live_data and abs(hs["power"] - as_["power"]) < 12: reason_code = "r_elo"

        # LOGO CHECK
        h_logo = hs.get("crest") if hs.get("crest") else CONFIG["DEFAULT_LOGO"]
        a_logo = as_.get("crest") if as_.get("crest") else CONFIG["DEFAULT_LOGO"]

        return {
            "p1": p1, "px": px, "p2": p2, 
            "score_h": best // 100, "score_a": best % 100,
            "conf": conf, "entropy": entropy, "reason_code": reason_code,
            "dc": {"1X": p1+px, "X2": px+p2},
            "goals": {"o25": o25, "btts": btts},
            "team_goals": {"h": h_o05, "a": a_o05},
            "names": {"h": h, "a": a},
            "logos": {"h": h_logo, "a": a_logo},
            "forms": {"h": hs["form_str"], "a": as_["form_str"]},
            "power": {"h": hs["power"], "a": as_["power"]}
        }

# -----------------------------------------------------------------------------
# 6. UI FUNCTIONS
# -----------------------------------------------------------------------------
def render_form_badges(form_str):
    form_str = form_str.replace(',', '')
    html = "<div class='form-badges'>"
    for char in form_str[-5:]:
        bg = "badge-W" if char == 'W' else "badge-D" if char == 'D' else "badge-L"
        html += f"<div class='badge {bg}'>{char}</div>"
    html += "</div>"
    return html

def create_radar(res, stats, t):
    h, a = res['names']['h'], res['names']['a']
    h_val = [stats[h]['att']*50, stats[h]['def']*40, stats[h]['power']*0.8]
    a_val = [stats[a]['att']*50, stats[a]['def']*40, stats[a]['power']*0.8]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_val, theta=['Att', 'Def', 'Power'], fill='toself', name='Home', line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_val, theta=['Att', 'Def', 'Power'], fill='toself', name='Away', line_color='#facc15'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), margin=dict(l=20,r=20,t=20,b=20), height=250, paper_bgcolor='rgba(0,0,0,0)', font_color='white', title=dict(text=t['power_balance'], font=dict(color='#00ff88')))
    return fig

# -----------------------------------------------------------------------------
# 7. MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Config")
        lang_code = st.selectbox("Language / Dil", list(LANGUAGES.keys()), format_func=lambda x: f"{LANGUAGES[x]['flag']} {LANGUAGES[x]['name']}")
        t = TRANSLATIONS.get(lang_code, TRANSLATIONS["en"])
        
        api_key = st.secrets.get("FOOTBALL_API_KEY")
        if not api_key:
            api_key = st.text_input("🔑 API Key", type="password")
            if not api_key:
                st.warning(t['warn_key'])
                st.stop()
    
    # CSS Direction Update
    if LANGUAGES[lang_code]['dir'] == 'rtl':
        st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)

    st.markdown(f"<div class='quantum-title'>{t['title']}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1: league_key = st.selectbox(t['sel_league'], list(LEAGUES.keys()))
    
    manager = SecureDataManager(api_key)
    with st.spinner(t['loading']):
        data = manager.fetch_data(LEAGUES[league_key])

    if not data or not data.get('matches'):
        st.warning(t['warn_no_data'])
        return

    # Process Stats
    table = data["standings"]["standings"][0]["table"]
    stats = {}
    tg = sum(r["goalsFor"] for r in table)
    tp = sum(r["playedGames"] for r in table)
    avg_goals = tg/tp if tp else 1.5

    for row in table:
        name = row["team"]["name"]
        played = row["playedGames"]
        pts = row["points"]
        form = row.get("form") or SecureDataManager.generate_smart_form(pts, played)
        form_val = np.mean([{"W":1.1,"D":1.0,"L":0.9}.get(x, 1.0) for x in form.replace(",","")])
        gd = row["goalsFor"] - row["goalsAgainst"]
        power = 100 + (pts/played * 20) + (gd/played * 5) + (form_val * 10)
        
        # LOGO URL ALIMI
        crest = row["team"].get("crest", "")

        stats[name] = {
            "att": (row["goalsFor"]/played)/avg_goals if played else 1,
            "def": (row["goalsAgainst"]/played)/avg_goals if played else 1,
            "form_val": form_val, "form_str": form, "power": power,
            "home_factor": 1.12, "away_factor": 0.88, "crest": crest
        }

    matches = {f'{m["homeTeam"]["name"]} - {m["awayTeam"]["name"]}': m for m in data["matches"]["matches"] if m["status"] in ["SCHEDULED", "TIMED", "UPCOMING"]}
    if not matches: st.warning(t['warn_no_data']); return

    with c2: game = st.selectbox(t['sel_match'], list(matches.keys()))

    live_on = st.toggle(t['live_mode'], value=False)
    live_data = None
    if live_on:
        lc1, lc2, lc3 = st.columns([1,1,1])
        with lc1: h_score = st.number_input(f"{t['home_goal']}", 0, 10, 0)
        with lc2: minute = st.slider(f"{t['minute']}", 0, 90, 45)
        with lc3: a_score = st.number_input(f"{t['away_goal']}", 0, 10, 0)
        live_data = {"minute": minute, "h_score": h_score, "a_score": a_score}

    if st.button(t['btn_analyze'], use_container_width=True):
        m = matches[game]
        bar = st.progress(0)
        for i in range(50): time.sleep(0.003); bar.progress(i*2)
        bar.empty()

        engine = QuantumEngine()
        # Save raw result to session
        st.session_state.q_data = engine.run(m["homeTeam"]["name"], m["awayTeam"]["name"], stats, avg_goals, t, live_data)

    # --- RENDER RESULTS (Dinamik Dil Desteği ile) ---
    if st.session_state.q_data:
        res = st.session_state.q_data
        
        # Calculate Labels based on current language
        decision_txt = t['play']
        if res["entropy"] > 1.05: decision_txt = f"{t['pass']} (Risk)"
        elif res["conf"] < 38: decision_txt = f"{t['pass']} ({t['low']} {t['confidence']})"
        
        dec_cls = "decision-play" if t['play'] in decision_txt else "decision-pass"
        st.markdown(f"<div style='text-align:center; margin-bottom:15px;'><span class='{dec_cls}'>{t['ai_decision']}: {decision_txt}</span></div>", unsafe_allow_html=True)

        if res['p1'] > res['p2'] and res['p1'] > res['px']: main_pred = f"{res['names']['h']} {t['win']}"
        elif res['p2'] > res['p1'] and res['p2'] > res['px']: main_pred = f"{res['names']['a']} {t['win']}"
        else: main_pred = t['draw']

        st.markdown(f"""
        <div class="ticket-container">
            <div style="color:#aaa; font-size:0.9rem;">{t['ticket_title']}</div>
            <div class="main-pred">{main_pred}</div>
            <div style="display:flex; justify-content:center; gap:20px; color:white; font-family:monospace;">
                <div>{t['score']}: {res['score_h']}-{res['score_a']}</div>
                <div style="color:#00ff88">{t['confidence']}: %{res['conf']:.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='team-header'><img src="{res['logos']['h']}" class='team-logo'><div class='team-name'>{res['names']['h']}</div></div>""", unsafe_allow_html=True)
            st.markdown(render_form_badges(res['forms']['h']), unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='team-header'><div class='team-name'>{res['names']['a']}</div><img src="{res['logos']['a']}" class='team-logo'></div>""", unsafe_allow_html=True)
            st.markdown(render_form_badges(res['forms']['a']), unsafe_allow_html=True)

        st.markdown("---")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='market-box'><div class='market-title'>🛡️ {t['market_dc']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>1X</span> <span class='prob-high'>%{res['dc']['1X']:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>X2</span> <span class='prob-med'>%{res['dc']['X2']:.1f}</span></div></div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='market-box'><div class='market-title'>🥅 {t['market_ou']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>2.5 {t['high']}</span> <span class='{ 'prob-high' if res['goals']['o25']>55 else 'prob-low' }'>%{res['goals']['o25']:.1f}</span></div>", unsafe_allow_html=True)

        with m2:
            st.markdown(f"<div class='market-box'><div class='market-title'>⚽ {t['team_goals']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>{t['home_goal']} > 0.5</span> <span class='prob-high'>%{res['team_goals']['h']:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>{t['away_goal']} > 0.5</span> <span class='prob-med'>%{res['team_goals']['a']:.1f}</span></div></div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='market-box'><div class='market-title'>🔥 {t['market_other']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>KG VAR</span> <span class='{ 'prob-high' if res['goals']['btts']>55 else 'prob-low' }'>%{res['goals']['btts']:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>1</span> <span>%{res['p1']:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='market-row'><span>2</span> <span>%{res['p2']:.1f}</span></div></div>", unsafe_allow_html=True)

        # Dynamic Reason Translation
        reason_txt = t.get(res['reason_code'], res['reason_code'])
        st.info(f"🧠 **{t['ai_reason']}:** {reason_txt}")
        
        st.plotly_chart(create_radar(res, stats, t), use_container_width=True)

    st.markdown(f"<div class='disclaimer'>{t['disclaimer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
