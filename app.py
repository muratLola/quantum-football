import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# -----------------------------------------------------------------------------
# 1. KONFİGÜRASYON
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "API_URL": "https://api.football-data.org/v4",
    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"}
}

st.set_page_config(page_title="Quantum Football xG", page_icon="⚽", layout="wide")

# -----------------------------------------------------------------------------
# 2. DİL VE ARAYÜZ METİNLERİ
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "tr": {
        "app_title": "QUANTUM FOOTBALL xG",
        "settings": "Analiz Ayarları",
        "api_ph": "Football-Data API Key",
        "sim_param": "Understat xG Simülasyonu",
        "match_count": "Simülasyon Sayısı",
        "form_set": "Takım Formu / Şans Faktörü",
        "missing_p": "Eksik Kilit Oyuncu (Sakat/Cezalı)",
        "h_miss": "Ev Sahibi Eksik",
        "a_miss": "Deplasman Eksik",
        "h_att": "Ev Sahibi Gücü",
        "a_att": "Deplasman Gücü",
        "league": "Lig Seçimi",
        "match": "Maç Seçimi",
        "start_btn": "XG ANALİZİNİ BAŞLAT",
        "calculating": "Understat verileri ve Poisson motoru çalışıyor...",
        "xg": "Beklenen Gol (xG)",
        "home": "EV SAHİBİ", "draw": "BERABERLİK", "away": "DEPLASMAN",
        "heatmap": "Skor Olasılık Matrisi",
        "top_scores": "En Olası Skorlar",
        "ht_ft": "İY/MS (HT/FT) Dağılımı",
        "total_goal": "Toplam Gol Beklentisi",
        "no_match": "Bu ligde yakında maç bulunamadı.",
        "footer": "Quantum Football v54.0 | Understat xG Logic | Yatırım tavsiyesi değildir."
    },
    "en": {
        "app_title": "QUANTUM FOOTBALL xG",
        "settings": "Analysis Settings",
        "api_ph": "Enter API Key",
        "sim_param": "Understat xG Simulation",
        "match_count": "Simulation Count",
        "form_set": "Team Form / Luck Factor",
        "missing_p": "Missing Key Players",
        "h_miss": "Home Missing",
        "a_miss": "Away Missing",
        "h_att": "Home Strength",
        "a_att": "Away Strength",
        "league": "Select League",
        "match": "Select Match",
        "start_btn": "START XG ANALYSIS",
        "calculating": "Running Understat logic & Poisson engine...",
        "xg": "Expected Goals (xG)",
        "home": "HOME WIN", "draw": "DRAW", "away": "AWAY WIN",
        "heatmap": "Score Probability Matrix",
        "top_scores": "Most Likely Scores",
        "ht_ft": "HT/FT Distribution",
        "total_goal": "Total Goal Expectancy",
        "no_match": "No upcoming matches found.",
        "footer": "Quantum Football v54.0 | Understat xG Logic | For statistical use only."
    }
}

# -----------------------------------------------------------------------------
# 3. CSS STİLİ
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc;}
    .main-title {
        font-family: 'Roboto Mono', monospace; font-size: 3rem; font-weight: 800;
        text-align: center; margin-bottom: 10px;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-card {
        background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 15px;
        text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stat-val { font-size: 2rem; font-weight: 700; color: #fff; }
    .stat-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .score-row { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #334155; font-family: 'Roboto Mono'; }
    .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #334155; text-align: center; color: #64748b; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DATA MANAGER (Understat xG Manuel Enjeksiyonlu)
# -----------------------------------------------------------------------------
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'match_info' not in st.session_state: st.session_state.match_info = None

TEAM_LOGOS = {
    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",
    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",
    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",
    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",
}

class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    @st.cache_data(ttl=3600)
    def fetch_data(_self, league_code):
        standings_data = {"standings": [{"table": []}]}
        matches_data = {"matches": []}
        try:
            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=_self.headers)
            if r1.status_code == 200: standings_data = r1.json()
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", 
                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            if r2.status_code == 200: matches_data = r2.json()
        except: pass
        return standings_data, matches_data

# -----------------------------------------------------------------------------
# 5. SIMULATION ENGINE (Understat xG Modeli Entegre Edildi)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        # UNDERSTAT MANTIĞI: gf (atılan gol) yerine xG_for, ga (yenen gol) yerine xG_against kullanımı
        # Eğer Understat verisi manuel enjekte edilmemişse standart gf/ga kullanılır.
        h_att_base = h_stats.get('xg_f', h_stats['gf'])
        h_def_base = h_stats.get('xg_a', h_stats['ga'])
        a_att_base = a_stats.get('xg_f', a_stats['gf'])
        a_def_base = a_stats.get('xg_a', a_stats['ga'])

        h_attack = (h_att_base / avg_g) * params['h_att_factor']
        h_def = (h_def_base / avg_g)
        a_attack = (a_att_base / avg_g) * params['a_att_factor']
        a_def = (a_def_base / avg_g)

        # Poisson Oranları
        lambda_h = h_attack * a_def * avg_g * params['home_adv']
        lambda_a = a_attack * h_def * avg_g

        # Sakatlık/Eksik Oyuncu Düzeltmesi (Her eksik %12 güç kaybı)
        if params['h_missing'] > 0: lambda_h *= (1 - (params['h_missing'] * 0.12))
        if params['a_missing'] > 0: lambda_a *= (1 - (params['a_missing'] * 0.12))

        sims = params['sim_count']
        # HT/FT ve Skor Simülasyonu
        h_ht = self.rng.poisson(lambda_h * 0.45, sims)
        a_ht = self.rng.poisson(lambda_a * 0.45, sims)
        h_ft = h_ht + self.rng.poisson(lambda_h * 0.55, sims)
        a_ft = a_ht + self.rng.poisson(lambda_a * 0.55, sims)

        return {"h": h_ft, "a": a_ft, "ht": (h_ht, a_ht), "xg": (lambda_h, lambda_a), "sims": sims}

    def analyze(self, data):
        h, a = data["h"], data["a"]
        sims = data["sims"]
        p_h, p_d, p_a = np.mean(h > a)*100, np.mean(h == a)*100, np.mean(h < a)*100
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100

        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:7]

        # HT/FT Dağılımı
        h_ht, a_ht = data["ht"]
        res_ht = np.where(h_ht > a_ht, "1", np.where(h_ht < a_ht, "2", "X"))
        res_ft = np.where(h > a, "1", np.where(h < a, "2", "X"))
        htft_list = [f"{i}/{j}" for i, j in zip(res_ht, res_ft)]
        u_htft, c_htft = np.unique(htft_list, return_counts=True)
        htft_final = sorted(zip(u_htft, c_htft/sims*100), key=lambda x: x[1], reverse=True)[:5]

        return {"1x2": [p_h, p_d, p_a], "matrix": matrix, "top_scores": top, "xg": data["xg"], "htft": htft_final}

# -----------------------------------------------------------------------------
# 6. APP MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        lang = st.selectbox("Language", ["tr", "en"])
        t = TRANSLATIONS[lang]
        st.divider()
        api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
        if not api_key:
            api_key = st.text_input(t['api_ph'], type="password")
            if not api_key: st.stop()
        
        sim_count = st.select_slider(t['match_count'], [10000, 100000, 500000], 100000)
        h_att = st.slider(t['h_att'], 80, 120, 100) / 100
        a_att = st.slider(t['a_att'], 80, 120, 100) / 100
        
        c1, c2 = st.columns(2)
        h_miss = c1.number_input(t['h_miss'], 0, 5, 0)
        a_miss = c2.number_input(t['a_miss'], 0, 5, 0)
        
    st.markdown(f"<div class='main-title'>{t['app_title']}</div>", unsafe_allow_html=True)
    dm = DataManager(api_key)

    L_MAP = {
        "Süper Lig & Kupa": "TR1", "Premier League": "PL", "La Liga": "PD", 
        "Bundesliga": "BL1", "Serie A": "SA", "Ligue 1": "FL1", 
        "Eredivisie": "DED", "Champions League": "CL"
    }
    
    col1, col2 = st.columns([1, 2])
    with col1: league = st.selectbox(t['league'], list(L_MAP.keys()))
    
    standings, fixtures = dm.fetch_data(L_MAP[league])
    if not fixtures: st.info(t['no_match']); st.stop()

    # --- UNDERSTAT xG GÜÇ ENJEKSİYONU ---
    # Bu verileri Understat.com'dan güncel xG rakamlarıyla doldurabilirsin
    # Örn: gf/ga yerine xG_For/xG_Against
    UNDERSTAT_DATA = {
        2054: {"xg_f": 2.45, "xg_a": 0.85}, # Galatasaray
        2052: {"xg_f": 2.52, "xg_a": 0.78}, # Fenerbahçe
        2036: {"xg_f": 2.05, "xg_a": 1.15}, # Beşiktaş
        8: {"xg_f": 2.10, "xg_a": 0.90},    # Örn: Man City
    }

    table = standings.get("standings", [{"table": []}])[0].get("table", [])
    teams = {}
    avg_l = 2.5
    if table:
        avg_l = sum(x["goalsFor"] for x in table) / sum(x["playedGames"] for x in table)
        for r in table:
            t_id = r["team"]["id"]
            teams[t_id] = {
                "name": r["team"]["name"], "crest": r["team"].get("crest"),
                "gf": r["goalsFor"]/r["playedGames"], "ga": r["goalsAgainst"]/r["playedGames"]
            }
            # Understat verisi varsa üzerine yaz
            if t_id in UNDERSTAT_DATA: teams[t_id].update(UNDERSTAT_DATA[t_id])

    match_dict = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})": m 
                  for m in fixtures.get("matches", []) if m["status"] in ["SCHEDULED", "TIMED"]}
    
    if not match_dict: st.info(t['no_match']); st.stop()
    with col2: sel_match = st.selectbox(t['match'], list(match_dict.keys()))

    if st.button(t['start_btn'], use_container_width=True):
        m = match_dict[sel_match]
        h_id, a_id = m["homeTeam"]["id"], m["awayTeam"]["id"]
        
        h_team = teams.get(h_id, {"name": m["homeTeam"]["name"], "gf": 1.5, "ga": 1.2, "crest": CONFIG["DEFAULT_LOGO"]})
        a_team = teams.get(a_id, {"name": m["awayTeam"]["name"], "gf": 1.3, "ga": 1.4, "crest": CONFIG["DEFAULT_LOGO"]})

        engine = SimulationEngine()
        with st.spinner(t['calculating']):
            raw = engine.run_monte_carlo(h_team, a_team, avg_l, 
                                        {"sim_count": sim_count, "h_att_factor": h_att, "a_att_factor": a_att, 
                                         "h_missing": h_miss, "a_missing": a_miss, "home_adv": 1.15})
            res = engine.analyze(raw)

        # SONUÇ GÖSTERİMİ
        c_h, c_vs, c_a = st.columns([2,1,2])
        with c_h: st.markdown(f"<div style='text-align:center'><img src='{h_team['crest']}' width='80'><h3>{h_team['name']}</h3></div>", unsafe_allow_html=True)
        with c_vs: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric(t['xg'], f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with c_a: st.markdown(f"<div style='text-align:center'><img src='{a_team['crest']}' width='80'><h3>{a_team['name']}</h3></div>", unsafe_allow_html=True)

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['home']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['draw']}</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['away']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)

        ca, cb = st.columns([2,1])
        with ca:
            fig_heat = go.Figure(data=go.Heatmap(z=res["matrix"], x=[0,1,2,3,4,5], y=[0,1,2,3,4,5], colorscale='Magma', texttemplate="%{z:.1f}%"))
            fig_heat.update_layout(title=t['heatmap'], paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
        with cb:
            st.markdown(f"### 🎯 {t['top_scores']}")
            for sc, pr in res["top_scores"]:
                st.markdown(f"<div class='score-row'><b>{sc}</b> <span style='color:#38bdf8'>%{pr:.1f}</span></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
