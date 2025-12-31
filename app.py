import streamlit as st

import pandas as pd

import numpy as np

import requests

import plotly.graph_objects as go

import plotly.express as px

from datetime import datetime, timedelta

import time

import os



# -----------------------------------------------------------------------------

# 1. KONFİGÜRASYON

# -----------------------------------------------------------------------------

CONFIG = {

    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",

    "API_URL": "https://api.football-data.org/v4",

    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"}

}



st.set_page_config(page_title="Quantum Football", page_icon="⚽", layout="wide")



# -----------------------------------------------------------------------------

# 2. DİL VE ARAYÜZ METİNLERİ

# -----------------------------------------------------------------------------

TRANSLATIONS = {

    "tr": {

        "app_title": "QUANTUM FOOTBALL",

        "settings": "Ayarlar",

        "api_ph": "API Anahtarını Giriniz",

        "sim_param": "Simülasyon Ayarları",

        "match_count": "Simülasyon Sayısı",

        "form_set": "Takım Form Ayarları (Varsayılan: %100)",

        "missing_p": "Eksik Kilit Oyuncu Sayısı (Sakat/Cezalı)",

        "h_miss": "Ev Sahibi Eksik",

        "a_miss": "Deplasman Eksik",

        "h_att": "Ev Sahibi Gücü",

        "a_att": "Deplasman Gücü",

        "league": "Lig Seçimi",

        "match": "Maç Seçimi",

        "start_btn": "ANALİZİ BAŞLAT",

        "calculating": "Kuantum motoru maç simülasyonunu yapıyor...",

        "xg": "Beklenen Gol (xG)",

        "home": "EV SAHİBİ", "draw": "BERABERLİK", "away": "DEPLASMAN",

        "heatmap": "Skor Olasılık Matrisi",

        "top_scores": "En Olası Skorlar",

        "ht_ft": "İY/MS (HT/FT) Dağılımı",

        "total_goal": "Toplam Gol Beklentisi",

        "no_match": "Bu ligde/turnuvada yakında planlanmış maç bulunamadı.",

        "footer": "Quantum Football v53.0 © 2026 | Model: Monte Carlo & Poisson Dağılımı | Uyarı: Bu yazılım sadece istatistiksel ve bilimsel analiz amaçlıdır. Yatırım tavsiyesi değildir."

    },

    "en": {

        "app_title": "QUANTUM FOOTBALL",

        "settings": "Settings",

        "api_ph": "Enter API Key",

        "sim_param": "Simulation Settings",

        "match_count": "Simulation Count",

        "form_set": "Team Form Settings (Default: 100%)",

        "missing_p": "Missing Key Players (Injured/Suspended)",

        "h_miss": "Home Missing",

        "a_miss": "Away Missing",

        "h_att": "Home Strength",

        "a_att": "Away Strength",

        "league": "Select League",

        "match": "Select Match",

        "start_btn": "START ANALYSIS",

        "calculating": "Quantum engine running match simulation...",

        "xg": "Expected Goals (xG)",

        "home": "HOME WIN", "draw": "DRAW", "away": "AWAY WIN",

        "heatmap": "Score Probability Matrix",

        "top_scores": "Most Likely Scores",

        "ht_ft": "HT/FT Distribution",

        "total_goal": "Total Goal Expectancy",

        "no_match": "No upcoming matches found in this league/tournament.",

        "footer": "Quantum Football v53.0 © 2026 | Model: Monte Carlo & Poisson Distribution | Disclaimer: This tool is for statistical analysis only. Not financial advice."

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

    

    /* xG Değerinin Sığması İçin Düzeltme */

    div[data-testid="stMetricValue"] {

        white-space: nowrap;

        overflow: visible;

        text-overflow: clip;

        font-size: 2.5rem !important; 

    }

    

    .stat-card {

        background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 15px;

        text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);

    }

    .stat-val { font-size: 2rem; font-weight: 700; color: #fff; }

    .stat-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

    

    .analysis-box {

        background: rgba(30, 41, 59, 0.4); border: 1px solid #334155; 

        border-radius: 12px; padding: 15px; height: 100%;

    }

    

    .score-row {

        display: flex; justify-content: space-between; padding: 8px; 

        border-bottom: 1px solid #334155; font-family: 'Roboto Mono';

    }

    

    .footer {

        margin-top: 50px; padding-top: 20px; border-top: 1px solid #334155;

        text-align: center; color: #64748b; font-size: 0.8rem;

    }

    

    div[data-testid="stButton"] button { border-radius: 8px; }

    </style>

    """, unsafe_allow_html=True)



# -----------------------------------------------------------------------------

# 4. DATA MANAGER

# -----------------------------------------------------------------------------

if 'sim_results' not in st.session_state: st.session_state.sim_results = None

if 'match_info' not in st.session_state: st.session_state.match_info = None



# TÜM TAKIM LOGOLARI LİSTESİ (Yedek Linkler)

TEAM_LOGOS = {

    # 4 BÜYÜKLER

    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",

    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",

    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",

    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",

    

    # ANADOLU VE DİĞERLERİ

    2058: "https://upload.wikimedia.org/wikipedia/tr/e/e0/Samsunspor_logo_2.png",

    4503: "https://upload.wikimedia.org/wikipedia/tr/d/d3/%C4%B0stanbul_Ba%C5%9Fak%C5%9Fehir_FK.png",

    2044: "https://upload.wikimedia.org/wikipedia/tr/6/6c/Kasimpasa_logo.png",

    2037: "https://upload.wikimedia.org/wikipedia/tr/5/52/Antalyaspor_logo.png",

    2053: "https://upload.wikimedia.org/wikipedia/tr/9/9f/Sivasspor.png",

    2049: "https://upload.wikimedia.org/wikipedia/commons/1/1b/Konyaspor_Logo.png",

    2043: "https://upload.wikimedia.org/wikipedia/tr/0/00/Kayserispor_logosu.png",

    5553: "https://upload.wikimedia.org/wikipedia/tr/5/5f/Alanyaspor_logo.png",

    2051: "https://upload.wikimedia.org/wikipedia/tr/e/ee/Gaziantep_FK.png",

    5642: "https://upload.wikimedia.org/wikipedia/tr/9/9a/Hatayspor_logo.png",

    2055: "https://upload.wikimedia.org/wikipedia/tr/2/2e/%C3%87aykur_Rizespor_logo.png",

    859: "https://upload.wikimedia.org/wikipedia/tr/2/26/Adana_Demirspor_logo.png",

    5532: "https://upload.wikimedia.org/wikipedia/tr/7/74/Fatih_Karag%C3%BCmr%C3%BCk_SK_logo.png",

    5529: "https://upload.wikimedia.org/wikipedia/tr/2/23/MKE_Ankarag%C3%BCc%C3%BC_logo.png",

    5523: "https://upload.wikimedia.org/wikipedia/tr/7/7d/Pendikspor_logo.png",

    5531: "https://upload.wikimedia.org/wikipedia/tr/7/77/G%C3%B6ztepe_logo.png",

    2032: "https://upload.wikimedia.org/wikipedia/tr/9/9a/Ey%C3%BCpspor_logo.png"

}



class DataManager:

    def __init__(self, api_key):

        self.headers = {"X-Auth-Token": api_key}



    @st.cache_data(ttl=3600)

    def fetch_data(_self, league_code):

        standings_data = {"standings": [{"table": []}]}

        matches_data = {"matches": []}

        

        try:

            # Standings her turnuvada olmayabilir (örn. kupa) o yüzden try-except

            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=_self.headers)

            if r1.status_code == 200:

                standings_data = r1.json()

            

            today = datetime.now().strftime("%Y-%m-%d")

            future = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

            

            # Fikstür Çekme

            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", 

                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})

            if r2.status_code == 200:

                matches_data = r2.json()

        except:

            pass



        # --- MANUEL SÜPER KUPA MAÇLARI ---

        if league_code == "TR1" or league_code == "SC": 

            manual_matches = [

                {

                    "id": 99901,

                    "homeTeam": {"name": "Galatasaray", "id": 2054, "crest": TEAM_LOGOS.get(2054)},

                    "awayTeam": {"name": "Trabzonspor", "id": 2061, "crest": TEAM_LOGOS.get(2061)},

                    "utcDate": "2026-01-05T17:30:00Z",

                    "status": "SCHEDULED",

                    "competition": {"name": "TFF Süper Kupa"}

                },

                {

                    "id": 99902,

                    "homeTeam": {"name": "Fenerbahçe", "id": 2052, "crest": TEAM_LOGOS.get(2052)},

                    "awayTeam": {"name": "Samsunspor", "id": 2058, "crest": TEAM_LOGOS.get(2058)},

                    "utcDate": "2026-01-06T17:30:00Z",

                    "status": "SCHEDULED",

                    "competition": {"name": "TFF Süper Kupa"}

                }

            ]

            if 'matches' not in matches_data:

                matches_data['matches'] = []

            matches_data['matches'].extend(manual_matches)

        

        # Boş veri kontrolü

        t_check = standings_data.get("standings", [])

        m_check = matches_data.get("matches", [])

        

        if not t_check and not m_check:

            return None, None

            

        return standings_data, matches_data



# -----------------------------------------------------------------------------

# 5. SIMULATION ENGINE

# -----------------------------------------------------------------------------

class SimulationEngine:

    def __init__(self):

        self.rng = np.random.default_rng()



    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):

        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']

        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']

        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']

        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']



        xg_h = h_attack * a_def * avg_g * params['home_adv']

        xg_a = a_attack * h_def * avg_g



        if params.get('h_missing', 0) > 0:

            xg_h = xg_h * (1 - (params['h_missing'] * 0.12))

        

        if params.get('a_missing', 0) > 0:

            xg_a = xg_a * (1 - (params['a_missing'] * 0.12))



        sims = params['sim_count']

        gh_ht = self.rng.poisson(xg_h * 0.45, sims)

        ga_ht = self.rng.poisson(xg_a * 0.45, sims)

        gh_ft = self.rng.poisson(xg_h * 0.55, sims)

        ga_ft = self.rng.poisson(xg_a * 0.55, sims)



        total_h = gh_ht + gh_ft

        total_a = ga_ht + ga_ft



        return {

            "h": total_h, "a": total_a,

            "ht": (gh_ht, ga_ht), "ft": (total_h, total_a),

            "xg": (xg_h, xg_a), "sims": sims

        }



    def analyze_results(self, data):

        h, a = data["h"], data["a"]

        sims = data["sims"]



        p_home = np.mean(h > a) * 100

        p_draw = np.mean(h == a) * 100

        p_away = np.mean(h < a) * 100



        matrix = np.zeros((6, 6))

        for i in range(6):

            for j in range(6):

                matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100



        scores = [f"{i}-{j}" for i, j in zip(h, a)]

        unique, counts = np.unique(scores, return_counts=True)

        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]



        total_goals = h + a

        goal_bins = {

            "0-1": np.sum(total_goals <= 1) / sims * 100,

            "2-3": np.sum((total_goals >= 2) & (total_goals <= 3)) / sims * 100,

            "4-6": np.sum((total_goals >= 4) & (total_goals <= 6)) / sims * 100,

            "7+": np.sum(total_goals >= 7) / sims * 100

        }



        h_ht, a_ht = data["ht"]

        ht_res = np.where(h_ht > a_ht, 1, np.where(h_ht < a_ht, 2, 0))

        ft_res = np.where(h > a, 1, np.where(h < a, 2, 0))

        

        htft = {}

        labels = {1: "1", 0: "X", 2: "2"}

        for i in [1, 0, 2]:

            for j in [1, 0, 2]:

                mask = (ht_res == i) & (ft_res == j)

                htft[f"{labels[i]}/{labels[j]}"] = np.sum(mask) / sims * 100



        return {

            "1x2": [p_home, p_draw, p_away],

            "matrix": matrix,

            "top_scores": top_scores,

            "goal_bins": goal_bins,

            "htft": htft,

            "xg": data["xg"]

        }



# -----------------------------------------------------------------------------

# 6. APP MAIN LOGIC

# -----------------------------------------------------------------------------

def main():

    # --- ÜST BAR ---

    with st.sidebar:

        col_lang, col_exit = st.columns([4, 1])

        with col_lang:

            lang_code = st.selectbox("Dil / Language", ["tr", "en"], label_visibility="collapsed")

        with col_exit:

            if st.button("❌", help="Reset"):

                st.session_state.clear()

                st.rerun()

        

        t = TRANSLATIONS[lang_code]

        st.divider()



    # API Key

    api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")

    

    with st.sidebar:

        st.header(f"🧪 {t['settings']}")

        if not api_key:

            api_key = st.text_input(t['api_ph'], type="password")

            if not api_key: st.stop()

            

        st.subheader(t['sim_param'])

        sim_count = st.select_slider(t['match_count'], options=[10000, 100000, 500000], value=100000)

        

        st.caption(t['form_set'])

        h_att = st.slider(t['h_att'], 80, 120, 100) / 100

        a_att = st.slider(t['a_att'], 80, 120, 100) / 100



        # EKSİK OYUNCU AYARI

        st.caption(f"🚑 {t['missing_p']}")

        c_m1, c_m2 = st.columns(2)

        with c_m1:

            h_miss = st.number_input(t['h_miss'], 0, 5, 0)

        with c_m2:

            a_miss = st.number_input(t['a_miss'], 0, 5, 0)

        

        params = {

            "sim_count": sim_count,

            "h_att_factor": h_att, "h_def_factor": 1.0,

            "a_att_factor": a_att, "a_def_factor": 1.0,

            "h_missing": h_miss, "a_missing": a_miss,

            "home_adv": 1.15

        }

        

        # BUY ME A COFFEE BUTONU

        st.divider()

        st.markdown(

            """

            <div style="text-align: center;">

                <a href="https://www.buymeacoffee.com/muratlola" target="_blank">

                    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 50px !important;width: 217px !important;" >

                </a>

            </div>

            """,

            unsafe_allow_html=True

        )



    st.markdown(f"<div class='main-title'>{t['app_title']}</div>", unsafe_allow_html=True)



    dm = DataManager(api_key)

    

    # --- YENİ LİG LİSTESİ (GÜNCELLENDİ) ---

    L_MAP = {

        "Süper Lig & Kupa (Türkiye)": "TR1",

        "Premier League (İngiltere)": "PL", 

        "Championship (İngiltere Alt)": "ELC", 

        "La Liga (İspanya)": "PD", 

        "Bundesliga (Almanya)": "BL1", 

        "Serie A (İtalya)": "SA",

        "Ligue 1 (Fransa)": "FL1", 

        "Eredivisie (Hollanda)": "DED",

        "Primeira Liga (Portekiz)": "PPL",

        "Série A (Brezilya)": "BSA",

        "UEFA Şampiyonlar Ligi": "CL",

        "FIFA Dünya Kupası": "WC",

        "Avrupa Şampiyonası": "EC"

    }

    

    c1, c2 = st.columns([1, 2])

    with c1: league = st.selectbox(t['league'], list(L_MAP.keys()))

    

    standings, fixtures = dm.fetch_data(L_MAP[league])

    

    if not standings and not fixtures:

        st.info(t['no_match'])

        st.stop()

    

    # Puan Durumu İşleme (Varsa)

    table = []

    if standings and "standings" in standings:

        standings_list = standings.get("standings", [])

        if standings_list:

            table = standings_list[0].get("table", [])



    teams = {}

    if table:

        total_goals = sum(t["goalsFor"] for t in table)

        total_games = sum(t["playedGames"] for t in table)

        avg_league = total_goals / total_games if total_games > 0 else 2.5

        

        for row in table:

            t_id = row["team"]["id"]

            crest = row["team"].get("crest") or TEAM_LOGOS.get(t_id, CONFIG["DEFAULT_LOGO"])

            

            teams[t_id] = {

                "name": row["team"]["name"], "crest": crest,

                "gf": row["goalsFor"]/row["playedGames"], "ga": row["goalsAgainst"]/row["playedGames"]

            }

    else:

        avg_league = 2.5 # Turnuvalar için varsayılan ortalama



    # Maçları Listele

    matches = {}

    if fixtures and "matches" in fixtures:

        for m in fixtures.get("matches", []):

            if m["status"] in ["SCHEDULED", "TIMED"]:

                match_label = f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})"

                matches[match_label] = m

    

    if not matches: st.info(t['no_match']); st.stop()



    with c2: sel_match = st.selectbox(t['match'], list(matches.keys()))



    if st.button(f"{t['start_btn']} ({sim_count//1000}K)", use_container_width=True):

        m = matches[sel_match]

        h_id, a_id = m["homeTeam"]["id"], m["awayTeam"]["id"]

        

        # LOGO ZORLAMASI (Maç objesinden almaya öncelik veriyoruz)

        h_crest = m["homeTeam"].get("crest") or teams.get(h_id, {}).get("crest") or TEAM_LOGOS.get(h_id, CONFIG["DEFAULT_LOGO"])

        a_crest = m["awayTeam"].get("crest") or teams.get(a_id, {}).get("crest") or TEAM_LOGOS.get(a_id, CONFIG["DEFAULT_LOGO"])



        h_team = teams.get(h_id, {"name": m["homeTeam"]["name"], "crest": h_crest, "gf": 1.5, "ga": 1.2})

        a_team = teams.get(a_id, {"name": m["awayTeam"]["name"], "crest": a_crest, "gf": 1.4, "ga": 1.3})

        

        # GÜÇ ENJEKSİYONU (Manuel Statlar - Sadece TR için örnek)

        MANUAL_STATS = {

            2054: {"gf": 2.50, "ga": 0.80}, # Galatasaray

            2052: {"gf": 2.55, "ga": 0.85}, # Fenerbahçe

            2061: {"gf": 1.75, "ga": 1.10}, # Trabzonspor

            2058: {"gf": 1.55, "ga": 1.20}  # Samsunspor

        }



        if h_id in MANUAL_STATS: h_team.update(MANUAL_STATS[h_id])

        if a_id in MANUAL_STATS: a_team.update(MANUAL_STATS[a_id])



        # SİMÜLASYON

        eng = SimulationEngine()

        with st.spinner(t['calculating']):

            current_params = params.copy()

            # Tarafsız saha kontrolü (Kupa finalleri vb.)

            comp_name = m.get("competition", {}).get("name", "")

            if "Kupa" in comp_name or "Cup" in comp_name or "Champions" in comp_name:

                # Şampiyonlar ligi finali vb değilse ev sahibi avantajını azalt ama sıfırlama

                # (Gruplarda ev sahibi var, ama finalde yok. Basitlik için düşürüyoruz)

                current_params["home_adv"] = 1.08 

            

            if "Süper Kupa" in comp_name:

                 current_params["home_adv"] = 1.0



            raw_data = eng.run_monte_carlo(h_team, a_team, avg_league, current_params)

            res = eng.analyze_results(raw_data)

            

        st.session_state.sim_results = res

        st.session_state.match_info = {"h": h_team, "a": a_team}



    # --- SONUÇ EKRANI ---

    if st.session_state.sim_results:

        res = st.session_state.sim_results

        info = st.session_state.match_info

        

        # 1. Header

        c_h, c_vs, c_a = st.columns([2,1,2])

        with c_h: st.markdown(f"<div style='text-align:center'><img src='{info['h']['crest']}' width='80'><br><h3>{info['h']['name']}</h3></div>", unsafe_allow_html=True)

        with c_vs: 

            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)

            st.metric(t['xg'], f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")

        with c_a: st.markdown(f"<div style='text-align:center'><img src='{info['a']['crest']}' width='80'><br><h3>{info['a']['name']}</h3></div>", unsafe_allow_html=True)



        st.divider()



        # 2. Olasılık Kartları

        k1, k2, k3 = st.columns(3)

        k1.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['home']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)

        k2.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['draw']}</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)

        k3.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['away']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)



        st.write("")



        # 3. DETAYLI ANALİZ

        c_heat, c_list = st.columns([2, 1])

        

        with c_heat:

            st.markdown(f"### 🔥 {t['heatmap']}")

            fig_heat = go.Figure(data=go.Heatmap(

                z=res["matrix"], x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],

                colorscale='Magma', texttemplate="%{z:.1f}%"

            ))

            fig_heat.update_layout(xaxis_title=t['away'], yaxis_title=t['home'], height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')

            st.plotly_chart(fig_heat, use_container_width=True)

            

        with c_list:

            st.markdown(f"### 🎯 {t['top_scores']}")

            with st.container():

                st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)

                for score, prob in res["top_scores"][:7]:

                    st.markdown(f"""

                    <div class='score-row'>

                        <span style='font-weight:bold; font-size:1.2rem'>{score}</span>

                        <span style='color:#38bdf8; font-weight:bold'>%{prob:.1f}</span>

                    </div>

                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)



        # 4. GOL ANALİZİ

        c_ht, c_goal = st.columns(2)

        

        with c_ht:

            st.markdown(f"### ⏱️ {t['ht_ft']}")

            htft_df = pd.DataFrame(list(res['htft'].items()), columns=['Result', 'Prob']).sort_values('Prob', ascending=False).head(7)

            fig_bar = px.bar(htft_df, x='Result', y='Prob', text_auto='.1f', color='Prob', color_continuous_scale='Viridis')

            fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white')

            st.plotly_chart(fig_bar, use_container_width=True)

            

        with c_goal:

            st.markdown(f"### 🥅 {t['total_goal']}")

            g_labels = list(res["goal_bins"].keys())

            g_vals = list(res["goal_bins"].values())

            fig_pie = go.Figure(data=[go.Pie(labels=g_labels, values=g_vals, hole=.4, marker=dict(colors=['#94a3b8', '#3b82f6', '#8b5cf6', '#f43f5e']))])

            fig_pie.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')

            st.plotly_chart(fig_pie, use_container_width=True)



    # --- ALT BİLGİ (FOOTER) ---

    st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)



if __name__ == "__main__":

    main()
