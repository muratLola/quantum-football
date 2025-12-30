import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# 1. ANALYTIC CONFIGURATION (SAF ANALİZ)
# -----------------------------------------------------------------------------
CONFIG = {
    "SIMULATION": {
        "COUNT": 100000, # Hız için 100k (İstenirse 1M yapılabilir)
        "HT_FACTOR": 0.45, # İlk yarı gol beklentisi çarpanı
        "FT_FACTOR": 0.55  # İkinci yarı gol beklentisi çarpanı
    },
    "MODEL": {
        "HOME_ADV": 1.15, # Ev sahibi xG avantajı
        "LEAGUE_AVG": 2.6 # Lig ortalaması (dinamik güncellenecek)
    },
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png"
}

st.set_page_config(page_title="Quantum Analyst v49", page_icon="🔬", layout="wide")

# -----------------------------------------------------------------------------
# 2. UI STYLES (LABORATORY DESIGN)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0e1117; font-family: 'Inter', sans-serif;}
    
    .lab-header {
        font-family: 'Roboto Mono', monospace; font-size: 2.5rem; font-weight: 800;
        text-align: center; color: #fff; margin-bottom: 20px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .stat-card {
        background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px;
        text-align: center; height: 100%;
    }
    .stat-title { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; }
    
    .prob-bar-bg { background: #334155; border-radius: 4px; height: 8px; width: 100%; margin-top: 5px; }
    .prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA MANAGER (GERÇEK VERİ AKIŞI)
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
        self.base_url = "https://api.football-data.org/v4"

    @st.cache_data(ttl=3600)
    def get_real_data(_self, league_code):
        try:
            # Puan Durumu
            r1 = requests.get(f"{_self.base_url}/competitions/{league_code}/standings", headers=_self.headers)
            r1.raise_for_status()
            # Fikstür
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{_self.base_url}/competitions/{league_code}/matches", 
                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            r2.raise_for_status()
            return r1.json(), r2.json()
        except: return None, None

# -----------------------------------------------------------------------------
# 4. QUANTUM SIMULATION ENGINE (HT/FT & GOALS)
# -----------------------------------------------------------------------------
class AnalyticEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def simulate_match(self, h_stats, a_stats, avg_g):
        # 1. xG Hesaplama (Gerçek Veriye Dayalı)
        h_attack = h_stats['goals_for'] / avg_g
        h_def = h_stats['goals_against'] / avg_g
        a_attack = a_stats['goals_for'] / avg_g
        a_def = a_stats['goals_against'] / avg_g

        # Ev Sahibi Beklenen Gol (xG)
        xg_h_total = h_attack * a_def * avg_g * CONFIG["MODEL"]["HOME_ADV"]
        # Deplasman Beklenen Gol (xG)
        xg_a_total = a_attack * h_def * avg_g

        # 2. Yarılara Bölme (HT / FT Simülasyonu için)
        # Genelde gollerin %45'i ilk yarı, %55'i ikinci yarı atılır
        xg_h_ht = xg_h_total * CONFIG["SIMULATION"]["HT_FACTOR"]
        xg_h_ft = xg_h_total * CONFIG["SIMULATION"]["FT_FACTOR"]
        
        xg_a_ht = xg_a_total * CONFIG["SIMULATION"]["HT_FACTOR"]
        xg_a_ft = xg_a_total * CONFIG["SIMULATION"]["FT_FACTOR"]

        # 3. Monte Carlo Simülasyonu (100.000 Maç)
        sims = CONFIG["SIMULATION"]["COUNT"]
        
        # İlk Yarı Simülasyonu
        goals_h_ht = self.rng.poisson(xg_h_ht, sims)
        goals_a_ht = self.rng.poisson(xg_a_ht, sims)
        
        # İkinci Yarı Simülasyonu
        goals_h_ft = self.rng.poisson(xg_h_ft, sims)
        goals_a_ft = self.rng.poisson(xg_a_ft, sims)

        # Toplam Skorlar
        total_h = goals_h_ht + goals_h_ft
        total_a = goals_a_ht + goals_a_ft

        return {
            "ht_score": (goals_h_ht, goals_a_ht),
            "ft_score": (total_h, total_a),
            "xg": (xg_h_total, xg_a_total),
            "sim_count": sims
        }

    def process_results(self, sim_data):
        h_ht, a_ht = sim_data["ht_score"]
        h_ft, a_ft = sim_data["ft_score"]
        sims = sim_data["sim_count"]

        # --- Analiz 1: Maç Sonucu (1X2) ---
        home_wins = np.sum(h_ft > a_ft)
        draws = np.sum(h_ft == a_ft)
        away_wins = np.sum(h_ft < a_ft)

        # --- Analiz 2: HT / FT Kombinasyonu ---
        # 1/1, X/1, 2/1 gibi kombinasyonlar
        # HT Sonucu: 0=Draw, 1=Home, 2=Away
        ht_res = np.where(h_ht > a_ht, 1, np.where(h_ht < a_ht, 2, 0))
        # FT Sonucu
        ft_res = np.where(h_ft > a_ft, 1, np.where(h_ft < a_ft, 2, 0))
        
        ht_ft_counts = {}
        codes = {0: "X", 1: "1", 2: "2"}
        for h in [1, 0, 2]:
            for f in [1, 0, 2]:
                mask = (ht_res == h) & (ft_res == f)
                count = np.sum(mask)
                ht_ft_counts[f"{codes[h]}/{codes[f]}"] = (count / sims) * 100

        # --- Analiz 3: Skor Matrisi (En Olası Skorlar) ---
        scores = [f"{h}-{a}" for h, a in zip(h_ft, a_ft)]
        unique, counts = np.unique(scores, return_counts=True)
        score_probs = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)

        # --- Analiz 4: Toplam Gol (0-1, 2-3, 4-6, 7+) ---
        total_goals = h_ft + a_ft
        goal_bins = {
            "0-1 Gol": np.sum((total_goals >= 0) & (total_goals <= 1)) / sims * 100,
            "2-3 Gol": np.sum((total_goals >= 2) & (total_goals <= 3)) / sims * 100,
            "4-6 Gol": np.sum((total_goals >= 4) & (total_goals <= 6)) / sims * 100,
            "7+ Gol": np.sum(total_goals >= 7) / sims * 100
        }

        return {
            "1x2": [home_wins/sims*100, draws/sims*100, away_wins/sims*100],
            "ht_ft": ht_ft_counts,
            "scores": score_probs[:5], # Top 5 skor
            "goals": goal_bins,
            "xg": sim_data["xg"]
        }

# -----------------------------------------------------------------------------
# 5. DASHBOARD UI
# -----------------------------------------------------------------------------
def main():
    api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
    
    with st.sidebar:
        st.header("🔬 Simulation Lab")
        if not api_key:
            api_key = st.text_input("API Key", type="password")
            if not api_key: st.stop()
        
        st.info("Bu modül, 100.000 maç simülasyonu yaparak saf istatistiksel olasılıkları hesaplar. Bahis önerisi içermez.")

    st.markdown("<div class='lab-header'>QUANTUM ANALYST v49</div>", unsafe_allow_html=True)

    dm = DataManager(api_key)
    L_MAP = {"Premier League": "PL", "Süper Lig": "TR1", "La Liga": "PD", "Bundesliga": "BL1", "Serie A": "SA"}
    
    c1, c2 = st.columns([1, 2])
    with c1: league = st.selectbox("Lig Seçimi", list(L_MAP.keys()))
    
    standings, fixtures = dm.get_real_data(L_MAP[league])
    if not standings: st.error("Veri Alınamadı"); st.stop()

    # İstatistik Hazırlığı
    table = standings["standings"][0]["table"]
    teams = {}
    total_goals = 0
    total_games = 0
    
    for row in table:
        teams[row["team"]["id"]] = {
            "name": row["team"]["name"],
            "goals_for": row["goalsFor"] / row["playedGames"],     # Maç başı atılan
            "goals_against": row["goalsAgainst"] / row["playedGames"], # Maç başı yenen
            "crest": row["team"].get("crest", CONFIG["DEFAULT_LOGO"])
        }
        total_goals += row["goalsFor"]
        total_games += row["playedGames"]
    
    avg_goals_league = total_goals / total_games if total_games > 0 else 2.5

    # Maç Seçimi
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m 
               for m in fixtures["matches"] if m["status"] == "SCHEDULED"}
    
    with c2: selected_match = st.selectbox("Analiz Edilecek Maç", list(matches.keys()))

    if st.button("SİMÜLASYONU BAŞLAT (100.000 MAÇ)", use_container_width=True):
        m = matches[selected_match]
        h_id, a_id = m["homeTeam"]["id"], m["awayTeam"]["id"]
        
        engine = AnalyticEngine()
        
        with st.spinner("Motor çalışıyor: 100.000 maç oynatılıyor..."):
            raw_sim = engine.simulate_match(teams[h_id], teams[a_id], avg_goals_league)
            res = engine.process_results(raw_sim)
        
        # --- SONUÇ EKRANI ---
        
        # 1. Başlık ve xG
        c_h, c_vs, c_a = st.columns([2,1,2])
        with c_h: st.markdown(f"<div style='text-align:center'><img src='{teams[h_id]['crest']}' width='80'><br><h3>{teams[h_id]['name']}</h3></div>", unsafe_allow_html=True)
        with c_vs: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric("Beklenen Gol (xG)", f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with c_a: st.markdown(f"<div style='text-align:center'><img src='{teams[a_id]['crest']}' width='80'><br><h3>{teams[a_id]['name']}</h3></div>", unsafe_allow_html=True)

        st.divider()

        # 2. Maç Sonucu Olasılıkları
        st.subheader("📊 Maç Sonucu Olasılıkları (1X2)")
        probs = res["1x2"]
        cols = st.columns(3)
        colors = ["#4ade80", "#94a3b8", "#f87171"]
        labels = ["Ev Sahibi Kazanır", "Beraberlik", "Deplasman Kazanır"]
        
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"<div class='stat-card'><div class='stat-title'>{labels[i]}</div><div class='stat-value'>%{probs[i]:.1f}</div><div class='prob-bar-bg'><div class='prob-bar-fill' style='width:{probs[i]}%; background:{colors[i]}'></div></div></div>", unsafe_allow_html=True)

        # 3. HT/FT Matrisi
        st.subheader("⏱️ İlk Yarı / Maç Sonu (HT/FT) Analizi")
        htft_df = pd.DataFrame.from_dict(res["ht_ft"], orient='index', columns=['Olasılık %'])
        htft_df = htft_df.sort_values(by='Olasılık %', ascending=False).head(5) # Top 5
        
        fig_htft = px.bar(htft_df, x=htft_df.index, y='Olasılık %', text_auto='.1f', 
                          color='Olasılık %', color_continuous_scale='Viridis')
        fig_htft.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_htft, use_container_width=True)

        # 4. En Olası Skorlar & Gol Aralığı
        c_score, c_goals = st.columns(2)
        
        with c_score:
            st.subheader("🎯 En Olası 5 Skor")
            for score, prob in res["scores"]:
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #334155;'>
                    <span style='font-weight:bold; font-size:1.1rem'>{score}</span>
                    <span style='color:#38bdf8'>%{prob:.1f}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with c_goals:
            st.subheader("🥅 Toplam Gol Aralığı")
            g_labels = list(res["goals"].keys())
            g_vals = list(res["goals"].values())
            
            fig_pie = go.Figure(data=[go.Pie(labels=g_labels, values=g_vals, hole=.4)])
            fig_pie.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
