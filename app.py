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
# 1. SCIENTIFIC CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "API_URL": "https://api.football-data.org/v4",
    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"}
}

st.set_page_config(page_title="Quantum Lab v51.1", page_icon="🧬", layout="wide")

# -----------------------------------------------------------------------------
# 2. LABORATORY UI STYLES
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc;}
    
    .lab-title {
        font-family: 'Roboto Mono', monospace; font-size: 3rem; font-weight: 800;
        text-align: center; margin-bottom: 10px;
        background: linear-gradient(90deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
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
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA & STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'match_info' not in st.session_state: st.session_state.match_info = None

class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    @st.cache_data(ttl=3600)
    def fetch_data(_self, league_code):
        try:
            # Puan Durumu
            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=_self.headers)
            r1.raise_for_status()
            
            # Fikstür: Gelecek 90 GÜN (Bütün maçları yakalamak için)
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
            
            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", 
                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            r2.raise_for_status()
            return r1.json(), r2.json()
        except: return None, None

# -----------------------------------------------------------------------------
# 4. QUANTUM SIMULATION ENGINE (v51 FUSION)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        # 1. Parametrik xG Hesaplama
        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        xg_h = h_attack * a_def * avg_g * params['home_adv']
        xg_a = a_attack * h_def * avg_g

        # 2. Simülasyon
        sims = params['sim_count']
        
        # İlk Yarı / İkinci Yarı (HT/FT)
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

        # 1X2 Olasılıkları
        p_home = np.mean(h > a) * 100
        p_draw = np.mean(h == a) * 100
        p_away = np.mean(h < a) * 100

        # Skor Matrisi (Heatmap Data)
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100

        # En Olası 10 Skor (Liste)
        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]

        # Toplam Gol Dağılımı (Pie Chart)
        total_goals = h + a
        goal_bins = {
            "0-1": np.sum(total_goals <= 1) / sims * 100,
            "2-3": np.sum((total_goals >= 2) & (total_goals <= 3)) / sims * 100,
            "4-6": np.sum((total_goals >= 4) & (total_goals <= 6)) / sims * 100,
            "7+": np.sum(total_goals >= 7) / sims * 100
        }

        # HT/FT
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
# 5. DASHBOARD UI
# -----------------------------------------------------------------------------
def main():
    api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
    
    with st.sidebar:
        st.header("🧪 Lab Settings")
        if not api_key:
            api_key = st.text_input("API Key", type="password")
            if not api_key: st.stop()
            
        st.subheader("Simülasyon Parametreleri")
        sim_count = st.select_slider("Maç Sayısı", options=[10000, 100000, 500000], value=100000)
        st.caption("Takım Form Ayarları (Varsayılan: %100)")
        h_att = st.slider("Ev Sahibi Saldırı Gücü", 80, 120, 100) / 100
        a_att = st.slider("Deplasman Saldırı Gücü", 80, 120, 100) / 100
        
        params = {
            "sim_count": sim_count,
            "h_att_factor": h_att, "h_def_factor": 1.0,
            "a_att_factor": a_att, "a_def_factor": 1.0,
            "home_adv": 1.15
        }

    st.markdown("<div class='lab-title'>QUANTUM LAB v51.1</div>", unsafe_allow_html=True)

    dm = DataManager(api_key)
    L_MAP = {"Premier League": "PL", "Süper Lig": "TR1", "La Liga": "PD", "Bundesliga": "BL1", "Serie A": "SA"}
    
    c1, c2 = st.columns([1, 2])
    with c1: league = st.selectbox("Lig Seçimi", list(L_MAP.keys()))
    
    standings, fixtures = dm.fetch_data(L_MAP[league])
    if not standings: st.error("Veri Alınamadı veya API limiti doldu."); st.stop()

    table = standings["standings"][0]["table"]
    teams = {}
    total_goals = sum(t["goalsFor"] for t in table)
    total_games = sum(t["playedGames"] for t in table)
    avg_league = total_goals / total_games if total_games > 0 else 2.5
    
    for t in table:
        teams[t["team"]["id"]] = {
            "name": t["team"]["name"], "crest": t["team"].get("crest", CONFIG["DEFAULT_LOGO"]),
            "gf": t["goalsFor"]/t["playedGames"], "ga": t["goalsAgainst"]/t["playedGames"]
        }
        
    # Genişletilmiş Tarih Aralığı + TIMED statüsü
    matches = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} (Tarih: {m['utcDate'][:10]})": m 
               for m in fixtures["matches"] if m["status"] in ["SCHEDULED", "TIMED"]}
    
    if not matches: st.info("Gelecek 90 gün içinde bu ligde planlanmış maç bulunamadı. Lütfen başka bir lig seçin."); st.stop()

    with c2: sel_match = st.selectbox("Analiz Edilecek Maç", list(matches.keys()))

    if st.button(f"DENEYİ BAŞLAT ({sim_count//1000}K SİMÜLASYON)", use_container_width=True):
        m = matches[sel_match]
        h_id, a_id = m["homeTeam"]["id"], m["awayTeam"]["id"]
        
        eng = SimulationEngine()
        with st.spinner("Kuantum motoru olasılıkları hesaplıyor..."):
            raw_data = eng.run_monte_carlo(teams[h_id], teams[a_id], avg_league, params)
            res = eng.analyze_results(raw_data)
            
        st.session_state.sim_results = res
        st.session_state.match_info = {"h": teams[h_id], "a": teams[a_id]}

    # --- SONUÇ PANELİ ---
    if st.session_state.sim_results:
        res = st.session_state.sim_results
        info = st.session_state.match_info
        
        # 1. Header
        c_h, c_vs, c_a = st.columns([2,1,2])
        with c_h: st.markdown(f"<div style='text-align:center'><img src='{info['h']['crest']}' width='80'><br><h3>{info['h']['name']}</h3></div>", unsafe_allow_html=True)
        with c_vs: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric("xG (Beklenen Gol)", f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with c_a: st.markdown(f"<div style='text-align:center'><img src='{info['a']['crest']}' width='80'><br><h3>{info['a']['name']}</h3></div>", unsafe_allow_html=True)

        st.divider()

        # 2. Ana Olasılıklar
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='stat-card'><div class='stat-lbl'>EV SAHİBİ</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stat-card'><div class='stat-lbl'>BERABERLİK</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='stat-card'><div class='stat-lbl'>DEPLASMAN</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)

        st.write("")

        # 3. DETAYLI ANALİZ (HEATMAP + SKOR LİSTESİ)
        c_heat, c_list = st.columns([2, 1])
        
        with c_heat:
            st.markdown("### 🔥 Skor Olasılık Matrisi")
            fig_heat = go.Figure(data=go.Heatmap(
                z=res["matrix"], x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],
                colorscale='Magma', texttemplate="%{z:.1f}%"
            ))
            fig_heat.update_layout(xaxis_title="Deplasman", yaxis_title="Ev Sahibi", height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with c_list:
            st.markdown("### 🎯 En Olası Skorlar")
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

        # 4. GOL ANALİZİ (HT/FT + PIE)
        c_ht, c_goal = st.columns(2)
        
        with c_ht:
            st.markdown("### ⏱️ İY/MS (HT/FT) Dağılımı")
            htft_df = pd.DataFrame(list(res['htft'].items()), columns=['Sonuç', 'Olasılık']).sort_values('Olasılık', ascending=False).head(7)
            fig_bar = px.bar(htft_df, x='Sonuç', y='Olasılık', text_auto='.1f', color='Olasılık', color_continuous_scale='Viridis')
            fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_goal:
            st.markdown("### 🥅 Toplam Gol Beklentisi")
            g_labels = list(res["goal_bins"].keys())
            g_vals = list(res["goal_bins"].values())
            fig_pie = go.Figure(data=[go.Pie(labels=g_labels, values=g_vals, hole=.4, marker=dict(colors=['#94a3b8', '#3b82f6', '#8b5cf6', '#f43f5e']))])
            fig_pie.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
