import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
from fpdf import FPDF
from typing import Dict, List, Any

# --- LOGGING ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
    except Exception as e: logger.error(f"Firebase Init Error: {e}")
try: db = firestore.client()
except: db = None

# -----------------------------------------------------------------------------
# 1. EVRENSEL SABİTLER
# -----------------------------------------------------------------------------
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.08, 
    "RHO": -0.13,
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "TACTICS": {
        "Dengeli": (1.0, 1.0), "Hücum (Gegenpressing)": (1.20, 1.15),
        "Savunma (Park the Bus)": (0.65, 0.60), "Kontra Atak": (0.90, 0.80)
    },
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92},
    "LEAGUES": {
        "Süper Lig (TR)": "TR1", "Premier League (EN)": "PL", "La Liga (ES)": "PD",
        "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1",
        "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Championship (EN)": "ELC"
    }
}

st.set_page_config(page_title="Quantum Flow v6.1", page_icon="🌊", layout="wide")

# -----------------------------------------------------------------------------
# 2. SÜPER ZEKA ÇEKİRDEĞİ (THE CORE v6.1)
# -----------------------------------------------------------------------------
class SingularityEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def dixon_coles_adjustment(self, prob_matrix):
        rho = CONSTANTS["RHO"]
        if prob_matrix.shape[0] < 2 or prob_matrix.shape[1] < 2: return prob_matrix
        prob_matrix[0, 0] *= (1 - rho); prob_matrix[1, 0] *= (1 + rho)
        prob_matrix[0, 1] *= (1 + rho); prob_matrix[1, 1] *= (1 - rho)
        return prob_matrix / np.sum(prob_matrix)

    def calculate_kelly_criterion(self, win_prob, odds):
        if odds <= 1.0: return 0.0
        b = odds - 1; p = win_prob / 100; q = 1 - p
        f = (b * p - q) / b
        return max(f * 100, 0.0)

    def determine_dna(self, gf, ga, avg_g):
        att = gf / avg_g; def_ = ga / avg_g
        if att > 1.3 and def_ < 0.8: return "TITAN", 1.2, 0.08
        if att > 1.2 and def_ > 1.2: return "CHAOS", 1.1, 0.25
        if att < 0.9 and def_ < 0.9: return "WALL", 0.8, 0.08
        if att < 0.8 and def_ > 1.3: return "FRAGILE", 0.7, 0.15
        return "BALANCED", 1.0, 0.12

    def simulate_match_momentum(self, xg_h, xg_a, sims):
        """Maçın hikayesini (Flow) ve skorunu simüle eder."""
        period_xg_h = xg_h / 6; period_xg_a = xg_a / 6
        current_h = np.zeros(sims); current_a = np.zeros(sims)
        momentum_history = [] 
        
        for period in range(6):
            fatigue_factor = 0.85 + (period * 0.08)
            diff = current_h - current_a
            factor_h = np.ones(sims) * fatigue_factor
            factor_a = np.ones(sims) * fatigue_factor
            
            mask_h_leads = diff > 0
            factor_h[mask_h_leads] *= 0.75; factor_a[mask_h_leads] *= 1.35
            
            mask_a_leads = diff < 0
            factor_h[mask_a_leads] *= 1.35; factor_a[mask_a_leads] *= 0.75
            
            if period == 5:
                mask_not_draw = diff != 0
                factor_h[mask_not_draw] *= 1.1; factor_a[mask_not_draw] *= 1.1

            # Momentum Kaydı (Ortalama Baskı)
            avg_p_h = np.mean(factor_h * period_xg_h)
            avg_p_a = np.mean(factor_a * period_xg_a)
            momentum_history.append((avg_p_h - avg_p_a) * 100) # Ev pozitif, Dep negatif

            p_h = self.rng.poisson(period_xg_h * factor_h)
            p_a = self.rng.poisson(period_xg_a * factor_a)
            current_h += p_h; current_a += p_a
            
        return current_h, current_a, momentum_history

    def run_analysis(self, h_stats, a_stats, avg_g, params, calibrated_adv):
        sims = params['sim_count']
        h_dna, h_mult, h_sig = self.determine_dna(h_stats['gf'], h_stats['ga'], avg_g)
        a_dna, a_mult, a_sig = self.determine_dna(a_stats['gf'], a_stats['ga'], avg_g)
        
        # Sentetik xG Türetme
        base_h = (h_stats['gf']/avg_g) * (a_stats['ga']/avg_g) * avg_g * h_mult
        base_a = (a_stats['gf']/avg_g) * (h_stats['ga']/avg_g) * avg_g * a_mult
        
        th, ta = CONSTANTS["TACTICS"][params['t_h']], CONSTANTS["TACTICS"][params['t_a']]
        base_h *= th[0] * ta[1] * calibrated_adv * params['weather_impact']
        base_a *= ta[0] * th[1] * params['weather_impact']
        
        if params['hk']: base_h *= 0.8
        if params['hgk']: base_a *= 1.2
        if params['ak']: base_a *= 0.8
        if params['agk']: base_h *= 1.2
        
        xg_h_rand = np.maximum(base_h * self.rng.normal(1, h_sig, sims), 0.05)
        xg_a_rand = np.maximum(base_a * self.rng.normal(1, a_sig, sims), 0.05)
        
        goals_h, goals_a, momentum = self.simulate_match_momentum(xg_h_rand, xg_a_rand, sims)
        return goals_h, goals_a, (base_h, base_a), (h_dna, a_dna), momentum

    def post_process(self, h, a, sims, odds=None):
        m = np.zeros((7,7))
        hc, ac = np.clip(h, 0, 6).astype(int), np.clip(a, 0, 6).astype(int)
        np.add.at(m, (hc, ac), 1)
        m = m / sims
        m = self.dixon_coles_adjustment(m)
        
        p_home = np.sum(np.tril(m, -1)) * 100
        p_draw = np.trace(m) * 100
        p_away = np.sum(np.triu(m, 1)) * 100
        
        kelly = {"h": 0, "d": 0, "a": 0}
        if odds:
            kelly["h"] = self.calculate_kelly_criterion(p_home, odds[0])
            kelly["d"] = self.calculate_kelly_criterion(p_draw, odds[1])
            kelly["a"] = self.calculate_kelly_criterion(p_away, odds[2])
            
        return {
            "1x2": [p_home, p_draw, p_away], "matrix": m * 100, "kelly": kelly,
            "btts": (1 - m[0,:].sum() - m[:,0].sum() + m[0,0]) * 100
        }

# -----------------------------------------------------------------------------
# 3. VERİ YÖNETİCİSİ (ARCHITECT MODE)
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers)
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers)
            return r1.json(), r2.json()
        except: return None, None
    
    def get_stats(self, s, m, tid):
        for st_ in s.get('standings', []):
            if st_['type']=='TOTAL':
                for t in st_['table']:
                    if t['team']['id']==tid:
                        return {
                            "name":t['team']['name'], 
                            "gf":t['goalsFor']/t['playedGames'], 
                            "ga":t['goalsAgainst']/t['playedGames'], 
                            "crest":t['team'].get('crest', ''),
                            "form": t.get('form', '-----')
                        }
        return {"name":"Takım", "gf":1.3, "ga":1.3, "crest":"", "form":""}

# -----------------------------------------------------------------------------
# 4. MAIN UI
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        .stApp {background-color: #050505; color: #fff; font-family: 'Courier New', monospace;}
        .big-stat {font-size: 2em; font-weight: bold; color: #00ff88;}
        .kelly-box {border: 1px solid #333; padding: 10px; border-radius: 5px; background: #111;}
    </style>""", unsafe_allow_html=True)
    
    st.title("🌊 QUANTUM FLOW v6.1")
    
    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key: st.error("API Key Yok"); st.stop()
    
    dm = DataManager(api_key)
    lid_key = st.selectbox("Lig Seç", list(CONSTANTS["LEAGUES"].keys()))
    lid = CONSTANTS["LEAGUES"][lid_key]
    
    standings, fixtures = dm.fetch(lid)
    if not standings: st.error("API Hatası (Limit dolmuş olabilir)"); st.stop()
    
    upcoming = [m for m in fixtures.get('matches',[]) if m['status'] in ['SCHEDULED','TIMED']]
    m_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upcoming}
    
    if not m_map: st.info("Bu ligde yaklaşan maç yok."); st.stop()

    c1, c2 = st.columns([2, 1])
    with c1: match_name = st.selectbox("Maç Seç", list(m_map.keys()))
    m = m_map[match_name]
    
    with c2:
        st.caption("💰 Bahis Oranları (Kelly İçin)")
        col_o1, col_oX, col_o2 = st.columns(3)
        odd_1 = col_o1.number_input("1", 1.0, 10.0, 1.0)
        odd_x = col_oX.number_input("X", 1.0, 10.0, 1.0)
        odd_2 = col_o2.number_input("2", 1.0, 10.0, 1.0)

    with st.expander("🛠️ Simülasyon Ayarları"):
        t_h = st.selectbox("Ev Taktik", list(CONSTANTS["TACTICS"].keys()))
        t_a = st.selectbox("Dep Taktik", list(CONSTANTS["TACTICS"].keys()))
        weather = st.selectbox("Hava", list(CONSTANTS["WEATHER"].keys()))
        hk = st.checkbox("Ev Golcü Yok"); hgk = st.checkbox("Ev Kaleci Yok")
    
    if st.button("🚀 SİMÜLE ET", use_container_width=True):
        engine = SingularityEngine()
        h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'])
        a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'])
        
        params = {
            "sim_count": 100000, "t_h": t_h, "t_a": t_a, 
            "hk": hk, "hgk": hgk, "ak": False, "agk": False,
            "weather_impact": CONSTANTS["WEATHER"][weather]
        }
        
        with st.spinner("Kuantum Akışı Hesaplanıyor..."):
            h_res, a_res, xg, dna, momentum = engine.run_analysis(h_stats, a_stats, 2.8, params, 1.05)
            final = engine.post_process(h_res, a_res, 100000, odds=[odd_1, odd_x, odd_2])
        
        st.divider()
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric(h_stats['name'], f"%{final['1x2'][0]:.1f}")
        c_res2.metric("Beraberlik", f"%{final['1x2'][1]:.1f}")
        c_res3.metric(a_stats['name'], f"%{final['1x2'][2]:.1f}")
        
        st.progress(final['1x2'][0]/100)
        
        # --- MOMENTUM GRAFİĞİ (THE FLOW) ---
        st.subheader("🌊 Maçın Hikayesi (Momentum Akışı)")
        periods = ["0-15'", "15-30'", "30-45'", "45-60'", "60-75'", "75-90'"]
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(
            x=periods, y=momentum, fill='tozeroy', mode='lines+markers',
            line=dict(width=3, color='#00ff88'), name='Baskı'
        ))
        fig_flow.add_hline(y=0, line_dash="dash", line_color="white")
        fig_flow.update_layout(
            title="Maç Dominasyon Analizi (Pozitif: Ev, Negatif: Dep)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=300
        )
        st.plotly_chart(fig_flow, use_container_width=True)

        st.subheader("🏦 Kelly Tavsiyesi")
        k = final['kelly']
        cols = st.columns(3)
        cols[0].info(f"Ev: %{k['h']:.1f}")
        cols[1].info(f"X: %{k['d']:.1f}")
        cols[2].info(f"Dep: %{k['a']:.1f}")

        st.info(f"🧬 DNA: {h_stats['name']} ({dna[0]}) vs {a_stats['name']} ({dna[1]})")
        
        st.subheader("🎯 Skor Matrisi")
        fig = go.Figure(data=go.Heatmap(z=final['matrix'], colorscale='Viridis'))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
