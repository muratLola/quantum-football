import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import time
import logging
import io
from fpdf import FPDF
from typing import Dict, Tuple, List, Any, Optional

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
# 1. EVRENSEL SABİTLER (UNIVERSAL CONSTANTS)
# -----------------------------------------------------------------------------
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.05, 
    "RHO": -0.13, # Dixon-Coles Korelasyon Katsayısı (Beraberlik Düzeltmesi)
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "TACTICS": {
        "Dengeli": (1.0, 1.0), "Hücum (Gegenpressing)": (1.20, 1.15),
        "Savunma (Park the Bus)": (0.65, 0.60), "Kontra Atak": (0.90, 0.80)
    },
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92}
}

st.set_page_config(page_title="Quantum Singularity v5.0", page_icon="🌌", layout="wide")

# -----------------------------------------------------------------------------
# 2. SÜPER ZEKA ÇEKİRDEĞİ (THE CORE)
# -----------------------------------------------------------------------------
class SingularityEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def dixon_coles_adjustment(self, prob_matrix):
        """
        Dixon-Coles Modeli: Düşük skorlu beraberlikleri düzeltir.
        Matematiksel olarak 0-0 ve 1-1 olasılıklarını artırır, 1-0 ve 0-1'i dengeler.
        """
        rho = CONSTANTS["RHO"]
        
        # Matris boyutlarını kontrol et ve gerekirse genişlet (En az 2x2 olmalı)
        if prob_matrix.shape[0] < 2 or prob_matrix.shape[1] < 2:
            return prob_matrix

        # 0-0 Düzeltmesi
        prob_matrix[0, 0] *= (1 - rho)
        # 1-0 Düzeltmesi
        prob_matrix[1, 0] *= (1 + rho)
        # 0-1 Düzeltmesi
        prob_matrix[0, 1] *= (1 + rho)
        # 1-1 Düzeltmesi
        prob_matrix[1, 1] *= (1 - rho)
        
        return prob_matrix / np.sum(prob_matrix) # Normalize et

    def calculate_kelly_criterion(self, win_prob, odds):
        """
        Kelly Kriteri: Kasanın ne kadarının riske edileceğini hesaplar.
        f* = (bp - q) / b
        """
        if odds <= 1.0: return 0.0
        b = odds - 1
        p = win_prob / 100
        q = 1 - p
        f = (b * p - q) / b
        return max(f * 100, 0.0) # Yüzde olarak döndür

    def determine_dna(self, gf, ga, avg_g):
        """Takım Karakteristiği Analizi"""
        att = gf / avg_g
        def_ = ga / avg_g
        
        if att > 1.3 and def_ < 0.8: return "TITAN", 1.2, 0.08  # Çok Güçlü
        if att > 1.2 and def_ > 1.2: return "CHAOS", 1.1, 0.25  # Kaotik
        if att < 0.9 and def_ < 0.9: return "WALL", 0.8, 0.08   # Duvar
        if att < 0.8 and def_ > 1.3: return "FRAGILE", 0.7, 0.15 # Kırılgan
        return "BALANCED", 1.0, 0.12

    def simulate_match_momentum(self, xg_h, xg_a, sims):
        """
        Momentum Simülasyonu: Maçı 15 dakikalık periyotlara böler.
        Skor avantajına göre xG dinamik olarak değişir.
        """
        # Toplam xG'yi 6 periyoda böl (90dk / 15dk = 6)
        period_xg_h = xg_h / 6
        period_xg_a = xg_a / 6
        
        # Simülasyon dizilerini başlat
        current_h = np.zeros(sims)
        current_a = np.zeros(sims)
        
        for period in range(6):
            # Duruma göre momentum ayarı (Vektörize işlem)
            diff = current_h - current_a
            
            # Dinamik Katsayılar (Maskeleme ile)
            factor_h = np.ones(sims)
            factor_a = np.ones(sims)
            
            # Ev öndeyse (Rölanti)
            mask_h_leads = diff > 0
            factor_h[mask_h_leads] *= 0.7
            factor_a[mask_h_leads] *= 1.4 # Dep bastırır
            
            # Dep öndeyse
            mask_a_leads = diff < 0
            factor_h[mask_a_leads] *= 1.4 # Ev bastırır
            factor_a[mask_a_leads] *= 0.7
            
            # Beraberlikte (80. dk sonrası risk alma) - Son periyot
            if period == 5:
                mask_draw = diff == 0
                factor_h[mask_draw] *= 1.3
                factor_a[mask_draw] *= 1.3

            # Poisson Simülasyonu (O periyot için)
            p_h = self.rng.poisson(period_xg_h * factor_h)
            p_a = self.rng.poisson(period_xg_a * factor_a)
            
            current_h += p_h
            current_a += p_a
            
        return current_h, current_a

    def run_analysis(self, h_stats, a_stats, avg_g, params, calibrated_adv):
        sims = params['sim_count']
        
        # 1. DNA Analizi
        h_dna, h_mult, h_sig = self.determine_dna(h_stats['gf'], h_stats['ga'], avg_g)
        a_dna, a_mult, a_sig = self.determine_dna(a_stats['gf'], a_stats['ga'], avg_g)
        
        # 2. Base xG Hesabı
        base_h = (h_stats['gf']/avg_g) * (a_stats['ga']/avg_g) * avg_g * h_mult
        base_a = (a_stats['gf']/avg_g) * (h_stats['ga']/avg_g) * avg_g * a_mult
        
        # 3. Taktik & Çevre
        th, ta = CONSTANTS["TACTICS"][params['t_h']], CONSTANTS["TACTICS"][params['t_a']]
        base_h *= th[0] * ta[1] * calibrated_adv * params['weather_impact']
        base_a *= ta[0] * th[1] * params['weather_impact']
        
        # 4. Eksikler
        if params['hk']: base_h *= 0.8
        if params['hgk']: base_a *= 1.2
        if params['ak']: base_a *= 0.8
        if params['agk']: base_h *= 1.2
        
        # 5. Momentum Simülasyonu (Singularity Engine Farkı)
        # Sigma (Şans faktörü) ekle
        xg_h_rand = base_h * self.rng.normal(1, h_sig, sims)
        xg_a_rand = base_a * self.rng.normal(1, a_sig, sims)
        
        # Negatif xG engelleme
        xg_h_rand = np.maximum(xg_h_rand, 0.05)
        xg_a_rand = np.maximum(xg_a_rand, 0.05)
        
        goals_h, goals_a = self.simulate_match_momentum(xg_h_rand, xg_a_rand, sims)
        
        return goals_h, goals_a, (base_h, base_a), (h_dna, a_dna)

    def post_process(self, h, a, sims, odds=None):
        # Matrix oluştur
        m = np.zeros((7,7))
        hc, ac = np.clip(h, 0, 6).astype(int), np.clip(a, 0, 6).astype(int)
        np.add.at(m, (hc, ac), 1)
        m = m / sims
        
        # DIXON-COLES DÜZELTMESİ (Matris üzerinde)
        m = self.dixon_coles_adjustment(m)
        
        # Olasılıkları matristen tekrar topla
        p_home = np.sum(np.tril(m, -1)) * 100
        p_draw = np.trace(m) * 100
        p_away = np.sum(np.triu(m, 1)) * 100
        
        # Kelly Criterion (Eğer oran varsa)
        kelly = {"h": 0, "d": 0, "a": 0}
        if odds:
            kelly["h"] = self.calculate_kelly_criterion(p_home, odds[0])
            kelly["d"] = self.calculate_kelly_criterion(p_draw, odds[1])
            kelly["a"] = self.calculate_kelly_criterion(p_away, odds[2])
            
        return {
            "1x2": [p_home, p_draw, p_away],
            "matrix": m * 100,
            "kelly": kelly,
            "btts": (1 - m[0,:].sum() - m[:,0].sum() + m[0,0]) * 100 # Yaklaşık BTTS
        }

# -----------------------------------------------------------------------------
# 3. YARDIMCI SINIFLAR (DATA & BRAIN)
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}

    @st.cache_data(ttl=1800, show_spinner=False)
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
                        return {"name":t['team']['name'], "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], "crest":t['team'].get('crest', '')}
        return {"name":"Takım", "gf":1.3, "ga":1.3, "crest":""}

# -----------------------------------------------------------------------------
# 4. MAIN UI
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        .stApp {background-color: #050505; color: #fff; font-family: 'Courier New', monospace;}
        .big-stat {font-size: 2em; font-weight: bold; color: #00ff88;}
        .kelly-box {border: 1px solid #333; padding: 10px; border-radius: 5px; background: #111;}
    </style>""", unsafe_allow_html=True)
    
    st.title("🌌 QUANTUM SINGULARITY v5.0")
    
    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key: st.error("API Key Yok"); st.stop()
    
    dm = DataManager(api_key)
    leagues = {"Süper Lig":"TR1","EPL":"PL","La Liga":"PD","Bundesliga":"BL1","Serie A":"SA"}
    lid = st.selectbox("Lig Seç", list(leagues.keys()))
    
    standings, fixtures = dm.fetch(leagues[lid])
    if not standings: st.error("API Hatası"); st.stop()
    
    upcoming = [m for m in fixtures.get('matches',[]) if m['status'] in ['SCHEDULED','TIMED']]
    m_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upcoming}
    
    if not m_map: st.warning("Maç yok."); st.stop()

    c1, c2 = st.columns([2, 1])
    with c1: match_name = st.selectbox("Maç Seç", list(m_map.keys()))
    m = m_map[match_name]
    
    # --- BAHİS ORANLARI GİRİŞİ (KELLY İÇİN) ---
    with c2:
        st.caption("💰 Bahis Oranları (Kelly Analizi İçin)")
        col_o1, col_oX, col_o2 = st.columns(3)
        odd_1 = col_o1.number_input("1", 1.0, 10.0, 1.0)
        odd_x = col_oX.number_input("X", 1.0, 10.0, 1.0)
        odd_2 = col_o2.number_input("2", 1.0, 10.0, 1.0)

    # --- AYARLAR ---
    with st.expander("🛠️ Simülasyon Ayarları"):
        t_h = st.selectbox("Ev Taktik", list(CONSTANTS["TACTICS"].keys()))
        t_a = st.selectbox("Dep Taktik", list(CONSTANTS["TACTICS"].keys()))
        weather = st.selectbox("Hava", list(CONSTANTS["WEATHER"].keys()))
        hk = st.checkbox("Ev Golcü Yok"); hgk = st.checkbox("Ev Kaleci Yok")
    
    if st.button("🚀 TEKİLLİK SİMÜLASYONU BAŞLAT", use_container_width=True):
        engine = SingularityEngine()
        h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'])
        a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'])
        
        params = {
            "sim_count": 100000, "t_h": t_h, "t_a": t_a, 
            "hk": hk, "hgk": hgk, "ak": False, "agk": False,
            "weather_impact": CONSTANTS["WEATHER"][weather]
        }
        
        with st.spinner("Evren simüle ediliyor..."):
            h_res, a_res, xg, dna = engine.run_analysis(h_stats, a_stats, 2.8, params, 1.05)
            # Dixon-Coles ve Kelly burada devreye giriyor
            final = engine.post_process(h_res, a_res, 100000, odds=[odd_1, odd_x, odd_2])
        
        # --- SONUÇ EKRANI ---
        st.divider()
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric(h_stats['name'], f"%{final['1x2'][0]:.1f}")
        c_res2.metric("Beraberlik", f"%{final['1x2'][1]:.1f}")
        c_res3.metric(a_stats['name'], f"%{final['1x2'][2]:.1f}")
        
        st.progress(final['1x2'][0]/100)
        
        # --- KELLY TAVSİYESİ (PARA YÖNETİMİ) ---
        st.subheader("🏦 Akıllı Kasa Yönetimi (Kelly Tavsiyesi)")
        k = final['kelly']
        
        cols = st.columns(3)
        if k['h'] > 0: 
            cols[0].success(f"Ev Sahibi: Kasanın %{k['h']:.1f}'ini bas")
        else: cols[0].error("Ev Sahibi: Değersiz Oran")
            
        if k['d'] > 0: 
            cols[1].success(f"Beraberlik: Kasanın %{k['d']:.1f}'ini bas")
        else: cols[1].error("Beraberlik: Değersiz Oran")
            
        if k['a'] > 0: 
            cols[2].success(f"Deplasman: Kasanın %{k['a']:.1f}'ini bas")
        else: cols[2].error("Deplasman: Değersiz Oran")

        # --- DNA & DETAY ---
        st.info(f"🧬 Takım DNA Analizi: {h_stats['name']} ({dna[0]}) vs {a_stats['name']} ({dna[1]})")
        
        # Skor Matrisi
        st.subheader("🎯 Skor Olasılıkları (Dixon-Coles Düzeltmeli)")
        fig = go.Figure(data=go.Heatmap(z=final['matrix'], colorscale='Viridis'))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
