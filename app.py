import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# -----------------------------------------------------------------------------
# 1. KONFİGÜRASYON & ANAHTARLAR
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "STD_API_URL": "https://api.football-data.org/v4",
    "PRO_API_URL": "https://api.sportmonks.com/v3/football",
    "PRO_TOKEN": "GL0xxZHLVkzEUypMQdNkKow4NI0FPrlzJ4IfalN7rV6Qlc2u3M1iXDlAfCzx", # Senin Sportmonks Anahtarın
    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"}
}

st.set_page_config(page_title="Quantum Football Hybrid", page_icon="⚽", layout="wide")

# -----------------------------------------------------------------------------
# 2. DİL VE CSS
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "tr": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Ayarlar",
        "api_ph": "Football-Data API Key Giriniz",
        "sim_param": "Simülasyon Ayarları",
        "match_count": "Simülasyon Sayısı",
        "start_btn": "ANALİZİ BAŞLAT",
        "calculating": "Kuantum motoru verileri işliyor...",
        "xg": "Beklenen Gol (xG)",
        "footer": "Quantum Football v90.0 Hybrid | SportMonks & Football-Data Integrated"
    },
    "en": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Settings",
        "api_ph": "Enter Football-Data API Key",
        "sim_param": "Simulation Settings",
        "match_count": "Simulation Count",
        "start_btn": "START ANALYSIS",
        "calculating": "Processing data...",
        "xg": "Expected Goals (xG)",
        "footer": "Quantum Football v90.0 Hybrid | SportMonks & Football-Data Integrated"
    }
}

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc;}
    .main-title { font-family: 'Roboto Mono', monospace; font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 10px; background: linear-gradient(90deg, #3b82f6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem !important; }
    .stat-card { background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stat-val { font-size: 2rem; font-weight: 700; color: #fff; }
    .stat-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .score-row { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #334155; font-family: 'Roboto Mono'; }
    .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #334155; text-align: center; color: #64748b; font-size: 0.8rem; }
    div[data-testid="stButton"] button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA MANAGERS (HYBRID)
# -----------------------------------------------------------------------------
class StandardDataManager:
    """Eski Sistem: Büyük Ligler İçin"""
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    def fetch_data(self, league_code):
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            r1 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/standings", headers=self.headers)
            r2 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/matches", headers=self.headers, params={"dateFrom": today, "dateTo": future})
            return r1.json(), r2.json(), False # False = Not Pro
        except: return None, None, False

class ProDataManager:
    """Yeni Sistem: Danimarka ve İskoçya İçin (Full Özellik)"""
    def __init__(self):
        self.token = CONFIG["PRO_TOKEN"]

    def fetch_fixtures(self, league_id):
        start = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        url = f"{CONFIG['PRO_API_URL']}/fixtures/between/{start}/{end}"
        params = {"api_token": self.token, "include": "participants;league", "filters": f"leagues:{league_id}"}
        try:
            return requests.get(url, params=params).json().get("data", [])
        except: return []

    def get_match_details(self, fixture_id):
        # BÜTÜN PRO ÖZELLİKLER BURADA ÇEKİLİYOR
        includes = "participants;scores;lineups.player;weatherReport;pressure.participant;predictions.type;odds.market"
        url = f"{CONFIG['PRO_API_URL']}/fixtures/{fixture_id}"
        params = {"api_token": self.token, "include": includes}
        try:
            return requests.get(url, params=params).json().get("data", {})
        except: return {}

# -----------------------------------------------------------------------------
# 4. GÖRSELLEŞTİRME MODÜLLERİ (PRO)
# -----------------------------------------------------------------------------
def display_pro_features(match_data):
    # 1. BASKI GRAFİĞİ (MOMENTUM)
    pressure = match_data.get('pressure', [])
    if pressure:
        st.write("---")
        st.subheader("⚡ Canlı Baskı Endeksi (Momentum)")
        h_id = match_data['participants'][0]['id']
        # Son 20 dakika
        recent = pressure[-20:]
        minutes = [p.get('minute') for p in recent]
        vals = [p.get('pressure') if p['participant_id'] == h_id else -p.get('pressure') for p in recent]
        
        fig = go.Figure(go.Bar(x=minutes, y=vals, marker_color=['#3b82f6' if v>0 else '#ef4444' for v in vals]))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", yaxis_title="Baskı")
        st.plotly_chart(fig, use_container_width=True)

    # 2. KADROLAR
    lineups = match_data.get('lineups', [])
    if lineups:
        st.write("---")
        st.subheader("📋 Sahaya Dizilişler")
        c1, c2 = st.columns(2)
        for i, col in enumerate([c1, c2]):
            p_id = match_data['participants'][i]['id']
            # İlk 11 (Type ID 11)
            starters = [l for l in lineups if l['participant_id'] == p_id and l.get('type_id') == 11]
            with col:
                st.markdown(f"**{match_data['participants'][i]['name']}**")
                for p in starters:
                    st.markdown(f"▫️ {p.get('player', {}).get('display_name', 'Oyuncu')}")

# -----------------------------------------------------------------------------
# 5. SIMULATION ENGINE (STANDART)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run(self, h_s, a_s, avg, params):
        # Eğer Pro modundaysa ve hava durumu yağmurluysa gücü kır
        w_factor = 1.0
        if params.get('weather_bad'): w_factor = 0.92

        xg_h = (h_s / avg) * params['h_att'] * params['adv'] * w_factor
        xg_a = (a_s / avg) * params['a_att'] * w_factor
        
        sims = params['count']
        h = self.rng.poisson(xg_h, sims)
        a = self.rng.poisson(xg_a, sims)
        
        return {
            "h": h, "a": a, "xg": (xg_h, xg_a),
            "p_h": np.mean(h > a)*100, "p_d": np.mean(h == a)*100, "p_a": np.mean(h < a)*100,
            "matrix": np.histogram2d(h, a, bins=[6,6], range=[[0,6],[0,6]], density=True)[0]*100
        }

# -----------------------------------------------------------------------------
# 6. MAIN APP
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        lang = st.selectbox("Dil", ["tr", "en"])
        t = TRANSLATIONS[lang]
        st.divider()
        
        # API Key (Sadece Standart Mod İçin Gerekli)
        api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
        if not api_key:
            api_key = st.text_input(t['api_ph'], type="password")
        
        st.header(t['settings'])
        sim_count = st.select_slider(t['match_count'], [10000, 50000, 100000], 50000)
        h_att = st.slider("Ev Güç Çarpanı", 0.8, 1.2, 1.0)
        a_att = st.slider("Dep Güç Çarpanı", 0.8, 1.2, 1.0)

    st.markdown(f"<div class='main-title'>{t['app_title']}</div>", unsafe_allow_html=True)

    # --- LIG LISTESI (HYBRID) ---
    # Normal ligler string kod, Pro ligler sayısal ID
    L_MAP = {
        "🇩🇰 Danimarka Superliga (PRO)": 271,
        "🏴󠁧󠁢󠁳󠁣󠁴󠁿 İskoçya Premiership (PRO)": 501,
        "🇹🇷 Süper Lig": "TR1",
        "🇬🇧 Premier League": "PL",
        "🇪🇸 La Liga": "PD",
        "🇩🇪 Bundesliga": "BL1",
        "🇮🇹 Serie A": "SA",
        "🇫🇷 Ligue 1": "FL1",
        "🇳🇱 Eredivisie": "DED",
        "🇪🇺 Şampiyonlar Ligi": "CL"
    }

    c1, c2 = st.columns([1, 2])
    with c1: league_sel = st.selectbox("Lig Seçiniz", list(L_MAP.keys()))
    
    league_val = L_MAP[league_sel]
    is_pro_mode = isinstance(league_val, int) # Eğer ID sayı ise Pro Moddur

    # --- VERİ ÇEKME ---
    matches = {}
    teams_stats = {}
    avg_goals = 2.5
    
    if is_pro_mode:
        # PRO MOD (SportMonks)
        pm = ProDataManager()
        fixtures = pm.fetch_fixtures(league_val)
        if not fixtures: st.error("Fikstür bulunamadı."); st.stop()
        
        for f in fixtures:
            label = f"{f['participants'][0]['name']} vs {f['participants'][1]['name']} ({f['starting_at'][:10]})"
            matches[label] = f # Tüm objeyi sakla
            
        st.info(f"✨ PRO MOD AKTİF: {league_sel} için detaylı Sportmonks verileri kullanılıyor.")
        
    else:
        # STANDART MOD (Football-Data)
        if not api_key: st.warning("Bu lig için API Key girmeniz gerekli."); st.stop()
        sm = StandardDataManager(api_key)
        standings, fixtures, _ = sm.fetch_data(league_val)
        
        if not fixtures: st.error("Fikstür bulunamadı."); st.stop()
        
        # Takım güçlerini hesapla
        if standings and "standings" in standings:
            tbl = standings["standings"][0].get("table", [])
            if tbl:
                avg_goals = sum(t['goalsFor'] for t in tbl) / sum(t['playedGames'] for t in tbl)
                for r in tbl:
                    teams_stats[r['team']['id']] = {'gf': r['goalsFor']/r['playedGames'], 'ga': r['goalsAgainst']/r['playedGames']}
        
        for m in fixtures.get('matches', []):
            if m['status'] == 'SCHEDULED':
                label = f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})"
                matches[label] = m

    with c2: sel_match = st.selectbox("Maç Seçiniz", list(matches.keys()))

    if st.button(t['start_btn'], use_container_width=True):
        match_data = matches[sel_match]
        
        # Verileri Hazırla
        if is_pro_mode:
            # Pro veriyi çek
            full_data = pm.get_match_details(match_data['id'])
            h_name = full_data['participants'][0]['name']
            a_name = full_data['participants'][1]['name']
            h_img = full_data['participants'][0]['image_path']
            a_img = full_data['participants'][1]['image_path']
            
            # Hava durumu kontrolü
            weather_bad = False
            w = full_data.get('weather_report')
            if w and ('rain' in w.get('type','').lower() or 'snow' in w.get('type','').lower()):
                weather_bad = True
            
            # Pro modda istatistik olmadığı için varsayılan güç + manuel ayar
            h_val, a_val = 1.6, 1.2 
            
        else:
            # Standart veri
            h_id = match_data['homeTeam']['id']
            a_id = match_data['awayTeam']['id']
            h_name = match_data['homeTeam']['name']
            a_name = match_data['awayTeam']['name']
            h_img = match_data['homeTeam'].get('crest', CONFIG['DEFAULT_LOGO'])
            a_img = match_data['awayTeam'].get('crest', CONFIG['DEFAULT_LOGO'])
            
            stats_h = teams_stats.get(h_id, {'gf': 1.5, 'ga': 1.2})
            stats_a = teams_stats.get(a_id, {'gf': 1.3, 'ga': 1.3})
            h_val = stats_h['gf']
            a_val = stats_a['gf']
            weather_bad = False
            full_data = None

        # Simülasyon
        eng = SimulationEngine()
        params = {
            'count': sim_count, 'h_att': h_att, 'a_att': a_att, 
            'adv': 1.15, 'weather_bad': weather_bad
        }
        res = eng.run(h_val, a_val, avg_goals, params)

        # GÖRSELLEŞTİRME
        col_h, col_vs, col_a = st.columns([2,1,2])
        with col_h: 
            st.image(h_img, width=80)
            st.subheader(h_name)
        with col_vs: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric(t['xg'], f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with col_a: 
            st.image(a_img, width=80)
            st.subheader(a_name)

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='stat-card'>EV SAHİBİ<br><span class='stat-val'>%{res['p_h']:.1f}</span></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stat-card'>BERABERLİK<br><span class='stat-val'>%{res['p_d']:.1f}</span></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='stat-card'>DEPLASMAN<br><span class='stat-val'>%{res['p_a']:.1f}</span></div>", unsafe_allow_html=True)

        # EĞER PRO MODSA EKSTRA ÖZELLİKLERİ GÖSTER
        if is_pro_mode and full_data:
            # Hava Durumu Bilgisi
            if full_data.get('weather_report'):
                w = full_data['weather_report']
                st.info(f"🌤️ Hava Durumu: {w.get('temp')}°C, {w.get('type')} (Simülasyona Etkisi: {'Var' if weather_bad else 'Yok'})")
            
            # Özel Fonksiyonları Çağır
            display_pro_features(full_data)
            
            # AI Tahmini
            preds = full_data.get('predictions', [])
            if preds:
                st.write("---")
                for p in preds[:1]:
                    st.success(f"🤖 Sportmonks AI: {p['type']['name']} -> {p['predictions']}")

        # Heatmap (Ortak)
        st.write("")
        fig = go.Figure(data=go.Heatmap(z=res["matrix"], colorscale='Magma'))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
