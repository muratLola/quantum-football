import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import csv

# -----------------------------------------------------------------------------
# 1. KONFIGURASYON
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "STD_API_URL": "https://api.football-data.org/v4",
    "PRO_API_URL": "https://api.sportmonks.com/v3/football",
    "PRO_TOKEN": "GL0xxZHLVkzEUypMQdNkKow4NI0FPrlzJ4IfalN7rV6Qlc2u3M1iXDlAfCzx", 
    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"},
    "ADMIN_PASS": "muratLola26", 
    "LOG_FILE": "user_activity_logs.csv"
}

st.set_page_config(page_title="Quantum Football Hybrid", page_icon="⚽", layout="wide")

# -----------------------------------------------------------------------------
# 2. KULLANICI TAKIP SISTEMI (LOGGING)
# -----------------------------------------------------------------------------
query_params = st.query_params
current_user_email = query_params.get("user_email", "Anonim_Ziyaretci")

def log_activity(league, match):
    """Kimin hangi maca baktigini kaydeder"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = current_user_email[0] if isinstance(current_user_email, list) else current_user_email
    new_data = [timestamp, user, league, match]
    
    file_exists = os.path.isfile(CONFIG["LOG_FILE"])
    try:
        with open(CONFIG["LOG_FILE"], mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Zaman", "Kullanici", "Lig", "Mac"]) 
            writer.writerow(new_data)
    except: pass

# -----------------------------------------------------------------------------
# 3. DIL VE ARAYUZ (EKSİK ANAHTARLAR EKLENDİ)
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "tr": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Ayarlar",
        "api_ph": "Football-Data API Key",
        "sim_param": "Simulasyon Ayarlari",
        "match_count": "Simulasyon Sayisi",
        "start_btn": "ANALIZI BASLAT",
        "xg": "Beklenen Gol (xG)",
        "league": "Lig Seçimi",       # EKLENDİ
        "match": "Maç Seçimi",        # EKLENDİ
        "form_set": "Takım Formu",    # EKLENDİ
        "home": "EV SAHİBİ",          # EKLENDİ
        "draw": "BERABERLİK",         # EKLENDİ
        "away": "DEPLASMAN",          # EKLENDİ
        "no_match": "Maç bulunamadı.",# EKLENDİ
        "footer": f"Quantum Football v99.1 | User: {current_user_email}"
    },
    "en": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Settings",
        "api_ph": "Football-Data API Key",
        "sim_param": "Simulation Settings",
        "match_count": "Simulation Count",
        "start_btn": "START ANALYSIS",
        "xg": "Expected Goals (xG)",
        "league": "Select League",    # EKLENDİ
        "match": "Select Match",      # EKLENDİ
        "form_set": "Team Form",      # EKLENDİ
        "home": "HOME WIN",           # EKLENDİ
        "draw": "DRAW",               # EKLENDİ
        "away": "AWAY WIN",           # EKLENDİ
        "no_match": "No match found.",# EKLENDİ
        "footer": f"Quantum Football v99.1 | User: {current_user_email}"
    }
}

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc;}
    .main-title { font-family: 'Roboto Mono', monospace; font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 10px; background: linear-gradient(90deg, #3b82f6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem !important; }
    .stat-card { background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 15px; text-align: center; }
    .stat-val { font-size: 2rem; font-weight: 700; color: #fff; }
    .score-row { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #334155; font-family: 'Roboto Mono'; }
    .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #334155; text-align: center; color: #64748b; font-size: 0.8rem; }
    div[data-testid="stButton"] button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DATA MANAGERS
# -----------------------------------------------------------------------------
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'match_info' not in st.session_state: st.session_state.match_info = None

TEAM_LOGOS = {
    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",
    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",
    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",
    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",
}

class StandardDataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
    @st.cache_data(ttl=3600)
    def fetch_data(_self, league_code):
        standings_data = {"standings": [{"table": []}]}
        matches_data = {"matches": []}
        try:
            r1 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/standings", headers=_self.headers)
            if r1.status_code == 200: standings_data = r1.json()
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/matches", headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            if r2.status_code == 200: matches_data = r2.json()
        except: pass
        return standings_data, matches_data

class ProDataManager:
    def __init__(self):
        self.token = CONFIG["PRO_TOKEN"]
    def fetch_fixtures(self, league_id):
        start = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        url = f"{CONFIG['PRO_API_URL']}/fixtures/between/{start}/{end}"
        params = {"api_token": self.token, "include": "participants;league", "filters": f"leagues:{league_id}"}
        try: return requests.get(url, params=params).json().get("data", [])
        except: return []
    def get_full_match_details(self, fixture_id):
        includes = "participants;scores;lineups.player;weatherReport;pressure.participant;predictions.type;odds.market"
        url = f"{CONFIG['PRO_API_URL']}/fixtures/{fixture_id}"
        params = {"api_token": self.token, "include": includes}
        try: return requests.get(url, params=params).json().get("data", {})
        except: return {}

# -----------------------------------------------------------------------------
# 5. GORSELLESTIRME (PRO)
# -----------------------------------------------------------------------------
def display_pro_features(match_data):
    pressure = match_data.get('pressure', [])
    if pressure:
        st.write("---")
        st.subheader("⚡ Canli Baski Endeksi")
        h_id = match_data['participants'][0]['id']
        recent = pressure[-20:]
        minutes = [p.get('minute') for p in recent]
        vals = [p.get('pressure') if p['participant_id'] == h_id else -p.get('pressure') for p in recent]
        fig = go.Figure(go.Bar(x=minutes, y=vals, marker_color=['#3b82f6' if v>0 else '#ef4444' for v in vals]))
        fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", yaxis_title="Baski")
        st.plotly_chart(fig, use_container_width=True)

    lineups = match_data.get('lineups', [])
    if lineups:
        st.write("---")
        st.subheader("📋 Sahaya Dizilisler")
        c1, c2 = st.columns(2)
        for i, col in enumerate([c1, c2]):
            p_id = match_data['participants'][i]['id']
            starters = [l for l in lineups if l['participant_id'] == p_id and l.get('type_id') == 11]
            with col:
                st.markdown(f"**{match_data['participants'][i]['name']}**")
                for p in starters: st.markdown(f"▫️ {p.get('player', {}).get('display_name', 'Oyuncu')}")

# -----------------------------------------------------------------------------
# 6. SIMULASYON MOTORU
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self): self.rng = np.random.default_rng()
    def run(self, h_val, a_val, avg, params):
        w_factor = 0.92 if params.get('weather_bad') else 1.0
        xg_h = (h_val / avg) * params['h_att'] * params['adv'] * w_factor
        xg_a = (a_val / avg) * params['a_att'] * w_factor
        sims = params['count']
        h_s = self.rng.poisson(xg_h, sims)
        a_s = self.rng.poisson(xg_a, sims)
        
        p_h, p_d, p_a = np.mean(h_s > a_s)*100, np.mean(h_s == a_s)*100, np.mean(h_s < a_s)*100
        matrix = np.histogram2d(h_s, a_s, bins=[6,6], range=[[0,6],[0,6]], density=True)[0]*100
        
        scores = [f"{i}-{j}" for i, j in zip(h_s, a_s)]
        u, c = np.unique(scores, return_counts=True)
        top = sorted(zip(u, c/sims*100), key=lambda x: x[1], reverse=True)[:7]
        
        tot = h_s + a_s
        gb = {"0-1": np.sum(tot<=1), "2-3": np.sum((tot>=2)&(tot<=3)), "4-6": np.sum((tot>=4)&(tot<=6)), "7+": np.sum(tot>=7)}
        gb = {k: v/sims*100 for k, v in gb.items()}

        h_ht = self.rng.poisson(xg_h*0.45, sims)
        a_ht = self.rng.poisson(xg_a*0.45, sims)
        res_ht = np.where(h_ht > a_ht, 1, np.where(h_ht < a_ht, 2, 0))
        res_ft = np.where(h_s > a_s, 1, np.where(h_s < a_s, 2, 0))
        htft_map = {1:"1", 0:"X", 2:"2"}
        htft = sorted(zip(*np.unique([f"{htft_map[h]}/{htft_map[f]}" for h, f in zip(res_ht, res_ft)], return_counts=True)), key=lambda x: x[1], reverse=True)[:5]
        htft = [(k, v/sims*100) for k, v in htft]

        return {"p": [p_h, p_d, p_a], "matrix": matrix, "top": top, "gb": gb, "htft": htft, "xg": (xg_h, xg_a)}

# -----------------------------------------------------------------------------
# 7. MAIN APP
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        lang = st.selectbox("Dil / Language", ["tr", "en"], label_visibility="collapsed")
        t = TRANSLATIONS[lang]
        st.divider()
        
        api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
        if not api_key: api_key = st.text_input(t['api_ph'], type="password")
        
        st.header(t['settings'])
        sim_count = st.select_slider(t['match_count'], [10000, 50000, 100000], 50000)
        st.caption(t['form_set'])
        h_att = st.slider("Ev Gucu", 80, 120, 100)/100
        a_att = st.slider("Dep Gucu", 80, 120, 100)/100
        c1, c2 = st.columns(2)
        h_miss = c1.number_input("Ev Eksik", 0, 5, 0)
        a_miss = c2.number_input("Dep Eksik", 0, 5, 0)

        # --- ADMIN PANELI (GIZLI) ---
        st.divider()
        with st.expander("🔐 Admin Paneli"):
            admin_pw = st.text_input("Sifre", type="password")
            if admin_pw == CONFIG["ADMIN_PASS"]:
                st.success("Hosgeldin Murat!")
                if os.path.exists(CONFIG["LOG_FILE"]):
                    df_log = pd.read_csv(CONFIG["LOG_FILE"])
                    st.write("### 📊 Kullanici Aktiviteleri")
                    st.dataframe(df_log, use_container_width=True)
                    if st.button("Loglari Temizle"):
                        os.remove(CONFIG["LOG_FILE"])
                        st.rerun()
                else:
                    st.info("Henuz veri yok.")
            elif admin_pw: st.error("Hatali Sifre")

    st.markdown(f"<div class='main-title'>{t['app_title']}</div>", unsafe_allow_html=True)
    
    # Kullanıcı emailini göster (V99 Özelliği)
    if current_user_email and current_user_email != "Anonim_Ziyaretci":
        user_display = current_user_email[0] if isinstance(current_user_email, list) else current_user_email
        st.caption(f"👤 Giris Yapan: **{user_display}**")

    L_MAP = {
        "Danimarka Superliga (PRO)": 271, "Iskocya Premiership (PRO)": 501,
        "Turkiye Super Lig": "TR1", "Ingiltere Premier League": "PL", "Ispanya La Liga": "PD",
        "Almanya Bundesliga": "BL1", "Italya Serie A": "SA", "Fransa Ligue 1": "FL1", 
        "Hollanda Eredivisie": "DED", "Sampiyonlar Ligi": "CL"
    }

    c1, c2 = st.columns([1, 2])
    with c1: league_sel = st.selectbox(t['league'], list(L_MAP.keys()))
    league_val = L_MAP[league_sel]
    is_pro_mode = isinstance(league_val, int)

    matches, teams_data, avg_goals = {}, {}, 2.5

    if is_pro_mode:
        pm = ProDataManager()
        fixtures = pm.fetch_fixtures(league_val)
        if not fixtures: st.warning(t['no_match']); st.stop()
        for f in fixtures: matches[f"{f['participants'][0]['name']} vs {f['participants'][1]['name']} ({f['starting_at'][:10]})"] = f
        st.success("✨ PRO MOD AKTIF (SportMonks)")
    else:
        if not api_key: st.warning("Bu ligler icin API Key giriniz."); st.stop()
        sm = StandardDataManager(api_key)
        standings, fixtures = sm.fetch_data(league_val)
        if not fixtures or not fixtures.get('matches'): st.warning(t['no_match']); st.stop()
        if standings and "standings" in standings:
            tbl = standings["standings"][0].get("table", [])
            if tbl:
                avg_goals = sum(x["goalsFor"] for x in tbl) / sum(x["playedGames"] for x in tbl)
                for r in tbl: teams_data[r['team']['id']] = {'name': r['team']['name'], 'crest': r['team'].get('crest'), 'gf': r['goalsFor']/r['playedGames'], 'ga': r['goalsAgainst']/r['playedGames']}
        for m in fixtures.get('matches', []):
            if m['status'] in ['SCHEDULED', 'TIMED']: matches[f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})"] = m

    with c2: sel_match = st.selectbox(t['match'], list(matches.keys()))

    if st.button(t['start_btn'], use_container_width=True):
        # --- LOGLAMA TETIKLENIR ---
        log_activity(league_sel, sel_match)
        
        m_data = matches[sel_match]
        h_val, a_val, h_name, a_name, weather_bad, full_pro_data = 1.5, 1.2, "Ev", "Dep", False, None
        h_img, a_img = CONFIG["DEFAULT_LOGO"], CONFIG["DEFAULT_LOGO"]

        if is_pro_mode:
            with st.spinner("SportMonks Pro verileri cekiliyor..."):
                full_pro_data = pm.get_full_match_details(m_data['id'])
                h_name, a_name = full_pro_data['participants'][0]['name'], full_pro_data['participants'][1]['name']
                h_img, a_img = full_pro_data['participants'][0]['image_path'], full_pro_data['participants'][1]['image_path']
                w = full_pro_data.get('weather_report')
                if w and ('rain' in w.get('type','').lower() or 'snow' in w.get('type','').lower()): weather_bad = True
                h_val, a_val = 1.6, 1.2 
        else:
            h_id, a_id = m_data['homeTeam']['id'], m_data['awayTeam']['id']
            h_info, a_info = teams_data.get(h_id, {}), teams_data.get(a_id, {})
            h_name, a_name = m_data['homeTeam']['name'], m_data['awayTeam']['name']
            h_img, a_img = h_info.get('crest', CONFIG["DEFAULT_LOGO"]), a_info.get('crest', CONFIG["DEFAULT_LOGO"])
            h_val, a_val = h_info.get('gf', 1.5), a_info.get('gf', 1.2)

        eng = SimulationEngine()
        h_val *= (1 - h_miss * 0.12)
        a_val *= (1 - a_miss * 0.12)
        res = eng.run(h_val, a_val, avg_goals, {'count': sim_count, 'h_att': h_att, 'a_att': a_att, 'adv': 1.15, 'weather_bad': weather_bad})

        c1, c2, c3 = st.columns([2,1,2])
        with c1: st.markdown(f"<div style='text-align:center'><img src='{h_img}' width='80'><br><h3>{h_name}</h3></div>", unsafe_allow_html=True)
        with c2: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric(t['xg'], f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with c3: st.markdown(f"<div style='text-align:center'><img src='{a_img}' width='80'><br><h3>{a_name}</h3></div>", unsafe_allow_html=True)

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='stat-card'>{t['home']}<br><span class='stat-val'>%{res['p'][0]:.1f}</span></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stat-card'>{t['draw']}<br><span class='stat-val'>%{res['p'][1]:.1f}</span></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='stat-card'>{t['away']}<br><span class='stat-val'>%{res['p'][2]:.1f}</span></div>", unsafe_allow_html=True)

        if is_pro_mode and full_pro_data:
            if full_pro_data.get('weather_report'): st.info(f"🌤️ Hava: {full_pro_data['weather_report'].get('temp')}C")
            display_pro_features(full_pro_data)
            preds = full_pro_data.get('predictions', [])
            if preds: 
                st.write("---")
                for p in preds[:1]: st.success(f"🤖 AI Tahmini: {p['predictions']}")

        st.write("")
        c_heat, c_list = st.columns([2, 1])
        with c_heat:
            st.write("### 🔥 Skor Matrisi")
            fig_heat = go.Figure(data=go.Heatmap(z=res["matrix"], x=[0,1,2,3,4,5], y=[0,1,2,3,4,5], colorscale='Magma', texttemplate="%{z:.1f}%"))
            fig_heat.update_layout(xaxis_title="Deplasman", yaxis_title="Ev Sahibi", height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with c_list:
            st.write("### 🎯 En Olası Skorlar")
            with st.container():
                for score, prob in res["top"]:
                    st.markdown(f"<div class='score-row'><span style='font-weight:bold; font-size:1.2rem'>{score}</span><span style='color:#38bdf8; font-weight:bold'>%{prob:.1f}</span></div>", unsafe_allow_html=True)

        c_ht, c_goal = st.columns(2)
        with c_ht:
            st.write("### ⏱️ IY/MS Tahmini")
            htft_df = pd.DataFrame(res['htft'], columns=['Sonuc', 'Olasilik'])
            fig_bar = px.bar(htft_df, x='Sonuc', y='Olasilik', text_auto='.1f', color='Olasilik', color_continuous_scale='Viridis')
            fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
        with c_goal:
            st.write("### 🥅 Toplam Gol")
            fig_pie = go.Figure(data=[go.Pie(labels=list(res["gb"].keys()), values=list(res["gb"].values()), hole=.4, marker=dict(colors=['#94a3b8', '#3b82f6', '#8b5cf6', '#f43f5e']))])
            fig_pie.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
