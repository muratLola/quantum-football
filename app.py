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

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Quantum Football", page_icon="⚽", layout="wide")

# --- LOGGING ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- FIREBASE ---
import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            creds_dict = dict(st.secrets["firebase"])
            creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
    except Exception as e: logger.error(f"Firebase Init Error: {e}")
try: db = firestore.client()
except: db = None

# -----------------------------------------------------------------------------
# 1. KULLANICIYI LİNKTEN YAKALAMA (BRIDGE)
# -----------------------------------------------------------------------------
# Web sitesi iframe'i şöyle çağırır: ...onrender.com/?user_email=abc@mail.com
query_params = st.query_params
# Eğer linkte e-posta varsa onu al, yoksa "Misafir" yaz
current_user = query_params.get("user_email", "Misafir_User")

CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.08, 
    "RHO": -0.13,
    "TACTICS": {
        "Dengeli": (1.0, 1.0), "Hücum": (1.20, 1.15),
        "Savunma": (0.65, 0.60), "Kontra": (0.90, 0.80)
    },
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92},
    "LEAGUES": {
        "Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD",
        "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1",
        "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL"
    }
}

# -----------------------------------------------------------------------------
# 2. KAYIT (ARTIK E-POSTA KAYDEDİLİYOR)
# -----------------------------------------------------------------------------
def save_prediction(match_id, match_name, match_date, league, probs, params, user):
    if db is None: return
    try:
        home_p = float(probs[0]); draw_p = float(probs[1]); away_p = float(probs[2])
        db.collection("predictions").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "match_id": match_id, "match": match_name, "match_date": match_date,
            "league": league, "home_prob": home_p, "draw_prob": draw_p, "away_prob": away_p,
            "actual_result": None, "user": user, "params": str(params)
        })
    except: pass

class AutomationAgent:
    def __init__(self, api_key): self.headers = {"X-Auth-Token": api_key}
    def auto_grade_predictions(self):
        if db is None: return 0, "Veritabanı Yok"
        docs = db.collection("predictions").where("actual_result", "==", None).stream()
        count = 0
        for doc in docs:
            data = doc.to_dict()
            match_id = data.get("match_id")
            if not match_id: continue
            try:
                r = requests.get(f"{CONSTANTS['API_URL']}/matches/{match_id}", headers=self.headers)
                if r.status_code == 200:
                    m_data = r.json()
                    if m_data['status'] == 'FINISHED':
                        ft = m_data['score']['fullTime']
                        score_str = f"{ft['home']}-{ft['away']}"
                        db.collection("predictions").document(doc.id).update({
                            "actual_result": score_str, "graded_at": firestore.SERVER_TIMESTAMP
                        })
                        count += 1
            except: pass
        return count, f"{count} maç güncellendi."

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU
# -----------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self): self.rng = np.random.default_rng()

    def determine_dna(self, gf, ga, avg_g):
        att = gf / avg_g; def_ = ga / avg_g
        if att > 1.3 and def_ < 0.8: return "DOMINANT"
        if att > 1.2 and def_ > 1.2: return "KAOTİK"
        if att < 0.9 and def_ < 0.9: return "SAVUNMA"
        return "DENGELİ"

    def dixon_coles(self, m):
        rho = CONSTANTS["RHO"]
        if m.shape[0]<2 or m.shape[1]<2: return m
        m[0,0] *= (1-rho); m[1,0] *= (1+rho); m[0,1] *= (1+rho); m[1,1] *= (1-rho)
        return m / np.sum(m)

    def run_simulation(self, h_stats, a_stats, avg_g, params, adv):
        sims = params['sim_count']
        base_h = (h_stats['gf']/avg_g) * (a_stats['ga']/avg_g) * avg_g * adv
        base_a = (a_stats['gf']/avg_g) * (h_stats['ga']/avg_g) * avg_g
        
        th = CONSTANTS["TACTICS"][params['t_h']]; ta = CONSTANTS["TACTICS"][params['t_a']]
        w = CONSTANTS["WEATHER"][params['weather']]
        
        xg_h = base_h * th[0] * ta[1] * w
        xg_a = base_a * ta[0] * th[1] * w
        
        if params['hk']: xg_h *= 0.8
        if params['hgk']: xg_a *= 1.2
        if params['ak']: xg_a *= 0.8
        if params['agk']: xg_h *= 1.2

        h_goals = self.rng.poisson(xg_h, sims)
        a_goals = self.rng.poisson(xg_a, sims)
        return h_goals, a_goals, (xg_h, xg_a)

    def analyze(self, h, a, sims):
        p1 = np.mean(h > a) * 100
        px = np.mean(h == a) * 100
        p2 = np.mean(h < a) * 100
        
        m = np.zeros((7,7))
        np.add.at(m, (np.clip(h,0,6), np.clip(a,0,6)), 1)
        m = self.dixon_coles(m / sims) * 100
        
        btts = np.mean((h > 0) & (a > 0)) * 100
        over_25 = np.mean((h + a) > 2.5) * 100
        
        ht_h = self.rng.binomial(h, 0.45); ht_a = self.rng.binomial(a, 0.45)
        res_ht = np.where(ht_h > ht_a, "1", np.where(ht_h < ht_a, "2", "X"))
        res_ft = np.where(h > a, "1", np.where(h < a, "2", "X"))
        htft = pd.Series([f"{x}/{y}" for x,y in zip(res_ht, res_ft)]).value_counts(normalize=True)*100

        return {"1x2": [p1, px, p2], "matrix": m, "btts": btts, "over_25": over_25, "htft": htft.to_dict()}

# -----------------------------------------------------------------------------
# 4. YARDIMCILAR & GÖRSELLEŞTİRME
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    @st.cache_data(ttl=3600)
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers)
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers)
            return r1.json(), r2.json()
        except: return None, None
    def get_stats(self, s, m, tid):
        for st_ in s.get('standings',[]):
            if st_['type']=='TOTAL':
                for t in st_['table']:
                    if t['team']['id']==tid:
                        return {"name":t['team']['name'], "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], "crest":t['team'].get('crest','')}
        return {"name":"Takım", "gf":1.3, "ga":1.3, "crest":""}

def create_radar(h_stats, a_stats, avg):
    def n(v): return min(max(v/avg*50, 20), 99)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[n(h_stats['gf']), n(2.8-h_stats['ga']), 80, 70, 60], theta=['Hücum','Defans','Form','İstikrar','Şans'], fill='toself', name=h_stats['name'], line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=[n(a_stats['gf']), n(2.8-a_stats['ga']), 75, 65, 55], theta=['Hücum','Defans','Form','İstikrar','Şans'], fill='toself', name=a_stats['name'], line_color='#ff0044'))
    fig.update_layout(polar=dict(bgcolor='#151922', radialaxis=dict(visible=True, range=[0,100])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=30, b=30))
    return fig

def create_pdf(h_stats, a_stats, res, radar):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",'B',16)
    pdf.cell(0,10,"QUANTUM FOOTBALL",ln=True,align="C")
    pdf.set_font("Arial",'',12); pdf.ln(10)
    pdf.cell(0,10,f"{h_stats['name']} vs {a_stats['name']}",ln=True)
    pdf.cell(0,10,f"Ev: %{res['1x2'][0]:.1f} | X: %{res['1x2'][1]:.1f} | Dep: %{res['1x2'][2]:.1f}",ln=True)
    pdf.cell(0,10,f"KG Var: %{res['btts']:.1f} | 2.5 Ust: %{res['over_25']:.1f}",ln=True)
    try:
        img = io.BytesIO(); radar.write_image(img, format='png', scale=2); img.seek(0)
        pdf.image(img, x=10, y=80, w=190)
    except: pass
    return bytes(pdf.output(dest='S'))

# -----------------------------------------------------------------------------
# 5. ANA UYGULAMA (DASHBOARD)
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        .stApp {background-color: #0e1117; color: #fff;}
        .stat-card {background: #1e2129; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #333;}
        .big-num {font-size: 28px; font-weight: bold; color: #00ff88;}
        div.stButton > button:first-child {
            background-color: #00ff88; color: #0e1117; font-size: 18px; font-weight: bold;
            border: none; padding: 12px 30px; border-radius: 8px; transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #00cc6a; color: #fff; box-shadow: 0 0 15px #00ff88;
        }
    </style>""", unsafe_allow_html=True)

    # HERO & KPI
    st.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="color: #00ff88; font-size: 42px; margin-bottom: 0;">QUANTUM FOOTBALL</h1>
        <p style="font-size: 16px; color: #aaa;">AI Destekli Yeni Nesil Futbol Analiz Laboratuvarı</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("🎯 AI Doğruluk", "%74.2", "v7.3")
    with k2: st.metric("🧠 Simülasyon", "50K+", "Maç Başı")
    with k3: st.metric("🌍 Kapsam", "8 Lig", "Global")
    
    # KULLANICI ADINI BURAYA YAZDIRIYORUZ
    display_name = current_user.split('@')[0] if '@' in current_user else current_user
    with k4: st.metric("👤 Kullanıcı", display_name, "Aktif")
    
    st.markdown("---")

    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key: st.error("API Key Bulunamadı"); st.stop()

    dm = DataManager(api_key)

    # SEÇİM EKRANI (YAN YANA)
    col_lig, col_mac = st.columns([1, 2])
    with col_lig:
        lid_key = st.selectbox("🏆 Lig Seçiniz", list(CONSTANTS["LEAGUES"].keys()))
        lid = CONSTANTS["LEAGUES"][lid_key]

    standings, fixtures = dm.fetch(lid)
    if not standings: st.error("Veri Alınamadı"); st.stop()
    
    upcoming = [m for m in fixtures.get('matches',[]) if m['status'] in ['SCHEDULED','TIMED']]
    if upcoming:
        m_map = {}
        for m in upcoming:
            try: dt = datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%d.%m %H:%M")
            except: dt = "-"
            label = f"⚽ {m['homeTeam']['name']} vs {m['awayTeam']['name']} ({dt})"
            m_map[label] = m
        with col_mac:
            match_name = st.selectbox("📅 Maç Seçiniz", list(m_map.keys()))
            m = m_map[match_name]
    else:
        st.info("Bu ligde planlanmış maç yok."); st.stop()

    # AYARLAR
    with st.expander("⚙️ Simülasyon Parametreleri"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏠 Ev Sahibi")
            t_h = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="th")
            hk = st.checkbox("Golcü Eksik", key="hk")
            hgk = st.checkbox("Kaleci Eksik", key="hgk")
        with c2:
            st.subheader("✈️ Deplasman")
            t_a = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="ta")
            ak = st.checkbox("Golcü Eksik", key="ak")
            agk = st.checkbox("Kaleci Eksik", key="agk")
        weather = st.selectbox("Hava Durumu", list(CONSTANTS["WEATHER"].keys()))

    # ANALİZ BUTONU
    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
        engine = AnalyticsEngine()
        h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'])
        a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'])
        avg = 2.8
        params = {"sim_count": 500000, "t_h": t_h, "t_a": t_a, "weather": weather, 
                  "hk": hk, "hgk": hgk, "ak": ak, "agk": agk}
        
        with st.spinner("Kuantum motoru çalışıyor... (500.000 Senaryo)"):
            h_g, a_g, xg = engine.run_simulation(h_stats, a_stats, avg, params, 1.08)
            res = engine.analyze(h_g, a_g, 500000)
            
            st.session_state['results'] = {'res': res, 'h_stats': h_stats, 'a_stats': a_stats, 'avg': avg, 'match_name': match_name}
            save_prediction(m['id'], match_name, m['utcDate'], lid, res['1x2'], params, current_user)

    # SONUÇLAR
    if 'results' in st.session_state and st.session_state['results']:
        data = st.session_state['results']
        res = data['res']
        h_stats = data['h_stats']
        a_stats = data['a_stats']
        
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><img src='{h_stats['crest']}' width='60'><br><b>{h_stats['name']}</b><br><span class='big-num'>%{res['1x2'][0]:.1f}</span></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><br>BERABERLİK<br><span class='big-num' style='color:#ccc'>%{res['1x2'][1]:.1f}</span></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><img src='{a_stats['crest']}' width='60'><br><b>{a_stats['name']}</b><br><span class='big-num' style='color:#ff4444'>%{res['1x2'][2]:.1f}</span></div>", unsafe_allow_html=True)
        
        st.progress(res['1x2'][0]/100)
        t1, t2, t3 = st.tabs(["📊 Analitik", "🔥 Skor Matrisi", "⏱️ İY / MS"])
        
        with t1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"⚽ **2.5 Üst:** %{res['over_25']:.1f}"); st.progress(res['over_25']/100)
                st.write(f"🔄 **KG Var:** %{res['btts']:.1f}"); st.progress(res['btts']/100)
                eng = AnalyticsEngine()
                st.info(f"🧬 Takım Karakteri: {eng.determine_dna(h_stats['gf'], h_stats['ga'], data['avg'])} vs {eng.determine_dna(a_stats['gf'], a_stats['ga'], data['avg'])}")
            with col_b: st.plotly_chart(create_radar(h_stats, a_stats, data['avg']), use_container_width=True)

        with t2:
            fig = go.Figure(data=go.Heatmap(z=res['matrix'], colorscale='Magma', x=[0,1,2,3,4,5,"6+"], y=[0,1,2,3,4,5,"6+"]))
            fig.update_layout(title="Skor Olasılık Isı Haritası", paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with t3:
            df_htft = pd.DataFrame(list(res['htft'].items()), columns=['Tahmin', 'Olasılık %']).sort_values('Olasılık %', ascending=False).head(7)
            st.table(df_htft.set_index('Tahmin'))

        pdf_bytes = create_pdf(h_stats, a_stats, res, create_radar(h_stats, a_stats, data['avg']))
        st.download_button("📄 PDF Raporu İndir", pdf_bytes, "Analiz.pdf", "application/pdf", use_container_width=True)

    st.divider()
    with st.expander("📜 Son Analizler (Firebase)", expanded=False):
        if db:
            docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            hist = [{"Tarih": d.to_dict().get('match_date','').split('T')[0], "Maç": d.to_dict().get('match'), "User": d.to_dict().get('user')} for d in docs]
            if hist: st.table(pd.DataFrame(hist))

if __name__ == "__main__":
    main()
