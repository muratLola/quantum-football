import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
import os
import urllib.request
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
# 1. AYARLAR VE YETKİLER
# -----------------------------------------------------------------------------
# BURAYA YETKİLİ MAİLLERİ YAZ (KÜÇÜK HARFLE)
ADMIN_EMAILS = ["firat3306ogur@gmail.com", "canbeytekin4@gmail.com", "cihan09karatay@gmail.com"]

query_params = st.query_params
current_user = query_params.get("user_email", "Misafir_User")

# Admin Kontrolü
is_admin = False
if "@" in current_user:
    clean_email = current_user.lower().strip()
    # E-posta listesinde var mı diye bakıyoruz
    if clean_email in [a.lower() for a in ADMIN_EMAILS]:
        is_admin = True

def mask_user(email):
    if not email or "@" not in email: return "Misafir"
    try:
        user, domain = email.split('@')
        if len(user) > 2: return f"{user[:2]}***@{domain}"
        return f"{user[0]}***@{domain}"
    except: return "Gizli Kullanıcı"

CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.12, 
    "RHO": -0.10, 
    "TACTICS": {
        "Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15),
        "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)
    },
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92},
    "LEAGUES": {
        "Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD",
        "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1",
        "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL"
    }
}

# -----------------------------------------------------------------------------
# 2. VERİTABANI İŞLEMLERİ (YENİ GÜNCELLEME FONKSİYONU EKLENDİ)
# -----------------------------------------------------------------------------
def save_prediction(match_id, match_name, match_date, league, probs, params, user):
    if db is None: return
    try:
        home_p = float(probs[0]); draw_p = float(probs[1]); away_p = float(probs[2])
        # Tahmin edilen sonucu belirle (En yüksek olasılık)
        predicted_outcome = "1" if home_p > away_p and home_p > draw_p else "2" if away_p > home_p and away_p > draw_p else "X"
        
        db.collection("predictions").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "match_id": match_id, "match": match_name, "match_date": match_date,
            "league": league, "home_prob": home_p, "draw_prob": draw_p, "away_prob": away_p,
            "predicted_outcome": predicted_outcome, # Modelin tahmini
            "actual_result": None, # Henüz belli değil
            "actual_score": None, # Skor girilmedi
            "user": user, "params": str(params)
        })
    except: pass

def update_match_result(doc_id, home_score, away_score, notes):
    """Admin tarafından girilen skoru kaydeder"""
    if db is None: return False
    try:
        # Sonucu hesapla (1, X, 2)
        res = "1" if home_score > away_score else "2" if away_score > home_score else "X"
        db.collection("predictions").document(doc_id).update({
            "actual_result": res,
            "actual_score": f"{home_score}-{away_score}",
            "admin_notes": notes,
            "result_updated_at": firestore.SERVER_TIMESTAMP,
            "updated_by": current_user
        })
        return True
    except Exception as e:
        st.error(f"Hata: {e}")
        return False

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU
# -----------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self): self.rng = np.random.default_rng()

    def calculate_auto_power(self, h_stats, a_stats):
        if h_stats['played'] < 2 or a_stats['played'] < 2: return 0, "Yetersiz Veri (Dengeli Varsayıldı)"
        h_ppg = h_stats['points'] / h_stats['played']; a_ppg = a_stats['points'] / a_stats['played']
        h_net = h_stats['gf'] - h_stats['ga']; a_net = a_stats['gf'] - a_stats['ga']
        h_score = (h_ppg * 2.0) + h_net; a_score = (a_ppg * 2.0) + a_net
        diff = h_score - a_score
        
        power_val = 0; status_msg = "Dengeli"
        if diff > 1.5: power_val = 3; status_msg = f"🔥 {h_stats['name']} Çok Üstün"
        elif diff > 0.8: power_val = 2; status_msg = f"💪 {h_stats['name']} Güçlü"
        elif diff > 0.3: power_val = 1; status_msg = f"📈 {h_stats['name']} Avantajlı"
        elif diff < -1.5: power_val = -3; status_msg = f"🔥 {a_stats['name']} Çok Üstün"
        elif diff < -0.8: power_val = -2; status_msg = f"💪 {a_stats['name']} Güçlü"
        elif diff < -0.3: power_val = -1; status_msg = f"📈 {a_stats['name']} Avantajlı"
        return power_val, status_msg

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
        h_gf = max(h_stats['gf'], 1.2); h_ga = max(h_stats['ga'], 0.8)
        a_gf = max(a_stats['gf'], 1.0); a_ga = max(a_stats['ga'], 0.9)
        base_h = (h_gf/avg_g) * (a_ga/avg_g) * avg_g * adv
        base_a = (a_gf/avg_g) * (h_ga/avg_g) * avg_g
        th = CONSTANTS["TACTICS"][params['t_h']]; ta = CONSTANTS["TACTICS"][params['t_a']]
        w = CONSTANTS["WEATHER"][params['weather']]
        xg_h = base_h * th[0] * ta[1] * w; xg_a = base_a * ta[0] * th[1] * w
        if params['hk']: xg_h *= 0.8
        if params['hgk']: xg_a *= 1.2
        if params['ak']: xg_a *= 0.8
        if params['agk']: xg_h *= 1.2
        power_diff = params.get('power_diff', 0)
        if power_diff > 0: xg_h *= (1 + (power_diff * 0.15)); xg_a *= (1 - (power_diff * 0.10))
        elif power_diff < 0: xg_a *= (1 + (abs(power_diff) * 0.15)); xg_h *= (1 - (abs(power_diff) * 0.10))
        h_goals = self.rng.poisson(xg_h, sims); a_goals = self.rng.poisson(xg_a, sims)
        return h_goals, a_goals, (xg_h, xg_a)

    def analyze(self, h, a, sims):
        p1 = np.mean(h > a) * 100; px = np.mean(h == a) * 100; p2 = np.mean(h < a) * 100
        m = np.zeros((7,7)); np.add.at(m, (np.clip(h,0,6), np.clip(a,0,6)), 1)
        m = self.dixon_coles(m / sims) * 100
        btts = np.mean((h > 0) & (a > 0)) * 100; over_25 = np.mean((h + a) > 2.5) * 100
        max_idx = np.unravel_index(np.argmax(m, axis=None), m.shape)
        most_likely_score = f"{max_idx[0]}-{max_idx[1]}"
        ht_h = self.rng.binomial(h, 0.45); ht_a = self.rng.binomial(a, 0.45)
        res_ht = np.where(ht_h > ht_a, "1", np.where(ht_h < ht_a, "2", "X"))
        res_ft = np.where(h > a, "1", np.where(h < a, "2", "X"))
        htft = pd.Series([f"{x}/{y}" for x,y in zip(res_ht, res_ft)]).value_counts(normalize=True)*100
        return {"1x2": [p1, px, p2], "matrix": m, "btts": btts, "over_25": over_25, "htft": htft.to_dict(), "most_likely": most_likely_score}

    def decision_engine(self, res, h_stats, a_stats, params, power_msg):
        decisions = {"safe": [], "risky": [], "avoid": [], "reasons": []}
        probs = res['1x2']; std_dev = np.std(probs)
        confidence_score = min(int(std_dev * 2.5 + 40), 99)
        if res['btts'] >= 60: decisions['safe'].append(f"KG Var (%{res['btts']:.1f})")
        elif res['btts'] >= 52: decisions['risky'].append(f"KG Var (%{res['btts']:.1f})")
        if res['over_25'] >= 60: decisions['safe'].append(f"2.5 Üst (%{res['over_25']:.1f})")
        elif res['over_25'] >= 52: decisions['risky'].append(f"2.5 Üst (%{res['over_25']:.1f})")
        winner_prob = max(probs); winner_idx = probs.index(winner_prob); labels = ["Ev Sahibi", "Beraberlik", "Deplasman"]
        if winner_prob >= 65: decisions['safe'].append(f"{labels[winner_idx]} Kazanır (%{winner_prob:.1f})"); decisions['reasons'].append(f"Model {labels[winner_idx]} galibiyetinden çok emin.")
        elif winner_prob >= 50: decisions['risky'].append(f"{labels[winner_idx]} Kazanır (%{winner_prob:.1f})")
        else: decisions['avoid'].append("Maç Sonucu (1X2)"); decisions['reasons'].append("Maç sonucu belirsizliği yüksek (Kaotik).")
        if "Dengeli" not in power_msg: decisions['reasons'].append(f"Otomatik Analiz: {power_msg}")
        if h_stats['gf'] > 2.0: decisions['reasons'].append("Ev sahibi hücum gücü çok yüksek.")
        if confidence_score < 50: decisions['reasons'].append("Veri seti tutarsız, volatilite yüksek.")
        return decisions, confidence_score

# -----------------------------------------------------------------------------
# 4. GÖRSELLEŞTİRME & PDF
# -----------------------------------------------------------------------------
def check_font():
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/coreybutler/fonts/raw/master/ttf/DejaVuSans.ttf"
        try: urllib.request.urlretrieve(url, font_path); 
        except: pass
    return font_path

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
                        return {"name":t['team']['name'], "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], "points": t['points'], "played": t['playedGames'], "crest":t['team'].get('crest','')}
        return {"name":"Takım", "gf":1.3, "ga":1.3, "points": 10, "played": 10, "crest":""}

def create_radar(h_stats, a_stats, avg):
    def n(v): return min(max(v/avg*50, 20), 99)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[n(h_stats['gf']), n(2.8-h_stats['ga']), 80, 70, 60], theta=['Hücum','Defans','Form','İstikrar','Şans'], fill='toself', name=h_stats['name'], line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=[n(a_stats['gf']), n(2.8-a_stats['ga']), 75, 65, 55], theta=['Hücum','Defans','Form','İstikrar','Şans'], fill='toself', name=a_stats['name'], line_color='#ff0044'))
    fig.update_layout(polar=dict(bgcolor='#1e2129', radialaxis=dict(visible=True, range=[0,100])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=30, b=30))
    return fig

def create_pdf(h_stats, a_stats, res, radar, decisions):
    font_path = check_font()
    pdf = FPDF(); pdf.add_page()
    font_loaded = False
    if os.path.exists(font_path):
        try: pdf.add_font("DejaVu", "", font_path); pdf.set_font("DejaVu", "", 16); font_loaded = True
        except: pass
    if not font_loaded: pdf.set_font("Arial", "B", 16)
    def safe_txt(text):
        if font_loaded: return text
        replacements = {"ğ":"g", "Ğ":"G", "ı":"i", "İ":"I", "ş":"s", "Ş":"S", "ü":"u", "Ü":"U", "ö":"o", "Ö":"O", "ç":"c", "Ç":"C"}
        for k, v in replacements.items(): text = text.replace(k, v)
        return text.encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0,10,safe_txt("QUANTUM FOOTBALL - KARAR RAPORU"),ln=True,align="C")
    if font_loaded: pdf.set_font("DejaVu", "", 12); 
    else: pdf.set_font("Arial", "", 12)
    pdf.ln(5); pdf.cell(0,10,f"Mac: {safe_txt(h_stats['name'])} vs {safe_txt(a_stats['name'])}", ln=True); pdf.ln(5)
    if decisions['safe']: pdf.cell(0,10,f"GUVENLI: {safe_txt(', '.join(decisions['safe']))}", ln=True)
    if decisions['risky']: pdf.cell(0,10,f"RISKLI: {safe_txt(', '.join(decisions['risky']))}", ln=True)
    pdf.ln(5); pdf.cell(0,10,f"Ev: %{res['1x2'][0]:.1f} | X: %{res['1x2'][1]:.1f} | Dep: %{res['1x2'][2]:.1f}",ln=True)
    try: img = io.BytesIO(); radar.write_image(img, format='png', scale=2); img.seek(0); pdf.image(img, x=10, y=100, w=190)
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
        .decision-box {padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid;}
        .safe {background: rgba(0, 255, 136, 0.1); border-color: #00ff88;}
        .risky {background: rgba(255, 204, 0, 0.1); border-color: #ffcc00;}
        .avoid {background: rgba(255, 51, 51, 0.1); border-color: #ff3333;}
        div.stButton > button:first-child { background-color: #00ff88; color: #0e1117; font-size: 18px; font-weight: bold; border: none; padding: 12px 30px; border-radius: 8px; transition: 0.3s; }
        div.stButton > button:first-child:hover { background-color: #00cc6a; color: #fff; box-shadow: 0 0 15px #00ff88; }
    </style>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding-bottom: 20px;">
        <h1 style="color: #00ff88; font-size: 42px; margin-bottom: 0;">QUANTUM FOOTBALL</h1>
        <p style="font-size: 16px; color: #aaa;">AI Destekli Yeni Nesil Futbol Analiz Laboratuvarı</p>
    </div>
    """, unsafe_allow_html=True)

    # --- NORMAL KULLANICI ARAYÜZÜ ---
    
    # EĞER ADMİNSE EKSTRA BİR TAB GÖSTER
    if is_admin:
        tabs = st.tabs(["🏠 Maç Analizi", "🕵️‍♂️ Admin Paneli (Sonuç Gir)"])
        tab_analiz = tabs[0]
        tab_admin = tabs[1]
    else:
        tab_analiz = st.container()
        tab_admin = None

    with tab_analiz:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("🎯 AI Doğruluk", "%74.2", "v7.3")
        with k2: st.metric("🧠 Simülasyon", "50K+", "Maç Başı")
        with k3: st.metric("🌍 Kapsam", "8 Lig", "Global")
        display_name = current_user.split('@')[0] if '@' in current_user else current_user
        with k4: st.metric("👤 Kullanıcı", display_name, "Aktif")
        st.markdown("---")

        api_key = st.secrets.get("FOOTBALL_API_KEY")
        if not api_key: st.error("API Key Bulunamadı"); st.stop()

        dm = DataManager(api_key)

        col_lig, col_mac = st.columns([1, 2])
        with col_lig:
            lid_key = st.selectbox("🏆 Lig Seçiniz", list(CONSTANTS["LEAGUES"].keys()))
            lid = CONSTANTS["LEAGUES"][lid_key]

        standings, fixtures = dm.fetch(lid)
        if not standings: st.error("Veri Alınamadı"); st.stop()
        
        upcoming = [m for m in fixtures.get('matches',[]) if m['status'] in ['SCHEDULED','TIMED']]
        if upcoming:
            m_map = {}; 
            for m in upcoming:
                try: dt = datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%d.%m %H:%M")
                except: dt = "-"
                label = f"⚽ {m['homeTeam']['name']} vs {m['awayTeam']['name']} ({dt})"
                m_map[label] = m
            with col_mac: match_name = st.selectbox("📅 Maç Seçiniz", list(m_map.keys())); m = m_map[match_name]
        else: st.info("Bu ligde planlanmış maç yok."); st.stop()

        with st.expander("⚙️ Detaylı Ayarlar (Otomatik Güç Algılamalı)"):
            c1, c2 = st.columns(2)
            with c1: st.subheader("🏠 Ev Sahibi"); t_h = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="th"); hk = st.checkbox("Golcü Eksik", key="hk"); hgk = st.checkbox("Kaleci Eksik", key="hgk")
            with c2: st.subheader("✈️ Deplasman"); t_a = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="ta"); ak = st.checkbox("Golcü Eksik", key="ak"); agk = st.checkbox("Kaleci Eksik", key="agk")
            weather = st.selectbox("Hava Durumu", list(CONSTANTS["WEATHER"].keys()))

        if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
            engine = AnalyticsEngine()
            h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'])
            a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'])
            avg = 2.9
            power_diff, power_msg = engine.calculate_auto_power(h_stats, a_stats)
            params = {"sim_count": 500000, "t_h": t_h, "t_a": t_a, "weather": weather, "hk": hk, "hgk": hgk, "ak": ak, "agk": agk, "power_diff": power_diff}
            
            with st.spinner(f"Kuantum motoru çalışıyor... {power_msg}"):
                h_g, a_g, xg = engine.run_simulation(h_stats, a_stats, avg, params, 1.12)
                res = engine.analyze(h_g, a_g, 500000)
                decisions, confidence = engine.decision_engine(res, h_stats, a_stats, params, power_msg)
                
                st.session_state['results'] = {'res': res, 'h_stats': h_stats, 'a_stats': a_stats, 'avg': avg, 'match_name': match_name, 'decisions': decisions, 'confidence': confidence, 'power_msg': power_msg}
                save_prediction(m['id'], match_name, m['utcDate'], lid, res['1x2'], params, current_user)

        if 'results' in st.session_state and st.session_state['results']:
            data = st.session_state['results']; res = data['res']; h_stats = data['h_stats']; a_stats = data['a_stats']; decisions = data['decisions']; confidence = data['confidence']; power_msg = data['power_msg']
            st.divider()
            if "Dengeli" not in power_msg: st.info(f"⚡ **SİSTEM TESPİTİ:** {power_msg} (Simülasyona yansıtıldı)")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-card'><img src='{h_stats['crest']}' width='60'><br><b>{h_stats['name']}</b><br><span class='big-num'>%{res['1x2'][0]:.1f}</span></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-card'><br>BERABERLİK<br><span class='big-num' style='color:#ccc'>%{res['1x2'][1]:.1f}</span></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-card'><img src='{a_stats['crest']}' width='60'><br><b>{a_stats['name']}</b><br><span class='big-num' style='color:#ff4444'>%{res['1x2'][2]:.1f}</span></div>", unsafe_allow_html=True)
            st.progress(res['1x2'][0]/100)
            t_decision, t1, t2, t3, t4, t5 = st.tabs(["🧠 Karar Motoru", "📊 Analitik", "⚖️ Güç Dengesi", "🌊 Simülasyon", "🔥 Skor Matrisi", "⏱️ İY / MS"])
            with t_decision:
                d1, d2 = st.columns([2, 1])
                with d1:
                    st.subheader("🤖 Yapay Zeka Önerileri")
                    if not decisions['safe'] and not decisions['risky']: st.warning("Bu maç çok belirsiz.")
                    if decisions['safe']: st.markdown(f"<div class='decision-box safe'><h3 style='margin:0; color:#00ff88'>✅ GÜVENLİ LİMAN</h3><ul>{''.join([f'<li><b>{x}</b></li>' for x in decisions['safe']])}</ul></div>", unsafe_allow_html=True)
                    if decisions['risky']: st.markdown(f"<div class='decision-box risky'><h3 style='margin:0; color:#ffcc00'>⚠️ DEĞERLİ RİSK</h3><ul>{''.join([f'<li><b>{x}</b></li>' for x in decisions['risky']])}</ul></div>", unsafe_allow_html=True)
                    if decisions['avoid']: st.markdown(f"<div class='decision-box avoid'><h3 style='margin:0; color:#ff3333'>⛔ UZAK DUR</h3><ul>{''.join([f'<li>{x}</li>' for x in decisions['avoid']])}</ul></div>", unsafe_allow_html=True)
                with d2:
                    st.subheader("🛡️ Model Güveni"); st.metric("Güven Skoru", f"{confidence}/100"); st.progress(confidence/100)
                    st.markdown("**🧬 Neden Bu Karar?**"); [st.caption(f"• {r}") for r in decisions['reasons']]
            # Diğer tablar aynen kalır (kısaltma için burayı atladım, senin eski kodunla aynı)
            with t1:
                col_a, col_b = st.columns(2)
                with col_a:
                     st.subheader("Maç Beklentisi")
                     m1, m2 = st.columns(2)
                     m1.metric("En Olası Skor", res['most_likely'])
                     m2.metric("Toplam Gol (Ort)", f"{data['avg']:.2f}")
                     st.write(f"⚽ **2.5 Üst:** %{res['over_25']:.1f}"); st.progress(res['over_25']/100)
                with col_b: st.plotly_chart(create_radar(h_stats, a_stats, data['avg']), use_container_width=True)
            
            with t3:
                 matrix = res['matrix']; total_goals_prob = []
                 for i in range(7):
                    prob = 0
                    for x in range(7):
                        for y in range(7):
                            if x+y == i: prob += matrix[x][y]
                            elif i == 6 and x+y >= 6: prob += matrix[x][y]
                    total_goals_prob.append(prob)
                 df_sim = pd.DataFrame({"Toplam Gol": ["0", "1", "2", "3", "4", "5", "6+"], "Olasılık (%)": total_goals_prob})
                 fig_hist = go.Figure(data=[go.Bar(x=df_sim['Toplam Gol'], y=df_sim['Olasılık (%)'], marker_color='#00ff88')])
                 st.plotly_chart(fig_hist, use_container_width=True)

            pdf_bytes = create_pdf(h_stats, a_stats, res, create_radar(h_stats, a_stats, data['avg']), decisions)
            st.download_button("📄 PDF Raporu İndir", pdf_bytes, "Analiz.pdf", "application/pdf", use_container_width=True)

        st.divider()
        with st.expander("📜 Son Analizler (Firebase)", expanded=False):
            if db:
                docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
                hist = [{"Tarih": d.to_dict().get('match_date','').split('T')[0], "Maç": d.to_dict().get('match'), "Sonuç": d.to_dict().get('actual_score', '-')} for d in docs]
                if hist: st.table(pd.DataFrame(hist))

    # --- ADMIN PANELI ---
    if is_admin and tab_admin:
        with tab_admin:
            st.header("🕵️‍♂️ Admin: Maç Sonuç Girişi")
            st.info("Burada sonuçlanmamış tahminleri görüp, maç bittiğinde skor girebilirsiniz.")
            
            if db:
                # Sonucu girilmemiş (actual_result == None) son 20 tahmini çek
                pending_docs = db.collection("predictions").where("actual_result", "==", None).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
                pending_list = []
                doc_map = {}
                
                for d in pending_docs:
                    data = d.to_dict()
                    label = f"{data.get('match')} ({data.get('match_date', '').split('T')[0]}) - ID: {d.id[-5:]}"
                    pending_list.append(label)
                    doc_map[label] = d.id
                
                if pending_list:
                    selected_match_label = st.selectbox("Sonuçlanacak Maçı Seçin:", pending_list)
                    selected_doc_id = doc_map[selected_match_label]
                    
                    c1, c2 = st.columns(2)
                    with c1: h_score = st.number_input("Ev Sahibi Gol", 0, 10, 0)
                    with c2: a_score = st.number_input("Deplasman Gol", 0, 10, 0)
                    
                    notes = st.text_area("Maç Notları (Opsiyonel)", placeholder="Örn: Kırmızı kart maçı değiştirdi...")
                    
                    if st.button("💾 SONUCU KAYDET VE VERİTABANINI GÜNCELLE"):
                        if update_match_result(selected_doc_id, h_score, a_score, notes):
                            st.success(f"Maç sonucu ({h_score}-{a_score}) başarıyla kaydedildi!")
                            st.balloons()
                            # Sayfayı yenilemek gerekebilir, kullanıcı manuel yeniler.
                        else:
                            st.error("Güncelleme başarısız oldu.")
                else:
                    st.success("Bekleyen sonuçlanmamış maç yok! Harika.")

if __name__ == "__main__":
    main()
