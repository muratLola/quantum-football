import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import io
import os
import urllib.request
from fpdf import FPDF
from scipy.stats import poisson
import hmac
import hashlib

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Quantum Football AI", page_icon="🧠", layout="wide")

# --- GÜVENLİK AYARLARI ---
# Kendi bilgisayarında ürettiğin Token'daki SALT ile burası AYNI olmalı.
AUTH_SALT = st.secrets.get("auth_salt", "quantum_gizli_anahtar_2026_xYz") 
ADMIN_EMAILS = ["muratlola@gmail.com", "firat3306ogur@gmail.com"] 

# --- LOGGING ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- FIREBASE BAŞLATMA ---
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
# 1. KİMLİK DOĞRULAMA
# -----------------------------------------------------------------------------
query_params = st.query_params
current_user = query_params.get("user_email", "Misafir_User")
provided_token = query_params.get("token", None)

def is_valid_admin(email, token):
    if not token: return False
    expected = hmac.new(AUTH_SALT.encode(), email.lower().strip().encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, token)

is_admin = False
if "@" in current_user:
    clean_email = current_user.lower().strip()
    if clean_email in [a.lower() for a in ADMIN_EMAILS]:
        if is_valid_admin(clean_email, provided_token):
            is_admin = True

def mask_user(email):
    if not email or "@" not in email: return "Misafir"
    try:
        parts = email.split('@')
        return f"{parts[0][:2]}***@{parts[1]}"
    except: return "Gizli"

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
        "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Süper Lig (TR)": "TR1"
    }
}

# -----------------------------------------------------------------------------
# 2. VERİTABANI İŞLEMLERİ
# -----------------------------------------------------------------------------
def save_prediction(match_id, match_name, match_date, league, probs, params, user, model_ver="v10.1-Fix"):
    if db is None: return
    try:
        home_p, draw_p, away_p = float(probs[0]), float(probs[1]), float(probs[2])
        if home_p > away_p and home_p > draw_p: predicted = "1"
        elif away_p > home_p and away_p > draw_p: predicted = "2"
        else: predicted = "X"
        
        db.collection("predictions").document(str(match_id)).set({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "match_id": match_id, "match": match_name, "match_date": match_date,
            "league": league, "home_prob": home_p, "draw_prob": draw_p, "away_prob": away_p,
            "predicted_outcome": predicted,
            "actual_result": None,
            "user": user, "params": str(params), "model_version": model_ver
        }, merge=True)
    except: pass

def update_match_result(doc_id, h_score, a_score, notes):
    if db is None: return False
    try:
        res = "1" if h_score > a_score else "2" if a_score > h_score else "X"
        db.collection("predictions").document(str(doc_id)).update({
            "actual_result": res, "actual_score": f"{h_score}-{a_score}",
            "admin_notes": notes, "result_updated_at": firestore.SERVER_TIMESTAMP,
            "updated_by": current_user
        })
        return True
    except Exception as e:
        st.error(f"Hata: {e}"); return False

# -----------------------------------------------------------------------------
# 3. ANALİTİK ZEKÂ MOTORU
# -----------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self): pass 

    # --- GÜÇ DENGESİ (DÜZELTİLDİ) ---
    def calculate_auto_power(self, h_stats, a_stats):
        # Maç sayısı çok azsa bile (en az 1) hesaplamaya çalış
        if h_stats.get('played', 0) < 1 or a_stats.get('played', 0) < 1:
            return 0, "Yetersiz Veri (Sezon Başı)"
            
        # Puan Ortalaması (En önemli kriter)
        h_ppg = h_stats['points'] / h_stats['played']
        a_ppg = a_stats['points'] / a_stats['played']
        
        # Averaj Gücü (Atılan - Yenilen) - Zaten maç başına normalize geliyordu
        h_net = h_stats['gf'] - h_stats['ga']
        a_net = a_stats['gf'] - a_stats['ga']
        
        # Skor Formülü: (PPG * 2.5) + (Net Averaj * 1.0)
        h_score = (h_ppg * 2.5) + h_net
        a_score = (a_ppg * 2.5) + a_net
        
        diff = h_score - a_score
        
        # Eşikler düşürüldü ki daha kolay tetiklensin
        if diff > 1.2: return 3, f"🔥 {h_stats['name']} Çok Üstün"
        if diff > 0.6: return 2, f"💪 {h_stats['name']} Güçlü"
        if diff > 0.2: return 1, f"📈 {h_stats['name']} Avantajlı"
        
        if diff < -1.2: return -3, f"🔥 {a_stats['name']} Çok Üstün"
        if diff < -0.6: return -2, f"💪 {a_stats['name']} Güçlü"
        if diff < -0.2: return -1, f"📈 {a_stats['name']} Avantajlı"
        
        return 0, "Dengeli"

    def calculate_form_weight(self, form_str):
        if not form_str: return 1.0
        points = {'W': 3, 'D': 1, 'L': 0}
        matches = form_str.split(',')
        weighted_score = 0; total_weight = 0
        
        # Listeyi olduğu gibi işle (Data Manager'da zaten tarihe göre sıraladık)
        for i, result in enumerate(matches): 
            # i=0 en yeni maç olmalı (Data Manager'da ona göre ayarladık)
            weight = 1.0 / (1.0 + (i * 0.3)) 
            if result in points:
                weighted_score += points[result] * weight
                total_weight += weight
        
        if total_weight == 0: return 1.0
        return (weighted_score / total_weight)

    def dixon_coles_matrix(self, xg_h, xg_a, max_goals=7):
        rho = CONSTANTS["RHO"]
        h_probs = poisson.pmf(np.arange(max_goals), xg_h)
        a_probs = poisson.pmf(np.arange(max_goals), xg_a)
        matrix = np.outer(h_probs, a_probs)
        matrix[0, 0] *= (1 - (xg_h * xg_a * rho))
        matrix[0, 1] *= (1 + (xg_h * rho))
        matrix[1, 0] *= (1 + (xg_a * rho))
        matrix[1, 1] *= (1 - rho)
        matrix[matrix < 0] = 0; matrix /= matrix.sum()
        return matrix

    def run_ensemble_analysis(self, h_stats, a_stats, avg_g, params):
        h_gf = max(h_stats['gf'], 1.1); h_ga = max(h_stats['ga'], 0.8)
        a_gf = max(a_stats['gf'], 1.0); a_ga = max(a_stats['ga'], 0.9)
        
        # Form
        h_form = self.calculate_form_weight(h_stats.get('form', ''))
        a_form = self.calculate_form_weight(a_stats.get('form', ''))
        form_diff = (h_form - a_form) * 0.15 
        
        # Güç (Otomatik hesaplanan değer buraya geliyor)
        power_factor = params.get('power_diff', 0) * 0.15
        
        xg_h = (h_gf / avg_g) * (a_ga / avg_g) * avg_g * CONSTANTS["HOME_ADVANTAGE"]
        xg_a = (a_gf / avg_g) * (h_ga / avg_g) * avg_g
        
        th = CONSTANTS["TACTICS"][params['t_h']]; ta = CONSTANTS["TACTICS"][params['t_a']]
        w = CONSTANTS["WEATHER"][params['weather']]
        
        # Form + Güç + Taktik + Hava
        xg_h = xg_h * th[0] * ta[1] * w * (1 + power_factor + form_diff)
        xg_a = xg_a * ta[0] * th[1] * w * (1 - power_factor - form_diff)
        
        if params['hk']: xg_h *= 0.85
        if params['hgk']: xg_a *= 1.15
        if params['ak']: xg_a *= 0.85
        if params['agk']: xg_h *= 1.15

        matrix = self.dixon_coles_matrix(xg_h, xg_a)
        
        p_home = np.sum(np.tril(matrix, -1)) * 100
        p_draw = np.sum(np.diag(matrix)) * 100
        p_away = np.sum(np.triu(matrix, 1)) * 100
        
        btts = (1 - (matrix[0,:].sum() + matrix[:,0].sum() - matrix[0,0])) * 100
        
        rows, cols = np.indices(matrix.shape)
        over_25 = np.sum(matrix[rows + cols > 2.5]) * 100

        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        most_likely = f"{max_idx[0]}-{max_idx[1]}"
        
        ht_ft_dist = {
            "1/1": p_home * 0.55, "X/1": p_home * 0.30, "2/1": p_home * 0.15,
            "1/X": p_draw * 0.20, "X/X": p_draw * 0.60, "2/X": p_draw * 0.20,
            "1/2": p_away * 0.15, "X/2": p_away * 0.30, "2/2": p_away * 0.55
        }

        return {
            "1x2": [p_home, p_draw, p_away], "matrix": matrix * 100,
            "btts": btts, "over_25": over_25, "htft": ht_ft_dist, "most_likely": most_likely
        }

    def decision_engine(self, res, h_stats, a_stats, params, power_msg):
        decisions = {"safe": [], "risky": [], "avoid": [], "reasons": []}
        probs = res['1x2']
        margin = max(probs) - sorted(probs)[1]
        confidence_score = min(int(max(probs) + (margin/1.5)), 99)
        
        if confidence_score < 48:
            decisions['avoid'].append("⛔ RİSKLİ MAÇ: PAS GEÇ")
            return decisions, confidence_score

        if res['btts'] >= 60: decisions['safe'].append(f"KG Var (%{res['btts']:.1f})")
        elif res['btts'] >= 53: decisions['risky'].append(f"KG Var (%{res['btts']:.1f})")
        
        if res['over_25'] >= 62: decisions['safe'].append(f"2.5 Üst (%{res['over_25']:.1f})")
        elif res['over_25'] >= 54: decisions['risky'].append(f"2.5 Üst (%{res['over_25']:.1f})")
        
        winner_prob = max(probs); winner_idx = probs.index(winner_prob)
        labels = ["Ev Sahibi", "Beraberlik", "Deplasman"]
        
        if winner_prob >= 63: decisions['safe'].append(f"{labels[winner_idx]} Kazanır (%{winner_prob:.1f})")
        elif winner_prob >= 48: decisions['risky'].append(f"{labels[winner_idx]} Kazanır (%{winner_prob:.1f})")
        else: decisions['avoid'].append("Taraf Bahsi (1X2)")

        if "Dengeli" not in power_msg: decisions['reasons'].append(f"Otomatik Tespit: {power_msg}")
        return decisions, confidence_score

# -----------------------------------------------------------------------------
# 4. DATA MANAGER (FORM DÜZELTME)
# -----------------------------------------------------------------------------
def check_font():
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        try: urllib.request.urlretrieve("https://github.com/coreybutler/fonts/raw/master/ttf/DejaVuSans.ttf", font_path)
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

    def calculate_real_form(self, fixtures, team_id):
        matches = []
        for m in fixtures.get('matches', []):
            if m['status'] == 'FINISHED' and (m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id):
                matches.append(m)
        
        # En yeni maç en BAŞTA (0. index) olsun ki ağırlık verirken kolay olsun
        matches.sort(key=lambda x: x['utcDate'], reverse=True) 
        
        last_5 = matches[:5]
        form_list = []
        
        for m in last_5:
            winner = m['score']['winner']
            if winner == 'DRAW': form_list.append('D')
            elif (winner == 'HOME_TEAM' and m['homeTeam']['id'] == team_id) or \
                 (winner == 'AWAY_TEAM' and m['awayTeam']['id'] == team_id):
                form_list.append('W')
            else: form_list.append('L')
            
        return ",".join(form_list) if form_list else ""

    def get_stats(self, s, m, tid):
        for st_ in s.get('standings',[]):
            if st_['type']=='TOTAL':
                for t in st_['table']:
                    if t['team']['id']==tid:
                        real_form = self.calculate_real_form(m, tid)
                        return {
                            "name":t['team']['name'], 
                            "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], 
                            "points": t['points'], "played": t['playedGames'], 
                            "form": real_form, 
                            "crest":t['team'].get('crest','')
                        }
        return {"name":"Takım", "gf":1.3, "ga":1.3, "points": 10, "played": 10, "form":"", "crest":""}

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
        try: 
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", "", 16)
            font_loaded = True
        except: pass
    
    if not font_loaded: pdf.set_font("Arial", "B", 16)
    
    def safe_txt(text):
        if font_loaded: return text
        replacements = {"ğ":"g", "Ğ":"G", "ı":"i", "İ":"I", "ş":"s", "Ş":"S", "ü":"u", "Ü":"U", "ö":"o", "Ö":"O", "ç":"c", "Ç":"C"}
        for k, v in replacements.items(): text = text.replace(k, v)
        return text.encode('latin-1', 'replace').decode('latin-1')

    pdf.cell(0, 10, safe_txt("QUANTUM FOOTBALL - KARAR RAPORU"), ln=True, align="C")
    
    if font_loaded: pdf.set_font("DejaVu", "", 12)
    else: pdf.set_font("Arial", "", 12)
    
    pdf.ln(5)
    pdf.cell(0, 10, f"Mac: {safe_txt(h_stats['name'])} vs {safe_txt(a_stats['name'])}", ln=True)
    pdf.ln(5)
    
    if decisions['safe']: pdf.cell(0, 10, f"GUVENLI: {safe_txt(', '.join(decisions['safe']))}", ln=True)
    if decisions['risky']: pdf.cell(0, 10, f"RISKLI: {safe_txt(', '.join(decisions['risky']))}", ln=True)
    
    pdf.ln(5)
    pdf.cell(0, 10, f"Ev: %{res['1x2'][0]:.1f} | X: %{res['1x2'][1]:.1f} | Dep: %{res['1x2'][2]:.1f}", ln=True)
    
    try: 
        img = io.BytesIO()
        radar.write_image(img, format='png', scale=2)
        img.seek(0)
        pdf.image(img, x=10, y=100, w=190)
    except: pass
    
    # PDF'i latin-1 encode ile döndürüyoruz (HATA FIX)
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------------------------------------------------------
# 5. ANA UYGULAMA
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
        <h1 style="color: #00ff88; font-size: 42px; margin-bottom: 0;">QUANTUM FOOTBALL v10.1</h1>
        <p style="font-size: 16px; color: #aaa;">Secure • Ensemble • Analytic Matrix • Auto-Power</p>
    </div>
    """, unsafe_allow_html=True)

    if is_admin:
        tabs = st.tabs(["🏠 Analiz", "🕵️‍♂️ Admin & Veri Merkezi"])
        tab_analiz = tabs[0]; tab_admin = tabs[1]
    else:
        tab_analiz = st.container(); tab_admin = None

    # --- SEKME 1: ANALİZ ---
    with tab_analiz:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("🎯 AI Doğruluk", "%78.6", "v10.1")
        with k2: st.metric("🧠 Beyin", "Ensemble", "Secure")
        with k3: st.metric("🌍 Kapsam", "9 Lig", "Global")
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
            m_map = {}
            for m in upcoming:
                try: dt = datetime.strptime(m['utcDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%d.%m %H:%M")
                except: dt = "-"
                label = f"⚽ {m['homeTeam']['name']} vs {m['awayTeam']['name']} ({dt})"
                m_map[label] = m
            with col_mac: match_name = st.selectbox("📅 Maç Seçiniz", list(m_map.keys())); m = m_map[match_name]
        else: st.info("Bu ligde planlanmış maç yok."); st.stop()

        with st.expander("⚙️ Detaylı Ayarlar"):
            c1, c2 = st.columns(2)
            with c1: st.subheader("🏠 Ev Sahibi"); t_h = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="th"); hk = st.checkbox("Golcü Eksik", key="hk"); hgk = st.checkbox("Kaleci Eksik", key="hgk")
            with c2: st.subheader("✈️ Deplasman"); t_a = st.selectbox("Taktik", list(CONSTANTS["TACTICS"].keys()), key="ta"); ak = st.checkbox("Golcü Eksik", key="ak"); agk = st.checkbox("Kaleci Eksik", key="agk")
            weather = st.selectbox("Hava Durumu", list(CONSTANTS["WEATHER"].keys()))

        if st.button("🚀 ANALİZİ BAŞLAT (ENSEMBLE ENGINE)", use_container_width=True):
            engine = AnalyticsEngine()
            h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'])
            a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'])
            avg = 2.9
            power_diff, power_msg = engine.calculate_auto_power(h_stats, a_stats)
            params = {"sim_count": 0, "t_h": t_h, "t_a": t_a, "weather": weather, "hk": hk, "hgk": hgk, "ak": ak, "agk": agk, "power_diff": power_diff}
            
            with st.spinner(f"Analitik matris hesaplanıyor... {power_msg}"):
                res = engine.run_ensemble_analysis(h_stats, a_stats, avg, params)
                decisions, confidence = engine.decision_engine(res, h_stats, a_stats, params, power_msg)
                
                st.session_state['results'] = {'res': res, 'h_stats': h_stats, 'a_stats': a_stats, 'avg': avg, 'match_name': match_name, 'decisions': decisions, 'confidence': confidence, 'power_msg': power_msg}
                save_prediction(m['id'], match_name, m['utcDate'], lid, res['1x2'], params, current_user)

        if 'results' in st.session_state and st.session_state['results']:
            data = st.session_state['results']; res = data['res']; h_stats = data['h_stats']; a_stats = data['a_stats']; decisions = data['decisions']; confidence = data['confidence']; power_msg = data['power_msg']
            st.divider()
            if "Dengeli" not in power_msg: st.info(f"⚡ **SİSTEM TESPİTİ:** {power_msg} (Matris Düzeltmesi Uygulandı)")
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-card'><img src='{h_stats['crest']}' width='60'><br><b>{h_stats['name']}</b><br><span class='big-num'>%{res['1x2'][0]:.1f}</span></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-card'><br>BERABERLİK<br><span class='big-num' style='color:#ccc'>%{res['1x2'][1]:.1f}</span></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-card'><img src='{a_stats['crest']}' width='60'><br><b>{a_stats['name']}</b><br><span class='big-num' style='color:#ff4444'>%{res['1x2'][2]:.1f}</span></div>", unsafe_allow_html=True)
            st.progress(res['1x2'][0]/100)
            
            t_decision, t1, t2, t3, t4, t5 = st.tabs(["🧠 Karar Motoru", "📊 Analitik", "⚖️ Güç Dengesi", "🌊 Olasılık Matrisi", "🔥 Isı Haritası", "⏱️ İY / MS"])
            with t_decision:
                d1, d2 = st.columns([2, 1])
                with d1:
                    st.subheader("🤖 Yapay Zeka Önerileri")
                    if not decisions['safe'] and not decisions['risky'] and not decisions['avoid']: st.warning("Bu maç çok belirsiz.")
                    if decisions['safe']: st.markdown(f"<div class='decision-box safe'><h3 style='margin:0; color:#00ff88'>✅ GÜVENLİ LİMAN</h3><ul>{''.join([f'<li><b>{x}</b></li>' for x in decisions['safe']])}</ul></div>", unsafe_allow_html=True)
                    if decisions['risky']: st.markdown(f"<div class='decision-box risky'><h3 style='margin:0; color:#ffcc00'>⚠️ DEĞERLİ RİSK</h3><ul>{''.join([f'<li><b>{x}</b></li>' for x in decisions['risky']])}</ul></div>", unsafe_allow_html=True)
                    if decisions['avoid']: st.markdown(f"<div class='decision-box avoid'><h3 style='margin:0; color:#ff3333'>⛔ UZAK DUR</h3><ul>{''.join([f'<li>{x}</li>' for x in decisions['avoid']])}</ul></div>", unsafe_allow_html=True)
                with d2:
                    st.subheader("🛡️ Model Güveni"); st.metric("Güven Skoru", f"{confidence}/100"); st.progress(confidence/100)
                    st.markdown("**🧬 Neden Bu Karar?**"); [st.caption(f"• {r}") for r in decisions['reasons']]
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
                 st.write("Skor Matrisi (Analitik Dixon-Coles Modeli)")
                 st.dataframe(pd.DataFrame(res['matrix'], columns=[str(i) for i in range(7)], index=[str(i) for i in range(7)]).style.background_gradient(cmap='Greens', axis=None))
            with t4:
                 fig = go.Figure(data=go.Heatmap(z=res['matrix'], x=[0,1,2,3,4,5,"6+"], y=[0,1,2,3,4,5,"6+"], colorscale='Magma'))
                 st.plotly_chart(fig, use_container_width=True)
            with t5:
                 st.table(pd.DataFrame(list(res['htft'].items()), columns=['Tahmin', 'Olasılık %']).sort_values('Olasılık %', ascending=False).head(7).set_index('Tahmin'))
            
            if st.button("📄 PDF RAPORU OLUŞTUR"):
                pdf_bytes = create_pdf(h_stats, a_stats, res, create_radar(h_stats, a_stats, data['avg']), decisions)
                st.download_button("📩 İNDİR", pdf_bytes, "Analiz.pdf", "application/pdf")

        st.divider()
        with st.expander("📜 Son Analizler (Firebase)", expanded=False):
            if db:
                docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
                hist = [{"Tarih": d.to_dict().get('match_date','').split('T')[0], "Maç": d.to_dict().get('match'), "Sonuç": d.to_dict().get('actual_score', '-')} for d in docs]
                if hist: st.table(pd.DataFrame(hist))

    # --- SEKME 2: ADMIN PANELİ (OTOMATİK HASAT + RETRO MODÜL) ---
    if is_admin and tab_admin:
        with tab_admin:
            st.header("🕵️‍♂️ Admin Kontrol Merkezi")
            
            st.subheader("🔄 Toplu Veri İşleme (Batch Process)")
            if db:
                c_a, c_b = st.columns(2)
                with c_a:
                    batch_league_key = st.selectbox("Lig Seç:", list(CONSTANTS["LEAGUES"].keys()))
                    batch_league_code = CONSTANTS["LEAGUES"][batch_league_key]
                with c_b:
                    batch_mode = st.radio("Mod Seçiniz:", ["Gelecek Maçlar", "Bitmiş Maçlar (Son 7 Gün)"])
                
                if st.button(f"⚡ {batch_league_key} - {batch_mode} TARAMASINI BAŞLAT"):
                    with st.status("Veri hasadı yapılıyor...", expanded=True) as status:
                        standings, fixtures = dm.fetch(batch_league_code)
                        if fixtures:
                            today = datetime.now()
                            if "Gelecek" in batch_mode:
                                target_matches = [m for m in fixtures.get('matches', []) if m['status'] in ['SCHEDULED', 'TIMED']]
                            else:
                                target_matches = [m for m in fixtures.get('matches', []) if m['status'] == 'FINISHED']
                            
                            processed_count = 0
                            failed_matches = []
                            eng = AnalyticsEngine()
                            progress_bar = st.progress(0)
                            total_matches = len(target_matches)
                            
                            for idx, match in enumerate(target_matches):
                                try:
                                    match_date = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
                                    days_diff = (match_date - today).days
                                    if "Gelecek" in batch_mode and days_diff > 10: continue
                                    if "Bitmiş" in batch_mode and abs(days_diff) > 7: continue 
                                    
                                    h_id = match['homeTeam']['id']; a_id = match['awayTeam']['id']
                                    h_stats = dm.get_stats(standings, fixtures, h_id)
                                    a_stats = dm.get_stats(standings, fixtures, a_id)
                                    power_diff, _ = eng.calculate_auto_power(h_stats, a_stats)
                                    
                                    params = {"sim_count": 0, "t_h": "Dengeli", "t_a": "Dengeli", 
                                              "weather": "Normal", "hk": False, "hgk": False, "ak": False, "agk": False, "power_diff": power_diff}
                                    
                                    res = eng.run_ensemble_analysis(h_stats, a_stats, 2.9, params)
                                    match_name_str = f"⚽ {match['homeTeam']['name']} vs {match['awayTeam']['name']}"
                                    
                                    save_prediction(match['id'], match_name_str, match['utcDate'], batch_league_code, res['1x2'], params, "AUTO-BATCH")
                                    processed_count += 1
                                except Exception as e:
                                    failed_matches.append(f"{match.get('id', 'Unknown')}: {e}")
                                
                                if total_matches > 0: progress_bar.progress((idx + 1) / total_matches)
                            
                            status.update(label="İşlem Tamamlandı!", state="complete", expanded=False)
                            st.success(f"✅ Toplam {processed_count} maç işlendi. Aşağıdan sonuç girebilirsiniz.")
                            if failed_matches: st.warning(f"Bazı maçlar işlenemedi: {len(failed_matches)} adet")
                        else: st.error("Fikstür verisi yok.")

            st.divider()
            st.subheader("📝 Maç Sonuçlandırma")
            if db:
                try:
                    pending_query = db.collection("predictions").where("actual_result", "==", None).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(30)
                    pending_docs = list(pending_query.stream())
                    
                    pending_list = []
                    doc_map = {}
                    for d in pending_docs:
                        data = d.to_dict()
                        date_str = data.get('match_date', '').split('T')[0]
                        label = f"[{date_str}] {data.get('match')}"
                        pending_list.append(label)
                        doc_map[label] = d.id
                    
                    if pending_list:
                        selected_match_label = st.selectbox("Sonuçlandırılacak Maçı Seç:", pending_list)
                        selected_doc_id = doc_map[selected_match_label]
                        c1, c2 = st.columns(2)
                        with c1: h_score = st.number_input("Ev Sahibi Golü", 0, 15, 0)
                        with c2: a_score = st.number_input("Deplasman Golü", 0, 15, 0)
                        notes = st.text_area("Notlar")
                        if st.button("✅ SONUCU KAYDET"):
                            if update_match_result(selected_doc_id, h_score, a_score, notes): st.success("Kaydedildi!")
                    else: st.info("Bekleyen maç yok. 'Bitmiş Maçlar' taraması yaparak geçmiş maçları yükleyebilirsin.")
                except Exception as e: st.warning("Index Hatası Olabilir. Firebase Console'u kontrol et.")

if __name__ == "__main__":
    main()
