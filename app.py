import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import time
import hmac
import hashlib
import firebase_admin
from firebase_admin import credentials, firestore
from scipy.stats import poisson

# --- 0. SİSTEM VE KONFIGURASYON ---
MODEL_VERSION = "v28.0-Singularity"

st.set_page_config(page_title="QUANTUM FOOTBALL", page_icon="⚽", layout="wide")
np.random.seed(42)

# CSS (Profesyonel Dashboard)
st.markdown("""
    <style>
        .stApp {background-color: #0b0f19; color: #e0e0e0;}
        .big-metric {font-size: 32px; font-weight: bold; color: #00ff88; font-family: 'Courier New', monospace;}
        .metric-label {font-size: 14px; color: #aaaaaa; text-transform: uppercase; letter-spacing: 1px;}
        .highlight-box {background: rgba(0, 255, 136, 0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(0, 255, 136, 0.2);}
        .narrative-box {background: rgba(0, 200, 255, 0.08); padding: 15px; border-radius: 8px; border-left: 4px solid #00c8ff; font-style: italic; font-family: 'Georgia', serif;}
        .stButton>button {background-color: #00ff88; color: #000; font-weight: 900; border: none; width: 100%; padding: 12px; transition: all 0.3s; text-transform: uppercase;}
        .stButton>button:hover {background-color: #00cc6a; color: #fff; transform: scale(1.02); box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);}
        .success-log {color: #00ff88; font-size: 0.85rem; font-family: monospace; border-left: 2px solid #00ff88; padding-left: 10px; margin-bottom: 5px;}
        .error-log {color: #ff4444; font-size: 0.85rem; font-family: monospace; border-left: 2px solid #ff4444; padding-left: 10px; margin-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# Dil
TRANS = {
    "EN": {"nav_sim": "🚀 Simulation Lab", "nav_perf": "📈 Model Performance", "nav_admin": "🗃️ Admin Core", "btn_sim": "⚡ INITIALIZE QUANTUM ENGINE", "scenarios": "📊 Probabilistic Scenarios"},
    "TR": {"nav_sim": "🚀 Simülasyon Laboratuvarı", "nav_perf": "📈 Performans & Kalibrasyon", "nav_admin": "🗃️ Yönetim Çekirdeği", "btn_sim": "⚡ KUANTUM MOTORUNU BAŞLAT", "scenarios": "📊 Olasılık Senaryoları"}
}

# Güvenlik & API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
FOOTBALL_API_KEY = st.secrets.get("FOOTBALL_API_KEY", None)
ADMIN_PASS = st.secrets.get("auth_salt", "admin123")

# Firebase Init
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
    except Exception as e: st.error(f"Veritabanı Bağlantı Hatası: {e}")
try: db = firestore.client()
except: db = None

CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4", "ELO_K": 32,
    "TACTICS": {"Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15), "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)},
    "LEAGUES": {"Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD", "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1", "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Süper Lig (TR)": "TR1"}
}
LEAGUE_PROFILES = {"PL": {"pace": 1.15}, "TR1": {"pace": 1.08}, "BL1": {"pace": 1.25}, "SA": {"pace": 0.98}, "DEFAULT": {"pace": 1.0}}

# --- 1. ENGINE: MATEMATİKSEL ÇEKİRDEK ---
class AnalyticsEngine:
    def __init__(self, elo_manager=None): self.elo_manager = elo_manager

    # Gemini Cache (Maliyet ve Hız İçin Kritik)
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_cached_ai_narrative(_self, h_name, a_name, probs, entropy, xg_h, xg_a, form_h, form_a):
        static = f"⚠️ **Yüksek Varyans:** {h_name} vs {a_name} maçında belirsizlik yüksek (Entropi: {entropy:.2f})." if entropy > 1.55 else f"✅ **İstatistiksel Avantaj:** {h_name if xg_h > xg_a else a_name} (Güven: %{max(probs.values()):.1f})."
        
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                prompt_text = (
                    f"Sen dünyanın en iyi futbol veri bilimcisisin. Asla bahis terimi kullanma. "
                    f"Maç: {h_name} (Form Endeksi: {form_h:.2f}) vs {a_name} (Form Endeksi: {form_a:.2f}). "
                    f"Modelin xG Tahmini: {xg_h:.2f} - {xg_a:.2f}. "
                    f"Kazanma Olasılıkları: Ev %{probs['1']:.1f}, Beraberlik %{probs['X']:.1f}, Deplasman %{probs['2']:.1f}. "
                    f"Kaos/Entropi: {entropy:.2f} (Yüksek olması sürprize açık demek). "
                    f"Bu veriler ışığında taktiksel ve istatistiksel bir öngörü paragrafı yaz (Max 2 cümle)."
                )
                payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
                response = requests.post(url, headers=headers, json=payload, timeout=8)
                
                if response.status_code == 200:
                    ai_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                    return f"🤖 **Gemini Neural Insight:** {ai_text}"
                else:
                    return f"{static} (AI Status: {response.status_code})"
            except Exception:
                return static
        return static

    def dixon_coles_tau(self, x, y, lambda_h, lambda_a, rho):
        """
        Akademik Dixon-Coles Tau Fonksiyonu (Düzeltilmiş)
        Düşük skorlu maçlardaki (0-0, 1-0, 0-1, 1-1) bağımlılığı modeller.
        """
        if x == 0 and y == 0: return 1 - (lambda_h * lambda_a * rho)
        elif x == 0 and y == 1: return 1 + (lambda_h * rho)
        elif x == 1 and y == 0: return 1 + (lambda_a * rho)
        elif x == 1 and y == 1: return 1 - rho
        else: return 1.0

    def calculate_rho(self, xg_sum):
        # Toplam gol beklentisi düştükçe, 0-0 ihtimali üzerindeki düzeltme katsayısı artmalı
        return -0.13 if xg_sum < 2.5 else -0.1

    def run_simulation(self, h_stats, a_stats, lg_stats, params, h_id, a_id, league_code, roster_factor, high_precision=False):
        l_prof = LEAGUE_PROFILES.get(league_code, LEAGUE_PROFILES["DEFAULT"])
        
        # 1. HÜCUM/SAVUNMA GÜCÜ (Safe Division)
        h_played = max(1, h_stats.get('home_played', h_stats.get('played', 1)/2))
        a_played = max(1, a_stats.get('away_played', a_stats.get('played', 1)/2))
        
        h_att = h_stats.get('home_gf', h_stats.get('gf', 0)/2) / h_played
        h_def = h_stats.get('home_ga', h_stats.get('ga', 0)/2) / h_played
        a_att = a_stats.get('away_gf', a_stats.get('gf', 0)/2) / a_played
        a_def = a_stats.get('away_ga', a_stats.get('ga', 0)/2) / a_played
        
        lg_h_att = lg_stats.get('home_avg_goals', 1.5)
        lg_a_att = lg_stats.get('away_avg_goals', 1.2)
        
        has = h_att / lg_h_att; adw = a_def / lg_h_att
        aas = a_att / lg_a_att; hdw = h_def / lg_a_att
        
        # 2. FORM (EWMA - Exponential Weighted Moving Average Benzeri)
        # Son maçlara daha çok ağırlık veren yapı
        h_form = (h_stats.get('form_home', 1.0) * 0.7) + (h_stats.get('form_overall', 1.0) * 0.3)
        a_form = (a_stats.get('form_away', 1.0) * 0.7) + (a_stats.get('form_overall', 1.0) * 0.3)
        form_impact = (h_form - a_form) * 0.20 # Katsayı artırıldı
        
        # 3. ELO (Logistic Impact - Lineer Yerine)
        elo_h = self.elo_manager.get_elo(h_id, h_stats['name']) if self.elo_manager else 1500
        elo_a = self.elo_manager.get_elo(a_id, a_stats['name']) if self.elo_manager else 1500
        elo_diff = elo_h - elo_a
        # Logistic fonksiyon: Fark 400 ise 10 kat, 0 ise 1 kat.
        elo_multiplier_h = 1 + (1 / (1 + 10 ** (-elo_diff / 400)) - 0.5) * 0.4
        elo_multiplier_a = 1 + (1 / (1 + 10 ** (elo_diff / 400)) - 0.5) * 0.4

        # 4. xG HESAPLAMA (Master Formula)
        xg_h = has * adw * lg_h_att * l_prof["pace"] * roster_factor[0] * elo_multiplier_h * (1 + form_impact)
        xg_a = aas * hdw * lg_a_att * l_prof["pace"] * roster_factor[1] * elo_multiplier_a * (1 - form_impact)
        
        # Taktiksel Çarpanlar
        xg_h *= params['t_h'][0] * params['t_a'][1]
        xg_a *= params['t_a'][0] * params['t_h'][1]
        
        # 5. POISSON & DIXON-COLES
        limit = 10
        h_probs = poisson.pmf(np.arange(limit), xg_h)
        a_probs = poisson.pmf(np.arange(limit), xg_a)
        matrix = np.outer(h_probs, a_probs)
        
        rho = self.calculate_rho(xg_h + xg_a)
        # Dixon-Coles Düzeltmesi (Sadece 0 ve 1 skorları için)
        for i in range(2):
            for j in range(2):
                tau_val = self.dixon_coles_tau(i, j, xg_h, xg_a, rho)
                matrix[i, j] *= tau_val
        
        # Normalizasyon ve Güvenlik
        matrix[matrix < 0] = 0
        sum_probs = matrix.sum()
        if sum_probs > 0: matrix /= sum_probs
        else: matrix[0,0] = 1.0
        
        # 6. SONUÇ ÇIKARIMI
        p_home = np.sum(np.tril(matrix, -1)) * 100
        p_draw = np.sum(np.diag(matrix)) * 100
        p_away = np.sum(np.triu(matrix, 1)) * 100
        o25 = np.sum(matrix[np.indices((limit,limit)).sum(0)>2.5]) * 100
        btts = (1 - matrix[0,:].sum() - matrix[:,0].sum() + matrix[0,0]) * 100
        
        # 7. MONTE CARLO (High Precision Override)
        if high_precision:
            sims = 100000 # 100k Simülasyon
            sim_h = np.random.poisson(xg_h, sims)
            sim_a = np.random.poisson(xg_a, sims)
            p_home = np.mean(sim_h > sim_a) * 100
            p_draw = np.mean(sim_h == sim_a) * 100
            p_away = np.mean(sim_h < sim_a) * 100
            o25 = np.mean((sim_h + sim_a) > 2.5) * 100
            btts = np.mean((sim_h > 0) & (sim_a > 0)) * 100
        
        # 8. SKOR TAHMİNİ (Akıllı Beraberlik Seçimi)
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        if max_idx[0] == max_idx[1]: # Eğer model 0-0 veya 1-1 arasında kaldıysa
             draw_probs = np.diag(matrix)
             best_draw = np.argmax(draw_probs)
             score_str = f"{best_draw}-{best_draw}"
        else:
             score_str = f"{max_idx[0]}-{max_idx[1]}"

        probs_dict = {"1": p_home, "X": p_draw, "2": p_away}
        entropy = -np.sum((np.array(list(probs_dict.values()))/100) * np.log2((np.array(list(probs_dict.values()))/100) + 1e-9))
        
        # Gemini'yi Çağır (Cache'li)
        narrative = self.get_cached_ai_narrative(h_stats['name'], a_stats['name'], probs_dict, entropy, xg_h, xg_a, h_form, a_form)
        ci_h = poisson.interval(0.90, xg_h); ci_a = poisson.interval(0.90, xg_a)
        
        return {
            "probs": {"1": p_home, "X": p_draw, "2": p_away, "o25": o25, "btts": btts},
            "xg": (xg_h, xg_a),
            "elo": (elo_h, elo_a),
            "score": score_str,
            "matrix": matrix,
            "vectors": (has, hdw, aas, adw),
            "narrative": narrative,
            "entropy": entropy,
            "ci": (ci_h, ci_a)
        }

# --- 2. DATA: VERİ YÖNETİMİ ---
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    
    @st.cache_data(ttl=1800) # 30 dk cache
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers).json()
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers).json()
            
            if 'errorCode' in r1 or 'errorCode' in r2:
                st.error(f"API Hatası: {r1.get('errorCode', 'Bilinmeyen Hata')}")
                return {}, {}, {}

            lg_stats = {"home_avg_goals": 1.5, "away_avg_goals": 1.2}; team_stats = {}
            if 'standings' in r1:
                h_goals=0; h_games=0; a_goals=0; a_games=0
                for group in r1['standings']:
                    for t in group['table']:
                        tid = t['team']['id']
                        if tid not in team_stats: team_stats[tid] = {'name': t['team']['name'], 'crest': t['team']['crest']}
                        g_type = group['type']
                        if g_type == 'TOTAL': team_stats[tid].update({'gf': t['goalsFor'], 'ga': t['goalsAgainst'], 'played': t['playedGames']})
                        elif g_type == 'HOME': team_stats[tid].update({'home_gf': t['goalsFor'], 'home_ga': t['goalsAgainst'], 'home_played': t['playedGames']}); h_goals+=t['goalsFor']; h_games+=t['playedGames']
                        elif g_type == 'AWAY': team_stats[tid].update({'away_gf': t['goalsFor'], 'away_ga': t['goalsAgainst'], 'away_played': t['playedGames']}); a_goals+=t['goalsFor']; a_games+=t['playedGames']
                if h_games > 0: lg_stats['home_avg_goals'] = h_goals / h_games
                if a_games > 0: lg_stats['away_avg_goals'] = a_goals / a_games
            return team_stats, r2, lg_stats
        except Exception as e:
            st.error(f"Veri Hatası: {e}")
            return {}, {}, {}

    def get_form(self, matches, team_id, filter_type='ALL'):
        # Gelişmiş Form Hesabı (Tarih Sıralı)
        played = [m for m in matches.get('matches', []) if m['status']=='FINISHED' and (m['homeTeam']['id']==team_id or m['awayTeam']['id']==team_id)]
        if filter_type == 'HOME': played = [m for m in played if m['homeTeam']['id']==team_id]
        if filter_type == 'AWAY': played = [m for m in played if m['awayTeam']['id']==team_id]
        
        played.sort(key=lambda x: x['utcDate'], reverse=True) # En yeni en üstte
        w_sum, tot_w = 0, 0
        
        # Son 5 maç, üstel ağırlık (En yeni maç en değerli)
        weights = [1.0, 0.85, 0.70, 0.55, 0.40]
        
        for i, m in enumerate(played[:5]):
            pts = 3 if (m['score']['winner']=='HOME_TEAM' and m['homeTeam']['id']==team_id) or (m['score']['winner']=='AWAY_TEAM' and m['awayTeam']['id']==team_id) else 1 if m['score']['winner']=='DRAW' else 0
            w = weights[i]
            w_sum += pts * w
            tot_w += w
            
        return (0.5 + (w_sum/tot_w/3.0)) if tot_w > 0 else 1.0

# --- 3. DB & SYNC ---
class EloManager:
    def __init__(self, db): self.db = db
    def get_elo(self, tid, name):
        if not self.db: return 1500
        doc = self.db.collection("ratings").document(str(tid)).get()
        return doc.to_dict().get("elo", 1500) if doc.exists else 1500
    def update(self, hid, hnm, aid, anm, hg, ag):
        eh = self.get_elo(hid, hnm); ea = self.get_elo(aid, anm)
        exp = 1/(1+10**((ea-eh)/400)); act = 1.0 if hg>ag else 0.0 if hg<ag else 0.5
        k = CONSTANTS["ELO_K"] * (1.5 if abs(hg-ag)>2 else 1.0); d = k*(act-exp)
        self.db.collection("ratings").document(str(hid)).set({"name":hnm, "elo":round(eh+d)}, merge=True)
        self.db.collection("ratings").document(str(aid)).set({"name":anm, "elo":round(ea-d)}, merge=True)

def update_result_db(doc_id, hg, ag, notes):
    if not db: return False
    try:
        ref = db.collection("predictions").document(str(doc_id)); d = ref.get().to_dict()
        res = "1" if int(hg) > int(ag) else "2" if int(ag) > int(hg) else "X"
        p_vec = np.array([d["home_prob"], d["draw_prob"], d["away_prob"]]) / 100; o_vec = np.zeros(3); o_vec[0 if res=="1" else 1 if res=="X" else 2] = 1
        brier = np.sum((p_vec - o_vec)**2)
        rps = (p_vec[0]-o_vec[0])**2 + (p_vec[0]+p_vec[1]-o_vec[0]-o_vec[1])**2 
        if "home_id" in d and "away_id" in d: 
            EloManager(db).update(d["home_id"], "", d["away_id"], "", int(hg), int(ag))
        ref.update({"actual_result": res, "actual_score": f"{hg}-{ag}", "brier_score": float(brier), "rps_score": float(rps), "validation_status": "VALIDATED", "admin_notes": notes})
        return True
    except Exception as e: st.error(f"DB Update Error: {e}"); return False

def auto_sync_results():
    if not db or not FOOTBALL_API_KEY: return 0
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}
    count = 0
    pending_docs = list(db.collection("predictions").where("actual_result", "==", None).stream())
    if not pending_docs: st.info("Senkronize edilecek eksik maç yok."); return 0
    
    st.info(f"{len(pending_docs)} maç kontrol ediliyor (Smart Sync)...")
    
    # Tarih filtresi (Son 10 gün) - API yükünü azaltmak için
    date_from = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
    target_leagues = set([d.to_dict().get("league") for d in pending_docs])
    
    for code in target_leagues:
        try:
            time.sleep(6) # Anti-Ban Delay
            url = f"{CONSTANTS['API_URL']}/competitions/{code}/matches?status=FINISHED&dateFrom={date_from}"
            r = requests.get(url, headers=headers).json()
            if 'matches' not in r: continue
            
            finished_matches = {str(m['id']): m for m in r['matches']}
            for doc in pending_docs:
                d = doc.to_dict(); mid = str(doc.id)
                if d.get("league") == code and mid in finished_matches:
                    m = finished_matches[mid]
                    update_result_db(mid, m['score']['fullTime']['home'], m['score']['fullTime']['away'], "Smart-Sync v28")
                    count += 1
                    st.markdown(f"<div class='success-log'>✅ {d['match_name']}: {m['score']['fullTime']['home']}-{m['score']['fullTime']['away']}</div>", unsafe_allow_html=True)
        except Exception as e: st.markdown(f"<div class='error-log'>⚠️ {code} Sync Hatası: {e}</div>", unsafe_allow_html=True)
    return count

def save_pred_db(m, probs, params, user, meta):
    if not db: return
    p1, p2, p3 = float(probs[0]), float(probs[1]), float(probs[2])
    doc = db.collection("predictions").document(str(m['id']))
    if not doc.get().exists:
        doc.set({
            "match_id": str(m['id']), "match_name": f"{meta['hn']} vs {meta['an']}", "match_date": m['utcDate'], 
            "league": meta['lg'], "home_prob": p1, "draw_prob": p2, "away_prob": p3, 
            "predicted_outcome": ("1" if p1>p2 and p1>p3 else "2" if p3>p1 and p3>p2 else "X"), 
            "confidence": meta['conf'], "dqi": meta['dqi'], "user": user, 
            "params": str(params), "model_version": MODEL_VERSION, "actual_result": None,
            "home_id": meta['hid'], "away_id": meta['aid']
        }, merge=True)

# --- 4. VISUALS ---
def create_score_heatmap(matrix, h_name, a_name):
    return go.Figure(data=go.Heatmap(z=matrix[:6, :6], x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], colorscale='Viridis', showscale=False, texttemplate="%{z:.1%}")).update_layout(title=f"{h_name} vs {a_name} Score Probability", xaxis_title=a_name, yaxis_title=h_name, width=400, height=400, font=dict(color="white"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

def create_radar(hs, as_, vectors):
    categories = ['Attack', 'Defense', 'Form (EWMA)', 'Elo Index']
    h_v = [min(100, vectors[0]*50), min(100, (2-vectors[1])*50), hs.get('form_home', 1)*80, 85]
    a_v = [min(100, vectors[2]*50), min(100, (2-vectors[3])*50), as_.get('form_away', 1)*80, 80]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_v, theta=categories, fill='toself', name=hs['name'], line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_v, theta=categories, fill='toself', name=as_['name'], line_color='#ff4444'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=350)
    return fig

# --- 5. MAIN ---
def main():
    q = st.query_params; user = q.get("user_email", "Guest"); is_admin = False
    
    # Session State Init
    if 'admin_logged_in' not in st.session_state: st.session_state.admin_logged_in = False

    with st.sidebar:
        st.header("🌐 Dil / Language"); lang = "TR" if "TR" in st.selectbox("Dil", ["Türkçe (TR)", "English (EN)"]) else "EN"; t = TRANS[lang]
        st.divider(); nav = st.radio("Menu", [t["nav_sim"], t["nav_perf"], t["nav_admin"]])
        
        if nav == t["nav_admin"]:
            st.divider()
            if not st.session_state.admin_logged_in:
                p = st.text_input("🔑 Admin Access", type="password")
                if p == ADMIN_PASS: st.session_state.admin_logged_in = True; st.success("Access Granted"); st.rerun()
            else:
                st.success("Yönetici Oturumu Aktif")
                if st.button("Çıkış Yap"): st.session_state.admin_logged_in = False; st.rerun()
                is_admin = True

        st.divider(); high_prec = st.checkbox("Monte Carlo (100k Sim)", help="Akademik hassasiyet için simülasyon sayısını artırır.")

    st.title("QUANTUM FOOTBALL"); st.caption(f"v{MODEL_VERSION} | AI-Powered Predictive Analytics")

    if nav == t["nav_sim"]:
        if not FOOTBALL_API_KEY: st.error("API Key Eksik"); st.stop()
        dm = DataManager(FOOTBALL_API_KEY); eng = AnalyticsEngine(EloManager(db))
        c1, c2 = st.columns([1, 2])
        with c1: lk = st.selectbox("League", list(CONSTANTS["LEAGUES"].keys())); lc = CONSTANTS["LEAGUES"][lk]
        team_stats, fixtures, lg_stats = dm.fetch(lc)
        
        if fixtures:
            upc = [m for m in fixtures['matches'] if m['status'] in ['SCHEDULED','TIMED','IN_PLAY','PAUSED']]
            if not upc: st.warning("Bu ligde yaklaşan maç bulunamadı.")
            else:
                matches_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upc}
                with c2: sel_m = st.selectbox("Match Selection", list(matches_map.keys())); m = matches_map[sel_m]
                with st.expander("🛠️ Tactical Parameters"):
                    ct1, ct2, ct3 = st.columns(3)
                    th = ct1.selectbox("Home Tactics", list(CONSTANTS["TACTICS"].keys()))
                    ta = ct2.selectbox("Away Tactics", list(CONSTANTS["TACTICS"].keys()))
                    roster = ct3.selectbox("Roster Status", ["Full Squad", "Home Missing Key", "Away Missing Key"])
                    roster_f = (0.85, 1.0) if roster == "Home Missing Key" else (1.0, 0.85) if roster == "Away Missing Key" else (1.0, 1.0)

                if st.button(t["btn_sim"]):
                    h_id = m['homeTeam']['id']; a_id = m['awayTeam']['id']
                    hs = team_stats.get(h_id, {'name': m['homeTeam']['name'], 'gf':1, 'ga':1, 'played':1}); as_ = team_stats.get(a_id, {'name': m['awayTeam']['name'], 'gf':1, 'ga':1, 'played':1})
                    hs['form_overall'] = dm.get_form(fixtures, h_id, 'ALL'); hs['form_home'] = dm.get_form(fixtures, h_id, 'HOME')
                    as_['form_overall'] = dm.get_form(fixtures, a_id, 'ALL'); as_['form_away'] = dm.get_form(fixtures, a_id, 'AWAY')
                    
                    with st.spinner("Quantum Engine Processing..."):
                        pars = {"t_h": CONSTANTS["TACTICS"][th], "t_a": CONSTANTS["TACTICS"][ta]}
                        res = eng.run_simulation(hs, as_, lg_stats, pars, h_id, a_id, lc, roster_f, high_prec)
                    
                    dqi = 100 if hs.get('played',0) > 5 else 80
                    conf = int(max(res['probs']['1'], res['probs']['X'], res['probs']['2']) * 0.95 * (dqi/100))
                    save_pred_db(m, [res['probs']['1'], res['probs']['X'], res['probs']['2']], pars, user, {'hn': hs['name'], 'an': as_['name'], 'hid': h_id, 'aid': a_id, 'lg': lc, 'conf': conf, 'dqi': dqi})

                    st.markdown(f"<div class='narrative-box'>{res['narrative']}</div>", unsafe_allow_html=True); st.write("")
                    c_h, c_d, c_a = st.columns(3)
                    c_h.markdown(f"<div class='highlight-box'><div class='metric-label'>{hs['name']}</div><div class='big-metric'>%{res['probs']['1']:.1f}</div></div>", unsafe_allow_html=True)
                    c_d.markdown(f"<div class='highlight-box'><div class='metric-label'>DRAW</div><div class='big-metric'>%{res['probs']['X']:.1f}</div></div>", unsafe_allow_html=True)
                    c_a.markdown(f"<div class='highlight-box'><div class='metric-label'>{as_['name']}</div><div class='big-metric'>%{res['probs']['2']:.1f}</div></div>", unsafe_allow_html=True)
                    
                    c_v1, c_v2 = st.columns([1,1])
                    with c_v1: st.plotly_chart(create_radar(hs, as_, res['vectors']), use_container_width=True)
                    with c_v2: 
                        st.subheader("🎯 Expected Goals (xG)"); st.metric("Home", f"{res['xg'][0]:.2f}", f"CI: {int(res['ci'][0][0])}-{int(res['ci'][0][1])}"); st.metric("Away", f"{res['xg'][1]:.2f}", f"CI: {int(res['ci'][1][0])}-{int(res['ci'][1][1])}")
                        st.plotly_chart(create_score_heatmap(res['matrix'], hs['name'], as_['name']), use_container_width=True)
                    st.subheader(t["scenarios"]); st.table(pd.DataFrame({"Scenario": ["Over 2.5 Goals", "Both Teams to Score"], "Probability": [f"%{res['probs']['o25']:.1f}", f"%{res['probs']['btts']:.1f}"]}))

    elif nav == t["nav_perf"]:
        st.header("📈 Model Performance Audit")
        if db:
            docs = list(db.collection("predictions").where("validation_status", "==", "VALIDATED").limit(200).stream())
            if docs:
                total = len(docs); correct = 0; brier_sum = 0; rps_sum = 0; cal_data = []
                for d in docs:
                    dd = d.to_dict(); pred = dd.get("predicted_outcome"); act = dd.get("actual_result")
                    if pred == act: correct += 1
                    brier_sum += dd.get("brier_score", 0); rps_sum += dd.get("rps_score", 0)
                    cal_data.append({"prob": max(dd["home_prob"], dd["draw_prob"], dd["away_prob"]), "correct": 1 if pred == act else 0})
                c1, c2, c3 = st.columns(3); c1.metric("Validated Matches", total); c2.metric("Accuracy", f"%{(correct/total)*100:.1f}"); c3.metric("RPS (Lower is Better)", f"{rps_sum/total:.4f}")
                st.subheader("Calibration Curve"); df_cal = pd.DataFrame(cal_data); df_cal['bin'] = pd.cut(df_cal['prob'], bins=np.arange(0, 101, 10))
                cal_plot = df_cal.groupby('bin').agg({'correct': 'mean', 'prob': 'mean'}).reset_index()
                st.plotly_chart(px.scatter(cal_plot, x='prob', y='correct', title="Reliability Diagram", labels={'prob': 'Confidence', 'correct': 'Accuracy'}).add_shape(type="line", x0=0, y0=0, x1=100, y1=1, line=dict(color="Red", dash="dash")), use_container_width=True)
                st.dataframe(pd.DataFrame([{"Match": d.to_dict().get("match_name"), "Prediction": d.to_dict().get("predicted_outcome"), "Result": d.to_dict().get("actual_result")} for d in docs]))
            else: st.info("No validated data available yet.")

    elif nav == t["nav_admin"]:
        if is_admin:
            st.header("🛡️ Admin Core"); at1, at2, at3 = st.tabs(["Smart Sync", "Manual Entry", "System"])
            with at1:
                st.info("Algoritma: Veritabanındaki eksik maçları (None) bulur ve sadece ilgili ligleri, son 10 gün filtresiyle tarar.")
                if st.button("🔄 EXECUTE SMART SYNC"):
                    with st.spinner("Synchronizing with API..."):
                        c = auto_sync_results()
                        if c > 0: st.success(f"{c} matches synchronized & validated!")
                        else: st.warning("All records are up to date.")
            with at2:
                if db:
                    pend = list(db.collection("predictions").where("actual_result", "==", None).limit(500).stream()); pend.sort(key=lambda x: x.to_dict().get('match_date', '0000'))
                    opts = {d.id: f"{d.to_dict().get('match_name')} ({str(d.to_dict().get('match_date'))[:10]})" for d in pend}
                    if opts:
                        with st.form("val"):
                            sid = st.selectbox("Select Pending Match", list(opts.keys()), format_func=lambda x: opts[x])
                            c1, c2 = st.columns(2); hg = c1.number_input("Home Goals", 0); ag = c2.number_input("Away Goals", 0); nt = st.text_area("Audit Notes")
                            if st.form_submit_button("Commit Result"): update_result_db(sid, hg, ag, nt); st.success("Record Updated"); time.sleep(1); st.rerun()
            with at3:
                if st.button("🧹 Cache Temizle (Force Refresh)"):
                    st.cache_data.clear()
                    st.success("Sistem önbelleği temizlendi.")
        else: st.error("Access Denied.")

if __name__ == "__main__":
    main()
