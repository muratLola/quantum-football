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
import functools

# --- 0. SİSTEM VE KONFIGURASYON ---
MODEL_VERSION = "v31.0-Lazarus"

st.set_page_config(page_title="QUANTUM FOOTBALL", page_icon="⚽", layout="wide")
np.random.seed(42)

# CSS
st.markdown("""
    <style>
        .stApp {background-color: #050505; color: #e0e0e0;}
        .big-metric {font-size: 36px; font-weight: 800; color: #00ff88; font-family: 'Helvetica Neue', sans-serif;}
        .metric-label {font-size: 12px; color: #888; letter-spacing: 2px; text-transform: uppercase;}
        .highlight-box {background: linear-gradient(145deg, rgba(0,255,136,0.05), rgba(0,0,0,0)); padding: 20px; border-radius: 15px; border: 1px solid rgba(0, 255, 136, 0.15);}
        .narrative-box {background: rgba(10, 20, 40, 0.8); padding: 20px; border-radius: 10px; border-left: 3px solid #00c8ff; font-family: 'Calibri', sans-serif; font-size: 1.1rem;}
        .stButton>button {background: linear-gradient(90deg, #00ff88, #00cc6a); color: #000; font-weight: 900; border: none; width: 100%; padding: 14px; text-transform: uppercase; border-radius: 8px;}
        .stButton>button:hover {transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0, 255, 136, 0.3);}
        .success-log {color: #00ff88; font-family: monospace; font-size: 0.8rem; padding: 2px;}
        .error-log {color: #ff5555; font-family: monospace; font-size: 0.8rem; padding: 2px;}
    </style>
""", unsafe_allow_html=True)

# Dil
TRANS = {
    "EN": {"nav_sim": "🚀 Analysis Lab", "nav_perf": "📈 Integrity Check", "nav_admin": "🗃️ System Core", "btn_sim": "⚡ EXECUTE PREDICTION MODEL", "scenarios": "📊 Probability Matrices"},
    "TR": {"nav_sim": "🚀 Analiz Laboratuvarı", "nav_perf": "📈 Doğruluk Kontrolü", "nav_admin": "🗃️ Sistem Çekirdeği", "btn_sim": "⚡ TAHMİN MODELİNİ ÇALIŞTIR", "scenarios": "📊 Olasılık Matrisleri"}
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
    except Exception as e: st.error(f"FATAL: Database Connection Failed - {e}")
try: db = firestore.client()
except: db = None

CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4", "ELO_K": 32,
    "TACTICS": {"Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15), "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)},
    "LEAGUES": {"Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD", "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1", "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Süper Lig (TR)": "TR1"}
}

LEAGUE_PROFILES = {
    "PL": {"pace": 1.15, "ha": 1.12},
    "TR1": {"pace": 1.08, "ha": 1.25},
    "BL1": {"pace": 1.25, "ha": 1.15},
    "SA": {"pace": 0.98, "ha": 1.10},
    "DEFAULT": {"pace": 1.0, "ha": 1.10}
}

# --- DECORATOR: API RETRY ---
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        return {}, {}, {}
                    time.sleep(backoff_in_seconds * 2 ** x)
                    x += 1
        return wrapper
    return decorator

# --- 1. ENGINE ---
class AnalyticsEngine:
    def __init__(self, elo_manager=None): self.elo_manager = elo_manager

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_cached_ai_narrative(_self, h_name, a_name, probs, entropy, xg_h, xg_a, form_h, form_a):
        static = f"⚠️ **Yüksek Belirsizlik:** {h_name} vs {a_name} (Entropi: {entropy:.2f})." if entropy > 1.58 else f"✅ **Model Favorisi:** {h_name if xg_h > xg_a else a_name} (Güven: %{max(probs.values()):.1f})."
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                prompt_text = (f"Futbol analisti. Maç: {h_name} (Form: {form_h:.2f}) - {a_name} (Form: {form_a:.2f}). xG: {xg_h:.2f}-{xg_a:.2f}. Olasılıklar: Ev %{probs['1']:.1f}, Dep %{probs['2']:.1f}. Kaos: {entropy:.2f}. 2 cümlelik teknik analiz.")
                payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
                response = requests.post(url, headers=headers, json=payload, timeout=9)
                if response.status_code == 200:
                    return f"🤖 **Gemini AI:** {response.json()['candidates'][0]['content']['parts'][0]['text']}"
            except: pass
        return static

    def dixon_coles_tau(self, x, y, lh, la, rho):
        if x==0 and y==0: return 1 - (lh*la*rho)
        elif x==0 and y==1: return 1 + (lh*rho)
        elif x==1 and y==0: return 1 + (la*rho)
        elif x==1 and y==1: return 1 - rho
        return 1.0

    def get_dynamic_rho(self, total_xg):
        if total_xg < 2.0: return -0.16
        if total_xg < 2.8: return -0.12
        return -0.09

    def run_simulation(self, h_stats, a_stats, lg_stats, params, h_id, a_id, league_code, roster_factor, high_precision=False):
        l_prof = LEAGUE_PROFILES.get(league_code, LEAGUE_PROFILES["DEFAULT"])
        
        # 1. TEMEL DEĞİŞKENLER (FIX: NameError Çözümü için değişkenlere atama)
        h_played = max(1, h_stats.get('home_played', h_stats.get('played', 1)/2))
        a_played = max(1, a_stats.get('away_played', a_stats.get('played', 1)/2))
        
        # Saldırı/Savunma Güçleri
        h_att_val = h_stats.get('home_gf', 0) / h_played / lg_stats.get('home_avg_goals', 1.5)
        h_def_val = h_stats.get('home_ga', 0) / h_played / lg_stats.get('away_avg_goals', 1.2)
        a_att_val = a_stats.get('away_gf', 0) / a_played / lg_stats.get('away_avg_goals', 1.2)
        a_def_val = a_stats.get('away_ga', 0) / a_played / lg_stats.get('home_avg_goals', 1.5)
        
        # FIX: Grafiklerde kullanılacak değişkenleri burada tanımlıyoruz
        has = h_att_val
        hdw = h_def_val
        aas = a_att_val
        adw = a_def_val
        
        # 2. EWMA FORM
        h_form = (h_stats.get('form_home', 1.0)*0.65 + h_stats.get('form_overall', 1.0)*0.35)
        a_form = (a_stats.get('form_away', 1.0)*0.65 + a_stats.get('form_overall', 1.0)*0.35)
        form_diff = (h_form - a_form) * 0.22 

        # 3. ELO & HOME ADVANTAGE
        elo_h = self.elo_manager.get_elo(h_id, h_stats['name']) if self.elo_manager else 1500
        elo_a = self.elo_manager.get_elo(a_id, a_stats['name']) if self.elo_manager else 1500
        elo_diff = elo_h - elo_a
        elo_prob_h = 1 / (1 + 10 ** (-elo_diff / 400))
        elo_mult = 1 + (elo_prob_h - 0.5) * 0.5

        # 4. xG HESAPLAMA
        xg_h = has * adw * lg_stats.get('home_avg_goals', 1.5) * l_prof["pace"] * roster_factor[0] * elo_mult * (1+form_diff) * params['t_h'][0] * params['t_a'][1] * l_prof["ha"]
        xg_a = aas * hdw * lg_stats.get('away_avg_goals', 1.2) * l_prof["pace"] * roster_factor[1] * (2-elo_mult) * (1-form_diff) * params['t_a'][0] * params['t_h'][1]
        
        limit = 10; h_probs = poisson.pmf(np.arange(limit), xg_h); a_probs = poisson.pmf(np.arange(limit), xg_a)
        matrix = np.outer(h_probs, a_probs)
        
        rho = self.get_dynamic_rho(xg_h + xg_a)
        for i in range(2):
            for j in range(2):
                matrix[i,j] *= self.dixon_coles_tau(i, j, xg_h, xg_a, rho)
        
        matrix[matrix < 0] = 0; s = matrix.sum(); matrix /= s if s > 0 else 1
        
        p_home = np.sum(np.tril(matrix, -1)) * 100
        p_draw = np.sum(np.diag(matrix)) * 100
        p_away = np.sum(np.triu(matrix, 1)) * 100
        o25 = np.sum(matrix[np.indices((limit,limit)).sum(0)>2.5]) * 100
        btts = (1 - matrix[0,:].sum() - matrix[:,0].sum() + matrix[0,0]) * 100
        
        if high_precision:
            sims = 100000; sh = np.random.poisson(xg_h, sims); sa = np.random.poisson(xg_a, sims)
            p_home = np.mean(sh > sa)*100; p_draw = np.mean(sh == sa)*100; p_away = np.mean(sh < sa)*100
            o25 = np.mean((sh+sa)>2.5)*100; btts = np.mean((sh>0)&(sa>0))*100

        midx = np.unravel_index(np.argmax(matrix), matrix.shape)
        score_str = f"{midx[0]}-{midx[1]}"
        if midx[0] == midx[1]: score_str = f"{np.argmax(np.diag(matrix))}-{np.argmax(np.diag(matrix))}"

        probs = {"1": p_home, "X": p_draw, "2": p_away}
        entropy = -np.sum((np.array(list(probs.values()))/100) * np.log2((np.array(list(probs.values()))/100) + 1e-9))
        narrative = self.get_cached_ai_narrative(h_stats['name'], a_stats['name'], probs, entropy, xg_h, xg_a, h_form, a_form)
        ci_h = poisson.interval(0.90, xg_h); ci_a = poisson.interval(0.90, xg_a)
        
        return {"probs": {"1": p_home, "X": p_draw, "2": p_away, "o25": o25, "btts": btts}, "xg": (xg_h, xg_a), "elo": (elo_h, elo_a), "score": score_str, "matrix": matrix, "vectors": (has, hdw, aas, adw), "narrative": narrative, "entropy": entropy, "ci": (ci_h, ci_a)}

# --- 2. DATA ---
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    
    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    @st.cache_data(ttl=1800)
    def fetch(_self, league):
        r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers).json()
        r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers).json()
        if 'errorCode' in r1: return {}, {}, {}
        
        lg_stats = {"home_avg_goals": 1.5, "away_avg_goals": 1.2}; team_stats = {}
        if 'standings' in r1:
            h_g, h_p, a_g, a_p = 0,0,0,0
            for group in r1['standings']:
                for t in group['table']:
                    tid = t['team']['id']
                    if tid not in team_stats: team_stats[tid] = {'name': t['team']['name'], 'crest': t['team']['crest']}
                    g_type = group['type']
                    if g_type == 'TOTAL': team_stats[tid].update({'gf': t['goalsFor'], 'ga': t['goalsAgainst'], 'played': t['playedGames']})
                    elif g_type == 'HOME': 
                        team_stats[tid].update({'home_gf': t['goalsFor'], 'home_ga': t['goalsAgainst'], 'home_played': t['playedGames']})
                        h_g+=t['goalsFor']; h_p+=t['playedGames']
                    elif g_type == 'AWAY': 
                        team_stats[tid].update({'away_gf': t['goalsFor'], 'away_ga': t['goalsAgainst'], 'away_played': t['playedGames']})
                        a_g+=t['goalsFor']; a_p+=t['playedGames']
            if h_p > 0: lg_stats = {'home_avg_goals': h_g/h_p, 'away_avg_goals': a_g/a_p}
        return team_stats, r2, lg_stats

    def get_form(self, matches, team_id, filter_type='ALL'):
        played = [m for m in matches.get('matches', []) if m['status']=='FINISHED' and (m['homeTeam']['id']==team_id or m['awayTeam']['id']==team_id)]
        if filter_type == 'HOME': played = [m for m in played if m['homeTeam']['id']==team_id]
        if filter_type == 'AWAY': played = [m for m in played if m['awayTeam']['id']==team_id]
        played.sort(key=lambda x: x['utcDate'], reverse=True)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]; w_sum = 0; tot = 0
        for i, m in enumerate(played[:5]):
            pts = 3 if (m['score']['winner']=='HOME_TEAM' and m['homeTeam']['id']==team_id) or (m['score']['winner']=='AWAY_TEAM' and m['awayTeam']['id']==team_id) else 1 if m['score']['winner']=='DRAW' else 0
            w_sum += pts * weights[i]; tot += weights[i]
        return (0.5 + (w_sum/tot/3.0)) if tot > 0 else 1.0

# --- 3. DB & SYNC ---
class EloManager:
    def __init__(self, db): self.db = db
    def get_elo(self, tid, name):
        if not self.db: return 1500
        doc = self.db.collection("ratings").document(str(tid)).get()
        return doc.to_dict().get("elo", 1500) if doc.exists else 1500
    def update(self, hid, hnm, aid, anm, hg, ag):
        eh = self.get_elo(hid, hnm); ea = self.get_elo(aid, anm)
        ex = 1/(1+10**((ea-eh)/400)); act = 1.0 if hg>ag else 0.0 if hg<ag else 0.5
        k = CONSTANTS["ELO_K"] * (1.5 if abs(hg-ag)>2 else 1.0); d = k*(act-ex)
        self.db.collection("ratings").document(str(hid)).set({"name":hnm, "elo":round(eh+d)}, merge=True)
        self.db.collection("ratings").document(str(aid)).set({"name":anm, "elo":round(ea-d)}, merge=True)

def update_result_db(doc_id, hg, ag, notes):
    if not db: return False
    try:
        ref = db.collection("predictions").document(str(doc_id)); d = ref.get().to_dict()
        res = "1" if int(hg)>int(ag) else "2" if int(ag)>int(hg) else "X"
        # FIX: Güvenli veri çekme (varsayılan değerler)
        p = np.array([d.get("home_prob",33), d.get("draw_prob",33), d.get("away_prob",33)])/100
        o = np.zeros(3); o[0 if res=="1" else 1 if res=="X" else 2] = 1
        brier = np.sum((p-o)**2); rps = (p[0]-o[0])**2 + (p[0]+p[1]-o[0]-o[1])**2 
        if "home_id" in d and "away_id" in d: EloManager(db).update(d["home_id"], "", d["away_id"], "", int(hg), int(ag))
        ref.update({"actual_result": res, "actual_score": f"{hg}-{ag}", "brier_score": float(brier), "rps_score": float(rps), "validation_status": "VALIDATED", "admin_notes": notes})
        return True
    except: return False

def auto_sync_results():
    if not db or not FOOTBALL_API_KEY: return 0
    headers = {"X-Auth-Token": FOOTBALL_API_KEY}; count = 0
    pending = list(db.collection("predictions").where("actual_result", "==", None).stream())
    
    if not pending: st.info("Veritabanında eksik maç yok."); return 0
    
    # FIX: Tarih filtresini kaldır ve son 60 günü zorla tara (En garanti yol)
    date_from = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
    leagues = set([d.to_dict().get("league") for d in pending])
    st.info(f"{len(pending)} açık tahmin bulundu. Son 60 gün taranıyor...")
    
    for code in leagues:
        try:
            time.sleep(6) # Anti-Ban
            url = f"{CONSTANTS['API_URL']}/competitions/{code}/matches?status=FINISHED&dateFrom={date_from}"
            r = requests.get(url, headers=headers).json()
            if 'matches' not in r: continue
            
            fin = {str(m['id']): m for m in r['matches']}
            for doc in pending:
                d = doc.to_dict(); mid = str(doc.id)
                if d.get("league") == code and mid in fin:
                    m = fin[mid]; hg=m['score']['fullTime']['home']; ag=m['score']['fullTime']['away']
                    if hg is not None:
                        update_result_db(mid, hg, ag, "AutoSync v31")
                        count += 1
                        st.markdown(f"<div class='success-log'>✅ {d['match_name']}: {hg}-{ag}</div>", unsafe_allow_html=True)
        except Exception as e: st.markdown(f"<div class='error-log'>⚠️ Sync Error ({code}): {e}</div>", unsafe_allow_html=True)
    return count

def save_pred_db(m, probs, params, user, meta):
    if not db: return
    p1,p2,p3 = float(probs[0]), float(probs[1]), float(probs[2])
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
    return go.Figure(data=go.Heatmap(z=matrix[:6, :6], x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], colorscale='Viridis', showscale=False, texttemplate="%{z:.1%}")).update_layout(title=f"Score Probabilities: {h_name} vs {a_name}", xaxis_title=a_name, yaxis_title=h_name, width=400, height=400, font=dict(color="white"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

def create_radar(hs, as_, vectors):
    cats = ['Attack', 'Defense', 'Momentum (Form)', 'Elo Rating']
    # FIX: Değişken isimleri düzeltildi (vectors tuple'dan çekildi)
    h_v = [min(100, vectors[0]*50), min(100, (2-vectors[1])*50), hs.get('form_home', 1)*80, 85]
    a_v = [min(100, vectors[2]*50), min(100, (2-vectors[3])*50), as_.get('form_away', 1)*80, 80]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_v, theta=cats, fill='toself', name=hs['name'], line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_v, theta=cats, fill='toself', name=as_['name'], line_color='#ff4444'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=350)
    return fig

# --- 5. MAIN ---
def main():
    if 'admin' not in st.session_state: st.session_state.admin = False
    
    with st.sidebar:
        st.header("🌐 Language"); lang = "TR" if "TR" in st.selectbox("Select", ["Türkçe (TR)", "English (EN)"]) else "EN"; t = TRANS[lang]
        st.divider(); nav = st.radio("Navigation", [t["nav_sim"], t["nav_perf"], t["nav_admin"]])
        
        if nav == t["nav_admin"]:
            st.divider()
            if not st.session_state.admin:
                if st.text_input("🔑 Password", type="password") == ADMIN_PASS:
                    st.session_state.admin = True; st.success("Authorized"); st.rerun()
            else:
                if st.button("Logout"): st.session_state.admin = False; st.rerun()
        
        st.divider(); high_prec = st.checkbox("Monte Carlo (100k)", help="Use experimental simulation")

    st.title("QUANTUM FOOTBALL"); st.caption(f"{MODEL_VERSION} | Neural-Statistical Hybrid Engine")

    if nav == t["nav_sim"]:
        if not FOOTBALL_API_KEY: st.error("System Halted: API Key Missing"); st.stop()
        dm = DataManager(FOOTBALL_API_KEY); eng = AnalyticsEngine(EloManager(db))
        c1, c2 = st.columns([1, 2])
        with c1: lk = st.selectbox("League", list(CONSTANTS["LEAGUES"].keys())); lc = CONSTANTS["LEAGUES"][lk]
        team_stats, fixtures, lg_stats = dm.fetch(lc)
        
        if fixtures:
            upc = [m for m in fixtures['matches'] if m['status'] in ['SCHEDULED','TIMED','IN_PLAY','PAUSED']]
            if not upc: st.warning("No upcoming matches found.")
            else:
                matches_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upc}
                with c2: sel_m = st.selectbox("Match", list(matches_map.keys())); m = matches_map[sel_m]
                with st.expander("⚙️ Advanced Parameters"):
                    c_1, c_2, c_3 = st.columns(3)
                    th = c_1.selectbox("Home Tactic", list(CONSTANTS["TACTICS"].keys()))
                    ta = c_2.selectbox("Away Tactic", list(CONSTANTS["TACTICS"].keys()))
                    roster = c_3.selectbox("Availability", ["Full Squad", "Home Missing Key", "Away Missing Key"])
                    rf = (0.85, 1.0) if roster == "Home Missing Key" else (1.0, 0.85) if roster == "Away Missing Key" else (1.0, 1.0)

                if st.button(t["btn_sim"]):
                    h_id = m['homeTeam']['id']; a_id = m['awayTeam']['id']
                    hs = team_stats.get(h_id, {'name': m['homeTeam']['name']}); as_ = team_stats.get(a_id, {'name': m['awayTeam']['name']})
                    hs['form_overall'] = dm.get_form(fixtures, h_id, 'ALL'); hs['form_home'] = dm.get_form(fixtures, h_id, 'HOME')
                    as_['form_overall'] = dm.get_form(fixtures, a_id, 'ALL'); as_['form_away'] = dm.get_form(fixtures, a_id, 'AWAY')
                    
                    with st.spinner("Processing Quantum Models..."):
                        pars = {"t_h": CONSTANTS["TACTICS"][th], "t_a": CONSTANTS["TACTICS"][ta]}
                        res = eng.run_simulation(hs, as_, lg_stats, pars, h_id, a_id, lc, rf, high_prec)
                    
                    dqi = 100 if hs.get('home_played',0) > 3 else 75
                    conf = int(max(res['probs']['1'], res['probs']['X'], res['probs']['2']) * 0.9 * (dqi/100))
                    save_pred_db(m, [res['probs']['1'], res['probs']['X'], res['probs']['2']], pars, user, {'hn': hs['name'], 'an': as_['name'], 'hid': h_id, 'aid': a_id, 'lg': lc, 'conf': conf, 'dqi': dqi})

                    st.markdown(f"<div class='narrative-box'>{res['narrative']}</div>", unsafe_allow_html=True); st.write("")
                    c_h, c_d, c_a = st.columns(3)
                    c_h.markdown(f"<div class='highlight-box'><div class='metric-label'>{hs['name']}</div><div class='big-metric'>%{res['probs']['1']:.1f}</div></div>", unsafe_allow_html=True)
                    c_d.markdown(f"<div class='highlight-box'><div class='metric-label'>DRAW</div><div class='big-metric'>%{res['probs']['X']:.1f}</div></div>", unsafe_allow_html=True)
                    c_a.markdown(f"<div class='highlight-box'><div class='metric-label'>{as_['name']}</div><div class='big-metric'>%{res['probs']['2']:.1f}</div></div>", unsafe_allow_html=True)
                    
                    v1, v2 = st.columns([1,1])
                    with v1: st.plotly_chart(create_radar(hs, as_, res['vectors']), use_container_width=True)
                    with v2: 
                        st.subheader("🎯 Expected Goals (xG)"); st.metric("Home", f"{res['xg'][0]:.2f}", f"CI: {res['ci'][0]}"); st.metric("Away", f"{res['xg'][1]:.2f}", f"CI: {res['ci'][1]}")
                        st.plotly_chart(create_score_heatmap(res['matrix'], hs['name'], as_['name']), use_container_width=True)
                    st.subheader(t["scenarios"]); st.table(pd.DataFrame({"Scenario": ["Over 2.5", "BTTS (KG Var)"], "Probability": [f"%{res['probs']['o25']:.1f}", f"%{res['probs']['btts']:.1f}"]}))

    elif nav == t["nav_perf"]:
        st.header("📈 Validation Center")
        if db:
            docs = list(db.collection("predictions").where("validation_status", "==", "VALIDATED").limit(200).stream())
            # FIX: KeyError'u önleyen güvenli filtreleme (predicted_outcome'u olmayanları atla)
            valid_docs = [d for d in docs if d.to_dict().get("predicted_outcome") and d.to_dict().get("actual_result")]
            
            if valid_docs:
                total = len(valid_docs)
                correct = sum(1 for d in valid_docs if d.to_dict().get("predicted_outcome") == d.to_dict().get("actual_result"))
                rps = sum(d.to_dict().get("rps_score", 0) for d in valid_docs) / total
                c1, c2, c3 = st.columns(3); c1.metric("Samples", total); c2.metric("Accuracy", f"%{(correct/total)*100:.1f}"); c3.metric("RPS Score", f"{rps:.4f}")
                
                cal_data = [{"prob": max(d.to_dict().get("home_prob",0), d.to_dict().get("draw_prob",0), d.to_dict().get("away_prob",0)), 
                             "correct": 1 if d.to_dict().get("predicted_outcome") == d.to_dict().get("actual_result") else 0} 
                            for d in valid_docs]
                
                df_cal = pd.DataFrame(cal_data); df_cal['bin'] = pd.cut(df_cal['prob'], bins=np.arange(0, 101, 10))
                cal_plot = df_cal.groupby('bin').agg({'correct': 'mean', 'prob': 'mean'}).reset_index()
                st.plotly_chart(px.scatter(cal_plot, x='prob', y='correct', title="Reliability Diagram", labels={'prob':'Model Confidence', 'correct':'Real Accuracy'}).add_shape(type="line", x0=0,y0=0,x1=100,y1=1, line=dict(color="red", dash="dash")), use_container_width=True)
                st.dataframe(pd.DataFrame([{"Match": d.to_dict().get("match_name"), "Pred": d.to_dict().get("predicted_outcome"), "Result": d.to_dict().get("actual_result")} for d in valid_docs]))
            else: st.info("No validated records yet.")

    elif nav == t["nav_admin"]:
        if st.session_state.admin:
            st.header("🛡️ System Core"); at1, at2, at3 = st.tabs(["Smart Sync", "Manual", "Tools"])
            with at1:
                st.info("Algoritma: Veritabanındaki eksik maçları (None) bulur ve tarih farketmeksizin son 60 günü tarayarak eşleştirir.")
                if st.button("🔄 START AUTO-SYNC (60 DAYS FORCE)"):
                    with st.spinner("Deep scanning..."):
                        c = auto_sync_results()
                        if c > 0: st.success(f"{c} matches synced.")
                        else: st.warning("Up to date.")
            with at2:
                if db:
                    pend = list(db.collection("predictions").where("actual_result", "==", None).limit(500).stream()); pend.sort(key=lambda x: x.to_dict().get('match_date', '0000'))
                    opts = {d.id: f"{d.to_dict().get('match_name')} ({str(d.to_dict().get('match_date'))[:10]})" for d in pend}
                    if opts:
                        with st.form("val"):
                            sid = st.selectbox("Select Match", list(opts.keys()), format_func=lambda x: opts[x])
                            c1, c2 = st.columns(2); hg = c1.number_input("HG", 0); ag = c2.number_input("AG", 0); nt = st.text_area("Note")
                            if st.form_submit_button("Save"): update_result_db(sid, hg, ag, nt); st.success("Saved"); time.sleep(1); st.rerun()
            with at3:
                if st.button("🧹 Flush Cache"): st.cache_data.clear(); st.success("Cache Cleared.")
        else: st.error("Access Denied.")

if __name__ == "__main__":
    main()
