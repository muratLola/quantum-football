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
MODEL_VERSION = "v24.0-PureScience-Lite"

st.set_page_config(page_title="QUANTUM FOOTBALL", page_icon="⚽", layout="wide")
np.random.seed(42)

# CSS (Temiz, Analitik Tasarım)
st.markdown("""
    <style>
        .stApp {background-color: #0b0f19; color: #e0e0e0;}
        .big-metric {font-size: 32px; font-weight: bold; color: #00ff88;}
        .metric-label {font-size: 14px; color: #aaaaaa;}
        .highlight-box {background: rgba(0, 255, 136, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #00ff88;}
        .chaos-box {background: rgba(255, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid #ff4444; text-align: center;}
        .narrative-box {background: rgba(0, 200, 255, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #00c8ff; font-style: italic;}
        .stButton>button {background-color: #00ff88; color: #000; font-weight: bold; border: none; width: 100%; transition: all 0.3s;}
        .stButton>button:hover {background-color: #00cc6a; color: #fff; transform: scale(1.02);}
    </style>
""", unsafe_allow_html=True)

# Dil
TRANS = {
    "EN": {"nav_sim": "🚀 Simulation Lab", "nav_perf": "📈 Model Performance", "nav_admin": "🗃️ Admin", "btn_sim": "⚡ RUN ANALYTICS ENGINE", "scenarios": "📊 Goal Scenarios"},
    "TR": {"nav_sim": "🚀 Simülasyon Laboratuvarı", "nav_perf": "📈 Performans & Kalibrasyon", "nav_admin": "🗃️ Admin", "btn_sim": "⚡ ANALİTİK MOTORUNU ÇALIŞTIR", "scenarios": "📊 Gol Senaryoları"}
}

# Güvenlik & API
AUTH_SALT = st.secrets.get("auth_salt", "quantum_research_key_2026")
ADMIN_EMAILS = ["muratlola@gmail.com", "firat3306ogur@gmail.com"]
logging.basicConfig(level=logging.INFO); logger = logging.getLogger(__name__)

# GEMINI API KEY (Secrets'tan çekiliyor)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

# Firebase Init
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
    except Exception as e: logger.error(f"Firebase Error: {e}")
try: db = firestore.client()
except: db = None

# Sabitler
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4", "ELO_K": 32,
    "TACTICS": {"Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15), "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)},
    "LEAGUES": {"Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD", "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1", "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Süper Lig (TR)": "TR1"}
}
LEAGUE_PROFILES = {"PL": {"pace": 1.15}, "TR1": {"pace": 1.05}, "BL1": {"pace": 1.25}, "SA": {"pace": 0.95}, "DEFAULT": {"pace": 1.0}}

def is_valid_admin(email, token):
    if not token: return False
    return hmac.compare_digest(hmac.new(AUTH_SALT.encode(), email.lower().strip().encode(), hashlib.sha256).hexdigest(), token)

# --- 1. ENGINE: MATEMATİKSEL ÇEKİRDEK ---
class AnalyticsEngine:
    def __init__(self, elo_manager=None): self.elo_manager = elo_manager

    def get_ai_narrative(self, h_name, a_name, probs, entropy, xg_h, xg_a, form_h, form_a):
        static = f"⚠️ **Yüksek Varyans:** {h_name} vs {a_name} maçında belirsizlik yüksek." if entropy > 1.55 else f"✅ **İstatistiksel Avantaj:** {h_name if xg_h > xg_a else a_name}."
        
        # REST API ile Gemini Bağlantısı (Kütüphanesiz - Hızlı Yükleme)
        if GEMINI_API_KEY:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
                headers = {'Content-Type': 'application/json'}
                
                prompt_text = f"""
                Sen bir Veri Bilimcisi ve Spor Analistisin. Asla bahis terimleri kullanma.
                Maç: {h_name} (Form Endeksi: {form_h:.2f}) vs {a_name} (Form Endeksi: {form_a:.2f}).
                Modelin xG Tahmini: {xg_h:.2f} - {xg_a:.2f}.
                Kazanma Olasılıkları: Ev %{probs['1']:.1f}, Dep %{probs['2']:.1f}.
                Entropi (Kaos/Belirsizlik): {entropy:.2f}.
                Bu veriler ışığında, takımların performans beklentilerini teknik bir dille 2 cümleyle özetle.
                """
                
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt_text}]
                    }]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=6)
                
                if response.status_code == 200:
                    ai_text = response.json()['candidates'][0]['content']['parts'][0]['text']
                    return f"🤖 **Gemini AI Analizi:** {ai_text}"
                else:
                    return static
            except Exception as e:
                logger.error(f"AI Error: {e}")
                return static
        return static

    def calculate_match_specific_rho(self, projected_total_xg):
        if projected_total_xg < 2.0: return -0.17 
        elif projected_total_xg < 2.6: return -0.13
        else: return -0.09

    def run_simulation(self, h_stats, a_stats, lg_stats, params, h_id, a_id, league_code, roster_factor, high_precision=False):
        l_prof = LEAGUE_PROFILES.get(league_code, LEAGUE_PROFILES["DEFAULT"])
        
        # Split Vectors (İç Saha / Dış Saha Ayrımı)
        h_att = h_stats.get('home_gf', h_stats['gf']/2) / max(1, h_stats.get('home_played', h_stats['played']/2))
        h_def = h_stats.get('home_ga', h_stats['ga']/2) / max(1, h_stats.get('home_played', h_stats['played']/2))
        a_att = a_stats.get('away_gf', a_stats['gf']/2) / max(1, a_stats.get('away_played', a_stats['played']/2))
        a_def = a_stats.get('away_ga', a_stats['ga']/2) / max(1, a_stats.get('away_played', a_stats['played']/2))

        lg_h_att = lg_stats.get('home_avg_goals', 1.5); lg_a_att = lg_stats.get('away_avg_goals', 1.2)
        has = h_att / lg_h_att; adw = a_def / lg_h_att; aas = a_att / lg_a_att; hdw = h_def / lg_a_att

        # Form (Weighted)
        h_form_final = (h_stats.get('form_home', 1.0) * 0.6) + (h_stats.get('form_overall', 1.0) * 0.4)
        a_form_final = (a_stats.get('form_away', 1.0) * 0.6) + (a_stats.get('form_overall', 1.0) * 0.4)
        form_impact = (h_form_final - a_form_final) * 0.18

        # Elo
        elo_h = self.elo_manager.get_elo(h_id, h_stats['name']) if self.elo_manager else 1500
        elo_a = self.elo_manager.get_elo(a_id, a_stats['name']) if self.elo_manager else 1500
        elo_impact = ((elo_h - elo_a) / 400.0) * 0.12

        # xG Calculation
        xg_h = has * adw * lg_h_att * l_prof["pace"] * roster_factor[0] * (1 + elo_impact + form_impact)
        xg_a = aas * hdw * lg_a_att * l_prof["pace"] * roster_factor[1] * (1 - elo_impact - form_impact)
        xg_h *= params['t_h'][0] * params['t_a'][1]; xg_a *= params['t_a'][0] * params['t_h'][1]

        # Poisson Matrix & Rho Correction (Scientific Approach)
        limit = 10
        h_probs = poisson.pmf(np.arange(limit), xg_h); a_probs = poisson.pmf(np.arange(limit), xg_a)
        matrix = np.outer(h_probs, a_probs)
        rho = self.calculate_match_specific_rho(xg_h + xg_a)
        correction = np.zeros((limit, limit)); correction[0,0] = 1-(xg_h*xg_a*rho); correction[0,1] = 1+(xg_h*rho); correction[1,0] = 1+(xg_a*rho); correction[1,1] = 1-rho
        matrix[0:2, 0:2] *= correction[0:2, 0:2]; matrix[matrix < 0] = 0; matrix /= matrix.sum()

        # Olasılıklar
        p_home = np.sum(np.tril(matrix, -1)) * 100; p_draw = np.sum(np.diag(matrix)) * 100; p_away = np.sum(np.triu(matrix, 1)) * 100
        o25 = np.sum(matrix[np.indices((limit,limit)).sum(0)>2.5])*100
        btts = (1 - matrix[0,:].sum() - matrix[:,0].sum() + matrix[0,0])*100

        # Monte Carlo (High Precision Scientific Simulation)
        if high_precision:
            sims = 50000 
            sim_h = np.random.poisson(xg_h, sims); sim_a = np.random.poisson(xg_a, sims)
            p_home = np.mean(sim_h > sim_a) * 100; p_draw = np.mean(sim_h == sim_a) * 100; p_away = np.mean(sim_h < sim_a) * 100
            o25 = np.mean((sim_h + sim_a) > 2.5) * 100
            btts = np.mean((sim_h > 0) & (sim_a > 0)) * 100

        probs_dict = {"1": p_home, "X": p_draw, "2": p_away}
        entropy = -np.sum((np.array(list(probs_dict.values()))/100) * np.log2((np.array(list(probs_dict.values()))/100) + 1e-9))
        narrative = self.get_ai_narrative(h_stats['name'], a_stats['name'], probs_dict, entropy, xg_h, xg_a, h_form_final, a_form_final)
        ci_h = poisson.interval(0.90, xg_h); ci_a = poisson.interval(0.90, xg_a)

        return {"probs": {"1": p_home, "X": p_draw, "2": p_away, "o25": o25, "btts": btts},
                "xg": (xg_h, xg_a), "elo": (elo_h, elo_a), "score": f"{np.unravel_index(np.argmax(matrix), matrix.shape)[0]}-{np.unravel_index(np.argmax(matrix), matrix.shape)[1]}",
                "matrix": matrix, "vectors": (has, hdw, aas, adw), "narrative": narrative, "entropy": entropy, "ci": (ci_h, ci_a)}

# --- 2. DATA: VERİ YÖNETİMİ ---
class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    @st.cache_data(ttl=3600)
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers).json()
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers).json()
            lg_stats = {"home_avg_goals": 1.5, "away_avg_goals": 1.2}
            team_stats = {}
            if 'standings' in r1:
                h_goals=0; h_games=0; a_goals=0; a_games=0
                for group in r1['standings']:
                    g_type = group['type']
                    for t in group['table']:
                        tid = t['team']['id']
                        if tid not in team_stats: team_stats[tid] = {'name': t['team']['name'], 'crest': t['team']['crest']}
                        if g_type == 'TOTAL': team_stats[tid].update({'gf': t['goalsFor'], 'ga': t['goalsAgainst'], 'played': t['playedGames']})
                        elif g_type == 'HOME': team_stats[tid].update({'home_gf': t['goalsFor'], 'home_ga': t['goalsAgainst'], 'home_played': t['playedGames']}); h_goals+=t['goalsFor']; h_games+=t['playedGames']
                        elif g_type == 'AWAY': team_stats[tid].update({'away_gf': t['goalsFor'], 'away_ga': t['goalsAgainst'], 'away_played': t['playedGames']}); a_goals+=t['goalsFor']; a_games+=t['playedGames']
                if h_games > 0: lg_stats['home_avg_goals'] = h_goals / h_games
                if a_games > 0: lg_stats['away_avg_goals'] = a_goals / a_games
            return team_stats, r2, lg_stats
        except: return {}, {}, {}

    def get_form(self, matches, team_id, filter_type='ALL'):
        played = []
        for m in matches.get('matches', []):
            if m['status'] != 'FINISHED': continue
            is_home = m['homeTeam']['id'] == team_id; is_away = m['awayTeam']['id'] == team_id
            if not (is_home or is_away): continue
            if filter_type == 'HOME' and not is_home: continue
            if filter_type == 'AWAY' and not is_away: continue
            played.append(m)
        played.sort(key=lambda x: x['utcDate'], reverse=True)
        w_sum, tot_w = 0, 0
        for i, m in enumerate(played[:5]):
            pts = 3 if (m['score']['winner']=='HOME_TEAM' and m['homeTeam']['id']==team_id) or (m['score']['winner']=='AWAY_TEAM' and m['awayTeam']['id']==team_id) else 1 if m['score']['winner']=='DRAW' else 0
            weight = 1.0 / (1 + i * 0.5); w_sum += pts * weight; tot_w += weight
        return (0.5 + (w_sum/tot_w/3.0)) if tot_w > 0 else 1.0

# --- 3. ELO & DB ---
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
        
        # Brier & RPS Calculation (Academic Metrics)
        brier = np.sum((p_vec - o_vec)**2)
        rps = (p_vec[0]-o_vec[0])**2 + (p_vec[0]+p_vec[1]-o_vec[0]-o_vec[1])**2 
        
        if "home_id" in d: EloManager(db).update(d["home_id"], "", d["away_id"], "", int(hg), int(ag))
        ref.update({"actual_result": res, "actual_score": f"{hg}-{ag}", "brier_score": float(brier), "rps_score": float(rps), "validation_status": "VALIDATED", "admin_notes": notes})
        return True
    except: return False

def save_pred_db(m, probs, params, user, meta):
    if not db: return
    p1, p2, p3 = float(probs[0]), float(probs[1]), float(probs[2])
    pred = "1" if p1>p2 and p1>p3 else "2" if p3>p1 and p3>p2 else "X"
    doc = db.collection("predictions").document(str(m['id']))
    if not doc.get().exists:
        doc.set({"match_id": str(m['id']), "match_name": f"{meta['hn']} vs {meta['an']}", "match_date": m['utcDate'], "league": meta['lg'], "home_prob": p1, "draw_prob": p2, "away_prob": p3, "predicted_outcome": pred, "confidence": meta['conf'], "dqi": meta['dqi'], "user": user, "params": str(params), "model_version": MODEL_VERSION, "actual_result": None}, merge=True)

# --- 4. VISUALS ---
def create_score_heatmap(matrix, h_name, a_name):
    return go.Figure(data=go.Heatmap(z=matrix[:6, :6], x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], colorscale='Viridis', showscale=False, texttemplate="%{z:.1%}")).update_layout(title=f"{h_name} vs {a_name} Olasılık Dağılımı", xaxis_title=f"{a_name}", yaxis_title=f"{h_name}", width=400, height=400, font=dict(color="white"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

def create_radar(hs, as_, vectors):
    has, hdw, aas, adw = vectors; categories = ['Hücum Gücü', 'Savunma Direnci', 'Form', 'Elo Endeksi']
    h_v = [min(100, has*50), min(100, (2-hdw)*50), hs.get('form_home', 1)*80, 80]
    a_v = [min(100, aas*50), min(100, (2-adw)*50), as_.get('form_away', 1)*80, 75]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_v, theta=categories, fill='toself', name=hs['name'], line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_v, theta=categories, fill='toself', name=as_['name'], line_color='#ff4444'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=350)
    return fig

# --- 5. MAIN ---
def main():
    q = st.query_params; user = q.get("user_email", "Guest"); is_admin = False
    if "@" in user and user.lower().strip() in [a.lower() for a in ADMIN_EMAILS]:
        if is_valid_admin(user.lower().strip(), q.get("token")): is_admin = True

    with st.sidebar:
        st.header("🌐 Dil / Language"); lang = "TR" if "TR" in st.selectbox("Dil", ["Türkçe (TR)", "English (EN)"]) else "EN"; t = TRANS[lang]
        st.divider(); nav = st.radio("Menu", [t["nav_sim"], t["nav_perf"]] + ([t["nav_admin"]] if is_admin else []))
        st.divider()
        high_prec = st.checkbox("Yüksek Hassasiyet (Monte Carlo 50k)", help="Akademik hassasiyet için simülasyon sayısını artırır.")

    st.title("QUANTUM FOOTBALL"); st.caption(f"v{MODEL_VERSION} | AI-Powered Sports Analytics Platform")

    if nav == t["nav_sim"]:
        api = st.secrets.get("FOOTBALL_API_KEY"); dm = DataManager(api); eng = AnalyticsEngine(EloManager(db))
        if not api: st.error("API Key Eksik"); st.stop()
        c1, c2 = st.columns([1, 2]); 
        with c1: lk = st.selectbox("League", list(CONSTANTS["LEAGUES"].keys())); lc = CONSTANTS["LEAGUES"][lk]
        team_stats, fixtures, lg_stats = dm.fetch(lc)
        
        if fixtures:
            upc = [m for m in fixtures['matches'] if m['status'] in ['SCHEDULED','TIMED','IN_PLAY','PAUSED']]
            if not upc: st.warning("Maç yok.")
            else:
                matches_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upc}
                with c2: sel_m = st.selectbox("Match", list(matches_map.keys())); m = matches_map[sel_m]
                
                with st.expander("🛠️ Simülasyon Parametreleri"):
                    ct1, ct2, ct3 = st.columns(3)
                    th = ct1.selectbox("Ev Taktik", list(CONSTANTS["TACTICS"].keys()))
                    ta = ct2.selectbox("Dep Taktik", list(CONSTANTS["TACTICS"].keys()))
                    roster = ct3.selectbox("Kadro Durumu", ["Tam Kadro", "Ev Sahibi Eksik", "Deplasman Eksik"])
                    roster_f = (0.85, 1.0) if roster == "Ev Sahibi Eksik" else (1.0, 0.85) if roster == "Deplasman Eksik" else (1.0, 1.0)

                if st.button(t["btn_sim"]):
                    h_id = m['homeTeam']['id']; a_id = m['awayTeam']['id']
                    hs = team_stats.get(h_id, {'name': m['homeTeam']['name'], 'gf':1, 'ga':1, 'played':1}); as_ = team_stats.get(a_id, {'name': m['awayTeam']['name'], 'gf':1, 'ga':1, 'played':1})
                    
                    hs['form_overall'] = dm.get_form(fixtures, h_id, 'ALL'); hs['form_home'] = dm.get_form(fixtures, h_id, 'HOME')
                    as_['form_overall'] = dm.get_form(fixtures, a_id, 'ALL'); as_['form_away'] = dm.get_form(fixtures, a_id, 'AWAY')
                    pars = {"t_h": CONSTANTS["TACTICS"][th], "t_a": CONSTANTS["TACTICS"][ta]}
                    
                    with st.spinner("Analitik Motoru Çalışıyor..."):
                        res = eng.run_simulation(hs, as_, lg_stats, pars, h_id, a_id, lc, roster_f, high_prec)
                    
                    dqi = 100 if hs.get('played',0) > 5 else 80
                    conf = int(max(res['probs']['1'], res['probs']['X'], res['probs']['2']) * 0.95 * (dqi/100))
                    save_pred_db(m, [res['probs']['1'], res['probs']['X'], res['probs']['2']], pars, user, {'hn': hs['name'], 'an': as_['name'], 'hid': h_id, 'aid': a_id, 'lg': lc, 'conf': conf, 'dqi': dqi})

                    st.markdown(f"<div class='narrative-box'>{res['narrative']}</div>", unsafe_allow_html=True); st.write("")
                    
                    c_h, c_d, c_a = st.columns(3)
                    c_h.markdown(f"<div class='highlight-box'><div class='metric-label'>{hs['name']}</div><div class='big-metric'>%{res['probs']['1']:.1f}</div></div>", unsafe_allow_html=True)
                    c_d.markdown(f"<div class='highlight-box'><div class='metric-label'>X</div><div class='big-metric'>%{res['probs']['X']:.1f}</div></div>", unsafe_allow_html=True)
                    c_a.markdown(f"<div class='highlight-box'><div class='metric-label'>{as_['name']}</div><div class='big-metric'>%{res['probs']['2']:.1f}</div></div>", unsafe_allow_html=True)
                    
                    c_vis1, c_vis2 = st.columns([1, 1])
                    with c_vis1: st.plotly_chart(create_radar(hs, as_, res['vectors']), use_container_width=True)
                    with c_vis2: 
                        st.subheader("🎯 xG & Aralık"); st.metric("Ev", f"{res['xg'][0]:.2f}", f"{int(res['ci'][0][0])}-{int(res['ci'][0][1])}"); st.metric("Dep", f"{res['xg'][1]:.2f}", f"{int(res['ci'][1][0])}-{int(res['ci'][1][1])}")
                        st.plotly_chart(create_score_heatmap(res['matrix'], hs['name'], as_['name']), use_container_width=True)
                    
                    st.subheader(t["scenarios"])
                    m_df = pd.DataFrame({"Senaryo": ["2.5 Gol Üstü", "Karşılıklı Gol (KG Var)"], "Olasılık (%)": [f"%{res['probs']['o25']:.1f}", f"%{res['probs']['btts']:.1f}"]})
                    st.table(m_df)

    elif nav == t["nav_perf"]:
        st.header("📈 Analitik Performans & Kalibrasyon")
        if db:
            docs = list(db.collection("predictions").where("validation_status", "==", "VALIDATED").limit(200).stream())
            if docs:
                total = len(docs); correct = 0; brier_sum = 0; rps_sum = 0; cal_data = []
                for d in docs:
                    dd = d.to_dict(); pred = dd.get("predicted_outcome"); act = dd.get("actual_result")
                    if pred == act: correct += 1
                    brier_sum += dd.get("brier_score", 0); rps_sum += dd.get("rps_score", 0)
                    max_prob = max(dd["home_prob"], dd["draw_prob"], dd["away_prob"])
                    is_correct = 1 if pred == act else 0
                    cal_data.append({"prob": max_prob, "correct": is_correct})
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Analiz Edilen Maç", total); c2.metric("İsabet Oranı", f"%{(correct/total)*100:.1f}"); c3.metric("RPS Skoru (Düşük İyi)", f"{rps_sum/total:.3f}")
                
                st.subheader("📊 Kalibrasyon Eğrisi (Güvenilirlik Testi)")
                df_cal = pd.DataFrame(cal_data)
                df_cal['bin'] = pd.cut(df_cal['prob'], bins=np.arange(0, 101, 10))
                cal_plot = df_cal.groupby('bin').agg({'correct': 'mean', 'prob': 'mean'}).reset_index()
                fig_cal = px.scatter(cal_plot, x='prob', y='correct', title="Tahmin Güveni vs Gerçekleşme (Çizgiye ne kadar yakınsa o kadar bilimsel)", labels={'prob': 'Model Güveni', 'correct': 'Gerçekleşme'})
                fig_cal.add_shape(type="line", x0=0, y0=0, x1=100, y1=1, line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_cal, use_container_width=True)

                st.subheader("Doğrulanmış Sonuçlar")
                data = [{"Maç": d.to_dict().get("match_name"), "Tahmin": d.to_dict().get("predicted_outcome"), "Sonuç": d.to_dict().get("actual_result")} for d in docs]
                st.dataframe(pd.DataFrame(data))
            else: st.info("Henüz yeterli veri seti oluşmadı.")

    elif is_admin and nav == t["nav_admin"]:
        st.header("Admin"); at1, at2 = st.tabs(["Batch", "Validation"])
        with at1:
            lb = st.slider("Geriye Dönük (Gün)", 0, 10, 3)
            if st.button("Batch Run"):
                api = st.secrets.get("FOOTBALL_API_KEY"); dm = DataManager(api); eng = AnalyticsEngine(EloManager(db))
                matches = []; 
                for code in CONSTANTS["LEAGUES"].values(): _, f, _ = dm.fetch(code); matches.extend(f['matches']) if f else None
                cutoff = datetime.utcnow() - timedelta(days=lb)
                target = [m for m in matches if m['status'] in ['SCHEDULED','TIMED','IN_PLAY'] or (m['status']=='FINISHED' and datetime.strptime(m['utcDate'],"%Y-%m-%dT%H:%M:%SZ") > cutoff)]
                pr = st.progress(0)
                for i, m in enumerate(target): pr.progress((i+1)/len(target))
                st.success(f"{len(target)} maç tarandı.")
        with at2:
            if db:
                pend = list(db.collection("predictions").where("actual_result", "==", None).limit(500).stream()); pend.sort(key=lambda x: x.to_dict().get('match_date', '0000'))
                opts = {}; seen = set()
                for d in pend: dd = d.to_dict(); lbl = dd.get('match_name'); dt = str(dd.get('match_date'))[:10]; opts[d.id] = f"{lbl} ({dt})"
                if opts:
                    with st.form("val"):
                        sid = st.selectbox("Maç", list(opts.keys()), format_func=lambda x: opts[x])
                        c1, c2 = st.columns(2); hg = c1.number_input("Ev", 0); ag = c2.number_input("Dep", 0); nt = st.text_area("Not")
                        if st.form_submit_button("Kaydet"): update_result_db(sid, hg, ag, nt); st.success("OK"); time.sleep(1); st.rerun()

if __name__ == "__main__":
    main()
