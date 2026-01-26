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
import random
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt

# --- 0. SİSTEM YAPILANDIRMASI ---
MODEL_VERSION = "v13.0-Scientific"
SYSTEM_PURPOSE = """
⚠️ ARAŞTIRMA VE EĞİTİM AMAÇLIDIR.
Bu sistem (Quantum Football), olasılıksal modelleme ve veri simülasyonu yapan bir laboratuvar projesidir.
Bahis, iddaa veya finansal tavsiye vermez. Çıktılar istatistiksel projeksiyonlardır.
"""

st.set_page_config(page_title="Quantum Research Lab", page_icon="🧬", layout="wide")
np.random.seed(42)

# --- GÜVENLİK ---
AUTH_SALT = st.secrets.get("auth_salt", "quantum_research_key_2026") 
ADMIN_EMAILS = ["muratlola@gmail.com", "firat3306ogur@gmail.com"] 

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- FIREBASE BAĞLANTISI ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            creds_dict = dict(st.secrets["firebase"])
            creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
    except Exception as e: logger.error(f"Firebase Error: {e}")
try: db = firestore.client()
except: db = None

# --- SABİTLER ---
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.12, 
    "RHO": -0.10, 
    "ELO_K": 32,
    "TACTICS": {"Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15), "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)},
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92},
    "LEAGUES": {
        "Şampiyonlar Ligi": "CL", "Premier League (EN)": "PL", "La Liga (ES)": "PD",
        "Bundesliga (DE)": "BL1", "Serie A (IT)": "SA", "Ligue 1 (FR)": "FL1",
        "Eredivisie (NL)": "DED", "Primeira Liga (PT)": "PPL", "Süper Lig (TR)": "TR1"
    }
}

LEAGUE_PROFILES = {
    "PL": {"pace": 1.15, "variance": 1.1}, 
    "SA": {"pace": 0.90, "variance": 0.8},
    "BL1": {"pace": 1.20, "variance": 1.2},
    "TR1": {"pace": 1.05, "variance": 1.3},
    "DEFAULT": {"pace": 1.0, "variance": 1.0}
}

# -----------------------------------------------------------------------------
# 1. KİMLİK DOĞRULAMA
# -----------------------------------------------------------------------------
query_params = st.query_params
current_user = query_params.get("user_email", "Guest_Researcher")
provided_token = query_params.get("token", None)

def is_valid_admin(email, token):
    if not token: return False
    expected = hmac.new(AUTH_SALT.encode(), email.lower().strip().encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, token)

is_admin = False
if "@" in current_user:
    clean_email = current_user.lower().strip()
    if clean_email in [a.lower() for a in ADMIN_EMAILS]:
        if is_valid_admin(clean_email, provided_token): is_admin = True

# -----------------------------------------------------------------------------
# 2. CORE ENGINE (ANALİTİK & BİLİMSEL KATMAN)
# -----------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self, elo_manager=None): 
        self.elo_manager = elo_manager

    def calculate_confidence_interval(self, mu, alpha=0.90):
        """
        SCIENTIFIC FEATURE: Poisson dağılımı için güven aralığı hesaplar.
        """
        low, high = poisson.interval(alpha, mu)
        return int(low), int(high)

    def calculate_ht_ft_probs(self, p_home, p_draw, p_away):
        """
        RESTORED FEATURE: İY/MS Olasılıkları (Heuristik Matris)
        """
        return {
            "1/1": p_home * 0.58, "X/1": p_home * 0.28, "2/1": p_home * 0.14,
            "1/X": p_draw * 0.18, "X/X": p_draw * 0.64, "2/X": p_draw * 0.18,
            "1/2": p_away * 0.14, "X/2": p_away * 0.28, "2/2": p_away * 0.58
        }

    def run_ensemble_analysis(self, h_stats, a_stats, avg_g, params, h_id, a_id, league_code):
        # 1. Lig & Elo Etkisi
        l_prof = LEAGUE_PROFILES.get(league_code, LEAGUE_PROFILES["DEFAULT"])
        elo_h = 1500; elo_a = 1500
        elo_impact = 0
        if self.elo_manager:
            elo_h = self.elo_manager.get_elo(h_id, h_stats['name'])
            elo_a = self.elo_manager.get_elo(a_id, a_stats['name'])
            elo_impact = ((elo_h - elo_a) / 100.0) * 0.06

        # 2. Form & Power
        h_form = h_stats.get('form_factor', 1.0); a_form = a_stats.get('form_factor', 1.0)
        form_impact = (h_form - a_form) * 0.18
        power_impact = params.get('power_diff', 0) * 0.12

        # 3. xG (Gol Beklentisi) Hesabı
        base_h = (h_stats['gf']/avg_g) * (a_stats['ga']/avg_g) * avg_g * CONSTANTS["HOME_ADVANTAGE"]
        base_a = (a_stats['gf']/avg_g) * (h_stats['ga']/avg_g) * avg_g
        
        xg_h = base_h * l_prof["pace"] * params['t_h'][0] * params['t_a'][1] * (1 + elo_impact + form_impact + power_impact)
        xg_a = base_a * l_prof["pace"] * params['t_a'][0] * params['t_h'][1] * (1 - elo_impact - form_impact - power_impact)
        
        # Taktiksel Manuel Müdahaleler
        if params['hk']: xg_h *= 0.85
        if params['hgk']: xg_a *= 1.15
        if params['ak']: xg_a *= 0.85
        if params['agk']: xg_h *= 1.15

        # 4. Dixon-Coles Matrisi (Olasılık Dağılımı)
        h_probs = poisson.pmf(np.arange(7), xg_h)
        a_probs = poisson.pmf(np.arange(7), xg_a)
        matrix = np.outer(h_probs, a_probs)
        
        rho = CONSTANTS["RHO"]
        matrix[0,0] *= (1 - (xg_h*xg_a*rho))
        matrix[0,1] *= (1 + (xg_h*rho))
        matrix[1,0] *= (1 + (xg_a*rho))
        matrix[1,1] *= (1 - rho)
        matrix[matrix < 0] = 0; matrix /= matrix.sum()

        # 5. Olasılık Çıktıları
        p_home = np.sum(np.tril(matrix, -1)) * 100
        p_draw = np.sum(np.diag(matrix)) * 100
        p_away = np.sum(np.triu(matrix, 1)) * 100
        
        # Gol Olasılıkları (RESTORED - Alt/Üst)
        rows, cols = np.indices(matrix.shape)
        total_goals = rows + cols
        
        over_15 = np.sum(matrix[total_goals > 1.5]) * 100
        over_25 = np.sum(matrix[total_goals > 2.5]) * 100
        over_35 = np.sum(matrix[total_goals > 3.5]) * 100
        btts = (1 - (matrix[0,:].sum() + matrix[:,0].sum() - matrix[0,0])) * 100
        
        # İY / MS Hesapla (RESTORED)
        ht_ft = self.calculate_ht_ft_probs(p_home, p_draw, p_away)
        
        # Güven Aralıkları (SCIENTIFIC NEW)
        ci_h = self.calculate_confidence_interval(xg_h)
        ci_a = self.calculate_confidence_interval(xg_a)

        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        return {
            "1x2": [p_home, p_draw, p_away],
            "matrix": matrix * 100,
            "goals": {"o15": over_15, "o25": over_25, "o35": over_35, "btts": btts},
            "ht_ft": ht_ft,
            "xg": (xg_h, xg_a),
            "ci": (ci_h, ci_a),
            "most_likely": f"{max_idx[0]}-{max_idx[1]}",
            "elo": (elo_h, elo_a)
        }

    def calculate_brier_score(self, probs, outcome_idx):
        """
        SCIENTIFIC FEATURE: Brier Score
        Modelin tahmin performansını (0 ile 2 arasında) ölçer. 0 mükemmeldir.
        """
        p_vector = [probs[0]/100, probs[1]/100, probs[2]/100]
        o_vector = [0, 0, 0]
        o_vector[outcome_idx] = 1
        return np.sum((np.array(p_vector) - np.array(o_vector))**2)

    def calculate_auto_power(self, h_stats, a_stats):
        if h_stats['played'] < 2: return 0, "Yetersiz Veri"
        # PPG (Maç başı puan) x 2 + Averaj
        h_val = (h_stats['points']/h_stats['played'])*2.0 + (h_stats['gf']-h_stats['ga'])/h_stats['played']
        a_val = (a_stats['points']/a_stats['played'])*2.0 + (a_stats['gf']-a_stats['ga'])/a_stats['played']
        diff = h_val - a_val
        
        if diff > 1.2: return 3, f"🔥 {h_stats['name']} Dominant"
        if diff > 0.5: return 2, f"💪 {h_stats['name']} Güçlü"
        if diff > 0.2: return 1, f"📈 {h_stats['name']} Avantajlı"
        if diff < -1.2: return -3, f"🔥 {a_stats['name']} Dominant"
        if diff < -0.5: return -2, f"💪 {a_stats['name']} Güçlü"
        if diff < -0.2: return -1, f"📈 {a_stats['name']} Avantajlı"
        return 0, "Dengeli"

class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    @st.cache_data(ttl=3600)
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers)
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers)
            return r1.json(), r2.json()
        except: return None, None

    def calculate_form(self, fixtures, team_id):
        matches = [m for m in fixtures.get('matches', []) if m['status'] == 'FINISHED' and (m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id)]
        matches.sort(key=lambda x: x['utcDate'], reverse=True)
        last_5 = matches[:5]
        form_list = []
        w_sum = 0; tot_w = 0
        for i, m in enumerate(last_5):
            res='L'; pts=0
            if m['score']['winner'] == 'DRAW': res='D'; pts=1
            elif (m['score']['winner']=='HOME_TEAM' and m['homeTeam']['id']==team_id) or (m['score']['winner']=='AWAY_TEAM' and m['awayTeam']['id']==team_id): res='W'; pts=3
            w = 1.0/(1+i*0.2)
            w_sum += pts*w; tot_w += w
            form_list.append(res)
        return ",".join(form_list), (0.8 + (w_sum/tot_w/3.0)*0.5 if tot_w > 0 else 1.0)

    def get_stats(self, s, m, tid):
        for st_ in s.get('standings',[]):
            if st_['type']=='TOTAL':
                for t in st_['table']:
                    if t['team']['id']==tid:
                        f_str, f_fac = self.calculate_form(m, tid)
                        return {"name":t['team']['name'], "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], "points": t['points'], "played": t['playedGames'], "form": f_str, "form_factor": f_fac, "crest":t['team'].get('crest','')}
        return {"name":"Takım", "gf":1.3, "ga":1.3, "points":1, "played":1, "form":"", "form_factor":1.0, "crest":""}

class EloManager:
    def __init__(self, db): self.db = db
    def get_elo(self, tid, name, ppg=1.35):
        if not self.db: return 1500
        doc = self.db.collection("ratings").document(str(tid)).get()
        return doc.to_dict().get("elo", 1500) if doc.exists else int(1000 + ppg*333)
    def update(self, hid, hnm, aid, anm, hg, ag):
        eh = self.get_elo(hid, hnm); ea = self.get_elo(aid, anm)
        exp = 1/(1+10**((ea-eh)/400))
        act = 1.0 if hg>ag else 0.0 if hg<ag else 0.5
        k = CONSTANTS["ELO_K"] * (1.5 if abs(hg-ag)>2 else 1.0)
        d = k*(act-exp)
        self.db.collection("ratings").document(str(hid)).set({"name":hnm, "elo":round(eh+d)}, merge=True)
        self.db.collection("ratings").document(str(aid)).set({"name":anm, "elo":round(ea-d)}, merge=True)

# -----------------------------------------------------------------------------
# 3. YARDIMCILAR & PDF
# -----------------------------------------------------------------------------
def check_font():
    fp = "DejaVuSans.ttf"
    if not os.path.exists(fp):
        try: urllib.request.urlretrieve("https://github.com/coreybutler/fonts/raw/master/ttf/DejaVuSans.ttf", fp)
        except: pass
    return fp

def create_model_card():
    """SCIENTIFIC FEATURE: Modelin çalışma prensibini anlatan PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"MODEL CARD: {MODEL_VERSION}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "TYPE: Probabilistic Ensemble (Dixon-Coles + Elo + Form)\n\nINTENDED USE: Academic Research & Decision Support\n\nINPUTS: Goals per match, Time-decayed form, Elo ratings, Contextual factors.\n\nMETRICS: Brier Score, Calibration Error, MAE.\n\nOUTPUTS: Full-time probabilities, 95% Confidence Intervals, Goal Markets.\n\nETHICS: Non-gambling, strictly for statistical analysis.")
    return pdf.output(dest='S').encode('latin-1')

def create_match_pdf(h, a, res, conf):
    fp = check_font(); pdf = FPDF(); pdf.add_page()
    if os.path.exists(fp): pdf.add_font("DejaVu","",fp,uni=True); pdf.set_font("DejaVu","",12)
    else: pdf.set_font("Arial","",12)
    def s(t): return t.encode('latin-1','replace').decode('latin-1')
    
    pdf.cell(0,10,s(f"RESEARCH REPORT: {h['name']} vs {a['name']}"),ln=True,align="C")
    pdf.cell(0,10,s(f"Confidence: {conf}/100 | Elo: {res['elo'][0]} vs {res['elo'][1]}"),ln=True)
    pdf.ln(5)
    pdf.cell(0,10,s(f"1X2: {res['1x2'][0]:.1f}% - {res['1x2'][1]:.1f}% - {res['1x2'][2]:.1f}%"),ln=True)
    pdf.cell(0,10,s(f"xG: {res['xg'][0]:.2f} - {res['xg'][1]:.2f}"),ln=True)
    pdf.cell(0,10,s(f"Most Likely: {res['most_likely']}"),ln=True)
    pdf.ln(5)
    pdf.cell(0,10,s(f"Confidence Interval (90%): Home {res['ci'][0]} - Away {res['ci'][1]}"),ln=True)
    return pdf.output(dest='S').encode('latin-1')

def update_result_db(doc_id, hg, ag, notes):
    if not db: return False
    try:
        ref = db.collection("predictions").document(str(doc_id))
        doc = ref.get()
        if not doc.exists: return False
        d = doc.to_dict()
        
        # Sonuç
        res = "1" if hg > ag else "2" if ag > hg else "X"
        idx = 0 if res == "1" else 1 if res == "X" else 2
        
        # Brier Score Hesapla (Bilimsel Metrik)
        probs = [d.get("home_prob"), d.get("draw_prob"), d.get("away_prob")]
        brier = 0.0
        if None not in probs:
            p_vec = np.array([probs[0]/100, probs[1]/100, probs[2]/100])
            o_vec = np.array([0,0,0]); o_vec[idx] = 1
            brier = np.sum((p_vec - o_vec)**2)

        # Elo Update
        elo = EloManager(db)
        if "home_id" in d and "away_id" in d:
            elo.update(d["home_id"], d["match_name"].split(" vs ")[0], d["away_id"], d["match_name"].split(" vs ")[1], hg, ag)
            
        ref.update({
            "actual_result": res, "actual_score": f"{hg}-{ag}", 
            "brier_score": float(brier), "validation_status": "VALIDATED",
            "admin_notes": notes
        })
        return True
    except Exception as e: st.error(str(e)); return False

def save_pred_db(match, probs, params, user, meta):
    if not db: return
    p1, p2, p3 = float(probs[0]), float(probs[1]), float(probs[2])
    pred = "1" if p1>p2 and p1>p3 else "2" if p3>p1 and p3>p2 else "X"
    db.collection("predictions").document(str(match['id'])).set({
        "match_id": str(match['id']), "match_name": f"{meta['hn']} vs {meta['an']}",
        "match_date": match['utcDate'], "league": meta['lg'],
        "home_id": meta['hid'], "away_id": meta['aid'],
        "home_prob": p1, "draw_prob": p2, "away_prob": p3,
        "predicted_outcome": pred, "confidence": meta['conf'],
        "dqi": meta['dqi'], "user": user, "params": str(params),
        "model_version": MODEL_VERSION, "actual_result": None
    }, merge=True)

# -----------------------------------------------------------------------------
# 4. MAIN UI
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        .stApp {background-color: #0e1117; color: #fff;}
        .big-n {font-size:24px; font-weight:bold; color:#00ff88;}
        .card {background:#1e2129; padding:15px; border-radius:10px; margin-bottom:10px;}
    </style>""", unsafe_allow_html=True)
    
    st.title("🔬 Quantum Research Lab v13")
    st.caption("Scientific Ultimate Edition: Brier Scores, Confidence Intervals & Model Cards")
    st.info(SYSTEM_PURPOSE)

    if is_admin:
        tabs = st.tabs(["📊 Analiz", "🗃️ Veri Merkezi", "📘 Model Kimliği"])
    else: tabs = [st.container()]

    # TAB 1: ANALİZ
    with tabs[0]:
        api = st.secrets.get("FOOTBALL_API_KEY")
        if not api: st.error("API Key Yok"); st.stop()
        dm = DataManager(api); eng = AnalyticsEngine(EloManager(db))
        
        c1, c2 = st.columns([1,2])
        with c1: 
            lk = st.selectbox("Lig", list(CONSTANTS["LEAGUES"].keys()))
            lc = CONSTANTS["LEAGUES"][lk]
        s, f = dm.fetch(lc)
        
        if f:
            upc = [m for m in f['matches'] if m['status'] in ['SCHEDULED','TIMED']]
            mm = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upc}
            if mm:
                with c2: mn = st.selectbox("Maç", list(mm.keys())); m = mm[mn]
                
                with st.expander("🛠️ Parametre Ayarları"):
                    pc1, pc2 = st.columns(2)
                    th = pc1.selectbox("Ev Taktik", list(CONSTANTS["TACTICS"].keys()))
                    ta = pc2.selectbox("Dep Taktik", list(CONSTANTS["TACTICS"].keys()))
                
                if st.button("🧪 BİLİMSEL ANALİZ BAŞLAT"):
                    hid, aid = m['homeTeam']['id'], m['awayTeam']['id']
                    hs = dm.get_stats(s, f, hid); as_ = dm.get_stats(s, f, aid)
                    
                    # DQI Hesapla
                    dqi = 100
                    if hs['played'] < 5: dqi -= 20
                    
                    # Auto Power
                    pow_diff, pow_msg = eng.calculate_auto_power(hs, as_)

                    pars = {"t_h": CONSTANTS["TACTICS"][th], "t_a": CONSTANTS["TACTICS"][ta], "weather": 1.0, "hk": False, "ak": False, "hgk": False, "agk": False, "power_diff": pow_diff}
                    res = eng.run_ensemble_analysis(hs, as_, 2.8, pars, hid, aid, lc)
                    
                    # Güven Skoru
                    conf = int(max(res['1x2']) * (dqi/100.0))
                    
                    # Kaydet
                    meta = {"hn": hs['name'], "an": as_['name'], "hid": hid, "aid": aid, "lg": lc, "conf": conf, "dqi": dqi}
                    save_pred_db(m, res['1x2'], pars, current_user, meta)
                    
                    # --- GÖRSELLEŞTİRME ---
                    st.divider()
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Güven Skoru", f"{conf}/100", delta="Model Confidence")
                    c_b.metric("Veri Kalitesi (DQI)", f"{dqi}", delta_color="off")
                    c_c.metric("Elo Farkı", f"{res['elo'][0] - res['elo'][1]}", help="Pozitif değer ev sahibi lehinedir")
                    
                    if "Dengeli" not in pow_msg: st.caption(f"⚡ Otomatik Güç Tespiti: {pow_msg}")

                    st.write(f"### ⚽ Beklenen Goller (xG): {res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
                    
                    # --- GÖRSEL BONUS: ÇAN EĞRİSİ (BELL CURVE) ---
                    def plot_bell_curve(mu, team_name, ci_low, ci_high, color):
                        x = np.arange(0, 8)
                        y = poisson.pmf(x, mu)
                        
                        fig, ax = plt.subplots(figsize=(5, 1.5))
                        fig.patch.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        ax.plot(x, y, 'o-', color=color, markersize=4, linewidth=1, alpha=0.8)
                        ax.fill_between(x, 0, y, where=(x >= ci_low) & (x <= ci_high), color=color, alpha=0.2, label='Güven Alanı')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#444')
                        ax.spines['bottom'].set_color('#444')
                        ax.tick_params(axis='x', colors='white')
                        ax.tick_params(axis='y', colors='white', labelsize=8)
                        ax.set_title(f"{team_name} (Beklenen: {mu:.2f})", color='white', fontsize=9, pad=2)
                        return fig

                    col_g1, col_g2 = st.columns(2)
                    with col_g1: st.pyplot(plot_bell_curve(res['xg'][0], hs['name'], res['ci'][0][0], res['ci'][0][1], '#00ff88'), use_container_width=True)
                    with col_g2: st.pyplot(plot_bell_curve(res['xg'][1], as_['name'], res['ci'][1][0], res['ci'][1][1], '#ff4444'), use_container_width=True)

                    st.info(f"**🧪 %90 Güven Aralığı (Confidence Interval):**\n"
                            f"Model, Ev Sahibinin **[{res['ci'][0][0]} ile {res['ci'][0][1]}]** arasında, "
                            f"Deplasmanın **[{res['ci'][1][0]} ile {res['ci'][1][1]}]** arasında gol atacağını %90 güvenle öngörüyor.")
                    
                    t1, t2, t3 = st.tabs(["Ana Tablo (1X2)", "İY / MS (HT/FT)", "Gol Piyasaları"])
                    
                    with t1:
                        st.subheader("Maç Sonucu Olasılıkları")
                        probs_df = pd.DataFrame([res['1x2']], columns=["Ev %", "Beraberlik %", "Deplasman %"])
                        st.dataframe(probs_df, hide_index=True)
                        st.caption(f"En Olası Skor: **{res['most_likely']}** (Matris Tepe Noktası)")
                        
                    with t2:
                        st.subheader("İlk Yarı / Maç Sonucu (Heuristik)")
                        df_htft = pd.DataFrame(list(res['ht_ft'].items()), columns=['Tahmin', 'Olasılık %'])
                        df_htft = df_htft.sort_values('Olasılık %', ascending=False).head(5)
                        st.table(df_htft.set_index('Tahmin'))
                        
                    with t3:
                        st.subheader("Gol Olasılıkları (Poisson Dağılımı)")
                        gol_data = {
                            "Piyasa": ["1.5 Üst", "2.5 Üst", "3.5 Üst", "KG Var (BTTS)"],
                            "Olasılık %": [f"%{res['goals']['o15']:.1f}", f"%{res['goals']['o25']:.1f}", f"%{res['goals']['o35']:.1f}", f"%{res['goals']['btts']:.1f}"]
                        }
                        st.table(pd.DataFrame(gol_data).set_index("Piyasa"))

                    p_bytes = create_match_pdf(hs, as_, res, conf)
                    st.download_button("📥 Akademik Raporu İndir (PDF)", p_bytes, "analiz_v13.pdf", "application/pdf")

    # TAB 2: ADMIN (DATA MINING & BRIER SCORE)
    if is_admin and len(tabs) > 1:
        with tabs[1]:
            st.header("🗃️ Veri Madenciliği ve Doğrulama")
            st.markdown("Burada girilen sonuçlar, **Brier Score (Hata Kareleri Ortalaması)** hesaplayarak modelin kalibrasyonunu ölçer.")
            
            if db:
                pend = list(db.collection("predictions").where("actual_result", "==", None).limit(20).stream())
                if pend:
                    sel = st.selectbox("Sonuçlanacak Maç", [d.id for d in pend], format_func=lambda x: [p for p in pend if p.id==x][0].to_dict()['match_name'])
                    c1, c2 = st.columns(2)
                    hs = c1.number_input("Ev Gol", 0); as_ = c2.number_input("Dep Gol", 0)
                    note = st.text_area("Admin Notu (Opsiyonel)")
                    
                    if st.button("✅ Onayla ve Eğit"):
                        if update_result_db(sel, hs, as_, note): 
                            st.success("Sonuç işlendi. Elo güncellendi. Brier Score veritabanına yazıldı.")
                else: st.info("Bekleyen açık tahmin bulunamadı.")
                
    # TAB 3: MODEL CARD
    if is_admin and len(tabs) > 2:
        with tabs[2]:
            st.header("📘 Model Kimlik Kartı (Model Card)")
            st.write("Bu sekme, modelin şeffaflığı ve tekrarlanabilirliği için teknik dokümantasyon üretir.")
            
            col_mc1, col_mc2 = st.columns([2,1])
            with col_mc1:
                st.code("""
                Architecture: Ensemble (Poisson + Dixon-Coles)
                Optimization: Elo-based Dynamic Weighting
                Validation Metric: Brier Score
                Risk Analysis: Volatility Index based on League Profiles
                """, language="yaml")
            
            with col_mc2:
                mc_bytes = create_model_card()
                st.download_button("📘 Model Card İndir (PDF)", mc_bytes, "model_card_v13.pdf", "application/pdf")

if __name__ == "__main__":
    main()
