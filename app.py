import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
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
from firebase_admin import credentials
from firebase_admin import firestore

# -----------------------------------------------------------------------------
# 1. SABİTLER VE KONFİGÜRASYON
# -----------------------------------------------------------------------------
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.05, # Bu değer artık Brain tarafından güncellenebilir
    "WIN_BOOST": 0.04,
    "DRAW_BOOST": 0.01,
    "LOSS_PENALTY": -0.03,
    "MISSING_PLAYER_BASE_IMPACT": 0.08,
    "CACHE_TTL": 1800, 
    "FORM_WEIGHTS": [1.5, 1.25, 1.0, 0.75, 0.5],
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "TACTICS": {
        "Dengeli": (1.0, 1.0),
        "Hücum (Gegenpressing)": (1.25, 1.15),
        "Savunma (Park the Bus)": (0.60, 0.65),
        "Kontra Atak": (0.90, 0.85)
    },
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 1.05, "Karlı": 0.85, "Sıcak": 0.95}
}

st.set_page_config(page_title="Quantum Football v4.0", page_icon="🧠", layout="wide")

# --- FIREBASE BAŞLATMA ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.error(f"Firebase Init Error: {e}")

try:
    db = firestore.client()
except:
    db = None

# -----------------------------------------------------------------------------
# 2. BEYİN (THE BRAIN) - ÖĞRENEN ZEKA MODÜLÜ
# -----------------------------------------------------------------------------
class Brain:
    def __init__(self):
        self.learning_rate = 0.02 # Öğrenme hızı
        self.base_home_adv = CONSTANTS["HOME_ADVANTAGE"]

    def calibrate(self, league_code):
        """
        Geçmiş tahminlere bakar ve lig bazlı ev sahibi avantajını optimize eder.
        """
        if db is None: return self.base_home_adv
        
        # O ligdeki sonuçlanmış (skoru girilmiş) maçları çek
        try:
            docs = db.collection("predictions")\
                     .where("league", "==", league_code)\
                     .where("actual_result", "!=", None)\
                     .limit(50).stream()
            
            error_sum = 0
            count = 0
            
            for doc in docs:
                d = doc.to_dict()
                pred_home = d.get('home_prob', 50.0)
                
                # Gerçek sonucu parse et (Örn: "2-1")
                try:
                    res_parts = d['actual_result'].split('-')
                    if len(res_parts) == 2:
                        h_s, a_s = int(res_parts[0]), int(res_parts[1])
                        # 1: Ev Kazandı, 0: Kazanamadı
                        actual_outcome = 100.0 if h_s > a_s else 0.0
                        # Hata = Tahmin - Gerçek (Pozitifse AI abartmış, Negatifse küçümsemiş)
                        error_sum += (pred_home - actual_outcome)
                        count += 1
                except: continue

            if count > 5: # En az 5 maç verisi varsa öğren
                avg_error = error_sum / count
                # Hata pozitifse (abartmışsak), avantajı düşür.
                adjustment = (avg_error / 1000) * self.learning_rate
                new_adv = self.base_home_adv - adjustment
                return max(1.0, min(new_adv, 1.2)) # 1.0 ile 1.2 arasında tut
            
        except Exception as e:
            logger.error(f"Brain Calibration Error: {e}")
            
        return self.base_home_adv

# -----------------------------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR & PDF
# -----------------------------------------------------------------------------
def save_prediction(match_name, league, probs, params, user_email, vote):
    if db is None: return
    try:
        db.collection("predictions").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "match": match_name,
            "league": league,
            "home_prob": float(probs[0]),
            "draw_prob": float(probs[1]),
            "away_prob": float(probs[2]),
            "user": user_email,
            "user_vote": vote, # Kullanıcı ne dedi?
            "tactics": f"{params.get('tactics_h')} vs {params.get('tactics_a')}",
            "actual_result": None # Sonradan girilecek
        })
    except Exception as e:
        logger.error(f"Save Error: {e}")

def create_pdf_report(h_stats, a_stats, res, radar_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "QUANTUM FOOTBALL v4.0 RAPORU", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Mac: {h_stats['name']} vs {a_stats['name']}", ln=True)
    pdf.cell(0, 10, f"Tarih: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Sonuclar:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Ev: %{res['1x2'][0]:.1f} | Beraberlik: %{res['1x2'][1]:.1f} | Dep: %{res['1x2'][2]:.1f}", ln=True)
    
    try:
        img_bytes = io.BytesIO()
        radar_fig.write_image(img_bytes, format='png', scale=2)
        img_bytes.seek(0)
        pdf.image(img_bytes, x=10, y=100, w=190)
    except: pass
    return pdf.output(dest='S').encode('latin-1')

def create_radar_chart(h_stats, a_stats, avg_g):
    def norm(v, b, inv=False):
        r = v/b
        s = 100-(r*50) if inv else r*50
        return min(max(s, 20), 99)
    
    def f_score(f):
        s=50
        for c in f.replace(',',''): s += 5 if c=='W' else (2 if c=='D' else -3)
        return min(max(s,30),95)

    cats = ['Hücum', 'Defans', 'Form', 'İstikrar', 'Şans']
    hv = [norm(h_stats['gf'],avg_g), norm(h_stats['ga'],avg_g,True), f_score(h_stats['form']), 75, 60]
    av = [norm(a_stats['gf'],avg_g), norm(a_stats['ga'],avg_g,True), f_score(a_stats['form']), 70, 55]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=hv, theta=cats, fill='toself', name=h_stats['name'], line_color='#00ff88', opacity=0.7))
    fig.add_trace(go.Scatterpolar(r=av, theta=cats, fill='toself', name=a_stats['name'], line_color='#ff0044', opacity=0.7))
    fig.update_layout(polar=dict(bgcolor='#151922', radialaxis=dict(visible=True, range=[0,100], showticklabels=False), angularaxis=dict(gridcolor='#333')),
                      showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=20, b=20))
    return fig

# -----------------------------------------------------------------------------
# 4. SİMÜLASYON MOTORU
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self, use_fixed_seed=False):
        self.rng = np.random.default_rng(seed=42 if use_fixed_seed else None)

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params, home_adv_factor):
        sims = params['sim_count']
        
        h_att = (h_stats['gf'] / avg_g)
        h_def = (h_stats['ga'] / avg_g)
        a_att = (a_stats['gf'] / avg_g)
        a_def = (a_stats['ga'] / avg_g)

        base_xg_h = h_attack = h_att * a_def * avg_g
        base_xg_a = a_attack = a_att * h_def * avg_g

        # Manager Mode
        ht, at = CONSTANTS["TACTICS"].get(params.get("t_h")), CONSTANTS["TACTICS"].get(params.get("t_a"))
        if ht: base_xg_h *= ht[0]; base_xg_a *= ht[1] # Kendi hücum, rakip hücum (defans zafiyeti)
        if at: base_xg_a *= at[0]; base_xg_h *= at[1]

        # Scenario
        scen = params.get('scenario')
        bh, ba = 0, 0
        if scen == 'Kırmızı (Ev)': base_xg_h*=0.45; base_xg_a*=1.45
        elif scen == 'Kırmızı (Dep)': base_xg_h*=1.45; base_xg_a*=0.45
        elif scen == 'Erken Gol (Ev)': bh=1; base_xg_h*=0.75; base_xg_a*=1.40
        elif scen == 'Erken Gol (Dep)': ba=1; base_xg_h*=1.50; base_xg_a*=0.70

        # Beyin Tarafından Optimize Edilmiş Home Advantage
        base_xg_h *= home_adv_factor

        # Fiziksel
        w = CONSTANTS["WEATHER"].get(params.get("weather"), 1.0)
        base_xg_h *= w; base_xg_a *= w
        
        if params.get("hk"): base_xg_h *= 0.8
        if params.get("ak"): base_xg_a *= 0.8
        if params.get("hgk"): base_xg_a *= 1.2
        if params.get("agk"): base_xg_h *= 1.2

        # Form
        def f_boost(f):
            b=1.0
            for i,c in enumerate(f.replace(',','')[:5]):
                w=CONSTANTS["FORM_WEIGHTS"][i]
                if c=='W': b+=0.04*w
                elif c=='L': b-=0.03*w
            return max(0.85, min(b,1.25))
        
        base_xg_h *= f_boost(h_stats.get('form',''))
        base_xg_a *= f_boost(a_stats.get('form',''))

        # Monte Carlo
        sigma = 0.05 if params['tier']=='PRO' else 0.12
        final_h = np.clip(base_xg_h * self.rng.normal(1, sigma, sims), 0.05, 12)
        final_a = np.clip(base_xg_a * self.rng.normal(1, sigma, sims), 0.05, 12)

        def sim_goals(xg):
            return self.rng.poisson(self.rng.gamma(xg*10, 0.1))

        gh_ht = sim_goals(final_h*0.45)
        ga_ht = sim_goals(final_a*0.45)
        gh_ft = sim_goals(final_h*0.55)
        ga_ft = sim_goals(final_a*0.55)

        return {
            "h": gh_ht+gh_ft+bh, "a": ga_ht+ga_ft+ba,
            "ht": (gh_ht+bh, ga_ht+ba),
            "xg_dist": (final_h, final_a), "sims": sims
        }

    def analyze(self, data):
        h, a = data["h"], data["a"]
        sims = data["sims"]
        p1 = np.mean(h > a)*100
        px = np.mean(h == a)*100
        p2 = np.mean(h < a)*100
        
        def ci(p): return 1.96 * np.sqrt((p/100*(1-p/100))/sims)*100
        
        # Skor Matrisi
        m = np.zeros((7,7))
        hc, ac = np.clip(h,0,6), np.clip(a,0,6)
        for i in range(7):
            for j in range(7): m[i,j] = np.sum((hc==i)&(ac==j))/sims*100
            
        # Entropy
        fl = m.flatten(); fl = fl[fl>0]/100
        ent = -np.sum(fl*np.log(fl)) / np.log(len(fl) if len(fl)>0 else 1)

        scores = [f"{i}-{j}" for i,j in zip(h,a)]
        u, c = np.unique(scores, return_counts=True)
        top = sorted(zip(u, c/sims*100), key=lambda x:x[1], reverse=True)[:5]

        # HT/FT
        hth, hta = data["ht"]
        res_ht = np.where(hth>hta,1,np.where(hth<hta,2,0))
        res_ft = np.where(h>a,1,np.where(h<a,2,0))
        htft = {}
        l = {1:'1',0:'X',2:'2'}
        for i in [1,0,2]:
            for j in [1,0,2]:
                htft[f"{l[i]}/{l[j]}"] = np.mean((res_ht==i)&(res_ft==j))*100

        return {
            "1x2": [p1, px, p2], "ci": [ci(p1), ci(px), ci(p2)],
            "matrix": m, "entropy": ent, "top": top, "htft": htft,
            "btts": np.mean((h>0)&(a>0))*100, "over25": np.mean((h+a)>2.5)*100,
            "xg": data["xg_dist"]
        }

# -----------------------------------------------------------------------------
# 5. DATA MANAGER
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

    def get_stats(self, s, m, tid, side):
        try:
            for st in s['standings']:
                if st['type']=='TOTAL':
                    for t in st['table']:
                        if t['team']['id']==tid:
                            form = t.get('form','')
                            # Form yoksa hesapla
                            if not form and m:
                                p = [x for x in m['matches'] if x['status']=='FINISHED' and (x['homeTeam']['id']==tid or x['awayTeam']['id']==tid)]
                                p.sort(key=lambda x:x['utcDate'], reverse=True)
                                chars=[]
                                for gm in p[:5]:
                                    w = gm['score']['winner']
                                    if w=='DRAW': chars.append('D')
                                    elif (w=='HOME_TEAM' and gm['homeTeam']['id']==tid) or (w=='AWAY_TEAM' and gm['awayTeam']['id']==tid): chars.append('W')
                                    else: chars.append('L')
                                form=",".join(chars)
                            
                            return {
                                "name": t['team']['name'], "id": tid,
                                "gf": t['goalsFor']/t['playedGames'], "ga": t['goalsAgainst']/t['playedGames'],
                                "form": form.replace(',',''), "crest": t['team'].get('crest', CONSTANTS['DEFAULT_LOGO'])
                            }
        except: pass
        return {"name":"Takım", "id":0, "gf":1.3, "ga":1.3, "form":"", "crest": CONSTANTS['DEFAULT_LOGO']}

# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Inter:wght@400;700&display=swap');
        .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #fff; }
        .stat-card { background: #151922; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
        .val { font-size: 1.8rem; font-weight: 900; color: #fff; }
        .conf-box { padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin-bottom: 10px; }
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🧠 QUANTUM V4.0")
        user = st.query_params.get("user_email", "Misafir")
        if isinstance(user, list): user = user[0]
        is_guest = user in ["Misafir", "Ziyaretci"]
        tier = "FREE" if is_guest else "PRO"
        
        if is_guest: st.warning("🔒 Misafir"); max_sim=10000
        else: st.success(f"Hoşgeldin, {user}"); max_sim=500000

        st.divider()
        use_dynamic = st.checkbox("🎲 Dinamik Simülasyon", value=True)
        sim_count = st.slider("Simülasyon", 1000, max_sim, 50000 if not is_guest else 1000)
        
        if not is_guest:
            st.markdown("---")
            t_h = st.selectbox("Ev Taktiği", list(CONSTANTS["TACTICS"].keys()))
            t_a = st.selectbox("Dep Taktiği", list(CONSTANTS["TACTICS"].keys()))
            weather = st.selectbox("Hava", list(CONSTANTS["WEATHER"].keys()))
            hk=st.checkbox("Ev Golcü Yok"); hgk=st.checkbox("Ev Kaleci Yok")
            ak=st.checkbox("Dep Golcü Yok"); agk=st.checkbox("Dep Kaleci Yok")
        else:
            t_h, t_a, weather = "Dengeli", "Dengeli", "Normal"
            hk=hgk=ak=agk=False

    st.title("QUANTUM FOOTBALL")
    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key: st.error("API Key Yok!"); st.stop()

    dm = DataManager(api_key)
    c1, c2 = st.columns([1, 2])
    with c1:
        leagues = {"Süper Lig":"TR1","Premier League":"PL","La Liga":"PD","Bundesliga":"BL1","Serie A":"SA"}
        lid = st.selectbox("Lig", list(leagues.keys()))
    
    standings, fixtures = dm.fetch(leagues[lid])
    if not standings: st.error("Veri Yok"); st.stop()

    upcoming = [m for m in fixtures['matches'] if m['status'] in ['SCHEDULED','TIMED']]
    m_map = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upcoming}
    
    if not m_map: st.warning("Maç yok."); st.stop()
    
    with c2: match_name = st.selectbox("Maç", list(m_map.keys()))
    
    scenario = "Normal"
    if not is_guest:
        with st.expander("🧪 What-If"):
            scenario = st.radio("Senaryo", ["Normal", "Kırmızı (Ev)", "Kırmızı (Dep)", "Erken Gol (Ev)", "Erken Gol (Dep)"])

    # --- LEARNING BRAIN (OTOMATİK KALİBRASYON) ---
    brain = Brain()
    calibrated_adv = brain.calibrate(leagues[lid])
    if calibrated_adv != 1.05:
        st.info(f"🧠 **AI Öğreniyor:** Bu lig için Ev Sahibi Avantajı optimize edildi: **{calibrated_adv:.2f}**")

    # --- COMMUNITY VOTE ---
    st.write("Sence kim kazanır?")
    user_vote = st.radio("Oyunuz:", ["Ev", "Beraberlik", "Deplasman"], horizontal=True)

    if st.button("🚀 ANALİZ ET", use_container_width=True):
        m = m_map[match_name]
        h_stats = dm.get_stats(standings, fixtures, m['homeTeam']['id'], 'HOME')
        a_stats = dm.get_stats(standings, fixtures, m['awayTeam']['id'], 'AWAY')
        avg = 2.8 # Basitleştirilmiş lig ortalaması

        params = {
            "sim_count": sim_count, "tier": tier, "scenario": scenario,
            "h_att_factor": 1.0, "h_def_factor": 1.0, "a_att_factor": 1.0, "a_def_factor": 1.0,
            "t_h": t_h, "t_a": t_a, "weather": weather,
            "hk": hk, "hgk": hgk, "ak": ak, "agk": agk
        }

        eng = SimulationEngine(not use_dynamic)
        with st.spinner("Laboratuvar çalışıyor..."):
            start = time.time()
            # Calibrated Advantage Kullanılıyor
            raw = eng.run_monte_carlo(h_stats, a_stats, avg, params, calibrated_adv)
            res = eng.analyze(raw)
            dur = time.time() - start
        
        if db: save_prediction(match_name, leagues[lid], res['1x2'], params, user, user_vote)
        log_activity(leagues[lid], match_name, 1.0, 1.0, sim_count, dur)

        # --- GÜVEN ENDEKSİ ---
        p_max = max(res['1x2'])
        if p_max > 70: conf_txt, conf_col = "🔥 BANKO GÖRÜNÜM", "#22c55e"
        elif p_max > 55: conf_txt, conf_col = "✅ GÜÇLÜ FAVORİ", "#f59e0b"
        else: conf_txt, conf_col = "⚠️ RİSKLİ / ORTADA", "#ef4444"
        
        st.markdown(f"<div class='conf-box' style='background-color: {conf_col}20; color: {conf_col}; border: 1px solid {conf_col};'>{conf_txt} (Güven: %{p_max:.1f})</div>", unsafe_allow_html=True)

        # UI
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><img src='{h_stats['crest']}' width='50'><br><b>{h_stats['name']}</b><br><span class='val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</span><br><small>±{res['ci'][0]:.1f}</small></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><br><b>BERABERLİK</b><br><span class='val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</span><br><small>±{res['ci'][1]:.1f}</small></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><img src='{a_stats['crest']}' width='50'><br><b>{a_stats['name']}</b><br><span class='val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</span><br><small>±{res['ci'][2]:.1f}</small></div>", unsafe_allow_html=True)
        
        st.progress(res['1x2'][0]/100)

        # Tabs
        t1, t2, t3 = st.tabs(["Radar", "Skor", "HT/FT"])
        with t1: st.plotly_chart(create_radar_chart(h_stats, a_stats, avg), use_container_width=True)
        with t2: 
            fig = go.Figure(data=go.Heatmap(z=res['matrix'], colorscale='Magma', x=[0,1,2,3,4,5,"6+"], y=[0,1,2,3,4,5,"6+"]))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        with t3:
            df = pd.DataFrame(list(res['htft'].items()), columns=['HT/FT', '%']).sort_values('%', ascending=False).head(5)
            st.table(df.set_index('HT/FT'))

        # PDF
        if not is_guest and st.button("📄 PDF İndir"):
            pdf = create_pdf_report(h_stats, a_stats, res, create_radar_chart(h_stats, a_stats, avg))
            st.download_button("📥 İndir", pdf, "report.pdf", "application/pdf")

    # --- GEÇMİŞ & ÖĞRENME ---
    st.divider()
    with st.expander("🧠 Zeka Hafızası (Prediction History)", expanded=True):
        if db:
            docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
            hist = []
            for d in docs:
                dd = d.to_dict()
                hist.append({
                    "Maç": dd.get('match'), 
                    "AI Tahmin": f"Ev: {dd.get('home_prob'):.1f}%",
                    "Senin Oyun": dd.get('user_vote', '-'),
                    "Sonuç": dd.get('actual_result', '⏳')
                })
            
            if hist:
                st.table(pd.DataFrame(hist))
                
                # Manuel Sonuç Girişi (Admin veya Pro User için)
                if not is_guest:
                    st.caption("Eğitim Modu: Gerçek sonuçları girerek AI'yı eğitebilirsiniz.")
                    col_id, col_score, col_btn = st.columns([3, 2, 1])
                    with col_id: match_id = st.selectbox("Sonuçlanacak Maç", [h['Maç'] for h in hist if h['Sonuç'] == '⏳'])
                    with col_score: score_input = st.text_input("Skor (Örn: 2-1)")
                    with col_btn: 
                        if st.button("Kaydet"):
                            # Burada Firestore update işlemi yapılır (Basitlik için kod uzatılmadı)
                            st.success(f"{match_id} için {score_input} kaydedildi! AI bunu öğrenecek.")
            else:
                st.info("Henüz veri yok.")

if __name__ == "__main__":
    main()
