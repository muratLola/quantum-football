import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os

# --- FIREBASE EKLENTİLERİ ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# -----------------------------------------------------------------------------
# 1. KONFİGÜRASYON & LOGOLAR
# -----------------------------------------------------------------------------
CONFIG = {
    "API_URL": "https://api.football-data.org/v4",
    # API Key ve Admin Şifresi secrets.toml üzerinden çekilecek
}

# TAKIM LOGOLARI (Genişletilebilir)
# ID'leri API'den gelen ID'lerdir.
TEAM_LOGOS = {
    # Türkiye
    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",
    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",
    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",
    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",
    2058: "https://upload.wikimedia.org/wikipedia/tr/e/e0/Samsunspor_logo_2.png",
    # Avrupa Devleri (Örnek)
    86:  "https://upload.wikimedia.org/wikipedia/tr/f/f4/Real_Madrid.png", # Real Madrid
    81:  "https://upload.wikimedia.org/wikipedia/tr/9/96/FC_Barcelona.png", # Barcelona
    65:  "https://upload.wikimedia.org/wikipedia/tr/b/b6/Manchester_City.png", # Man City
    64:  "https://upload.wikimedia.org/wikipedia/tr/0/03/Liverpool_FC.png", # Liverpool
    5:   "https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg", # Bayern
}
DEFAULT_LOGO = "https://cdn-icons-png.flaticon.com/512/53/53283.png"

st.set_page_config(page_title="Quantum Football AI", page_icon="⚽", layout="wide")

# --- FIREBASE BAŞLATMA ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
    except Exception as e:
        st.warning("Veritabanı bağlantısı kurulamadı. (Loglama Devre Dışı)")
        print(f"Firebase Init Error: {e}")

try:
    db = firestore.client()
except:
    db = None

# -----------------------------------------------------------------------------
# 2. KULLANICI YÖNETİMİ
# -----------------------------------------------------------------------------
query_params = st.query_params
current_user_email = query_params.get("user_email", "Misafir")
if isinstance(current_user_email, list): current_user_email = current_user_email[0]

IS_GUEST = (current_user_email == "Misafir" or current_user_email == "Ziyaretci")
USER_TIER = "FREE" if IS_GUEST else "PRO"

# -----------------------------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def log_activity(league, match, h_att, a_att, sim_count):
    if db is None: return
    try:
        db.collection("analysis_logs").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user": current_user_email,
            "tier": USER_TIER,
            "league": league,
            "match": match,
            "sims": sim_count,
            "settings": {"h": h_att, "a": a_att}
        })
    except Exception as e:
        print(f"Log Error: {e}") 

def generate_commentary(h_name, a_name, res):
    """xG ve Olasılık bazlı gelişmiş yorum üretir"""
    p_h, p_d, p_a = res['1x2']
    xg_h, xg_a = res['xg']
    total_xg = xg_h + xg_a
    xg_diff = xg_h - xg_a

    # 1. Galibiyet Analizi
    if p_h > 65:
        main_msg = f"🔥 **{h_name}** sahasında çok baskın (%{p_h:.1f})."
        if xg_diff > 1.0: main_msg += f" Gol beklentisi farkı ({xg_diff:.2f}) ezici bir oyuna işaret ediyor."
    elif p_a > 60:
        main_msg = f"🚨 **{a_name}** deplasmanda olmasına rağmen net favori (%{p_a:.1f}). Kadro kalitesi fark yaratıyor."
    elif p_d > 32:
        main_msg = f"⚖️ **Taktik Savaşı.** Beraberlik ihtimali (%{p_d:.1f}) lig ortalamasının çok üzerinde."
    elif abs(p_h - p_a) < 10:
        main_msg = f"⚔️ **Ortada Bir Maç.** Kazanma şansları çok yakın (%{p_h:.1f} vs %{p_a:.1f}). İlk golü atan maçı %70 çözer."
    else:
        favori = h_name if p_h > p_a else a_name
        main_msg = f"📊 **Çekişmeli Mücadele.** Net favori yok ama ibre hafifçe **{favori}** tarafına kayıyor."
    
    # 2. Gol Analizi (Over/Under)
    if total_xg > 3.0:
        goal_msg = " ⚽ **Gol Yağmuru:** İki takımın da hücum gücü yüksek, 2.5 Üst ihtimali güçlü."
    elif total_xg < 2.0:
        goal_msg = " 🛡️ **Kısır Maç:** Savunma ağırlıklı, düşük skorlu bir mücadele beklenebilir."
    else:
        goal_msg = ""

    return main_msg + goal_msg

# -----------------------------------------------------------------------------
# 4. SİMÜLASYON MOTORU
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        sims = params['sim_count']
        
        # 1. GÜÇ HESABI
        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        base_xg_h = h_attack * a_def * avg_g
        base_xg_a = a_attack * h_def * avg_g

        base_xg_h *= params['home_adv']

        # Eksik Oyuncu Cezası
        if params.get('h_missing', 0) > 0: base_xg_h *= (1 - (params['h_missing'] * 0.08))
        if params.get('a_missing', 0) > 0: base_xg_a *= (1 - (params['a_missing'] * 0.08))

        # 2. BELİRSİZLİK (PRO vs FREE Farkı)
        sigma = 0.05 if params['tier'] == 'PRO' else 0.12
        
        random_factors_h = self.rng.normal(1, sigma, sims)
        random_factors_a = self.rng.normal(1, sigma, sims)
        
        # Negatif xG engellemek için np.clip
        final_xg_h = np.clip(base_xg_h * random_factors_h, 0.05, None)
        final_xg_a = np.clip(base_xg_a * random_factors_a, 0.05, None)

        # 3. POISSON SİMÜLASYONU
        gh_ht = self.rng.poisson(final_xg_h * 0.45)
        ga_ht = self.rng.poisson(final_xg_a * 0.45)
        gh_ft = self.rng.poisson(final_xg_h * 0.55)
        ga_ft = self.rng.poisson(final_xg_a * 0.55)

        total_h = gh_ht + gh_ft
        total_a = ga_ht + ga_ft

        return {
            "h": total_h, "a": total_a,
            "ht": (gh_ht, ga_ht), "ft": (total_h, total_a),
            "xg": (np.mean(final_xg_h), np.mean(final_xg_a)), "sims": sims
        }

    def analyze_results(self, data):
        h, a = data["h"], data["a"]
        sims = data["sims"]

        p_home = np.mean(h > a) * 100
        p_draw = np.mean(h == a) * 100
        p_away = np.mean(h < a) * 100

        # Skor Matrisi
        matrix = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100

        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]

        # HT/FT
        h_ht, a_ht = data["ht"]
        ht_res = np.where(h_ht > a_ht, 1, np.where(h_ht < a_ht, 2, 0))
        ft_res = np.where(h > a, 1, np.where(h < a, 2, 0))
        
        htft = {}
        labels = {1: "1", 0: "X", 2: "2"}
        for i in [1, 0, 2]:
            for j in [1, 0, 2]:
                mask = (ht_res == i) & (ft_res == j)
                htft[f"{labels[i]}/{labels[j]}"] = np.sum(mask) / sims * 100

        return {
            "1x2": [p_home, p_draw, p_away],
            "matrix": matrix,
            "top_scores": top_scores,
            "htft": htft,
            "xg": data["xg"]
        }

# -----------------------------------------------------------------------------
# 5. DATA MANAGER
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    @st.cache_data(ttl=1800)
    def fetch_data(_self, league_code):
        try:
            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=_self.headers)
            if r1.status_code == 429:
                st.warning("⚠️ API Limiti: Lütfen 30 saniye bekleyin.")
                return None, None
            
            standings = r1.json()
            
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", 
                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            matches = r2.json() if r2.status_code == 200 else {}
            
            return standings, matches
        except Exception as e:
            print(f"Data Fetch Error: {e}")
            return None, None

    def get_team_stats(self, standings, team_id, table_type='TOTAL', default_name="Takım"):
        try:
            target_table = []
            s_list = standings.get('standings', [])
            if not s_list: return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4}

            for item in s_list:
                if item.get('type') == table_type:
                    target_table = item.get('table', [])
                    break
            
            if not target_table and table_type != 'TOTAL':
                return self.get_team_stats(standings, team_id, 'TOTAL', default_name)

            for row in target_table:
                if row['team']['id'] == team_id:
                    played = row['playedGames']
                    if played < 2: 
                        return {"name": row['team']['name'], "id": team_id, "gf": 1.5, "ga": 1.5}
                    
                    return {
                        "name": row['team']['name'],
                        "id": team_id,
                        "gf": row['goalsFor'] / played,
                        "ga": row['goalsAgainst'] / played
                    }
        except Exception as e:
            print(f"Stats Parse Error: {e}")
            pass
        return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4}

    def get_league_avg(self, standings):
        try:
            table = standings.get('standings', [])[0].get('table', [])
            total_goals = sum(t['goalsFor'] for t in table)
            total_games = sum(t['playedGames'] for t in table)
            return total_goals / (total_games / 2) if total_games > 10 else 2.8
        except:
            return 2.8

# -----------------------------------------------------------------------------
# 6. UI MAIN
# -----------------------------------------------------------------------------
def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Inter:wght@400;700&display=swap');
        .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #fff; }
        .tier-badge { padding: 5px 10px; border-radius: 5px; font-weight: bold; font-family: 'Orbitron'; text-align: center; margin-bottom: 20px; }
        .free-tier { background: #334155; color: #94a3b8; border: 1px solid #475569; }
        .pro-tier { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; box-shadow: 0 0 10px rgba(0,255,136,0.2); }
        .stat-card { background: #151922; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
        .stat-val { font-size: 1.8rem; font-weight: 900; color: #fff; }
        .stat-lbl { font-size: 0.8rem; color: #888; text-transform: uppercase; }
        .stProgress > div > div > div > div { background-color: #00ff88; }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 👤 Kullanıcı Paneli")
        if IS_GUEST:
            st.markdown(f"<div class='tier-badge free-tier'>PLAN: {USER_TIER}</div>", unsafe_allow_html=True)
            st.warning("🔒 Misafir Modu\nAnaliz kapasitesi sınırlı.")
        else:
            st.markdown(f"<div class='tier-badge pro-tier'>PLAN: {USER_TIER} ⚡</div>", unsafe_allow_html=True)
            st.success(f"Hoşgeldin,\n{current_user_email}")
        
        # --- ÜYELİK TABLOSU (YENİ EKLENDİ) ---
        with st.expander("ℹ️ Üyelik Avantajları", expanded=False):
            st.markdown("""
            | Özellik | FREE | PRO ⚡ |
            |---|---|---|
            | **Simülasyon** | 5.000 | **250.000** |
            | **İsabet Oranı** | Standart | **Yüksek** |
            | **Eksik Oyuncu** | ❌ | ✅ |
            | **Detaylı xG** | ❌ | ✅ |
            """)

        st.divider()
        st.subheader("⚙️ Ayarlar")
        
        max_sim = 250000 if not IS_GUEST else 5000
        default_sim = 50000 if not IS_GUEST else 1000
        sim_count = st.slider("Simülasyon Sayısı", 1000, max_sim, default_sim, step=1000)
        
        st.caption("Takım Form Çarpanları")
        h_att = st.slider("Ev Sahibi", 0.8, 1.2, 1.0)
        a_att = st.slider("Deplasman", 0.8, 1.2, 1.0)

        if not IS_GUEST:
            st.caption("🚑 Eksik Oyuncu (PRO)")
            h_miss = st.number_input("Ev Sahibi Eksik", 0, 5, 0)
            a_miss = st.number_input("Deplasman Eksik", 0, 5, 0)
        else:
            h_miss, a_miss = 0, 0

        st.markdown("---")
        with st.expander("🔐 Admin"):
            pw = st.text_input("Şifre", type="password")
            admin_pass = st.secrets.get("ADMIN_PASS")
            if admin_pass and pw == admin_pass:
                if st.button("Logları Getir") and db:
                    try:
                        docs = db.collection("analysis_logs").order_by("timestamp", direction="DESCENDING").limit(20).stream()
                        st.dataframe(pd.DataFrame([d.to_dict() for d in docs]))
                    except: st.error("Veri yok.")

    st.markdown("<h1 style='text-align:center; color:#00ff88;'>QUANTUM FOOTBALL AI</h1>", unsafe_allow_html=True)
    
    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key: st.error("⚠️ API Key Eksik!"); st.stop()

    dm = DataManager(api_key)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        leagues = {"Süper Lig": "TR1", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1", "Serie A": "SA", "Şampiyonlar Ligi": "CL"}
        sel_league = st.selectbox("Lig Seçiniz", list(leagues.keys()))
    
    standings, fixtures = dm.fetch_data(leagues[sel_league])
    if not fixtures: st.stop()
        
    matches = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in fixtures['matches'] if m['status'] == 'SCHEDULED'}
    if not matches: st.warning("Planlanmış maç yok."); st.stop()

    with c2: sel_match_name = st.selectbox("Maç Seçiniz", list(matches.keys()))
    
    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
        m = matches[sel_match_name]
        log_activity(sel_league, sel_match_name, h_att, a_att, sim_count)
        
        h_id, a_id = m['homeTeam']['id'], m['awayTeam']['id']
        
        # --- GÜÇLENDİRİLMİŞ VERİ ÇEKME ---
        h_stats = dm.get_team_stats(standings, h_id, 'HOME', m['homeTeam']['name'])
        a_stats = dm.get_team_stats(standings, a_id, 'AWAY', m['awayTeam']['name'])
        
        # --- LOGO ÇEKME MANTIĞI (DÜZELTİLDİ) ---
        h_logo = TEAM_LOGOS.get(h_stats['id'], DEFAULT_LOGO)
        a_logo = TEAM_LOGOS.get(a_stats['id'], DEFAULT_LOGO)
        
        league_avg = dm.get_league_avg(standings)

        params = {
            "sim_count": sim_count,
            "h_att_factor": h_att, "h_def_factor": 1.0,
            "a_att_factor": a_att, "a_def_factor": 1.0,
            "h_missing": h_miss, "a_missing": a_miss,
            "home_adv": 1.05,
            "tier": USER_TIER
        }

        eng = SimulationEngine()
        with st.spinner(f"Kuantum motoru hesaplıyor: {h_stats['name']} (Ev) vs {a_stats['name']} (Dep)..."):
            raw = eng.run_monte_carlo(h_stats, a_stats, league_avg, params)
            res = eng.analyze_results(raw)

        st.divider()
        st.info(generate_commentary(h_stats['name'], a_stats['name'], res))
        
        # --- KARTLARDA LOGO KULLANIMI (HTML İLE) ---
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><img src='{h_logo}' width='60'><br><div class='stat-lbl'>{h_stats['name']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><br><br><div class='stat-lbl'>BERABERLİK</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><img src='{a_logo}' width='60'><br><div class='stat-lbl'>{a_stats['name']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)
        
        st.write("")
        st.progress(res['1x2'][0]/100, text=f"Ev Sahibi Galibiyet İhtimali: %{res['1x2'][0]:.1f}")

        st.markdown(f"""
        ### 📊 Detaylı Analiz 
        <span style='color:#00ff88; font-weight:bold'>Ev xG: {res['xg'][0]:.2f}</span> - 
        <span style='color:#ff4b4b; font-weight:bold'>Dep xG: {res['xg'][1]:.2f}</span>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Skor Matrisi", "En Olası Skorlar", "HT/FT"])
        
        with tab1:
            fig = go.Figure(data=go.Heatmap(
                z=res['matrix'], 
                colorscale='Magma',
                x=[0, 1, 2, 3, 4, 5, 6],
                y=[0, 1, 2, 3, 4, 5, 6]
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350, 
                margin=dict(l=0,r=0,t=20,b=0),
                xaxis_title="Deplasman Gol", yaxis_title="Ev Sahibi Gol"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            for s, p in res['top_scores']:
                st.progress(p/100, text=f"Skor {s} - Olasılık: %{p:.1f}")
        
        with tab3:
            st.caption("İlk Yarı / Maç Sonucu Olasılıkları (> %5)")
            htft_data = [{"Tercih": k, "Olasılık": v} for k, v in res['htft'].items() if v > 5]
            if htft_data:
                df_htft = pd.DataFrame(htft_data).sort_values("Olasılık", ascending=False)
                st.bar_chart(df_htft.set_index("Tercih"))
            else:
                st.write("Belirgin bir olasılık bulunamadı.")

if __name__ == "__main__":
    main()
