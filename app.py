import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import logging

# --- LOGGING AYARLARI (KURUMSAL SEVİYE) ---
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

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
TEAM_LOGOS = {
    # Türkiye
    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",
    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",
    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",
    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",
    2058: "https://upload.wikimedia.org/wikipedia/tr/e/e0/Samsunspor_logo_2.png",
    # Avrupa (Örnek)
    86:  "https://upload.wikimedia.org/wikipedia/tr/f/f4/Real_Madrid.png",
    81:  "https://upload.wikimedia.org/wikipedia/tr/9/96/FC_Barcelona.png",
    65:  "https://upload.wikimedia.org/wikipedia/tr/b/b6/Manchester_City.png",
    64:  "https://upload.wikimedia.org/wikipedia/tr/0/03/Liverpool_FC.png",
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
        logger.error(f"Firebase Init Error: {e}")
        # Kullanıcıya sessiz kal, loga yaz

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
def log_activity(league, match, h_att, a_att, sim_count, duration):
    if db is None: return
    try:
        db.collection("analysis_logs").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user": current_user_email,
            "tier": USER_TIER,
            "league": league,
            "match": match,
            "sims": sim_count,
            "duration_sec": duration,
            "settings": {"h": h_att, "a": a_att}
        })
    except Exception as e:
        logger.error(f"Firestore Log Error: {e}")

def generate_commentary(h_stats, a_stats, res):
    """xG, Olasılık ve Over/Under bazlı gelişmiş yorum üretir"""
    p_h, p_d, p_a = res['1x2']
    xg_h, xg_a = res['xg']
    xg_diff = xg_h - xg_a
    
    # Form bilgisini temizle
    h_form = h_stats.get('form', '').replace(',', '')
    a_form = a_stats.get('form', '').replace(',', '')

    # 1. Galibiyet Analizi
    if p_h > 60:
        main_msg = f"🔥 **{h_stats['name']}** sahasında favori (%{p_h:.1f})."
        if h_form.startswith('W'): main_msg += " Son maçı kazanmanın moraliyle sahada!"
    elif p_a > 55:
        main_msg = f"🚨 **{a_stats['name']}** deplasmanda baskın (%{p_a:.1f})."
        if a_form.startswith('W'): main_msg += " Form grafiği yükselişte."
    elif p_d > 30:
        main_msg = f"⚖️ **Denge Maçı.** Beraberlik ihtimali (%{p_d:.1f}) yüksek."
    else:
        main_msg = f"📊 **Çekişmeli Mücadele.** Net favori yok."
    
    # 2. xG Yorumu
    if abs(xg_diff) < 0.25:
        goal_msg = " Gol beklentileri kafa kafaya, ilk golü atan avantajı kapar."
    elif (xg_h + xg_a) > 2.8:
        goal_msg = " ⚽ **Gol Şöleni:** İstatistikler 2.5 Üst ihtimalini destekliyor."
    else:
        goal_msg = ""

    return main_msg + goal_msg

# -----------------------------------------------------------------------------
# 4. SİMÜLASYON MOTORU (THE SINGULARITY EDITION)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self, use_fixed_seed=False):
        # Hibrit Seed Kontrolü (CEO Önerisi)
        if use_fixed_seed:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        sims = params['sim_count']
        
        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        base_xg_h = h_attack * a_def * avg_g
        base_xg_a = a_attack * h_def * avg_g

        # --- AĞIRLIKLI FORM (Zaman Bazlı) ---
        def calculate_weighted_form(form_str):
            if not form_str: return 1.0
            matches = form_str.replace(',', '')
            boost = 1.0
            # [En Yeni ... En Eski] -> Yeni maçın etkisi daha fazla
            weights = [1.5, 1.25, 1.0, 0.75, 0.5] 
            recent = matches[:5] 
            current_weights = weights[:len(recent)]

            for i, char in enumerate(recent):
                w = current_weights[i]
                if char == 'W': boost += 0.04 * w
                elif char == 'D': boost += 0.01 * w
                elif char == 'L': boost -= 0.03 * w
            
            return max(0.85, min(boost, 1.25))

        base_xg_h *= calculate_weighted_form(h_stats.get('form', ''))
        base_xg_a *= calculate_weighted_form(a_stats.get('form', ''))

        base_xg_h *= params['home_adv']

        if params.get('h_missing', 0) > 0: base_xg_h *= (1 - (params['h_missing'] * 0.08))
        if params.get('a_missing', 0) > 0: base_xg_a *= (1 - (params['a_missing'] * 0.08))

        sigma = 0.05 if params['tier'] == 'PRO' else 0.12
        
        random_factors_h = self.rng.normal(1, sigma, sims)
        random_factors_a = self.rng.normal(1, sigma, sims)
        
        final_xg_h = np.clip(base_xg_h * random_factors_h, 0.05, None)
        final_xg_a = np.clip(base_xg_a * random_factors_a, 0.05, None)

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

        # Olasılıklar
        home_wins = (h > a)
        draws = (h == a)
        away_wins = (h < a)

        p_home = np.mean(home_wins) * 100
        p_draw = np.mean(draws) * 100
        p_away = np.mean(away_wins) * 100

        # --- GÜVEN ARALIĞI (TAM KAPSAMLI) ---
        def calc_ci(p, n):
            return 1.96 * np.sqrt((p/100 * (1 - p/100)) / n) * 100
        
        ci = {
            "h": calc_ci(p_home, sims),
            "d": calc_ci(p_draw, sims),
            "a": calc_ci(p_away, sims)
        }

        # --- SKOR MATRİSİ (BUCKET SİSTEMİ - 6+) ---
        matrix = np.zeros((7, 7))
        # 6'dan büyük skorları 6. indekste topla (Veri kaybını önler)
        h_clipped = np.clip(h, 0, 6)
        a_clipped = np.clip(a, 0, 6)
        
        for i in range(7):
            for j in range(7):
                matrix[i, j] = np.sum((h_clipped == i) & (a_clipped == j)) / sims * 100

        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]

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
            "ci": ci,
            "matrix": matrix,
            "top_scores": top_scores,
            "htft": htft,
            "xg": data["xg"]
        }

# -----------------------------------------------------------------------------
# 5. DATA MANAGER (IMMORTAL - CRASH PROOF)
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    def fetch_data(self, league_code):
        cache_key = f"data_{league_code}"
        
        # 1. Önce Cache Kontrolü
        if cache_key in st.session_state:
            last_fetch, data = st.session_state[cache_key]
            if (datetime.now() - last_fetch).seconds < 1800:
                return data, last_fetch

        # 2. API İsteği (Tüm Sezon)
        try:
            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=self.headers)
            if r1.status_code == 429:
                st.warning("⚠️ API Limiti. Önbellek kullanılıyor.")
                if cache_key in st.session_state:
                    return st.session_state[cache_key][1], st.session_state[cache_key][0]
                return None, None
            
            standings = r1.json()
            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", headers=self.headers)
            matches = r2.json() if r2.status_code == 200 else {}
            
            # Veriyi Hafızaya Al
            st.session_state[cache_key] = (datetime.now(), (standings, matches))
            return (standings, matches), datetime.now()

        except Exception as e:
            logger.error(f"API Fetch Error: {e}")
            if cache_key in st.session_state:
                st.warning("⚠️ Bağlantı hatası. Önbellek kullanılıyor.")
                return st.session_state[cache_key][1], st.session_state[cache_key][0]
            return None, None

    def calculate_form_from_matches(self, matches_data, team_id):
        if not matches_data or 'matches' not in matches_data: return ""
        
        played_matches = []
        for m in matches_data['matches']:
            if m['status'] == 'FINISHED':
                if m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id:
                    played_matches.append(m)
        
        # Yeniden Eskiye Sırala
        played_matches.sort(key=lambda x: x['utcDate'], reverse=True)
        
        form_chars = []
        for m in played_matches[:5]: 
            winner = m['score']['winner']
            if winner == 'DRAW':
                form_chars.append('D')
            elif (winner == 'HOME_TEAM' and m['homeTeam']['id'] == team_id) or \
                 (winner == 'AWAY_TEAM' and m['awayTeam']['id'] == team_id):
                form_chars.append('W')
            else:
                form_chars.append('L')
                
        return ",".join(form_chars) 

    def get_team_stats(self, standings, matches, team_id, table_type='TOTAL', default_name="Takım"):
        try:
            target_table = []
            s_list = standings.get('standings', [])
            if not s_list: return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4, "form": ""}

            for item in s_list:
                if item.get('type') == table_type:
                    target_table = item.get('table', [])
                    break
            
            if not target_table and table_type != 'TOTAL':
                return self.get_team_stats(standings, matches, team_id, 'TOTAL', default_name)

            for row in target_table:
                if row['team']['id'] == team_id:
                    played = row['playedGames']
                    form = row.get('form', '')
                    # Otonom Form Hesaplama
                    if not form and matches:
                        form = self.calculate_form_from_matches(matches, team_id)
                    
                    form = form.replace(',', '') if form else ""

                    if played < 2: 
                        return {"name": row['team']['name'], "id": team_id, "gf": 1.5, "ga": 1.5, "form": form}
                    
                    return {
                        "name": row['team']['name'],
                        "id": team_id,
                        "gf": row['goalsFor'] / played,
                        "ga": row['goalsAgainst'] / played,
                        "form": form
                    }
        except Exception as e:
            logger.error(f"Parsing Stats Error: {e}")
            pass
        return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4, "form": ""}

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
        
        with st.expander("ℹ️ Üyelik Avantajları", expanded=False):
            st.markdown("""
            | Özellik | FREE | PRO ⚡ |
            |---|---|---|
            | **Simülasyon** | 5.000 | **250.000** |
            | **Form Analizi** | ❌ | ✅ |
            | **Eksik Oyuncu** | ❌ | ✅ |
            """)

        st.divider()
        st.subheader("⚙️ Ayarlar")
        
        use_dynamic = st.checkbox("🎲 Dinamik Simülasyon", value=True, help="Her analizde farklı sonuçlar üretir (Gerçek Monte Carlo).")
        
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
            
            if h_miss > 5 or a_miss > 5:
                st.warning("⚠️ 5'ten fazla eksik oyuncu maçın iptaline yol açabilir.")
        else:
            h_miss, a_miss = 0, 0

        st.markdown("---")
        with st.expander("🔐 Admin"):
            pw = st.text_input("Şifre", type="password")
            admin_pass = st.secrets.get("ADMIN_PASS")
            if admin_pass and pw == admin_pass:
                if st.button("🗑️ Önbelleği Temizle"):
                    st.session_state.clear()
                    st.rerun()
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
    
    data_tuple, fetch_time = dm.fetch_data(leagues[sel_league])
    
    if not data_tuple:
        st.error("Veri alınamadı. Lütfen daha sonra tekrar deneyin.")
        st.stop()
        
    standings, fixtures = data_tuple
    
    if fetch_time:
        age_min = int((datetime.now() - fetch_time).total_seconds() / 60)
        st.caption(f"🕒 Veri Güncelliği: {age_min} dakika önce güncellendi.")

    upcoming_matches = []
    past_matches = []
    
    for m in fixtures.get('matches', []):
        match_date = m['utcDate'][:10]
        match_label = f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({match_date})"
        
        if m['status'] in ['SCHEDULED', 'TIMED']:
            upcoming_matches.append((match_label, m))
        elif m['status'] == 'FINISHED':
            score = f"[{m['score']['fullTime']['home']}-{m['score']['fullTime']['away']}]"
            past_label = f"{match_label} {score} ✅"
            past_matches.append((past_label, m))

    all_matches_dict = {}
    upcoming_matches.sort(key=lambda x: x[1]['utcDate'])
    for lbl, m in upcoming_matches: all_matches_dict[lbl] = m
    past_matches.sort(key=lambda x: x[1]['utcDate'], reverse=True)
    for lbl, m in past_matches: all_matches_dict[lbl] = m

    if not all_matches_dict: st.warning("Maç verisi bulunamadı."); st.stop()

    with c2: 
        sel_match_name = st.selectbox("Maç Seçiniz", list(all_matches_dict.keys()))
    
    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
        start_time = time.time()
        
        m = all_matches_dict[sel_match_name]
        
        h_id, a_id = m['homeTeam']['id'], m['awayTeam']['id']
        
        h_stats = dm.get_team_stats(standings, fixtures, h_id, 'HOME', m['homeTeam']['name'])
        a_stats = dm.get_team_stats(standings, fixtures, a_id, 'AWAY', m['awayTeam']['name'])
        
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

        eng = SimulationEngine(use_fixed_seed=not use_dynamic)
        
        with st.spinner(f"Kuantum motoru hesaplıyor: {h_stats['name']} (Ev) vs {a_stats['name']} (Dep)..."):
            raw = eng.run_monte_carlo(h_stats, a_stats, league_avg, params)
            res = eng.analyze_results(raw)

        duration = time.time() - start_time
        log_activity(leagues[sel_league], sel_match_name, h_att, a_att, sim_count, duration)

        st.divider()
        st.info(generate_commentary(h_stats, a_stats, res))
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><img src='{h_logo}' width='60'><br><div class='stat-lbl'>{h_stats['name']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><br><br><div class='stat-lbl'>BERABERLİK</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><img src='{a_logo}' width='60'><br><div class='stat-lbl'>{a_stats['name']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)
        
        st.write("")
        st.progress(res['1x2'][0]/100, text=f"Ev Sahibi Galibiyet İhtimali: %{res['1x2'][0]:.1f}")
        
        st.caption(f"📊 Güven Aralığı (Ev): ±%{res['ci']['h']:.1f} | (Dep): ±%{res['ci']['a']:.1f} | ⏱️ Hesaplama: {duration:.3f} sn")

        st.markdown(f"""
        ### 📊 Detaylı Analiz 
        <span style='color:#00ff88; font-weight:bold'>Ev xG: {res['xg'][0]:.2f}</span> - 
        <span style='color:#ff4b4b; font-weight:bold'>Dep xG: {res['xg'][1]:.2f}</span>
        """, unsafe_allow_html=True)
        
        h_form_display = " ".join(list(h_stats['form'])) if h_stats['form'] else "Nötr"
        a_form_display = " ".join(list(a_stats['form'])) if a_stats['form'] else "Nötr"
        st.caption(f"📈 Form Durumu (Yeni -> Eski): {h_stats['name']} [{h_form_display}] - {a_stats['name']} [{a_form_display}]")

        tab1, tab2, tab3 = st.tabs(["Skor Matrisi", "En Olası Skorlar", "HT/FT"])
        
        with tab1:
            fig = go.Figure(data=go.Heatmap(
                z=res['matrix'], 
                colorscale='Magma',
                x=[0, 1, 2, 3, 4, 5, "6+"],
                y=[0, 1, 2, 3, 4, 5, "6+"]
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
