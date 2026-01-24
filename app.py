import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import logging
from typing import Dict, Tuple, List, Any, Optional

# --- LOGGING AYARLARI ---
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
    "HOME_ADVANTAGE": 1.05,
    "WIN_BOOST": 0.04,
    "DRAW_BOOST": 0.01,
    "LOSS_PENALTY": -0.03,
    "MISSING_PLAYER_BASE_IMPACT": 0.08,
    "CACHE_TTL": 1800, 
    "FORM_WEIGHTS": [1.5, 1.25, 1.0, 0.75, 0.5],
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png"
}

st.set_page_config(page_title="Quantum Football AI", page_icon="⚽", layout="wide")

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
# 2. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def log_activity(league: str, match: str, h_att: float, a_att: float, sim_count: int, duration: float) -> None:
    if db is None: return
    try:
        q_params = st.query_params
        user = q_params.get("user_email", "Misafir")
        if isinstance(user, list): user = user[0]
        
        tier = "FREE" if user in ["Misafir", "Ziyaretci"] else "PRO"

        db.collection("analysis_logs").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "user": user,
            "tier": tier,
            "league": league,
            "match": match,
            "sims": sim_count,
            "duration_sec": duration,
            "settings": {"h": h_att, "a": a_att}
        })
    except Exception as e:
        logger.error(f"Firestore Log Error: {e}")

# -----------------------------------------------------------------------------
# 3. SİMÜLASYON MOTORU (THE FINAL TRUTH)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self, use_fixed_seed: bool = False):
        if use_fixed_seed:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats: Dict, a_stats: Dict, avg_g: float, params: Dict) -> Dict:
        sims = params['sim_count']
        
        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        base_xg_h = h_attack * a_def * avg_g
        base_xg_a = a_attack * h_def * avg_g

        # Form Hesabı
        def _calc_form_boost(form_str: str) -> float:
            if not form_str: return 1.0
            matches = form_str.replace(',', '')
            boost = 1.0
            weights = CONSTANTS["FORM_WEIGHTS"]
            recent = matches[:5]
            curr_weights = weights[:len(recent)]

            for i, char in enumerate(recent):
                w = curr_weights[i]
                if char == 'W': boost += CONSTANTS["WIN_BOOST"] * w
                elif char == 'D': boost += CONSTANTS["DRAW_BOOST"] * w
                elif char == 'L': boost += CONSTANTS["LOSS_PENALTY"] * w
            
            return max(0.85, min(boost, 1.25))

        base_xg_h *= _calc_form_boost(h_stats.get('form', ''))
        base_xg_a *= _calc_form_boost(a_stats.get('form', ''))

        base_xg_h *= CONSTANTS["HOME_ADVANTAGE"]

        # Convex Eksik Oyuncu Etkisi
        if params.get('h_missing', 0) > 0: 
            impact_h = 1 - (1 - CONSTANTS["MISSING_PLAYER_BASE_IMPACT"]) ** params['h_missing']
            base_xg_h *= (1 - impact_h)
        
        if params.get('a_missing', 0) > 0: 
            impact_a = 1 - (1 - CONSTANTS["MISSING_PLAYER_BASE_IMPACT"]) ** params['a_missing']
            base_xg_a *= (1 - impact_a)

        sigma = 0.05 if params['tier'] == 'PRO' else 0.12
        
        random_factors_h = self.rng.normal(1, sigma, sims)
        random_factors_a = self.rng.normal(1, sigma, sims)
        
        # Clip Upper Bound (12 gol) - Grok Önerisi
        final_xg_h = np.clip(base_xg_h * random_factors_h, 0.05, 12.0)
        final_xg_a = np.clip(base_xg_a * random_factors_a, 0.05, 12.0)

        # --- GAMMA-POISSON (VEKTÖREL HIZLANDIRILMIŞ) ---
        def simulate_goals(xg_array):
            alpha = 10.0 # Dispersion
            # Shape parametresi vektörel olabilir (NumPy destekler)
            gamma_variate = self.rng.gamma(shape=xg_array * alpha, scale=1/alpha)
            return self.rng.poisson(gamma_variate)

        gh_ht = simulate_goals(final_xg_h * 0.45)
        ga_ht = simulate_goals(final_xg_a * 0.45)
        gh_ft = simulate_goals(final_xg_h * 0.55)
        ga_ft = simulate_goals(final_xg_a * 0.55)

        total_h = gh_ht + gh_ft
        total_a = ga_ht + ga_ft

        return {
            "h": total_h, "a": total_a,
            "ht": (gh_ht, ga_ht), "ft": (total_h, total_a),
            "xg_dist": (final_xg_h, final_xg_a),
            "sims": sims
        }

    def analyze_results(self, data: Dict) -> Dict:
        h, a = data["h"], data["a"]
        ht_h, ht_a = data["ht"]
        sims = data["sims"]

        p_home = np.mean(h > a) * 100
        p_draw = np.mean(h == a) * 100
        p_away = np.mean(h < a) * 100

        def calc_ci(p, n):
            return 1.96 * np.sqrt((p/100 * (1 - p/100)) / n) * 100
        
        ci = {
            "h": calc_ci(p_home, sims),
            "d": calc_ci(p_draw, sims),
            "a": calc_ci(p_away, sims)
        }

        # Skor Matrisi (6+ Bucket)
        matrix = np.zeros((7, 7))
        h_clipped = np.clip(h, 0, 6)
        a_clipped = np.clip(a, 0, 6)
        for i in range(7):
            for j in range(7):
                matrix[i, j] = np.sum((h_clipped == i) & (a_clipped == j)) / sims * 100

        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]

        # HT/FT Hesaplama
        ht_res = np.where(ht_h > ht_a, 1, np.where(ht_h < ht_a, 2, 0))
        ft_res = np.where(h > a, 1, np.where(h < a, 2, 0))
        
        htft_probs = {}
        outcome_labels = {1: "1", 0: "X", 2: "2"}
        for ht in [1, 0, 2]:
            for ft in [1, 0, 2]:
                mask = (ht_res == ht) & (ft_res == ft)
                htft_probs[f"{outcome_labels[ht]}/{outcome_labels[ft]}"] = np.mean(mask) * 100

        # Bilimsel Entropy (Düzeltildi)
        flat_matrix = matrix.flatten()
        flat_matrix = flat_matrix[flat_matrix > 0] / 100
        raw_entropy = -np.sum(flat_matrix * np.log(flat_matrix))
        max_entropy = np.log(len(flat_matrix)) if len(flat_matrix) > 0 else 1
        normalized_entropy = (raw_entropy / max_entropy) # 0-1 arası

        # Pazarlar
        dc = {"1X": p_home + p_draw, "X2": p_away + p_draw, "12": p_home + p_away}
        btts_yes = np.mean((h > 0) & (a > 0)) * 100
        over_25 = np.mean((h + a) > 2.5) * 100
        
        goal_diff = h - a
        diff_bins = {
            "≤-3": np.mean(goal_diff <= -3) * 100,
            "-2": np.mean(goal_diff == -2) * 100,
            "-1": np.mean(goal_diff == -1) * 100,
            "0": np.mean(goal_diff == 0) * 100,
            "+1": np.mean(goal_diff == 1) * 100,
            "+2": np.mean(goal_diff == 2) * 100,
            "≥+3": np.mean(goal_diff >= 3) * 100,
        }

        return {
            "1x2": [p_home, p_draw, p_away],
            "ci": ci,
            "matrix": matrix,
            "top_scores": top_scores,
            "htft": htft_probs,
            "goal_diff": diff_bins,
            "entropy": normalized_entropy,
            "dc": dc,
            "btts": btts_yes,
            "over_25": over_25,
            "xg_dist": data["xg_dist"]
        }

# -----------------------------------------------------------------------------
# 4. VERİ YÖNETİCİSİ
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, api_key: str):
        self.headers = {"X-Auth-Token": api_key}

    def fetch_data(self, league_code: str) -> Tuple[Optional[Tuple], Optional[datetime]]:
        cache_key = f"data_{league_code}"
        
        if cache_key in st.session_state:
            last_fetch, data = st.session_state[cache_key]
            if (datetime.now() - last_fetch).seconds < CONSTANTS["CACHE_TTL"]:
                return data, last_fetch

        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league_code}/standings", headers=self.headers)
            if r1.status_code == 429:
                st.warning("⚠️ API Limiti. Önbellek kullanılıyor.")
                if cache_key in st.session_state:
                    return st.session_state[cache_key][1], st.session_state[cache_key][0]
                return None, None
            
            standings = r1.json()
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league_code}/matches", headers=self.headers)
            matches = r2.json() if r2.status_code == 200 else {}
            
            st.session_state[cache_key] = (datetime.now(), (standings, matches))
            return (standings, matches), datetime.now()

        except Exception as e:
            logger.error(f"API Fetch Error: {e}")
            if cache_key in st.session_state:
                st.warning("⚠️ Bağlantı hatası. Önbellek kullanılıyor.")
                return st.session_state[cache_key][1], st.session_state[cache_key][0]
            return None, None

    def _calculate_form_from_matches(self, matches_data: Dict, team_id: int) -> str:
        if not matches_data or 'matches' not in matches_data: return ""
        played = [m for m in matches_data['matches'] 
                  if m['status'] == 'FINISHED' and 
                  (m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id)]
        played.sort(key=lambda x: x['utcDate'], reverse=True)
        form_chars = []
        for m in played[:5]: 
            winner = m['score']['winner']
            if winner == 'DRAW': form_chars.append('D')
            elif (winner == 'HOME_TEAM' and m['homeTeam']['id'] == team_id) or \
                 (winner == 'AWAY_TEAM' and m['awayTeam']['id'] == team_id): form_chars.append('W')
            else: form_chars.append('L')
        return ",".join(form_chars) 

    def get_team_stats(self, standings: Dict, matches: Dict, team_id: int, table_type: str = 'TOTAL', default_name: str = "Takım") -> Dict:
        try:
            target_table = []
            s_list = standings.get('standings', [])
            if not s_list: 
                return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4, "form": "", "crest": CONSTANTS["DEFAULT_LOGO"]}

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
                    crest = row['team'].get('crest', CONSTANTS["DEFAULT_LOGO"])

                    if not form and matches:
                        form = self._calculate_form_from_matches(matches, team_id)
                    
                    form = form.replace(',', '') if form else ""

                    if played < 2: 
                        return {"name": row['team']['name'], "id": team_id, "gf": 1.5, "ga": 1.5, "form": form, "crest": crest}
                    
                    return {
                        "name": row['team']['name'],
                        "id": team_id,
                        "gf": row['goalsFor'] / played,
                        "ga": row['goalsAgainst'] / played,
                        "form": form,
                        "crest": crest
                    }
        except Exception as e:
            logger.error(f"Stats Parse Error: {e}")
            pass
        return {"name": default_name, "id": team_id, "gf": 1.4, "ga": 1.4, "form": "", "crest": CONSTANTS["DEFAULT_LOGO"]}

    def get_league_avg(self, standings: Dict) -> float:
        try:
            table = standings.get('standings', [])[0].get('table', [])
            total_goals = sum(t['goalsFor'] for t in table)
            total_games = sum(t['playedGames'] for t in table)
            return total_goals / (total_games / 2) if total_games > 10 else 2.8
        except:
            return 2.8

# -----------------------------------------------------------------------------
# 5. UI MAIN
# -----------------------------------------------------------------------------
def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Inter:wght@400;700&display=swap');
        .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #fff; }
        .stat-card { background: #151922; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center; }
        .stat-val { font-size: 1.8rem; font-weight: 900; color: #fff; }
        .stat-lbl { font-size: 0.8rem; color: #888; text-transform: uppercase; }
        .stProgress > div > div > div > div { background-color: #00ff88; }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 👤 Kullanıcı Paneli")
        
        q_params = st.query_params
        user_email = q_params.get("user_email", "Misafir")
        if isinstance(user_email, list): user_email = user_email[0]
        is_guest = (user_email in ["Misafir", "Ziyaretci"])
        user_tier = "FREE" if is_guest else "PRO"

        if is_guest:
            st.warning("🔒 Misafir Modu")
        else:
            st.success(f"Hoşgeldin, {user_email}")
        
        with st.expander("ℹ️ Üyelik Avantajları", expanded=False):
            st.markdown("""
            | Özellik | FREE | PRO ⚡ |
            |---|---|---|
            | **Simülasyon** | 10.000 | **500.000** |
            | **Form Analizi** | ❌ | ✅ |
            """)

        st.divider()
        st.subheader("⚙️ Ayarlar")
        
        use_dynamic = st.checkbox("🎲 Dinamik Simülasyon", value=True, help="Gerçek Monte Carlo (Her seferinde farklı).")
        
        max_sim = 500000 if not is_guest else 10000
        default_sim = 100000 if not is_guest else 1000
        sim_count = st.slider("Simülasyon Sayısı", 1000, max_sim, default_sim, step=1000)
        
        st.caption("Takım Form Çarpanları")
        h_att = st.slider("Ev Sahibi", 0.8, 1.2, 1.0)
        a_att = st.slider("Deplasman", 0.8, 1.2, 1.0)

        h_miss, a_miss = 0, 0
        if not is_guest:
            st.caption("🚑 Eksik Oyuncu (PRO)")
            h_miss = st.number_input("Ev Sahibi Eksik", 0, 5, 0)
            a_miss = st.number_input("Deplasman Eksik", 0, 5, 0)
            if h_miss > 5 or a_miss > 5:
                st.warning("⚠️ 5'ten fazla eksik kritik hataya yol açabilir.")

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
        
        h_logo = h_stats.get('crest') or CONSTANTS["DEFAULT_LOGO"]
        a_logo = a_stats.get('crest') or CONSTANTS["DEFAULT_LOGO"]
        
        league_avg = dm.get_league_avg(standings)

        params = {
            "sim_count": sim_count,
            "h_att_factor": h_att, "h_def_factor": 1.0,
            "a_att_factor": a_att, "a_def_factor": 1.0,
            "h_missing": h_miss, "a_missing": a_miss,
            "home_adv": CONSTANTS["HOME_ADVANTAGE"],
            "tier": user_tier
        }

        eng = SimulationEngine(use_fixed_seed=not use_dynamic)
        
        with st.spinner(f"Kuantum motoru {sim_count} simülasyonu hesaplıyor..."):
            raw = eng.run_monte_carlo(h_stats, a_stats, league_avg, params)
            res = eng.analyze_results(raw)

        duration = time.time() - start_time
        log_activity(leagues[sel_league], sel_match_name, h_att, a_att, sim_count, duration)

        st.divider()
        
        # --- UI: PURE DATA DASHBOARD ---
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><img src='{h_logo}' width='60'><br><div class='stat-lbl'>{h_stats['name']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div><small>±{res['ci']['h']:.1f}</small></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><br><br><div class='stat-lbl'>BERABERLİK</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div><small>±{res['ci']['d']:.1f}</small></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><img src='{a_logo}' width='60'><br><div class='stat-lbl'>{a_stats['name']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div><small>±{res['ci']['a']:.1f}</small></div>", unsafe_allow_html=True)
        
        st.write("")
        st.progress(res['1x2'][0]/100, text=f"Ev Sahibi Kazanma Olasılığı: %{res['1x2'][0]:.1f}")
        
        st.subheader("📊 Gelişmiş İhtimaller")
        col_dc, col_goal = st.columns(2)
        
        with col_dc:
            st.markdown("**Çifte Şans**")
            st.progress(res['dc']['1X'] / 100, text=f"1X (Ev Yenilmez): %{res['dc']['1X']:.1f}")
            st.progress(res['dc']['X2'] / 100, text=f"X2 (Dep Yenilmez): %{res['dc']['X2']:.1f}")
            st.progress(res['dc']['12'] / 100, text=f"12 (Beraberlik Yok): %{res['dc']['12']:.1f}")
            
        with col_goal:
            st.markdown("**Gol Pazarı**")
            st.progress(res['btts'] / 100, text=f"KG Var (BTTS): %{res['btts']:.1f}")
            st.progress(res['over_25'] / 100, text=f"2.5 Üst: %{res['over_25']:.1f}")
            
        st.subheader("⏱️ İlk Yarı / Maç Sonucu (HT/FT)")
        htft_df = pd.DataFrame(list(res['htft'].items()), columns=['Tercih', 'Olasılık'])
        htft_df = htft_df.sort_values('Olasılık', ascending=False).head(5)
        st.dataframe(htft_df.set_index('Tercih'), use_container_width=True)

        st.subheader("🌪️ Maç Kaos Seviyesi (Entropy)")
        entropy_val = res["entropy"]
        st.progress(min(entropy_val, 1.0)) # 0-1 arası zaten normalize
        st.caption(f"Entropy: {entropy_val:.3f} (Yüksek = Sürprize Açık)")

        st.subheader("⚖️ Gol Farkı Dağılımı")
        df_diff = pd.DataFrame({"Fark": list(res["goal_diff"].keys()), "Olasılık": list(res["goal_diff"].values())})
        fig_diff = px.bar(df_diff, x="Fark", y="Olasılık", text="Olasılık")
        fig_diff.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_diff.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_diff, use_container_width=True)

        st.subheader("📈 xG Dağılımı (Simülasyon)")
        xg_h, xg_a = res["xg_dist"]
        fig_xg = go.Figure()
        fig_xg.add_trace(go.Histogram(x=xg_h, name="Ev xG", opacity=0.6))
        fig_xg.add_trace(go.Histogram(x=xg_a, name="Dep xG", opacity=0.6))
        fig_xg.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_xg, use_container_width=True)

        with st.expander("📉 Adil Bahis Oranları (Implied Odds)"):
            odds = {k: round(100/v, 2) if v > 0 else 0 for k, v in zip(["1", "X", "2"], res['1x2'])}
            st.table(pd.DataFrame([odds]))

        h_form_display = " ".join(list(h_stats['form'])) if h_stats['form'] else "Nötr"
        a_form_display = " ".join(list(a_stats['form'])) if a_stats['form'] else "Nötr"
        st.caption(f"📈 Form (Yeni → Eski): {h_stats['name']} [{h_form_display}] - {a_stats['name']} [{a_form_display}]")

        tab1, tab2 = st.tabs(["Skor Matrisi", "En Olası Skorlar"])
        
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

if __name__ == "__main__":
    main()
