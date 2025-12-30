import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# -----------------------------------------------------------------------------
# 1. KONFİGÜRASYON & API ANAHTARLARI
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    # ESKİ API (Yedek)
    "STD_API_URL": "https://api.football-data.org/v4",
    # YENİ API (Pro - SportMonks)
    "PRO_API_URL": "https://api.sportmonks.com/v3/football",
    "PRO_TOKEN": "GL0xxZHLVkzEUypMQdNkKow4NI0FPrlzJ4IfalN7rV6Qlc2u3M1iXDlAfCzx", # Senin Yeni Key'in
    "COLORS": {"H": "#3b82f6", "D": "#94a3b8", "A": "#ef4444"}
}

st.set_page_config(page_title="Quantum Football", page_icon="⚽", layout="wide")

# -----------------------------------------------------------------------------
# 2. DİL VE ARAYÜZ METİNLERİ
# -----------------------------------------------------------------------------
TRANSLATIONS = {
    "tr": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Ayarlar",
        "source_sel": "Veri Kaynağı (Data Source)",
        "api_ph": "Eski API Key (Opsiyonel)",
        "sim_param": "Simülasyon Ayarları",
        "match_count": "Simülasyon Sayısı",
        "form_set": "Takım Form Ayarları",
        "missing_p": "Eksik Kilit Oyuncu",
        "h_miss": "Ev Sahibi Eksik",
        "a_miss": "Deplasman Eksik",
        "h_att": "Ev Sahibi Gücü",
        "a_att": "Deplasman Gücü",
        "league": "Lig Seçimi",
        "match": "Maç Seçimi",
        "start_btn": "ANALİZİ BAŞLAT",
        "calculating": "Kuantum motoru verileri işliyor...",
        "xg": "Beklenen Gol (xG)",
        "home": "EV SAHİBİ", "draw": "BERABERLİK", "away": "DEPLASMAN",
        "heatmap": "Skor Matrisi",
        "top_scores": "En Olası Skorlar",
        "ht_ft": "İY/MS Dağılımı",
        "total_goal": "Toplam Gol",
        "no_match": "Bu ligde yakında maç bulunamadı.",
        "footer": "Quantum Football v55.0 Dual-Core © 2026 | Powered by SportMonks & Football-Data"
    },
    "en": {
        "app_title": "QUANTUM FOOTBALL",
        "settings": "Settings",
        "source_sel": "Data Source",
        "api_ph": "Old API Key (Optional)",
        "sim_param": "Simulation Settings",
        "match_count": "Simulation Count",
        "form_set": "Team Form Settings",
        "missing_p": "Missing Key Players",
        "h_miss": "Home Missing",
        "a_miss": "Away Missing",
        "h_att": "Home Strength",
        "a_att": "Away Strength",
        "league": "Select League",
        "match": "Select Match",
        "start_btn": "START ANALYSIS",
        "calculating": "Quantum engine processing data...",
        "xg": "Expected Goals (xG)",
        "home": "HOME WIN", "draw": "DRAW", "away": "AWAY WIN",
        "heatmap": "Score Matrix",
        "top_scores": "Most Likely Scores",
        "ht_ft": "HT/FT Distribution",
        "total_goal": "Total Goals",
        "no_match": "No upcoming matches found.",
        "footer": "Quantum Football v55.0 Dual-Core © 2026 | Powered by SportMonks & Football-Data"
    }
}

# -----------------------------------------------------------------------------
# 3. CSS STİLİ
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #0f172a; font-family: 'Inter', sans-serif; color: #f8fafc;}
    
    .main-title {
        font-family: 'Roboto Mono', monospace; font-size: 3rem; font-weight: 800;
        text-align: center; margin-bottom: 10px;
        background: linear-gradient(90deg, #3b82f6, #10b981);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important; 
    }
    
    .stat-card {
        background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 15px;
        text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stat-val { font-size: 2rem; font-weight: 700; color: #fff; }
    .stat-lbl { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    
    .analysis-box {
        background: rgba(30, 41, 59, 0.4); border: 1px solid #334155; 
        border-radius: 12px; padding: 15px; height: 100%;
    }
    
    .score-row {
        display: flex; justify-content: space-between; padding: 8px; 
        border-bottom: 1px solid #334155; font-family: 'Roboto Mono';
    }
    
    .footer {
        margin-top: 50px; padding-top: 20px; border-top: 1px solid #334155;
        text-align: center; color: #64748b; font-size: 0.8rem;
    }
    
    div[data-testid="stButton"] button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. DATA MANAGERS (HYBRID SYSTEM)
# -----------------------------------------------------------------------------
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'match_info' not in st.session_state: st.session_state.match_info = None

# SportMonks Lig ID Haritası (Manual Mapping)
SM_LEAGUE_MAP = {
    "TR1": 600,   # Süper Lig
    "PL": 8,      # Premier League
    "CL": 2,      # Champions League
    "ELC": 9,     # Championship
    "PD": 564,    # La Liga
    "BL1": 82,    # Bundesliga
    "SA": 384,    # Serie A
    "FL1": 301,   # Ligue 1
    "DED": 72,    # Eredivisie
    "PPL": 462,   # Liga Portugal
    "BSA": 2026   # Brazil Serie A
}

class StandardDataManager:
    """Eski Sistem (Yedek)"""
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    def fetch_data(self, league_code):
        standings_data = {"standings": [{"table": []}]}
        matches_data = {"matches": []}
        try:
            r1 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/standings", headers=self.headers)
            if r1.status_code == 200: standings_data = r1.json()
            
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{CONFIG['STD_API_URL']}/competitions/{league_code}/matches", 
                              headers=self.headers, params={"dateFrom": today, "dateTo": future})
            if r2.status_code == 200: matches_data = r2.json()
        except: pass
        return standings_data, matches_data, "std"

class ProDataManager:
    """Yeni Sistem (SportMonks)"""
    def __init__(self, api_token):
        self.token = api_token

    def fetch_data(self, league_code):
        # Lig kodunu SportMonks ID'sine çevir
        league_id = SM_LEAGUE_MAP.get(league_code)
        if not league_id: return None, None, "pro"

        # 1. Standings (Puan Durumu) - Güç hesaplama için
        standings_url = f"{CONFIG['PRO_API_URL']}/standings/seasons/latest" 
        # Not: SportMonks v3'te yapı biraz karışık, basitlik için fikstüre odaklanacağız
        # Puan durumu çekmek karmaşık olduğu için burada fikstür odaklı gidiyoruz.
        
        # 2. Fikstür (Gelecek Maçlar)
        start = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d")
        
        matches_formatted = {"matches": []}
        
        try:
            # Lig ID'sine göre fikstür
            url = f"{CONFIG['PRO_API_URL']}/fixtures/between/{start}/{end}"
            params = {
                "api_token": self.token,
                "include": "participants;league;venue", # Önemli: Takımları dahil et
                "filters": f"leagues:{league_id}" # Sadece seçilen lig
            }
            
            r = requests.get(url, params=params)
            data = r.json().get("data", [])
            
            for m in data:
                # Veriyi standart formata dönüştür
                home = next((p for p in m['participants'] if p['meta']['location'] == 'home'), {})
                away = next((p for p in m['participants'] if p['meta']['location'] == 'away'), {})
                
                matches_formatted["matches"].append({
                    "id": m['id'],
                    "utcDate": m['starting_at'],
                    "status": "SCHEDULED",
                    "homeTeam": {
                        "id": home.get('id'), 
                        "name": home.get('name'), 
                        "crest": home.get('image_path') # GERÇEK LOGO BURADA!
                    },
                    "awayTeam": {
                        "id": away.get('id'), 
                        "name": away.get('name'), 
                        "crest": away.get('image_path') # GERÇEK LOGO BURADA!
                    },
                    "competition": {"name": m.get('league', {}).get('name', 'Cup')}
                })
        except Exception as e:
            st.error(f"SportMonks Hatası: {str(e)}")
            
        return None, matches_formatted, "pro"

# -----------------------------------------------------------------------------
# 5. SIMULATION ENGINE (DEĞİŞMEDİ)
# -----------------------------------------------------------------------------
class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        xg_h = h_attack * a_def * avg_g * params['home_adv']
        xg_a = a_attack * h_def * avg_g

        if params.get('h_missing', 0) > 0: xg_h *= (1 - (params['h_missing'] * 0.12))
        if params.get('a_missing', 0) > 0: xg_a *= (1 - (params['a_missing'] * 0.12))

        sims = params['sim_count']
        gh_ht = self.rng.poisson(xg_h * 0.45, sims)
        ga_ht = self.rng.poisson(xg_a * 0.45, sims)
        gh_ft = self.rng.poisson(xg_h * 0.55, sims)
        ga_ft = self.rng.poisson(xg_a * 0.55, sims)

        return {"h": gh_ht + gh_ft, "a": ga_ht + ga_ft, "ht": (gh_ht, ga_ht), "xg": (xg_h, xg_a), "sims": sims}

    def analyze_results(self, data):
        h, a = data["h"], data["a"]
        sims = data["sims"]
        
        p_home = np.mean(h > a) * 100
        p_draw = np.mean(h == a) * 100
        p_away = np.mean(h < a) * 100
        
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6): matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100
            
        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]
        
        total_goals = h + a
        goal_bins = {
            "0-1": np.sum(total_goals <= 1)/sims*100, "2-3": np.sum((total_goals>=2)&(total_goals<=3))/sims*100,
            "4-6": np.sum((total_goals>=4)&(total_goals<=6))/sims*100, "7+": np.sum(total_goals>=7)/sims*100
        }
        
        # HT/FT Basitleştirilmiş
        h_ht, a_ht = data["ht"]
        ht_res = np.where(h_ht > a_ht, 1, np.where(h_ht < a_ht, 2, 0))
        ft_res = np.where(h > a, 1, np.where(h < a, 2, 0))
        htft = {}
        labels = {1: "1", 0: "X", 2: "2"}
        for i in [1,0,2]:
            for j in [1,0,2]:
                mask = (ht_res == i) & (ft_res == j)
                htft[f"{labels[i]}/{labels[j]}"] = np.sum(mask)/sims*100

        return {"1x2": [p_home, p_draw, p_away], "matrix": matrix, "top_scores": top_scores, "goal_bins": goal_bins, "htft": htft, "xg": data["xg"]}

# -----------------------------------------------------------------------------
# 6. MAIN APP
# -----------------------------------------------------------------------------
def main():
    with st.sidebar:
        col_lang, col_exit = st.columns([4, 1])
        with col_lang:
            lang_code = st.selectbox("Dil / Language", ["tr", "en"], label_visibility="collapsed")
        with col_exit:
            if st.button("❌"): st.session_state.clear(); st.rerun()
        
        t = TRANSLATIONS[lang_code]
        st.divider()
        
        st.header(f"🧪 {t['settings']}")
        
        # --- VERİ KAYNAĞI SEÇİMİ (HYBRID) ---
        source_option = st.radio(t['source_sel'], ["PRO (SportMonks)", "STANDARD (Free)"], index=0)
        
        api_key = None
        if "STANDARD" in source_option:
            api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
            if not api_key:
                api_key = st.text_input(t['api_ph'], type="password")
        
        st.subheader(t['sim_param'])
        sim_count = st.select_slider(t['match_count'], options=[10000, 100000, 500000], value=100000)
        
        st.caption(t['form_set'])
        h_att = st.slider(t['h_att'], 80, 120, 100) / 100
        a_att = st.slider(t['a_att'], 80, 120, 100) / 100

        st.caption(f"🚑 {t['missing_p']}")
        c1, c2 = st.columns(2)
        h_miss = c1.number_input(t['h_miss'], 0, 5, 0)
        a_miss = c2.number_input(t['a_miss'], 0, 5, 0)
        
        params = {"sim_count": sim_count, "h_att_factor": h_att, "h_def_factor": 1.0, 
                  "a_att_factor": a_att, "a_def_factor": 1.0, "h_missing": h_miss, "a_missing": a_miss, "home_adv": 1.15}
        
        st.divider()
        st.markdown("""<div style="text-align: center;"><a href="https://www.buymeacoffee.com/muratlola" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" style="height: 40px !important;" ></a></div>""", unsafe_allow_html=True)

    st.markdown(f"<div class='main-title'>{t['app_title']}</div>", unsafe_allow_html=True)

    # --- LIG LISTESI ---
    L_MAP = {
        "Süper Lig & Kupa (Türkiye)": "TR1", "Premier League (İngiltere)": "PL", "La Liga (İspanya)": "PD",
        "Bundesliga (Almanya)": "BL1", "Serie A (İtalya)": "SA", "Ligue 1 (Fransa)": "FL1",
        "Eredivisie (Hollanda)": "DED", "Primeira Liga (Portekiz)": "PPL", "Série A (Brezilya)": "BSA",
        "UEFA Şampiyonlar Ligi": "CL"
    }
    
    col1, col2 = st.columns([1, 2])
    with col1: league = st.selectbox(t['league'], list(L_MAP.keys()))
    
    # --- VERİ ÇEKME ---
    if "PRO" in source_option:
        manager = ProDataManager(CONFIG["PRO_TOKEN"])
    else:
        if not api_key: st.stop()
        manager = StandardDataManager(api_key)
        
    standings, fixtures, source_type = manager.fetch_data(L_MAP[league])
    
    if not fixtures or not fixtures.get("matches"):
        st.info(t['no_match'])
        st.stop()

    # --- TAKIM VERİLERİ (GÜÇ HESABI) ---
    teams = {}
    avg_league = 2.5
    
    # Standard Mode: Puan tablosundan veri çek
    if source_type == "std" and standings and "standings" in standings:
        table = standings["standings"][0].get("table", [])
        if table:
            total_g = sum(x["goalsFor"] for x in table)
            total_m = sum(x["playedGames"] for x in table)
            avg_league = total_g / total_m if total_m > 0 else 2.5
            for row in table:
                teams[row["team"]["id"]] = {
                    "name": row["team"]["name"], 
                    "crest": row["team"].get("crest"),
                    "gf": row["goalsFor"]/row["playedGames"], 
                    "ga": row["goalsAgainst"]/row["playedGames"]
                }
    
    # Pro Mode: Veri tablosu gelmediği için varsayılan güç + Manuel Ayar
    elif source_type == "pro":
        # Pro modda takımları maçlardan topluyoruz
        for m in fixtures["matches"]:
            h = m["homeTeam"]
            a = m["awayTeam"]
            # Pro modda varsayılan güç atıyoruz (Manuel Stats ile düzelecek)
            if h["id"] not in teams: teams[h["id"]] = {"name": h["name"], "crest": h["crest"], "gf": 1.4, "ga": 1.1}
            if a["id"] not in teams: teams[a["id"]] = {"name": a["name"], "crest": a["crest"], "gf": 1.3, "ga": 1.2}

    matches_list = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})": m for m in fixtures["matches"]}
    with col2: sel_match = st.selectbox(t['match'], list(matches_list.keys()))

    if st.button(f"{t['start_btn']} ({sim_count//1000}K)", use_container_width=True):
        m = matches_list[sel_match]
        h_id, a_id = m["homeTeam"]["id"], m["awayTeam"]["id"]
        
        # Logo Seçimi: Pro modda API'den, yoksa yedekten
        h_crest = m["homeTeam"].get("crest") or CONFIG["DEFAULT_LOGO"]
        a_crest = m["awayTeam"].get("crest") or CONFIG["DEFAULT_LOGO"]
        
        h_team = teams.get(h_id, {"name": m["homeTeam"]["name"], "crest": h_crest, "gf": 1.5, "ga": 1.2})
        a_team = teams.get(a_id, {"name": m["awayTeam"]["name"], "crest": a_crest, "gf": 1.4, "ga": 1.3})
        
        # MANUEL GÜÇ ENJEKSİYONU (Özellikle Pro Mod için kritik çünkü standings çekmedik)
        # Burası takımların saldırı/savunma gücünü belirler
        MANUAL_STATS = {
            2054: {"gf": 2.50, "ga": 0.80}, # Galatasaray
            2052: {"gf": 2.55, "ga": 0.85}, # Fenerbahçe (Std ID)
            88: {"gf": 2.55, "ga": 0.85},   # Fenerbahçe (SportMonks ID)
            2061: {"gf": 1.75, "ga": 1.10}, # Trabzonspor
            688: {"gf": 1.75, "ga": 1.10},  # Trabzonspor (SM ID)
            554: {"gf": 2.10, "ga": 1.00},  # Beşiktaş (SM ID)
            2036: {"gf": 2.10, "ga": 1.00}  # Beşiktaş
        }
        
        if h_id in MANUAL_STATS: h_team.update(MANUAL_STATS[h_id])
        if a_id in MANUAL_STATS: a_team.update(MANUAL_STATS[a_id])

        eng = SimulationEngine()
        with st.spinner(t['calculating']):
            # Kupa kontrolü
            if "Cup" in m.get("competition", {}).get("name", "") or "Kupa" in m.get("competition", {}).get("name", ""):
                params["home_adv"] = 1.05
            
            raw_data = eng.run_monte_carlo(h_team, a_team, avg_league, params)
            res = eng.analyze_results(raw_data)
            
        st.session_state.sim_results = res
        st.session_state.match_info = {"h": h_team, "a": a_team}

    # --- SONUÇ EKRANI ---
    if st.session_state.sim_results:
        res = st.session_state.sim_results
        info = st.session_state.match_info
        
        c_h, c_vs, c_a = st.columns([2,1,2])
        with c_h: st.markdown(f"<div style='text-align:center'><img src='{info['h']['crest']}' width='80'><br><h3>{info['h']['name']}</h3></div>", unsafe_allow_html=True)
        with c_vs: 
            st.markdown("<h1 style='text-align:center; color:#94a3b8'>VS</h1>", unsafe_allow_html=True)
            st.metric(t['xg'], f"{res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
        with c_a: st.markdown(f"<div style='text-align:center'><img src='{info['a']['crest']}' width='80'><br><h3>{info['a']['name']}</h3></div>", unsafe_allow_html=True)

        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['home']}</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['draw']}</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='stat-card'><div class='stat-lbl'>{t['away']}</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)

        st.write("")
        c_heat, c_list = st.columns([2, 1])
        with c_heat:
            st.markdown(f"### 🔥 {t['heatmap']}")
            fig_heat = go.Figure(data=go.Heatmap(z=res["matrix"], x=[0,1,2,3,4,5], y=[0,1,2,3,4,5], colorscale='Magma', texttemplate="%{z:.1f}%"))
            fig_heat.update_layout(xaxis_title=t['away'], yaxis_title=t['home'], height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with c_list:
            st.markdown(f"### 🎯 {t['top_scores']}")
            with st.container():
                st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
                for score, prob in res["top_scores"][:7]:
                    st.markdown(f"<div class='score-row'><span style='font-weight:bold; font-size:1.2rem'>{score}</span><span style='color:#38bdf8; font-weight:bold'>%{prob:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        c_ht, c_goal = st.columns(2)
        with c_ht:
            st.markdown(f"### ⏱️ {t['ht_ft']}")
            htft_df = pd.DataFrame(list(res['htft'].items()), columns=['Result', 'Prob']).sort_values('Prob', ascending=False).head(7)
            fig_bar = px.bar(htft_df, x='Result', y='Prob', text_auto='.1f', color='Prob', color_continuous_scale='Viridis')
            fig_bar.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_goal:
            st.markdown(f"### 🥅 {t['total_goal']}")
            fig_pie = go.Figure(data=[go.Pie(labels=list(res["goal_bins"].keys()), values=list(res["goal_bins"].values()), hole=.4, marker=dict(colors=['#94a3b8', '#3b82f6', '#8b5cf6', '#f43f5e']))])
            fig_pie.update_layout(height=300, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
