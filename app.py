import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import json

# --- FIREBASE EKLENTİLERİ ---
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# -----------------------------------------------------------------------------
# 1. KONFİGÜRASYON VE BAĞLANTILAR
# -----------------------------------------------------------------------------
CONFIG = {
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png",
    "API_URL": "https://api.football-data.org/v4",
    "ADMIN_PASS": "muratLola26", 
}

st.set_page_config(page_title="Quantum Football AI", page_icon="⚽", layout="wide")

# --- FIREBASE BAŞLATMA (GÜVENLİ) ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))
            firebase_admin.initialize_app(cred)
        else:
            st.warning("Firebase secrets bulunamadı. Loglama devre dışı.")
    except Exception as e:
        st.error(f"Firebase Bağlantı Hatası: {e}")

try:
    db = firestore.client()
except:
    db = None

# -----------------------------------------------------------------------------
# 2. KULLANICI VE ÜYELİK YÖNETİMİ (SAAS MANTIĞI)
# -----------------------------------------------------------------------------
# URL'den kullanıcıyı yakala
query_params = st.query_params
current_user_email = query_params.get("user_email", "Misafir")

# Kullanıcı listelerden geliyorsa düzelt
if isinstance(current_user_email, list):
    current_user_email = current_user_email[0]

# Üyelik Seviyesini Belirle
IS_GUEST = (current_user_email == "Misafir" or current_user_email == "Ziyaretci")
USER_TIER = "FREE" if IS_GUEST else "PRO"

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU VE LOGLAMA
# -----------------------------------------------------------------------------
def log_activity(league, match, h_att, a_att, sim_count):
    """Analizi buluta kaydeder"""
    if db is None: return

    log_data = {
        "timestamp": firestore.SERVER_TIMESTAMP,
        "user": current_user_email,
        "tier": USER_TIER,
        "league": league,
        "match": match,
        "simulations": sim_count,
        "settings": {"h": h_att, "a": a_att}
    }
    
    try:
        db.collection("analysis_logs").add(log_data)
        print("Log başarıyla kaydedildi.")
    except Exception as e:
        st.error(f"Veri kayıt hatası: {e}")

class SimulationEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def run_monte_carlo(self, h_stats, a_stats, avg_g, params):
        # Kuantum belirsizliği (Rastgelelik faktörü)
        uncertainty = 0.05 if params['tier'] == 'PRO' else 0.15 # Pro'da hata payı daha az

        h_attack = (h_stats['gf'] / avg_g) * params['h_att_factor']
        h_def = (h_stats['ga'] / avg_g) * params['h_def_factor']
        a_attack = (a_stats['gf'] / avg_g) * params['a_att_factor']
        a_def = (a_stats['ga'] / avg_g) * params['a_def_factor']

        xg_h = h_attack * a_def * avg_g * params['home_adv']
        xg_a = a_attack * h_def * avg_g

        # Eksik oyuncu etkisi
        if params.get('h_missing', 0) > 0: xg_h *= (1 - (params['h_missing'] * 0.08))
        if params.get('a_missing', 0) > 0: xg_a *= (1 - (params['a_missing'] * 0.08))

        sims = params['sim_count']
        
        # Poisson Dağılımı ile Maç Simülasyonu
        gh_ht = self.rng.poisson(xg_h * 0.45, sims)
        ga_ht = self.rng.poisson(xg_a * 0.45, sims)
        gh_ft = self.rng.poisson(xg_h * 0.55, sims)
        ga_ft = self.rng.poisson(xg_a * 0.55, sims)

        total_h = gh_ht + gh_ft
        total_a = ga_ht + ga_ft

        return {
            "h": total_h, "a": total_a,
            "ht": (gh_ht, ga_ht), "ft": (total_h, total_a),
            "xg": (xg_h, xg_a), "sims": sims
        }

    def analyze_results(self, data):
        h, a = data["h"], data["a"]
        sims = data["sims"]

        # Olasılık Hesapları
        p_home = np.mean(h > a) * 100
        p_draw = np.mean(h == a) * 100
        p_away = np.mean(h < a) * 100

        # Skor Matrisi
        matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                matrix[i, j] = np.sum((h == i) & (a == j)) / sims * 100

        # En Olası Skorlar
        scores = [f"{i}-{j}" for i, j in zip(h, a)]
        unique, counts = np.unique(scores, return_counts=True)
        top_scores = sorted(zip(unique, counts/sims*100), key=lambda x: x[1], reverse=True)[:10]

        # İY/MS
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
# 4. DATA MANAGER & LOGOS
# -----------------------------------------------------------------------------
TEAM_LOGOS = {
    2054: "https://upload.wikimedia.org/wikipedia/commons/f/f6/Galatasaray_Sports_Club_Logo.png",
    2052: "https://upload.wikimedia.org/wikipedia/tr/8/86/Fenerbah%C3%A7e_SK.png",
    2036: "https://upload.wikimedia.org/wikipedia/commons/2/20/Besiktas_jk.png",
    2061: "https://upload.wikimedia.org/wikipedia/tr/a/ab/Trabzonspor_Amblemi.png",
    2058: "https://upload.wikimedia.org/wikipedia/tr/e/e0/Samsunspor_logo_2.png",
}

class DataManager:
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}

    @st.cache_data(ttl=3600)
    def fetch_data(_self, league_code):
        try:
            # Puan Durumu
            r1 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/standings", headers=_self.headers)
            standings = r1.json() if r1.status_code == 200 else {}
            
            # Fikstür
            today = datetime.now().strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            r2 = requests.get(f"{CONFIG['API_URL']}/competitions/{league_code}/matches", 
                              headers=_self.headers, params={"dateFrom": today, "dateTo": future})
            matches = r2.json() if r2.status_code == 200 else {}
            
            return standings, matches
        except:
            return None, None

# -----------------------------------------------------------------------------
# 5. UYGULAMA ARAYÜZÜ (UI)
# -----------------------------------------------------------------------------
def main():
    # --- CSS TASARIM ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Inter:wght@400;700&display=swap');
        .stApp {background-color: #0b0f19; font-family: 'Inter', sans-serif;}
        
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #fff; }
        
        .tier-badge {
            padding: 5px 10px; border-radius: 5px; font-weight: bold; font-family: 'Orbitron';
            text-align: center; margin-bottom: 20px;
        }
        .free-tier { background: #334155; color: #94a3b8; border: 1px solid #475569; }
        .pro-tier { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; box-shadow: 0 0 10px rgba(0,255,136,0.2); }
        
        .stat-card {
            background: #151922; border: 1px solid #333; border-radius: 10px; padding: 15px; text-align: center;
        }
        .stat-val { font-size: 1.8rem; font-weight: 900; color: #fff; }
        .stat-lbl { font-size: 0.8rem; color: #888; text-transform: uppercase; }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: KULLANICI PROFİLİ ---
    with st.sidebar:
        st.markdown("## 👤 Kullanıcı Paneli")
        
        if IS_GUEST:
            st.markdown(f"<div class='tier-badge free-tier'>PLAN: {USER_TIER}</div>", unsafe_allow_html=True)
            st.warning("🔒 Misafir Modu\n\nAnaliz kapasitesi sınırlı. Full erişim için giriş yapın.")
        else:
            st.markdown(f"<div class='tier-badge pro-tier'>PLAN: {USER_TIER} ⚡</div>", unsafe_allow_html=True)
            st.success(f"Hoşgeldin,\n{current_user_email}")

        st.divider()
        
        # AYARLAR (Tier'a göre kilitli)
        st.subheader("⚙️ Simülasyon Ayarları")
        
        # Free: Max 10k, Pro: Max 1M
        max_sim = 1000000 if not IS_GUEST else 10000
        default_sim = 100000 if not IS_GUEST else 5000
        
        sim_count = st.slider("Simülasyon Sayısı", 1000, max_sim, default_sim, step=1000)
        
        st.markdown("---")
        st.caption("Takım Güç Çarpanları")
        h_att = st.slider("Ev Sahibi Form", 0.8, 1.2, 1.0)
        a_att = st.slider("Deplasman Form", 0.8, 1.2, 1.0)

        # Sadece PRO Özelliği: Eksik Oyuncu
        st.markdown("---")
        if not IS_GUEST:
            st.caption("🚑 Eksik Oyuncu Analizi (PRO)")
            h_miss = st.number_input("Ev Sahibi Eksik", 0, 5, 0)
            a_miss = st.number_input("Deplasman Eksik", 0, 5, 0)
        else:
            st.caption("🚑 Eksik Oyuncu Analizi (KİLİTLİ)")
            st.info("Bu özelliği açmak için giriş yapın.")
            h_miss, a_miss = 0, 0

        # --- ADMIN PANELİ (Firebase) ---
        st.markdown("---")
        with st.expander("🔐 Yönetici Girişi"):
            pw = st.text_input("Admin Şifresi", type="password")
            if pw == CONFIG["ADMIN_PASS"]:
                st.success("Admin Erişimi Aktif")
                if st.button("Son Kayıtları Getir"):
                    try:
                        docs = db.collection("analysis_logs").order_by("timestamp", direction="DESCENDING").limit(20).stream()
                        data = [d.to_dict() for d in docs]
                        st.dataframe(pd.DataFrame(data))
                    except:
                        st.error("Veri yok veya hata oluştu.")

    # --- ANA EKRAN ---
    st.markdown("<h1 style='text-align:center; color:#00ff88;'>QUANTUM FOOTBALL AI</h1>", unsafe_allow_html=True)
    
    # API KEY KONTROL
    api_key = st.secrets.get("FOOTBALL_API_KEY")
    if not api_key:
        st.error("API Anahtarı (secrets.toml) bulunamadı!")
        st.stop()

    dm = DataManager(api_key)
    
    # LİG SEÇİMİ
    col_l, col_m = st.columns([1, 2])
    with col_l:
        leagues = {"Süper Lig": "TR1", "Premier League": "PL", "La Liga": "PD", "Bundesliga": "BL1", "Serie A": "SA", "Şampiyonlar Ligi": "CL"}
        sel_league = st.selectbox("Lig Seçiniz", list(leagues.keys()))
    
    standings, fixtures = dm.fetch_data(leagues[sel_league])
    
    if not fixtures:
        st.info("Bu ligde planlanmış maç bulunamadı.")
        st.stop()
        
    matches = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in fixtures['matches'] if m['status'] == 'SCHEDULED'}
    
    with col_m:
        sel_match_name = st.selectbox("Maç Seçiniz", list(matches.keys()))
    
    # ANALİZ BUTONU
    if st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True):
        m = matches[sel_match_name]
        
        # Loglama
        log_activity(sel_league, sel_match_name, h_att, a_att, sim_count)
        
        # Takım Verilerini Hazırla (Basit Mock Data + API)
        h_id, a_id = m['homeTeam']['id'], m['awayTeam']['id']
        h_team = {"name": m['homeTeam']['name'], "gf": 1.6, "ga": 1.1} # Varsayılan
        a_team = {"name": m['awayTeam']['name'], "gf": 1.3, "ga": 1.4} # Varsayılan

        # Simülasyon Parametreleri
        params = {
            "sim_count": sim_count,
            "h_att_factor": h_att, "h_def_factor": 1.0,
            "a_att_factor": a_att, "a_def_factor": 1.0,
            "h_missing": h_miss, "a_missing": a_miss,
            "home_adv": 1.15,
            "tier": USER_TIER
        }

        eng = SimulationEngine()
        with st.spinner("Kuantum motoru olasılıkları hesaplıyor..."):
            raw = eng.run_monte_carlo(h_team, a_team, 2.8, params)
            res = eng.analyze_results(raw)

        # SONUÇ GÖSTERİMİ
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='stat-card'><div class='stat-lbl'>EV SAHİBİ</div><div class='stat-val' style='color:#3b82f6'>%{res['1x2'][0]:.1f}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-card'><div class='stat-lbl'>BERABERLİK</div><div class='stat-val' style='color:#94a3b8'>%{res['1x2'][1]:.1f}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-card'><div class='stat-lbl'>DEPLASMAN</div><div class='stat-val' style='color:#ef4444'>%{res['1x2'][2]:.1f}</div></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Detaylı Analiz")
        tab1, tab2 = st.tabs(["Skor Matrisi", "En Olası Skorlar"])
        
        with tab1:
            fig = go.Figure(data=go.Heatmap(z=res['matrix'], colorscale='Magma'))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            for s, p in res['top_scores']:
                st.progress(p/100, text=f"Skor {s} - Olasılık: %{p:.1f}")

if __name__ == "__main__":
    main()
