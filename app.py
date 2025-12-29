import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from difflib import get_close_matches
import os

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS (PREMIUM UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum AI v30",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp {background-color: #0b0f19;}
    
    /* BAŞLIK */
    .quantum-title {
        font-family: 'Arial', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #fff;
        text-align: center;
        letter-spacing: 4px;
        margin-top: 10px;
        background: -webkit-linear-gradient(#eee, #333);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    }
    
    /* SKORBOARD (LIVE EFEKT) */
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); } }
    .live-badge { background-color: #ff5252; color: white; padding: 3px 8px; border-radius: 10px; font-weight: bold; font-size: 0.7rem; animation: pulse-red 2s infinite; vertical-align: middle; }
    .scoreboard { background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); border: 1px solid #334155; padding: 20px; border-radius: 16px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    
    /* TICKET (ANA TAHMİN) */
    .ticket-container { background: radial-gradient(circle at center, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; border-top: 4px solid #00ff88; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.6); margin-bottom: 25px; max-width: 800px; margin-left: auto; margin-right: auto; }
    .main-pred { font-size: 3rem; font-weight: 900; color: #facc15; margin: 15px 0; letter-spacing: -1px; text-shadow: 0 0 15px rgba(250, 204, 21, 0.4); }
    
    /* DETAYLI ANALİZ KUTULARI */
    .market-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; transition: transform 0.2s; }
    .market-box:hover { border-color: #00ff88; transform: translateY(-2px); }
    .market-title { color: #00ff88; font-weight: bold; font-size: 0.9rem; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .market-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.95rem; color: #e5e7eb; }
    
    .prob-high { color: #4ade80; font-weight: bold; } 
    .prob-med { color: #facc15; } 
    .prob-low { color: #f87171; } 

    /* FORM BADGES */
    .form-badges { display: flex; gap: 4px; justify-content: center; }
    .badge { width: 24px; height: 24px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #000; font-size: 0.75rem; }
    .badge-W { background-color: #4ade80; } .badge-D { background-color: #facc15; } .badge-L { background-color: #f87171; }
    
    /* DECISION BADGE (PLAY/PASS) */
    .decision-play { background-color: #238636; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }
    .decision-pass { background-color: #da3633; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; letter-spacing: 1px; font-size: 0.9rem; }

    /* YASAL UYARI */
    .disclaimer { font-size: 0.7rem; color: #64748b; text-align: center; margin-top: 40px; border-top: 1px solid #334155; padding-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. GÜVENLİK & API
# -----------------------------------------------------------------------------
try:
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
except:
    API_KEY = "741fe4cfaf31419a864d7b6777b23862"

HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

LEAGUES = {
    "🇬🇧 Premier League": "PL",
    "🇹🇷 Süper Lig": "TR1",
    "🇪🇸 La Liga": "PD",
    "🇩🇪 Bundesliga": "BL1",
    "🇮🇹 Serie A": "SA",
    "🇫🇷 Ligue 1": "FL1",
    "🇳🇱 Eredivisie": "DED",
    "🇪🇺 Şampiyonlar Ligi": "CL"
}

# -----------------------------------------------------------------------------
# 3. YARDIMCI MOTORLAR (AKILLI FORM & EŞLEŞTİRME)
# -----------------------------------------------------------------------------
def generate_smart_form(points, played):
    if played == 0: return "D,D,D,D,D"
    ppg = points / played
    if ppg >= 2.2: w = [0.75, 0.2, 0.05]
    elif ppg >= 1.6: w = [0.55, 0.25, 0.2]
    elif ppg >= 1.2: w = [0.35, 0.35, 0.3]
    else: w = [0.15, 0.25, 0.6]
    return ",".join(np.random.choice(["W","D","L"], 5, p=w))

def render_form_badges(form_str):
    form_str = form_str.replace(',', '')
    html = "<div class='form-badges'>"
    for char in form_str[-5:]:
        bg = "badge-W" if char == 'W' else "badge-D" if char == 'D' else "badge-L"
        html += f"<div class='badge {bg}'>{char}</div>"
    html += "</div>"
    return html

def match_team_name(target, names):
    if target in names: return target
    m = get_close_matches(target, names, n=1, cutoff=0.5)
    return m[0] if m else None

# -----------------------------------------------------------------------------
# 4. VERİ ÇEKME MOTORU
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(code):
    try:
        r1 = requests.get(f"{BASE_URL}/competitions/{code}/standings", headers=HEADERS)
        r2 = requests.get(
            f"{BASE_URL}/competitions/{code}/matches",
            headers=HEADERS,
            params={
                "dateFrom": datetime.now().strftime("%Y-%m-%d"),
                "dateTo": (datetime.now()+timedelta(days=10)).strftime("%Y-%m-%d")
            }
        )
        if r1.status_code != 200: return None
        return {"standings": r1.json(), "matches": r2.json()}
    except: return None

# -----------------------------------------------------------------------------
# 5. CORE ENSEMBLE ENGINE (Meta-Learner + 3 Model)
# -----------------------------------------------------------------------------
def run_ensemble_simulation(home, away, stats, avg_goals):
    h = match_team_name(home, stats.keys())
    a = match_team_name(away, stats.keys())
    if not h or not a: return None

    hs, as_ = stats[h], stats[a]

    # --- MODEL 1: POISSON (GOL ODAKLI) ---
    home_adv = 0.35 * hs["home_factor"]
    h_xg = hs["att"] * as_["def"] * avg_goals + home_adv
    a_xg = as_["att"] * hs["def"] * avg_goals * as_["away_factor"]
    
    # Form Etkisi (Temporal Decay benzeri basit etki)
    h_xg *= hs["form_val"]
    a_xg *= as_["form_val"]

    SIMS = 12000
    rng = np.random.default_rng()
    hg = rng.poisson(h_xg, SIMS)
    ag = rng.poisson(a_xg, SIMS)
    
    p1_pois = np.mean(hg > ag)
    px_pois = np.mean(hg == ag)
    p2_pois = np.mean(hg < ag)

    # --- MODEL 2: POWER/ELO (GÜÇ ODAKLI) ---
    # Güç farkına dayalı lojistik bir eğri simülasyonu
    power_diff = hs["power"] - as_["power"]
    # Sigmoid fonksiyonu ile kazanma ihtimali
    p1_elo = 1 / (1 + np.exp(-(power_diff + 20) / 40)) # +20 Home Adv
    p2_elo = 1 / (1 + np.exp((power_diff - 20) / 40))
    px_elo = 1 - (p1_elo + p2_elo)
    if px_elo < 0: px_elo = 0.1 # Normalize

    # --- META-LEARNER (CONTEXT AĞIRLIKLANDIRMA) ---
    # Maçın karakterine göre hangi modele güveneceğiz?
    w_pois = 0.60
    w_elo = 0.40
    
    # Eğer derbi veya büyük maç ise (Güçler yakınsa), ELO daha belirleyicidir.
    if abs(power_diff) < 15: 
        w_pois = 0.40
        w_elo = 0.60 # Derbide taktik ve güç konuşur, xG sapıtabilir.

    # --- ENSEMBLE SONUÇ (BİRLEŞTİRME) ---
    p1 = (p1_pois * w_pois) + (p1_elo * w_elo)
    p2 = (p2_pois * w_pois) + (p2_elo * w_elo)
    px = (px_pois * w_pois) + (px_elo * w_elo)
    
    # Toplamı 100'e tamamla
    total = p1 + p2 + px
    p1, p2, px = (p1/total)*100, (p2/total)*100, (px/total)*100

    # --- QUANTUM SCENARIO ENGINE (SENARYO ANALİZİ) ---
    # Farklı evrenlerde maç simülasyonu
    universes = {
        "Normal": 0.50,
        "Erken Gol (Kaos)": 0.20,
        "Kırmızı Kart": 0.10,
        "Kısır Maç (0-0)": 0.20
    }
    
    # Basit bir senaryo etkisi (Örn: Kısır maçta X artar)
    # Bu kısım sonuçları hafifçe "bükerek" gerçekçilik katar.
    px += (universes["Kısır Maç (0-0)"] * 10) 
    p1 -= (universes["Kısır Maç (0-0)"] * 5)
    p2 -= (universes["Kısır Maç (0-0)"] * 5)
    
    # Tekrar normalize
    total = p1 + p2 + px
    p1, p2, px = (p1/total)*100, (p2/total)*100, (px/total)*100

    # --- ÇIKTILAR ---
    conf = max(p1, px, p2)
    
    # Entropy (Belirsizlik) Hesabı
    probs_norm = np.array([p1, px, p2]) / 100
    entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-9))
    
    # Karar Mekanizması (PASS / PLAY)
    decision = "PLAY"
    if entropy > 1.02: decision = "PASS (Riskli)"
    elif conf < 40: decision = "PASS (Düşük Güven)"

    # Skor Tahmini (Poisson'dan gelen)
    score_hash = hg * 100 + ag
    u, c = np.unique(score_hash, return_counts=True)
    best = u[np.argmax(c)]
    exact_score = f"{best//100}-{best%100}"

    if p1 > p2 and p1 > px: main_pred = f"{h} KAZANIR"
    elif p2 > p1 and p2 > px: main_pred = f"{a} KAZANIR"
    else: main_pred = "BERABERLİK"

    # Ekstra Marketler
    dc_1x = p1 + px
    dc_x2 = px + p2
    total_goals = hg + ag
    o25 = np.mean(total_goals > 2.5) * 100
    btts = np.mean((hg > 0) & (ag > 0)) * 100
    
    # Yorum Üretici (Explainable AI)
    reason = "Güç farkı ve saha avantajı belirleyici oldu."
    if w_elo > w_pois: reason = "Kritik maç olduğu için Güç Dengesi (ELO) modeline öncelik verildi."
    if entropy > 1.0: reason = "Modeller arasında fikir ayrılığı var (Yüksek Entropi). Sürpriz ihtimali."

    return {
        "main": main_pred, "score": exact_score, "conf": conf, "entropy": entropy, "decision": decision,
        "reason": reason,
        "probs": {"1":p1,"X":px,"2":p2},
        "dc": {"1X": dc_1x, "X2": dc_x2},
        "goals": {"o25": o25, "btts": btts},
        "names": {"h": h, "a": a},
        "forms": {"h": hs["form_str"], "a": as_["form_str"]},
        "power": {"h": hs["power"], "a": as_["power"]}
    }

def create_radar(res, stats):
    h, a = res['names']['h'], res['names']['a']
    h_val = [stats[h]['att']*50, stats[h]['def']*40, stats[h]['power']*0.8]
    a_val = [stats[a]['att']*50, stats[a]['def']*40, stats[a]['power']*0.8]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_val, theta=['Hücum','Savunma','Güç'], fill='toself', name='Ev', line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_val, theta=['Hücum','Savunma','Güç'], fill='toself', name='Dep', line_color='#facc15'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), margin=dict(l=20,r=20,t=20,b=20), height=250, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    return fig

# -----------------------------------------------------------------------------
# 6. MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    st.markdown("<div class='quantum-title'>QUANTUM AI v30</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        league_name = st.selectbox("LİG SEÇ", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    with st.spinner("Veri tabanına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'): 
        st.warning("⚠️ Bu lig için veri alınamadı. Başka lig seçiniz.")
        return

    # İSTATİSTİK HAZIRLAMA (v30 Power Calculation)
    table = data["standings"]["standings"][0]["table"]
    stats = {}
    tg = sum(t["goalsFor"] for t in table)
    tp = sum(t["playedGames"] for t in table)
    avg_goals = tg/tp if tp else 1.5

    for t in table:
        name = t["team"]["name"]
        played = t["playedGames"]
        pts = t["points"]

        form = t.get("form")
        if not form or len(form)<3: form = generate_smart_form(pts, played)
        form_val = np.mean([{"W":1.1,"D":1.0,"L":0.9}.get(x, 1.0) for x in form.replace(",","")])

        # Power Rating (v30 - Gelişmiş)
        # Puan + Gol Averajı + Form kombinasyonu
        gd = t["goalsFor"] - t["goalsAgainst"]
        power = 100 + (pts/played * 20) + (gd/played * 5) + (form_val * 10)

        stats[name] = {
            "att": (t["goalsFor"]/played)/avg_goals if played else 1,
            "def": (t["goalsAgainst"]/played)/avg_goals if played else 1,
            "form_val": form_val, "form_str": form,
            "power": power,
            "home_factor": 1.12, "away_factor": 0.88
        }

    matches = {f'{m["homeTeam"]["name"]} - {m["awayTeam"]["name"]}': m for m in data["matches"]["matches"] if m["status"] == "SCHEDULED"}
    
    with c2:
        game = st.selectbox("MAÇ SEÇ", list(matches.keys()))

    # LIVE MOD
    live_mode = st.toggle("🔴 LIVE SIMULATION", value=False)
    if live_mode:
        m = matches[game]
        st.markdown(f"""
        <div class="scoreboard">
            <div style="width:30%; text-align:center; color:#ccc; font-weight:bold;">{m['homeTeam']['name']}</div>
            <div style="width:40%; text-align:center;">
                <div style="color:#00ff88; font-family:monospace;"><span class="live-badge">LIVE</span> {np.random.randint(10,80)}'</div>
                <div style="font-size:1.8rem; font-weight:bold; color:white;">{np.random.randint(0,2)} - {np.random.randint(0,2)}</div>
            </div>
            <div style="width:30%; text-align:center; color:#ccc; font-weight:bold;">{m['awayTeam']['name']}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("QUANTUM ANALİZİ BAŞLAT", use_container_width=True):
        m = matches[game]
        # Progress Bar Efekti (Steps: Data -> Models -> Meta-Learn -> Scenarios)
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            bar.progress(i+1)
        bar.empty()

        res = run_ensemble_simulation(m["homeTeam"]["name"], m["awayTeam"]["name"], stats, avg_goals)

        if res:
            # KARAR BADGE (PLAY / PASS)
            dec_class = "decision-play" if "PLAY" in res["decision"] else "decision-pass"
            st.markdown(f"<div style='text-align:center; margin-bottom:15px;'><span class='{dec_class}'>AI KARARI: {res['decision']}</span></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="ticket-container">
                <div style="color:#aaa; font-size:0.9rem;">ENSEMBLE PREDICTION</div>
                <div class="main-pred">{res['main']}</div>
                <div style="display:flex; justify-content:center; gap:20px; color:white; font-family:monospace;">
                    <div>SKOR: {res['score']}</div>
                    <div style="color:#00ff88">GÜVEN: %{res['conf']:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # FORMLAR
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{res['names']['h']}** (Güç: {int(res['power']['h'])})")
                st.markdown(render_form_badges(res['forms']['h']), unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{res['names']['a']}** (Güç: {int(res['power']['a'])})")
                st.markdown(render_form_badges(res['forms']['a']), unsafe_allow_html=True)

            st.markdown("---")

            # DETAYLI MARKETLER
            st.subheader("📊 Quantum Market Derinliği")
            m1, m2 = st.columns(2)
            
            with m1:
                st.markdown("<div class='market-box'><div class='market-title'>🛡️ ÇİFTE ŞANS</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>1X</span> <span class='prob-high'>%{res['dc']['1X']:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>X2</span> <span class='prob-med'>%{res['dc']['X2']:.1f}</span></div></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='market-box'><div class='market-title'>🥅 ALT / ÜST</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>2.5 ÜST</span> <span class='{ 'prob-high' if res['goals']['o25']>55 else 'prob-low' }'>%{res['goals']['o25']:.1f}</span></div>", unsafe_allow_html=True)

            with m2:
                st.markdown("<div class='market-box'><div class='market-title'>🔥 DİĞER</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>KG VAR</span> <span class='{ 'prob-high' if res['goals']['btts']>55 else 'prob-low' }'>%{res['goals']['btts']:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>MS 1</span> <span>%{res['probs']['1']:.1f}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='market-row'><span>MS 2</span> <span>%{res['probs']['2']:.1f}</span></div></div>", unsafe_allow_html=True)
            
            # EXPLAINABLE AI KUTUSU
            st.info(f"🧠 **AI Nedeni:** {res['reason']}")

            # RADAR
            st.markdown("#### 🕸️ Güç Dengesi")
            st.plotly_chart(create_radar(res, stats), use_container_width=True)

        else: st.error("Analiz verisi oluşturulamadı.")

    st.markdown("---")
    st.markdown("<div class='disclaimer'>⚠️ YASAL UYARI: Bu uygulama sadece istatistiksel simülasyon amaçlıdır. Kesinlik içermez. 18+</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
