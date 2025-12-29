import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS (PROFESYONEL GÖRÜNÜM)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Intelligence",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed" # Yan menüyü kapalı başlatıyoruz, odak merkezde
)

st.markdown("""
    <style>
    /* GENEL ARKA PLAN */
    .stApp {background-color: #0b0f19;}
    
    /* QUANTUM BAŞLIK */
    .quantum-header {
        text-align: center;
        font-family: 'Courier New', monospace;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        margin-bottom: 30px;
    }
    
    /* KUPON KARTI (TICKET) */
    .ticket-container {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-top: 3px solid #00ff88;
        border-radius: 12px;
        padding: 25px;
        margin: 20px auto;
        max-width: 650px;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.1);
        position: relative;
    }
    
    .ticket-match {
        font-size: 1.5rem; font-weight: bold; color: white; text-align: center; margin-bottom: 10px;
    }
    
    .prediction-box {
        background: rgba(0, 255, 136, 0.05);
        border: 1px solid rgba(0, 255, 136, 0.2);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 15px 0;
    }
    
    .main-pred {
        font-size: 2.5rem; font-weight: 900; color: #facc15; 
        text-shadow: 0 0 15px rgba(250, 204, 21, 0.3);
        letter-spacing: 2px;
    }
    
    .sub-stats {
        display: flex; justify-content: space-around;
        color: #94a3b8; font-family: monospace; font-size: 0.9rem;
        margin-top: 10px;
    }
    
    .confidence-badge {
        background-color: #00ff88; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold;
    }
    
    /* İSTATİSTİK KARTLARI */
    .stat-card {
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #30363d;
        text-align: center;
        height: 100%;
    }
    
    .green-text {color: #4ade80;} .red-text {color: #f87171;} .yellow-text {color: #fbbf24;}
    
    /* YÜKLEME EKRANI */
    .stSpinner > div { border-top-color: #00ff88 !important; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SABİTLER
# -----------------------------------------------------------------------------
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 3. VERİ ÇEKME MOTORU
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(league_code):
    # TR1 (Süper Lig) ÖZEL SCRAPER
    if league_code == 'TR1':
        try:
            url = "https://www.tff.org/default.aspx?pageID=198"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers)
            if r.status_code != 200: return None
            
            try: tables = pd.read_html(r.content)
            except: return None
            if not tables: return None

            df = tables[0]
            if "Takım" not in df.columns:
                df.columns = df.iloc[0]
                df = df[1:]
            
            standings = []
            for idx, row in df.iterrows():
                try:
                    raw_team = str(row.get('Takım', 'Bilinmiyor'))
                    parts = raw_team.split(" ")
                    if parts[0].replace('.', '').isdigit(): parts = parts[1:]
                    team_name = " ".join(parts).replace("A.Ş.", "").strip()
                    
                    # MOMENTUM İÇİN RASTGELE AMA TUTARLI FORM ÜRETİMİ
                    # (TFF form verisi vermediği için puan durumuna göre üretiyoruz)
                    points = int(row.get('P', 0))
                    rank = idx + 1
                    if rank <= 3: base_form = ['W','W','D','W']
                    elif rank <= 8: base_form = ['W','D','L','W']
                    elif rank >= 16: base_form = ['L','L','D','L']
                    else: base_form = ['D','L','W','D']
                    
                    # Son maçı rastgele ekle ki grafik canlı dursun
                    import random
                    base_form.append(random.choice(['W','D','L']))
                    form_str = ",".join(base_form)

                    standings.append({
                        "team": {"name": team_name},
                        "playedGames": int(row.get('O', 0)),
                        "form": form_str, 
                        "goalsFor": int(row.get('A', 0)),
                        "goalsAgainst": int(row.get('Y', 0)),
                        "points": points,
                        "position": rank
                    })
                except: continue
            
            # Sanal Fikstür
            matches = []
            if len(standings) > 0:
                top = [t['team']['name'] for t in standings[:10]]
                import itertools
                # İlk 10 takımdan rastgele 5 maç oluştur (Demo için)
                for i in range(0, 10, 2):
                    if i+1 < len(top):
                        matches.append({"homeTeam": {"name": top[i]}, "awayTeam": {"name": top[i+1]}, "utcDate": datetime.now().isoformat()})

            return {"standings": {"standings": [{"table": standings}]}, "matches": {"matches": matches}, "scorers": {"scorers": []}}
        except: return None

    # GLOBAL API
    try:
        data = {}
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        data['standings'] = r1.json() if r1.status_code == 200 else None
        
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        return data
    except: return None

# -----------------------------------------------------------------------------
# 4. QUANTUM SİMÜLASYON MOTORU (TUTARLI VERSİYON)
# -----------------------------------------------------------------------------
def get_momentum_data(form_str):
    """ Grafiğin düz çizgi olmasını engeller """
    if not form_str: 
        # Veri yoksa rastgele dalgalanma yarat
        return np.random.randint(-1, 4, 5).cumsum().tolist()
        
    form_str = form_str.replace(',', '')
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    
    points = [0]
    current_val = 0
    for char in last_5:
        if char == 'W': current_val += 3
        elif char == 'D': current_val += 1
        elif char == 'L': current_val -= 2 # Kaybedince düşüş sert olsun
        points.append(current_val)
    return points

def simulate_consistent_match(home, away, stats, avg_goals):
    if home not in stats or away not in stats: return None
    h, a = stats[home], stats[away]
    
    # xG Hesaplama (Lig Ortalamasına Göre)
    # Form faktörünü biraz kıstık ki abartılı sonuçlar çıkmasın
    h_xg = h['att'] * a['def'] * avg_goals * 1.10 * (0.7 + (h['form_val']*0.3))
    a_xg = a['att'] * h['def'] * avg_goals * (0.7 + (a['form_val']*0.3))
    
    # QUANTUM SİMÜLASYONU (50.000 Maç)
    SIMS = 50000
    rng = np.random.default_rng()
    
    # Poisson ile gol sayılarını üret
    h_goals = rng.poisson(h_xg, SIMS)
    a_goals = rng.poisson(a_xg, SIMS)
    
    # 1. MAÇ SONUCU OLASILIKLARI
    home_wins = np.sum(h_goals > a_goals)
    draws = np.sum(h_goals == a_goals)
    away_wins = np.sum(h_goals < a_goals)
    
    prob_1 = (home_wins / SIMS) * 100
    prob_X = (draws / SIMS) * 100
    prob_2 = (away_wins / SIMS) * 100
    
    # 2. EN OLASI SKORU BUL (Tutarlılık İçin)
    # Skorları string yapıp sayıyoruz (Örn: "2-1")
    # Numpy ile hızlıca yapalım: h*100 + a (Örn: 201 = 2-1)
    score_hashes = h_goals * 100 + a_goals
    unique, counts = np.unique(score_hashes, return_counts=True)
    best_score_idx = np.argmax(counts)
    best_score_hash = unique[best_score_idx]
    
    pred_h_score = best_score_hash // 100
    pred_a_score = best_score_hash % 100
    exact_score_str = f"{pred_h_score}-{pred_a_score}"
    
    # 3. İY/MS MANTIĞINI SKORA GÖRE KUR (Çelişkiyi Önle)
    # Eğer skor 2-0 ise, mantıken İY de muhtemelen 1-0 veya 0-0'dır.
    # Simülasyon yerine, en olası skora en uygun İY senaryosunu seçiyoruz.
    if pred_h_score > pred_a_score:
        ht_ft_pred = "1 / 1" # Ev alıyorsa genelde 1/1 biter
    elif pred_a_score > pred_h_score:
        ht_ft_pred = "2 / 2" # Deplasman alıyorsa 2/2
    else:
        ht_ft_pred = "X / X" # Berabereyse X/X
        
    # Eğer skor 0-0 ise HT kesin X'tir.
    if pred_h_score == 0 and pred_a_score == 0:
        ht_ft_pred = "X / X"

    # 4. GÜVEN SKORU VE TAHMİN
    max_prob = max(prob_1, prob_X, prob_2)
    
    prediction_text = ""
    if prob_1 == max_prob: prediction_text = f"{home} KAZANIR"
    elif prob_2 == max_prob: prediction_text = f"{away} KAZANIR"
    else: prediction_text = "MAÇ BERABERE"

    # KG VAR / YOK
    btts_prob = (np.sum((h_goals > 0) & (a_goals > 0)) / SIMS) * 100
    o25_prob = (np.sum((h_goals + a_goals) > 2.5) / SIMS) * 100

    return {
        'probs': {'1': prob_1, 'X': prob_X, '2': prob_2},
        'score': exact_score_str,
        'ht_ft': ht_ft_pred,
        'main_pred': prediction_text,
        'conf': max_prob,
        'goals': {'btts': btts_prob, 'o25': o25_prob},
        'stats': {'h': h, 'a': a} # Radar için
    }

def create_radar(h, h_stats, a, a_stats):
    categories = ['Hücum', 'Savunma', 'Form', 'Gol Pot.', 'İstikrar']
    h_vals = [
        min(h_stats['att']*55, 100), min((3.5-h_stats['def'])*30, 100),
        min(h_stats['form_val']*85, 100), min(h_stats['att']*45 + h_stats['form_val']*20, 100),
        min(h_stats['form_val']*95, 100)
    ]
    a_vals = [
        min(a_stats['att']*55, 100), min((3.5-a_stats['def'])*30, 100),
        min(a_stats['form_val']*85, 100), min(a_stats['att']*45 + a_stats['form_val']*20, 100),
        min(a_stats['form_val']*95, 100)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories, fill='toself', name=h, line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a, line_color='#facc15')) # Sarı
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

# -----------------------------------------------------------------------------
# 5. MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.markdown("<div class='quantum-header'><h1>⚛️ QUANTUM INTELLIGENCE v19</h1></div>", unsafe_allow_html=True)
    
    # LIG SEÇİMİ (Sidebar değil, üstte temiz bir selectbox)
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        league_name = st.selectbox("LİG SEÇİNİZ", list(LEAGUES.keys()))
    
    league_code = LEAGUES[league_name]
    
    with st.spinner("Quantum veri tabanına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'):
        st.error("Veri alınamadı veya lig tatilde.")
        return

    # İSTATİSTİK HAZIRLIĞI
    stats = {}
    avg_goals = 1.5
    if data['standings']:
        table = data['standings']['standings'][0]['table']
        tg = sum(t['goalsFor'] for t in table); tp = sum(t['playedGames'] for t in table)
        avg_goals = tg/tp if tp>0 else 1.5
        for t in table:
            name = t['team']['name']; played = t['playedGames']
            raw_form = t.get('form', 'D,L,D,L,D')
            form_val = 1.0
            if raw_form:
                score = sum({'W':1.1, 'D':1.0, 'L':0.9}.get(c, 1.0) for c in raw_form.replace(',',''))
                form_val = score/len(raw_form.replace(',',''))
            stats[name] = {'att': (t['goalsFor']/played)/avg_goals if played>0 else 1, 'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1, 'form_val': form_val, 'form_str': raw_form}

    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    if not matches: st.warning("Maç bulunamadı."); return

    with col_sel2:
        selected = st.selectbox("MAÇI SEÇİN", list(matches.keys()))

    if st.button("🚀 SİMÜLASYONU BAŞLAT (50.000 Maç)", use_container_width=True):
        m_data = matches[selected]
        h_name, a_name = m_data['homeTeam']['name'], m_data['awayTeam']['name']
        
        # PROGRES BAR EFEKTİ
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        my_bar.empty()
        
        res = simulate_consistent_match(h_name, a_name, stats, avg_goals)
        
        if res:
            # --- 1. ANA KUPON KARTI ---
            st.markdown(f"""
            <div class="ticket-container">
                <div class="ticket-match">{h_name} vs {a_name}</div>
                <div class="prediction-box">
                    <div style="color:#94a3b8; font-size:0.9rem;">QUANTUM ANA TAHMİNİ</div>
                    <div class="main-pred">{res['main_pred']}</div>
                </div>
                <div class="sub-stats">
                    <div>GÜVEN: <span class="confidence-badge">%{res['conf']:.1f}</span></div>
                    <div>SKOR: <span style="color:#fff; font-weight:bold;">{res['score']}</span></div>
                    <div>İY/MS: <span style="color:#fff; font-weight:bold;">{res['ht_ft']}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 2. GRAFİKLER ---
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🕸️ Güç Dengesi")
                st.plotly_chart(create_radar(h_name, stats[h_name], a_name, stats[a_name]), use_container_width=True)
            
            with c2:
                st.markdown("### 📈 Momentum (Son 5 Maç)")
                h_mom = get_momentum_data(stats[h_name]['form_str'])
                a_mom = get_momentum_data(stats[a_name]['form_str'])
                # Grafiğin düz çizgi olmaması için garantiye aldık
                chart_data = pd.DataFrame({h_name: h_mom, a_name: a_mom})
                st.line_chart(chart_data, color=["#00ff88", "#facc15"])
            
            # --- 3. DETAYLI İSTATİSTİKLER ---
            st.markdown("### 📊 Simülasyon Verileri")
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='stat-card'><h3 class='green-text'>%{res['probs']['1']:.1f}</h3><p>Ev Sahibi</p></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='stat-card'><h3 class='yellow-text'>%{res['probs']['X']:.1f}</h3><p>Beraberlik</p></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='stat-card'><h3 class='red-text'>%{res['probs']['2']:.1f}</h3><p>Deplasman</p></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='stat-card'><h3>%{res['goals']['o25']:.1f}</h3><p>2.5 Üst</p></div>", unsafe_allow_html=True)
            
        else:
            st.error("Analiz için yeterli veri yok.")

if __name__ == "__main__":
    main()
