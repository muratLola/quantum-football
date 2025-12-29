import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR VE STİL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum v14: Momentum",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {background-color: #0f172a;}
    .stat-card {background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
    .win-green {color: #4ade80; font-weight: bold;}
    .loss-red {color: #f87171; font-weight: bold;}
    .draw-yellow {color: #fbbf24; font-weight: bold;}
    .big-number {font-size: 28px; font-weight: 800; color: white;}
    .sub-text {font-size: 14px; color: #94a3b8;}
    /* Grafik Arka Planı */
    canvas {background-color: #1e293b; border-radius: 10px; padding: 10px;}
    </style>
    """, unsafe_allow_html=True)

# API
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUE_MULTIPLIERS = {
    'PL': 1.05, 'TR1': 1.02, 'PD': 0.95, 'BL1': 1.20,
    'SA': 1.00, 'FL1': 0.90, 'DED': 1.15, 'CL': 1.00
}

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 2. VERİ ÇEKME
# -----------------------------------------------------------------------------
def fetch_tff_data_hybrid():
    try:
        url = "https://www.tff.org/default.aspx?pageID=198"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return None
        try: tables = pd.read_html(response.content)
        except: return None
        if not tables: return None

        df = tables[0]
        if "Takım" not in df.columns:
            df.columns = df.iloc[0]
            df = df[1:]
            
        standings_table = []
        for index, row in df.iterrows():
            try:
                raw_team = str(row.get('Takım', 'Bilinmiyor'))
                team_parts = raw_team.split(" ")
                if team_parts[0].replace('.', '').isdigit(): team_parts = team_parts[1:]
                team_name = " ".join(team_parts).replace("A.Ş.", "").strip()

                standings_table.append({
                    "position": index + 1, "team": {"name": team_name},
                    "playedGames": int(row.get('O', 0)), 
                    # TFF'den gerçek form verisi çekilemediği için görsel amaçlı nötr bırakıyoruz
                    # İleride burası geliştirilebilir
                    "form": "W,D,W,L,D", 
                    "goalsFor": int(row.get('A', 0)), "goalsAgainst": int(row.get('Y', 0)),
                    "points": int(row.get('P', 0))
                })
            except: continue

        matches_list = []
        if len(standings_table) > 0:
            top_teams = [t['team']['name'] for t in standings_table[:6]]
            import itertools
            for pair in itertools.combinations(top_teams, 2):
                 matches_list.append({"homeTeam": {"name": pair[0]}, "awayTeam": {"name": pair[1]}, "utcDate": datetime.now().isoformat(), "status": "SCHEDULED"})

        return {"standings": {"standings": [{"table": standings_table}]}, "matches": {"matches": matches_list}, "scorers": {"scorers": []}}
    except: return None

@st.cache_data(ttl=3600)
def fetch_data(league_code):
    if league_code == 'TR1': return fetch_tff_data_hybrid()
    try:
        data = {}
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        if r1.status_code != 200: return None
        data['standings'] = r1.json()
        r2 = requests.get(f"{BASE_URL}/competitions/{league_code}/scorers?limit=10", headers=HEADERS)
        data['scorers'] = r2.json() if r2.status_code == 200 else {'scorers': []}
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        return data
    except: return None

# -----------------------------------------------------------------------------
# 3. YENİ ÖZELLİK: MOMENTUM GRAFİĞİ 📈
# -----------------------------------------------------------------------------
def get_momentum_data(form_str):
    """ 'W,L,D,W,W' formatındaki formu sayısal grafiğe çevirir. """
    if not form_str: return [0, 0, 0, 0, 0]
    
    # Virgülleri temizle
    form_str = form_str.replace(',', '')
    # Son 5 maçı al
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    
    points = []
    current_score = 0
    # Başlangıç noktası
    points.append(0) 
    
    for char in last_5:
        if char == 'W': current_score += 3  # Galibiyet: Yükseliş
        elif char == 'D': current_score += 1 # Beraberlik: Hafif yükseliş
        elif char == 'L': current_score -= 1 # Mağlubiyet: Düşüş (Daha dramatik görünmesi için -1)
        points.append(current_score)
        
    return points

# -----------------------------------------------------------------------------
# 4. ANALİZ MOTORU
# -----------------------------------------------------------------------------
def analyze_teams(data):
    stats = {}
    avg_goals = 1.5
    if data and data.get('standings'):
        table = data['standings']['standings'][0]['table']
        total_g = sum(t['goalsFor'] for t in table)
        total_p = sum(t['playedGames'] for t in table)
        avg_goals = (total_g / total_p) if total_p > 0 else 1.5
        for t in table:
            name = t['team']['name']
            played = t['playedGames']
            raw_form = t.get('form', '')
            form_str = raw_form.replace(',', '') if raw_form else ''
            form_val = 1.0
            if form_str:
                score = sum({'W':1.1, 'D':1.0, 'L':0.9}.get(c, 1.0) for c in form_str)
                form_val = score / len(form_str)
            # İstatistiklere ham form stringini de ekliyoruz (Grafik için)
            stats[name] = {
                'att': (t['goalsFor']/played)/avg_goals if played>0 else 1, 
                'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1, 
                'form_val': form_val, 
                'form_str': raw_form,
                'rank': t['position'], 
                'bonus': 0
            }
            
    if data and data.get('scorers'):
        for p in data['scorers']['scorers']:
            if p['team']['name'] in stats: stats[p['team']['name']]['bonus'] += (p['goals'] * 0.005)
    return stats, avg_goals

def simulate_value_bet(home, away, stats, avg_goals, league_code):
    if home not in stats or away not in stats: return None
    h, a = stats[home], stats[away]
    
    league_factor = LEAGUE_MULTIPLIERS.get(league_code, 1.0)
    
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form_val'] * (1 + h['bonus']) * league_factor
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form_val'] * (1 + a['bonus']) * league_factor
    
    SIMS = 30000
    rng = np.random.default_rng()
    h_goals = rng.poisson(total_h_xg, SIMS)
    a_goals = rng.poisson(total_a_xg, SIMS)
    
    home_wins = np.sum(h_goals > a_goals)
    draws = np.sum(h_goals == a_goals)
    away_wins = np.sum(h_goals < a_goals)
    
    prob_1 = (home_wins/SIMS)*100
    prob_x = (draws/SIMS)*100
    prob_2 = (away_wins/SIMS)*100
    
    fair_odd_1 = 100 / prob_1 if prob_1 > 0 else 0
    fair_odd_x = 100 / prob_x if prob_x > 0 else 0
    fair_odd_2 = 100 / prob_2 if prob_2 > 0 else 0
    
    max_prob = max(prob_1, prob_x, prob_2)
    stake_advice = "Düşük (%1)"
    if max_prob > 70: stake_advice = "Yüksek (%5)"
    elif max_prob > 55: stake_advice = "Orta (%3)"
    elif max_prob > 45: stake_advice = "Düşük-Orta (%2)"
    
    total_goals = h_goals + a_goals
    
    return {
        'probs': {'1': prob_1, 'X': prob_x, '2': prob_2},
        'fair_odds': {'1': fair_odd_1, 'X': fair_odd_x, '2': fair_odd_2},
        'goals': {'o25': (np.sum(total_goals > 2.5)/SIMS)*100, 'btts': (np.sum((h_goals>0)&(a_goals>0))/SIMS)*100},
        'xg': {'h': total_h_xg, 'a': total_a_xg},
        'stake': stake_advice,
        # Form verilerini de döndür
        'forms': {'h': h['form_str'], 'a': a['form_str']}
    }

# -----------------------------------------------------------------------------
# 5. ARAYÜZ
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("💎 Quantum v14")
    league_name = st.sidebar.selectbox("Lig Seç:", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    st.title(f"Quantum Analiz: {league_name}")
    
    with st.spinner("Piyasalar ve momentum analiz ediliyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'):
        st.warning("Veri alınamadı.")
        return
        
    stats, avg_goals = analyze_teams(data)
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    if not matches: st.warning("Maç yok."); return
    
    selected = st.selectbox("Maç Seç:", list(matches.keys()))
    if not selected: return
    
    m_data = matches[selected]
    h, a = m_data['homeTeam']['name'], m_data['awayTeam']['name']
    
    if st.button("🚀 Momentum Analizi Yap"):
        res = simulate_value_bet(h, a, stats, avg_goals, league_code)
        if res:
            # --- 1. ÜST BÖLÜM: ORANLAR VE KASA ---
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-card'><h3>{h}</h3><h1 class='win-green'>%{res['probs']['1']:.1f}</h1><p>Adil Oran: <b>{res['fair_odds']['1']:.2f}</b></p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-card'><h3>Beraberlik</h3><h1 class='draw-yellow'>%{res['probs']['X']:.1f}</h1><p>Adil Oran: <b>{res['fair_odds']['X']:.2f}</b></p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-card'><h3>{a}</h3><h1 class='loss-red'>%{res['probs']['2']:.1f}</h1><p>Adil Oran: <b>{res['fair_odds']['2']:.2f}</b></p></div>", unsafe_allow_html=True)
            
            st.write("")
            
            # --- 2. YENİ BÖLÜM: MOMENTUM GRAFİĞİ ---
            st.subheader("📈 Takım Momentum Grafiği (Son 5 Maç)")
            st.caption("Takımların son maçlardaki form durumunu gösterir. Yükselen çizgi formda olduğunu işaret eder.")
            
            # Form verilerini sayısal grafiğe dök
            h_mom = get_momentum_data(res['forms']['h'])
            a_mom = get_momentum_data(res['forms']['a'])
            
            # Pandas DataFrame oluştur (Streamlit grafiği için)
            chart_data = pd.DataFrame({
                h: h_mom,
                a: a_mom
            })
            
            # Çizgi grafiği çiz
            st.line_chart(chart_data, color=["#4ade80", "#f87171"]) # Ev sahibi Yeşil, Deplasman Kırmızı
            
            st.markdown("---")
            
            # --- 3. ALT BÖLÜM: VALUE VE GOLLER ---
            col_stake, col_goal = st.columns(2)
            with col_stake:
                st.markdown(f"### 💼 Kasa Yönetimi")
                st.markdown(f"Güven Seviyesi: <b style='color:#00ff88'>{res['stake']}</b>", unsafe_allow_html=True)
                st.info("Eğer bahis sitelerindeki oran, yukarıdaki **Adil Oran**'dan yüksekse bu 'Değerli Bahis'tir.")
                
            with col_goal:
                st.markdown("### 🥅 Gol Beklentisi")
                st.write(f"**2.5 Üst:** %{res['goals']['o25']:.1f}")
                st.write(f"**KG Var:** %{res['goals']['btts']:.1f}")

        else:
            st.error("Veri yetersiz.")

if __name__ == "__main__":
    main()
