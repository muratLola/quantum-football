import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR VE STİL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum v15: The Ticket",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {background-color: #0f172a;}
    .stat-card {background-color: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
    
    /* --- YENİ KUPON KARTI TASARIMI (TICKET CSS) --- */
    .ticket-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 20px;
        position: relative;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
    }
    .ticket-header {
        border-bottom: 2px dashed #334155;
        padding-bottom: 10px;
        margin-bottom: 15px;
        text-align: center;
        color: #00ff88;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        letter-spacing: 2px;
    }
    .ticket-body {
        text-align: center;
        color: white;
    }
    .ticket-match {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .ticket-prediction {
        font-size: 2rem;
        font-weight: 900;
        color: #facc15; /* Sarı */
        margin: 10px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .ticket-confidence {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .ticket-footer {
        margin-top: 15px;
        border-top: 2px dashed #334155;
        padding-top: 10px;
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
        font-family: 'Courier New', monospace;
    }
    .barcode {
        font-family: 'Libre Barcode 39 Text', cursive; /* Google Font eklenebilir ama şimdilik temsili */
        font-size: 2rem;
        opacity: 0.5;
        letter-spacing: 5px;
    }
    
    /* Renkler */
    .win-green {color: #4ade80; font-weight: bold;}
    .loss-red {color: #f87171; font-weight: bold;}
    .draw-yellow {color: #fbbf24; font-weight: bold;}
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
# 3. ANALİZ MOTORU & MOMENTUM
# -----------------------------------------------------------------------------
def get_momentum_data(form_str):
    if not form_str: return [0, 0, 0, 0, 0]
    form_str = form_str.replace(',', '')
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    points = [0]
    current = 0
    for char in last_5:
        if char == 'W': current += 3
        elif char == 'D': current += 1
        elif char == 'L': current -= 1
        points.append(current)
    return points

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
    
    # KUPON İÇİN ANA TAHMİNİ BELİRLE
    main_prediction = "BELİRSİZ"
    if max_prob == prob_1: main_prediction = f"{home} KAZANIR (MS 1)"
    elif max_prob == prob_2: main_prediction = f"{away} KAZANIR (MS 2)"
    else: main_prediction = "BERABERLİK (MS 0)"
    
    # Alternatif (Güvenli) Tahmin
    alt_prediction = "YOK"
    if (np.sum(total_goals > 1.5)/SIMS)*100 > 75: alt_prediction = "1.5 ÜST"
    elif (np.sum((h_goals>0)&(a_goals>0))/SIMS)*100 > 60: alt_prediction = "KG VAR"
    
    return {
        'probs': {'1': prob_1, 'X': prob_x, '2': prob_2},
        'fair_odds': {'1': fair_odd_1, 'X': fair_odd_x, '2': fair_odd_2},
        'goals': {'o25': (np.sum(total_goals > 2.5)/SIMS)*100, 'btts': (np.sum((h_goals>0)&(a_goals>0))/SIMS)*100},
        'stake': stake_advice,
        'forms': {'h': h['form_str'], 'a': a['form_str']},
        'ticket': {'main': main_prediction, 'alt': alt_prediction, 'conf': max_prob}
    }

# -----------------------------------------------------------------------------
# 5. ARAYÜZ (MAIN)
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("💎 Quantum v15")
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
    
    if st.button("🎫 Kuponu Oluştur"):
        res = simulate_value_bet(h, a, stats, avg_goals, league_code)
        if res:
            # --- 1. KUPON KARTI (YENİ ÖZELLİK) ---
            st.markdown(f"""
            <div class="ticket-container">
                <div class="ticket-header">QUANTUM INTELLIGENCE TICKET</div>
                <div class="ticket-body">
                    <div class="ticket-match">{h} vs {a}</div>
                    <div class="ticket-confidence">ÖNERİLEN TAHMİN</div>
                    <div class="ticket-prediction">{res['ticket']['main']}</div>
                    
                    <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                        <div>
                            <div class="ticket-confidence">GÜVEN</div>
                            <div style="color: #00ff88; font-weight: bold;">%{res['ticket']['conf']:.1f}</div>
                        </div>
                        <div>
                            <div class="ticket-confidence">ADİL ORAN</div>
                            <div style="color: #00ff88; font-weight: bold;">{res['fair_odds']['1'] if 'MS 1' in res['ticket']['main'] else res['fair_odds']['2'] if 'MS 2' in res['ticket']['main'] else res['fair_odds']['X']:.2f}</div>
                        </div>
                        <div>
                            <div class="ticket-confidence">ALTERNATİF</div>
                            <div style="color: #fbbf24; font-weight: bold;">{res['ticket']['alt']}</div>
                        </div>
                    </div>
                </div>
                <div class="ticket-footer">
                    <div class="barcode">||| || ||| | |||| |||</div>
                    <div>{datetime.now().strftime("%d.%m.%Y • %H:%M")}</div>
                    <div>ID: {str(int(res['ticket']['conf']*12345))}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown("---")

            # --- 2. DETAYLI VERİLER ---
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{h}", f"%{res['probs']['1']:.1f}")
            c2.metric("Beraberlik", f"%{res['probs']['X']:.1f}")
            c3.metric(f"{a}", f"%{res['probs']['2']:.1f}")
            
            st.subheader("📈 Momentum Grafiği")
            h_mom = get_momentum_data(res['forms']['h'])
            a_mom = get_momentum_data(res['forms']['a'])
            st.line_chart(pd.DataFrame({h: h_mom, a: a_mom}), color=["#4ade80", "#f87171"])
            
        else:
            st.error("Veri yetersiz.")

if __name__ == "__main__":
    main()
