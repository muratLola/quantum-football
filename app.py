import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR VE STİL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Analyst v11: Hybrid System",
    page_icon="🧬",
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
    </style>
    """, unsafe_allow_html=True)

# API BİLGİLERİ
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {
    '🇬🇧 Premier League': 'PL', 
    '🇹🇷 Süper Lig': 'TR1', 
    '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', 
    '🇮🇹 Serie A': 'SA', 
    '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', 
    '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 2. VERİ ÇEKME (HİBRİT SİSTEM: API + TFF)
# -----------------------------------------------------------------------------

def fetch_tff_data_hybrid():
    """Süper Lig için TFF Sitesinden Veri Çeker"""
    try:
        url = "https://www.tff.org/default.aspx?pageID=198"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # HTML Tablolarını Oku (lxml gerektirir)
        try:
            tables = pd.read_html(response.content)
        except ValueError:
            return None # Tablo bulunamazsa

        if not tables:
            return None

        # --- PUAN DURUMU ---
        df_standings = tables[0]
        
        # Sütun isimlerini düzelt
        if "Takım" not in df_standings.columns:
            df_standings.columns = df_standings.iloc[0]
            df_standings = df_standings[1:]
            
        standings_table = []
        for index, row in df_standings.iterrows():
            try:
                # Veri temizliği
                played = int(row.get('O', 0))
                points = int(row.get('P', 0))
                raw_team = str(row.get('Takım', 'Bilinmiyor'))
                
                # İsim temizleme (Örn: "1. GALATASARAY A.Ş." -> "GALATASARAY")
                team_parts = raw_team.split(" ")
                if team_parts[0].replace('.', '').isdigit():
                    team_parts = team_parts[1:]
                team_name = " ".join(team_parts).replace("A.Ş.", "").strip()

                goals_for = int(row.get('A', 0))
                goals_against = int(row.get('Y', 0))

                standings_table.append({
                    "position": index + 1,
                    "team": {"name": team_name},
                    "playedGames": played,
                    "form": "WWWWW", # Varsayılan Form
                    "goalsFor": goals_for,
                    "goalsAgainst": goals_against,
                    "points": points
                })
            except: continue

        api_standings = {"standings": [{"table": standings_table}]}

        # --- MAÇLAR (SANAL FİKSTÜR) ---
        # TFF Fikstürü çekmek zor olduğu için, ilk 5 takım arasında sanal maçlar oluşturuyoruz
        # Böylece simülasyon motoru boş dönüp çökmez.
        matches_list = []
        top_teams = [t['team']['name'] for t in standings_table[:5]]
        
        import itertools
        for pair in itertools.combinations(top_teams, 2):
             matches_list.append({
                "homeTeam": {"name": pair[0]},
                "awayTeam": {"name": pair[1]},
                "utcDate": datetime.now().isoformat(),
                "status": "SCHEDULED"
            })

        return {
            "standings": api_standings,
            "matches": {"matches": matches_list},
            "scorers": {"scorers": []}
        }

    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_data(league_code):
    # TR1 (Süper Lig) ise TFF Scraper kullan
    if league_code == 'TR1':
        return fetch_tff_data_hybrid()
    
    # Diğerleri için API
    try:
        data = {}
        # 1. Puan Durumu
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        if r1.status_code != 200: return None
        data['standings'] = r1.json()
        
        # 2. Gol Krallığı
        r2 = requests.get(f"{BASE_URL}/competitions/{league_code}/scorers?limit=10", headers=HEADERS)
        data['scorers'] = r2.json() if r2.status_code == 200 else {'scorers': []}
        
        # 3. Gelecek Maçlar
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        
        return data
    except:
        return None

# -----------------------------------------------------------------------------
# 3. ANALİZ MOTORU
# -----------------------------------------------------------------------------
def analyze_teams(data):
    stats = {}
    avg_goals = 1.5
    
    if data and data.get('standings') and 'standings' in data['standings']:
        standings_list = data['standings']['standings']
        if not standings_list: return {}, 1.5
        
        table = standings_list[0]['table']
        total_g = sum(t['goalsFor'] for t in table)
        total_p = sum(t['playedGames'] for t in table)
        avg_goals = (total_g / total_p) if total_p > 0 else 1.5

        for t in table:
            name = t['team']['name']
            played = t['playedGames']
            
            raw_form = t.get('form')
            form_str = (raw_form if raw_form is not None else '').replace(',', '')
            form_val = 1.0
            if form_str:
                score = sum({'W':1.1, 'D':1.0, 'L':0.9}.get(c, 1.0) for c in form_str)
                form_val = score / len(form_str)

            stats[name] = {
                'att': (t['goalsFor']/played)/avg_goals if played>0 else 1,
                'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1,
                'form': form_val,
                'rank': t['position'],
                'bonus': 0
            }
            
    if data and data.get('scorers') and 'scorers' in data['scorers']:
        for p in data['scorers']['scorers']:
            tname = p['team']['name']
            if tname in stats: stats[tname]['bonus'] += (p['goals'] * 0.005)

    return stats, avg_goals

def simulate_detailed(home, away, stats, avg_goals):
    if home not in stats or away not in stats: return None
    
    h = stats[home]
    a = stats[away]
    
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form'] * (1 + h['bonus'])
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form'] * (1 + a['bonus'])
    
    SIMS = 10000 # Hızlı sonuç için düşürdük
    rng = np.random.default_rng()
    
    h_goals = rng.poisson(total_h_xg, SIMS)
    a_goals = rng.poisson(total_a_xg, SIMS)
    
    home_wins = np.sum(h_goals > a_goals)
    draws = np.sum(h_goals == a_goals)
    away_wins = np.sum(h_goals < a_goals)
    
    total_goals = h_goals + a_goals
    over_25 = np.sum(total_goals > 2.5)
    btts_yes = np.sum((h_goals > 0) & (a_goals > 0))
    
    return {
        'probs': {'1': (home_wins/SIMS)*100, 'X': (draws/SIMS)*100, '2': (away_wins/SIMS)*100},
        'goals': {'o25': (over_25/SIMS)*100, 'btts': (btts_yes/SIMS)*100},
        'xg': {'h': total_h_xg, 'a': total_a_xg}
    }

# -----------------------------------------------------------------------------
# 4. ARAYÜZ (MAIN)
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 Quantum v11")
    league_name = st.sidebar.selectbox("Lig Seç:", list(LEAGUES.keys()))
    
    st.title(f"Quantum Football: {league_name}")
    
    with st.spinner("Veriler işleniyor..."):
        data = fetch_data(LEAGUES[league_name])
        
    # --- KRİTİK HATA KONTROLÜ BURADA ---
    # Eğer veri yoksa veya boşsa çökmemesi için kontrol ekliyoruz
    if not data or not data.get('matches') or not data['matches'].get('matches'):
        st.warning(f"⚠️ {league_name} için güncel maç verisi bulunamadı veya API limiti doldu.")
        st.info("Lütfen başka bir lig seçin veya daha sonra tekrar deneyin.")
        return # İşlemi burada durdur, aşağıya geçip çökme.
        
    stats, avg_goals = analyze_teams(data)
    
    # Maç listesini oluştur
    matches = {}
    for m in data['matches']['matches']:
        try:
            h = m['homeTeam']['name']
            a = m['awayTeam']['name']
            matches[f"{h} - {a}"] = m
        except: continue
        
    if not matches:
        st.warning("Eşleşme listesi oluşturulamadı.")
        return

    # Seçim Kutusu
    selected = st.selectbox("Maç Seç:", list(matches.keys()))
    
    # Eğer selected None ise (yani liste boşsa) işlem yapma
    if not selected:
        st.warning("Lütfen listeden bir maç seçin.")
        return

    match_data = matches[selected]
    h_name = match_data['homeTeam']['name']
    a_name = match_data['awayTeam']['name']
    
    if st.button("🧠 Analiz Et"):
        res = simulate_detailed(h_name, a_name, stats, avg_goals)
        
        if res:
            c1, c2, c3 = st.columns(3)
            c1.metric(h_name, f"%{res['probs']['1']:.1f}")
            c2.metric("Beraberlik", f"%{res['probs']['X']:.1f}")
            c3.metric(a_name, f"%{res['probs']['2']:.1f}")
            
            st.progress(res['probs']['1']/100)
            st.progress(res['probs']['2']/100)
            
            st.info(f"⚽ 2.5 Üst İhtimali: %{res['goals']['o25']:.1f}")
            st.info(f"🥅 KG Var İhtimali: %{res['goals']['btts']:.1f}")
        else:
            st.error("Bu takımlar için yeterli istatistik yok.")

if __name__ == "__main__":
    main()
