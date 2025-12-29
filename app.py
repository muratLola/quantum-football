import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR VE STİL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Analyst v12: Full Detail",
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
    /* Tablo ayarları */
    div[data-testid="stTable"] {background-color: #1e293b; border-radius: 10px;}
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
        # Hata durumunda None dön
        if response.status_code != 200: return None
        
        try:
            tables = pd.read_html(response.content)
        except ValueError:
            return None 

        if not tables:
            return None

        # --- PUAN DURUMU ---
        df_standings = tables[0]
        
        if "Takım" not in df_standings.columns:
            df_standings.columns = df_standings.iloc[0]
            df_standings = df_standings[1:]
            
        standings_table = []
        for index, row in df_standings.iterrows():
            try:
                played = int(row.get('O', 0))
                points = int(row.get('P', 0))
                raw_team = str(row.get('Takım', 'Bilinmiyor'))
                
                # İsim temizleme
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
                    "form": "WWWWW", 
                    "goalsFor": goals_for,
                    "goalsAgainst": goals_against,
                    "points": points
                })
            except: continue

        api_standings = {"standings": [{"table": standings_table}]}

        # --- MAÇLAR (SANAL FİKSTÜR) ---
        # Süper Lig maçlarını bulamazsa ilk 6 takımı birbiriyle eşleştir
        matches_list = []
        if len(standings_table) > 0:
            top_teams = [t['team']['name'] for t in standings_table[:6]]
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

    except Exception:
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
# 3. ANALİZ MOTORU (DETAYLI VERSİYON)
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
    
    # xG Hesaplama
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form'] * (1 + h['bonus'])
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form'] * (1 + a['bonus'])
    
    # Simülasyon Sayısı
    SIMS = 100000 
    rng = np.random.default_rng()
    
    # 1. Yarı ve 2. Yarı Simülasyonu (IY/MS için)
    h_goals_1 = rng.poisson(total_h_xg * 0.45, SIMS)
    h_goals_2 = rng.poisson(total_h_xg * 0.55, SIMS)
    a_goals_1 = rng.poisson(total_a_xg * 0.45, SIMS)
    a_goals_2 = rng.poisson(total_a_xg * 0.55, SIMS)
    
    h_total = h_goals_1 + h_goals_2
    a_total = a_goals_1 + a_goals_2
    
    # Sonuçlar
    home_wins = np.sum(h_total > a_total)
    draws = np.sum(h_total == a_total)
    away_wins = np.sum(h_total < a_total)
    
    # Gol Piyasaları
    total_goals = h_total + a_total
    over_15 = np.sum(total_goals > 1.5)
    over_25 = np.sum(total_goals > 2.5)
    btts_yes = np.sum((h_total > 0) & (a_total > 0))
    
    # Çifte Şans
    dc_1x = home_wins + draws
    dc_x2 = away_wins + draws
    dc_12 = home_wins + away_wins
    
    # İY/MS Analizi
    ht_1 = (h_goals_1 > a_goals_1)
    ht_x = (h_goals_1 == a_goals_1)
    ht_2 = (h_goals_1 < a_goals_1)
    
    ft_1 = (h_total > a_total)
    ft_x = (h_total == a_total)
    ft_2 = (h_total < a_total)
    
    htft_1_1 = np.sum(ht_1 & ft_1)
    htft_x_1 = np.sum(ht_x & ft_1)
    htft_x_x = np.sum(ht_x & ft_x)
    htft_2_2 = np.sum(ht_2 & ft_2)

    # Skor Analizi (En Olası 5 Skor)
    score_hashes = h_total * 100 + a_total
    unique, counts = np.unique(score_hashes, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    top_scores = []
    for i in range(min(5, len(unique))):
        score_val = unique[sorted_indices[i]]
        h_s = score_val // 100
        a_s = score_val % 100
        prob = (counts[sorted_indices[i]] / SIMS) * 100
        top_scores.append((f"{h_s}-{a_s}", prob))

    return {
        'probs': {'1': (home_wins/SIMS)*100, 'X': (draws/SIMS)*100, '2': (away_wins/SIMS)*100},
        'goals': {'o15': (over_15/SIMS)*100, 'o25': (over_25/SIMS)*100, 'btts': (btts_yes/SIMS)*100},
        'dc': {'1X': (dc_1x/SIMS)*100, '12': (dc_12/SIMS)*100, 'X2': (dc_x2/SIMS)*100},
        'htft': {'1/1': (htft_1_1/SIMS)*100, 'X/1': (htft_x_1/SIMS)*100, 'X/X': (htft_x_x/SIMS)*100, '2/2': (htft_2_2/SIMS)*100},
        'correct_scores': top_scores,
        'xg': {'h': total_h_xg, 'a': total_a_xg}
    }

# -----------------------------------------------------------------------------
# 4. ARAYÜZ (MAIN)
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 Quantum v12")
    league_name = st.sidebar.selectbox("Lig Seç:", list(LEAGUES.keys()))
    
    st.title(f"Quantum Football: {league_name}")
    
    with st.spinner("Veriler işleniyor..."):
        data = fetch_data(LEAGUES[league_name])
        
    # Hata Kontrolü
    if not data or not data.get('matches') or not data['matches'].get('matches'):
        st.warning(f"⚠️ {league_name} verisi şu an alınamıyor.")
        return 
        
    stats, avg_goals = analyze_teams(data)
    
    # Maç listesi
    matches = {}
    for m in data['matches']['matches']:
        try:
            h = m['homeTeam']['name']
            a = m['awayTeam']['name']
            matches[f"{h} - {a}"] = m
        except: continue
        
    if not matches:
        st.warning("Eşleşme bulunamadı.")
        return

    selected = st.selectbox("Maç Seç:", list(matches.keys()))
    
    if not selected:
        return

    match_data = matches[selected]
    h_name = match_data['homeTeam']['name']
    a_name = match_data['awayTeam']['name']
    
    if st.button("🧠 Derinlemesine Analiz Et"):
        res = simulate_detailed(h_name, a_name, stats, avg_goals)
        
        if res:
            # --- TABLI TASARIM GERİ GELDİ ---
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Genel Bakış", "⚽ Gol & Skor", "📊 İY/MS", "💰 Bahis Stratejisi"])
            
            with tab1:
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='stat-card'><div class='sub-text'>{h_name}</div><div class='big-number win-green'>%{res['probs']['1']:.1f}</div><div>xG: {res['xg']['h']:.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='stat-card'><div class='sub-text'>Beraberlik</div><div class='big-number draw-yellow'>%{res['probs']['X']:.1f}</div><div>Risk: Orta</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='stat-card'><div class='sub-text'>{a_name}</div><div class='big-number loss-red'>%{res['probs']['2']:.1f}</div><div>xG: {res['xg']['a']:.2f}</div></div>", unsafe_allow_html=True)
                
                st.write("")
                st.subheader("Kazanma Olasılıkları")
                st.progress(res['probs']['1']/100, text=f"{h_name}")
                st.progress(res['probs']['2']/100, text=f"{a_name}")

            with tab2:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("### 🎯 En Olası 5 Skor")
                    for score, prob in res['correct_scores']:
                        st.markdown(f"**{score}** — %{prob:.1f}")
                        st.progress(prob/100)
                with col_b:
                    st.markdown("### 🥅 Gol Piyasaları")
                    st.markdown(f"**2.5 Üst:** %{res['goals']['o25']:.1f}")
                    st.progress(res['goals']['o25']/100)
                    st.markdown(f"**KG Var (BTTS):** %{res['goals']['btts']:.1f}")
                    st.progress(res['goals']['btts']/100)

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ⏳ İY / MS Tahminleri")
                    st.table(pd.DataFrame({
                        'Senaryo': ['1/1 (Ev/Ev)', 'X/1 (Ber/Ev)', 'X/X (Ber/Ber)', '2/2 (Dep/Dep)'],
                        'Olasılık': [f"%{res['htft']['1/1']:.1f}", f"%{res['htft']['X/1']:.1f}", f"%{res['htft']['X/X']:.1f}", f"%{res['htft']['2/2']:.1f}"]
                    }))
                with c2:
                    st.markdown("### 🛡️ Çifte Şans")
                    st.markdown(f"**1X (Ev Yenilmez):** %{res['dc']['1X']:.1f}")
                    st.markdown(f"**X2 (Dep Yenilmez):** %{res['dc']['X2']:.1f}")

            with tab4:
                st.markdown("### 🧠 AI Bahis Stratejisi")
                
                best_bet = "Maç Sonucu 1" if res['probs']['1'] > 50 else "Maç Sonucu 2" if res['probs']['2'] > 50 else "Çifte Şans 1X"
                best_prob = max(res['probs']['1'], res['probs']['2']) if max(res['probs']['1'], res['probs']['2']) > 50 else res['dc']['1X']
                
                st.info(f"💎 **Ana Öneri:** {best_bet} (Güven: %{best_prob:.1f})")
                
                # Adil Oran Tablosu
                st.markdown("#### ⚖️ Adil Oran Tablosu (Bookie vs AI)")
                fair_odds = {
                    'MS 1': 100/res['probs']['1'] if res['probs']['1']>0 else 0,
                    'Beraberlik': 100/res['probs']['X'] if res['probs']['X']>0 else 0,
                    'MS 2': 100/res['probs']['2'] if res['probs']['2']>0 else 0,
                    '2.5 Üst': 100/res['goals']['o25'] if res['goals']['o25']>0 else 0
                }
                cols = st.columns(4)
                cols[0].metric("MS 1 Oranı", f"{fair_odds['MS 1']:.2f}")
                cols[1].metric("MS X Oranı", f"{fair_odds['Beraberlik']:.2f}")
                cols[2].metric("MS 2 Oranı", f"{fair_odds['MS 2']:.2f}")
                cols[3].metric("2.5 Üst Oranı", f"{fair_odds['2.5 Üst']:.2f}")

        else:
            st.error("Bu takımlar için yeterli istatistik yok.")

if __name__ == "__main__":
    main()
