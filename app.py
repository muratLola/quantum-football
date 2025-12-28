import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from collections import Counter

# -----------------------------------------------------------------------------
# 1. AYARLAR VE STİL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Analyst v10: Ultra Detail",
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

# API
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 2. VERİ ÇEKME VE İŞLEME (HATA DÜZELTMELERİ DAHİL)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(league_code):
    try:
        data = {}
        # Puan Durumu
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        data['standings'] = r1.json() if r1.status_code == 200 else None
        
        # Gol Krallığı
        r2 = requests.get(f"{BASE_URL}/competitions/{league_code}/scorers?limit=10", headers=HEADERS)
        data['scorers'] = r2.json() if r2.status_code == 200 else None
        
        # Gelecek Maçlar (14 Günlük)
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else None
        
        return data
    except: return None

def analyze_teams(data):
    stats = {}
    avg_goals = 1.5
    
    if data.get('standings') and 'standings' in data['standings']:
        standings_list = data['standings']['standings']
        if not standings_list: return {}, 1.5
        
        table = standings_list[0]['table']
        total_g = sum(t['goalsFor'] for t in table)
        total_p = sum(t['playedGames'] for t in table)
        avg_goals = (total_g / total_p) if total_p > 0 else 1.5

        for t in table:
            name = t['team']['name']
            played = t['playedGames']
            
            # Form Analizi
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
            
    if data.get('scorers') and 'scorers' in data['scorers']:
        for p in data['scorers']['scorers']:
            tname = p['team']['name']
            if tname in stats: stats[tname]['bonus'] += (p['goals'] * 0.005) # Golcü etkisi

    return stats, avg_goals

# -----------------------------------------------------------------------------
# 3. QUANTUM SİMÜLASYON MOTORU (DETAYLI MOD)
# -----------------------------------------------------------------------------
def simulate_detailed(home, away, stats, avg_goals):
    if home not in stats or away not in stats: return None
    
    h = stats[home]
    a = stats[away]
    
    # xG Hesaplama (İlk Yarı / İkinci Yarı Ayrımı için)
    # Futbolda gollerin %45'i ilk yarı, %55'i ikinci yarı atılır (yaklaşık)
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form'] * (1 + h['bonus'])
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form'] * (1 + a['bonus'])
    
    # Simülasyon Sayısı (Hız/Doğruluk Dengesi)
    SIMS = 500_000 
    rng = np.random.default_rng()
    
    # 1. Yarı ve 2. Yarı Simülasyonu (Ayrı Ayrı)
    h_goals_1 = rng.poisson(total_h_xg * 0.45, SIMS)
    h_goals_2 = rng.poisson(total_h_xg * 0.55, SIMS)
    a_goals_1 = rng.poisson(total_a_xg * 0.45, SIMS)
    a_goals_2 = rng.poisson(total_a_xg * 0.55, SIMS)
    
    # Toplam Skorlar
    h_total = h_goals_1 + h_goals_2
    a_total = a_goals_1 + a_goals_2
    
    # --- İSTATİSTİK HESAPLAMALARI ---
    
    # 1. Maç Sonucu (1X2)
    home_wins = np.sum(h_total > a_total)
    draws = np.sum(h_total == a_total)
    away_wins = np.sum(h_total < a_total)
    
    # 2. Alt/Üst Piyasaları
    total_goals = h_total + a_total
    over_15 = np.sum(total_goals > 1.5)
    over_25 = np.sum(total_goals > 2.5)
    over_35 = np.sum(total_goals > 3.5)
    
    # 3. KG Var (BTTS)
    btts_yes = np.sum((h_total > 0) & (a_total > 0))
    
    # 4. Çifte Şans
    dc_1x = home_wins + draws
    dc_x2 = away_wins + draws
    dc_12 = home_wins + away_wins
    
    # 5. İY/MS (HT/FT) Analizi
    # İlk yarı sonuçları
    ht_1 = (h_goals_1 > a_goals_1)
    ht_x = (h_goals_1 == a_goals_1)
    ht_2 = (h_goals_1 < a_goals_1)
    
    # Maç sonu sonuçları
    ft_1 = (h_total > a_total)
    ft_x = (h_total == a_total)
    ft_2 = (h_total < a_total)
    
    # Kombinasyonlar
    htft_1_1 = np.sum(ht_1 & ft_1)
    htft_x_1 = np.sum(ht_x & ft_1)
    htft_x_x = np.sum(ht_x & ft_x)
    htft_2_2 = np.sum(ht_2 & ft_2)
    # (Diğerleri de eklenebilir ama en popülerleri bunlar)

    # 6. Skor Analizi (En Olası 5 Skor)
    # Numpy arraylerini stringe çevirmek yavaş olduğu için matematiksel hashing kullanıyoruz
    # Skor ID = Home*100 + Away (Örn: 2-1 -> 201)
    score_hashes = h_total * 100 + a_total
    unique, counts = np.unique(score_hashes, return_counts=True)
    sorted_indices = np.argsort(-counts) # En çoktan aza
    
    top_scores = []
    for i in range(min(5, len(unique))):
        score_val = unique[sorted_indices[i]]
        h_s = score_val // 100
        a_s = score_val % 100
        prob = (counts[sorted_indices[i]] / SIMS) * 100
        top_scores.append((f"{h_s}-{a_s}", prob))

    return {
        'probs': {
            '1': (home_wins/SIMS)*100, 'X': (draws/SIMS)*100, '2': (away_wins/SIMS)*100
        },
        'goals': {
            'o15': (over_15/SIMS)*100, 'o25': (over_25/SIMS)*100, 'o35': (over_35/SIMS)*100,
            'btts': (btts_yes/SIMS)*100
        },
        'dc': {
            '1X': (dc_1x/SIMS)*100, '12': (dc_12/SIMS)*100, 'X2': (dc_x2/SIMS)*100
        },
        'htft': {
            '1/1': (htft_1_1/SIMS)*100, 'X/1': (htft_x_1/SIMS)*100,
            'X/X': (htft_x_x/SIMS)*100, '2/2': (htft_2_2/SIMS)*100
        },
        'correct_scores': top_scores,
        'xg': {'h': total_h_xg, 'a': total_a_xg}
    }

# -----------------------------------------------------------------------------
# 4. ARAYÜZ
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 Quantum v10 Pro")
    league_name = st.sidebar.selectbox("Lig Seç:", list(LEAGUES.keys()))
    
    st.title(f"Quantum Football Analyst: {league_name}")
    st.caption("🚀 500,000 Simülasyon | İY/MS Analizi | Detaylı Skorlar")
    
    with st.spinner("Veriler işleniyor..."):
        data = fetch_data(LEAGUES[league_name])
        
    if not data or not data['matches']:
        st.error("Veri alınamadı.")
        return
        
    stats, avg_goals = analyze_teams(data)
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches']}
    
    selected = st.selectbox("Maç Seç:", list(matches.keys()))
    match_data = matches[selected]
    h_name = match_data['homeTeam']['name']
    a_name = match_data['awayTeam']['name']
    
    if st.button("🧠 Derinlemesine Analiz Et"):
        res = simulate_detailed(h_name, a_name, stats, avg_goals)
        
        if res:
            # --- TAB 1: GENEL BAKIŞ ---
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Genel Bakış", "⚽ Gol & Skor", "📊 İY/MS & Detay", "💰 Bahis Stratejisi"])
            
            with tab1:
                # Ana Oranlar
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='stat-card'><div class='sub-text'>{h_name}</div><div class='big-number win-green'>%{res['probs']['1']:.1f}</div><div>xG: {res['xg']['h']:.2f}</div></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='stat-card'><div class='sub-text'>Beraberlik</div><div class='big-number draw-yellow'>%{res['probs']['X']:.1f}</div><div>Risk: Orta</div></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='stat-card'><div class='sub-text'>{a_name}</div><div class='big-number loss-red'>%{res['probs']['2']:.1f}</div><div>xG: {res['xg']['a']:.2f}</div></div>", unsafe_allow_html=True)
                
                st.write("")
                st.subheader("Simülasyon Özeti")
                # Progress Bar
                st.progress(res['probs']['1']/100, text=f"{h_name} Kazanma İhtimali")
                st.progress(res['probs']['2']/100, text=f"{a_name} Kazanma İhtimali")
                
            with tab2:
                # Skor ve Goller
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
                    st.markdown(f"**1.5 Üst (Banko):** %{res['goals']['o15']:.1f}")
            
            with tab3:
                # İY/MS ve Çifte Şans
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### ⏳ İY / MS Tahminleri")
                    st.table(pd.DataFrame({
                        'Senaryo': ['1/1 (Ev/Ev)', 'X/1 (Ber/Ev)', 'X/X (Ber/Ber)', '2/2 (Dep/Dep)'],
                        'Olasılık': [f"%{res['htft']['1/1']:.1f}", f"%{res['htft']['X/1']:.1f}", f"%{res['htft']['X/X']:.1f}", f"%{res['htft']['2/2']:.1f}"]
                    }))
                with c2:
                    st.markdown("### 🛡️ Çifte Şans (Sigorta)")
                    st.markdown(f"**1X (Ev Yenilmez):** %{res['dc']['1X']:.1f}")
                    st.markdown(f"**X2 (Dep Yenilmez):** %{res['dc']['X2']:.1f}")
                    st.markdown(f"**12 (Beraberlik Olmaz):** %{res['dc']['12']:.1f}")

            with tab4:
                # Value Bet & Strategy
                st.markdown("### 🧠 AI Bahis Stratejisi")
                
                # Güvenilir Seçimler
                best_bet = "Maç Sonucu 1" if res['probs']['1'] > 50 else "Maç Sonucu 2" if res['probs']['2'] > 50 else "Çifte Şans 1X"
                best_prob = max(res['probs']['1'], res['probs']['2']) if max(res['probs']['1'], res['probs']['2']) > 50 else res['dc']['1X']
                
                st.info(f"💎 **Ana Öneri:** {best_bet} (Güven: %{best_prob:.1f})")
                
                if res['goals']['o25'] > 55:
                    st.success(f"🔥 **Alternatif:** 2.5 Gol Üst (Güven: %{res['goals']['o25']:.1f})")
                elif res['goals']['btts'] > 55:
                    st.success(f"🔥 **Alternatif:** KG Var (Güven: %{res['goals']['btts']:.1f})")
                else:
                    st.warning("⚠️ **Alternatif:** 3.5 Gol Alt (Düşük Tempo Bekleniyor)")

                # Adil Oranlar Tablosu
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

if __name__ == "__main__":
    main()