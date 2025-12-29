import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum v18: Master Edition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {background-color: #0f172a;}
    
    /* KARTLAR */
    .stat-card {
        background-color: #1e293b; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #334155; 
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* KUPON (TICKET) */
    .ticket-container {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        margin: 20px auto;
        max-width: 600px;
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.15);
        position: relative;
    }
    .ticket-header { 
        color: #00ff88; font-family: monospace; text-align: center; 
        letter-spacing: 3px; border-bottom: 1px dashed #475569; padding-bottom: 10px;
    }
    .ticket-main-pred {
        font-size: 2.2rem; font-weight: 900; color: #facc15; 
        text-align: center; margin: 15px 0; text-shadow: 0 0 10px rgba(250, 204, 21, 0.4);
    }
    .ticket-sub-info {
        display: flex; justify-content: space-between; font-family: monospace; color: #cbd5e1; font-size: 0.9rem;
    }
    
    /* RENKLER */
    .text-green {color: #4ade80;} .text-red {color: #f87171;} .text-yellow {color: #fbbf24;}
    
    /* TABLO BAŞLIKLARI */
    .market-header {
        font-size: 1.1rem; font-weight: bold; color: #e2e8f0; 
        border-bottom: 2px solid #334155; margin-bottom: 10px; padding-bottom: 5px;
    }
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
# 3. VERİ ÇEKME (HİBRİT)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(league_code):
    # TFF ÖZEL (Süper Lig)
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
                    
                    # TFF Form verisi vermediği için rastgele gerçekçi form üretiyoruz (Görsellik için)
                    # Gerçek senaryoda buraya maç sonuçları scraper bağlanmalı
                    standings.append({
                        "team": {"name": team_name},
                        "playedGames": int(row.get('O', 0)),
                        "form": "W,D,W,L,D", # Varsayılan
                        "goalsFor": int(row.get('A', 0)),
                        "goalsAgainst": int(row.get('Y', 0)),
                        "points": int(row.get('P', 0)),
                        "position": idx+1
                    })
                except: continue
                
            # Maç Listesi (İlk 6 takımı eşleştir - Demo Amaçlı)
            matches = []
            if len(standings) > 0:
                top = [t['team']['name'] for t in standings[:6]]
                import itertools
                for p in itertools.combinations(top, 2):
                    matches.append({"homeTeam": {"name": p[0]}, "awayTeam": {"name": p[1]}, "utcDate": datetime.now().isoformat()})

            return {"standings": {"standings": [{"table": standings}]}, "matches": {"matches": matches}, "scorers": {"scorers": []}}
        except: return None

    # GLOBAL API
    try:
        data = {}
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        data['standings'] = r1.json() if r1.status_code == 200 else None
        
        r2 = requests.get(f"{BASE_URL}/competitions/{league_code}/scorers?limit=10", headers=HEADERS)
        data['scorers'] = r2.json() if r2.status_code == 200 else {'scorers': []}
        
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        return data
    except: return None

# -----------------------------------------------------------------------------
# 4. GRAFİK VE HESAPLAMA MOTORU
# -----------------------------------------------------------------------------
def get_momentum_data(form_str):
    """ DÜZELTİLMİŞ GRAFİK MANTIĞI: Kümülatif Puanlama """
    if not form_str: return [0]*5
    
    # Virgülleri sil ve listeye çevir
    form_str = form_str.replace(',', '')
    # Son 5 maçı al
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    
    # Başlangıç noktası 0
    points = [0]
    current_val = 0
    
    # API genelde "En yeni en sağda" verir.
    for char in last_5:
        if char == 'W': current_val += 3   # Galibiyet: Yüksel
        elif char == 'D': current_val += 1  # Beraberlik: Az Yüksel
        elif char == 'L': current_val -= 2  # Mağlubiyet: Düş (Cezalandır)
        points.append(current_val)
        
    return points

def create_radar(h, h_stats, a, a_stats):
    categories = ['Hücum', 'Savunma', 'Form', 'Gol Pot.', 'İstikrar']
    
    # Değerleri normalize et
    h_vals = [
        min(h_stats['att']*50, 100), min((3.5-h_stats['def'])*30, 100),
        min(h_stats['form_val']*80, 100), min(h_stats['att']*40 + h_stats['form_val']*20, 100),
        min(h_stats['form_val']*90, 100)
    ]
    a_vals = [
        min(a_stats['att']*50, 100), min((3.5-a_stats['def'])*30, 100),
        min(a_stats['form_val']*80, 100), min(a_stats['att']*40 + a_stats['form_val']*20, 100),
        min(a_stats['form_val']*90, 100)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories, fill='toself', name=h, line_color='#4ade80'))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a, line_color='#f87171'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(orientation="h", y=1.1)
    )
    return fig

# -----------------------------------------------------------------------------
# 5. SİMÜLASYON (GENİŞLETİLMİŞ BAHİS TÜRLERİ)
# -----------------------------------------------------------------------------
def analyze_match_advanced(home, away, stats, avg_goals, multipliers):
    if home not in stats or away not in stats: return None
    h, a = stats[home], stats[away]
    
    # Manuel Çarpanlar (Sidebar'dan gelen)
    manual_h_impact = multipliers.get('home_impact', 1.0)
    manual_a_impact = multipliers.get('away_impact', 1.0)
    
    # xG Hesaplama (İlk Yarı ve İkinci Yarı Ayrı)
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form_val'] * (1+h['bonus']) * manual_h_impact
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form_val'] * (1+a['bonus']) * manual_a_impact
    
    SIMS = 15000
    rng = np.random.default_rng()
    
    # 1. Yarı ve 2. Yarı Simülasyonu
    h_ht = rng.poisson(total_h_xg * 0.45, SIMS) # İlk yarı golleri
    h_ft = h_ht + rng.poisson(total_h_xg * 0.55, SIMS) # İkinci yarı eklenir = Maç Sonu
    
    a_ht = rng.poisson(total_a_xg * 0.45, SIMS)
    a_ft = a_ht + rng.poisson(total_a_xg * 0.55, SIMS)
    
    # --- SONUÇ HESAPLAMALARI ---
    
    # 1. Maç Sonucu (1X2)
    ms_1 = np.sum(h_ft > a_ft)
    ms_x = np.sum(h_ft == a_ft)
    ms_2 = np.sum(h_ft < a_ft)
    
    # 2. Çifte Şans
    cs_1x = ms_1 + ms_x
    cs_12 = ms_1 + ms_2
    cs_x2 = ms_2 + ms_x
    
    # 3. İY / MS (HT/FT)
    # HT Sonuçları
    ht_1 = (h_ht > a_ht)
    ht_x = (h_ht == a_ht)
    ht_2 = (h_ht < a_ht)
    # FT Sonuçları
    ft_1 = (h_ft > a_ft)
    ft_x = (h_ft == a_ft)
    ft_2 = (h_ft < a_ft)
    
    # Kombinasyonlar
    htft_1_1 = np.sum(ht_1 & ft_1)
    htft_x_1 = np.sum(ht_x & ft_1)
    htft_2_2 = np.sum(ht_2 & ft_2)
    htft_x_x = np.sum(ht_x & ft_x)
    
    # 4. Gol Baremleri
    total_goals = h_ft + a_ft
    o15 = np.sum(total_goals > 1.5)
    o25 = np.sum(total_goals > 2.5)
    u35 = np.sum(total_goals < 3.5)
    btts = np.sum((h_ft > 0) & (a_ft > 0))
    
    # 5. Skor Tahmini (En olası 3)
    hashes = h_ft * 100 + a_ft
    unique, counts = np.unique(hashes, return_counts=True)
    sorted_idx = np.argsort(-counts)
    top_scores = []
    for i in range(3):
        val = unique[sorted_idx[i]]
        s_h, s_a = val // 100, val % 100
        top_scores.append(f"{s_h}-{s_a}")

    # 6. Beraberlikte İade (Draw No Bet)
    # Beraberlikleri yok sayıp oranlıyoruz
    total_decisive = ms_1 + ms_2
    dnb_1 = (ms_1 / total_decisive * 100) if total_decisive > 0 else 0
    dnb_2 = (ms_2 / total_decisive * 100) if total_decisive > 0 else 0

    return {
        '1x2': {'1': ms_1/SIMS*100, 'X': ms_x/SIMS*100, '2': ms_2/SIMS*100},
        'dc': {'1X': cs_1x/SIMS*100, '12': cs_12/SIMS*100, 'X2': cs_x2/SIMS*100},
        'htft': {'1/1': htft_1_1/SIMS*100, 'X/1': htft_x_1/SIMS*100, 'X/X': htft_x_x/SIMS*100, '2/2': htft_2_2/SIMS*100},
        'goals': {'o15': o15/SIMS*100, 'o25': o25/SIMS*100, 'u35': u35/SIMS*100, 'btts': btts/SIMS*100},
        'scores': top_scores,
        'dnb': {'1': dnb_1, '2': dnb_2},
        'stats': {'h': h, 'a': a}
    }

# -----------------------------------------------------------------------------
# 6. ANA ARAYÜZ
# -----------------------------------------------------------------------------
def main():
    st.sidebar.header("⚙️ Analiz Ayarları")
    
    # --- MANUEL FAKTÖRLER (KODSUZ MÜDAHALE) ---
    st.sidebar.markdown("### 🏟️ Saha Dışı Faktörler")
    st.sidebar.info("Yapay zekaya ekstra bilgi vererek analizi keskinleştir.")
    
    h_impact = 1.0
    a_impact = 1.0
    
    # Ev Sahibi Faktörleri
    st.sidebar.caption("Ev Sahibi Durumu")
    if st.sidebar.checkbox("Ev Sahibi: Kritik Eksik Var 🚑", key="h_inj"): h_impact -= 0.15
    if st.sidebar.checkbox("Ev Sahibi: Seyirci Cezası 🔇", key="h_fan"): h_impact -= 0.10
    
    # Deplasman Faktörleri
    st.sidebar.caption("Deplasman Durumu")
    if st.sidebar.checkbox("Deplasman: Yorgun (Avrupa Dönüşü) ✈️", key="a_tired"): a_impact -= 0.20
    if st.sidebar.checkbox("Deplasman: Teknik Direktör Krizi 📉", key="a_crisis"): a_impact -= 0.15
    
    # Ortak Faktörler
    st.sidebar.caption("Maç Koşulları")
    weather = st.sidebar.selectbox("Hava/Zemin Durumu:", ["Normal", "Yağmurlu/Ağır Zemin", "Karlı/Buzlu"])
    if weather == "Yağmurlu/Ağır Zemin": 
        h_impact *= 0.9; a_impact *= 0.9 # Gol ihtimali düşer
    elif weather == "Karlı/Buzlu":
        h_impact *= 0.8; a_impact *= 0.8
    
    multipliers = {'home_impact': h_impact, 'away_impact': a_impact}

    # --- LİG SEÇİMİ ---
    league_name = st.sidebar.selectbox("Ligi Seçiniz:", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    st.title("🧠 Quantum v18: Master Analiz")
    
    with st.spinner("Veriler işleniyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'): st.error("Veri alınamadı."); return
    
    # Takım İstatistiklerini Hazırla
    stats = {}
    avg_goals = 1.5
    if data['standings']:
        table = data['standings']['standings'][0]['table']
        tg = sum(t['goalsFor'] for t in table); tp = sum(t['playedGames'] for t in table)
        avg_goals = tg/tp if tp>0 else 1.5
        for t in table:
            name = t['team']['name']; played = t['playedGames']
            raw_form = t.get('form', 'D,D,D,D,D')
            form_val = 1.0
            if raw_form:
                score = sum({'W':1.1, 'D':1.0, 'L':0.9}.get(c, 1.0) for c in raw_form.replace(',',''))
                form_val = score/len(raw_form.replace(',',''))
            stats[name] = {'att': (t['goalsFor']/played)/avg_goals if played>0 else 1, 'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1, 'form_val': form_val, 'form_str': raw_form, 'bonus': 0}
            
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    if not matches: st.warning("Maç yok."); return
    
    selected = st.selectbox("Maç Seç:", list(matches.keys()))
    if not selected: return
    
    m_data = matches[selected]
    h_name, a_name = m_data['homeTeam']['name'], m_data['awayTeam']['name']
    
    if st.button("🚀 ANALİZİ BAŞLAT"):
        res = analyze_match_advanced(h_name, a_name, stats, avg_goals, multipliers)
        
        if res:
            # --- 1. TICKET (ÖZET) ---
            conf = max(res['1x2']['1'], res['1x2']['X'], res['1x2']['2'])
            main_pred = "EV SAHİBİ" if res['1x2']['1'] == conf else "DEPLASMAN" if res['1x2']['2'] == conf else "BERABERLİK"
            
            st.markdown(f"""
            <div class="ticket-container">
                <div class="ticket-header">QUANTUM INTELLIGENCE</div>
                <div class="ticket-main-pred">{main_pred}</div>
                <div class="ticket-sub-info">
                    <span>GÜVEN: %{conf:.1f}</span>
                    <span>SKOR: {res['scores'][0]}</span>
                    <span>İY/MS: {"1/1" if res['htft']['1/1']>20 else "X/X"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 2. GÖRSEL ANALİZ (TABLAR) ---
            tab_vis, tab_markets = st.tabs(["📊 Görsel Analiz", "💰 Detaylı Bahisler"])
            
            with tab_vis:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Güç Dengesi (Radar)")
                    st.plotly_chart(create_radar(h_name, stats[h_name], a_name, stats[a_name]), use_container_width=True)
                with c2:
                    st.subheader("Momentum (Son 5 Maç)")
                    h_mom = get_momentum_data(stats[h_name]['form_str'])
                    a_mom = get_momentum_data(stats[a_name]['form_str'])
                    chart_data = pd.DataFrame({h_name: h_mom, a_name: a_mom})
                    st.line_chart(chart_data, color=["#4ade80", "#f87171"])
            
            with tab_markets:
                # 3 KOLONLU PİYASA EKRANI
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='market-header'>📌 Maç Sonucu</div>", unsafe_allow_html=True)
                    st.write(f"**MS 1:** %{res['1x2']['1']:.1f}")
                    st.write(f"**MS X:** %{res['1x2']['X']:.1f}")
                    st.write(f"**MS 2:** %{res['1x2']['2']:.1f}")
                    
                    st.markdown("<div class='market-header' style='margin-top:20px'>🛡️ Çifte Şans</div>", unsafe_allow_html=True)
                    st.write(f"**1X:** %{res['dc']['1X']:.1f}")
                    st.write(f"**12:** %{res['dc']['12']:.1f}")
                    st.write(f"**X2:** %{res['dc']['X2']:.1f}")

                with col2:
                    st.markdown("<div class='market-header'>⚽ Gol Piyasaları</div>", unsafe_allow_html=True)
                    st.write(f"**1.5 Üst:** %{res['goals']['o15']:.1f}")
                    st.write(f"**2.5 Üst:** %{res['goals']['o25']:.1f}")
                    st.write(f"**3.5 Alt:** %{res['goals']['u35']:.1f}")
                    st.write(f"**KG Var:** %{res['goals']['btts']:.1f}")
                    
                    st.markdown("<div class='market-header' style='margin-top:20px'>🔢 Skor Tahmini</div>", unsafe_allow_html=True)
                    st.write(f"1. {res['scores'][0]}")
                    st.write(f"2. {res['scores'][1]}")
                    st.write(f"3. {res['scores'][2]}")

                with col3:
                    st.markdown("<div class='market-header'>⏳ İY / MS</div>", unsafe_allow_html=True)
                    st.write(f"**1 / 1:** %{res['htft']['1/1']:.1f}")
                    st.write(f"**X / 1:** %{res['htft']['X/1']:.1f}")
                    st.write(f"**X / X:** %{res['htft']['X/X']:.1f}")
                    st.write(f"**2 / 2:** %{res['htft']['2/2']:.1f}")
                    
                    st.markdown("<div class='market-header' style='margin-top:20px'>🔄 Beraberlikte İade</div>", unsafe_allow_html=True)
                    st.write(f"**DNB 1:** %{res['dnb']['1']:.1f}")
                    st.write(f"**DNB 2:** %{res['dnb']['2']:.1f}")

        else: st.error("Analiz yapılamadı.")

if __name__ == "__main__":
    main()
