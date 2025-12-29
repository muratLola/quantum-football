import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. AYARLAR VE CSS TASARIMI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Ultimate",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* GENEL ARKA PLAN */
    .stApp {background-color: #0f172a;}
    
    /* KARTLAR */
    .stat-card {
        background-color: #1e293b; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #334155; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* KUPON KARTI (TICKET) TASARIMI */
    .ticket-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 20px;
        position: relative;
        margin-top: 20px;
        margin-bottom: 30px;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.15);
        max-width: 600px;
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
        letter-spacing: 3px; 
        font-size: 1.2rem;
    }
    .ticket-body { text-align: center; color: white; }
    .ticket-match { font-size: 1.4rem; font-weight: bold; margin-bottom: 5px; color: #e2e8f0; }
    .ticket-prediction { 
        font-size: 2.5rem; 
        font-weight: 900; 
        color: #facc15; /* Sarı */
        margin: 10px 0; 
        text-shadow: 0 4px 10px rgba(250, 204, 21, 0.3);
        letter-spacing: 1px;
    }
    .ticket-info-row {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
        background: rgba(255,255,255,0.03);
        padding: 10px;
        border-radius: 8px;
    }
    .ticket-label { color: #94a3b8; font-size: 0.8rem; font-family: 'Courier New', monospace; }
    .ticket-value { color: #00ff88; font-weight: bold; font-size: 1.1rem; }
    .ticket-alt { color: #fbbf24; font-weight: bold; font-size: 1.1rem; }
    
    .ticket-footer { 
        margin-top: 20px; 
        border-top: 2px dashed #334155; 
        padding-top: 10px; 
        text-align: center; 
        font-size: 0.7rem; 
        color: #64748b; 
        font-family: 'Courier New', monospace; 
    }
    .barcode { font-size: 2.5rem; opacity: 0.3; letter-spacing: 6px; margin-bottom: 5px; }

    /* RENKLER */
    .win-green {color: #4ade80;}
    .loss-red {color: #f87171;}
    .draw-yellow {color: #fbbf24;}
    
    /* RİSK GÖSTERGESİ */
    .risk-box {
        background-color: #1e293b; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        margin-top: 10px;
        border: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AYARLAR & SABİTLER
# -----------------------------------------------------------------------------
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

# Lig Karakteristikleri (Gol Çarpanları)
LEAGUE_MULTIPLIERS = { 
    'PL': 1.05, 'TR1': 1.05, 'PD': 0.95, 'BL1': 1.25, 
    'SA': 1.00, 'FL1': 0.90, 'DED': 1.20, 'CL': 1.00 
}

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 3. VERİ ÇEKME MOTORU (HİBRİT)
# -----------------------------------------------------------------------------
def fetch_tff_data_hybrid():
    """ Süper Lig için TFF Scraper """
    try:
        url = "https://www.tff.org/default.aspx?pageID=198"
        headers = {"User-Agent": "Mozilla/5.0"}
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
                    # TFF form verisi vermediği için varsayılan bir form atıyoruz
                    "form": "W,D,L,W,D", 
                    "goalsFor": int(row.get('A', 0)), "goalsAgainst": int(row.get('Y', 0)),
                    "points": int(row.get('P', 0))
                })
            except: continue

        # Sanal Fikstür (Eğer gerçek maç yoksa ilk 6 takımı eşleştir)
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
    """ Ana Veri Çekme Fonksiyonu """
    if league_code == 'TR1': return fetch_tff_data_hybrid()
    try:
        data = {}
        # Puan Durumu
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        if r1.status_code != 200: return None
        data['standings'] = r1.json()
        
        # Gol Krallığı (Bonus için)
        r2 = requests.get(f"{BASE_URL}/competitions/{league_code}/scorers?limit=10", headers=HEADERS)
        data['scorers'] = r2.json() if r2.status_code == 200 else {'scorers': []}
        
        # Fikstür
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        return data
    except: return None

# -----------------------------------------------------------------------------
# 4. GRAFİK VE HESAPLAMA YARDIMCILARI
# -----------------------------------------------------------------------------
def get_momentum_data(form_str):
    """ W,L,D stringini sayısal grafiğe çevirir """
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

def create_radar_chart(h_name, h_stats, a_name, a_stats):
    """ Plotly ile Radar (Örümcek) Grafiği Çizer """
    categories = ['Hücum', 'Savunma', 'Form', 'Gol Gücü', 'İstikrar']
    
    # Normalize edilmiş puanlar (0-100 arası)
    h_values = [
        min(h_stats['att'] * 55, 100),
        min((3.5 - h_stats['def']) * 30, 100),
        min(h_stats['form_val'] * 85, 100),
        min(h_stats['att'] * 40 + h_stats['form_val'] * 30, 100),
        min(h_stats['form_val'] * 95, 100)
    ]
    
    a_values = [
        min(a_stats['att'] * 55, 100),
        min((3.5 - a_stats['def']) * 30, 100),
        min(a_stats['form_val'] * 85, 100),
        min(a_stats['att'] * 40 + a_stats['form_val'] * 30, 100),
        min(a_stats['form_val'] * 95, 100)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=h_values, theta=categories, fill='toself', name=h_name,
        line=dict(color='#00ff88'), fillcolor='rgba(0, 255, 136, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=a_values, theta=categories, fill='toself', name=a_name,
        line=dict(color='#f87171'), fillcolor='rgba(248, 113, 113, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# 5. ANALİZ VE SİMÜLASYON MOTORU
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
            stats[name] = {
                'att': (t['goalsFor']/played)/avg_goals if played>0 else 1, 
                'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1, 
                'form_val': form_val, 
                'form_str': raw_form,
                'rank': t['position'], 
                'bonus': 0
            }
    # Golcülerden bonus ekle
    if data and data.get('scorers'):
        for p in data['scorers']['scorers']:
            if p['team']['name'] in stats: stats[p['team']['name']]['bonus'] += (p['goals'] * 0.005)
    return stats, avg_goals

def simulate_full_match(home, away, stats, avg_goals, league_code):
    if home not in stats or away not in stats: return None
    h, a = stats[home], stats[away]
    
    # Lig Çarpanı Uygula
    league_factor = LEAGUE_MULTIPLIERS.get(league_code, 1.0)
    
    # xG (Gol Beklentisi) Hesaplama
    total_h_xg = h['att'] * a['def'] * avg_goals * 1.15 * h['form_val'] * (1 + h['bonus']) * league_factor
    total_a_xg = a['att'] * h['def'] * avg_goals * a['form_val'] * (1 + a['bonus']) * league_factor
    
    # Monte Carlo Simülasyonu
    SIMS = 20000
    rng = np.random.default_rng()
    h_goals = rng.poisson(total_h_xg, SIMS)
    a_goals = rng.poisson(total_a_xg, SIMS)
    
    # Olasılıklar
    prob_1 = (np.sum(h_goals > a_goals)/SIMS)*100
    prob_x = (np.sum(h_goals == a_goals)/SIMS)*100
    prob_2 = (np.sum(h_goals < a_goals)/SIMS)*100
    
    # Adil Oranlar
    fair_odd_1 = 100 / prob_1 if prob_1 > 0 else 0
    fair_odd_x = 100 / prob_x if prob_x > 0 else 0
    fair_odd_2 = 100 / prob_2 if prob_2 > 0 else 0
    
    # En Güçlü Tahmin
    max_prob = max(prob_1, prob_x, prob_2)
    main_pred = "BELİRSİZ"
    if max_prob == prob_1: main_pred = f"{home} KAZANIR"
    elif max_prob == prob_2: main_pred = f"{away} KAZANIR"
    else: main_pred = "BERABERLİK"
    
    # Gol ve Alternatif Tahminler
    total_goals = h_goals + a_goals
    prob_o25 = (np.sum(total_goals > 2.5)/SIMS)*100
    prob_btts = (np.sum((h_goals>0)&(a_goals>0))/SIMS)*100
    
    alt_pred = "RİSKLİ"
    if prob_o25 > 65: alt_pred = "2.5 ÜST"
    elif prob_btts > 60: alt_pred = "KG VAR"
    elif (np.sum(total_goals < 3.5)/SIMS)*100 > 70: alt_pred = "3.5 ALT"

    return {
        'probs': {'1': prob_1, 'X': prob_x, '2': prob_2},
        'fair_odds': {'1': fair_odd_1, 'X': fair_odd_x, '2': fair_odd_2},
        'goals': {'o25': prob_o25, 'btts': prob_btts},
        'stats': {'h': h, 'a': a},
        'forms': {'h': h['form_str'], 'a': a['form_str']},
        'ticket': {'main': main_pred, 'alt': alt_pred, 'conf': max_prob}
    }

# -----------------------------------------------------------------------------
# 6. ARAYÜZ (MAIN LOOP)
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🧬 Quantum Ultimate")
    league_name = st.sidebar.selectbox("Ligi Seçiniz:", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    st.title(f"Quantum Intelligence: {league_name}")
    st.caption("Advanced AI Sports Analytics • v17.0")
    
    # Veri Yükleme
    with st.spinner("Veri tabanlarına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    # Hata Kontrolü
    if not data or not data.get('matches'):
        st.error(f"⚠️ {league_name} verisi şu an alınamıyor. Lütfen başka bir lig seçin.")
        return
        
    stats, avg_goals = analyze_teams(data)
    
    # Maç Listesi
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    if not matches: st.warning("Bu ligde yakında oynanacak maç bulunamadı."); return
    
    selected = st.selectbox("Analiz Edilecek Maçı Seçin:", list(matches.keys()))
    if not selected: return
    
    # Seçilen Maç
    m_data = matches[selected]
    h, a = m_data['homeTeam']['name'], m_data['awayTeam']['name']
    
    if st.button("🚀 QUANTUM ANALİZİ BAŞLAT"):
        res = simulate_full_match(h, a, stats, avg_goals, league_code)
        
        if res:
            # --- 1. KUPON KARTI (THE TICKET) ---
            st.markdown(f"""
            <div class="ticket-container">
                <div class="ticket-header">QUANTUM INTELLIGENCE</div>
                <div class="ticket-body">
                    <div class="ticket-match">{h} vs {a}</div>
                    <div class="ticket-prediction">{res['ticket']['main']}</div>
                    
                    <div class="ticket-info-row">
                        <div>
                            <div class="ticket-label">GÜVEN</div>
                            <div class="ticket-value">%{res['ticket']['conf']:.1f}</div>
                        </div>
                        <div>
                            <div class="ticket-label">ADİL ORAN</div>
                            <div class="ticket-value">{res['fair_odds']['1'] if 'MS 1' in res['ticket']['main'] else res['fair_odds']['2'] if 'MS 2' in res['ticket']['main'] else res['fair_odds']['X']:.2f}</div>
                        </div>
                        <div>
                            <div class="ticket-label">ALTERNATİF</div>
                            <div class="ticket-alt">{res['ticket']['alt']}</div>
                        </div>
                    </div>
                </div>
                <div class="ticket-footer">
                    <div class="barcode">||| || ||| | |||| |||</div>
                    <div>SESSION ID: {str(int(res['ticket']['conf']*9999))} • {datetime.now().strftime("%H:%M")}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 2. RADAR & MOMENTUM (GÖRSEL ANALİZ) ---
            st.markdown("---")
            col_radar, col_mom = st.columns(2)
            
            with col_radar:
                st.subheader("🕸️ 360° Güç Dengesi")
                radar_fig = create_radar_chart(h, res['stats']['h'], a, res['stats']['a'])
                st.plotly_chart(radar_fig, use_container_width=True)
                
            with col_mom:
                st.subheader("📈 Momentum (Son 5 Maç)")
                h_mom = get_momentum_data(res['forms']['h'])
                a_mom = get_momentum_data(res['forms']['a'])
                
                chart_df = pd.DataFrame({h: h_mom, a: a_mom})
                st.line_chart(chart_df, color=["#00ff88", "#f87171"])
                
                # Risk İbresi
                risk_color = "#4ade80" if res['ticket']['conf'] > 60 else "#facc15" if res['ticket']['conf'] > 45 else "#f87171"
                risk_text = "DÜŞÜK RİSK" if res['ticket']['conf'] > 60 else "ORTA RİSK" if res['ticket']['conf'] > 45 else "YÜKSEK RİSK"
                
                st.markdown(f"""
                <div class="risk-box">
                    <span style="color:#94a3b8; font-size:0.8rem;">YAPAY ZEKA RİSK SEVİYESİ</span><br>
                    <span style="color:{risk_color}; font-size:1.5rem; font-weight:bold;">{risk_text}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # --- 3. DETAYLI OLASILIKLAR & VALUE ---
            st.markdown("---")
            st.subheader("📊 Detaylı Olasılıklar")
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-card'><h3>{h}</h3><h2 class='win-green'>%{res['probs']['1']:.1f}</h2><p>Adil: {res['fair_odds']['1']:.2f}</p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-card'><h3>Beraberlik</h3><h2 class='draw-yellow'>%{res['probs']['X']:.1f}</h2><p>Adil: {res['fair_odds']['X']:.2f}</p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-card'><h3>{a}</h3><h2 class='loss-red'>%{res['probs']['2']:.1f}</h2><p>Adil: {res['fair_odds']['2']:.2f}</p></div>", unsafe_allow_html=True)
            
            st.write("")
            col_g1, col_g2 = st.columns(2)
            col_g1.progress(res['goals']['o25']/100, text=f"2.5 ÜST Gol İhtimali: %{res['goals']['o25']:.1f}")
            col_g2.progress(res['goals']['btts']/100, text=f"KG VAR (BTTS) İhtimali: %{res['goals']['btts']:.1f}")
            
        else:
            st.error("Bu maç için yeterli veri bulunamadı.")

if __name__ == "__main__":
    main()
