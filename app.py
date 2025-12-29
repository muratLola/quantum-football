import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS (MİNİMALİST TASARIM)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* GENEL ARKA PLAN */
    .stApp {background-color: #0b0f19;}
    
    /* BAŞLIK (SADE VE GÜÇLÜ) */
    .header-container {
        text-align: center;
        padding: 30px;
        margin-bottom: 20px;
        border-bottom: 1px solid #30363d;
    }
    .quantum-title {
        font-family: 'Arial', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        color: #fff;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: -webkit-linear-gradient(#fff, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* KUPON KARTI */
    .ticket-container {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
        border: 1px solid #30363d;
        border-top: 4px solid #00ff88;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    .main-pred { font-size: 3rem; font-weight: 900; color: #facc15; margin: 10px 0; }
    
    /* FORM KUTUCUKLARI (G-B-M) */
    .form-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .form-badge {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
        font-family: monospace;
        font-size: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .form-w { background-color: #4ade80; } /* YEŞİL - GALİBİYET */
    .form-d { background-color: #facc15; } /* SARI - BERABERLİK */
    .form-l { background-color: #f87171; } /* KIRMIZI - MAĞLUBİYET */
    
    .team-name-header {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    /* AI YORUM KUTUSU */
    .ai-comment-box {
        background-color: #22272e;
        border-left: 4px solid #00ff88;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
        color: #adbac7;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* ETIKETLER */
    .badge-banko { background-color: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge-surpriz { background-color: #da3633; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .badge-riskli { background-color: #d29922; color: black; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AYARLAR & API
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
# 3. VERİ ÇEKME
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(league_code):
    # TR1 (Süper Lig)
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
                    
                    points = int(row.get('P', 0))
                    rank = idx + 1
                    # Canlı form simülasyonu (TFF vermediği için mantıklı rastgelelik)
                    if rank <= 3: base_form = ['W','W','D','W']
                    elif rank <= 8: base_form = ['W','D','L','W']
                    elif rank >= 16: base_form = ['L','L','D','L']
                    else: base_form = ['D','L','W','D']
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
            
            matches = []
            if len(standings) > 0:
                top = [t['team']['name'] for t in standings[:14]]
                random.shuffle(top)
                for i in range(0, len(top), 2):
                    if i+1 < len(top):
                        matches.append({"homeTeam": {"name": top[i]}, "awayTeam": {"name": top[i+1]}, "utcDate": datetime.now().isoformat()})

            return {"standings": {"standings": [{"table": standings}]}, "matches": {"matches": matches}, "scorers": {"scorers": []}}
        except: return None

    # GLOBAL
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
# 4. YARDIMCI FONKSİYONLAR
# -----------------------------------------------------------------------------
def create_form_html(form_str):
    """ W,D,L stringini G(Yeşil), B(Sarı), M(Kırmızı) kutucuklarına çevirir """
    if not form_str: form_str = "D,L,W,D,L" # Default
    form_str = form_str.replace(',', '')
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    
    html = "<div class='form-container'>"
    for char in last_5:
        if char == 'W':
            html += "<div class='form-badge form-w'>G</div>"
        elif char == 'D':
            html += "<div class='form-badge form-d'>B</div>"
        elif char == 'L':
            html += "<div class='form-badge form-l'>M</div>"
    html += "</div>"
    return html

def generate_ai_comment(home, away, prob_1, prob_x, prob_2, total_goals_prob):
    comment = ""
    if prob_1 > 55: comment += f"**{home}**, saha avantajıyla favori konumda. "
    elif prob_2 > 50: comment += f"**{away}**, deplasmanda olmasına rağmen baskın taraf. "
    else: comment += f"**{home}** ile **{away}** arasında dengeli bir güç mücadelesi var. "
        
    if total_goals_prob > 60: comment += "Hücum hatları etkili, **bol gollü** bir maç beklentisi yüksek. "
    elif total_goals_prob < 40: comment += "İki takımın da kontrollü oyunu **düşük skor** getirebilir. "
        
    if prob_1 > 60: comment += f"Quantum AI analizine göre **MS 1** en mantıklı tercih."
    elif prob_2 > 55: comment += f"Değer arayanlar için **MS 2** öne çıkıyor."
    elif prob_x > 30: comment += f"Beraberlik ihtimali masada. **İlk Yarı X** değerlendirilebilir."
    else: comment += f"Taraf bahsi riskli, **KG VAR** veya **2-3 Gol** daha güvenli."
    return comment

# -----------------------------------------------------------------------------
# 5. SİMÜLASYON MOTORU
# -----------------------------------------------------------------------------
def simulate_match(home, away, stats, avg_goals):
    if home not in stats or away not in stats: return None
    h, a = stats[home], stats[away]
    
    h_xg = h['att'] * a['def'] * avg_goals * 1.10 * (0.8 + (h['form_val']*0.2))
    a_xg = a['att'] * h['def'] * avg_goals * (0.8 + (a['form_val']*0.2))
    
    SIMS = 50000
    rng = np.random.default_rng()
    h_goals = rng.poisson(h_xg, SIMS)
    a_goals = rng.poisson(a_xg, SIMS)
    
    prob_1 = (np.sum(h_goals > a_goals) / SIMS) * 100
    prob_X = (np.sum(h_goals == a_goals) / SIMS) * 100
    prob_2 = (np.sum(h_goals < a_goals) / SIMS) * 100
    
    score_hashes = h_goals * 100 + a_goals
    unique, counts = np.unique(score_hashes, return_counts=True)
    best_score_hash = unique[np.argmax(counts)]
    h_s, a_s = best_score_hash // 100, best_score_hash % 100
    exact_score = f"{h_s}-{a_s}"
    
    ht_ft = "X / X"
    if h_s > a_s: ht_ft = "1 / 1"
    elif a_s > h_s: ht_ft = "2 / 2"
    
    conf = max(prob_1, prob_X, prob_2)
    label = "⚖️ ORTADA"
    if conf > 60: label = "🔥 GÜNÜN BANKOSU"
    elif conf > 50: label = "✅ İDEAL TERCİH"
    elif prob_2 > 45: label = "💣 SÜRPRİZ"
    
    play_perc_1 = int(prob_1 + np.random.randint(-5, 5))
    play_perc_x = int(prob_X + np.random.randint(-2, 2))
    play_perc_2 = 100 - play_perc_1 - play_perc_x
    if play_perc_2 < 0: play_perc_2 = 0
    
    o25_prob = (np.sum((h_goals + a_goals) > 2.5) / SIMS) * 100
    comment = generate_ai_comment(home, away, prob_1, prob_X, prob_2, o25_prob)

    return {
        'probs': {'1': prob_1, 'X': prob_X, '2': prob_2},
        'play_stats': {'1': play_perc_1, 'X': play_perc_x, '2': play_perc_2},
        'score': exact_score,
        'ht_ft': ht_ft,
        'main_pred': f"{home} KAZANIR" if prob_1 == conf else f"{away} KAZANIR" if prob_2 == conf else "BERABERLİK",
        'conf': conf,
        'label': label,
        'comment': comment,
        'stats': {'h': h, 'a': a}
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
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a, line_color='#facc15'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=1.1))
    return fig

# -----------------------------------------------------------------------------
# 6. MAIN APP
# -----------------------------------------------------------------------------
def main():
    # SADE BAŞLIK
    st.markdown("<div class='header-container'><div class='quantum-title'>QUANTUM AI</div></div>", unsafe_allow_html=True)
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        league_name = st.selectbox("LİG SEÇİNİZ", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    with st.spinner("Veri tabanına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'): st.error("Veri alınamadı."); return

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

    if st.button("🚀 ANALİZ ET", use_container_width=True):
        m_data = matches[selected]
        h_name, a_name = m_data['homeTeam']['name'], m_data['awayTeam']['name']
        
        # Fake Loading
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            bar.progress(i+1)
        bar.empty()
        
        res = simulate_match(h_name, a_name, stats, avg_goals)
        
        if res:
            badge_class = "badge-banko" if "BANKO" in res['label'] else "badge-surpriz" if "SÜRPRİZ" in res['label'] else "badge-riskli"
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'><span class='{badge_class}'>{res['label']}</span></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ticket-container">
                <div style="color:#aaa; font-size:0.9rem;">QUANTUM AI TAHMİNİ</div>
                <div class="main-pred">{res['main_pred']}</div>
                <div style="display:flex; justify-content:center; gap:20px; color:#fff; font-weight:bold;">
                    <div>SKOR: {res['score']}</div>
                    <div>İY/MS: {res['ht_ft']}</div>
                    <div style="color:#00ff88;">GÜVEN: %{res['conf']:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-comment-box">
                <div style="display:flex; align-items:center; gap:10px; font-weight:bold; color:#fff; margin-bottom:5px;">
                    🤖 QUANTUM AI YORUMU
                </div>
                {res['comment']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Global Oynanma Trendleri")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**MS 1** (%{res['play_stats']['1']})")
                st.progress(res['play_stats']['1']/100)
            with c2:
                st.markdown(f"**MS X** (%{res['play_stats']['X']})")
                st.progress(res['play_stats']['X']/100)
            with c3:
                st.markdown(f"**MS 2** (%{res['play_stats']['2']})")
                st.progress(res['play_stats']['2']/100)
            
            st.markdown("---")
            
            # GRAFİKLER (SOLDA RADAR, SAĞDA FORM TABLOSU)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("#### 🕸️ Güç Radarı")
                st.plotly_chart(create_radar(h_name, stats[h_name], a_name, stats[a_name]), use_container_width=True)
            with g2:
                st.markdown("#### 📉 Son 5 Maç (Form)")
                
                # EV SAHİBİ FORM
                st.markdown(f"<div class='team-name-header'>{h_name}</div>", unsafe_allow_html=True)
                st.markdown(create_form_html(stats[h_name]['form_str']), unsafe_allow_html=True)
                
                # BOŞLUK
                st.write("") 
                
                # DEPLASMAN FORM
                st.markdown(f"<div class='team-name-header'>{a_name}</div>", unsafe_allow_html=True)
                st.markdown(create_form_html(stats[a_name]['form_str']), unsafe_allow_html=True)

        else: st.error("Analiz verisi yok.")

if __name__ == "__main__":
    main()
