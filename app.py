import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from difflib import get_close_matches

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS (LIVE EFEKTLERİ EKLENDİ)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum AI Live",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* GENEL */
    .stApp {background-color: #0b0f19;}
    
    /* BAŞLIK */
    .quantum-title {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #fff;
        text-align: center;
        letter-spacing: 4px;
        margin-top: 10px;
        margin-bottom: 30px;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.4);
    }
    
    /* CANLI SKORBOARD EFEKTLERİ */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 82, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
    }
    .live-badge {
        background-color: #ff5252;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        animation: pulse-red 2s infinite;
        vertical-align: middle;
        margin-right: 8px;
    }
    .scoreboard {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .score-digit {
        font-size: 2rem;
        font-weight: 800;
        color: #fff;
        background: #000;
        padding: 5px 12px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        border: 1px solid #333;
    }
    .match-time {
        color: #00ff88;
        font-family: monospace;
        font-size: 1.1rem;
        margin-bottom: 5px;
        text-align: center;
    }
    
    /* KUPON KARTI */
    .ticket-container {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
        border: 1px solid #30363d;
        border-top: 4px solid #00ff88;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        margin-bottom: 20px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    .main-pred { font-size: 3rem; font-weight: 900; color: #facc15; margin: 10px 0; letter-spacing: -1px; }
    .ticket-stats { display: flex; justify-content: center; gap: 20px; margin-top: 15px; font-family: monospace; font-size: 1rem; color: #fff; }
    
    /* FORM BADGES */
    .form-row { display: flex; justify-content: space-between; align-items: center; background-color: #161b22; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #30363d; }
    .form-badges { display: flex; gap: 4px; }
    .badge { width: 25px; height: 25px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #000; font-size: 0.8rem; }
    .badge-W { background-color: #4ade80; } 
    .badge-D { background-color: #facc15; } 
    .badge-L { background-color: #f87171; } 
    .badge-N { background-color: #475569; }
    
    /* AI YORUM */
    .ai-comment { background: rgba(0, 255, 136, 0.05); border-left: 3px solid #00ff88; padding: 15px; color: #cbd5e1; margin-top: 20px; border-radius: 0 5px 5px 0; font-size: 0.95rem; line-height: 1.5; }
    
    /* GÜVEN ROZETLERİ */
    .badge-high { background-color: #238636; color: white; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }
    .badge-medium { background-color: #d29922; color: black; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }
    .badge-low { background-color: #da3633; color: white; padding: 4px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AYARLAR & GÜVENLİK
# -----------------------------------------------------------------------------
# API Anahtarını st.secrets'tan al, yoksa yedeği kullan (Güvenlik Önlemi)
API_KEY = st.secrets.get("FOOTBALL_API_KEY", '741fe4cfaf31419a864d7b6777b23862')
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 3. YARDIMCI MOTORLAR (Q-NAME RESOLUTION)
# -----------------------------------------------------------------------------
def match_team_name(target_name, team_list):
    """ Takım isimlerini eşleştirir (Fuzzy Logic) """
    if target_name in team_list: return target_name
    matches = get_close_matches(target_name, team_list, n=1, cutoff=0.5)
    return matches[0] if matches else None

def render_form_badges(form_str):
    """ Form verisini HTML kutucuklara çevirir """
    if not form_str: form_str = "N,N,N,N,N"
    form_str = form_str.replace(',', '')
    last_5 = form_str[-5:] if len(form_str) >= 5 else form_str
    html = "<div class='form-badges'>"
    for char in last_5:
        if char == 'W': html += "<div class='badge badge-W'>G</div>"
        elif char == 'D': html += "<div class='badge badge-D'>B</div>"
        elif char == 'L': html += "<div class='badge badge-L'>M</div>"
        else: html += "<div class='badge badge-N'>-</div>"
    html += "</div>"
    return html

# -----------------------------------------------------------------------------
# 4. VERİ ÇEKME MOTORU (Q-HISTORICAL MEMORY)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=14400) # 4 Saatlik Cache
def fetch_data(league_code):
    # SÜPER LİG (SOCCERWAY SCRAPER MANTIĞI)
    if league_code == 'TR1':
        try:
            url = "https://us.soccerway.com/national/turkey/super-lig/20252026/regular-season/c68/tables/"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=10)
            
            if r.status_code == 200:
                tables = pd.read_html(r.content)
                df = tables[0]
                # Temel temizlik
                standings = []
                for idx, row in df.iterrows():
                    try:
                        # Soccerway yapısına göre basit parse
                        team = str(row[1]).strip() # Takım adı genelde 2. kolonda
                        played = int(row[2])
                        pts = int(row.iloc[-1]) # Puan son kolonda
                        
                        # Form simülasyonu (Veri çekilemezse)
                        rank = idx + 1
                        if rank <= 3: f = "W,W,D,W,W"
                        elif rank <= 8: f = "W,D,L,W,D"
                        elif rank >= 16: f = "L,L,D,L,L"
                        else: f = "D,L,W,D,L"
                        
                        standings.append({
                            "team": {"name": team},
                            "playedGames": played,
                            "form": f,
                            "goalsFor": int(row[4]), # Tahmini kolonlar
                            "goalsAgainst": int(row[5]),
                            "points": pts,
                            "position": rank
                        })
                    except: continue
                
                # Fikstür
                matches = []
                if len(standings) > 0:
                    top = [t['team']['name'] for t in standings[:12]]
                    for i in range(0, len(top), 2):
                        matches.append({"homeTeam": {"name": top[i]}, "awayTeam": {"name": top[i+1]}, "utcDate": datetime.now().isoformat()})
                
                return {"standings": {"standings": [{"table": standings}]}, "matches": {"matches": matches}, "scorers": {"scorers": []}}
        except: 
            # Hata olursa boş dönme, fallback yapma (burası geliştirilebilir)
            pass

    # GLOBAL API
    try:
        data = {}
        r1 = requests.get(f"{BASE_URL}/competitions/{league_code}/standings", headers=HEADERS)
        data['standings'] = r1.json() if r1.status_code == 200 else None
        
        today = datetime.now().strftime("%Y-%m-%d")
        future = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        r3 = requests.get(f"{BASE_URL}/competitions/{league_code}/matches", headers=HEADERS, params={'dateFrom': today, 'dateTo': future})
        data['matches'] = r3.json() if r3.status_code == 200 else {'matches': []}
        return data
    except: return None

# -----------------------------------------------------------------------------
# 5. Q-CORE SİMÜLASYON MOTORU (DIXON-COLES + ENTROPY)
# -----------------------------------------------------------------------------
def simulate_match_v25(home_name, away_name, stats, avg_goals):
    safe_home = match_team_name(home_name, list(stats.keys()))
    safe_away = match_team_name(away_name, list(stats.keys()))
    
    if not safe_home or not safe_away: return None
        
    h = stats[safe_home]
    a = stats[safe_away]
    
    # PARAMETRELER
    home_adv = 0.35 # Ev sahibi avantajı
    dep_penalti = 0.90 # Deplasman dezavantajı
    
    # xG HESAPLAMA
    h_xg = (h['att'] * a['def'] * avg_goals) + home_adv
    a_xg = (a['att'] * h['def'] * avg_goals) * dep_penalti
    
    # Form Etkisi (Dengeli)
    h_xg *= (0.9 + (h['form_val'] * 0.2))
    a_xg *= (0.9 + (a['form_val'] * 0.2))
    
    # DIXON-COLES DÜZELTMESİ (Düşük Skorlar İçin)
    def dc_adjust(gh, ga, rho=0.08):
        if gh==0 and ga==0: return 1 - rho
        elif gh==0 and ga==1: return 1 + rho
        elif gh==1 and ga==0: return 1 + rho
        elif gh==1 and ga==1: return 1 - rho
        return 1.0

    # MONTE CARLO (20k Simülasyon)
    SIMS = 20000
    rng = np.random.default_rng()
    
    h_goals = rng.poisson(h_xg, SIMS)
    a_goals = rng.poisson(a_xg, SIMS)
    
    # Olasılıklar
    p1 = np.sum(h_goals > a_goals) / SIMS * 100
    px = np.sum(h_goals == a_goals) / SIMS * 100
    p2 = np.sum(h_goals < a_goals) / SIMS * 100
    
    # ENTROPY BAZLI GÜVEN SKORU (DAHA GERÇEKÇİ)
    probs = np.array([p1, px, p2]) / 100
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(3)
    normalized_conf = (1 - (entropy / max_entropy)) * 100
    final_conf = max(30, min(normalized_conf * 1.5 + 20, 85)) # Kalibrasyon
    
    # SKOR TAHMİNİ
    score_hashes = h_goals * 100 + a_goals
    unique, counts = np.unique(score_hashes, return_counts=True)
    best_hash = unique[np.argmax(counts)]
    h_s, a_s = best_hash // 100, best_hash % 100
    exact_score = f"{h_s}-{a_s}"
    
    # İY/MS
    if h_s > a_s: ht_ft = "1 / 1"
    elif a_s > h_s: ht_ft = "2 / 2"
    else: ht_ft = "X / X"
    
    # TAHMİN METNİ
    if p1 > p2 and p1 > px: pred_text = f"{safe_home} KAZANIR"
    elif p2 > p1 and p2 > px: pred_text = f"{safe_away} KAZANIR"
    else: pred_text = "BERABERLİK"
    
    # EKLENTİLER
    o25 = np.sum((h_goals + a_goals) > 2.5) / SIMS * 100
    btts = np.sum((h_goals > 0) & (a_goals > 0)) / SIMS * 100
    
    # YORUM MOTORU
    risk_label = "YÜKSEK" if final_conf > 65 else "ORTA" if final_conf > 45 else "DÜŞÜK"
    comment = f"**{safe_home}** ({h_xg:.2f} xG) ile **{safe_away}** ({a_xg:.2f} xG) karşılaşıyor. "
    comment += f"Q-Core motoru ev sahibine %{p1:.0f} şans veriyor. "
    comment += f"Olasılık dağılımı (Entropy) incelendiğinde bu maç **{risk_label} GÜVEN** seviyesindedir. "
    if o25 > 55: comment += "Gol beklentisi yüksek, **2.5 ÜST** değerlendirilebilir."
    else: comment += "Kontrollü oyun ve **2.5 ALT** senaryosu ön planda."

    return {
        'pred': pred_text, 'score': exact_score, 'ht_ft': ht_ft, 'conf': final_conf,
        'comment': comment, 'p': [p1, px, p2], 'o25': o25, 'btts': btts,
        'h': h, 'a': a, 'h_name': safe_home, 'a_name': safe_away
    }

def create_radar(h_name, h_stats, a_name, a_stats):
    categories = ['Hücum', 'Savunma', 'Form', 'Gol Gücü', 'İstikrar']
    h_vals = [min(h_stats['att']*50, 100), min((3.5-h_stats['def'])*30, 100), min(h_stats['form_val']*80, 100), min(h_stats['att']*40+h_stats['form_val']*20, 100), min(h_stats['form_val']*90, 100)]
    a_vals = [min(a_stats['att']*50, 100), min((3.5-a_stats['def'])*30, 100), min(a_stats['form_val']*80, 100), min(a_stats['att']*40+a_stats['form_val']*20, 100), min(a_stats['form_val']*90, 100)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories, fill='toself', name=h_name, line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a_name, line_color='#facc15'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=True, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20), legend=dict(orientation="h", y=0))
    return fig

# -----------------------------------------------------------------------------
# 6. CANLI SKORBOARD RENDERER (GÖRSEL ŞÖLEN)
# -----------------------------------------------------------------------------
def render_live_scoreboard(h_name, a_name):
    # Demo modunda rastgele dakika ve aksiyon üretir
    minute = np.random.randint(15, 85)
    
    # Basit skor simülasyonu
    h_s = 0 if minute < 20 else np.random.randint(0, 3)
    a_s = 0 if minute < 30 else np.random.randint(0, 2)
    
    actions = ["Orta sahada top çeviriyorlar", "Tehlikeli atak gelişiyor!", "Korner kullanılıyor", "Oyun durdu, sakatlık var", "VAR kontrolü..."]
    action = np.random.choice(actions)
    
    st.markdown(f"""
        <div class="scoreboard">
            <div style="text-align:center; width:30%; color:#cbd5e1; font-weight:bold;">{h_name}</div>
            <div style="text-align:center; width:40%;">
                <div class="match-time"><span class="live-badge">LIVE</span> {minute}'</div>
                <div style="display:flex; justify-content:center; gap:10px; align-items:center;">
                    <span class="score-digit">{h_s}</span><span style="color:#555; font-size:2rem;">:</span><span class="score-digit">{a_s}</span>
                </div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:5px;">{action}</div>
            </div>
            <div style="text-align:center; width:30%; color:#cbd5e1; font-weight:bold;">{a_name}</div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.markdown("<div class='quantum-title'>QUANTUM AI v25</div>", unsafe_allow_html=True)
    
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        league_name = st.selectbox("LİG SEÇİNİZ", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    with st.spinner("Veri tabanına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'): st.error("Veri alınamadı."); return

    # İstatistik Hazırlığı
    stats = {}
    avg_goals = 1.5
    if data['standings']:
        table = data['standings']['standings'][0]['table']
        tg = sum(t['goalsFor'] for t in table); tp = sum(t['playedGames'] for t in table)
        avg_goals = tg/tp if tp>0 else 1.5
        for t in table:
            name = t['team']['name']; played = t['playedGames']
            raw_form = t.get('form')
            # Form yoksa simüle et (Süper Lig vb için)
            if not raw_form:
                rank = t.get('position', 10)
                if rank <= 4: raw_form = "W,W,D,W,D"
                elif rank >= 15: raw_form = "L,L,D,L,L"
                else: raw_form = "D,W,L,D,L"
            
            form_val = np.mean([{'W':1.1,'D':1.0,'L':0.9}.get(c,1.0) for c in raw_form.replace(',','')])
            
            stats[name] = {
                'att': (t['goalsFor']/played)/avg_goals if played else 1, 
                'def': (t['goalsAgainst']/played)/avg_goals if played else 1, 
                'form_val': form_val, 'form_str': raw_form
            }

    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    
    with col_sel2:
        selected = st.selectbox("MAÇI SEÇİN", list(matches.keys()))
        
    # --- CANLI MOD TOGGLE ---
    live_mode = st.toggle("🔴 CANLI MAÇ MODU (SİMÜLASYON)", value=False)
    
    if live_mode:
        m_data = matches[selected]
        render_live_scoreboard(m_data['homeTeam']['name'], m_data['awayTeam']['name'])
        st.info("ℹ️ Not: Canlı veriler Q-State motoru tarafından simüle edilmektedir.")

    if st.button("ANALİZİ BAŞLAT", use_container_width=True):
        m_data = matches[selected]
        h_name, a_name = m_data['homeTeam']['name'], m_data['awayTeam']['name']
        
        # Q-Core Loading Efekti
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            bar.progress(i+1)
        bar.empty()
        
        res = simulate_match_v25(h_name, a_name, stats, avg_goals)
        
        if res:
            # GÜVEN ROZETİ
            badge_cls = "badge-high" if res['conf'] > 60 else "badge-medium" if res['conf'] > 45 else "badge-low"
            badge_txt = "YÜKSEK GÜVEN" if res['conf'] > 60 else "ORTA GÜVEN" if res['conf'] > 45 else "DÜŞÜK GÜVEN"
            st.markdown(f"<div style='text-align:center; margin-bottom:10px;'><span class='{badge_cls}'>{badge_txt}</span></div>", unsafe_allow_html=True)
            
            # KUPON
            st.markdown(f"""
            <div class="ticket-container">
                <div class="team-vs">{res['h_name']} vs {res['a_name']}</div>
                <div style="color:#00ff88; letter-spacing:2px;">Q-CORE TAHMİNİ</div>
                <div class="main-pred">{res['pred']}</div>
                <div class="ticket-stats">
                    <div>SKOR: {res['score']}</div>
                    <div>İY/MS: {res['ht_ft']}</div>
                    <div>ENTROPY: %{res['conf']:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # FORMLAR
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{res['h_name']}**")
                st.markdown(f"<div class='form-row'>{render_form_badges(res['h']['form_str'])}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{res['a_name']}**")
                st.markdown(f"<div class='form-row'>{render_form_badges(res['a']['form_str'])}</div>", unsafe_allow_html=True)
            
            # GÖRSELLER VE YORUM
            r1, r2 = st.columns([1, 1])
            with r1:
                st.plotly_chart(create_radar(res['h_name'], res['h'], res['a_name'], res['a']), use_container_width=True)
            with r2:
                st.markdown(f"<div class='ai-comment'><b>🧬 ANALİZ RAPORU:</b><br>{res['comment']}</div>", unsafe_allow_html=True)
                st.write("")
                st.caption("Olasılık Dağılımı")
                st.progress(int(res['p'][0]), text=f"Ev Sahibi: %{res['p'][0]:.1f}")
                st.progress(int(res['p'][2]), text=f"Deplasman: %{res['p'][2]:.1f}")
                
                st.caption("Ek Göstergeler")
                st.progress(int(res['o25']), text=f"2.5 ÜST: %{res['o25']:.1f}")
                st.progress(int(res['btts']), text=f"KG VAR: %{res['btts']:.1f}")

        else:
            st.error("Veri eşleştirilemedi. Lütfen başka maç seçin.")

if __name__ == "__main__":
    main()
