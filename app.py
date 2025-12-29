import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
# İsim eşleştirme için difflib kullanacağız (Python'un kendi kütüphanesidir, ekstra kuruluma gerek yok)
from difflib import get_close_matches 

# -----------------------------------------------------------------------------
# 1. AYARLAR & CSS (MİNİMALİST & PROFESYONEL)
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
    
    /* SADE BAŞLIK */
    .quantum-title {
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #fff;
        text-align: center;
        letter-spacing: 4px;
        margin-top: 20px;
        margin-bottom: 40px;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    /* KUPON KARTI */
    .ticket-container {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
        border: 1px solid #30363d;
        border-top: 4px solid #00ff88;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        margin-bottom: 30px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .team-vs { font-size: 1.2rem; color: #cbd5e1; margin-bottom: 15px; }
    .main-pred { font-size: 3.5rem; font-weight: 900; color: #facc15; margin: 10px 0; letter-spacing: -1px; }
    
    .ticket-stats {
        display: flex; justify-content: center; gap: 30px; margin-top: 20px;
        font-family: monospace; font-size: 1.1rem; color: #fff;
    }
    
    /* FORM KUTUCUKLARI (G-B-M) */
    .form-row {
        display: flex; justify-content: space-between; align-items: center;
        background-color: #161b22; padding: 15px; border-radius: 10px; margin-bottom: 10px;
        border: 1px solid #30363d;
    }
    .form-badges { display: flex; gap: 5px; }
    .badge {
        width: 30px; height: 30px; border-radius: 4px;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; color: #000; font-size: 0.9rem;
    }
    .badge-W { background-color: #4ade80; } /* YEŞİL */
    .badge-D { background-color: #facc15; } /* SARI */
    .badge-L { background-color: #f87171; } /* KIRMIZI */
    .badge-N { background-color: #475569; } /* GRİ (Veri Yok) */
    
    /* DİĞER */
    .ai-comment {
        background: rgba(0, 255, 136, 0.05); border-left: 3px solid #00ff88;
        padding: 15px; color: #cbd5e1; margin-top: 20px; border-radius: 0 5px 5px 0;
    }
    .share-box {
        background-color: #0d1117; padding: 20px; border-radius: 10px;
        border: 1px dashed #30363d; margin-top: 30px; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. AYARLAR & API
# -----------------------------------------------------------------------------
# GÜVENLİK NOTU: Gerçek projede bunu st.secrets içine almalısın.
API_KEY = '741fe4cfaf31419a864d7b6777b23862'
HEADERS = {'X-Auth-Token': API_KEY}
BASE_URL = 'https://api.football-data.org/v4'

LEAGUES = {
    '🇬🇧 Premier League': 'PL', '🇹🇷 Süper Lig': 'TR1', '🇪🇸 La Liga': 'PD',
    '🇩🇪 Bundesliga': 'BL1', '🇮🇹 Serie A': 'SA', '🇫🇷 Ligue 1': 'FL1',
    '🇳🇱 Eredivisie': 'DED', '🇪🇺 Şampiyonlar Ligi': 'CL'
}

# -----------------------------------------------------------------------------
# 3. AKILLI İSİM EŞLEŞTİRİCİ (CRASH ÖNLEYİCİ)
# -----------------------------------------------------------------------------
def match_team_name(target_name, team_list):
    """ API'den gelen isimle istatistiklerdeki ismi eşleştirir """
    if target_name in team_list:
        return target_name
    
    # En yakın eşleşmeyi bul
    matches = get_close_matches(target_name, team_list, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None

# -----------------------------------------------------------------------------
# 4. VERİ ÇEKME MOTORU
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_data(league_code):
    # --- TÜRKİYE SÜPER LİG (MANUEL SCRAPER) ---
    if league_code == 'TR1':
        try:
            url = "https://www.tff.org/default.aspx?pageID=198"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=10)
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
                    
                    # TFF sitesinde form verisi yok, puan durumuna göre 'tahmini' form üretiyoruz
                    # Ama bunu her açılışta sabit tutmak için random seed kullanmıyoruz, basit mantık:
                    rank = idx + 1
                    if rank <= 3: form_str = "W,W,D,W,W"
                    elif rank <= 8: form_str = "W,D,L,W,D"
                    elif rank >= 16: form_str = "L,L,D,L,L"
                    else: form_str = "D,L,W,D,L"

                    standings.append({
                        "team": {"name": team_name},
                        "playedGames": int(row.get('O', 0)),
                        "form": form_str, 
                        "goalsFor": int(row.get('A', 0)),
                        "goalsAgainst": int(row.get('Y', 0)),
                        "points": int(row.get('P', 0)),
                        "position": rank
                    })
                except: continue
            
            # Fikstür: İlk 10 takımı kendi arasında eşleştir (Demo için)
            matches = []
            if len(standings) > 0:
                top_teams = [t['team']['name'] for t in standings[:12]]
                # Rastgeleliği kaldırdık, her zaman aynı eşleşmeler çıksın ki stabil olsun
                for i in range(0, len(top_teams), 2):
                    matches.append({"homeTeam": {"name": top_teams[i]}, "awayTeam": {"name": top_teams[i+1]}, "utcDate": datetime.now().isoformat()})

            return {"standings": {"standings": [{"table": standings}]}, "matches": {"matches": matches}, "scorers": {"scorers": []}}
        except: return None

    # --- GLOBAL LİGLER (API) ---
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
# 5. İSTATİSTİK VE FORM GÖRSELLEŞTİRME
# -----------------------------------------------------------------------------
def render_form_badges(form_str):
    """ API form stringini (W,D,L) alıp HTML kutucuklara çevirir """
    if not form_str: form_str = "N,N,N,N,N"
    form_str = form_str.replace(',', '')
    # Son 5 maçı al
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
# 6. QUANTUM SİMÜLASYON MOTORU (GERÇEKÇİ MOD)
# -----------------------------------------------------------------------------
def simulate_match_realism(home_name, away_name, stats, avg_goals):
    # İsimleri güvenli şekilde eşleştir
    safe_home = match_team_name(home_name, stats.keys())
    safe_away = match_team_name(away_name, stats.keys())
    
    if not safe_home or not safe_away:
        return None
        
    h = stats[safe_home]
    a = stats[safe_away]
    
    # 1. HOME ADVANTAGE (Ev Sahibi Avantajı)
    # Futbolda ev sahibi ortalama +0.3 ile +0.4 gol avantajına sahiptir.
    home_advantage = 0.35 
    
    # xG Hesaplama (Daha gerçekçi formül)
    h_xg = (h['att'] * a['def'] * avg_goals) + home_advantage
    a_xg = (a['att'] * h['def'] * avg_goals)
    
    # Form Etkisi (Sonuçları %10-15 saptırır)
    h_xg *= (0.9 + (h['form_val'] * 0.2))
    a_xg *= (0.9 + (a['form_val'] * 0.2))
    
    # MONTE CARLO SİMÜLASYONU (20.000 Maç yeterli ve hızlıdır)
    SIMS = 20000
    rng = np.random.default_rng()
    
    h_goals = rng.poisson(h_xg, SIMS)
    a_goals = rng.poisson(a_xg, SIMS)
    
    # Olasılıklar
    prob_1 = (np.sum(h_goals > a_goals) / SIMS) * 100
    prob_X = (np.sum(h_goals == a_goals) / SIMS) * 100
    prob_2 = (np.sum(h_goals < a_goals) / SIMS) * 100
    
    # En Olası Skor
    score_hashes = h_goals * 100 + a_goals
    unique, counts = np.unique(score_hashes, return_counts=True)
    best_idx = np.argmax(counts)
    best_hash = unique[best_idx]
    h_s, a_s = best_hash // 100, best_hash % 100
    exact_score = f"{h_s}-{a_s}"
    
    # İY/MS MANTIĞI (Skora göre tutarlı)
    # Skor 0-0 ise İY X olur.
    # Skor 2-1 ise İY X veya 1 olabilir. Biz en olası senaryoyu seçiyoruz.
    if h_s > a_s: 
        ht_ft = "1 / 1"
    elif a_s > h_s: 
        ht_ft = "2 / 2"
    else: 
        ht_ft = "X / X"
        
    # Güven Skoru
    conf = max(prob_1, prob_X, prob_2)
    
    # Ana Tahmin Yazısı
    if prob_1 > prob_2 and prob_1 > prob_X: main_text = f"{home_name} KAZANIR"
    elif prob_2 > prob_1 and prob_2 > prob_X: main_text = f"{away_name} KAZANIR"
    else: main_text = "BERABERLİK"
    
    # Yorum Üretimi
    comment = f"Ev sahibi **{home_name}**, Quantum simülasyonlarında maçların **%{prob_1:.0f}**'ini kazandı. "
    if conf > 60: comment += "İstatistiksel olarak **güçlü bir favori**."
    elif abs(prob_1 - prob_2) < 10: comment += "Maç ortada görünüyor, **taraf bahsinden kaçınılmalı**."
    else: comment += "Rakip takımın sürpriz potansiyeli var."
    
    if (h_xg + a_xg) > 2.6: comment += " Gol beklentisi (xG) yüksek, **2.5 ÜST** ihtimali güçlü."
    else: comment += " Düşük tempolu, taktiksel bir maç bekleniyor (**2.5 ALT**)."

    return {
        'pred': main_text,
        'score': exact_score,
        'ht_ft': ht_ft,
        'conf': conf,
        'comment': comment,
        'stats': {'h': h, 'a': a, 'h_name': safe_home, 'a_name': safe_away},
        'raw_probs': [prob_1, prob_X, prob_2]
    }

def create_radar(h_name, h_stats, a_name, a_stats):
    categories = ['Hücum', 'Savunma', 'Form', 'Gol Gücü', 'İstikrar']
    
    # Verileri 0-100 arasına çek
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
    fig.add_trace(go.Scatterpolar(r=h_vals, theta=categories, fill='toself', name=h_name, line_color='#00ff88'))
    fig.add_trace(go.Scatterpolar(r=a_vals, theta=categories, fill='toself', name=a_name, line_color='#facc15'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#334155'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(orientation="h", y=0, x=0.3)
    )
    return fig

# -----------------------------------------------------------------------------
# 7. MAIN APP LOOP
# -----------------------------------------------------------------------------
def main():
    st.markdown("<div class='quantum-title'>QUANTUM AI</div>", unsafe_allow_html=True)
    
    # 1. Lig Seçimi
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        league_name = st.selectbox("LİG SEÇİNİZ", list(LEAGUES.keys()))
    league_code = LEAGUES[league_name]
    
    # 2. Veri Çekme
    with st.spinner("Veri tabanına bağlanılıyor..."):
        data = fetch_data(league_code)
    
    if not data or not data.get('matches'):
        st.error("Bu lig için şu an veri alınamıyor veya maç yok.")
        return

    # 3. İstatistikleri İşle
    stats = {}
    avg_goals = 1.5
    if data['standings']:
        table = data['standings']['standings'][0]['table']
        tg = sum(t['goalsFor'] for t in table); tp = sum(t['playedGames'] for t in table)
        avg_goals = tg/tp if tp>0 else 1.5
        for t in table:
            name = t['team']['name']; played = t['playedGames']
            # Form verisini al
            raw_form = t.get('form', 'D,L,D,L,D')
            # Formu sayısal değere çevir (1.0 = Nötr)
            form_val = 1.0
            if raw_form:
                score = sum({'W':1.1, 'D':1.0, 'L':0.9}.get(c, 1.0) for c in raw_form.replace(',',''))
                form_val = score/len(raw_form.replace(',',''))
            
            stats[name] = {
                'att': (t['goalsFor']/played)/avg_goals if played>0 else 1, 
                'def': (t['goalsAgainst']/played)/avg_goals if played>0 else 1, 
                'form_val': form_val, 
                'form_str': raw_form
            }

    # 4. Maç Listesi
    matches = {f"{m['homeTeam']['name']} - {m['awayTeam']['name']}": m for m in data['matches']['matches'] if 'homeTeam' in m}
    
    with col_sel2:
        selected = st.selectbox("MAÇI SEÇİN", list(matches.keys()))

    # 5. Analiz Butonu
    if st.button("SİMÜLASYONU BAŞLAT", use_container_width=True):
        m_data = matches[selected]
        h_name_api = m_data['homeTeam']['name']
        a_name_api = m_data['awayTeam']['name']
        
        # Yükleniyor efekti
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            bar.progress(i+1)
        bar.empty()
        
        # Simülasyonu Çalıştır
        res = simulate_match_realism(h_name_api, a_name_api, stats, avg_goals)
        
        if res:
            # --- SONUÇ EKRANI ---
            
            # KUPON KARTI
            st.markdown(f"""
            <div class="ticket-container">
                <div class="team-vs">{res['stats']['h_name']} vs {res['stats']['a_name']}</div>
                <div style="color:#00ff88; letter-spacing:2px;">QUANTUM TAHMİNİ</div>
                <div class="main-pred">{res['pred']}</div>
                <div class="ticket-stats">
                    <div>SKOR: {res['score']}</div>
                    <div>İY/MS: {res['ht_ft']}</div>
                    <div>GÜVEN: %{res['conf']:.0f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # TAKIM FORMLARI (G-B-M)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{res['stats']['h_name']}** (Ev)")
                st.markdown(f"""
                <div class="form-row">
                    <div>Son 5 Maç</div>
                    {render_form_badges(res['stats']['h']['form_str'])}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{res['stats']['a_name']}** (Dep)")
                st.markdown(f"""
                <div class="form-row">
                    <div>Son 5 Maç</div>
                    {render_form_badges(res['stats']['a']['form_str'])}
                </div>
                """, unsafe_allow_html=True)
            
            # RADAR VE YORUM
            r1, r2 = st.columns([1, 1])
            with r1:
                st.plotly_chart(create_radar(res['stats']['h_name'], res['stats']['h'], res['stats']['a_name'], res['stats']['a']), use_container_width=True)
            with r2:
                st.markdown(f"<div class='ai-comment'><b>🤖 ANALİZ RAPORU:</b><br>{res['comment']}</div>", unsafe_allow_html=True)
                
                # Olasılık Barları
                st.write("")
                st.caption("Kazanma Olasılıkları")
                st.progress(int(res['raw_probs'][0]), text=f"Ev Sahibi: %{res['raw_probs'][0]:.1f}")
                st.progress(int(res['raw_probs'][2]), text=f"Deplasman: %{res['raw_probs'][2]:.1f}")

            # PAYLAŞIM ALANI
            st.markdown("""<div class='share-box'>
            <p style='color:#aaa'>📸 Ekran görüntüsü alıp paylaşabilirsin.</p>
            </div>""", unsafe_allow_html=True)

        else:
            st.error("Takım verileri eşleştirilemedi. Lütfen başka bir maç deneyin.")

if __name__ == "__main__":
    main()
