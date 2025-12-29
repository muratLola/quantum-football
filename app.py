import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# -----------------------------------------------------------------------------
# 1. ELITE QUANT CONFIG (v40.1 Final Masterpiece)
# -----------------------------------------------------------------------------
CONFIG = {
    "BASE_SIM": 20000, 
    "MAX_SIM": 35000, 
    "HOME_ADV": {"PL": 0.40, "TR1": 0.45, "PD": 0.38, "BL1": 0.42, "SA": 0.32, "FL1": 0.34},
    "EDGE_THRESHOLD": 0.05, 
    "KELLY_FRAC": 0.25, # Fractional Kelly for maximum safety
    "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png"
}

st.set_page_config(page_title="Quantum Quant v40.1", page_icon="🏦", layout="wide")

# -----------------------------------------------------------------------------
# 2. BLOOMBERG TERMINAL UI DESIGN
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #020617; font-family: 'Inter', sans-serif;}
    
    .terminal-header {
        font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 900;
        text-align: center; color: #fff; margin-bottom: 20px;
        background: linear-gradient(135deg, #fbbf24, #d97706);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    .value-box {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981; border-radius: 16px; padding: 25px;
        text-align: center; box-shadow: 0 15px 35px rgba(16, 185, 129, 0.2);
    }
    
    .stats-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; text-align: center; }
    .label { font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .val { font-size: 1.8rem; font-weight: 800; color: #f8fafc; }
    
    .team-crest { width: 80px; height: 80px; object-fit: contain; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(255,255,255,0.1)); }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. QUANTUM ARCHITECT: THE DECISION CORE
# -----------------------------------------------------------------------------
class QuantumArchitect:
    def __init__(self, key):
        self.headers = {"X-Auth-Token": key}
        self.base_url = "https://api.football-data.org/v4"

    @st.cache_data(ttl=3600)
    def fetch_data(_self, league):
        try:
            stnd = requests.get(f"{_self.base_url}/competitions/{league}/standings", headers=_self.headers).json()
            fixt = requests.get(f"{_self.base_url}/competitions/{league}/matches", headers=_self.headers, 
                               params={"dateFrom": datetime.now().strftime("%Y-%m-%d"), 
                                       "dateTo": (datetime.now()+timedelta(days=21)).strftime("%Y-%m-%d")}).json()
            return stnd, fixt
        except: return None, None

    def analyze(self, h_id, a_id, stats, avg_g, odds, bankroll, league="PL"):
        hs, as_ = stats[h_id], stats[a_id]
        
        # Base xG + Power Factors
        h_xg = hs["att"] * as_["def"] * avg_g + (CONFIG["HOME_ADV"].get(league, 0.38) * 1.15)
        a_xg = as_["att"] * hs["def"] * avg_g * 0.90
        
        rng = np.random.default_rng()
        
        # Adaptive Deep Simulation
        # Önce varyansı ölç
        temp_hg = rng.poisson(h_xg, 5000)
        temp_ag = rng.poisson(a_xg, 5000)
        ent = -np.sum(np.unique((temp_hg > temp_ag), return_counts=True)[1]/5000 * np.log(np.unique((temp_hg > temp_ag), return_counts=True)[1]/5000 + 1e-9))
        
        sim_count = CONFIG["MAX_SIM"] if ent > 0.6 else CONFIG["BASE_SIM"]
        
        # Negative Binomial for overdispersion
        hg = rng.negative_binomial(n=6, p=6/(6+h_xg), size=sim_count)
        ag = rng.negative_binomial(n=6, p=6/(6+a_xg), size=sim_count)
        
        probs = [np.mean(hg > ag), np.mean(hg == ag), np.mean(hg < ag)]
        
        # Financial Quant Analysis (EV & Kelly)
        evs = [(probs[i] * odds[i]) - 1 for i in range(3)]
        best_idx = np.argmax(evs)
        edge = evs[best_idx]
        
        # Kelly Criterion Calculation
        b = odds[best_idx] - 1
        p = probs[best_idx]
        q = 1 - p
        kelly_perc = max(0, (b * p - q) / b) * CONFIG["KELLY_FRAC"]
        stake_amt = bankroll * kelly_perc

        # Clusters
        score_hash = hg * 100 + ag
        u, c = np.unique(score_hash, return_counts=True)
        top3 = [f"{u[i]//100}-{u[i]%100}" for i in np.argsort(c)[-3:][::-1]]

        return {
            "probs": [p*100 for p in probs], "evs": evs, "kelly": kelly_perc*100, 
            "stake": stake_amt, "edge": edge, "best_idx": best_idx, "top3": top3, 
            "h_name": hs["name"], "a_name": as_["name"], "h_crest": hs["crest"], "a_crest": as_["crest"],
            "implied": [100/o for o in odds]
        }

# -----------------------------------------------------------------------------
# 4. MAIN APP INTERFACE
# -----------------------------------------------------------------------------
def main():
    api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
    
    with st.sidebar:
        st.markdown("<h2 style='color:#fbbf24'>💰 BANKROLL MGT</h2>", unsafe_allow_html=True)
        if not api_key:
            api_key = st.text_input("QUANT AUTH KEY", type="password")
            if not api_key: st.info("Enter API key to access terminal."); st.stop()
        
        bankroll = st.number_input("Total Bankroll ($)", 10, 1000000, 1000)
        st.divider()
        st.subheader("Market Odds")
        o_h = st.number_input("Home Win Odds", 1.01, 50.0, 2.00)
        o_x = st.number_input("Draw Odds", 1.01, 50.0, 3.40)
        o_a = st.number_input("Away Win Odds", 1.01, 50.0, 3.80)

    st.markdown("<div class='terminal-header'>QUANTUM QUANT TERMINAL v40.1</div>", unsafe_allow_html=True)

    arc = QuantumArchitect(api_key)
    l_dict = {"Premier League": "PL", "Süper Lig": "TR1", "La Liga": "PD", "Bundesliga": "BL1", "Serie A": "SA"}
    
    col1, col2 = st.columns([1, 2])
    with col1: league = st.selectbox("Market", list(l_dict.keys()))
    
    stnd, matches = arc.fetch_data(l_dict[league])
    if not stnd: st.error("API link failed."); return

    stats = {}
    table = stnd["standings"][0]["table"]
    avg_g = sum(r["goalsFor"] for r in table) / sum(r["playedGames"] for r in table)
    for r in table:
        stats[r["team"]["id"]] = {
            "name": r["team"]["name"], "crest": r["team"].get("crest", CONFIG["DEFAULT_LOGO"]),
            "att": (r["goalsFor"]/r["playedGames"])/avg_g,
            "def": (r["goalsAgainst"]/r["playedGames"])/avg_g
        }

    fixt = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in matches["matches"] if m["status"] in ["SCHEDULED", "TIMED"]}
    if not fixt: st.warning("No upcoming data."); return
    with col2: selected = st.selectbox("Fixture Selection", list(fixt.keys()))
    
    if st.button("EXECUTE QUANT SIMULATION", use_container_width=True):
        m = fixt[selected]
        res = arc.analyze(m["homeTeam"]["id"], m["awayTeam"]["id"], stats, avg_g, [o_h, o_x, o_a], bankroll, league=l_dict[league])
        
        # 1. ELITE VALUE ALERT
        labels = ["HOME", "DRAW", "AWAY"]
        if res["edge"] > CONFIG["EDGE_THRESHOLD"]:
            st.markdown(f"""
            <div class="value-box">
                <h1 style="color:#10b981; margin:0;">💎 {labels[res['best_idx']]} VALUE: +{res['edge']*100:.1f}% EDGE</h1>
                <p style="font-size:1.2rem; color:#f1f5f9;">Kelly Stake: <b>{res['kelly']:.2f}%</b> | Recommended Bet: <b>${res['stake']:.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("NO EDGE DETECTED. EXPECTED VALUE IS NEGATIVE. PASS.")

        # 2. DATA GRID
        st.write("")
        g1, g2, g3 = st.columns(3)
        with g1: st.markdown(f"<div class='stats-card'><p class='label'>Most Likely</p><p class='val'>{res['top3'][0]}</p></div>", unsafe_allow_html=True)
        with g2: st.markdown(f"<div class='stats-card'><p class='label'>Home Win %</p><p class='val'>{res['probs'][0]:.1f}</p></div>", unsafe_allow_html=True)
        with g3: st.markdown(f"<div class='stats-card'><p class='label'>Away Win %</p><p class='val'>{res['probs'][2]:.1f}</p></div>", unsafe_allow_html=True)

        # 3. EDGE VISUALIZATION
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=res["probs"], name="AI Probability %", marker_color='#3b82f6'))
        fig.add_trace(go.Bar(x=labels, y=res["implied"], name="Market Probability %", marker_color='#64748b'))
        fig.update_layout(title="AI Model vs Market Implied Probability", barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

        # 4. LOGOS
        lc1, lc2 = st.columns(2)
        with lc1: st.markdown(f"<div style='text-align:center'><img src='{res['h_crest']}' class='team-crest'><br><b>{res['h_name']}</b></div>", unsafe_allow_html=True)
        with lc2: st.markdown(f"<div style='text-align:center'><img src='{res['a_crest']}' class='team-crest'><br><b>{res['a_name']}</b></div>", unsafe_allow_html=True)

    st.markdown("<div class='disclaimer'>Quantum v40.1 Final Gold | Hedge Fund Grade | Kelly Integrated</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
