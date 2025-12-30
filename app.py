import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# 1. INSTITUTIONAL CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "SYSTEM": {
        "SIM_COUNT_BASE": 20000,
        "CACHE_TTL": 3600,
        "LOOKAHEAD_DAYS": 30,
        "DEFAULT_LOGO": "https://cdn-icons-png.flaticon.com/512/53/53283.png"
    },
    "MODEL": {
        "HOME_ADV_BASE": 0.35,
        "AWAY_PENALTY": 0.88,
        "LEAGUE_WEIGHTS": {"PL": 0.38, "TR1": 0.45, "PD": 0.35, "BL1": 0.42, "SA": 0.32, "FL1": 0.34},
        "NB_SIZE_PARAM": 6.0,
        "LIVE_DECAY_EXP": 0.6,
        "URGENCY_BOOST": 1.35
    },
    "RISK": {
        "KELLY_FRACTION": 0.20,
        "EDGE_THRESHOLD": 0.04,
        "MAX_EXPOSURE": 0.15,
        "VAR_CONFIDENCE": 0.95
    }
}

st.set_page_config(page_title="Quantum Ultimate v48", page_icon="🏛️", layout="wide")

# -----------------------------------------------------------------------------
# 2. SESSION STATE
# -----------------------------------------------------------------------------
if 'mode' not in st.session_state: st.session_state.mode = "PRE_MATCH"
if 'portfolio' not in st.session_state: st.session_state.portfolio = {"bankroll": 1000.0, "exposure": 0.0, "history": [], "pnl": 0.0}
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = None
if 'match_state' not in st.session_state: st.session_state.match_state = None
if 'backtest_data' not in st.session_state: st.session_state.backtest_data = None

@dataclass
class MatchStats:
    name: str
    att: float
    def_: float
    power: float
    form_val: float
    form_str: str
    crest: str
    home_factor: float

@dataclass
class LiveState:
    minute: int = 0
    h_goals: int = 0
    a_goals: int = 0
    pressure: float = 0.5
    momentum: float = 0.0

# -----------------------------------------------------------------------------
# 3. UI STYLES
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;900&display=swap');
    .stApp {background-color: #020617; font-family: 'Inter', sans-serif;}
    
    .app-header {
        font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 800;
        text-align: center; color: #fff; margin-bottom: 20px;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .metric-box { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.25rem; font-weight: 700; color: #f8fafc; }
    
    .value-card {
        background: rgba(6, 78, 59, 0.6); border: 1px solid #059669; border-radius: 12px; padding: 20px;
        text-align: center; margin-bottom: 20px; backdrop-filter: blur(10px); box-shadow: 0 0 20px rgba(5, 150, 105, 0.2);
    }
    
    .live-monitor {
        background: #1e293b; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 20px;
        text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .trade-row {
        display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #1e293b;
        font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #cbd5e1;
    }
    .success-badge { color: #4ade80; font-weight: bold; }
    .loss-badge { color: #f87171; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. ROBUST DATA MANAGER
# -----------------------------------------------------------------------------
class DataManager:
    def __init__(self, api_key: str):
        self.headers = {"X-Auth-Token": api_key}
        self.base_url = "https://api.football-data.org/v4"

    @st.cache_data(ttl=CONFIG["SYSTEM"]["CACHE_TTL"])
    def fetch_league_data(_self, league_code: str) -> Tuple[Optional[dict], Optional[dict]]:
        for attempt in range(3):
            try:
                r1 = requests.get(f"{_self.base_url}/competitions/{league_code}/standings", headers=_self.headers, timeout=10)
                r1.raise_for_status()
                today = datetime.now().strftime("%Y-%m-%d")
                future = (datetime.now() + timedelta(days=CONFIG["SYSTEM"]["LOOKAHEAD_DAYS"])).strftime("%Y-%m-%d")
                r2 = requests.get(
                    f"{_self.base_url}/competitions/{league_code}/matches", 
                    headers=_self.headers, 
                    params={"dateFrom": today, "dateTo": future},
                    timeout=10
                )
                r2.raise_for_status()
                return r1.json(), r2.json()
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
        return None, None

# -----------------------------------------------------------------------------
# 5. QUANTUM ENGINE (v48 - MULTI-LEAGUE SIM)
# -----------------------------------------------------------------------------
class QuantumEngine:
    def __init__(self):
        self.rng = np.random.default_rng()

    def _negative_binomial_sim(self, mu: float, size: int) -> np.ndarray:
        if mu <= 0.05: return np.zeros(size)
        n = CONFIG["MODEL"]["NB_SIZE_PARAM"]
        p = n / (n + mu)
        return self.rng.negative_binomial(n, p, size)

    def analyze_pre_match(self, h_stats: MatchStats, a_stats: MatchStats, avg_g: float, odds: List[float], league_code: str) -> Dict:
        l_weight = CONFIG["MODEL"]["LEAGUE_WEIGHTS"].get(league_code, CONFIG["MODEL"]["HOME_ADV_BASE"])
        h_xg = h_stats.att * a_stats.def_ * avg_g + (l_weight * h_stats.home_factor)
        a_xg = a_stats.att * h_stats.def_ * avg_g * CONFIG["MODEL"]["AWAY_PENALTY"]
        
        hg = self._negative_binomial_sim(h_xg, CONFIG["SYSTEM"]["SIM_COUNT_BASE"])
        ag = self._negative_binomial_sim(a_xg, CONFIG["SYSTEM"]["SIM_COUNT_BASE"])
        
        probs = [np.mean(hg > ag), np.mean(hg == ag), np.mean(hg < ag)]
        evs = [(probs[i] * odds[i]) - 1 for i in range(3)]
        best_idx = np.argmax(evs)
        edge = evs[best_idx]
        
        kelly_stake = 0.0
        if edge > 0:
            b = odds[best_idx] - 1
            p = probs[best_idx]
            q = 1 - p
            kelly_raw = (b * p - q) / b
            kelly_stake = max(0.0, kelly_raw) * CONFIG["RISK"]["KELLY_FRACTION"]

        var_95 = kelly_stake * (1 - probs[best_idx])
        return {
            "probs": probs, "evs": evs, "kelly": kelly_stake, "edge": edge, "best_idx": best_idx,
            "xg": (h_xg, a_xg), "risk": {"var": var_95},
            "h_name": h_stats.name, "a_name": a_stats.name
        }

    def run_multi_league_backtest(self) -> Dict:
        """ Simulates 3 Leagues with Performance Breakdown """
        leagues = ["PL", "TR1", "PD"]
        bankroll = 1000.0
        history = [1000.0]
        returns = []
        league_pnl = {l: 0.0 for l in leagues}
        trades = 0
        wins = 0
        
        for league in leagues:
            teams = [{"id": i, "name": f"{league}-{i}", "att": self.rng.normal(1.0, 0.2), "def": self.rng.normal(1.0, 0.2)} for i in range(20)]
            
            for _ in range(30): # 30 matches per league
                h, a = self.rng.choice(teams, 2, replace=False)
                h_stats = MatchStats(name=h["name"], att=h["att"], def_=h["def"], power=100, form_val=1, form_str="", crest="", home_factor=1.15)
                a_stats = MatchStats(name=a["name"], att=a["att"], def_=a["def"], power=100, form_val=1, form_str="", crest="", home_factor=1.15)
                
                # True Prob calc
                res = self.analyze_pre_match(h_stats, a_stats, 2.6, [1,1,1], league)
                true_probs = res["probs"]
                
                # Odds with Margin
                odds = [1.05/p if p > 0 else 100.0 for p in true_probs]
                
                analysis = self.analyze_pre_match(h_stats, a_stats, 2.6, odds, league)
                
                if analysis["edge"] > CONFIG["RISK"]["EDGE_THRESHOLD"]:
                    stake = bankroll * analysis["kelly"]
                    trades += 1
                    
                    h_goals = self.rng.negative_binomial(6, 6/(6+analysis["xg"][0]))
                    a_goals = self.rng.negative_binomial(6, 6/(6+analysis["xg"][1]))
                    actual = 0 if h_goals > a_goals else 2 if a_goals > h_goals else 1
                    
                    pnl = stake * (odds[analysis["best_idx"]] - 1) if analysis["best_idx"] == actual else -stake
                    bankroll += pnl
                    league_pnl[league] += pnl
                    history.append(bankroll)
                    returns.append(pnl/stake if stake > 0 else 0)
                    if pnl > 0: wins += 1
        
        # Risk Metrics
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(90) if len(returns) > 1 else 0
        peak = 1000.0
        max_dd = 0.0
        for val in history:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd

        return {
            "final_bankroll": bankroll, "roi": (bankroll - 1000) / 1000, 
            "win_rate": wins/trades if trades > 0 else 0, "history": history, 
            "sharpe": sharpe, "max_dd": max_dd, "league_pnl": league_pnl
        }

class LiveExecutionEngine:
    def pricing(self, state: LiveState, pre_xg: Tuple[float, float]) -> Dict[str, float]:
        h_xg, a_xg = pre_xg
        rem_ratio = max(0, (95 - state.minute) / 95)
        decay = rem_ratio ** CONFIG["MODEL"]["LIVE_DECAY_EXP"]
        
        live_h_xg = h_xg * decay + (state.momentum * 0.15)
        live_a_xg = a_xg * decay - (state.momentum * 0.15)
        
        if state.h_goals < state.a_goals: live_h_xg *= CONFIG["MODEL"]["URGENCY_BOOST"]
        if state.a_goals < state.h_goals: live_a_xg *= CONFIG["MODEL"]["URGENCY_BOOST"]
        
        rng = np.random.default_rng()
        sims = 5000
        hg_rem = rng.poisson(max(0.05, live_h_xg), sims)
        ag_rem = rng.poisson(max(0.05, live_a_xg), sims)
        
        total_h = hg_rem + state.h_goals
        total_a = ag_rem + state.a_goals
        
        return {
            "H": np.mean(total_h > total_a),
            "D": np.mean(total_h == total_a),
            "A": np.mean(total_h < total_a),
            "OU_2.5": np.mean((total_h + total_a) > 2.5)
        }

# -----------------------------------------------------------------------------
# 6. APP MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    api_key = os.environ.get("FOOTBALL_API_KEY") or st.secrets.get("FOOTBALL_API_KEY")
    
    with st.sidebar:
        st.header("🎯 Sniper Hub")
        if not api_key:
            api_key = st.text_input("API Key", type="password")
            if not api_key: st.stop()
            
        b_roll = st.session_state.portfolio["bankroll"]
        exp = st.session_state.portfolio["exposure"]
        st.metric("Total Capital", f"${b_roll:.2f}", delta=f"{st.session_state.portfolio['pnl']:+.2f}")
        
        st.divider()
        st.subheader("🛠️ Simulation Lab")
        if st.button("Run Multi-League Sim"):
            eng = QuantumEngine()
            with st.spinner("Simulating PL, TR1, PD..."):
                st.session_state.backtest_data = eng.run_multi_league_backtest()
        
        if st.session_state.backtest_data:
            bd = st.session_state.backtest_data
            c1, c2 = st.columns(2)
            c1.metric("Sharpe Ratio", f"{bd['sharpe']:.2f}")
            c2.metric("Max Drawdown", f"{bd['max_dd']*100:.1f}%")
            
            # League Breakdown
            st.caption("Performance by League:")
            for l, pnl in bd["league_pnl"].items():
                st.write(f"{l}: ${pnl:.2f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=bd["history"], mode='lines', fill='tozeroy', line=dict(color='#3b82f6')))
            fig.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.mode == "LIVE":
            if st.button("⬅️ Exit Sniper"):
                st.session_state.mode = "PRE_MATCH"
                st.rerun()

    st.markdown("<div class='app-header'>QUANTUM ULTIMATE v48</div>", unsafe_allow_html=True)

    # --- PRE-MATCH QUANT ---
    if st.session_state.mode == "PRE_MATCH":
        dm = DataManager(api_key)
        L_MAP = {"Premier League": "PL", "Süper Lig": "TR1", "La Liga": "PD", "Bundesliga": "BL1"}
        
        c1, c2 = st.columns([1,2])
        with c1: l_sel = st.selectbox("Market", list(L_MAP.keys()))
        
        stnd, fixt = dm.fetch_league_data(L_MAP[l_sel])
        if not stnd: st.stop()
        
        stats = {}
        tbl = stnd["standings"][0]["table"]
        avg_g = sum(t["goalsFor"] for t in tbl) / sum(t["playedGames"] for t in tbl)
        
        for t in tbl:
            stats[t["team"]["id"]] = MatchStats(
                name=t["team"]["name"], 
                att=(t["goalsFor"]/t["playedGames"])/avg_g,
                def_=(t["goalsAgainst"]/t["playedGames"])/avg_g,
                power=100.0, form_val=1.0, form_str=t.get("form", "N/A"),
                crest=t["team"].get("crest", CONFIG["SYSTEM"]["DEFAULT_LOGO"]),
                home_factor=1.15
            )
            
        matches = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m 
                   for m in fixt["matches"] if m["status"] in ["SCHEDULED", "TIMED"]}
        
        with c2: m_sel = st.selectbox("Fixture", list(matches.keys()))
        
        st.subheader("Market Odds Input")
        oc1, oc2, oc3 = st.columns(3)
        odds = [
            oc1.number_input("Home (1)", 1.01, 20.0, 2.00),
            oc2.number_input("Draw (X)", 1.01, 20.0, 3.40),
            oc3.number_input("Away (2)", 1.01, 20.0, 3.80)
        ]
        
        if st.button("SCAN FOR VALUE", use_container_width=True):
            m = matches[m_sel]
            engine = QuantumEngine()
            res = engine.analyze_pre_match(stats[m["homeTeam"]["id"]], stats[m["awayTeam"]["id"]], avg_g, odds, L_MAP[l_sel])
            
            # FIX: Ensure 'match' name is stored for history logging
            st.session_state.pre_analysis = {
                **res, 
                "h_name": m["homeTeam"]["name"], 
                "a_name": m["awayTeam"]["name"], 
                "match_name": f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}", # Fix for Live History
                "ids": (m["homeTeam"]["id"], m["awayTeam"]["id"])
            }
            
            if res["edge"] > CONFIG["RISK"]["EDGE_THRESHOLD"]:
                rec_stake = res["kelly"] * b_roll
                st.markdown(f"""
                <div class='value-card'>
                    <h2 style='color:#34d399; margin:0'>🎯 SNIPER SIGNAL: +{res['edge']*100:.1f}% EDGE</h2>
                    <p>Target: <b>{['HOME','DRAW','AWAY'][res['best_idx']]}</b> | Allocation: <b>${rec_stake:.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No value detected. Market is efficient.")
            
            if st.button("🚀 ACTIVATE LIVE MONITOR"):
                st.session_state.mode = "LIVE"
                st.session_state.match_state = LiveState()
                st.rerun()

    # --- LIVE EXECUTION ---
    elif st.session_state.mode == "LIVE":
        pre = st.session_state.pre_analysis
        ms = st.session_state.match_state
        
        col_l1, col_l2 = st.columns([1, 1])
        
        with col_l1:
            st.markdown(f"### 🏟️ {pre['h_name']} vs {pre['a_name']}")
            c_min, c_h, c_a = st.columns([2,1,1])
            ms.minute = c_min.slider("Minute", 0, 95, ms.minute)
            ms.h_goals = c_h.number_input("Home", 0, 10, ms.h_goals)
            ms.a_goals = c_a.number_input("Away", 0, 10, ms.a_goals)
            
            st.markdown(f"""
            <div class='live-monitor'>
                <h1 style='font-size:3rem; margin:0'>{ms.h_goals} - {ms.a_goals}</h1>
                <p style='color:#f43f5e'>{ms.minute}' LIVE</p>
            </div>
            """, unsafe_allow_html=True)

        with col_l2:
            st.markdown("### 💹 Sniper Execution")
            exec_engine = LiveExecutionEngine()
            live_probs = exec_engine.pricing(ms, pre['xg'])
            
            live_odds = {
                "H": round(1/live_probs["H"]*0.92, 2) if live_probs["H"] > 0 else 1.01,
                "D": round(1/live_probs["D"]*0.92, 2) if live_probs["D"] > 0 else 1.01,
                "A": round(1/live_probs["A"]*0.92, 2) if live_probs["A"] > 0 else 1.01
            }
            
            for mkt, prob in live_probs.items():
                if mkt == "OU_2.5": continue
                odds_val = live_odds[mkt]
                live_edge = (prob * odds_val) - 1
                
                with st.container():
                    c_btn, c_info = st.columns([1, 2])
                    
                    btn_label = f"BUY {mkt}"
                    if live_edge > 0.05: 
                        btn_label = f"🔥 BUY {mkt} (VALUE!)"
                    
                    if c_btn.button(btn_label, key=mkt):
                        stake = 50.0 
                        pnl = (stake * odds_val) - stake if np.random.random() < prob else -stake
                        st.session_state.portfolio["bankroll"] += pnl
                        st.session_state.portfolio["pnl"] += pnl
                        st.session_state.portfolio["history"].append({
                            "match": pre['match_name'], "pnl": f"{pnl:+.2f}", 
                            "cls": "success-badge" if pnl > 0 else "loss-badge"
                        })
                        st.success(f"FILLED @ {odds_val}")
                    
                    edge_txt = f"<span style='color:#4ade80'>Edge: +{live_edge*100:.1f}%</span>" if live_edge > 0.05 else "Fair Price"
                    c_info.markdown(f"**{mkt}** @ {odds_val} | {edge_txt}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
