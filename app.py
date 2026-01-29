import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import io
import os
import urllib.request
from fpdf import FPDF
from scipy.stats import poisson
import hmac
import hashlib
import random
import time
import firebase_admin
from firebase_admin import credentials, firestore
import matplotlib.pyplot as plt

# --- 0. SİSTEM YAPILANDIRMASI ---
MODEL_VERSION = "v14.0-Global"

st.set_page_config(page_title="QUANTUM FOOTBALL", page_icon="⚽", layout="wide")
np.random.seed(42)

# --- DİL SÖZLÜĞÜ (TRANSLATION DICTIONARY) ---
TRANS = {
    "EN": {
        "page_title": "QUANTUM FOOTBALL",
        "legal_warning": "⚠️ DISCLAIMER:\nThis system is a statistical simulation tool for educational purposes.\nIt does NOT provide betting or financial advice.",
        "tab_sim": "📊 Simulation",
        "tab_admin": "🗃️ Admin Panel",
        "tab_model": "📘 Model Card",
        "lbl_league": "Select League",
        "lbl_match": "Select Match",
        "exp_params": "🛠️ Parameter Settings",
        "lbl_tac_home": "Home Tactics",
        "lbl_tac_away": "Away Tactics",
        "btn_start": "🚀 RUN SIMULATION",
        "res_conf": "Confidence Score",
        "res_dqi": "Data Quality (DQI)",
        "res_elo": "Elo Diff",
        "res_auto_power": "⚡ Auto Power Detect",
        "res_xg": "⚽ Expected Goals (xG)",
        "res_ci": "🧪 90% Confidence Interval",
        "ci_desc": "Model projects Home goals between **[{0}-{1}]**, Away goals between **[{2}-{3}]**.",
        "tab_res_1": "Main Table (1X2)",
        "tab_res_2": "HT / FT Probabilities",
        "tab_res_3": "Goal Markets",
        "col_home": "Home %",
        "col_draw": "Draw %",
        "col_away": "Away %",
        "market_o15": "Over 1.5",
        "market_o25": "Over 2.5",
        "market_o35": "Over 3.5",
        "market_btts": "BTTS (Both Teams Score)",
        "admin_batch_title": "⚡ Batch Processing Center",
        "admin_batch_desc": "Analyzes all upcoming and live matches in the selected league.",
        "admin_batch_btn": "⚡ ANALYZE FULL LEAGUE",
        "admin_batch_success": "✅ Operation Complete: {0} matches added to database.",
        "admin_valid_title": "📝 Result Validation (Pending)",
        "admin_completed_title": "✅ Validated Matches (History)",
        "admin_valid_sel": "Select Match to Validate",
        "admin_valid_btn": "✅ Save Result & Update Elo",
        "admin_valid_success": "Result saved successfully! Updating system...",
        "msg_no_match": "No matches found.",
        "msg_wait": "Pending...",
        "pow_dominant": "Dominant",
        "pow_strong": "Strong",
        "pow_adv": "Advantage",
        "pow_balanced": "Balanced",
        "dl_report": "📥 Download Report (PDF)"
    },
    "TR": {
        "page_title": "QUANTUM FOOTBALL",
        "legal_warning": "⚠️ YASAL UYARI:\nBu sistem, istatistiksel veri simülasyonu yapan bir analiz aracıdır.\nKesinlikle bahis veya finansal yatırım tavsiyesi vermez.",
        "tab_sim": "📊 Simülasyon",
        "tab_admin": "🗃️ Admin Paneli",
        "tab_model": "📘 Model Kimliği",
        "lbl_league": "Lig Seçin",
        "lbl_match": "Maç Seçin",
        "exp_params": "🛠️ Parametre Ayarları",
        "lbl_tac_home": "Ev Taktik",
        "lbl_tac_away": "Dep Taktik",
        "btn_start": "🚀 SİMÜLASYONU BAŞLAT",
        "res_conf": "Güven Skoru",
        "res_dqi": "Veri Kalitesi (DQI)",
        "res_elo": "Elo Farkı",
        "res_auto_power": "⚡ Otomatik Güç Tespiti",
        "res_xg": "⚽ Beklenen Goller (xG)",
        "res_ci": "🧪 %90 Güven Aralığı",
        "ci_desc": "Model, Ev Sahibinin **[{0}-{1}]**, Deplasmanın **[{2}-{3}]** gol atacağını öngörüyor.",
        "tab_res_1": "Ana Tablo (1X2)",
        "tab_res_2": "İY / MS (HT/FT)",
        "tab_res_3": "Gol Piyasaları",
        "col_home": "Ev %",
        "col_draw": "Berabere %",
        "col_away": "Dep %",
        "market_o15": "1.5 Üst",
        "market_o25": "2.5 Üst",
        "market_o35": "3.5 Üst",
        "market_btts": "KG Var",
        "admin_batch_title": "⚡ Toplu İşlem Merkezi",
        "admin_batch_desc": "Seçili ligdeki tüm gelecek ve canlı maçları analiz eder.",
        "admin_batch_btn": "⚡ TÜM LİGİ ANALİZ ET",
        "admin_batch_success": "✅ İşlem Tamamlandı: {0} maç eklendi.",
        "admin_valid_title": "📝 Sonuç Doğrulama (Bekleyenler)",
        "admin_completed_title": "✅ Tamamlanan Maçlar (Geçmiş)",
        "admin_valid_sel": "Sonuçlanacak Maçı Seç",
        "admin_valid_btn": "✅ Sonucu Kaydet ve Elo'yu İşle",
        "admin_valid_success": "Maç sonucu başarıyla kaydedildi! Liste güncelleniyor...",
        "msg_no_match": "Kriterlere uygun maç bulunamadı.",
        "msg_wait": "Bekleniyor...",
        "pow_dominant": "Dominant",
        "pow_strong": "Güçlü",
        "pow_adv": "Avantajlı",
        "pow_balanced": "Dengeli",
        "dl_report": "📥 Raporu İndir (PDF)"
    },
     "DE": {
        "page_title": "QUANTUM FUSSBALL",
        "legal_warning": "⚠️ HAFTUNGSAUSSCHLUSS:\nDies ist ein statistisches Simulationswerkzeug.\nEs bietet KEINE Wett- oder Finanzberatung.",
        "tab_sim": "📊 Simulation",
        "tab_admin": "🗃️ Admin-Bereich",
        "tab_model": "📘 Modellkarte",
        "lbl_league": "Liga Wählen",
        "lbl_match": "Spiel Wählen",
        "exp_params": "🛠️ Parametereinstellungen",
        "lbl_tac_home": "Heim Taktik",
        "lbl_tac_away": "Auswärts Taktik",
        "btn_start": "🚀 SIMULATION STARTEN",
        "res_conf": "Konfidenz-Score",
        "res_dqi": "Datenqualität (DQI)",
        "res_elo": "Elo-Diff",
        "res_auto_power": "⚡ Auto-Stärke",
        "res_xg": "⚽ Erwartete Tore (xG)",
        "res_ci": "🧪 90% Konfidenzintervall",
        "ci_desc": "Modell prognostiziert Heimtore zwischen **[{0}-{1}]**, Auswärtstore zwischen **[{2}-{3}]**.",
        "tab_res_1": "Haupttabelle (1X2)",
        "tab_res_2": "HZ / ES Wahrsch.",
        "tab_res_3": "Tormärkte",
        "col_home": "Heim %",
        "col_draw": "Remis %",
        "col_away": "Gast %",
        "market_o15": "Über 1.5",
        "market_o25": "Über 2.5",
        "market_o35": "Über 3.5",
        "market_btts": "Beide Treffen (BTTS)",
        "admin_batch_title": "⚡ Stapelverarbeitung",
        "admin_batch_desc": "Analysiert alle kommenden Spiele.",
        "admin_batch_btn": "⚡ LIGA ANALYSIEREN",
        "admin_batch_success": "✅ Fertig: {0} Spiele hinzugefügt.",
        "admin_valid_title": "📝 Ergebnisvalidierung",
        "admin_completed_title": "✅ Abgeschlossene Spiele",
        "admin_valid_sel": "Spiel auswählen",
        "admin_valid_btn": "✅ Bestätigen & Trainieren",
        "admin_valid_success": "Ergebnis gespeichert!",
        "msg_no_match": "Keine Spiele gefunden.",
        "msg_wait": "Warten...",
        "pow_dominant": "Dominant",
        "pow_strong": "Stark",
        "pow_adv": "Vorteil",
        "pow_balanced": "Ausgeglichen",
        "dl_report": "📥 Bericht Herunterladen (PDF)"
    },
    "FR": {
        "page_title": "FOOTBALL QUANTIQUE",
        "legal_warning": "⚠️ AVERTISSEMENT:\nCe système est un outil de simulation statistique.\nIl ne fournit PAS de conseils de paris.",
        "tab_sim": "📊 Simulation",
        "tab_admin": "🗃️ Panneau Admin",
        "tab_model": "📘 Carte Modèle",
        "lbl_league": "Choisir la Ligue",
        "lbl_match": "Choisir le Match",
        "exp_params": "🛠️ Paramètres",
        "lbl_tac_home": "Tactique Domicile",
        "lbl_tac_away": "Tactique Extérieur",
        "btn_start": "🚀 LANCER SIMULATION",
        "res_conf": "Score Confiance",
        "res_dqi": "Qualité Données",
        "res_elo": "Diff. Elo",
        "res_auto_power": "⚡ Détection Puissance",
        "res_xg": "⚽ Buts Attendus (xG)",
        "res_ci": "🧪 Intervalle de Confiance (90%)",
        "ci_desc": "Le modèle prévoit buts Domicile entre **[{0}-{1}]**, Extérieur entre **[{2}-{3}]**.",
        "tab_res_1": "Tableau Principal",
        "tab_res_2": "Mi-temps / Fin",
        "tab_res_3": "Marchés des Buts",
        "col_home": "Dom %",
        "col_draw": "Nul %",
        "col_away": "Ext %",
        "market_o15": "Plus de 1.5",
        "market_o25": "Plus de 2.5",
        "market_o35": "Plus de 3.5",
        "market_btts": "Les 2 Marquent",
        "admin_batch_title": "⚡ Traitement par Lots",
        "admin_batch_desc": "Analyse tous les matchs à venir.",
        "admin_batch_btn": "⚡ ANALYSER LA LIGUE",
        "admin_batch_success": "✅ Terminé: {0} matchs ajoutés.",
        "admin_valid_title": "📝 Validation Résultats",
        "admin_completed_title": "✅ Matchs Validés",
        "admin_valid_sel": "Match à Valider",
        "admin_valid_btn": "✅ Confirmer & Entraîner",
        "admin_valid_success": "Résultat enregistré!",
        "msg_no_match": "Aucun match trouvé.",
        "msg_wait": "En attente...",
        "pow_dominant": "Dominant",
        "pow_strong": "Fort",
        "pow_adv": "Avantage",
        "pow_balanced": "Équilibré",
        "dl_report": "📥 Télécharger Rapport (PDF)"
    },
    "AR": {
        "page_title": "كرة القدم الكمومية",
        "legal_warning": "⚠️ إخلاء مسؤولية:\nهذا النظام هو أداة محاكاة إحصائية للأغراض التعليمية.\nلا يقدم نصائح للمراهنة أو الاستثمار المالي.",
        "tab_sim": "📊 محاكاة",
        "tab_admin": "🗃️ لوحة الإدارة",
        "tab_model": "📘 بطاقة النموذج",
        "lbl_league": "اختر الدوري",
        "lbl_match": "اختر المباراة",
        "exp_params": "🛠️ إعدادات المعلمات",
        "lbl_tac_home": "تكتيكات المضيف",
        "lbl_tac_away": "تكتيكات الضيف",
        "btn_start": "🚀 ابدأ المحاكاة",
        "res_conf": "درجة الثقة",
        "res_dqi": "جودة البيانات (DQI)",
        "res_elo": "فرق Elo",
        "res_auto_power": "⚡ كشف القوة التلقائي",
        "res_xg": "⚽ الأهداف المتوقعة (xG)",
        "res_ci": "🧪 فاصل الثقة 90%",
        "ci_desc": "يتوقع النموذج أهداف المضيف بين **[{0}-{1}]**، والضيف بين **[{2}-{3}]**.",
        "tab_res_1": "الجدول الرئيسي",
        "tab_res_2": "الشوط الأول / المباراة",
        "tab_res_3": "أسواق الأهداف",
        "col_home": "مضيف %",
        "col_draw": "تعادل %",
        "col_away": "ضيف %",
        "market_o15": "أكثر من 1.5",
        "market_o25": "أكثر من 2.5",
        "market_o35": "أكثر من 3.5",
        "market_btts": "كلا الفريقين يسجل",
        "admin_batch_title": "⚡ مركز المعالجة المجمعة",
        "admin_batch_desc": "يحلل جميع المباريات القادمة.",
        "admin_batch_btn": "⚡ تحليل الدوري بالكامل",
        "admin_batch_success": "✅ اكتملت العملية: تمت إضافة {0} مباراة.",
        "admin_valid_title": "📝 التحقق من النتائج",
        "admin_completed_title": "✅ المباريات المكتملة",
        "admin_valid_sel": "المباراة للتحقق",
        "admin_valid_btn": "✅ تأكيد وتدريب",
        "admin_valid_success": "تم حفظ النتيجة بنجاح!",
        "msg_no_match": "لا توجد مباريات قادمة.",
        "msg_wait": "قيد الانتظار...",
        "pow_dominant": "مهيمن",
        "pow_strong": "قوي",
        "pow_adv": "أفضلية",
        "pow_balanced": "متوازن",
        "dl_report": "📥 تحميل التقرير (PDF)"
    }
}

# --- GÜVENLİK ---
AUTH_SALT = st.secrets.get("auth_salt", "quantum_research_key_2026") 
ADMIN_EMAILS = ["muratlola@gmail.com", "firat3306ogur@gmail.com"] 

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# --- FIREBASE BAĞLANTISI ---
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            creds_dict = dict(st.secrets["firebase"])
            creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred)
    except Exception as e: logger.error(f"Firebase Error: {e}")
try: db = firestore.client()
except: db = None

# --- SABİTLER ---
CONSTANTS = {
    "API_URL": "https://api.football-data.org/v4",
    "HOME_ADVANTAGE": 1.12, 
    "RHO": -0.10, 
    "ELO_K": 32,
    "TACTICS": {"Dengeli": (1.0, 1.0), "Hücum": (1.25, 1.15), "Savunma": (0.65, 0.60), "Kontra": (0.95, 0.85)},
    "WEATHER": {"Normal": 1.0, "Yağmurlu": 0.95, "Karlı": 0.85, "Sıcak": 0.92},
    "LEAGUES": {
        "Şampiyonlar Ligi": "CL", 
        "Premier League (EN)": "PL", 
        "Championship (EN)": "ELC", 
        "La Liga (ES)": "PD",
        "Bundesliga (DE)": "BL1", 
        "Serie A (IT)": "SA", 
        "Ligue 1 (FR)": "FL1",
        "Eredivisie (NL)": "DED", 
        "Primeira Liga (PT)": "PPL", 
        "Süper Lig (TR)": "TR1"
    }
}

LEAGUE_PROFILES = {
    "PL": {"pace": 1.15, "variance": 1.1}, 
    "ELC": {"pace": 1.10, "variance": 1.2}, 
    "SA": {"pace": 0.90, "variance": 0.8},
    "BL1": {"pace": 1.20, "variance": 1.2},
    "TR1": {"pace": 1.05, "variance": 1.3},
    "DEFAULT": {"pace": 1.0, "variance": 1.0}
}

# -----------------------------------------------------------------------------
# 1. KİMLİK DOĞRULAMA
# -----------------------------------------------------------------------------
query_params = st.query_params
current_user = query_params.get("user_email", "Guest")
provided_token = query_params.get("token", None)

def is_valid_admin(email, token):
    if not token: return False
    expected = hmac.new(AUTH_SALT.encode(), email.lower().strip().encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, token)

is_admin = False
if "@" in current_user:
    clean_email = current_user.lower().strip()
    if clean_email in [a.lower() for a in ADMIN_EMAILS]:
        if is_valid_admin(clean_email, provided_token): is_admin = True

# -----------------------------------------------------------------------------
# 2. CORE ENGINE
# -----------------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self, elo_manager=None): 
        self.elo_manager = elo_manager

    def calculate_confidence_interval(self, mu, alpha=0.90):
        low, high = poisson.interval(alpha, mu)
        return int(low), int(high)

    def calculate_ht_ft_probs(self, p_home, p_draw, p_away):
        return {
            "1/1": p_home * 0.58, "X/1": p_home * 0.28, "2/1": p_home * 0.14,
            "1/X": p_draw * 0.18, "X/X": p_draw * 0.64, "2/X": p_draw * 0.18,
            "1/2": p_away * 0.14, "X/2": p_away * 0.28, "2/2": p_away * 0.58
        }

    def run_ensemble_analysis(self, h_stats, a_stats, avg_g, params, h_id, a_id, league_code):
        l_prof = LEAGUE_PROFILES.get(league_code, LEAGUE_PROFILES["DEFAULT"])
        elo_h = 1500; elo_a = 1500
        elo_impact = 0
        if self.elo_manager:
            elo_h = self.elo_manager.get_elo(h_id, h_stats['name'])
            elo_a = self.elo_manager.get_elo(a_id, a_stats['name'])
            elo_impact = ((elo_h - elo_a) / 100.0) * 0.06

        h_form = h_stats.get('form_factor', 1.0); a_form = a_stats.get('form_factor', 1.0)
        form_impact = (h_form - a_form) * 0.18
        power_impact = params.get('power_diff', 0) * 0.12

        base_h = (h_stats['gf']/avg_g) * (a_stats['ga']/avg_g) * avg_g * CONSTANTS["HOME_ADVANTAGE"]
        base_a = (a_stats['gf']/avg_g) * (h_stats['ga']/avg_g) * avg_g
        
        xg_h = base_h * l_prof["pace"] * params['t_h'][0] * params['t_a'][1] * (1 + elo_impact + form_impact + power_impact)
        xg_a = base_a * l_prof["pace"] * params['t_a'][0] * params['t_h'][1] * (1 - elo_impact - form_impact - power_impact)
        
        if params['hk']: xg_h *= 0.85
        if params['hgk']: xg_a *= 1.15
        if params['ak']: xg_a *= 0.85
        if params['agk']: xg_h *= 1.15

        h_probs = poisson.pmf(np.arange(7), xg_h)
        a_probs = poisson.pmf(np.arange(7), xg_a)
        matrix = np.outer(h_probs, a_probs)
        
        rho = CONSTANTS["RHO"]
        matrix[0,0] *= (1 - (xg_h*xg_a*rho))
        matrix[0,1] *= (1 + (xg_h*rho))
        matrix[1,0] *= (1 + (xg_a*rho))
        matrix[1,1] *= (1 - rho)
        matrix[matrix < 0] = 0; matrix /= matrix.sum()

        p_home = np.sum(np.tril(matrix, -1)) * 100
        p_draw = np.sum(np.diag(matrix)) * 100
        p_away = np.sum(np.triu(matrix, 1)) * 100
        
        rows, cols = np.indices(matrix.shape)
        total_goals = rows + cols
        
        over_15 = np.sum(matrix[total_goals > 1.5]) * 100
        over_25 = np.sum(matrix[total_goals > 2.5]) * 100
        over_35 = np.sum(matrix[total_goals > 3.5]) * 100
        btts = (1 - (matrix[0,:].sum() + matrix[:,0].sum() - matrix[0,0])) * 100
        
        ht_ft = self.calculate_ht_ft_probs(p_home, p_draw, p_away)
        ci_h = self.calculate_confidence_interval(xg_h)
        ci_a = self.calculate_confidence_interval(xg_a)
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        return {
            "1x2": [p_home, p_draw, p_away],
            "matrix": matrix * 100,
            "goals": {"o15": over_15, "o25": over_25, "o35": over_35, "btts": btts},
            "ht_ft": ht_ft,
            "xg": (xg_h, xg_a),
            "ci": (ci_h, ci_a),
            "most_likely": f"{max_idx[0]}-{max_idx[1]}",
            "elo": (elo_h, elo_a)
        }

    def calculate_auto_power(self, h_stats, a_stats, t):
        if h_stats['played'] < 2: return 0, t["msg_wait"]
        h_val = (h_stats['points']/h_stats['played'])*2.0 + (h_stats['gf']-h_stats['ga'])/h_stats['played']
        a_val = (a_stats['points']/a_stats['played'])*2.0 + (a_stats['gf']-a_stats['ga'])/a_stats['played']
        diff = h_val - a_val
        
        if diff > 1.2: return 3, f"🔥 {h_stats['name']} {t['pow_dominant']}"
        if diff > 0.5: return 2, f"💪 {h_stats['name']} {t['pow_strong']}"
        if diff > 0.2: return 1, f"📈 {h_stats['name']} {t['pow_adv']}"
        if diff < -1.2: return -3, f"🔥 {a_stats['name']} {t['pow_dominant']}"
        if diff < -0.5: return -2, f"💪 {a_stats['name']} {t['pow_strong']}"
        if diff < -0.2: return -1, f"📈 {a_stats['name']} {t['pow_adv']}"
        return 0, t['pow_balanced']

class DataManager:
    def __init__(self, key): self.headers = {"X-Auth-Token": key}
    @st.cache_data(ttl=3600)
    def fetch(_self, league):
        try:
            r1 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/standings", headers=_self.headers)
            r2 = requests.get(f"{CONSTANTS['API_URL']}/competitions/{league}/matches", headers=_self.headers)
            return r1.json(), r2.json()
        except: return None, None

    def calculate_form(self, fixtures, team_id):
        matches = [m for m in fixtures.get('matches', []) if m['status'] == 'FINISHED' and (m['homeTeam']['id'] == team_id or m['awayTeam']['id'] == team_id)]
        matches.sort(key=lambda x: x['utcDate'], reverse=True)
        last_5 = matches[:5]
        form_list = []
        w_sum = 0; tot_w = 0
        for i, m in enumerate(last_5):
            res='L'; pts=0
            if m['score']['winner'] == 'DRAW': res='D'; pts=1
            elif (m['score']['winner']=='HOME_TEAM' and m['homeTeam']['id']==team_id) or (m['score']['winner']=='AWAY_TEAM' and m['awayTeam']['id']==team_id): res='W'; pts=3
            w = 1.0/(1+i*0.2)
            w_sum += pts*w; tot_w += w
            form_list.append(res)
        return ",".join(form_list), (0.8 + (w_sum/tot_w/3.0)*0.5 if tot_w > 0 else 1.0)

    def get_stats(self, s, m, tid):
        for st_ in s.get('standings',[]):
            if st_['type']=='TOTAL':
                for t in st_['table']:
                    if t['team']['id']==tid:
                        f_str, f_fac = self.calculate_form(m, tid)
                        return {"name":t['team']['name'], "gf":t['goalsFor']/t['playedGames'], "ga":t['goalsAgainst']/t['playedGames'], "points": t['points'], "played": t['playedGames'], "form": f_str, "form_factor": f_fac, "crest":t['team'].get('crest','')}
        return {"name":"Takım", "gf":1.3, "ga":1.3, "points":1, "played":1, "form":"", "form_factor":1.0, "crest":""}

class EloManager:
    def __init__(self, db): self.db = db
    def get_elo(self, tid, name, ppg=1.35):
        if not self.db: return 1500
        doc = self.db.collection("ratings").document(str(tid)).get()
        return doc.to_dict().get("elo", 1500) if doc.exists else int(1000 + ppg*333)
    def update(self, hid, hnm, aid, anm, hg, ag):
        eh = self.get_elo(hid, hnm); ea = self.get_elo(aid, anm)
        exp = 1/(1+10**((ea-eh)/400))
        act = 1.0 if hg>ag else 0.0 if hg<ag else 0.5
        k = CONSTANTS["ELO_K"] * (1.5 if abs(hg-ag)>2 else 1.0)
        d = k*(act-exp)
        self.db.collection("ratings").document(str(hid)).set({"name":hnm, "elo":round(eh+d)}, merge=True)
        self.db.collection("ratings").document(str(aid)).set({"name":anm, "elo":round(ea-d)}, merge=True)

# -----------------------------------------------------------------------------
# 3. YARDIMCILAR & PDF
# -----------------------------------------------------------------------------
def check_font():
    fp = "DejaVuSans.ttf"
    if not os.path.exists(fp):
        try: urllib.request.urlretrieve("https://github.com/coreybutler/fonts/raw/master/ttf/DejaVuSans.ttf", fp)
        except: pass
    return fp

def create_model_card():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"MODEL CARD: {MODEL_VERSION}", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, "TYPE: Probabilistic Ensemble (Dixon-Coles + Elo + Form)\n\nINTENDED USE: Decision Support\n\nINPUTS: Goals per match, Time-decayed form, Elo ratings, Contextual factors.\n\nMETRICS: Brier Score, Calibration Error, MAE.\n\nOUTPUTS: Full-time probabilities, 95% Confidence Intervals, Goal Markets.\n\nETHICS: Non-gambling, strictly for statistical analysis.")
    return pdf.output(dest='S').encode('latin-1')

def create_match_pdf(h, a, res, conf):
    fp = check_font(); pdf = FPDF(); pdf.add_page()
    if os.path.exists(fp): pdf.add_font("DejaVu","",fp,uni=True); pdf.set_font("DejaVu","",12)
    else: pdf.set_font("Arial","",12)
    def s(t): return t.encode('latin-1','replace').decode('latin-1')
    
    pdf.cell(0,10,s(f"QUANTUM FOOTBALL REPORT: {h['name']} vs {a['name']}"),ln=True,align="C")
    pdf.cell(0,10,s(f"Confidence: {conf}/100 | Elo: {res['elo'][0]} vs {res['elo'][1]}"),ln=True)
    pdf.ln(5)
    pdf.cell(0,10,s(f"1X2: {res['1x2'][0]:.1f}% - {res['1x2'][1]:.1f}% - {res['1x2'][2]:.1f}%"),ln=True)
    pdf.cell(0,10,s(f"xG: {res['xg'][0]:.2f} - {res['xg'][1]:.2f}"),ln=True)
    pdf.cell(0,10,s(f"Most Likely: {res['most_likely']}"),ln=True)
    pdf.ln(5)
    pdf.cell(0,10,s(f"Confidence Interval (90%): Home {res['ci'][0]} - Away {res['ci'][1]}"),ln=True)
    return pdf.output(dest='S').encode('latin-1')

def update_result_db(doc_id, hg, ag, notes):
    if not db: return False
    try:
        ref = db.collection("predictions").document(str(doc_id))
        doc = ref.get()
        if not doc.exists: return False
        d = doc.to_dict()
        
        # --- [CRITICAL FIX] Ensure integers ---
        hg_int = int(hg)
        ag_int = int(ag)
        
        # Sonuç
        res = "1" if hg_int > ag_int else "2" if ag_int > hg_int else "X"
        idx = 0 if res == "1" else 1 if res == "X" else 2
        
        # Brier Score
        probs = [d.get("home_prob"), d.get("draw_prob"), d.get("away_prob")]
        brier = 0.0
        if None not in probs:
            p_vec = np.array([probs[0]/100, probs[1]/100, probs[2]/100])
            o_vec = np.array([0,0,0]); o_vec[idx] = 1
            brier = np.sum((p_vec - o_vec)**2)

        # Elo Update
        match_str = d.get("match_name") or d.get("match", "Unknown vs Unknown")
        if " vs " in match_str:
            home_name = match_str.split(" vs ")[0]
            away_name = match_str.split(" vs ")[1]
            elo = EloManager(db)
            if "home_id" in d and "away_id" in d:
                elo.update(d["home_id"], home_name, d["away_id"], away_name, hg_int, ag_int)
            
        ref.update({
            "actual_result": res, 
            "actual_score": f"{hg_int}-{ag_int}",
            "brier_score": float(brier), 
            "validation_status": "VALIDATED",
            "admin_notes": notes
        })
        return True
    except Exception as e: st.error(f"Kayıt Hatası: {e}"); return False

def save_pred_db(match, probs, params, user, meta):
    if not db: return
    p1, p2, p3 = float(probs[0]), float(probs[1]), float(probs[2])
    pred = "1" if p1>p2 and p1>p3 else "2" if p3>p1 and p3>p2 else "X"
    
    doc_ref = db.collection("predictions").document(str(match['id']))
    existing = doc_ref.get()
    
    data = {
        "match_id": str(match['id']), "match_name": f"{meta['hn']} vs {meta['an']}",
        "match_date": match['utcDate'], "league": meta['lg'],
        "home_id": meta['hid'], "away_id": meta['aid'],
        "home_prob": p1, "draw_prob": p2, "away_prob": p3,
        "predicted_outcome": pred, "confidence": meta['conf'],
        "dqi": meta['dqi'], "user": user, "params": str(params),
        "model_version": MODEL_VERSION
    }
    
    if not existing.exists:
        data["actual_result"] = None
        
    doc_ref.set(data, merge=True)

# -----------------------------------------------------------------------------
# 4. MAIN UI
# -----------------------------------------------------------------------------
def main():
    st.markdown("""<style>
        .stApp {background-color: #0e1117; color: #fff;}
        .big-n {font-size:24px; font-weight:bold; color:#00ff88;}
        .card {background:#1e2129; padding:15px; border-radius:10px; margin-bottom:10px;}
    </style>""", unsafe_allow_html=True)
    
    # --- DİL SEÇİCİ ---
    with st.sidebar:
        st.header("🌐 Language")
        # Product Hunt için EN varsayılan
        lang_sel = st.selectbox("Select Language / Dil Seçin", ["English (EN)", "Türkçe (TR)", "Deutsch (DE)", "Français (FR)", "العربية (AR)"], index=0)
        
        lang_map = {"English (EN)": "EN", "Türkçe (TR)": "TR", "Deutsch (DE)": "DE", "Français (FR)": "FR", "العربية (AR)": "AR"}
        curr_lang = lang_map[lang_sel]
        t = TRANS[curr_lang]

    st.title(t["page_title"])
    st.info(t["legal_warning"])

    if is_admin:
        tabs = st.tabs([t["tab_sim"], t["tab_admin"], t["tab_model"]])
    else: tabs = [st.container()]

    # TAB 1: ANALİZ
    with tabs[0]:
        api = st.secrets.get("FOOTBALL_API_KEY")
        if not api: st.error("API Key Yok"); st.stop()
        dm = DataManager(api); eng = AnalyticsEngine(EloManager(db))
        
        c1, c2 = st.columns([1,2])
        with c1: 
            lk = st.selectbox(t["lbl_league"], list(CONSTANTS["LEAGUES"].keys()))
            lc = CONSTANTS["LEAGUES"][lk]
        s, f = dm.fetch(lc)
        
        if f:
            upc = [m for m in f['matches'] if m['status'] in ['SCHEDULED','TIMED', 'IN_PLAY', 'PAUSED']]
            if not upc: st.warning(t["msg_no_match"])
            
            mm = {f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}": m for m in upc}
            if mm:
                with c2: mn = st.selectbox(t["lbl_match"], list(mm.keys())); m = mm[mn]
                
                with st.expander(t["exp_params"]):
                    pc1, pc2 = st.columns(2)
                    th = pc1.selectbox(t["lbl_tac_home"], list(CONSTANTS["TACTICS"].keys()))
                    ta = pc2.selectbox(t["lbl_tac_away"], list(CONSTANTS["TACTICS"].keys()))
                
                if st.button(t["btn_start"]):
                    hid, aid = m['homeTeam']['id'], m['awayTeam']['id']
                    hs = dm.get_stats(s, f, hid); as_ = dm.get_stats(s, f, aid)
                    dqi = 100; 
                    if hs['played'] < 5: dqi -= 20
                    
                    pow_diff, pow_msg = eng.calculate_auto_power(hs, as_, t)
                    pars = {"t_h": CONSTANTS["TACTICS"][th], "t_a": CONSTANTS["TACTICS"][ta], "weather": 1.0, "hk": False, "ak": False, "hgk": False, "agk": False, "power_diff": pow_diff}
                    res = eng.run_ensemble_analysis(hs, as_, 2.8, pars, hid, aid, lc)
                    conf = int(max(res['1x2']) * (dqi/100.0))
                    
                    meta = {"hn": hs['name'], "an": as_['name'], "hid": h_id, "aid": a_id, "lg": lc, "conf": conf, "dqi": dqi}
                    save_pred_db(m, res['1x2'], pars, current_user, meta)
                    
                    st.divider()
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric(t["res_conf"], f"{conf}/100", delta="Model Confidence")
                    c_b.metric(t["res_dqi"], f"{dqi}", delta_color="off")
                    c_c.metric(t["res_elo"], f"{res['elo'][0] - res['elo'][1]}", help="Elo Diff")
                    
                    if t["pow_balanced"] not in pow_msg: st.caption(f"{t['res_auto_power']}: {pow_msg}")
                    
                    st.write(f"### {t['res_xg']}: {res['xg'][0]:.2f} - {res['xg'][1]:.2f}")
                    
                    def plot_bell_curve(mu, team_name, ci_low, ci_high, color):
                        x = np.arange(0, 8); y = poisson.pmf(x, mu)
                        fig, ax = plt.subplots(figsize=(5, 1.5))
                        fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
                        ax.plot(x, y, 'o-', color=color, markersize=4, linewidth=1, alpha=0.8)
                        ax.fill_between(x, 0, y, where=(x >= ci_low) & (x <= ci_high), color=color, alpha=0.2)
                        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#444'); ax.spines['bottom'].set_color('#444')
                        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white', labelsize=8)
                        ax.set_title(f"{team_name} (Exp: {mu:.2f})", color='white', fontsize=9, pad=2)
                        return fig

                    col_g1, col_g2 = st.columns(2)
                    with col_g1: st.pyplot(plot_bell_curve(res['xg'][0], hs['name'], res['ci'][0][0], res['ci'][0][1], '#00ff88'), use_container_width=True)
                    with col_g2: st.pyplot(plot_bell_curve(res['xg'][1], as_['name'], res['ci'][1][0], res['ci'][1][1], '#ff4444'), use_container_width=True)

                    st.info(f"**{t['res_ci']}:**\n" + t['ci_desc'].format(res['ci'][0][0], res['ci'][0][1], res['ci'][1][0], res['ci'][1][1]))
                    
                    t1, t2, t3 = st.tabs([t['tab_res_1'], t['tab_res_2'], t['tab_res_3']])
                    with t1:
                        st.dataframe(pd.DataFrame([res['1x2']], columns=[t["col_home"], t["col_draw"], t["col_away"]]), hide_index=True)
                        st.caption(f"Max Prob: **{res['most_likely']}**")
                    with t2:
                        df_htft = pd.DataFrame(list(res['ht_ft'].items()), columns=['Pick', 'Prob %']).sort_values('Prob %', ascending=False).head(5)
                        st.table(df_htft.set_index('Pick'))
                    with t3:
                        gol_data = {"Market": [t["market_o15"], t["market_o25"], t["market_o35"], t["market_btts"]], "Prob %": [f"%{res['goals']['o15']:.1f}", f"%{res['goals']['o25']:.1f}", f"%{res['goals']['o35']:.1f}", f"%{res['goals']['btts']:.1f}"]}
                        st.table(pd.DataFrame(gol_data).set_index("Market"))

                    p_bytes = create_match_pdf(hs, as_, res, conf)
                    st.download_button(t["dl_report"], p_bytes, "analiz_v14.pdf", "application/pdf")

    # TAB 2: ADMIN
    if is_admin and len(tabs) > 1:
        with tabs[1]:
            st.header(t["tab_admin"])
            
            # --- YENİ SEKMELER ---
            adm_t1, adm_t2, adm_t3 = st.tabs([t["admin_batch_title"], t["admin_valid_title"], t["admin_completed_title"]])
            
            # 1. BATCH PROCESSING
            with adm_t1:
                st.write(t["admin_batch_desc"])
                
                # Tarih Aralığı Seçimi (Geçmiş maçlar için)
                lookback_days = st.slider("Geriye Dönük Tarama (Gün)", 0, 14, 3, help="Kaç gün önceki bitmiş maçları da sisteme ekleyelim?")
                
                if f:
                    if st.button(t["admin_batch_btn"]):
                        # Şimdiki zaman
                        now = datetime.utcnow()
                        cutoff_date = now - timedelta(days=lookback_days)
                        
                        target_matches = []
                        for m in f['matches']:
                            # Maç tarihi (String -> Datetime)
                            m_date_str = m['utcDate']
                            m_date = datetime.strptime(m_date_str, "%Y-%m-%dT%H:%M:%SZ")
                            
                            # 1. Gelecek ve Canlı maçları her türlü al
                            if m['status'] in ['SCHEDULED', 'TIMED', 'IN_PLAY', 'PAUSED']:
                                target_matches.append(m)
                            
                            # 2. Geçmiş maçları (Eğer tarih limitine uyuyorsa) al
                            elif m['status'] == 'FINISHED' and m_date > cutoff_date:
                                target_matches.append(m)

                        if not target_matches:
                            st.warning("Kriterlere uygun maç bulunamadı.")
                        else:
                            progress_bar = st.progress(0)
                            count = 0
                            for i, tm in enumerate(target_matches):
                                try:
                                    h_id, a_id = tm['homeTeam']['id'], tm['awayTeam']['id']
                                    hs = dm.get_stats(s, f, h_id); as_ = dm.get_stats(s, f, a_id)
                                    pars = {"t_h": (1,1), "t_a": (1,1), "weather": 1.0, "hk": False, "ak": False, "hgk": False, "agk": False, "power_diff": 0}
                                    dqi = 100; 
                                    if hs['played'] < 5: dqi -= 20
                                    res = eng.run_ensemble_analysis(hs, as_, 2.8, pars, h_id, a_id, lc)
                                    conf = int(max(res['1x2']) * (dqi/100.0))
                                    meta = {"hn": hs['name'], "an": as_['name'], "hid": h_id, "aid": a_id, "lg": lc, "conf": conf, "dqi": dqi}
                                    
                                    save_pred_db(tm, res['1x2'], pars, "Auto-Batch", meta)
                                    count += 1
                                except Exception as e: pass 
                                progress_bar.progress((i + 1) / len(target_matches))
                            
                            st.success(t["admin_batch_success"].format(count))

            # 2. SONUÇ DOĞRULAMA (PENDING)
            with adm_t2:
                if db is None:
                    st.error("Veritabanı bağlantısı yok!")
                else:
                    try:
                        # Limit 1000 ve Eski maçlar üstte
                        pend_ref = db.collection("predictions").where("actual_result", "==", None).limit(1000)
                        pend = list(pend_ref.stream())
                        pend.sort(key=lambda x: x.to_dict().get('match_date', '0000'), reverse=False)
                        
                        match_options = {}
                        seen_matches = set()

                        for d in pend:
                            data = d.to_dict()
                            # İsim bulma garantisi (Fallback)
                            label = data.get('match_name') or data.get('match') or f"Maç {d.id}"
                            date = str(data.get('match_date', ''))[:10]
                            unique_key = f"{label}_{date}"
                            
                            if unique_key not in seen_matches:
                                match_options[d.id] = f"{label} ({date})"
                                seen_matches.add(unique_key)
                        
                        if match_options:
                            with st.form("validation_form", clear_on_submit=False):
                                st.write("### 📝 Maç Sonucu Gir")
                                c_sel1, c_sel2 = st.columns([2, 1])
                                with c_sel1:
                                    selected_option_id = st.selectbox(t["admin_valid_sel"], options=list(match_options.keys()), format_func=lambda x: match_options[x])
                                with c_sel2:
                                    manual_id = st.text_input("Match ID (Manuel - Opsiyonel)")
                                
                                final_id = manual_id if manual_id else selected_option_id
                                c1, c2 = st.columns(2)
                                hs = c1.number_input("Home Goal", min_value=0, step=1)
                                as_ = c2.number_input("Away Goal", min_value=0, step=1)
                                note = st.text_area("Admin Note")
                                submitted = st.form_submit_button(t["admin_valid_btn"])
                                
                                if submitted:
                                    if not final_id:
                                        st.error("Lütfen bir maç seçin.")
                                    else:
                                        with st.spinner("Veritabanı güncelleniyor..."):
                                            if update_result_db(final_id, hs, as_, note):
                                                st.success(f"✅ {match_options.get(final_id, final_id)} başarıyla kaydedildi!")
                                                time.sleep(1.0)
                                                st.rerun()
                                            else:
                                                st.error("❌ Kayıt hatası.")
                        else:
                            st.info(t["msg_no_match"])
                    except Exception as e:
                        st.error(f"Panel Hatası: {e}")

            # 3. GEÇMİŞ (COMPLETED MATCHES)
            with adm_t3:
                if db:
                    try:
                        # [CRITICAL FIX] İsim Görünmeme Sorunu Çözümü
                        validated_refs = list(db.collection("predictions").where("validation_status", "==", "VALIDATED").limit(100).stream())
                        # Yeniden eskiye sırala
                        validated_refs.sort(key=lambda x: x.to_dict().get('match_date', '0000'), reverse=True)
                        
                        val_data = []
                        for v in validated_refs:
                            vd = v.to_dict()
                            # İsim yoksa 'match' anahtarına bak, o da yoksa ID yaz
                            match_label = vd.get("match_name") or vd.get("match") or f"Maç {v.id}"
                            
                            val_data.append({
                                "Match": match_label,
                                "Date": vd.get("match_date", "")[:10],
                                "Score": vd.get("actual_score") or f"{vd.get('home_score',0)}-{vd.get('away_score',0)}",
                                "Brier": f"{vd.get('brier_score', 0):.4f}",
                                "Note": vd.get("admin_notes")
                            })
                        
                        if val_data:
                            st.dataframe(pd.DataFrame(val_data))
                        else:
                            st.info("Henüz doğrulanmış maç yok.")
                    except Exception as e:
                        st.error(f"History Error: {e}")

    # TAB 3: MODEL CARD
    if is_admin and len(tabs) > 2:
        with tabs[2]:
            st.header(t["tab_model"])
            col_mc1, col_mc2 = st.columns([2,1])
            with col_mc1:
                st.code("""
                Architecture: Ensemble (Poisson + Dixon-Coles)
                Optimization: Elo-based Dynamic Weighting
                Validation Metric: Brier Score
                Risk Analysis: Volatility Index based on League Profiles
                Training Data: Last 5 Seasons / 10k+ Matches
                Update Frequency: Real-time
                """, language="yaml")
            with col_mc2:
                mc_bytes = create_model_card()
                st.download_button(t["dl_report"], mc_bytes, "model_card_v14.pdf", "application/pdf")

if __name__ == "__main__":
    main()
