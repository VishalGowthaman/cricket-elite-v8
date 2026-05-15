import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. SETUP & PAGE CONFIG
st.set_page_config(page_title="Cricket Elite V8", page_icon="🏏", layout="wide")

@st.cache_resource
def load_assets():
    # REMOVE the D:\ check. Just check if the files exist in the current directory.
    files = ['cricket_v8_final_global.pkl', 'stadiums.csv', 
             'pure_death_batter_dna.csv', 'pure_death_bowler_dna.csv']
    
    if not all(os.path.exists(f) for f in files):
        return None, None, None, None
        
    model = joblib.load('cricket_v8_final_global.pkl')
    stadium_df = pd.read_csv('stadiums.csv')
    bat_dna = pd.read_csv('pure_death_batter_dna.csv').set_index('player_name')
    bowl_dna = pd.read_csv('pure_death_bowler_dna.csv').set_index('player_name')
    return model, stadium_df, bat_dna, bowl_dna

# 2. ERROR CHECKING
if model is None:
    st.error("❌ Missing Files! Ensure PKL and all CSVs are in D:\GCT\mini_project")
    st.stop()

st.title("🏏 Cricket Elite V8: Death Over Predictor")
st.markdown("---")

# 3. UI LAYOUT
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏟️ Match Context")
    b_team = st.selectbox("Batting Team", ["IND", "AUS", "ENG", "SA", "PAK", "NZ", "WI", "SL", "AFG", "BAN"])
    f_team = st.selectbox("Bowling Team", ["IND", "AUS", "ENG", "SA", "PAK", "NZ", "WI", "SL", "AFG", "BAN"])
    venue = st.selectbox("Select Stadium", stadium_df['stadium_name'].unique())
    inn = st.radio("Innings", [1, 2], horizontal=True)
    
    st.subheader("🧬 Player DNA")
    batters = st.text_input("Enter Batters (e.g., V KOHLI, HH PANDYA)", "V KOHLI, HH PANDYA")
    bowlers = st.text_input("Enter Death Bowlers (e.g., JJ BUMRAH, MA STARC)", "JJ BUMRAH, MA STARC")

with col2:
    st.subheader("📊 Live Match Stats")
    score = st.number_input("Score at 15.0 Overs", value=120)
    wicks = st.slider("Wickets Lost", 0, 9, 3)
    l5 = st.number_input("Runs in last 5 overs (10-15)", value=45)
    m15 = st.number_input("Runs in 15th over", value=12)
    set_b = st.slider("Balls faced by set batter", 0, 80, 25)

st.markdown("---")

# 4. PREDICTION LOGIC
if st.button("🚀 RUN AI PREDICTION", use_container_width=True):
    # --- Stadium Stats ---
    st_data = stadium_df[stadium_df['stadium_name'] == venue]
    h15 = 120
    for c in ['avg_score_15', 'average_15', 'hist_avg_15']:
        if not st_data.empty and c in st_data.columns:
            h15 = float(st_data[c].iloc[0]); break
    
    bnd = 65
    if not st_data.empty and 'avg_boundary_min' in st_data.columns:
        bnd = float(st_data['avg_boundary_min'].iloc[0])

    # --- DNA Lookup Logic ---
    b_list = [name.strip().upper() for name in batters.split(',')]
    f_list = [name.strip().upper() for name in bowlers.split(',')]
    
    sr_vals = [bat_dna.loc[p].mean() if p in bat_dna.index else 135.0 for p in b_list]
    m_sr = np.mean(sr_vals)
    
    eco_vals = [bowl_dna.loc[p, 'death_eco_overall'] if p in bowl_dna.index else 10.5 for p in f_list]
    m_eco = np.mean(eco_vals)

    # --- Feature Engineering ---
    pf = score / h15 if h15 > 0 else 1.0
    pwr = (l5 / 30 * 6) * (10 - wicks)
    
    # Feature vector for the Stacking Regressor
    features = pd.DataFrame([[
        score, 10-wicks, pwr, m_sr, m_eco, m15, set_b, bnd, pf, inn
    ]], columns=['s15', 'wL', 'pwr', 'matchup_sr', 'matchup_eco', 
                 'mom15', 'set_factor', 'boundary', 'pitch_factor', 'inn'])
    
    # --- Prediction ---
    pred_death_runs = model.predict(features)[0]
    final_total = score + pred_death_runs
    
    # 5. RESULTS DISPLAY
    st.balloons()
    st.subheader(f"Projected Scorecard: {b_team} vs {f_team}")
    
    # Just showing the two main results now
    res1, res2 = st.columns(2)
    
    with res1:
        st.metric("Predicted Death Over Score (16-20)", f"{round(pred_death_runs)} runs")
        st.info("Based on current momentum and player matchups.")

    with res2:
        st.metric("Projected Final Total", f"{round(final_total)} runs")
        st.success(f"Expected final score for {b_team} at {venue}.")