import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ---- Page Configuration ----
st.set_page_config(
    page_title="⚽ EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Load Models ----
@st.cache_resource
def load_models():
    models = {}
    model_path = 'models/'
    
    try:
        models['outcome'] = joblib.load(os.path.join(model_path, 'model_outcome.pkl'))
        models['home_goals'] = joblib.load(os.path.join(model_path, 'model_home_goals.pkl'))
        models['away_goals'] = joblib.load(os.path.join(model_path, 'model_away_goals.pkl'))
        models['over25'] = joblib.load(os.path.join(model_path, 'model_over25.pkl'))
        models['btts'] = joblib.load(os.path.join(model_path, 'model_btts.pkl'))
        models['corners'] = joblib.load(os.path.join(model_path, 'model_corners.pkl'))
        models['imputer'] = joblib.load(os.path.join(model_path, 'imputer.pkl'))
        models['loaded'] = True
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        models['loaded'] = False
    
    try:
        models['team_stats'] = joblib.load(os.path.join(model_path, 'team_stats_lookup.pkl'))
        models['lookup_loaded'] = True
        # Get sorted list of team names
        models['team_list'] = sorted(models['team_stats'].keys())
    except:
        models['team_stats'] = {}
        models['lookup_loaded'] = False
        models['team_list'] = []
        st.warning("⚠️ Team stats lookup not found. Please run the precomputation script in your notebook.")
    
    return models

# ---- Prediction Function ----
def predict_match(home_team, away_team, models):
    """Fetch stats from lookup and predict"""
    team_stats = models['team_stats']
    
    if home_team not in team_stats or away_team not in team_stats:
        return None, "Team not found in lookup. Please check spelling or run precomputation."
    
    hs = team_stats[home_team]
    aw = team_stats[away_team]
    
    # Build feature dictionary
    data = {
        'HTGS': int(hs.get('HTGS', 0)),
        'ATGS': int(aw.get('ATGS', 0)),
        'HTGC': int(hs.get('HTGC', 0)),
        'ATGC': int(aw.get('ATGC', 0)),
        'HTP': int(hs.get('HTP', 0)),
        'ATP': int(aw.get('ATP', 0)),
        'HTGD': int(hs.get('HTGD', 0)),
        'ATGD': int(aw.get('ATGD', 0)),
        'DiffPts': int(hs.get('HTP', 0) - aw.get('ATP', 0)),
        'DiffFormPts': int(hs.get('HTFormPts', 0) - aw.get('ATFormPts', 0)),
        'HTFormPts': int(min(15, max(0, hs.get('HTFormPts', 0)))),
        'ATFormPts': int(min(15, max(0, aw.get('ATFormPts', 0)))),
        'B365H': float(hs.get('B365H', 1.50)),
        'B365D': float(hs.get('B365D', 3.50)),
        'B365A': float(hs.get('B365A', 2.50)),
    }
    
    # Prepare features in correct order
    feature_order = ['HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 
                     'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts',
                     'HTFormPts', 'ATFormPts', 
                     'B365H', 'B365D', 'B365A']
    
    input_df = pd.DataFrame([data])[feature_order]
    input_imputed = models['imputer'].transform(input_df)
    
    # 1. Outcome
    probs = models['outcome'].predict_proba(input_imputed)[0]
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    pred_idx = np.argmax(probs)
    prediction = outcome_labels[pred_idx]
    confidence = probs[pred_idx]
    
    # 2. Exact Score
    home_goals = max(0, round(models['home_goals'].predict(input_imputed)[0]))
    away_goals = max(0, round(models['away_goals'].predict(input_imputed)[0]))
    
    # Fix consistency
    if pred_idx == 0:  # Home Win
        if home_goals <= away_goals:
            home_goals = away_goals + 1
    elif pred_idx == 1:  # Draw
        avg_goals = (home_goals + away_goals) // 2
        home_goals = avg_goals
        away_goals = avg_goals
    else:  # Away Win
        if away_goals <= home_goals:
            away_goals = home_goals + 1
    
    # 3. Over/Under 2.5
    prob_over25 = models['over25'].predict_proba(input_imputed)[0][1]
    over25_pred = "Over 2.5" if prob_over25 >= 0.5 else "Under 2.5"
    
    # 4. BTTS
    prob_btts = models['btts'].predict_proba(input_imputed)[0][1]
    btts_pred = "Yes" if prob_btts >= 0.5 else "No"
    
    # 5. Corners
    total_corners = max(0, models['corners'].predict(input_imputed)[0])
    
    # Confidence level
    if confidence >= 0.60:
        confidence_level = "🔒 High"
    elif confidence >= 0.45:
        confidence_level = "⚖️ Medium"
    else:
        confidence_level = "⚠️ Low"
    
    result = {
        'outcome': prediction,
        'confidence': f"{confidence_level} ({confidence*100:.1f}%)",
        'probabilities': {
            'Home Win': probs[0],
            'Draw': probs[1],
            'Away Win': probs[2]
        },
        'score': f"{home_goals} - {away_goals}",
        'over_under': f"{over25_pred} ({prob_over25*100:.1f}%)",
        'btts': f"{btts_pred} ({prob_btts*100:.1f}%)",
        'corners': f"{total_corners:.1f}",
        'key_stats': data,
        'betting_odds': {'B365H': data['B365H'], 'B365D': data['B365D'], 'B365A': data['B365A']}
    }
    return result, None

# ---- Load models ----
models = load_models()
models_loaded = models['loaded']

# ---- App Title ----
st.title("⚽ EPL Match Predictor")
st.markdown("*Predict match outcomes, exact scores, corners, and more using machine learning*")

# ---- Sidebar ----
with st.sidebar:
    st.header("📊 Select Teams")
    
    if models['lookup_loaded']:
        team_list = models['team_list']
        home_team = st.selectbox("🏠 Home Team", team_list, index=0)
        away_team = st.selectbox("✈️ Away Team", team_list, index=min(1, len(team_list)-1))
    else:
        st.warning("Team lookup not available. Please run precomputation.")
        home_team = st.text_input("🏠 Home Team", "Arsenal")
        away_team = st.text_input("✈️ Away Team", "Chelsea")
    
    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ---- Main Content ----
if predict_clicked:
    if not models_loaded:
        st.error("❌ Models not loaded. Please check the models folder.")
    elif not models['lookup_loaded']:
        st.error("❌ Team lookup not loaded. Run precomputation in your notebook.")
    else:
        result, error = predict_match(home_team, away_team, models)
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Prediction Complete!")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"🏆 Match: {home_team} vs {away_team}")
                st.markdown(f"### 🎯 Prediction: **{result['outcome']}**")
                st.markdown(f"### ✅ Confidence: {result['confidence']}")
                st.markdown("### 📊 Outcome Probabilities:")
                for outcome, prob in result['probabilities'].items():
                    bar = "█" * int(prob * 20)
                    st.markdown(f"- {outcome}: {prob*100:.1f}% {bar}")
            
            with col2:
                st.markdown("### ⚽ Match Stats")
                st.metric("Exact Score", result['score'])
                st.metric("Over/Under 2.5", result['over_under'])
                st.metric("BTTS", result['btts'])
                st.metric("Total Corners", result['corners'])
            
            st.markdown("### 📈 Key Stats Used")
            stats = result['key_stats']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Home Goals Scored", stats['HTGS'])
                st.metric("Home Goals Conceded", stats['HTGC'])
                st.metric("Home Points", stats['HTP'])
            with col2:
                st.metric("Away Goals Scored", stats['ATGS'])
                st.metric("Away Goals Conceded", stats['ATGC'])
                st.metric("Away Points", stats['ATP'])
            with col3:
                st.metric("Points Diff", stats['DiffPts'])
                st.metric("Form Diff", stats['DiffFormPts'])
            
            st.markdown("### 💡 Value Bet Analysis")
            odds = result['betting_odds']
            bookmaker_implied = 1 / np.array([odds['B365H'], odds['B365D'], odds['B365A']])
            probs_arr = np.array(list(result['probabilities'].values()))
            edge = probs_arr - bookmaker_implied
            labels = ['Home Win', 'Draw', 'Away Win']
            best_bet_idx = np.argmax(edge)
            
            if edge[best_bet_idx] > 0.05:
                st.success(f"🎯 **Value Bet:** {labels[best_bet_idx]} (Edge: {edge[best_bet_idx]*100:.1f}%)")
                st.info("The model gives this outcome a higher probability than the bookmaker odds suggest!")
            else:
                st.info("📊 No clear value bet detected at current odds.")
else:
    st.info("👈 Select teams and click 'Predict'")
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1. Select Home and Away teams from the dropdown.
    2. Click 'Predict' – the system automatically fetches pre‑match stats.
    3. Get instant predictions for:
    - **Match Outcome** (Home/Draw/Away) with confidence
    - **Exact Score**
    - **Over/Under 2.5 Goals**
    - **Both Teams to Score (BTTS)**
    - **Total Corners**
    - **Value Bet** detection
    """)
    if not models['lookup_loaded']:
        st.warning("⚠️ Team lookup not loaded. Run the precomputation script in your Jupyter notebook to enable auto‑fetch.")
        with st.expander("ℹ️ How to enable auto‑fetch"):
            st.code("""
# In your notebook, after you have the 'data' DataFrame with all features:
import joblib
teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
team_stats = {}
for team in teams:
    # ... compute stats as shown in previous instructions ...
    team_stats[team] = { ... }
joblib.dump(team_stats, 'models/team_stats_lookup.pkl')
""", language='python')

# ---- Footer ----
st.markdown("---")
st.caption("⚽ EPL Match Predictor • Built with Python, XGBoost & Streamlit")