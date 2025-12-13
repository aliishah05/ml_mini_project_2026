import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Group 10 - ML Mini-Project 2026",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Libre Baskerville', serif !important;
        color: #e8d5b7 !important;
    }
    
    .stMarkdown, .stText, p, span, label {
        font-family: 'Source Sans Pro', sans-serif !important;
        color: #c9d6df !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e3a5f, #0d2137);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #3d5a80;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .event-card {
        background: linear-gradient(145deg, #243b55, #141e30);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #e63946;
    }
    
    .explanation-box {
        background: rgba(230, 57, 70, 0.1);
        border: 1px solid #e63946;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e63946, #f77f00) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(230, 57, 70, 0.4) !important;
    }
    
    .sidebar .stSelectbox, .sidebar .stNumberInput {
        background: #1e3a5f !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #4ecdc4 !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    .stExpander {
        background: rgba(30, 58, 95, 0.5) !important;
        border: 1px solid #3d5a80 !important;
        border-radius: 10px !important;
    }
    /* Hide default arrow/text in st.expander */
    .stExpander > div > button {
        color: transparent !important;   /* hides the text */
    }
    .stExpander > div > button:focus {
        color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Models and Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load the trained XGBoost model and SHAP explainer."""
    try:
        with open('xgb_volume_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('shap_explainer.pkl', 'rb') as f:
            explainer = pickle.load(f)
        return model, explainer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

@st.cache_data
def load_rock_data():
    """Load and preprocess rock data."""
    try:
        df = pd.read_csv('all_data_cleaned.csv', sep=';', decimal=',')
        df.columns = df.columns.str.strip()
        
        # Convert Type_event to numeric
        df['Type_event_num'] = df['Type_event'].map({'rain': 0, 'snow': 1})
        
        # Get unique rock properties
        rock_cols = ['Rock_number', 'EC_rock', 'Ph_rock', 'Corg_rock (%)', 
                     'Ca_rock', 'K_rock', 'Mg_rock', 'Na_rock', 'SAR_rock',
                     'SiO2_rock', 'Al2O3_rock', 'Fe2O3_rock', 'TiO2_rock',
                     'MnO_rock', 'CaO_rock', 'MgO_rock', 'Na2O_rock', 
                     'K2O_rock', 'SO3_rock', 'P2O5_rock']
        rocks = df.groupby('Rock_number')[rock_cols].first().reset_index(drop=True)
        rocks.set_index('Rock_number', inplace=True)
        
        return df, rocks
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load resources
model, explainer = load_models()
df, rocks_data = load_rock_data()

# Feature names (must match training order)
FEATURE_NAMES = [
    'Volume_leachate_lag1', 'Volume_leachate_lag2', 'Volume_leachate_lag3',
    'Volume_leachate_roll_mean3', 'Volume_leachate_roll_std3',
    'Event_quantity_lag1', 'Event_quantity_lag2', 'Event_quantity_lag3',
    'Event_quantity_roll_mean3', 'Event_quantity_roll_std3',
    'Temp_lag1', 'Temp_lag2', 'Temp_lag3',
    'Temp_roll_mean3', 'Temp_roll_std3',
    'Acid_lag1', 'Acid_lag2', 'Acid_lag3',
    'Acid_roll_mean3', 'Acid_roll_std3',
    'Type_event_num_lag1', 'Type_event_num_lag2', 'Type_event_num_lag3',
    'Type_event_num_roll_mean3', 'Type_event_num_roll_std3'
]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Group 10 - Rock Leachate Volume Predictor")
st.markdown("### Syed Muhammad Ali SHAH - Muhammad Usman ASGHER - Syed Umar HASANY")
st.markdown("### Predict leachate volume from rock characteristics and event sequences")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar - Rock Selection
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏔️ Rock Selection")

rock_options = ['Custom'] + sorted(rocks_data.index.tolist())
rock_selection = st.sidebar.selectbox(
    "Choose a rock or enter custom values",
    options=rock_options,
    format_func=lambda x: f"Rock {x}" if x != 'Custom' else "✏️ Custom Values"
)

if rock_selection != 'Custom':
    selected_rock = rocks_data.loc[rock_selection]
    st.sidebar.success(f"✓ Loaded Rock {rock_selection}")
    
    # Display rock properties directly (always visible)
    st.sidebar.markdown("### 📊 Rock Properties")
    st.sidebar.markdown("---")
    for col in selected_rock.index:
        if col != 'Rock_number':
            st.sidebar.markdown(f"**{col}:** `{selected_rock[col]:.4f}`")
    st.sidebar.markdown("---")
else:
    selected_rock = None
    st.sidebar.info("⚠️ Please select a rock to view its properties")

# ─────────────────────────────────────────────────────────────────────────────
# Main Content - Event Sequence Input
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📋 Event Sequence Configuration")
st.markdown("""
Define your sequence of events **S = [(e₁, a₁, t₁), ..., (eₙ, aₙ, tₙ)]** where:
- **eᵢ** = Event type (rain or snow)
- **aᵢ** = Acid treatment (0 = No, 1 = Yes)
- **tᵢ** = Temperature (-5°C or 50°C)
""")

# Number of events
num_events = st.slider(
    "Number of events in sequence",
    min_value=1,
    max_value=15,
    value=5,
    help="Select how many events to simulate"
)

# Event input grid
st.markdown("### Define Each Event")

events = []
cols_per_row = 3

for i in range(num_events):
    with st.container():
        st.markdown(f"**Event {i+1}**")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            event_type = st.selectbox(
                "Type",
                options=['rain', 'snow'],
                key=f"type_{i}",
                label_visibility="collapsed"
            )
        
        with col2:
            acid = st.selectbox(
                "Acid",
                options=[0, 1],
                format_func=lambda x: '💧 No Acid' if x == 0 else '⚗️ Acid',
                key=f"acid_{i}",
                label_visibility="collapsed"
            )
        
        with col3:
            temp = st.selectbox(
                "Temperature",
                options=[-5, 50],
                format_func=lambda x: f"❄️ {x}°C" if x == -5 else f"🔥 {x}°C",
                key=f"temp_{i}",
                label_visibility="collapsed"
            )
        
        with col4:
            # Event quantity based on type
            event_qty = 140 if event_type == 'rain' else 170
            st.markdown(f"**{event_qty}ml**")
        
        events.append({
            'type': event_type,
            'type_num': 0 if event_type == 'rain' else 1,
            'acid': acid,
            'temp': temp,
            'event_quantity': event_qty
        })

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────────────────────
def build_features(event_history, current_event):
    """
    Build the 25-feature vector for prediction.
    
    Features (in order):
    - Volume_leachate: lag1, lag2, lag3, roll_mean3, roll_std3
    - Event_quantity: lag1, lag2, lag3, roll_mean3, roll_std3
    - Temp: lag1, lag2, lag3, roll_mean3, roll_std3
    - Acid: lag1, lag2, lag3, roll_mean3, roll_std3
    - Type_event_num: lag1, lag2, lag3, roll_mean3, roll_std3
    """
    # Get historical values (pad with defaults if not enough history)
    default_volume = 125.0  # median volume from training data
    default_qty = 140.0
    default_temp = -5.0
    default_acid = 0.0
    default_type = 0.0
    
    n = len(event_history)
    
    def get_lag_values(history, key, default, n):
        """Get lag1, lag2, lag3 values from history, padded with defaults."""
        lag1 = history[-1][key] if n >= 1 else default
        lag2 = history[-2][key] if n >= 2 else default
        lag3 = history[-3][key] if n >= 3 else default
        return lag1, lag2, lag3
    
    def get_rolling_stats(history, key, default, n):
        """Get rolling mean and std over last 3 values."""
        vals = []
        for k in range(1, 4):  # k=1,2,3 for the last 3 events
            if n >= k:
                vals.append(history[-k][key])
            else:
                vals.append(default)
        roll_mean = np.mean(vals)
        roll_std = np.std(vals) if len(vals) > 1 else 0.0
        return roll_mean, roll_std
    
    # Volume lags and rolling
    vol_lag1, vol_lag2, vol_lag3 = get_lag_values(event_history, 'volume', default_volume, n)
    vol_roll_mean, vol_roll_std = get_rolling_stats(event_history, 'volume', default_volume, n)
    
    # Event quantity lags and rolling
    qty_lag1, qty_lag2, qty_lag3 = get_lag_values(event_history, 'event_quantity', default_qty, n)
    qty_roll_mean, qty_roll_std = get_rolling_stats(event_history, 'event_quantity', default_qty, n)
    
    # Temperature lags and rolling
    temp_lag1, temp_lag2, temp_lag3 = get_lag_values(event_history, 'temp', default_temp, n)
    temp_roll_mean, temp_roll_std = get_rolling_stats(event_history, 'temp', default_temp, n)
    
    # Acid lags and rolling
    acid_lag1, acid_lag2, acid_lag3 = get_lag_values(event_history, 'acid', default_acid, n)
    acid_roll_mean, acid_roll_std = get_rolling_stats(event_history, 'acid', default_acid, n)
    
    # Type_event_num lags and rolling
    type_lag1, type_lag2, type_lag3 = get_lag_values(event_history, 'type_num', default_type, n)
    type_roll_mean, type_roll_std = get_rolling_stats(event_history, 'type_num', default_type, n)
    
    # Build feature vector in exact order
    features = np.array([[
        vol_lag1, vol_lag2, vol_lag3, vol_roll_mean, vol_roll_std,
        qty_lag1, qty_lag2, qty_lag3, qty_roll_mean, qty_roll_std,
        temp_lag1, temp_lag2, temp_lag3, temp_roll_mean, temp_roll_std,
        acid_lag1, acid_lag2, acid_lag3, acid_roll_mean, acid_roll_std,
        type_lag1, type_lag2, type_lag3, type_roll_mean, type_roll_std
    ]])
    
    return features

def get_plain_language_explanation(shap_values, feature_names, features, prediction, baseline):
    """Generate a plain-language explanation for non-experts."""
    
    # Get top 5 contributing features
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-5:][::-1]
    
    explanation_parts = []
    
    # Map feature names to human-readable descriptions
    feature_descriptions = {
        'Volume_leachate_lag1': 'the previous event\'s volume',
        'Volume_leachate_lag2': 'volume from 2 events ago',
        'Volume_leachate_lag3': 'volume from 3 events ago',
        'Volume_leachate_roll_mean3': 'average volume over last 3 events',
        'Volume_leachate_roll_std3': 'variability in recent volumes',
        'Event_quantity_lag1': 'previous water/snow amount',
        'Event_quantity_lag2': 'water amount from 2 events ago',
        'Event_quantity_lag3': 'water amount from 3 events ago',
        'Event_quantity_roll_mean3': 'average water amount recently',
        'Event_quantity_roll_std3': 'variability in water amounts',
        'Temp_lag1': 'previous temperature',
        'Temp_lag2': 'temperature from 2 events ago',
        'Temp_lag3': 'temperature from 3 events ago',
        'Temp_roll_mean3': 'average recent temperature',
        'Temp_roll_std3': 'temperature variability',
        'Acid_lag1': 'previous acid treatment',
        'Acid_lag2': 'acid treatment 2 events ago',
        'Acid_lag3': 'acid treatment 3 events ago',
        'Acid_roll_mean3': 'frequency of acid treatments',
        'Acid_roll_std3': 'acid treatment pattern',
        'Type_event_num_lag1': 'previous event type (rain/snow)',
        'Type_event_num_lag2': 'event type 2 events ago',
        'Type_event_num_lag3': 'event type 3 events ago',
        'Type_event_num_roll_mean3': 'recent rain/snow mix',
        'Type_event_num_roll_std3': 'weather pattern variability'
    }
    
    for idx in top_indices:
        feat_name = feature_names[idx]
        shap_val = shap_values[idx]
        feat_val = features[0, idx]
        desc = feature_descriptions.get(feat_name, feat_name)
        
        direction = "increased" if shap_val > 0 else "decreased"
        impact = abs(shap_val)
        
        if impact > 1:
            strength = "significantly"
        elif impact > 0.5:
            strength = "moderately"
        else:
            strength = "slightly"
        
        explanation_parts.append(f"• **{desc.capitalize()}** ({feat_val:.1f}) {strength} {direction} the prediction by {impact:.2f} ml")
    
    return explanation_parts

# ─────────────────────────────────────────────────────────────────────────────
# Prediction Button and Results
# ─────────────────────────────────────────────────────────────────────────────

# Check if rock is selected
rock_selected = rock_selection != 'Custom'

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if not rock_selected:
        st.warning("⚠️ Please select a rock from the sidebar before making predictions.")
        predict_button = st.button("🔮 Predict Leachate Volume for All Events", type="primary", use_container_width=True, disabled=True)
    else:
        st.success(f"✓ Rock {rock_selection} selected. Ready to predict!")
        predict_button = st.button("🔮 Predict Leachate Volume for All Events", type="primary", use_container_width=True)

if predict_button and rock_selected:
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Run predictions sequentially
    predictions = []
    event_history = []
    
    progress_bar = st.progress(0)
    
    for idx, event in enumerate(events):
        # Build features based on history
        features = build_features(event_history, event)
        
        # Predict
        pred_volume = model.predict(features)[0]
        
        # Get SHAP values
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Store prediction
        predictions.append({
            'event_num': idx + 1,
            'type': event['type'],
            'acid': event['acid'],
            'temp': event['temp'],
            'event_quantity': event['event_quantity'],
            'predicted_volume': pred_volume,
            'features': features,
            'shap_values': shap_vals
        })
        
        # Add to history for next prediction
        event_history.append({
            'volume': pred_volume,
            'event_quantity': event['event_quantity'],
            'temp': event['temp'],
            'acid': event['acid'],
            'type_num': event['type_num']
        })
        
        progress_bar.progress((idx + 1) / len(events))
    
    progress_bar.empty()
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary Chart
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Volume Predictions Over Event Sequence")
    
    # Create summary dataframe
    summary_df = pd.DataFrame([{
        'Event': f"E{p['event_num']}",
        'Type': p['type'].title(),
        'Acid': 'Yes' if p['acid'] == 1 else 'No',
        'Temp (°C)': p['temp'],
        'Predicted Volume (ml)': round(p['predicted_volume'], 2)
    } for p in predictions])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    colors = ['#e63946' if p['type'] == 'rain' else '#4ecdc4' for p in predictions]
    bars = ax.bar(range(len(predictions)), 
                  [p['predicted_volume'] for p in predictions],
                  color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='white', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Event Number', color='white', fontsize=12)
    ax.set_ylabel('Predicted Volume (ml)', color='white', fontsize=12)
    ax.set_title('Leachate Volume Predictions', color='#e8d5b7', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(predictions)))
    ax.set_xticklabels([f"E{i+1}" for i in range(len(predictions))], color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=explainer.expected_value, color='#f77f00', linestyle='--', 
               label=f'Baseline: {explainer.expected_value:.1f} ml', linewidth=2)
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Summary table
    st.markdown("### 📋 Summary Table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # Detailed Explanations for Each Event
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔍 Detailed Explanations")
    st.markdown("*Explanations for each event showing what factors influenced the prediction*")


    for pred in predictions:
        event_icon = "🌧️" if pred['type'] == 'rain' else "❄️"
        acid_icon = "⚗️" if pred['acid'] == 1 else "💧"
        temp_icon = "🔥" if pred['temp'] == 50 else "🧊"
        
        # Event header (always visible, no dropdown)
        st.markdown("---")
        st.markdown(f"### {event_icon} Event {pred['event_num']}: {pred['type'].title()} | {acid_icon} | {temp_icon} {pred['temp']}°C → **{pred['predicted_volume']:.1f} ml**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Predicted Volume",
                value=f"{pred['predicted_volume']:.1f} ml",
                delta=f"{pred['predicted_volume'] - explainer.expected_value:.1f} ml from baseline"
            )
            
            st.markdown("**Event Details:**")
            st.markdown(f"- Type: {pred['type'].title()}")
            st.markdown(f"- Acid Treatment: {'Yes' if pred['acid'] == 1 else 'No'}")
            st.markdown(f"- Temperature: {pred['temp']}°C")
            st.markdown(f"- Water Amount: {pred['event_quantity']} ml")
        
        with col2:
            st.markdown("#### 💡 Why This Prediction?")
            st.markdown(f"""
            The model predicted **{pred['predicted_volume']:.1f} ml** of leachate volume.
            Starting from a baseline of **{explainer.expected_value:.1f} ml**, the following factors 
            influenced this prediction:
            """)
            
            # Get plain language explanation
            explanations = get_plain_language_explanation(
                pred['shap_values'], 
                FEATURE_NAMES, 
                pred['features'],
                pred['predicted_volume'],
                explainer.expected_value
            )
            
            for exp in explanations:
                st.markdown(exp)
        
        # SHAP waterfall plot
        st.markdown("#### 📊 Feature Impact Visualization")
        
        fig_shap, ax_shap = plt.subplots(figsize=(12, 6))
        fig_shap.patch.set_facecolor('#1a1a2e')
        ax_shap.set_facecolor('#16213e')
        
        # Sort features by absolute SHAP value
        sorted_idx = np.argsort(np.abs(pred['shap_values']))[-10:]
        sorted_shap = pred['shap_values'][sorted_idx]
        sorted_names = [FEATURE_NAMES[i].replace('_', ' ').replace('leachate ', '') for i in sorted_idx]
        
        # Calculate percentages (based on absolute contribution)
        total_abs_shap = np.sum(np.abs(pred['shap_values']))
        sorted_percentages = (np.abs(sorted_shap) / total_abs_shap) * 100 if total_abs_shap > 0 else np.zeros_like(sorted_shap)
        
        colors_bar = ['#e63946' if v > 0 else '#4ecdc4' for v in sorted_shap]
        
        bars = ax_shap.barh(range(len(sorted_idx)), sorted_shap, color=colors_bar, edgecolor='white', height=0.7)
        ax_shap.set_yticks(range(len(sorted_idx)))
        ax_shap.set_yticklabels(sorted_names, color='white', fontsize=10)
        ax_shap.set_xlabel('SHAP Value (Impact on Prediction)', color='white', fontsize=11)
        ax_shap.axvline(x=0, color='white', linewidth=0.8)
        ax_shap.tick_params(colors='white')
        ax_shap.spines['bottom'].set_color('white')
        ax_shap.spines['left'].set_color('white')
        ax_shap.spines['top'].set_visible(False)
        ax_shap.spines['right'].set_visible(False)
        
        # Add percentage labels on bars
        for bar, shap_val, pct in zip(bars, sorted_shap, sorted_percentages):
            width = bar.get_width()
            # Position label outside the bar
            if width >= 0:
                label_x = width + 0.1
                ha = 'left'
            else:
                label_x = width - 0.1
                ha = 'right'
            
            ax_shap.annotate(
                f'{pct:.1f}%',
                xy=(label_x, bar.get_y() + bar.get_height() / 2),
                ha=ha, va='center',
                color='#f77f00', fontsize=10, fontweight='bold'
            )
            
            # Also show the actual SHAP value inside/near the bar
            if abs(width) > 0.5:
                ax_shap.annotate(
                    f'{shap_val:.2f}',
                    xy=(width / 2, bar.get_y() + bar.get_height() / 2),
                    ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold'
                )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e63946', label='Increases Volume'),
            Patch(facecolor='#4ecdc4', label='Decreases Volume')
        ]
        ax_shap.legend(handles=legend_elements, loc='lower right', 
                      facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
        
        # Add title showing total contribution of top 10
        top10_pct = np.sum(sorted_percentages)
        ax_shap.set_title(f'Top 10 Features (explaining {top10_pct:.1f}% of prediction)', 
                         color='#e8d5b7', fontsize=12, fontweight='bold', pad=10)
        
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🪨 Group 10 - Rock Leachate Predictor | Built with XGBoost & SHAP</p>
    <p style='font-size: 0.8em;'>Model achieves R² = 0.926 on test data</p>
</div>
""", unsafe_allow_html=True)
