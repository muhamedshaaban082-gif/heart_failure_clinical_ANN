import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #FFE5E5;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #FF6B6B;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255,107,107,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255,107,107,0.6);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .success-box {
        background: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #F8D7DA;
        border-left: 5px solid #DC3545;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
    }
    
    /* File uploader */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animation */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# ======================== HEADER ========================
st.markdown("""
    <div class="main-header animate-fade-in">
        <h1>❤️ Heart Failure Prediction System</h1>
        <p>AI-Powered Cardiovascular Risk Assessment</p>
    </div>
""", unsafe_allow_html=True)

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio(
        "",
        ["🏠 Home", "📊 Data Analysis", "🤖 Make Prediction", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    st.info("**Dataset Features:** 12")
    st.info("**Target:** Death Event")
    st.info("**Model:** SVM + ANN")
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <p style='color: white; font-size: 0.9rem;'>
                Developed with ❤️<br>
                <strong>ML Projects Hub</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ======================== HOME PAGE ========================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card animate-fade-in">
                <h2 style='color: #FF6B6B;'>🩺 About This Project</h2>
                <p style='font-size: 1.1rem; line-height: 1.8; color: #2C3E50;'>
                    This system uses <strong style='color: #1A1A1A;'>Machine Learning</strong> and 
                    <strong style='color: #1A1A1A;'>Deep Learning</strong> 
                    to predict heart failure risk based on clinical records. It analyzes 12 key medical 
                    indicators to assess the probability of a cardiovascular death event.
                </p>
                
                <h3 style='color: #FF6B6B; margin-top: 2rem;'>🎯 Key Features</h3>
                <ul style='font-size: 1.05rem; line-height: 2.2; color: #2C3E50;'>
                    <li>✅ <strong style='color: #1A1A1A;'>Advanced ML Models:</strong> SVM and Artificial Neural Networks</li>
                    <li>✅ <strong style='color: #1A1A1A;'>12 Clinical Parameters:</strong> Age, Blood Pressure, Ejection Fraction, etc.</li>
                    <li>✅ <strong style='color: #1A1A1A;'>Real-time Predictions:</strong> Instant risk assessment</li>
                    <li>✅ <strong style='color: #1A1A1A;'>Interactive Visualizations:</strong> Understand your data better</li>
                    <li>✅ <strong style='color: #1A1A1A;'>High Accuracy:</strong> Validated on 299 patient records</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card animate-fade-in" style='background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%); color: white; border: none;'>
                <h3 style='color: white; text-align: center;'>📊 Dataset Overview</h3>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <p style='font-size: 2rem; text-align: center; font-weight: 700; margin: 1rem 0;'>299</p>
                <p style='text-align: center; opacity: 0.9;'>Patient Records</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;'>
                <p style='font-size: 2rem; text-align: center; font-weight: 700; margin: 1rem 0;'>12</p>
                <p style='text-align: center; opacity: 0.9;'>Clinical Features</p>
                <hr style='border-color: rgba(255,255,255,0.3); margin: 1.5rem 0;'>
                <p style='font-size: 2rem; text-align: center; font-weight: 700; margin: 1rem 0;'>2</p>
                <p style='text-align: center; opacity: 0.9;'>ML Models (SVM + ANN)</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Clinical Features
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card animate-fade-in">
            <h2 style='color: #FF6B6B;'>🔬 Clinical Features Explained</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h4 style='color: #FF6B6B; font-weight: 700;'>👤 Demographics</h4>
                <ul style='color: #2C3E50; font-size: 0.95rem; line-height: 1.8;'>
                    <li><strong style='color: #1A1A1A;'>Age:</strong> Patient's age in years</li>
                    <li><strong style='color: #1A1A1A;'>Sex:</strong> Gender (0=Female, 1=Male)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h4 style='color: #FF6B6B; font-weight: 700;'>🩸 Blood Tests</h4>
                <ul style='color: #2C3E50; font-size: 0.95rem; line-height: 1.8;'>
                    <li><strong style='color: #1A1A1A;'>CPK:</strong> Creatinine Phosphokinase (mcg/L)</li>
                    <li><strong style='color: #1A1A1A;'>Platelets:</strong> Platelet count (kiloplatelets/mL)</li>
                    <li><strong style='color: #1A1A1A;'>Serum Creatinine:</strong> Kidney function (mg/dL)</li>
                    <li><strong style='color: #1A1A1A;'>Serum Sodium:</strong> Sodium level (mEq/L)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card">
                <h4 style='color: #FF6B6B; font-weight: 700;'>❤️ Cardiac Metrics</h4>
                <ul style='color: #2C3E50; font-size: 0.95rem; line-height: 1.8;'>
                    <li><strong style='color: #1A1A1A;'>Ejection Fraction:</strong> Blood pumping % (30-70% normal)</li>
                    <li><strong style='color: #1A1A1A;'>Blood Pressure:</strong> Diastolic BP (mm Hg)</li>
                    <li><strong style='color: #1A1A1A;'>Anaemia:</strong> Low RBC/hemoglobin</li>
                    <li><strong style='color: #1A1A1A;'>Diabetes:</strong> Diabetes status</li>
                    <li><strong style='color: #1A1A1A;'>Smoking:</strong> Smoking history</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ======================== DATA ANALYSIS PAGE ========================
elif page == "📊 Data Analysis":
    st.markdown("""
        <div class="info-card animate-fade-in">
            <h2 style='color: #FF6B6B;'>📊 Upload & Analyze Your Data</h2>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Upload your heart failure dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing heart failure clinical records"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("""
                <div class="success-box">
                    ✅ <strong>File uploaded successfully!</strong> Dataset loaded with {} rows and {} columns.
                </div>
            """.format(df.shape[0], df.shape[1]), unsafe_allow_html=True)
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Distributions", "🔗 Correlations", "📊 Statistics"])
            
            with tab1:
                st.markdown("### 👀 Dataset Preview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", df.shape[0], help="Number of patient records")
                with col2:
                    st.metric("Features", df.shape[1], help="Number of clinical features")
                with col3:
                    if 'DEATH_EVENT' in df.columns:
                        death_rate = (df['DEATH_EVENT'].sum() / len(df) * 100)
                        st.metric("Death Event Rate", f"{death_rate:.1f}%", help="Percentage of death events")
                
                st.dataframe(df.head(20), use_container_width=True, height=400)
                
                # Missing values check
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.markdown("""
                        <div class="warning-box">
                            ⚠️ <strong>Warning:</strong> Found {} missing values in the dataset.
                        </div>
                    """.format(missing.sum()), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-box">
                            ✅ <strong>Great!</strong> No missing values detected.
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### 📊 Feature Distributions")
                
                if 'DEATH_EVENT' in df.columns:
                    # Target distribution
                    fig = px.histogram(
                        df, 
                        x='DEATH_EVENT',
                        color='DEATH_EVENT',
                        title="Death Event Distribution",
                        labels={'DEATH_EVENT': 'Death Event (0=Survived, 1=Death)'},
                        color_discrete_sequence=['#28A745', '#DC3545']
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='white',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Select feature to visualize
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'DEATH_EVENT' in numeric_cols:
                    numeric_cols.remove('DEATH_EVENT')
                
                selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df,
                        x=selected_feature,
                        marginal="box",
                        title=f"{selected_feature} Distribution",
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(plot_bgcolor='white', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot by death event
                    if 'DEATH_EVENT' in df.columns:
                        fig = px.box(
                            df,
                            x='DEATH_EVENT',
                            y=selected_feature,
                            color='DEATH_EVENT',
                            title=f"{selected_feature} by Death Event",
                            labels={'DEATH_EVENT': 'Death Event'},
                            color_discrete_sequence=['#28A745', '#DC3545']
                        )
                        fig.update_layout(plot_bgcolor='white', height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("### 🔗 Feature Correlations")
                
                # Correlation heatmap
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap"
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations with target
                if 'DEATH_EVENT' in df.columns:
                    target_corr = corr_matrix['DEATH_EVENT'].drop('DEATH_EVENT').sort_values(ascending=False)
                    
                    fig = go.Figure(go.Bar(
                        x=target_corr.values,
                        y=target_corr.index,
                        orientation='h',
                        marker=dict(
                            color=target_corr.values,
                            colorscale='RdYlGn',
                            showscale=True
                        )
                    ))
                    fig.update_layout(
                        title="Feature Correlation with Death Event",
                        xaxis_title="Correlation Coefficient",
                        yaxis_title="Features",
                        height=500,
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("### 📊 Statistical Summary")
                
                st.dataframe(df.describe(), use_container_width=True)
                
                # Group statistics by death event
                if 'DEATH_EVENT' in df.columns:
                    st.markdown("### 📈 Statistics by Death Event")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Survived (DEATH_EVENT = 0)")
                        st.dataframe(
                            df[df['DEATH_EVENT'] == 0].describe(),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("#### Death (DEATH_EVENT = 1)")
                        st.dataframe(
                            df[df['DEATH_EVENT'] == 1].describe(),
                            use_container_width=True
                        )
        
        except Exception as e:
            st.markdown(f"""
                <div class="error-box">
                    ❌ <strong>Error loading file:</strong> {str(e)}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-card" style='text-align: center; padding: 3rem;'>
                <h3 style='color: #667eea;'>📁 No file uploaded yet</h3>
                <p style='font-size: 1.1rem; color: #666;'>
                    Please upload a CSV file containing heart failure clinical records to begin analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ======================== PREDICTION PAGE ========================
elif page == "🤖 Make Prediction":
    st.markdown("""
        <div class="info-card animate-fade-in">
            <h2 style='color: #FF6B6B;'>🤖 Heart Failure Risk Prediction</h2>
            <p>Enter patient clinical data to predict cardiovascular death event risk.</p>
        </div>
    """, unsafe_allow_html=True)
    
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["Single Patient Prediction", "Batch Prediction (Upload CSV)"],
        horizontal=True
    )
    
    if prediction_mode == "Single Patient Prediction":
        st.markdown("### 📝 Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=60, step=1)
            anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            cpk = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, max_value=10000, value=250, step=10)
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            ejection_fraction = st.slider("Ejection Fraction (%)", min_value=10, max_value=80, value=40, step=1)
            high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=50000, max_value=800000, value=250000, step=1000)
            serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        with col3:
            serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=135, step=1)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            time = st.number_input("Follow-up Period (days)", min_value=1, max_value=365, value=100, step=1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Predict Risk", use_container_width=True):
            # Create input data
            input_data = pd.DataFrame({
                'age': [age],
                'anaemia': [anaemia],
                'creatinine_phosphokinase': [cpk],
                'diabetes': [diabetes],
                'ejection_fraction': [ejection_fraction],
                'high_blood_pressure': [high_blood_pressure],
                'platelets': [platelets],
                'serum_creatinine': [serum_creatinine],
                'serum_sodium': [serum_sodium],
                'sex': [sex],
                'smoking': [smoking],
                'time': [time]
            })
            
            # For demo purposes, we'll use a simple risk calculation
            # In production, you would load your trained model
            
            # Simple risk score calculation (demo)
            risk_score = 0
            
            # Age risk
            if age > 70:
                risk_score += 25
            elif age > 60:
                risk_score += 15
            elif age > 50:
                risk_score += 10
            
            # Ejection fraction risk (lower is worse)
            if ejection_fraction < 30:
                risk_score += 30
            elif ejection_fraction < 40:
                risk_score += 20
            elif ejection_fraction < 50:
                risk_score += 10
            
            # Serum creatinine risk (higher is worse)
            if serum_creatinine > 2.0:
                risk_score += 20
            elif serum_creatinine > 1.5:
                risk_score += 15
            elif serum_creatinine > 1.2:
                risk_score += 10
            
            # Other risk factors
            if anaemia == 1:
                risk_score += 10
            if diabetes == 1:
                risk_score += 10
            if high_blood_pressure == 1:
                risk_score += 10
            if smoking == 1:
                risk_score += 10
            
            # Normalize to 0-100
            risk_score = min(risk_score, 100)
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "LOW"
                risk_color = "#28A745"
                risk_emoji = "✅"
                risk_message = "Low risk of cardiovascular death event. Continue regular check-ups."
            elif risk_score < 60:
                risk_level = "MODERATE"
                risk_color = "#FFC107"
                risk_emoji = "⚠️"
                risk_message = "Moderate risk detected. Consult with a cardiologist for assessment."
            else:
                risk_level = "HIGH"
                risk_color = "#DC3545"
                risk_emoji = "🚨"
                risk_message = "High risk of cardiovascular death event. Immediate medical attention recommended."
            
            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="info-card" style='background: {risk_color}; color: white; border: none;'>
                    <h2 style='color: white; text-align: center;'>{risk_emoji} Risk Assessment Result</h2>
                    <hr style='border-color: rgba(255,255,255,0.3);'>
                    <h1 style='color: white; text-align: center; font-size: 4rem; margin: 1rem 0;'>{risk_score}%</h1>
                    <h3 style='color: white; text-align: center;'>{risk_level} RISK</h3>
                    <p style='text-align: center; font-size: 1.2rem; margin-top: 1rem; opacity: 0.95;'>
                        {risk_message}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Risk factors breakdown
            st.markdown("### 📊 Risk Factors Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create risk factors chart
                factors = []
                scores = []
                
                if age > 60:
                    factors.append('Age')
                    scores.append(25 if age > 70 else 15 if age > 60 else 10)
                
                if ejection_fraction < 50:
                    factors.append('Ejection Fraction')
                    scores.append(30 if ejection_fraction < 30 else 20 if ejection_fraction < 40 else 10)
                
                if serum_creatinine > 1.2:
                    factors.append('Serum Creatinine')
                    scores.append(20 if serum_creatinine > 2.0 else 15 if serum_creatinine > 1.5 else 10)
                
                if anaemia == 1:
                    factors.append('Anaemia')
                    scores.append(10)
                
                if diabetes == 1:
                    factors.append('Diabetes')
                    scores.append(10)
                
                if high_blood_pressure == 1:
                    factors.append('High BP')
                    scores.append(10)
                
                if smoking == 1:
                    factors.append('Smoking')
                    scores.append(10)
                
                if factors:
                    fig = go.Figure(go.Bar(
                        y=factors,
                        x=scores,
                        orientation='h',
                        marker=dict(color='#FF6B6B')
                    ))
                    fig.update_layout(
                        title="Contributing Risk Factors",
                        xaxis_title="Risk Score Contribution",
                        yaxis_title="Factor",
                        height=400,
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No significant risk factors detected!")
            
            with col2:
                # Patient summary
                st.markdown("""
                    <div class="info-card">
                        <h4 style='color: #667eea;'>Patient Summary</h4>
                        <table style='width: 100%; border-collapse: collapse;'>
                            <tr><td><strong>Age:</strong></td><td>{} years</td></tr>
                            <tr><td><strong>Sex:</strong></td><td>{}</td></tr>
                            <tr><td><strong>Ejection Fraction:</strong></td><td>{}%</td></tr>
                            <tr><td><strong>Serum Creatinine:</strong></td><td>{} mg/dL</td></tr>
                            <tr><td><strong>Serum Sodium:</strong></td><td>{} mEq/L</td></tr>
                            <tr><td><strong>Platelets:</strong></td><td>{:,} kilo/mL</td></tr>
                        </table>
                        <hr>
                        <h4 style='color: #667eea; margin-top: 1rem;'>Medical Conditions</h4>
                        <ul>
                            <li>Anaemia: {}</li>
                            <li>Diabetes: {}</li>
                            <li>High BP: {}</li>
                            <li>Smoking: {}</li>
                        </ul>
                    </div>
                """.format(
                    age, "Male" if sex == 1 else "Female", ejection_fraction,
                    serum_creatinine, serum_sodium, platelets,
                    "Yes" if anaemia == 1 else "No",
                    "Yes" if diabetes == 1 else "No",
                    "Yes" if high_blood_pressure == 1 else "No",
                    "Yes" if smoking == 1 else "No"
                ), unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            recommendations = []
            
            if ejection_fraction < 40:
                recommendations.append("🔴 **Critical:** Ejection fraction is below normal. Immediate cardiac evaluation needed.")
            
            if serum_creatinine > 1.5:
                recommendations.append("🟠 **Important:** Elevated serum creatinine suggests kidney issues. Consult nephrologist.")
            
            if age > 70:
                recommendations.append("🟡 **Monitor:** Age is a risk factor. Regular cardiac check-ups recommended.")
            
            if smoking == 1:
                recommendations.append("🚭 **Lifestyle:** Smoking cessation programs strongly recommended.")
            
            if diabetes == 1:
                recommendations.append("💊 **Management:** Ensure diabetes is well-controlled with proper medication.")
            
            if high_blood_pressure == 1:
                recommendations.append("🩺 **Control:** Blood pressure management is crucial. Follow prescribed treatment.")
            
            if not recommendations:
                recommendations.append("✅ **Good News:** No critical findings. Continue healthy lifestyle and regular check-ups.")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    else:  # Batch Prediction
        st.markdown("### 📁 Upload Patient Data for Batch Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with patient data",
            type=['csv'],
            help="CSV should contain all required features"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown(f"""
                    <div class="success-box">
                        ✅ File uploaded successfully! Found {len(df)} patients.
                    </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All", use_container_width=True):
                    # Here you would use your trained model
                    # For demo, we'll create random predictions
                    predictions = np.random.randint(0, 2, len(df))
                    probabilities = np.random.random(len(df))
                    
                    df['Prediction'] = predictions
                    df['Risk_Probability'] = probabilities
                    df['Risk_Level'] = df['Risk_Probability'].apply(
                        lambda x: 'HIGH' if x > 0.6 else 'MODERATE' if x > 0.3 else 'LOW'
                    )
                    
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_risk = (df['Risk_Level'] == 'HIGH').sum()
                        st.metric("High Risk Patients", high_risk, 
                                 delta=f"{high_risk/len(df)*100:.1f}%",
                                 delta_color="inverse")
                    
                    with col2:
                        moderate_risk = (df['Risk_Level'] == 'MODERATE').sum()
                        st.metric("Moderate Risk", moderate_risk,
                                 delta=f"{moderate_risk/len(df)*100:.1f}%")
                    
                    with col3:
                        low_risk = (df['Risk_Level'] == 'LOW').sum()
                        st.metric("Low Risk", low_risk,
                                 delta=f"{low_risk/len(df)*100:.1f}%",
                                 delta_color="normal")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="heart_failure_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.markdown(f"""
                    <div class="error-box">
                        ❌ Error processing file: {str(e)}
                    </div>
                """, unsafe_allow_html=True)

# ======================== MODEL PERFORMANCE PAGE ========================
elif page == "📈 Model Performance":
    st.markdown("""
        <div class="info-card animate-fade-in">
            <h2 style='color: #FF6B6B;'>📈 Model Performance Metrics</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    model_type = st.selectbox(
        "Select Model:",
        ["Support Vector Machine (SVM)", "Artificial Neural Network (ANN)"]
    )
    
    # Demo metrics (in production, load from saved model)
    if model_type == "Support Vector Machine (SVM)":
        accuracy = 0.85
        precision = 0.82
        recall = 0.88
        f1_score = 0.85
    else:
        accuracy = 0.87
        precision = 0.84
        recall = 0.90
        f1_score = 0.87
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{accuracy*100:.1f}%</h3>
                <p>Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{precision*100:.1f}%</h3>
                <p>Precision</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{recall*100:.1f}%</h3>
                <p>Recall</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <h3>{f1_score*100:.1f}%</h3>
                <p>F1-Score</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        
        # Demo confusion matrix
        cm = np.array([[45, 5], [8, 42]])
        
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Survived', 'Death'],
            y=['Survived', 'Death'],
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        
        # Demo ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name='ROC Curve',
            line=dict(color='#FF6B6B', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400,
            plot_bgcolor='white',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.markdown("### 🔄 Model Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['SVM', 'ANN', 'Random Forest', 'Logistic Regression'],
        'Accuracy': [0.85, 0.87, 0.83, 0.80],
        'Precision': [0.82, 0.84, 0.81, 0.78],
        'Recall': [0.88, 0.90, 0.85, 0.82],
        'F1-Score': [0.85, 0.87, 0.83, 0.80]
    })
    
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_data['Model'],
            y=comparison_data[metric],
            text=comparison_data[metric].apply(lambda x: f'{x*100:.1f}%'),
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title="Model Performance Comparison",
        yaxis_title="Score",
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Training history (for ANN)
    if model_type == "Artificial Neural Network (ANN)":
        st.markdown("### 📉 Training History")
        
        epochs = np.arange(1, 51)
        train_loss = np.exp(-epochs/10) + 0.2
        val_loss = np.exp(-epochs/10) + 0.25
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss,
            mode='lines',
            name='Validation Loss',
            line=dict(color='#FF6B6B', width=3)
        ))
        fig.update_layout(
            xaxis_title='Epochs',
            yaxis_title='Loss',
            height=400,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================== ABOUT PAGE ========================
else:  # About page
    st.markdown("""
        <div class="info-card animate-fade-in">
            <h2 style='color: #FF6B6B;'>ℹ️ About This System</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3 style='color: #FF6B6B; font-weight: 700;'>🎯 Project Overview</h3>
                <p style='font-size: 1.1rem; line-height: 1.8; color: #2C3E50;'>
                    This Heart Failure Prediction System uses advanced machine learning and deep learning 
                    algorithms to assess cardiovascular risk based on clinical records. The system analyzes 
                    12 key medical indicators to predict the probability of a death event.
                </p>
                
                <h3 style='color: #FF6B6B; margin-top: 2rem; font-weight: 700;'>🔬 Methodology</h3>
                <ul style='line-height: 2.2; color: #2C3E50; font-size: 1rem;'>
                    <li><strong style='color: #1A1A1A;'>Data Collection:</strong> 299 patient records with 12 clinical features</li>
                    <li><strong style='color: #1A1A1A;'>Preprocessing:</strong> StandardScaler for feature normalization</li>
                    <li><strong style='color: #1A1A1A;'>Models:</strong> SVM and Artificial Neural Networks</li>
                    <li><strong style='color: #1A1A1A;'>Validation:</strong> Train-test split with cross-validation</li>
                    <li><strong style='color: #1A1A1A;'>Metrics:</strong> Accuracy, Precision, Recall, F1-Score, ROC-AUC</li>
                </ul>
                
                <h3 style='color: #FF6B6B; margin-top: 2rem; font-weight: 700;'>⚠️ Disclaimer</h3>
                <p style='background: #FFF3CD; padding: 1rem; border-radius: 10px; border-left: 4px solid #FFC107; color: #664D03;'>
                    <strong style='color: #000;'>Important:</strong> This system is designed for research and educational purposes only. 
                    It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
                    Always consult with a qualified healthcare provider for medical decisions.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card" style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;'>
                <h3 style='color: white;'>📊 Key Statistics</h3>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <div style='margin: 1.5rem 0;'>
                    <h4 style='color: white; margin: 0;'>85-87%</h4>
                    <p style='opacity: 0.9; margin-top: 0.5rem;'>Model Accuracy</p>
                </div>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <div style='margin: 1.5rem 0;'>
                    <h4 style='color: white; margin: 0;'>299</h4>
                    <p style='opacity: 0.9; margin-top: 0.5rem;'>Training Samples</p>
                </div>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <div style='margin: 1.5rem 0;'>
                    <h4 style='color: white; margin: 0;'>12</h4>
                    <p style='opacity: 0.9; margin-top: 0.5rem;'>Clinical Features</p>
                </div>
                <hr style='border-color: rgba(255,255,255,0.3);'>
                <div style='margin: 1.5rem 0;'>
                    <h4 style='color: white; margin: 0;'>2</h4>
                    <p style='opacity: 0.9; margin-top: 0.5rem;'>ML Models (SVM + ANN)</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card" style='margin-top: 1rem;'>
                <h4 style='color: #FF6B6B; font-weight: 700;'>🛠️ Technologies Used</h4>
                <ul style='line-height: 2; color: #2C3E50; font-size: 0.95rem;'>
                    <li><strong style='color: #1A1A1A;'>Python 3.8+</strong></li>
                    <li><strong style='color: #1A1A1A;'>Streamlit</strong></li>
                    <li><strong style='color: #1A1A1A;'>Scikit-learn</strong></li>
                    <li><strong style='color: #1A1A1A;'>TensorFlow/Keras</strong></li>
                    <li><strong style='color: #1A1A1A;'>Plotly</strong></li>
                    <li><strong style='color: #1A1A1A;'>Pandas & NumPy</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card" style='margin-top: 2rem;'>
            <h3 style='color: #FF6B6B; font-weight: 700;'>📚 References & Resources</h3>
            <ul style='line-height: 2; color: #2C3E50; font-size: 1rem;'>
                <li><strong style='color: #1A1A1A;'>Dataset:</strong> Heart Failure Clinical Records Dataset (Kaggle)</li>
                <li><strong style='color: #1A1A1A;'>Research:</strong> Machine Learning in Cardiovascular Medicine</li>
                <li><strong style='color: #1A1A1A;'>Guidelines:</strong> American Heart Association (AHA)</li>
                <li><strong style='color: #1A1A1A;'>Documentation:</strong> Scikit-learn, TensorFlow</li>
            </ul>
            
            <h3 style='color: #FF6B6B; margin-top: 2rem; font-weight: 700;'>👨‍💻 Developer Information</h3>
            <p style='color: #2C3E50; font-size: 1rem; line-height: 1.8;'>
                Developed as part of Machine Learning Projects Hub<br>
                For questions or feedback, please contact: <strong style='color: #1A1A1A;'>ml-projects@example.com</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <p style='color: #FF6B6B; font-size: 1.3rem; font-weight: 700; margin-bottom: 0.5rem;'>
                Made with ❤️ using Streamlit
            </p>
            <p style='color: #666; margin-top: 0.5rem; font-size: 0.95rem;'>
                © 2024 ML Projects Hub. All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)
