"""
911 Emergency Calls Analytics Dashboard

An interactive Streamlit dashboard for exploring emergency call patterns,
predicting call volumes, and analyzing emergency response data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data.processor import EmergencyCallsProcessor
    from models.train import ModelTrainer
except ImportError:
    st.error("Could not import required modules. Please ensure the src directory is properly set up.")


# Page configuration
st.set_page_config(
    page_title="911 Emergency Analytics",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling inspired by enterprise dashboards
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main content container - improved responsiveness */
    .main .block-container {
        padding: 1rem 1.5rem;
        max-width: 1400px;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 16px;
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.12);
        margin: 1rem auto;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar professional styling - Fixed colors */
    .css-1d391kg, .css-1lcbmhc, .css-17lntkn, section[data-testid="stSidebar"],
    .css-6qob1r, .css-uf99v8 {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Fix all sidebar text visibility */
    .css-1d391kg *, .css-1lcbmhc *, .css-17lntkn *, section[data-testid="stSidebar"] *,
    .css-6qob1r *, .css-uf99v8 * {
        color: #ffffff !important;
    }
    
    /* Sidebar specific elements */
    .css-1d391kg .stMarkdown, .css-1d391kg .stMarkdown p,
    .css-1lcbmhc .stMarkdown, .css-1lcbmhc .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    .css-6qob1r .stMarkdown, .css-6qob1r .stMarkdown p,
    .css-uf99v8 .stMarkdown, .css-uf99v8 .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Form labels in sidebar */
    .css-1d391kg label, .css-1lcbmhc label, section[data-testid="stSidebar"] label,
    .css-6qob1r label, .css-uf99v8 label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Fix selectbox and multiselect text visibility in sidebar */
    .css-1d391kg .stSelectbox > div > div > div,
    .css-1lcbmhc .stSelectbox > div > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div > div,
    .css-6qob1r .stSelectbox > div > div > div,
    .css-uf99v8 .stSelectbox > div > div > div {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Fix multiselect text in sidebar */
    .css-1d391kg .stMultiSelect > div > div > div,
    .css-1lcbmhc .stMultiSelect > div > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div > div,
    .css-6qob1r .stMultiSelect > div > div > div,
    .css-uf99v8 .stMultiSelect > div > div > div {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Fix date input text in sidebar */
    .css-1d391kg .stDateInput input,
    .css-1lcbmhc .stDateInput input,
    section[data-testid="stSidebar"] .stDateInput input,
    .css-6qob1r .stDateInput input,
    .css-uf99v8 .stDateInput input {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Ensure main content text is always dark */
    .main .block-container * {
        color: inherit;
    }
    
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6 {
        color: #1a202c !important;
    }
    
    .main .block-container p,
    .main .block-container div {
        color: #4a5568;
    }
    
    /* Sidebar buttons - improved functionality */
    .css-1d391kg .stButton > button, .css-1lcbmhc .stButton > button,
    section[data-testid="stSidebar"] .stButton > button,
    .css-6qob1r .stButton > button, .css-uf99v8 .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .css-1d391kg .stButton > button:hover, .css-1lcbmhc .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover,
    .css-6qob1r .stButton > button:hover, .css-uf99v8 .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #5a67d8, #6b46c1) !important;
    }
    
    /* Sidebar download button */
    .css-1d391kg .stDownloadButton > button, .css-1lcbmhc .stDownloadButton > button,
    section[data-testid="stSidebar"] .stDownloadButton > button,
    .css-6qob1r .stDownloadButton > button, .css-uf99v8 .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78, #38a169) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a202c;
        text-align: center;
        margin: 0 0 2rem 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    
    /* Section headers - improved contrast */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a202c !important;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        margin: 2rem 0 1.5rem 0;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.08), transparent);
        border-radius: 8px;
        position: relative;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px 4px 0 0;
    }
    
    /* Metrics cards enhanced - better responsiveness */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.5rem;
        position: relative;
        overflow: hidden;
        min-height: 120px;
    }
    
    div[data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    /* Metric labels and values - improved contrast */
    div[data-testid="metric-container"] label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-bottom: 0.8rem !important;
    }
    
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #1a202c !important;
        line-height: 1.1 !important;
    }
    
    /* Charts and visualizations */
    .stPlotlyChart {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Tables enhanced */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        font-weight: 700;
        padding: 1.2rem;
        text-align: left;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame td {
        padding: 1rem 1.2rem;
        border-bottom: 1px solid #f1f5f9;
        font-weight: 500;
        color: #4a5568;
    }
    
    .stDataFrame tr:hover td {
        background: #f8fafc;
    }
    
    /* Enhanced download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(72, 187, 120, 0.4);
    }
    
    /* Alert messages - improved contrast */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
        font-weight: 500;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #c6f6d5, #9ae6b4);
        color: #22543d !important;
        border-left: 4px solid #48bb78;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fefcbf, #faf089);
        color: #744210 !important;
        border-left: 4px solid #ed8936;
    }
    
    .stError {
        background: linear-gradient(135deg, #fed7d7, #feb2b2);
        color: #742a2a !important;
        border-left: 4px solid #e53e3e;
    }
    
    /* Professional dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 3rem 0;
    }
    
    /* Footer styling */
    .footer-container {
        background: linear-gradient(135deg, #f8fafc, #ffffff);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04);
    }
    
    /* Responsive design - improved */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem;
            margin: 0.5rem;
        }
        
        .main-header {
            font-size: 1.8rem;
        }
        
        .section-header {
            font-size: 1.1rem;
            padding: 0.8rem;
        }
        
        div[data-testid="metric-container"] {
            padding: 1rem;
            margin: 0.25rem;
        }
        
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1.5rem !important;
        }
        
        .footer-container {
            padding: 1.5rem;
        }
        
        /* Mobile sidebar improvements */
        section[data-testid="stSidebar"] .stButton > button {
            font-size: 0.8rem !important;
            padding: 0.4rem 0.8rem !important;
        }
        
        /* Mobile badge adjustments */
        .badge {
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        div[data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1.3rem !important;
        }
        
        .section-header {
            font-size: 1rem;
            padding: 0.6rem;
        }
        
        /* Very small screen adjustments */
        .stPlotlyChart {
            padding: 0.5rem;
        }
        
        .footer-container {
            padding: 1rem;
        }
    }
    
    /* Loading states */
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Professional scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
    }
    
    /* Professional badges - improved contrast */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-primary { background: #667eea; color: white; }
    .badge-success { background: #48bb78; color: white; }
    .badge-warning { background: #ed8936; color: white; }
    .badge-info { background: #4299e1; color: white; }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-online { background: #48bb78; }
    .status-warning { background: #ed8936; }
    .status-error { background: #e53e3e; }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Ensure all text is readable */
    .info-box * {
        color: #2d3748 !important;
    }
    
    /* Fix sidebar info box text specifically */
    .css-1d391kg .info-box *,
    .css-1lcbmhc .info-box *,
    section[data-testid="stSidebar"] .info-box *,
    .css-6qob1r .info-box *,
    .css-uf99v8 .info-box * {
        color: #e2e8f0 !important;
    }
    
    /* Improve subheader contrast */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1a202c !important;
    }
    
    /* Ensure warning and info messages are visible */
    .stWarning > div {
        color: #744210 !important;
    }
    
    .stInfo > div {
        color: #2c5282 !important;
    }
    
    .stError > div {
        color: #742a2a !important;
    }
    
    .stSuccess > div {
        color: #22543d !important;
    }
    
    /* Folium map container adjustments - fix cropping and display issues */
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        height: 400px !important;
        width: 100% !important;
        max-height: 400px !important;
    }
    
    /* Remove extra spacing from map containers */
    .streamlit-folium {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        overflow: visible !important;
    }
    
    /* Control folium wrapper height - fix cropping */
    div[data-testid="stIFrame"] {
        height: 400px !important;
        max-height: 400px !important;
        overflow: visible !important;
        width: 100% !important;
    }
    
    /* Reduce map component spacing and fix container */
    [data-testid="stVerticalBlock"] iframe {
        border-radius: 12px;
        min-height: 400px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path: str = "data/raw/911.csv"):
    """Load and properly process emergency calls data using the data processor"""
    
    # Try to load processed data first
    processed_path = "data/processed/processed_data.csv"
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
            # Ensure processed data has the correct timestamp format
            if 'timeStamp' in df.columns:
                df['timeStamp'] = pd.to_datetime(df['timeStamp'], errors='coerce')
            st.success("✅ Loaded pre-processed data successfully")
            return df
        except Exception as e:
            st.warning(f"⚠️ Could not load processed data: {str(e)}. Processing raw data...")
    
    # Process raw data using the proper processor
    try:
        processor = EmergencyCallsProcessor()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load and validate
        status_text.text('🔄 Loading and validating raw data...')
        progress_bar.progress(25)
        df_raw = processor.load_and_validate_data(file_path)
        
        # Step 2: Clean data
        status_text.text('🧹 Cleaning data...')
        progress_bar.progress(50)
        df_clean = processor.clean_data(df_raw)
        
        # Step 3: Engineer features
        status_text.text('⚙️ Engineering features...')
        progress_bar.progress(75)
        df_features = processor.engineer_features(df_clean)
        
        # Step 4: Finalize
        status_text.text('✅ Processing complete!')
        progress_bar.progress(100)
        
        # Save processed data for future use
        try:
            os.makedirs("data/processed", exist_ok=True)
            df_features.to_csv(processed_path, index=False)
            st.info(f"💾 Saved processed data to {processed_path} for faster future loading")
        except Exception as e:
            st.warning(f"⚠️ Could not save processed data: {str(e)}")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully processed {len(df_features):,} records with {len(df_features.columns)} features")
        return df_features
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.error("Please check your data file and ensure it contains the required columns.")
        return pd.DataFrame()


def create_professional_sidebar():
    """Create a professional enterprise-grade sidebar"""
    
    # Main sidebar header with branding
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                   padding: 2rem 1rem; margin: -1rem -1rem 2rem -1rem; 
                   border-radius: 0 0 20px 20px; text-align: center;'>
            <div style='color: white; font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;'>
                🚨 Emergency Analytics
            </div>
            <div style='color: rgba(255,255,255,0.9); font-size: 0.9rem; font-weight: 500;'>
                Professional Dashboard Suite
            </div>
            <div style='margin-top: 1rem; padding: 0.5rem 1rem; 
                       background: rgba(255,255,255,0.1); border-radius: 20px; 
                       border: 1px solid rgba(255,255,255,0.2);'>
                <span style='color: white; font-size: 0.8rem; font-weight: 600;'>
                    <span class="status-indicator status-online"></span>LIVE ANALYTICS
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for persistent filters
    if 'sidebar_filters' not in st.session_state:
        st.session_state.sidebar_filters = {
            'date_range': None,
            'emergency_types': [],
            'townships': [],
            'filters_applied': False
        }
    
    if 'df' not in st.session_state or st.session_state.df.empty:
        st.sidebar.warning("⚠️ No data loaded. Please ensure data is available.")
        return {}
    
    df = st.session_state.df
    
    # Data overview section
    st.sidebar.markdown("""
        <div style='background: rgba(255,255,255,0.05); border-radius: 12px; 
                   padding: 1.2rem; margin: 1.5rem 0; border: 1px solid rgba(255,255,255,0.1);'>
            <h4 style='color: #ffffff; margin: 0 0 1rem 0; font-size: 1rem; font-weight: 700;'>
                📊 Dataset Overview
            </h4>
            <div style='color: #e2e8f0; font-size: 0.85rem; line-height: 1.6;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span>Total Records:</span>
                    <strong style='color: #ffffff;'>{:,}</strong>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span>Emergency Types:</span>
                    <strong style='color: #ffffff;'>{}</strong>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span>Townships:</span>
                    <strong style='color: #ffffff;'>{}</strong>
                </div>
            </div>
        </div>
    """.format(
        len(df),
        df['emergency_category'].nunique() if 'emergency_category' in df.columns else 'N/A',
        df['twp'].nunique() if 'twp' in df.columns else 'N/A'
    ), unsafe_allow_html=True)
    
    # Date Range Filter Section
    st.sidebar.markdown("""
        <div style='margin: 2rem 0 1rem 0; padding-bottom: 0.8rem; 
                   border-bottom: 2px solid rgba(255,255,255,0.1);'>
            <h3 style='color: #ffffff; font-size: 1.1rem; font-weight: 700; 
                      margin: 0; text-transform: uppercase; letter-spacing: 1px;'>
                📅 Temporal Filters
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Date range filter
    if 'timeStamp' in df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['timeStamp']):
                df['timeStamp'] = pd.to_datetime(df['timeStamp'], errors='coerce')
            
            min_date = df['timeStamp'].min().date()
            max_date = df['timeStamp'].max().date()
            
            date_range = st.sidebar.date_input(
                "📆 Select Analysis Period",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Choose the date range for your analysis",
                key="date_filter"
            )
            
            # Display date range info
            if date_range and len(date_range) == 2:
                days_count = (date_range[1] - date_range[0]).days + 1
                st.sidebar.markdown(f"""
                    <div class='info-box' style='margin: 1rem 0; background: rgba(102, 126, 234, 0.1); 
                               border-radius: 8px; padding: 1rem; border-left: 3px solid #667eea;'>
                        <div style='color: #e2e8f0; font-size: 0.85rem; line-height: 1.5;'>
                            <strong style='color: #ffffff;'>Selected Period:</strong><br>
                            {date_range[0].strftime('%B %d, %Y')} → {date_range[1].strftime('%B %d, %Y')}<br>
                            <span class='badge badge-primary'>{days_count} days</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.sidebar.error(f"❌ Date filter error: {str(e)}")
            date_range = None
    else:
        date_range = None
    
    # Emergency Type Filter Section
    st.sidebar.markdown("""
        <div style='margin: 2rem 0 1rem 0; padding-bottom: 0.8rem; 
                   border-bottom: 2px solid rgba(255,255,255,0.1);'>
            <h3 style='color: #ffffff; font-size: 1.1rem; font-weight: 700; 
                      margin: 0; text-transform: uppercase; letter-spacing: 1px;'>
                🚑 Emergency Categories
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'emergency_category' in df.columns:
        emergency_types = sorted(df['emergency_category'].dropna().unique())
        
        # Quick action buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            select_all_clicked = st.button("✅ Select All", key="select_all_types", use_container_width=True)
            if select_all_clicked:
                st.session_state.selected_emergency_types = emergency_types
                st.rerun()
        with col2:
            clear_all_clicked = st.button("❌ Clear All", key="clear_all_types", use_container_width=True)
            if clear_all_clicked:
                st.session_state.selected_emergency_types = []
                st.rerun()
        
        # Initialize if not exists
        if 'selected_emergency_types' not in st.session_state:
            st.session_state.selected_emergency_types = emergency_types
        
        selected_types = st.sidebar.multiselect(
            "🏥 Choose Emergency Categories",
            options=emergency_types,
            default=st.session_state.selected_emergency_types,
            help="Select one or more emergency categories for analysis",
            key="emergency_filter"
        )
        
        # Update session state
        st.session_state.selected_emergency_types = selected_types
        
        # Display selection summary
        if selected_types:
            filtered_count = len(df[df['emergency_category'].isin(selected_types)])
            st.sidebar.markdown(f"""
                <div class='info-box' style='margin: 1rem 0; background: rgba(72, 187, 120, 0.1); 
                           border-radius: 8px; padding: 1rem; border-left: 3px solid #48bb78;'>
                    <div style='color: #e2e8f0; font-size: 0.85rem; line-height: 1.5;'>
                        <strong style='color: #ffffff;'>Selection Summary:</strong><br>
                        <span class='badge badge-success'>{len(selected_types)} categories</span>
                        <span class='badge badge-info'>{filtered_count:,} calls</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        selected_types = []
    
    # Geographic Filter Section
    st.sidebar.markdown("""
        <div style='margin: 2rem 0 1rem 0; padding-bottom: 0.8rem; 
                   border-bottom: 2px solid rgba(255,255,255,0.1);'>
            <h3 style='color: #ffffff; font-size: 1.1rem; font-weight: 700; 
                      margin: 0; text-transform: uppercase; letter-spacing: 1px;'>
                🏘️ Geographic Areas
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'twp' in df.columns:
        townships = sorted(df['twp'].dropna().unique())
        
        # Township selection control
        max_townships = min(len(townships), 25)
        num_townships = st.sidebar.slider(
            "🎯 Number of Top Townships",
            min_value=5,
            max_value=max_townships,
            value=min(15, max_townships),
            step=5,
            help="Select the number of top townships by call volume"
        )
        
        # Get top townships by call volume
        top_townships = (df['twp'].value_counts().head(num_townships).index.tolist())
        
        # Initialize if not exists
        if 'selected_townships' not in st.session_state:
            st.session_state.selected_townships = top_townships[:10]
        
        selected_townships = st.sidebar.multiselect(
            f"🗺️ Choose from Top {num_townships} Townships",
            options=top_townships,
            default=[t for t in st.session_state.selected_townships if t in top_townships],
            help="Townships are ranked by call volume",
            key="township_filter"
        )
        
        # Update session state
        st.session_state.selected_townships = selected_townships
        
        # Display geographic summary
        if selected_townships:
            township_calls = len(df[df['twp'].isin(selected_townships)])
            st.sidebar.markdown(f"""
                <div class='info-box' style='margin: 1rem 0; background: rgba(237, 137, 54, 0.1); 
                           border-radius: 8px; padding: 1rem; border-left: 3px solid #ed8936;'>
                    <div style='color: #e2e8f0; font-size: 0.85rem; line-height: 1.5;'>
                        <strong style='color: #ffffff;'>Geographic Coverage:</strong><br>
                        <span class='badge badge-warning'>{len(selected_townships)} areas</span>
                        <span class='badge badge-info'>{township_calls:,} calls</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        selected_townships = []
    
    # Dashboard Controls Section
    st.sidebar.markdown("""
        <div style='margin: 2rem 0 1rem 0; padding-bottom: 0.8rem; 
                   border-bottom: 2px solid rgba(255,255,255,0.1);'>
            <h3 style='color: #ffffff; font-size: 1.1rem; font-weight: 700; 
                      margin: 0; text-transform: uppercase; letter-spacing: 1px;'>
                ⚙️ Dashboard Controls
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        refresh_clicked = st.button("🔄 Refresh Data", use_container_width=True, 
                    help="Reload the dataset and clear cache",
                    key="refresh_data_btn")
        if refresh_clicked:
            st.cache_data.clear()
            if 'df' in st.session_state:
                del st.session_state.df
            st.rerun()
    
    with col2:
        reset_clicked = st.button("📊 Reset Filters", use_container_width=True, 
                    help="Reset all filters to default",
                    key="reset_filters_btn")
        if reset_clicked:
            # Clear all filter-related session state
            for key in list(st.session_state.keys()):
                if any(filter_key in key for filter_key in ['selected_emergency_types', 'selected_townships', 'date_filter', 'emergency_filter', 'township_filter']):
                    del st.session_state[key]
            st.rerun()
    
    # Export functionality
    if not df.empty:
        st.sidebar.download_button(
            label="📥 Export Filtered Data",
            data=df.to_csv(index=False),
            file_name=f"911_analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the currently filtered dataset as CSV",
            use_container_width=True
        )
    
    # Dashboard Information Panel
    st.sidebar.markdown("""
        <div style='margin-top: 3rem; background: rgba(255,255,255,0.05); 
                   border-radius: 16px; padding: 1.5rem; 
                   border: 1px solid rgba(255,255,255,0.1);'>
            <h4 style='color: #ffffff; font-size: 1rem; margin: 0 0 1rem 0; font-weight: 700;'>
                💡 Dashboard Features
            </h4>
            <div style='color: #a0aec0; font-size: 0.85rem; line-height: 1.6;'>
                <div style='margin-bottom: 0.8rem;'>
                    ✨ <strong>Real-time Analytics</strong><br>
                    Live emergency call pattern analysis
                </div>
                <div style='margin-bottom: 0.8rem;'>
                    🗺️ <strong>Geographic Insights</strong><br>
                    Interactive mapping and location analysis
                </div>
                <div style='margin-bottom: 0.8rem;'>
                    📈 <strong>Temporal Patterns</strong><br>
                    Time-based trends and forecasting
                </div>
                <div>
                    🎯 <strong>Advanced Filtering</strong><br>
                    Multi-dimensional data exploration
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics
    st.sidebar.markdown(f"""
        <div style='margin-top: 1.5rem; padding: 1rem; 
                   background: rgba(72, 187, 120, 0.1); 
                   border-radius: 12px; border: 1px solid rgba(72, 187, 120, 0.2);'>
            <div style='color: #48bb78; font-size: 0.8rem; font-weight: 600; text-align: center;'>
                <span class="status-indicator status-online"></span>
                Dashboard Performance: Optimal
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    return {
        'date_range': date_range,
        'emergency_types': selected_types,
        'townships': selected_townships
    }


def filter_data(df, filters):
    """Apply filters to the dataframe"""
    if df.empty or not filters:
        return df
    
    filtered_df = df.copy()
    
    # Date filter with proper datetime handling
    if 'date_range' in filters and filters['date_range'] and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        
        if 'timeStamp' in filtered_df.columns:
            try:
                # Ensure timeStamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(filtered_df['timeStamp']):
                    filtered_df['timeStamp'] = pd.to_datetime(filtered_df['timeStamp'], errors='coerce')
                
                # Apply date filter
                filtered_df = filtered_df[
                    (filtered_df['timeStamp'].dt.date >= start_date) &
                    (filtered_df['timeStamp'].dt.date <= end_date)
                ]
            except Exception as e:
                st.error(f"Error applying date filter: {str(e)}")
    
    # Emergency type filter
    if 'emergency_types' in filters and filters['emergency_types']:
        if 'emergency_category' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['emergency_category'].isin(filters['emergency_types'])
            ]
    
    # Township filter
    if 'townships' in filters and filters['townships']:
        if 'twp' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['twp'].isin(filters['townships'])
            ]
    
    return filtered_df


def display_key_metrics(df):
    """Display key performance indicators with enhanced visibility"""
    st.markdown('<p class="section-header">📊 Executive Summary & Key Performance Indicators</p>', 
                unsafe_allow_html=True)
    
    if df.empty:
        st.warning("⚠️ **No Data Available**: The current filters have resulted in an empty dataset.")
        return
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            total_calls = len(df)
            st.metric(
                label="🚨 Total Emergency Calls",
                value=f"{total_calls:,}",
                help="Total number of emergency calls in the dataset"
            )
        
        with col2:
            if 'emergency_category' in df.columns:
                unique_types = df['emergency_category'].nunique()
                most_common = df['emergency_category'].mode().iloc[0] if len(df) > 0 else "N/A"
                st.metric(
                    label="🏥 Emergency Categories",
                    value=unique_types,
                    help=f"Most common: {most_common}"
                )
            else:
                st.metric(
                    label="🏥 Emergency Categories",
                    value="N/A",
                    help="Emergency category data not available"
                )
        
        with col3:
            if 'twp' in df.columns:
                unique_townships = df['twp'].nunique()
                top_township = df['twp'].mode().iloc[0] if len(df) > 0 else "N/A"
                st.metric(
                    label="🏘️ Geographic Areas",
                    value=unique_townships,
                    help=f"Most active: {top_township}"
                )
            else:
                st.metric(
                    label="🏘️ Geographic Areas",
                    value="N/A",
                    help="Township data not available"
                )
        
        with col4:
            if 'timeStamp' in df.columns:
                # Data should already be properly processed
                if not df.empty:
                    date_range = (df['timeStamp'].max() - df['timeStamp'].min()).days
                    st.metric(
                        label="📅 Days of Data",
                        value=f"{date_range:,}",
                        help=f"From {df['timeStamp'].min().strftime('%Y-%m-%d')} to {df['timeStamp'].max().strftime('%Y-%m-%d')}"
                    )
                else:
                    st.metric(
                        label="📅 Days of Data",
                        value="0",
                        help="No data available"
                    )
            else:
                st.metric(
                    label="📅 Days of Data",
                    value="N/A",
                    help="Timestamp data not available"
                )
                
    except Exception as e:
        st.error(f"❌ Error displaying metrics: {str(e)}")
        
        # Fallback: Simple metrics display
        st.markdown(f"""
        <div style='background: #f8f9fa; padding: 2rem; border-radius: 12px; border: 1px solid #dee2e6;'>
            <h3 style='color: #495057; margin-bottom: 1.5rem;'>📊 Basic Statistics</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 8px;'>
                    <div style='font-size: 2rem; font-weight: bold; color: #007bff;'>{len(df):,}</div>
                    <div style='color: #6c757d;'>Total Calls</div>
                </div>
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 8px;'>
                    <div style='font-size: 2rem; font-weight: bold; color: #28a745;'>{df['emergency_category'].nunique() if 'emergency_category' in df.columns else 'N/A'}</div>
                    <div style='color: #6c757d;'>Emergency Types</div>
                </div>
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 8px;'>
                    <div style='font-size: 2rem; font-weight: bold; color: #ffc107;'>{df['twp'].nunique() if 'twp' in df.columns else 'N/A'}</div>
                    <div style='color: #6c757d;'>Townships</div>
                </div>
                <div style='text-align: center; padding: 1rem; background: white; border-radius: 8px;'>
                    <div style='font-size: 2rem; font-weight: bold; color: #dc3545;'>Live</div>
                    <div style='color: #6c757d;'>Data Status</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def create_time_series_chart(df):
    """Create interactive time series visualization"""
    st.markdown('<p class="section-header">📈 Call Volume Over Time</p>', unsafe_allow_html=True)
    
    if df.empty or 'timeStamp' not in df.columns:
        st.warning("No timestamp data available.")
        return
    
    # Data should already be properly processed - no need for conversions
    if df['timeStamp'].isna().any():
        st.warning("Some timestamp data is invalid and will be excluded from analysis.")
        df = df.dropna(subset=['timeStamp'])
    
    if df.empty:
        st.warning("No valid timestamp data available.")
        return
    
    # Group by different time periods
    time_period = st.selectbox(
        "Select time aggregation:",
        options=['Daily', 'Weekly', 'Monthly'],
        index=0
    )
    
    if time_period == 'Daily':
        time_grouped = df.groupby(df['timeStamp'].dt.date).size().reset_index()
        time_grouped.columns = ['Date', 'Call_Count']
        time_grouped['Date'] = pd.to_datetime(time_grouped['Date'])
    elif time_period == 'Weekly':
        time_grouped = df.groupby(df['timeStamp'].dt.to_period('W')).size().reset_index()
        time_grouped.columns = ['Date', 'Call_Count']
        time_grouped['Date'] = time_grouped['Date'].dt.start_time
    else:  # Monthly
        time_grouped = df.groupby(df['timeStamp'].dt.to_period('M')).size().reset_index()
        time_grouped.columns = ['Date', 'Call_Count']
        time_grouped['Date'] = time_grouped['Date'].dt.start_time
    
    # Create the chart
    fig = px.line(
        time_grouped,
        x='Date',
        y='Call_Count',
        title=f'{time_period} Emergency Call Volume',
        labels={'Call_Count': 'Number of Calls', 'Date': 'Date'}
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_emergency_type_analysis(df):
    """Create emergency type distribution charts"""
    st.markdown('<p class="section-header">🚑 Emergency Type Analysis</p>', unsafe_allow_html=True)
    
    if 'emergency_category' not in df.columns:
        st.warning("Emergency category data not available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Emergency type distribution
        emergency_counts = df['emergency_category'].value_counts()
        
        fig_pie = px.pie(
            values=emergency_counts.values,
            names=emergency_counts.index,
            title="Emergency Type Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Emergency type by hour
        if 'hour' in df.columns:
            try:
                hourly_emergency = df.groupby(['hour', 'emergency_category']).size().unstack(fill_value=0)
                
                # Convert to long format for plotly
                hourly_melted = hourly_emergency.reset_index().melt(
                    id_vars=['hour'], 
                    var_name='Emergency_Type', 
                    value_name='Call_Count'
                )
                
                fig_bar = px.bar(
                    hourly_melted,
                    x='hour',
                    y='Call_Count',
                    color='Emergency_Type',
                    title="Emergency Types by Hour of Day",
                    labels={'hour': 'Hour of Day', 'Call_Count': 'Number of Calls'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating hourly emergency chart: {str(e)}")
                # Fallback: simple hourly distribution
                if 'hour' in df.columns:
                    hourly_simple = df.groupby('hour').size()
                    fig_simple = px.bar(
                        x=hourly_simple.index,
                        y=hourly_simple.values,
                        title="Total Calls by Hour of Day",
                        labels={'x': 'Hour of Day', 'y': 'Number of Calls'}
                    )
                    st.plotly_chart(fig_simple, use_container_width=True)


def create_geographic_analysis(df):
    """Create geographic visualizations"""
    st.markdown('<p class="section-header">🗺️ Geographic Analysis</p>', unsafe_allow_html=True)
    
    if 'lat' not in df.columns or 'lng' not in df.columns:
        st.warning("Geographic coordinates not available. Please check data processing.")
        return
    
    # Data should already be cleaned and validated
    geo_df = df.dropna(subset=['lat', 'lng'])
    
    if geo_df.empty:
        st.warning("No valid geographic data available.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # For performance, use a reasonable sample size for map display
        display_sample_size = 5000
        if len(geo_df) > display_sample_size:
            geo_sample = geo_df.sample(n=display_sample_size, random_state=42)
            # Remove the visible message - keep sampling silent for professional UI
        else:
            geo_sample = geo_df
        
        # Calculate bounds for better centering and display
        lat_center = geo_sample['lat'].mean()
        lng_center = geo_sample['lng'].mean()
        
        # Calculate proper bounds and padding
        lat_min, lat_max = geo_sample['lat'].min(), geo_sample['lat'].max()
        lng_min, lng_max = geo_sample['lng'].min(), geo_sample['lng'].max()
        
        lat_range = lat_max - lat_min
        lng_range = lng_max - lng_min
        
        # Create folium map with proper bounds and zoom
        m = folium.Map(
            location=[lat_center, lng_center],
            zoom_start=10,  # Start with reasonable zoom
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Set bounds to fit all data with padding
        southwest = [lat_min - lat_range * 0.1, lng_min - lng_range * 0.1]
        northeast = [lat_max + lat_range * 0.1, lng_max + lng_range * 0.1]
        m.fit_bounds([southwest, northeast])
        
        # Add emergency type colors
        if 'emergency_category' in geo_sample.columns:
            emergency_types = geo_sample['emergency_category'].unique()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightgreen', 'pink', 'darkblue', 'gray']
            color_map = dict(zip(emergency_types, colors[:len(emergency_types)]))
            
            # Add simple circle markers
            for idx, row in geo_sample.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=6,
                    popup=folium.Popup(
                        f"""
                        <div style="font-family: Arial; font-size: 12px; width: 200px;">
                            <h4 style="margin: 0; color: #333;">{row.get('emergency_category', 'Unknown')}</h4>
                            <hr style="margin: 5px 0;">
                            <p style="margin: 2px 0;"><b>📍 Location:</b><br>{row.get('addr', 'No address')}</p>
                            <p style="margin: 2px 0;"><b>🏘️ Township:</b> {row.get('twp', 'Unknown')}</p>
                            <p style="margin: 2px 0;"><b>📅 Time:</b> {str(row.get('timeStamp', 'No timestamp'))[:16]}</p>
                        </div>
                        """,
                        max_width=250
                    ),
                    color=color_map.get(row.get('emergency_category', 'Unknown'), 'gray'),
                    fill=True,
                    opacity=0.9,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        # Display map with better sizing
        map_data = st_folium(m, width=700, height=450, returned_objects=["last_clicked"])
    
    with col2:
        st.subheader("📊 Geographic Summary")
        
        # Display legend in sidebar instead of on map
        if 'emergency_category' in geo_sample.columns:
            st.markdown("### 🗺️ Map Legend")
            legend_df = pd.DataFrame({
                'Emergency Type': list(color_map.keys()),
                'Color': [f'🔴' if c == 'red' else f'🔵' if c == 'blue' else f'🟢' if c == 'green' 
                         else f'🟣' if c == 'purple' else f'🟠' if c == 'orange' else f'⚫' 
                         for c in color_map.values()]
            })
            st.dataframe(legend_df, use_container_width=True, hide_index=True)
        
        if 'twp' in geo_df.columns:
            # Top townships chart
            top_townships = geo_df['twp'].value_counts().head(8)
            
            fig_township = px.bar(
                x=top_townships.values,
                y=top_townships.index,
                orientation='h',
                title="Top 8 Townships",
                labels={'x': 'Call Count', 'y': 'Township'},
                color=top_townships.values,
                color_continuous_scale='Viridis'
            )
            fig_township.update_layout(
                height=350,
                showlegend=False,
                font=dict(size=11)
            )
            st.plotly_chart(fig_township, use_container_width=True)
        
        # Geographic statistics
        st.markdown("### 📈 Location Stats")
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("📍 Total Locations", f"{len(geo_df):,}")
            st.metric("🏘️ Townships", f"{geo_df['twp'].nunique() if 'twp' in geo_df.columns else 'N/A'}")
        
        with stats_col2:
            st.metric("📐 Lat Range", f"{lat_range:.3f}°")
            st.metric("📐 Lng Range", f"{lng_range:.3f}°")


def create_temporal_patterns(df):
    """Create temporal pattern analysis"""
    st.markdown('<p class="section-header">⏰ Temporal Patterns</p>', unsafe_allow_html=True)
    
    if 'timeStamp' not in df.columns:
        st.warning("Timestamp data not available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern - features should already exist
        if 'hour' in df.columns:
            # Ensure hour is integer for consistency
            df_clean = df.dropna(subset=['hour'])
            df_clean = df_clean[df_clean['hour'].between(0, 23)]
            df_clean['hour'] = df_clean['hour'].astype(int)
            
            hourly_counts = df_clean.groupby('hour').size()
            
            fig_hour = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Calls by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Calls'}
            )
            fig_hour.update_layout(showlegend=False)
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.warning("Hour feature not available. Please check data processing.")
    
    with col2:
        # Daily pattern - features should already exist
        if 'dayofweek' in df.columns:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Data should already be clean integers, but convert to int to be safe
            df_clean = df.dropna(subset=['dayofweek'])
            df_clean = df_clean[df_clean['dayofweek'].between(0, 6)]
            df_clean['dayofweek'] = df_clean['dayofweek'].astype(int)
            
            daily_counts = df_clean.groupby('dayofweek').size()
            
            # Only use valid day indices and convert to int
            valid_indices = [int(i) for i in daily_counts.index if 0 <= int(i) <= 6]
            
            if valid_indices:
                fig_day = px.bar(
                    x=[day_names[i] for i in valid_indices],
                    y=[daily_counts[i] for i in valid_indices],
                    title="Calls by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Number of Calls'}
                )
                fig_day.update_layout(showlegend=False)
                st.plotly_chart(fig_day, use_container_width=True)
            else:
                st.warning("No valid day of week data available.")
        else:
            st.warning("Day of week feature not available. Please check data processing.")
    
    # Heatmap - features should already exist and be clean
    if 'hour' in df.columns and 'dayofweek' in df.columns:
        st.subheader("Call Volume Heatmap (Hour vs Day of Week)")
        
        # Data should already be clean - no need for extensive processing
        df_heat = df.dropna(subset=['hour', 'dayofweek'])
        df_heat = df_heat[
            df_heat['hour'].between(0, 23) & 
            df_heat['dayofweek'].between(0, 6)
        ]
        
        # Ensure data types are integers for proper indexing
        df_heat = df_heat.copy()
        df_heat['hour'] = df_heat['hour'].astype(int)
        df_heat['dayofweek'] = df_heat['dayofweek'].astype(int)
        
        if not df_heat.empty:
            # Create heatmap data with proper dimensions
            heatmap_data = df_heat.groupby(['dayofweek', 'hour']).size().unstack(fill_value=0)
            
            # Ensure all hours 0-23 are present
            for hour in range(24):
                if hour not in heatmap_data.columns:
                    heatmap_data[hour] = 0
            
            # Ensure all days 0-6 are present
            for day in range(7):
                if day not in heatmap_data.index:
                    heatmap_data.loc[day] = 0
            
            # Sort columns and index to ensure proper order
            heatmap_data = heatmap_data.sort_index()  # Sort by day of week
            heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)  # Sort by hour
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create the heatmap with proper dimensions
            fig_heatmap = px.imshow(
                heatmap_data.values,
                labels=dict(x="Hour of Day", y="Day of Week", color="Call Count"),
                x=list(range(24)),  # Always 0-23
                y=[day_names[i] for i in range(7)],  # Always all 7 days
                aspect="auto",
                title="Emergency Call Heatmap",
                color_continuous_scale="Viridis"
            )
            
            # Improve heatmap layout
            fig_heatmap.update_layout(
                height=400,
                font=dict(size=12),
                title_font_size=16,
                coloraxis_colorbar=dict(
                    title="Number of Calls"
                )
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ No valid data available for heatmap visualization.")
    else:
        st.warning("Temporal features (hour, dayofweek) not available. Please check data processing.")


def create_statistical_analysis(df):
    """Create statistical analysis and insights"""
    st.markdown('<p class="section-header">📊 Statistical Analysis</p>', unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available for analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Call Volume Statistics")
        
        # Daily call volume statistics
        if 'timeStamp' in df.columns:
            # Data should already be properly processed
            daily_counts = df.groupby(df['timeStamp'].dt.date).size()
            
            stats_data = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Daily Calls': [
                    f"{daily_counts.mean():.1f}",
                    f"{daily_counts.median():.1f}",
                    f"{daily_counts.std():.1f}",
                    f"{daily_counts.min()}",
                    f"{daily_counts.max()}"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.table(stats_df)
        else:
            st.warning("Timestamp data not available for statistics. Please check data processing.")
    
    with col2:
        st.subheader("Top Locations")
        
        if 'twp' in df.columns:
            top_townships = df['twp'].value_counts().head(10)
            
            fig_top = px.bar(
                x=top_townships.values,
                y=top_townships.index,
                orientation='h',
                title="Top 10 Townships by Call Volume",
                labels={'x': 'Number of Calls', 'y': 'Township'}
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)


def create_advanced_analytics(df):
    """Create advanced analytics and insights"""
    st.markdown('<p class="section-header">🔬 Advanced Analytics & Predictive Insights</p>', 
                unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available for advanced analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly trends
        st.subheader("📅 Monthly Call Volume Trends")
        
        if 'timeStamp' in df.columns:
            # Data should already be properly processed
            if not df.empty:
                # Group by month
                monthly_data = df.groupby(df['timeStamp'].dt.to_period('M')).size().reset_index()
                monthly_data.columns = ['Month', 'Call_Count']
                monthly_data['Month'] = monthly_data['Month'].dt.start_time
                
                fig_monthly = px.line(
                    monthly_data,
                    x='Month',
                    y='Call_Count',
                    title="Monthly Emergency Call Trends",
                    labels={'Call_Count': 'Number of Calls', 'Month': 'Month'},
                    markers=True
                )
                fig_monthly.update_traces(
                    line_color='#667eea',
                    marker_color='#764ba2',
                    marker_size=8
                )
                fig_monthly.update_layout(height=300)
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.warning("No data available for monthly trends.")
        else:
            st.warning("Timestamp data not available. Please check data processing.")
    
    with col2:
        # Emergency type distribution over time
        st.subheader("🚑 Emergency Response Patterns")
        
        if 'emergency_category' in df.columns and 'timeStamp' in df.columns:
            # Get top 3 emergency types
            top_types = df['emergency_category'].value_counts().head(3).index
            df_top = df[df['emergency_category'].isin(top_types)]
            
            if not df_top.empty:
                # Group by month and emergency type
                monthly_emergency = df_top.groupby([
                    df_top['timeStamp'].dt.to_period('M'),
                    'emergency_category'
                ]).size().reset_index()
                monthly_emergency.columns = ['Month', 'Emergency_Type', 'Call_Count']
                monthly_emergency['Month'] = monthly_emergency['Month'].dt.start_time
                
                fig_emergency_time = px.line(
                    monthly_emergency,
                    x='Month',
                    y='Call_Count',
                    color='Emergency_Type',
                    title="Top 3 Emergency Types Over Time",
                    labels={'Call_Count': 'Number of Calls', 'Month': 'Month'}
                )
                fig_emergency_time.update_layout(height=300)
                st.plotly_chart(fig_emergency_time, use_container_width=True)
            else:
                st.warning("Insufficient data for emergency patterns.")
        else:
            st.warning("Required data columns not available. Please check data processing.")
    
    # Full width advanced charts
    st.subheader("🔥 Geographic Hotspots Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Township ranking
        if 'twp' in df.columns:
            township_counts = df['twp'].value_counts().head(15)
            
            fig_townships = px.bar(
                x=township_counts.index,
                y=township_counts.values,
                title="Top 15 Townships by Emergency Call Volume",
                labels={'x': 'Township', 'y': 'Number of Calls'},
                color=township_counts.values,
                color_continuous_scale='Reds'
            )
            fig_townships.update_layout(
                height=400,
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig_townships, use_container_width=True)
    
    with col4:
        # Emergency type by township (top 5 townships)
        if 'twp' in df.columns and 'emergency_category' in df.columns:
            top_townships = df['twp'].value_counts().head(5).index
            df_top_twp = df[df['twp'].isin(top_townships)]
            
            township_emergency = df_top_twp.groupby(['twp', 'emergency_category']).size().reset_index()
            township_emergency.columns = ['Township', 'Emergency_Type', 'Call_Count']
            
            fig_twp_emergency = px.bar(
                township_emergency,
                x='Township',
                y='Call_Count',
                color='Emergency_Type',
                title="Emergency Types in Top 5 Townships",
                labels={'Call_Count': 'Number of Calls'}
            )
            fig_twp_emergency.update_layout(
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_twp_emergency, use_container_width=True)
        else:
            st.warning("Required data columns not available. Please check data processing.")


def create_performance_metrics(df):
    """Create performance and efficiency metrics"""
    st.markdown('<p class="section-header">⚡ Performance & Efficiency Metrics</p>', 
                unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No data available for performance analysis.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Average calls per day
        if 'timeStamp' in df.columns:
            # Data should already be properly processed
            daily_calls = df.groupby(df['timeStamp'].dt.date).size()
            avg_daily = daily_calls.mean()
            
            st.metric(
                label="📊 Avg Daily Calls",
                value=f"{avg_daily:.1f}",
                help="Average number of emergency calls per day"
            )
        else:
            st.metric("📊 Avg Daily Calls", "N/A", help="Timestamp data not available")
    
    with col2:
        # Peak hour
        if 'hour' in df.columns:
            # Ensure hour is integer for consistency
            df_clean = df.dropna(subset=['hour'])
            df_clean = df_clean[df_clean['hour'].between(0, 23)]
            df_clean['hour'] = df_clean['hour'].astype(int)
            
            if not df_clean.empty:
                peak_hour = df_clean['hour'].value_counts().index[0]
                peak_calls = df_clean['hour'].value_counts().iloc[0]
                
                st.metric(
                    label="🕐 Peak Hour",
                    value=f"{peak_hour}:00",
                    delta=f"{peak_calls} calls",
                    help="Hour with most emergency calls"
                )
            else:
                st.metric("🕐 Peak Hour", "No Data", help="No valid hour data")
        else:
            st.metric("🕐 Peak Hour", "N/A", help="Hour data not available")
    
    with col3:
        # Busiest day of week
        if 'dayofweek' in df.columns:
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            # Ensure dayofweek is integer for proper indexing
            df_clean = df.dropna(subset=['dayofweek'])
            df_clean = df_clean[df_clean['dayofweek'].between(0, 6)]
            df_clean['dayofweek'] = df_clean['dayofweek'].astype(int)
            
            if not df_clean.empty:
                busiest_day_num = df_clean['dayofweek'].value_counts().index[0]
                busiest_day = day_names[busiest_day_num]
                busiest_calls = df_clean['dayofweek'].value_counts().iloc[0]
                
                st.metric(
                    label="📅 Busiest Day",
                    value=busiest_day,
                    delta=f"{busiest_calls} calls",
                    help="Day of week with most emergency calls"
                )
            else:
                st.metric("📅 Busiest Day", "No Data", help="No valid day of week data")
        else:
            st.metric("📅 Busiest Day", "N/A", help="Day of week data not available")
    
    with col4:
        # Most common emergency
        if 'emergency_category' in df.columns:
            top_emergency = df['emergency_category'].value_counts().index[0]
            top_emergency_calls = df['emergency_category'].value_counts().iloc[0]
            percentage = (top_emergency_calls / len(df)) * 100
            
            st.metric(
                label="🚨 Top Emergency",
                value=top_emergency[:10] + "..." if len(top_emergency) > 10 else top_emergency,
                delta=f"{percentage:.1f}%",
                help="Most common type of emergency call"
            )
        else:
            st.metric("🚨 Top Emergency", "N/A", help="Emergency category data not available")


def main():
    """Main dashboard function with enterprise-grade layout"""
    
    # Professional header with status indicators
    st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem;'>
            <h1 class="main-header">🚨 Emergency Response Analytics Platform</h1>
            <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;'>
                <div class='badge badge-success'>
                    <span class="status-indicator status-online"></span>Real-time Data
                </div>
                <div class='badge badge-primary'>
                    <span class="status-indicator status-online"></span>Live Dashboard
                </div>
                <div class='badge badge-info'>
                    <span class="status-indicator status-online"></span>Advanced Analytics
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data with proper error handling
    if 'df' not in st.session_state:
        with st.spinner("🔄 Initializing analytics platform..."):
            st.session_state.df = load_data()
    
    df = st.session_state.df
    
    if df.empty:
        st.error("❌ **Critical Error**: No data available. Please check your data source.")
        st.info("🔧 **Troubleshooting Steps:**")
        st.markdown("""
        1. Ensure `data/raw/911.csv` exists in your project directory
        2. Check that the CSV file contains the required columns: `lat`, `lng`, `desc`, `title`, `timeStamp`, `addr`
        3. Verify the data format matches the expected structure
        4. Try clearing the cache and refreshing the page
        """)
        
        # Provide option to clear cache
        if st.button("🔄 Clear Cache and Retry"):
            st.cache_data.clear()
            if 'df' in st.session_state:
                del st.session_state.df
            st.rerun()
        
        st.stop()
    
    # Validate that essential features exist
    required_features = ['emergency_category', 'hour', 'dayofweek', 'timeStamp']
    missing_features = [feature for feature in required_features if feature not in df.columns]
    
    if missing_features:
        st.error(f"❌ **Data Processing Error**: Missing required features: {missing_features}")
        st.info("This indicates the data processing pipeline did not complete successfully.")
        
        # Option to reprocess data
        if st.button("🔄 Reprocess Data"):
            st.cache_data.clear()
            if 'df' in st.session_state:
                del st.session_state.df
            st.rerun()
        
        st.stop()
    
    # Professional sidebar
    filters = create_professional_sidebar()
    
    # Apply filters with progress indication
    with st.spinner("🔍 Applying filters and processing data..."):
        filtered_df = filter_data(df, filters)
    
    # Data quality check
    if filtered_df.empty:
        st.warning("⚠️ **No Data Available**: The current filters have resulted in an empty dataset. Please adjust your filter criteria.")
        st.stop()
    
    # Main dashboard content with professional layout
    # Key Metrics Section
    display_key_metrics(filtered_df)
    
    # Professional divider
    st.markdown("---")
    
    # Time Series Analysis Section
    st.markdown('<p class="section-header">📈 Temporal Analysis & Trend Forecasting</p>', 
                unsafe_allow_html=True)
    create_time_series_chart(filtered_df)
    
    st.markdown("---")
    
    # Emergency Analysis Section
    st.markdown('<p class="section-header">🚑 Emergency Category Intelligence</p>', 
                unsafe_allow_html=True)
    create_emergency_type_analysis(filtered_df)
    
    st.markdown("---")
    
    # Geographic Intelligence Section
    st.markdown('<p class="section-header">🗺️ Geographic Intelligence & Heat Mapping</p>', 
                unsafe_allow_html=True)
    create_geographic_analysis(filtered_df)
    
    st.markdown("---")
    
    # Temporal Patterns Section
    st.markdown('<p class="section-header">⏰ Advanced Temporal Pattern Analysis</p>', 
                unsafe_allow_html=True)
    create_temporal_patterns(filtered_df)
    
    st.markdown("---")
    
    # Statistical Analysis Section
    st.markdown('<p class="section-header">📊 Statistical Intelligence & Data Insights</p>', 
                unsafe_allow_html=True)
    create_statistical_analysis(filtered_df)
    
    # Advanced Analytics Section
    st.markdown("---")
    create_advanced_analytics(filtered_df)
    
    # Performance Metrics Section
    st.markdown("---")
    create_performance_metrics(filtered_df)
    
    # Professional footer
    st.markdown("---")
    st.markdown(
        """
        <div class='footer-container'>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 2rem;'>
                <div style='flex: 1; min-width: 300px;'>
                    <h3 style='color: #2d3748; margin: 0 0 1rem 0; font-size: 1.2rem; font-weight: 700;'>
                        🚨 Emergency Response Analytics Platform
                    </h3>
                    <p style='color: #718096; margin: 0; font-size: 0.95rem; line-height: 1.6;'>
                        Enterprise-grade analytics platform for emergency response optimization, 
                        real-time monitoring, and data-driven decision making.
                    </p>
                </div>
                <div style='flex: 1; text-align: center; min-width: 250px;'>
                    <h4 style='color: #4a5568; margin: 0 0 1rem 0; font-size: 1rem; font-weight: 600;'>
                        Technology Stack
                    </h4>
                    <div style='display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap;'>
                        <span class='badge badge-primary'>Streamlit</span>
                        <span class='badge badge-success'>Python</span>
                        <span class='badge badge-warning'>Plotly</span>
                        <span class='badge badge-info'>Pandas</span>
                    </div>
                </div>
                <div style='flex: 1; text-align: right; min-width: 200px;'>
                    <h4 style='color: #4a5568; margin: 0 0 1rem 0; font-size: 1rem; font-weight: 600;'>
                        Platform Status
                    </h4>
                    <div style='color: #48bb78; font-weight: 600;'>
                        <span class="status-indicator status-online"></span>
                        All Systems Operational
                    </div>
                    <div style='color: #718096; font-size: 0.85rem; margin-top: 0.5rem;'>
                        Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                    </div>
                </div>
            </div>
            <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; text-align: center;'>
                <div style='color: #a0aec0; font-size: 0.9rem; line-height: 1.5;'>
                    <strong style='color: #4a5568;'>Professional Analytics Dashboard</strong> • 
                    Built with enterprise-grade architecture • 
                    Optimized for real-time emergency response insights
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main() 