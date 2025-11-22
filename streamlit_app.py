"""
Saudi Arabia Air Quality Monitor
Real-time pollution tracking using satellite data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import os
from typing import Dict, List, Optional
import json

# Import local modules
from satellite_fetcher import SatelliteDataFetcher
from analyzer import PollutionAnalyzer
from visualizer import PollutionVisualizer
import config

# Page configuration
st.set_page_config(
    page_title="Saudi Air Quality Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/pollution-monitor',
        'Report a bug': "https://github.com/yourusername/pollution-monitor/issues",
        'About': "Real-time air quality monitoring using Sentinel-5P satellite data"
    }
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        border-radius: 10px;
        border: 1px solid;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .violation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    h1 {
        color: #1e3a8a;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = 'Yanbu'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'pollution_data' not in st.session_state:
    st.session_state.pollution_data = {}

@st.cache_resource
def initialize_services():
    """Initialize services"""
    try:
        vertex_project = st.secrets.get("GEE_PROJECT", os.getenv("GEE_PROJECT"))
        vertex_location = st.secrets.get("VERTEX_LOCATION", os.getenv("VERTEX_LOCATION", "us-central1"))
        gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

        fetcher = SatelliteDataFetcher()
        analyzer = PollutionAnalyzer(
            gemini_api_key=gemini_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location
        )
        visualizer = PollutionVisualizer()

        return fetcher, analyzer, visualizer
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()

def create_header():
    """Header section"""
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        st.image("https://via.placeholder.com/150x50/1e3a8a/ffffff?text=AQ+Monitor", width=150)

    with col2:
        st.title("🌍 Saudi Arabia Air Quality Monitor")
        st.caption("Real-time pollution monitoring using Sentinel-5P satellite data")

    with col3:
        ksa_tz = pytz.timezone(config.TIMEZONE)
        current_time = datetime.now(ksa_tz).strftime("%H:%M KSA")
        st.metric("Time", current_time)

def create_sidebar():
    """Sidebar controls"""
    with st.sidebar:
        st.header("⚙️ Control Panel")
        selected_city = st.selectbox(
            "Select City",
            options=list(config.CITIES.keys()),
            index=list(config.CITIES.keys()).index(st.session_state.selected_city),
            help="Choose the city to monitor"
        )
        st.session_state.selected_city = selected_city

        days_back = st.slider(
            "Historical Data (days)",
            min_value=1,
            max_value=7,
            value=3,
            help="Number of days to analyze"
        )

        st.session_state.auto_refresh = st.toggle(
            "Auto-refresh (30 min)",
            value=st.session_state.auto_refresh,
            help="Automatically update data every 30 minutes"
        )

        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.session_state.pollution_data = {}
            st.rerun()

        st.divider()
        if st.session_state.last_update:
            st.info(f"Last update: {st.session_state.last_update}")

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        This system monitors air quality using:
        - **Sentinel-5P** satellite data
        - **WHO 2021** air quality guidelines
        - **Real-time** wind data
        - **AI-powered** source attribution
        """)

        st.subheader("🔬 Monitored Gases")
        for gas, info in config.GAS_PRODUCTS.items():
            st.caption(f"• **{gas}**: {info['name']}")

        return selected_city, days_back

@st.cache_data(ttl=1800)
def fetch_pollution_data(city: str, days_back: int):
    """Fetch pollution data"""
    fetcher, analyzer, _ = initialize_services()
    all_data = {}

    progress = st.progress(0)
    status = st.empty()

    gases = list(config.GAS_PRODUCTS.keys())
    for i, gas in enumerate(gases):
        status.text(f"Fetching {gas} data...")
        progress.progress((i + 1) / len(gases))

        try:
            data = fetcher.fetch_gas_data(city, gas, days_back=days_back)
            if data['success']:
                all_data[gas] = data
        except Exception as e:
            st.warning(f"Could not fetch {gas} data: {str(e)}")

    progress.empty()
    status.empty()

    return all_data

def display_metrics(pollution_data: Dict):
    """Display metrics"""
    st.subheader("📊 Current Air Quality Metrics")

    cols = st.columns(len(pollution_data))

    for i, (gas, data) in enumerate(pollution_data.items()):
        if not data.get('success'):
            continue

        with cols[i]:
            threshold_info = config.GAS_THRESHOLDS.get(gas, {})
            threshold = threshold_info.get('column_threshold', float('inf'))
            critical = threshold_info.get('critical_threshold', float('inf'))

            max_val = data['statistics']['max']
            if max_val >= critical:
                status = "🔴 Critical"
                delta_color = "inverse"
            elif max_val >= threshold:
                status = "🟡 Moderate"
                delta_color = "normal"
            else:
                status = "🟢 Normal"
                delta_color = "normal"

            st.metric(
                label=f"{gas} ({config.GAS_PRODUCTS[gas]['name']})",
                value=f"{max_val:.2f}",
                delta=f"{status} | Threshold: {threshold:.1f}",
                delta_color=delta_color
            )

            with st.expander("Details"):
                st.write(f"**Mean:** {data['statistics']['mean']:.2f}")
                st.write(f"**Min:** {data['statistics']['min']:.2f}")
                st.write(f"**Unit:** {data['unit']}")
                if max_val >= threshold:
                    exceeded = ((max_val - threshold) / threshold * 100)
                    st.error(f"Exceeded by {exceeded:.1f}%")

def display_violations(pollution_data: Dict, city: str):
    """Display violations with AI analysis"""
    st.subheader("⚠️ Violation Analysis")

    fetcher, analyzer, visualizer = initialize_services()
    violations = []

    for gas, data in pollution_data.items():
        if not data.get('success'):
            continue

        # Check for violations
        threshold_check = analyzer.check_threshold_violation(gas, data['statistics']['max'])

        if threshold_check['violated']:
            # Find hotspot
            hotspot = analyzer.find_hotspot(data)
            if hotspot:
                # Get nearby factories
                factories = analyzer.find_nearby_factories(hotspot, city)

                # Get wind data
                wind_data = data.get('wind', {})

                # Rank factories
                if wind_data.get('success'):
                    factories = analyzer.calculate_wind_vector_to_factories(
                        hotspot, factories, wind_data
                    )

                violation_info = {
                    'gas': gas,
                    'gas_name': config.GAS_PRODUCTS[gas]['name'],
                    'max_value': data['statistics']['max'],
                    'threshold': threshold_check['threshold'],
                    'unit': data['unit'],
                    'severity': threshold_check['severity'],
                    'percentage_over': threshold_check['percentage_over'],
                    'hotspot': hotspot,
                    'city': city,
                    'timestamp_ksa': data.get('timestamp_ksa', 'N/A'),
                    'wind': wind_data,
                    'nearby_factories': factories[:5]
                }

                violations.append(violation_info)

    if violations:
        for violation in violations:
            with st.container():
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.error(f"**{violation['gas']} Violation Detected**")
                    st.write(f"**Severity:** {violation['severity'].upper()}")
                    st.write(f"**Value:** {violation['max_value']:.2f} {violation['unit']}")
                    st.write(f"**Threshold:** {violation['threshold']:.1f} {violation['unit']}")
                    st.write(f"**Exceeded by:** {violation['percentage_over']:.1f}%")

                with col2:
                    # AI Analysis
                    with st.spinner("Analyzing source..."):
                        analysis = analyzer.ai_analysis(violation)
                    st.info(analysis[:300] + "..." if len(analysis) > 300 else analysis)
    else:
        st.success("✅ No violations detected - Air quality is within safe limits")

def display_map(pollution_data: Dict, city: str):
    """Display interactive pollution map"""
    st.subheader("🗺️ Pollution Heatmap")

    # Get city coordinates
    city_info = config.CITIES[city]
    center_lat, center_lon = city_info['center']

    # Create map figure
    fig = go.Figure()

    # Add factory markers
    factories = config.FACTORIES.get(city, [])
    if factories:
        factory_lats = [f['location'][0] for f in factories]
        factory_lons = [f['location'][1] for f in factories]
        factory_names = [f['name'] for f in factories]

        fig.add_trace(go.Scattermapbox(
            mode='markers',
            lon=factory_lons,
            lat=factory_lats,
            marker={'size': 10, 'color': 'red'},
            text=factory_names,
            name='Factories'
        ))

    # Add pollution data if available
    for gas, data in pollution_data.items():
        if data.get('success') and data.get('pixels'):
            pixels = data['pixels'][:100]  # Limit for performance
            if pixels:
                lats = [p['lat'] for p in pixels]
                lons = [p['lon'] for p in pixels]
                values = [p['value'] for p in pixels]

                fig.add_trace(go.Densitymapbox(
                    lat=lats,
                    lon=lons,
                    z=values,
                    radius=20,
                    name=gas,
                    colorscale='Hot',
                    showscale=True
                ))

    # Update layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)

def display_trends(pollution_data: Dict):
    """Display trend charts"""
    st.subheader("📈 Pollution Trends")

    # Create trend data
    trend_data = []
    for gas, data in pollution_data.items():
        if data.get('success'):
            trend_data.append({
                'Gas': gas,
                'Mean': data['statistics']['mean'],
                'Max': data['statistics']['max'],
                'Threshold': config.GAS_THRESHOLDS.get(gas, {}).get('column_threshold', 0)
            })

    if trend_data:
        df = pd.DataFrame(trend_data)

        # Bar chart
        fig = px.bar(df, x='Gas', y=['Max', 'Mean', 'Threshold'],
                     title="Gas Concentrations vs Thresholds",
                     barmode='group',
                     color_discrete_map={'Max': '#ef4444', 'Mean': '#3b82f6', 'Threshold': '#10b981'})

        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    # Create header
    create_header()

    # Create sidebar and get settings
    city, days_back = create_sidebar()

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        st.write("Auto-refresh enabled - Updates every 30 minutes")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Map View", "📈 Trends", "⚠️ Violations"])

    # Fetch data
    if not st.session_state.pollution_data:
        with st.spinner(f"Fetching pollution data for {city}..."):
            st.session_state.pollution_data = fetch_pollution_data(city, days_back)
            st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pollution_data = st.session_state.pollution_data

    with tab1:
        st.header(f"Air Quality Overview - {city}")
        display_metrics(pollution_data)

        # Summary statistics
        st.divider()
        col1, col2, col3 = st.columns(3)

        violations_count = sum(1 for gas, data in pollution_data.items()
                             if data.get('success') and
                             data['statistics']['max'] >= config.GAS_THRESHOLDS.get(gas, {}).get('column_threshold', float('inf')))

        with col1:
            st.metric("Total Gases Monitored", len(pollution_data))
        with col2:
            st.metric("Violations Detected", violations_count)
        with col3:
            st.metric("Data Quality", "High" if all(d.get('success') for d in pollution_data.values()) else "Partial")

    with tab2:
        st.header(f"Pollution Map - {city}")
        display_map(pollution_data, city)

    with tab3:
        st.header("Historical Trends")
        display_trends(pollution_data)

    with tab4:
        st.header("Violation Details")
        display_violations(pollution_data, city)

    # Footer
    st.divider()
    st.caption("Data source: ESA Sentinel-5P | Guidelines: WHO 2021 | Last satellite pass: Check timestamp above")

if __name__ == "__main__":
    main()