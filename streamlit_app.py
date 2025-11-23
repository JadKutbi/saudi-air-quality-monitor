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
from visualizer import MapVisualizer
from data_validator import DataValidator
from dashboard_components import (
    create_aqi_dashboard,
    create_health_risk_panel,
    create_data_quality_panel,
    create_insights_panel,
    create_historical_comparison
)
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
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 6  # Default 6 hours
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'pollution_data' not in st.session_state:
    st.session_state.pollution_data = {}
if 'alert_thresholds' not in st.session_state:
    st.session_state.alert_thresholds = {}

@st.cache_resource
def initialize_services():
    """Initialize services with graceful error handling"""
    services = {}

    vertex_project = st.secrets.get("VERTEX_PROJECT_ID", os.getenv("VERTEX_PROJECT_ID"))
    vertex_location = st.secrets.get("VERTEX_LOCATION", os.getenv("VERTEX_LOCATION", "us-central1"))
    gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

    # Initialize each service with error handling
    try:
        fetcher = SatelliteDataFetcher()
        services['fetcher'] = fetcher
    except Exception as e:
        st.warning(f"Satellite data fetcher initialization issue: {str(e)}")
        services['fetcher'] = None

    try:
        analyzer = PollutionAnalyzer(
            gemini_api_key=gemini_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location
        )
        services['analyzer'] = analyzer
    except Exception as e:
        st.warning(f"AI analyzer initialization issue: {str(e)}")
        services['analyzer'] = None

    try:
        visualizer = MapVisualizer()
        services['visualizer'] = visualizer
    except Exception as e:
        st.warning(f"Map visualizer initialization issue: {str(e)}")
        services['visualizer'] = None

    try:
        validator = DataValidator()
        services['validator'] = validator
    except Exception as e:
        st.warning(f"Data validator initialization issue: {str(e)}")
        services['validator'] = None

    # Return services (some may be None)
    return (services.get('fetcher'), services.get('analyzer'),
            services.get('visualizer'), services.get('validator'))

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
            "Search Period (days)",
            min_value=1,
            max_value=14,
            value=3,
            help="How far back to search for satellite data. System uses median of all observations from the most recent day (ensures wind sync while filling cloud gaps). Values will be identical across all slider positions."
        )

        # Auto-refresh settings
        st.subheader("🔄 Refresh Settings")

        refresh_enabled = st.toggle(
            "Enable Auto-refresh",
            value=st.session_state.get('auto_refresh', False),
            help="Automatically update data at specified interval"
        )
        st.session_state.auto_refresh = refresh_enabled

        if refresh_enabled:
            refresh_hours = st.select_slider(
                "Refresh Interval",
                options=[0.5, 1, 2, 3, 4, 6, 8, 12, 24],
                value=st.session_state.get('refresh_interval', 6),
                format_func=lambda x: f"{x} hours" if x >= 1 else f"{int(x*60)} minutes",
                help="How often to refresh the data"
            )
            st.session_state.refresh_interval = refresh_hours

            # Show next refresh time
            if st.session_state.last_update:
                from datetime import datetime, timedelta
                import pytz
                last_update_dt = datetime.strptime(st.session_state.last_update, "%Y-%m-%d %H:%M:%S")
                ksa_tz = pytz.timezone(config.TIMEZONE)
                last_update_ksa = ksa_tz.localize(last_update_dt)
                next_refresh = last_update_ksa + timedelta(hours=refresh_hours)
                st.caption(f"Next refresh: {next_refresh.strftime('%H:%M:%S KSA')}")

        if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
            st.session_state.pollution_data = {}
            st.rerun()

        st.divider()
        if st.session_state.last_update:
            st.info(f"Last update: {st.session_state.last_update}")

        # Connection diagnostics
        st.divider()
        with st.expander("🔧 Connection Diagnostics"):
            if st.button("Test Earth Engine Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    try:
                        import ee
                        # Test basic connection
                        test_number = ee.Number(1).getInfo()
                        st.success("✅ Earth Engine connection successful!")

                        # Test project access
                        try:
                            test_collection = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_NO2").limit(1).getInfo()
                            st.success("✅ Can access Sentinel-5P data!")
                        except Exception as e:
                            st.error(f"❌ Cannot access Sentinel-5P: {str(e)}")

                        # Check credentials
                        if st.secrets.get("GEE_SERVICE_ACCOUNT"):
                            st.info(f"Using service account: {st.secrets['GEE_SERVICE_ACCOUNT'][:30]}...")
                        else:
                            st.warning("No service account configured - using default auth")

                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        st.info("Please check:")
                        st.write("1. GEE_SERVICE_ACCOUNT in secrets")
                        st.write("2. GEE_PRIVATE_KEY in secrets")
                        st.write("3. Service account has Earth Engine access")
                        st.write("4. Project ID is correct")

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
    """Fetch pollution data with improved error handling"""
    fetcher, analyzer, _, _ = initialize_services()

    if not fetcher:
        st.error("❌ Cannot connect to Earth Engine satellite data")
        st.info("Please check:")
        st.write("• Google Earth Engine authentication")
        st.write("• Service account credentials in Streamlit secrets")
        st.write("• Use the Connection Diagnostics tool in the sidebar")
        return {}

    all_data = {}
    errors = []

    progress = st.progress(0)
    status = st.empty()

    gases = list(config.GAS_PRODUCTS.keys())
    for i, gas in enumerate(gases):
        status.text(f"🔍 Searching for {gas} data (past {days_back} days)...")
        progress.progress((i + 1) / len(gases))

        try:
            data = fetcher.fetch_gas_data(city, gas, days_back=days_back)
            if data and data.get('success'):
                all_data[gas] = data
                pixel_count = data.get('statistics', {}).get('pixel_count', 0)
                status.text(f"✅ {gas} data fetched ({pixel_count} pixels)")
            else:
                error_msg = data.get('error', 'No data available') if data else 'No data available'
                errors.append(f"{gas}: {error_msg}")
                status.text(f"⚠️ {gas}: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            errors.append(f"{gas}: {error_msg}")
            # Log but continue with other gases
            print(f"Error fetching {gas}: {error_msg}")

    progress.empty()
    status.empty()

    # Show summary of errors if any
    if errors and len(errors) == len(gases):
        # All gases failed - likely a systemic issue
        st.error("Failed to fetch data for all gases")
        with st.expander("Error Details"):
            for error in errors:
                st.write(f"• {error}")
    elif errors:
        # Some gases failed
        with st.expander(f"⚠️ Partial data issues ({len(errors)} gases)"):
            for error in errors:
                st.write(f"• {error}")

    return all_data

def display_metrics(pollution_data: Dict):
    """Display metrics"""
    st.subheader("📊 Current Air Quality Metrics")

    # Filter for successful data
    valid_data = {gas: data for gas, data in pollution_data.items() if data.get('success')}

    if not valid_data:
        st.warning("No pollution data available. Please try again later.")
        return

    # Create columns based on valid data count
    cols = st.columns(min(len(valid_data), 6))  # Max 6 columns for better display

    for i, (gas, data) in enumerate(valid_data.items()):
        with cols[i % len(cols)]:
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

    fetcher, analyzer, visualizer, validator = initialize_services()
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

                    # Add wind information
                    if violation['wind'].get('success'):
                        st.write(f"**Wind:** {violation['wind']['speed_ms']:.1f} m/s from {violation['wind']['direction_cardinal']} ({violation['wind']['direction_deg']:.0f}°)")
                        st.write(f"**Wind Confidence:** {violation['wind']['confidence']:.0f}%")

                    # Display hotspot location
                    if violation.get('hotspot'):
                        st.write(f"**Hotspot Location:** ({violation['hotspot']['lat']:.4f}, {violation['hotspot']['lon']:.4f})")

                with col2:
                    # AI Analysis
                    st.write("**🤖 AI Source Analysis:**")
                    with st.spinner("Analyzing pollution source..."):
                        analysis = analyzer.ai_analysis(violation)

                    # Display full analysis in an expandable section
                    if len(analysis) > 300:
                        st.info(analysis[:300] + "...")
                        with st.expander("View Full Analysis"):
                            st.write(analysis)
                    else:
                        st.info(analysis)

                # Add factory list if available
                if violation.get('nearby_factories'):
                    with st.expander(f"📍 Nearby Industrial Facilities ({len(violation['nearby_factories'])} found)"):
                        for factory in violation['nearby_factories'][:5]:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{factory['name']}**")
                                st.write(f"Type: {factory['type']}")
                            with col2:
                                st.write(f"Distance: {factory['distance_km']:.1f} km")
                                if factory.get('likely_upwind'):
                                    st.write("⚠️ **UPWIND**")
                            with col3:
                                st.write(f"Confidence: {factory.get('confidence', 0):.0f}%")
                                st.write(f"Emissions: {', '.join(factory['emissions'][:2])}")

                st.divider()
    else:
        st.success("✅ No violations detected - Air quality is within safe limits")

def display_map(pollution_data: Dict, city: str):
    """Display interactive pollution map"""
    st.subheader("🗺️ Pollution Heatmap")

    # Import streamlit_folium for displaying folium maps
    from streamlit_folium import st_folium

    # Initialize visualizer
    fetcher, analyzer, visualizer, validator = initialize_services()

    # Get ALL gases that have successful data (with or without pixels)
    available_gases = [gas for gas, data in pollution_data.items()
                      if data.get('success')]

    if not available_gases:
        st.warning("No pollution data available to display on the map")
        return

    # Also check for gases with pixel data for heatmap
    gases_with_pixels = [gas for gas in available_gases
                        if pollution_data[gas].get('pixels')]

    # Find gas with violation (if any)
    violation_gas = None
    for gas in available_gases:
        threshold_check = analyzer.check_threshold_violation(
            gas, pollution_data[gas]['statistics']['max']
        )
        if threshold_check['violated']:
            violation_gas = gas
            break

    # Gas selector
    default_index = 0
    if violation_gas:
        default_index = available_gases.index(violation_gas)

    selected_gas = st.selectbox(
        "Select Gas to Display:",
        available_gases,
        index=default_index,
        format_func=lambda x: f"{x} - {config.GAS_PRODUCTS[x]['name']} {'⚠️ VIOLATION' if x == violation_gas else ''} {'📊' if x in gases_with_pixels else '📈 Stats Only'}"
    )

    if selected_gas:
        gas_data = pollution_data[selected_gas]

        # Display comprehensive data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{selected_gas} Max",
                     f"{gas_data['statistics']['max']:.2f}",
                     f"{gas_data['unit']}")
        with col2:
            st.metric(f"{selected_gas} Mean",
                     f"{gas_data['statistics']['mean']:.2f}",
                     f"Min: {gas_data['statistics']['min']:.2f}")
        with col3:
            if gas_data.get('wind', {}).get('success'):
                wind = gas_data['wind']
                st.metric("Wind",
                         f"{wind['speed_ms']:.1f} m/s",
                         f"From {wind['direction_cardinal']} ({wind['direction_deg']:.0f}°)")
            else:
                st.metric("Wind", "No data", "—")
        with col4:
            # Show pixel count if available
            pixel_count = gas_data.get('statistics', {}).get('pixel_count', 0)
            st.metric("Data Points",
                     pixel_count,
                     "pixels" if pixel_count > 0 else "No spatial data")

        # Show detailed timing information
        with st.expander("🕐 Detailed Timing Information (All times in KSA)", expanded=True):
            col1, col2, col3 = st.columns(3)

            # Satellite observation time
            with col1:
                sat_time_ksa = gas_data.get('timestamp_ksa', 'N/A')
                if hasattr(sat_time_ksa, 'strftime'):
                    st.info(f"**🛰️ Satellite Pass:**\n{sat_time_ksa.strftime('%Y-%m-%d %H:%M:%S KSA')}")
                else:
                    st.info(f"**🛰️ Satellite Pass:**\n{sat_time_ksa}")

            # Wind observation time
            with col2:
                if gas_data.get('wind', {}).get('timestamp_ksa'):
                    wind_time = gas_data['wind']['timestamp_ksa']
                    if hasattr(wind_time, 'strftime'):
                        st.info(f"**💨 Wind Reading:**\n{wind_time.strftime('%Y-%m-%d %H:%M:%S KSA')}")
                    else:
                        st.info(f"**💨 Wind Reading:**\n{wind_time}")
                else:
                    st.info("**💨 Wind Reading:**\nNo wind data")

            # Time synchronization info
            with col3:
                if gas_data.get('wind', {}).get('time_difference_minutes') is not None:
                    time_diff = gas_data['wind']['time_difference_minutes']
                    confidence = gas_data['wind'].get('confidence', 0)
                    if time_diff < 30:
                        quality = "🟢 Excellent"
                    elif time_diff < 60:
                        quality = "🟡 Good"
                    else:
                        quality = "🔴 Poor"
                    st.info(f"**⏱️ Sync Quality:**\n{quality}\nΔt: {time_diff:.0f} min\nConfidence: {confidence:.0f}%")
                else:
                    st.info("**⏱️ Sync Quality:**\nNo sync data")

        # Find hotspot
        hotspot = analyzer.find_hotspot(gas_data)

        # Get nearby factories
        factories = None
        if hotspot:
            factories = analyzer.find_nearby_factories(hotspot, city)

            # Add wind data to factory ranking
            wind_data = gas_data.get('wind', {})
            if wind_data.get('success'):
                factories = analyzer.calculate_wind_vector_to_factories(
                    hotspot, factories, wind_data
                )

        # Check if this is a violation
        threshold_check = analyzer.check_threshold_violation(selected_gas, gas_data['statistics']['max'])
        violation = threshold_check['violated']

        # Create the folium map with heatmap
        gas_data_for_map = {
            'city': city,
            'gas': selected_gas,
            'pixels': gas_data.get('pixels', []),
            'statistics': gas_data.get('statistics', {}),
            'unit': gas_data.get('unit', '')
        }

        wind_data = gas_data.get('wind', {})

        # Create the pollution map
        pollution_map = visualizer.create_pollution_map(
            gas_data=gas_data_for_map,
            wind_data=wind_data,
            hotspot=hotspot,
            factories=factories,
            violation=violation
        )

        # Display the map
        st_folium(pollution_map, width=None, height=600, returned_objects=[])

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
        refresh_interval = st.session_state.get('refresh_interval', 6)
        interval_text = f"{refresh_interval} hours" if refresh_interval >= 1 else f"{int(refresh_interval*60)} minutes"
        st.info(f"🔄 Auto-refresh enabled - Updates every {interval_text}")

    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🌡️ AQI Dashboard",
        "🗺️ Map View",
        "📈 Analysis",
        "⚠️ Violations",
        "💡 Insights"
    ])

    # Fetch data
    if not st.session_state.pollution_data:
        with st.spinner(f"Fetching pollution data for {city}..."):
            st.session_state.pollution_data = fetch_pollution_data(city, days_back)
            ksa_tz = pytz.timezone(config.TIMEZONE)
            st.session_state.last_update = datetime.now(ksa_tz).strftime("%Y-%m-%d %H:%M:%S KSA")

    pollution_data = st.session_state.pollution_data

    # Check if we have any data
    if not pollution_data:
        st.error("❌ Unable to fetch pollution data. Please check your connection and try again.")
        st.info("Possible reasons: Google Earth Engine authentication issues, network problems, or no recent satellite data.")
        if st.button("Retry"):
            st.session_state.pollution_data = {}
            st.rerun()
        return

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
            if pollution_data:
                st.metric("Data Quality", "High" if all(d.get('success') for d in pollution_data.values()) else "Partial")
            else:
                st.metric("Data Quality", "No Data")

    with tab2:
        st.header("🌡️ Air Quality Index Dashboard")
        # Initialize validator
        _, _, _, validator = initialize_services()

        # AQI Dashboard
        create_aqi_dashboard(pollution_data, validator)
        st.divider()

        # Health Risk Panel
        create_health_risk_panel(pollution_data, validator)

    with tab3:
        st.header(f"🗺️ Pollution Map - {city}")
        display_map(pollution_data, city)

    with tab4:
        st.header("📈 Detailed Analysis")

        # Display trends only
        display_trends(pollution_data)

    with tab5:
        st.header("⚠️ Violation Details")
        display_violations(pollution_data, city)

    with tab6:
        st.header("💡 Intelligent Insights & Predictions")
        _, _, _, validator = initialize_services()

        # Insights panel
        create_insights_panel(pollution_data, city, validator)

        # Additional analytics
        with st.expander("🔬 Advanced Analytics"):
            st.subheader("Data Validation Report")
            for gas, data in pollution_data.items():
                if data.get('success'):
                    validation = validator.validate_measurement(gas, data['statistics']['max'], data['unit'])
                    if validation['warnings'] or validation['errors']:
                        st.write(f"**{gas}:**")
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")
                        for error in validation['errors']:
                            st.error(f"❌ {error}")

    # Footer with enhanced information
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("**Data Source:** ESA Sentinel-5P TROPOMI")
    with col2:
        st.caption("**Standards:** WHO 2021 Guidelines")
    with col3:
        ksa_tz = pytz.timezone(config.TIMEZONE)
        current_time = datetime.now(ksa_tz)
        st.caption(f"**System Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S KSA')}")

if __name__ == "__main__":
    main()