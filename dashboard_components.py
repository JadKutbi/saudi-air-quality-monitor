"""
Advanced Dashboard Components for World-Class Air Quality Monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List
import pytz
from datetime import datetime
import config

def create_aqi_dashboard(pollution_data: Dict, validator) -> None:
    """Create comprehensive AQI dashboard"""
    st.subheader("🌡️ Air Quality Index (AQI) Dashboard")

    # Calculate AQI for all gases
    aqi_data = []
    for gas, data in pollution_data.items():
        if data.get('success'):
            aqi_info = validator.calculate_aqi(gas, data['statistics']['max'])
            if aqi_info['aqi']:
                aqi_data.append({
                    'Gas': gas,
                    'AQI': aqi_info['aqi'],
                    'Category': aqi_info['category'],
                    'Color': aqi_info['color']
                })

    if aqi_data:
        # Display overall AQI (worst case)
        max_aqi = max(aqi_data, key=lambda x: x['AQI'])

        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            # AQI Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_aqi['AQI'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall AQI"},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': max_aqi['Color']},
                    'steps': [
                        {'range': [0, 50], 'color': "#E8F5E9"},
                        {'range': [50, 100], 'color': "#FFF9C4"},
                        {'range': [100, 150], 'color': "#FFE0B2"},
                        {'range': [150, 200], 'color': "#FFCCBC"},
                        {'range': [200, 300], 'color': "#E1BEE7"},
                        {'range': [300, 500], 'color': "#FFCDD2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Air Quality Status", max_aqi['Category'])
            st.markdown(f"**Dominant Pollutant:** {max_aqi['Gas']}")

            # AQI breakdown by gas
            df_aqi = pd.DataFrame(aqi_data)
            fig_bar = px.bar(df_aqi, x='Gas', y='AQI', color='Category',
                            color_discrete_map={
                                'Good': '#00E400',
                                'Moderate': '#FFFF00',
                                'Unhealthy for Sensitive': '#FF7E00',
                                'Unhealthy': '#FF0000',
                                'Very Unhealthy': '#8F3F97',
                                'Hazardous': '#7E0023'
                            },
                            title="AQI by Pollutant")
            fig_bar.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col3:
            # Health recommendations
            st.info("**Health Advice:**")
            aqi_info = validator.calculate_aqi(max_aqi['Gas'],
                                              pollution_data[max_aqi['Gas']]['statistics']['max'])
            st.write(aqi_info['health_implications'])

def create_health_risk_panel(pollution_data: Dict, validator) -> None:
    """Create health risk assessment panel"""
    st.subheader("🏥 Health Risk Assessment")

    risk_info = validator.calculate_health_risk_index(pollution_data)

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        # Risk level indicator
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: {risk_info['color']}20;
                    border: 2px solid {risk_info['color']}; border-radius: 10px;">
            <h2 style="color: {risk_info['color']}; margin: 0;">{risk_info['risk_level']}</h2>
            <p style="margin: 5px 0;">Risk Score: {risk_info['overall_risk']:.0f}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Risk by pollutant
        if risk_info['gas_risks']:
            risk_df = pd.DataFrame([
                {'Pollutant': gas, 'Risk Score': score}
                for gas, score in risk_info['gas_risks'].items()
            ])
            fig = px.bar(risk_df, x='Risk Score', y='Pollutant', orientation='h',
                        color='Risk Score', color_continuous_scale='RdYlGn_r',
                        title="Risk by Pollutant")
            fig.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Recommendations
        st.markdown("**📋 Recommendations:**")
        for rec in risk_info['recommendations']:
            st.write(f"• {rec}")

def create_data_quality_panel(pollution_data: Dict, validator) -> None:
    """Create data quality indicators panel"""
    st.subheader("📊 Data Quality Indicators")

    quality_scores = {}
    for gas, data in pollution_data.items():
        if data.get('success'):
            quality_scores[gas] = validator.calculate_data_quality_score(data)

    if quality_scores:
        # Create quality matrix
        quality_df = pd.DataFrame([
            {
                'Gas': gas,
                'Overall': scores['overall'],
                'Spatial': scores['spatial_coverage'],
                'Temporal': scores['temporal_accuracy'],
                'Validity': scores['measurement_validity'],
                'Wind Sync': scores['wind_sync_quality'],
                'Status': scores['label']
            }
            for gas, scores in quality_scores.items()
        ])

        # Display quality heatmap
        fig = go.Figure(data=go.Heatmap(
            z=quality_df[['Spatial', 'Temporal', 'Validity', 'Wind Sync']].values,
            x=['Spatial Coverage', 'Temporal Accuracy', 'Data Validity', 'Wind Sync'],
            y=quality_df['Gas'].values,
            colorscale='RdYlGn',
            text=quality_df[['Spatial', 'Temporal', 'Validity', 'Wind Sync']].values,
            texttemplate='%{text:.0f}',
            textfont={"size": 10},
            colorbar=dict(title="Quality Score")
        ))

        fig.update_layout(
            title="Data Quality Matrix",
            height=200 + len(quality_scores) * 30,
            xaxis_title="Quality Metric",
            yaxis_title="Pollutant"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Overall quality summary
        avg_quality = quality_df['Overall'].mean()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average Quality", f"{avg_quality:.0f}%")
        with col2:
            best_quality = quality_df.loc[quality_df['Overall'].idxmax()]
            st.metric("Best Quality", f"{best_quality['Gas']}: {best_quality['Overall']:.0f}%")
        with col3:
            worst_quality = quality_df.loc[quality_df['Overall'].idxmin()]
            st.metric("Needs Attention", f"{worst_quality['Gas']}: {worst_quality['Overall']:.0f}%")
        with col4:
            high_quality_count = len(quality_df[quality_df['Overall'] >= 80])
            st.metric("High Quality", f"{high_quality_count}/{len(quality_df)} gases")

def create_insights_panel(pollution_data: Dict, city: str, validator) -> None:
    """Create intelligent insights panel"""
    st.subheader("💡 Intelligent Insights")

    insights = validator.generate_data_insights(pollution_data, city)

    if insights:
        # Display insights in a clean format
        cols = st.columns(2)
        for i, insight in enumerate(insights):
            with cols[i % 2]:
                st.info(insight)
    else:
        st.info("No significant patterns detected in current data")

    # Add trend analysis
    with st.expander("📈 Detailed Trend Analysis"):
        # Correlation analysis
        st.write("**Pollutant Correlations:**")

        correlation_data = []
        gases = list(pollution_data.keys())
        for i, gas1 in enumerate(gases):
            for gas2 in gases[i+1:]:
                if pollution_data[gas1].get('success') and pollution_data[gas2].get('success'):
                    # Simple correlation based on violation status
                    thresh1 = config.GAS_THRESHOLDS.get(gas1, {}).get('column_threshold', float('inf'))
                    thresh2 = config.GAS_THRESHOLDS.get(gas2, {}).get('column_threshold', float('inf'))

                    viol1 = pollution_data[gas1]['statistics']['max'] > thresh1
                    viol2 = pollution_data[gas2]['statistics']['max'] > thresh2

                    if viol1 and viol2:
                        correlation_data.append(f"• {gas1} and {gas2} both elevated - possible common source")

        if correlation_data:
            for corr in correlation_data:
                st.write(corr)
        else:
            st.write("No significant correlations detected")

def create_historical_comparison(pollution_data: Dict) -> None:
    """Create WHO standards comparison view"""
    st.subheader("📊 Compliance with WHO Air Quality Standards")

    # Compare current satellite measurements against WHO 2021 guidelines
    comparison_data = []
    for gas, data in pollution_data.items():
        if data.get('success'):
            threshold = config.GAS_THRESHOLDS.get(gas, {}).get('column_threshold', 100)
            max_val = data['statistics']['max']
            mean_val = data['statistics']['mean']

            # Calculate compliance percentage
            max_compliance = (max_val / threshold * 100) if threshold > 0 else 0
            mean_compliance = (mean_val / threshold * 100) if threshold > 0 else 0

            comparison_data.append({
                'Gas': gas,
                'Peak Level': max_val,
                'Average Level': mean_val,
                'WHO Guideline': threshold,
                'Peak % of Limit': max_compliance,
                'Status': '🔴 Violation' if max_val > threshold else '🟢 Compliant'
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Peak Concentration',
            x=df['Gas'],
            y=df['Peak Level'],
            marker_color='#ef4444',
            text=df['Peak % of Limit'].round(0),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Spatial Average',
            x=df['Gas'],
            y=df['Average Level'],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Scatter(
            name='WHO 2021 Guideline',
            x=df['Gas'],
            y=df['WHO Guideline'],
            mode='lines+markers',
            marker=dict(color='#22c55e', size=12, symbol='line-ew'),
            line=dict(color='#22c55e', width=3, dash='dash')
        ))

        fig.update_layout(
            title="Current Satellite Measurements vs WHO 2021 Air Quality Guidelines",
            barmode='group',
            height=350,
            xaxis_title="Pollutant Gas",
            yaxis_title="Concentration (Column Density)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add compliance summary
        violations = df[df['Status'].str.contains('Violation')]
        if len(violations) > 0:
            st.warning(f"⚠️ **{len(violations)} pollutant(s) exceeding WHO guidelines**: {', '.join(violations['Gas'].tolist())}")
        else:
            st.success("✅ All pollutants within WHO 2021 air quality guidelines")