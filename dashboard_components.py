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
from translations import get_text


def t(key: str) -> str:
    """Get translated text for current language."""
    return get_text(key, st.session_state.get('language', 'en'))


def get_spi_category_translated(category: str) -> str:
    """Translate Satellite Pollution Index category name"""
    category_map = {
        'Background': 'spi_background',
        'Normal': 'spi_normal',
        'Elevated': 'spi_elevated',
        'Violation': 'spi_violation',
        'Critical': 'spi_critical'
    }
    key = category_map.get(category, 'unknown')
    return t(key)


def create_aqi_dashboard(pollution_data: Dict, validator) -> None:
    """Create Satellite Pollution Index (SPI) dashboard based on Sentinel-5P thresholds"""
    st.subheader(f"📊 {t('spi_dashboard')}")

    # Info box explaining the metric
    with st.expander(f"ℹ️ {t('about_spi')}", expanded=False):
        st.markdown(t('spi_explanation'))

    # Calculate SPI for all gases
    spi_data = []
    for gas, data in pollution_data.items():
        if data.get('success'):
            spi_info = validator.calculate_satellite_pollution_index(gas, data['statistics']['max'])
            if spi_info['index'] is not None:
                spi_data.append({
                    'Gas': gas,
                    'Index': spi_info['index'],
                    'Percentage': spi_info['percentage'],
                    'Category': spi_info['category'],
                    'CategoryTranslated': get_spi_category_translated(spi_info['category']),
                    'Color': spi_info['color'],
                    'Value': data['statistics']['max'],
                    'Unit': data['unit']
                })

    if spi_data:
        # Display overall SPI (worst case)
        max_spi = max(spi_data, key=lambda x: x['Index'])

        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            # SPI Gauge - scale 0-200 based on threshold percentages
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=max_spi['Index'],
                domain={'x': [0, 1], 'y': [0.15, 1]},
                gauge={
                    'axis': {'range': [0, 200], 'tickwidth': 1, 'tickvals': [0, 50, 75, 100, 150, 200]},
                    'bar': {'color': max_spi['Color']},
                    'steps': [
                        {'range': [0, 50], 'color': "#E8F5E9"},    # Background - dark green bg
                        {'range': [50, 75], 'color': "#C8E6C9"},   # Normal - light green bg
                        {'range': [75, 100], 'color': "#FFF9C4"},  # Elevated - yellow bg
                        {'range': [100, 150], 'color': "#FFE0B2"}, # Violation - orange bg
                        {'range': [150, 200], 'color': "#FFCDD2"}  # Critical - red bg
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100  # Violation threshold
                    }
                }
            ))
            # Add centered number and title as annotations
            fig.add_annotation(
                x=0.5, y=0.35,
                text=f"<b>{int(max_spi['Percentage'])}%</b>",
                font=dict(size=48, color="#333"),
                showarrow=False,
                xanchor='center',
                yanchor='middle'
            )
            fig.add_annotation(
                x=0.5, y=0.15,
                text=t('of_threshold'),
                font=dict(size=14, color="#666"),
                showarrow=False,
                xanchor='center',
                yanchor='middle'
            )
            fig.add_annotation(
                x=0.5, y=0.95,
                text=t('worst_pollutant'),
                font=dict(size=16, color="#333"),
                showarrow=False,
                xanchor='center',
                yanchor='top'
            )
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=40, t=30, b=20),
                autosize=True
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with col2:
            st.metric(t('pollution_status'), max_spi['CategoryTranslated'])
            st.markdown(f"**{t('dominant_pollutant')}:** {max_spi['Gas']} ({max_spi['Percentage']}% {t('of_threshold_label')})")

            # SPI breakdown by gas - show percentage of threshold
            df_spi = pd.DataFrame(spi_data)
            fig_bar = px.bar(df_spi, x='Gas', y='Percentage', color='CategoryTranslated',
                            color_discrete_map={
                                t('spi_background'): '#00E400',
                                t('spi_normal'): '#90EE90',
                                t('spi_elevated'): '#FFFF00',
                                t('spi_violation'): '#FF7E00',
                                t('spi_critical'): '#FF0000',
                                # Fallback for English categories
                                'Background': '#00E400',
                                'Normal': '#90EE90',
                                'Elevated': '#FFFF00',
                                'Violation': '#FF7E00',
                                'Critical': '#FF0000'
                            },
                            title=t('threshold_by_pollutant'))
            # Add 100% reference line
            fig_bar.add_hline(y=100, line_dash="dash", line_color="red",
                            annotation_text=t('violation_threshold'))
            fig_bar.update_layout(height=220, showlegend=False,
                                yaxis_title=t('percent_of_threshold'))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col3:
            # Health recommendations
            st.info(f"**{t('health_advice')}:**")
            spi_info = validator.calculate_satellite_pollution_index(
                max_spi['Gas'],
                pollution_data[max_spi['Gas']]['statistics']['max']
            )
            st.write(spi_info['health_implications'])

            # Show actual values
            st.markdown(f"**{t('measured_values')}:**")
            for item in sorted(spi_data, key=lambda x: x['Percentage'], reverse=True)[:3]:
                status_icon = "🔴" if item['Percentage'] >= 100 else "🟡" if item['Percentage'] >= 75 else "🟢"
                st.caption(f"{status_icon} {item['Gas']}: {item['Value']:.2f} {item['Unit']}")
    else:
        st.warning(t('no_data_for_index'))

def create_gas_health_effects_panel(pollution_data: Dict, validator) -> None:
    """Create detailed health effects panel for each gas"""
    st.subheader(f"⚠️ {t('health_effects_title')}")

    st.caption(t('health_effects_note'))

    # Get gases with elevated levels (sorted by severity)
    gas_status = []
    for gas, data in pollution_data.items():
        if data.get('success'):
            spi_info = validator.calculate_satellite_pollution_index(gas, data['statistics']['max'])
            if spi_info['index'] is not None:
                gas_status.append({
                    'gas': gas,
                    'percentage': spi_info['percentage'],
                    'category': spi_info['category'],
                    'color': spi_info['color'],
                    'value': data['statistics']['max'],
                    'unit': data['unit']
                })

    # Sort by severity (percentage of threshold)
    gas_status.sort(key=lambda x: x['percentage'], reverse=True)

    if not gas_status:
        st.info(t('no_data_available'))
        return

    # Display health effects for each gas
    for item in gas_status:
        gas = item['gas']
        pct = item['percentage']

        # Determine severity styling
        if pct >= 100:
            severity_color = "#dc2626"  # Red
            severity_bg = "#fee2e2"
            severity_icon = "🔴"
            severity_text = t('critical') if pct >= 150 else t('violation')
        elif pct >= 75:
            severity_color = "#f59e0b"  # Orange
            severity_bg = "#fef3c7"
            severity_icon = "🟡"
            severity_text = t('spi_elevated')
        else:
            severity_color = "#10b981"  # Green
            severity_bg = "#d1fae5"
            severity_icon = "🟢"
            severity_text = t('normal')

        with st.expander(f"{severity_icon} **{gas}** - {config.GAS_PRODUCTS[gas]['name']} ({pct:.0f}% {t('of_threshold_label')})", expanded=(pct >= 75)):
            # Current level indicator
            st.markdown(f"""
            <div style="background-color: {severity_bg}; border-left: 4px solid {severity_color}; padding: 10px; margin-bottom: 15px; border-radius: 5px;">
                <strong>{t('current_exposure_level')}:</strong> {severity_text} ({item['value']:.2f} {item['unit']})
            </div>
            """, unsafe_allow_html=True)

            # Health effects in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**🩺 {t('short_term_effects')}:**")
                st.write(t(f'{gas}_effects_short'))

                st.markdown(f"**⏳ {t('long_term_effects')}:**")
                st.write(t(f'{gas}_effects_long'))

            with col2:
                st.markdown(f"**👥 {t('sensitive_groups')}:**")
                st.warning(t(f'{gas}_sensitive'))

                st.markdown(f"**🔍 {t('symptoms_to_watch')}:**")
                st.info(t(f'{gas}_symptoms'))


def create_health_risk_panel(pollution_data: Dict, validator) -> None:
    """Create health risk assessment panel"""
    st.subheader(f"🏥 {t('health_risk_assessment')}")

    risk_info = validator.calculate_health_risk_index(pollution_data)

    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        # Risk level indicator
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: {risk_info['color']}20;
                    border: 2px solid {risk_info['color']}; border-radius: 10px;">
            <h2 style="color: {risk_info['color']}; margin: 0;">{risk_info['risk_level']}</h2>
            <p style="margin: 5px 0;">{t('risk_score')}: {risk_info['overall_risk']:.0f}/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Risk by pollutant
        if risk_info['gas_risks']:
            risk_df = pd.DataFrame([
                {t('pollutant'): gas, t('risk_score'): score}
                for gas, score in risk_info['gas_risks'].items()
            ])
            fig = px.bar(risk_df, x=t('risk_score'), y=t('pollutant'), orientation='h',
                        color=t('risk_score'), color_continuous_scale='RdYlGn_r',
                        title=t('risk_by_pollutant'))
            fig.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Recommendations
        st.markdown(f"**📋 {t('recommendations')}:**")
        for rec in risk_info['recommendations']:
            st.write(f"• {rec}")

def create_data_quality_panel(pollution_data: Dict, validator) -> None:
    """Create data quality indicators panel"""
    st.subheader(f"📊 {t('data_quality')}")

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
            x=[t('spatial_coverage'), t('temporal_accuracy'), t('measurement_validity'), t('wind_sync')],
            y=quality_df['Gas'].values,
            colorscale='RdYlGn',
            text=quality_df[['Spatial', 'Temporal', 'Validity', 'Wind Sync']].values,
            texttemplate='%{text:.0f}',
            textfont={"size": 10},
            colorbar=dict(title=t('quality_score'))
        ))

        fig.update_layout(
            title=t('data_quality_matrix'),
            height=200 + len(quality_scores) * 30,
            xaxis_title=t('quality_metric'),
            yaxis_title=t('pollutant')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Overall quality summary
        avg_quality = quality_df['Overall'].mean()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(t('average_quality'), f"{avg_quality:.0f}%")
        with col2:
            best_quality = quality_df.loc[quality_df['Overall'].idxmax()]
            st.metric(t('best_quality'), f"{best_quality['Gas']}: {best_quality['Overall']:.0f}%")
        with col3:
            worst_quality = quality_df.loc[quality_df['Overall'].idxmin()]
            st.metric(t('needs_attention'), f"{worst_quality['Gas']}: {worst_quality['Overall']:.0f}%")
        with col4:
            high_quality_count = len(quality_df[quality_df['Overall'] >= 80])
            st.metric(t('high_quality'), f"{high_quality_count}/{len(quality_df)} {t('gases')}")

def create_insights_panel(pollution_data: Dict, city: str, validator) -> None:
    """Create intelligent insights panel"""
    st.subheader(f"💡 {t('intelligent_insights')}")

    insights = validator.generate_data_insights(pollution_data, city)

    if insights:
        # Display insights in a clean format
        cols = st.columns(2)
        for i, insight in enumerate(insights):
            with cols[i % 2]:
                st.info(insight)
    else:
        st.info(t('no_patterns_detected'))

    # Add trend analysis
    with st.expander(f"📈 {t('detailed_trend_analysis')}"):
        # Correlation analysis
        st.write(f"**{t('pollutant_correlations')}:**")

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
                        correlation_data.append(f"• {gas1} & {gas2} {t('both_elevated')}")

        if correlation_data:
            for corr in correlation_data:
                st.write(corr)
        else:
            st.write(t('no_correlations_detected'))

def create_historical_comparison(pollution_data: Dict) -> None:
    """Create satellite threshold comparison view"""
    st.subheader(f"📊 {t('who_compliance')}")

    # Compare current satellite measurements against S5P typical ranges
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
                t('peak_level'): max_val,
                t('average_level'): mean_val,
                t('who_guideline'): threshold,
                t('peak_percent_limit'): max_compliance,
                t('status'): f"🔴 {t('violation')}" if max_val > threshold else f"🟢 {t('compliant')}"
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        # Create comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name=t('peak_concentration'),
            x=df['Gas'],
            y=df[t('peak_level')],
            marker_color='#ef4444',
            text=df[t('peak_percent_limit')].round(0),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name=t('spatial_average'),
            x=df['Gas'],
            y=df[t('average_level')],
            marker_color='#3b82f6'
        ))
        fig.add_trace(go.Scatter(
            name=t('who_guideline'),
            x=df['Gas'],
            y=df[t('who_guideline')],
            mode='lines+markers',
            marker=dict(color='#22c55e', size=12, symbol='line-ew'),
            line=dict(color='#22c55e', width=3, dash='dash')
        ))

        fig.update_layout(
            title=t('current_vs_who'),
            barmode='group',
            height=350,
            xaxis_title=t('pollutant_gas'),
            yaxis_title=t('concentration'),
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
        violations = df[df[t('status')].str.contains(t('violation'))]
        if len(violations) > 0:
            st.warning(f"⚠️ **{len(violations)} {t('pollutants_exceeding')}**: {', '.join(violations['Gas'].tolist())}")
        else:
            st.success(f"✅ {t('all_within_guidelines')}")