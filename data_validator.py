"""
Data Validator and Quality Assurance Module
Ensures data integrity and provides quality metrics for world-class monitoring
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pytz
import config
from translations import get_text

logger = logging.getLogger(__name__)


def get_current_language():
    """Get the current language from Streamlit session state, or default to English."""
    try:
        import streamlit as st
        return st.session_state.get('language', 'en')
    except Exception:
        return 'en'

class DataValidator:
    """Validate and quality-check pollution data"""

    def __init__(self):
        """Initialize validator with quality thresholds"""
        self.quality_thresholds = {
            'pixel_count_min': 10,  # Minimum pixels for reliable data
            'time_sync_excellent': 30,  # Minutes for excellent sync
            'time_sync_good': 60,  # Minutes for good sync
            'confidence_high': 80,  # High confidence threshold
            'confidence_medium': 50,  # Medium confidence threshold
        }

    def calculate_aqi(self, gas: str, concentration: float) -> Dict:
        """
        Calculate Air Quality Index (AQI) based on concentration
        Using US EPA and WHO standards combined

        Returns:
            Dict with AQI value, category, color, and health implications
        """
        # AQI breakpoints for different gases (simplified)
        aqi_breakpoints = {
            'NO2': [
                (0, 53, 0, 50, "Good", "#00E400", "Air quality is satisfactory"),
                (54, 100, 51, 100, "Moderate", "#FFFF00", "Acceptable for most"),
                (101, 360, 101, 150, "Unhealthy for Sensitive", "#FF7E00", "Sensitive groups may experience effects"),
                (361, 649, 151, 200, "Unhealthy", "#FF0000", "Everyone may experience effects"),
                (650, 1249, 201, 300, "Very Unhealthy", "#8F3F97", "Health warnings"),
                (1250, 2049, 301, 500, "Hazardous", "#7E0023", "Emergency conditions")
            ],
            'SO2': [
                (0, 35, 0, 50, "Good", "#00E400", "Air quality is satisfactory"),
                (36, 75, 51, 100, "Moderate", "#FFFF00", "Acceptable for most"),
                (76, 185, 101, 150, "Unhealthy for Sensitive", "#FF7E00", "Sensitive groups may experience effects"),
                (186, 304, 151, 200, "Unhealthy", "#FF0000", "Everyone may experience effects"),
                (305, 604, 201, 300, "Very Unhealthy", "#8F3F97", "Health warnings"),
                (605, 1004, 301, 500, "Hazardous", "#7E0023", "Emergency conditions")
            ],
            'CO': [
                (0, 4.4, 0, 50, "Good", "#00E400", "Air quality is satisfactory"),
                (4.5, 9.4, 51, 100, "Moderate", "#FFFF00", "Acceptable for most"),
                (9.5, 12.4, 101, 150, "Unhealthy for Sensitive", "#FF7E00", "Sensitive groups may experience effects"),
                (12.5, 15.4, 151, 200, "Unhealthy", "#FF0000", "Everyone may experience effects"),
                (15.5, 30.4, 201, 300, "Very Unhealthy", "#8F3F97", "Health warnings"),
                (30.5, 50.4, 301, 500, "Hazardous", "#7E0023", "Emergency conditions")
            ],
            'O3': [
                (0, 54, 0, 50, "Good", "#00E400", "Air quality is satisfactory"),
                (55, 70, 51, 100, "Moderate", "#FFFF00", "Acceptable for most"),
                (71, 85, 101, 150, "Unhealthy for Sensitive", "#FF7E00", "Sensitive groups may experience effects"),
                (86, 105, 151, 200, "Unhealthy", "#FF0000", "Everyone may experience effects"),
                (106, 200, 201, 300, "Very Unhealthy", "#8F3F97", "Health warnings"),
                (201, 400, 301, 500, "Hazardous", "#7E0023", "Emergency conditions")
            ]
        }

        # Default AQI if gas not in breakpoints
        lang = get_current_language()
        if gas not in aqi_breakpoints:
            return {
                'aqi': None,
                'category': get_text('unknown', lang),
                'color': '#808080',
                'description': get_text('aqi_not_available', lang),
                'health_implications': get_text('refer_who', lang)
            }

        # Find appropriate breakpoint
        for bp in aqi_breakpoints[gas]:
            c_low, c_high, i_low, i_high, category, color, description = bp
            if c_low <= concentration <= c_high:
                # Linear interpolation
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                return {
                    'aqi': round(aqi),
                    'category': category,
                    'color': color,
                    'description': description,
                    'health_implications': self._get_health_recommendations(category)
                }

        # If concentration exceeds all breakpoints
        return {
            'aqi': 500,
            'category': get_text('aqi_hazardous', lang),
            'color': '#7E0023',
            'description': get_text('emergency_conditions', lang),
            'health_implications': get_text('avoid_outdoor', lang)
        }

    def _get_health_recommendations(self, category: str) -> str:
        """Get health recommendations based on AQI category"""
        lang = get_current_language()
        recommendations = {
            "Good": get_text('health_good', lang),
            "Moderate": get_text('health_moderate', lang),
            "Unhealthy for Sensitive": get_text('health_sensitive', lang),
            "Unhealthy": get_text('health_unhealthy', lang),
            "Very Unhealthy": get_text('health_very_unhealthy', lang),
            "Hazardous": get_text('health_hazardous', lang)
        }
        return recommendations.get(category, get_text('follow_advisories', lang))

    def validate_measurement(self, gas: str, value: float, unit: str) -> Dict:
        """
        Validate a single measurement value

        Returns:
            Dict with validation status and any warnings
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 100
        }

        # Check for negative values
        if value < 0:
            validation['errors'].append("Negative value detected - likely sensor error")
            validation['valid'] = False
            validation['quality_score'] = 0
            return validation

        # Check for unrealistic values based on gas type
        max_realistic = {
            'NO2': 1000,  # 10^15 molecules/cm²
            'SO2': 500,
            'CO': 100,  # 10^18 molecules/cm²
            'O3': 600,
            'HCHO': 50,
            'CH4': 2000
        }

        if gas in max_realistic and value > max_realistic[gas]:
            validation['warnings'].append(f"Unusually high value ({value:.2f} {unit})")
            validation['quality_score'] -= 20

        # Check against WHO critical thresholds
        threshold_info = config.GAS_THRESHOLDS.get(gas, {})
        critical = threshold_info.get('critical_threshold', float('inf'))

        if value > critical * 2:
            validation['warnings'].append("Value exceeds 2x critical threshold - verify data accuracy")
            validation['quality_score'] -= 30

        return validation

    def calculate_data_quality_score(self, data: Dict) -> Dict:
        """
        Calculate comprehensive data quality score

        Returns:
            Dict with overall score and component scores
        """
        scores = {
            'spatial_coverage': 0,
            'temporal_accuracy': 0,
            'measurement_validity': 0,
            'wind_sync_quality': 0,
            'overall': 0
        }

        # Spatial coverage score
        pixel_count = data.get('statistics', {}).get('pixel_count', 0)
        if pixel_count >= 50:
            scores['spatial_coverage'] = 100
        elif pixel_count >= 20:
            scores['spatial_coverage'] = 80
        elif pixel_count >= 10:
            scores['spatial_coverage'] = 60
        elif pixel_count > 0:
            scores['spatial_coverage'] = 40
        else:
            scores['spatial_coverage'] = 0

        # Temporal accuracy (data freshness)
        if 'timestamp_utc' in data:
            ksa_tz = pytz.timezone(config.TIMEZONE)
            now = datetime.now(pytz.UTC)
            data_time = data['timestamp_utc']
            if isinstance(data_time, str):
                data_time = datetime.fromisoformat(data_time.replace('Z', '+00:00'))
            age_hours = (now - data_time).total_seconds() / 3600

            if age_hours <= 3:
                scores['temporal_accuracy'] = 100
            elif age_hours <= 6:
                scores['temporal_accuracy'] = 90
            elif age_hours <= 12:
                scores['temporal_accuracy'] = 70
            elif age_hours <= 24:
                scores['temporal_accuracy'] = 50
            else:
                scores['temporal_accuracy'] = max(0, 100 - int(age_hours * 2))

        # Measurement validity
        if data.get('success'):
            max_val = data.get('statistics', {}).get('max', 0)
            gas = data.get('gas', '')
            validation = self.validate_measurement(gas, max_val, data.get('unit', ''))
            scores['measurement_validity'] = validation['quality_score']

        # Wind synchronization quality
        if data.get('wind', {}).get('confidence'):
            scores['wind_sync_quality'] = data['wind']['confidence']

        # Calculate overall score (weighted average)
        weights = {
            'spatial_coverage': 0.25,
            'temporal_accuracy': 0.25,
            'measurement_validity': 0.30,
            'wind_sync_quality': 0.20
        }

        scores['overall'] = sum(scores[key] * weights[key] for key in weights)

        # Add quality label
        lang = get_current_language()
        if scores['overall'] >= 80:
            scores['label'] = f"🟢 {get_text('quality_excellent', lang)}"
        elif scores['overall'] >= 60:
            scores['label'] = f"🟡 {get_text('quality_good', lang)}"
        elif scores['overall'] >= 40:
            scores['label'] = f"🟠 {get_text('quality_fair', lang)}"
        else:
            scores['label'] = f"🔴 {get_text('quality_poor', lang)}"

        return scores

    def generate_data_insights(self, pollution_data: Dict, city: str) -> List[str]:
        """
        Generate intelligent insights from the data

        Returns:
            List of insight strings
        """
        insights = []
        lang = get_current_language()

        # Check for simultaneous violations
        violations = []
        for gas, data in pollution_data.items():
            if data.get('success'):
                threshold = config.GAS_THRESHOLDS.get(gas, {}).get('column_threshold', float('inf'))
                if data['statistics']['max'] > threshold:
                    violations.append(gas)

        if len(violations) > 1:
            insight_text = get_text('insight_multiple_violations', lang).format(gases=', '.join(violations))
            insights.append(insight_text)

        # Check for unusual patterns
        high_variance_gases = []
        for gas, data in pollution_data.items():
            if data.get('success') and data.get('statistics'):
                mean = data['statistics'].get('mean', 0)
                max_val = data['statistics'].get('max', 0)
                if mean > 0 and (max_val / mean) > 3:
                    high_variance_gases.append(gas)

        if high_variance_gases:
            insight_text = get_text('insight_high_variance', lang).format(gases=', '.join(high_variance_gases))
            insights.append(insight_text)

        # Check wind patterns
        wind_speeds = []
        for gas, data in pollution_data.items():
            if data.get('wind', {}).get('speed_ms'):
                wind_speeds.append(data['wind']['speed_ms'])

        if wind_speeds:
            avg_wind = np.mean(wind_speeds)
            if avg_wind < 2:
                insights.append(get_text('insight_low_wind', lang))
            elif avg_wind > 10:
                insights.append(get_text('insight_high_wind', lang))

        # Time-based insights
        ksa_tz = pytz.timezone(config.TIMEZONE)
        current_hour = datetime.now(ksa_tz).hour
        if 6 <= current_hour <= 9:
            insights.append(get_text('insight_morning_rush', lang))
        elif 17 <= current_hour <= 20:
            insights.append(get_text('insight_evening_rush', lang))

        # Seasonal insights (simplified)
        current_month = datetime.now().month
        if current_month in [6, 7, 8]:
            insights.append(get_text('insight_summer', lang))
        elif current_month in [11, 12, 1, 2]:
            insights.append(get_text('insight_winter', lang))

        return insights

    def calculate_health_risk_index(self, pollution_data: Dict) -> Dict:
        """
        Calculate comprehensive health risk index

        Returns:
            Dict with risk scores and recommendations
        """
        risk_scores = {}
        total_risk = 0
        gas_count = 0

        for gas, data in pollution_data.items():
            if data.get('success'):
                max_val = data['statistics']['max']
                threshold = config.GAS_THRESHOLDS.get(gas, {}).get('column_threshold', float('inf'))
                critical = config.GAS_THRESHOLDS.get(gas, {}).get('critical_threshold', float('inf'))

                # Calculate risk score (0-100)
                if max_val <= threshold * 0.5:
                    risk = 0
                elif max_val <= threshold:
                    risk = 25 * (max_val / threshold)
                elif max_val <= critical:
                    risk = 25 + 50 * ((max_val - threshold) / (critical - threshold))
                else:
                    risk = min(100, 75 + 25 * ((max_val - critical) / critical))

                risk_scores[gas] = risk
                total_risk += risk
                gas_count += 1

        # Calculate overall risk
        overall_risk = total_risk / gas_count if gas_count > 0 else 0

        # Determine risk level and recommendations
        lang = get_current_language()
        if overall_risk < 20:
            risk_level = get_text('risk_low', lang)
            color = "#00E400"
            recommendations = [get_text('safe_outdoor', lang), get_text('no_precautions', lang)]
        elif overall_risk < 40:
            risk_level = get_text('risk_moderate', lang)
            color = "#FFFF00"
            recommendations = [get_text('monitor_symptoms', lang), get_text('limit_exertion', lang)]
        elif overall_risk < 60:
            risk_level = get_text('risk_high', lang)
            color = "#FF7E00"
            recommendations = [get_text('reduce_outdoor', lang), get_text('keep_windows_closed', lang), get_text('use_purifiers', lang)]
        elif overall_risk < 80:
            risk_level = get_text('risk_very_high', lang)
            color = "#FF0000"
            recommendations = [get_text('avoid_outdoor_activities', lang), get_text('seal_indoor', lang), get_text('wear_masks', lang)]
        else:
            risk_level = get_text('risk_severe', lang)
            color = "#7E0023"
            recommendations = [get_text('stay_indoors', lang), get_text('emergency_measures', lang), get_text('follow_advisories', lang)]

        return {
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'color': color,
            'gas_risks': risk_scores,
            'recommendations': recommendations
        }