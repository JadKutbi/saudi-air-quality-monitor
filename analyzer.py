"""
Pollution Analyzer Module

Analyzes satellite pollution data to detect threshold violations and
attribute sources using multi-factor scoring and AI-powered analysis.

Features:
    - Sentinel-5P based threshold violation detection
    - Wind-based upwind factory identification
    - Multi-factor confidence scoring (wind, distance, emissions)
    - AI source attribution using Gemini with optional vision analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import os
import config


def get_current_language():
    """Get the current language from Streamlit session state, or default to English."""
    try:
        import streamlit as st
        return st.session_state.get('language', 'en')
    except Exception:
        return 'en'

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part, Image
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    try:
        import google.generativeai as genai
    except ImportError:
        genai = None

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class PollutionAnalyzer:
    """Analyze pollution data and attribute sources to industrial facilities."""

    def __init__(self, gemini_api_key: Optional[str] = None,
                 vertex_project: Optional[str] = None,
                 vertex_location: Optional[str] = None):
        """
        Initialize analyzer with AI capabilities.

        Args:
            gemini_api_key: Gemini API key (fallback)
            vertex_project: Google Cloud project ID for Vertex AI (preferred)
            vertex_location: Vertex AI location (e.g., 'us-central1')
        """
        self.model = None
        self.use_vertex = False

        if VERTEX_AI_AVAILABLE and vertex_project and vertex_location:
            try:
                vertexai.init(project=vertex_project, location=vertex_location)
                self.model = GenerativeModel("gemini-3-pro-preview-11-2025")
                self.use_vertex = True
                logger.info(f"Vertex AI initialized (project: {vertex_project})")
            except Exception as e:
                logger.warning(f"Vertex AI initialization failed: {e}")

        if not self.model and gemini_api_key and genai:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-3-pro-preview')
            self.use_vertex = False
            logger.info("Gemini API initialized")
    
    def find_hotspot(self, gas_data: Dict) -> Optional[Dict]:
        """
        Find the most intense pixel (hotspot) for a gas
        
        Args:
            gas_data: Gas data from satellite fetcher
            
        Returns:
            Dictionary with hotspot location and value
        """
        pixels = gas_data.get('pixels', [])
        
        if not pixels:
            logger.warning(f"No pixels available for {gas_data['gas']}")
            return None
        
        # Find pixel with maximum value
        max_pixel = max(pixels, key=lambda p: p['value'])
        
        logger.info(f"Hotspot found at ({max_pixel['lat']:.4f}, {max_pixel['lon']:.4f}) "
                   f"with value {max_pixel['value']:.2f} {gas_data['unit']}")
        
        return {
            'lat': max_pixel['lat'],
            'lon': max_pixel['lon'],
            'value': max_pixel['value'],
            'gas': gas_data['gas'],
            'unit': gas_data['unit']
        }
    
    def check_threshold_violation(self, gas: str, value: float) -> Dict:
        """
        Check if gas concentration exceeds satellite-based thresholds

        Args:
            gas: Gas type
            value: Measured value in display units (10^15 or 10^18 molec/cm²)

        Returns:
            Dictionary with violation status and severity
        """
        threshold_config = config.GAS_THRESHOLDS.get(gas, {})
        threshold = threshold_config.get('column_threshold')
        critical = threshold_config.get('critical_threshold')
        threshold_unit = threshold_config.get('unit')
        gas_unit = config.GAS_PRODUCTS[gas]['display_unit']
        unit_mismatch = False

        if threshold_unit and gas_unit != threshold_unit:
            unit_mismatch = True
            logger.warning(
                "Unit mismatch detected for %s: gas data in %s vs threshold in %s",
                gas,
                gas_unit,
                threshold_unit
            )
        
        if threshold is None:
            return {'violated': False, 'severity': 'unknown'}
        
        if value >= critical:
            severity = 'critical'
            violated = True
        elif value >= threshold:
            severity = 'moderate'
            violated = True
        else:
            severity = 'normal'
            violated = False
        
        return {
            'violated': violated,
            'severity': severity,
            'threshold': threshold,
            'critical_threshold': critical,
            'measured_value': value,
            'percentage_over': ((value - threshold) / threshold * 100) if violated else 0,
            'unit': threshold_unit or gas_unit,
            'unit_mismatch': unit_mismatch,
            'threshold_source': threshold_config.get('source', 'Unknown')
        }
    
    def find_nearby_factories(self, hotspot: Dict, city: str,
                            max_distance_km: float = 20.0) -> List[Dict]:
        """
        Find factories near the pollution hotspot

        Args:
            hotspot: Hotspot location dictionary
            city: City name
            max_distance_km: Maximum search radius

        Returns:
            List of nearby factories with distances and default wind attributes
        """
        factories = config.FACTORIES.get(city, [])
        nearby = []

        for factory in factories:
            distance = self._haversine_distance(
                hotspot['lat'], hotspot['lon'],
                factory['location'][0], factory['location'][1]
            )

            if distance <= max_distance_km:
                nearby.append({
                    **factory,
                    'distance_km': distance,
                    'likely_upwind': False,
                    'confidence': 0.0,
                    'bearing_to_hotspot': 0.0,
                    'angle_from_wind': None,
                    'scores': {},
                    'composite_score': 0.0,
                    'confidence_breakdown': {}
                })

        # Sort by distance
        nearby.sort(key=lambda f: f['distance_km'])

        logger.info(f"Found {len(nearby)} factories within {max_distance_km}km of hotspot")
        return nearby
    
    def calculate_wind_vector_to_factories(self, hotspot: Dict,
                                          factories: List[Dict],
                                          wind_data: Dict) -> List[Dict]:
        """
        Multi-factor scoring for factory attribution.

        Weights: Wind alignment (40%), Distance (30%), Emission match (20%),
        Wind data quality (10% adjustment).

        Returns factories ranked by confidence score.
        """
        wind_direction = wind_data.get('direction_deg')
        wind_success = wind_data.get('success', False)
        wind_confidence = wind_data.get('confidence', 0)  # Default to 0 if missing
        detected_gas = hotspot.get('gas', '')

        for factory in factories:
            # Calculate bearing from factory to hotspot
            bearing = self._calculate_bearing(
                factory['location'][0], factory['location'][1],
                hotspot['lat'], hotspot['lon']
            )

            factory['bearing_to_hotspot'] = bearing
            scores = {}

            if wind_success and wind_direction is not None:
                # Check if factory is upwind (wind blows from factory to hotspot)
                reverse_bearing = (bearing + 180) % 360
                angle_diff = abs((reverse_bearing - wind_direction + 180) % 360 - 180)

                if angle_diff <= 15:
                    scores['wind'] = 100.0 - (angle_diff * 2.0)
                elif angle_diff <= 30:
                    scores['wind'] = 70.0 - ((angle_diff - 15) * 2.0)
                elif angle_diff <= 60:
                    scores['wind'] = 40.0 - ((angle_diff - 30) * 1.0)
                elif angle_diff <= 90:
                    scores['wind'] = max(0.0, 10.0 - ((angle_diff - 60) * 0.33))
                else:
                    scores['wind'] = 0.0

                factory['angle_from_wind'] = angle_diff
                factory['likely_upwind'] = bool(angle_diff < 30)

                logger.info(
                    f"Factory: {factory['name'][:30]:<30} | "
                    f"Factory→Hotspot: {bearing:>3.0f}° | "
                    f"Hotspot→Factory: {reverse_bearing:>3.0f}° | "
                    f"Wind from: {wind_direction:>3.0f}° | "
                    f"Deviation: {angle_diff:>3.0f}° | "
                    f"Upwind: {'YES' if angle_diff < 30 else 'NO':<3} | "
                    f"Wind score: {scores['wind']:>3.0f}"
                )
            else:
                scores['wind'] = 0.0
                factory['angle_from_wind'] = None
                factory['likely_upwind'] = False

            # Distance score (exponential decay)
            distance_km = factory.get('distance_km', 100)
            scores['distance'] = 100.0 * np.exp(-distance_km / 5.0)

            # Emission match score
            factory_emissions = factory.get('emissions', [])
            if detected_gas in factory_emissions:
                scores['emission'] = 100.0
                logger.debug(f"✓ Emission match: {factory['name']} produces {detected_gas}")
            else:
                scores['emission'] = 0.0
                logger.debug(f"✗ No emission match: {factory['name']} doesn't produce {detected_gas}")

            # Composite score (weighted average)
            weights = {'wind': 0.40, 'distance': 0.30, 'emission': 0.20}
            composite_score = sum(scores[k] * weights[k] for k in ['wind', 'distance', 'emission'])

            if scores['emission'] == 0:
                composite_score *= 0.1

            # Wind data quality adjustment
            if wind_confidence >= 70:
                confidence_multiplier = 1.0
            elif wind_confidence >= 40:
                confidence_multiplier = 0.8
            else:
                confidence_multiplier = 0.5

            final_confidence = composite_score * confidence_multiplier

            factory['scores'] = scores
            factory['composite_score'] = composite_score
            factory['confidence'] = float(max(0.0, min(100.0, final_confidence)))
            factory['confidence_breakdown'] = {
                'wind_score': scores.get('wind', 0),
                'distance_score': scores.get('distance', 0),
                'emission_score': scores.get('emission', 0),
                'wind_confidence_multiplier': confidence_multiplier,
                'final_confidence': final_confidence
            }

        # === FALLBACK LOGIC: If NO factories are upwind, use distance-based ranking ===
        # Check if ANY factory is upwind
        upwind_list = [f['name'] for f in factories if f.get('likely_upwind', False)]
        has_upwind_factory = len(upwind_list) > 0

        logger.info(f"=== UPWIND CHECK: {len(upwind_list)} upwind factories out of {len(factories)} total ===")
        if upwind_list:
            logger.info(f"Upwind factories: {', '.join(upwind_list)}")
        else:
            logger.info("NO upwind factories detected")

        if not has_upwind_factory:
            logger.info("⚠️ NO UPWIND FACTORIES - Switching to distance-based ranking")

            # Recalculate confidence using distance + emission match only
            for factory in factories:
                emission_score = factory['scores'].get('emission', 0)
                distance_score = factory['scores'].get('distance', 0)

                if emission_score > 0:
                    # Prioritize emission match + distance
                    # Emission match factories: Use distance as primary factor
                    factory['confidence'] = distance_score * 0.9  # Up to 90% confidence
                    factory['fallback_ranking'] = 'emission_match'
                else:
                    # No emission match: Very low confidence
                    factory['confidence'] = distance_score * 0.1  # Max 10% confidence
                    factory['fallback_ranking'] = 'no_emission_match'

                logger.info(
                    f"Fallback ranking: {factory['name'][:30]:<30} | "
                    f"Distance: {factory.get('distance_km', 0):>5.1f} km | "
                    f"Emits {detected_gas}: {'YES' if emission_score > 0 else 'NO':<3} | "
                    f"Confidence: {factory['confidence']:>3.0f}%"
                )

        # Sort by final confidence (highest first)
        return sorted(factories, key=lambda f: f['confidence'], reverse=True)
    
    def ai_analysis(self, violation_data: Dict, map_image_path: Optional[str] = None) -> str:
        """
        Use Google Gemini to analyze pollution violation and determine likely source.

        Uses hybrid approach:
        - Simple cases (1 obvious upwind factory): Fast rule-based analysis
        - Complex cases (multiple upwind, low confidence, unusual patterns): Full AI reasoning

        Args:
            violation_data: Complete violation data including gas, factories, wind
            map_image_path: Optional path to map visualization image for vision analysis

        Returns:
            AI-generated analysis text
        """
        factories = violation_data.get('nearby_factories', [])

        if not self.model:
            return self._rule_based_analysis(violation_data)

        upwind_factories = [f for f in factories if f.get('likely_upwind', False) and
                           violation_data['gas'] in f.get('emissions', [])]

        # Simple case: Use fast rule-based analysis
        if len(upwind_factories) == 1 and upwind_factories[0].get('confidence', 0) > 70:
            logger.info("Simple case detected: Using rule-based analysis")
            return self._rule_based_analysis(violation_data)

        # Complex case: Let Gemini do real analysis
        if not upwind_factories or all(f.get('confidence', 0) < 40 for f in upwind_factories):
            logger.info("No clear source detected: Using AI for uncertainty analysis")
        elif len(upwind_factories) >= 3:
            logger.info("Cluster pollution detected: Using AI for multi-source analysis")
        else:
            logger.info("Complex case: Using AI for detailed attribution")

        try:
            current_lang = get_current_language()
            lang_instruction = ("Respond in formal Arabic (العربية) suitable for environmental reports."
                              if current_lang == 'ar' else "Respond in English.")

            # Build factory data as raw JSON for Gemini to analyze
            factory_data = []
            for factory in violation_data.get('nearby_factories', [])[:8]:
                factory_data.append({
                    'name': factory['name'],
                    'type': factory['type'],
                    'distance_km': round(factory['distance_km'], 1),
                    'bearing_deg': round(factory.get('bearing_to_hotspot', 0)),
                    'emissions': factory['emissions'],
                    'produces_detected_gas': violation_data['gas'] in factory['emissions']
                })

            # Construct analysis prompt focusing on reasoning
            prompt = f"""You are an expert environmental scientist analyzing industrial air pollution.

**POLLUTION EVENT:**
- Detected: {violation_data['gas_name']} ({violation_data['gas']})
- Concentration: {violation_data['max_value']:.2f} {violation_data['unit']} (threshold: {violation_data['threshold']:.2f})
- Exceedance: {violation_data['percentage_over']:.1f}% above threshold
- Severity: {violation_data['severity']}
- Location: {violation_data['city']} ({violation_data['hotspot']['lat']:.4f}, {violation_data['hotspot']['lon']:.4f})
- Time: {violation_data['timestamp_ksa']}

**METEOROLOGICAL CONDITIONS:**
- Wind from: {violation_data['wind']['direction_deg']:.0f}° ({violation_data['wind']['direction_cardinal']})
- Wind speed: {violation_data['wind']['speed_ms']:.1f} m/s
- Data quality: {violation_data['wind'].get('confidence', 0):.0f}% ({violation_data['wind'].get('source_label', 'unknown')})
- Wind measured: {violation_data['wind'].get('timestamp_ksa', 'N/A')}
- Time offset from satellite: {violation_data['wind'].get('time_offset_hours', 'N/A'):.1f}h

**INDUSTRIAL FACILITIES:**
{json.dumps(factory_data, indent=2)}

**YOUR ANALYSIS TASKS:**

1. **Source Attribution:**
   Using atmospheric dispersion principles, determine which facility (if any) is most likely responsible.
   Consider:
   - Wind direction (pollutant transport from source to hotspot)
   - Distance and plume dispersion
   - Emission profile matching
   - Wind speed effects on concentration

2. **Reasoning:**
   Explain your analytical process. Why did you select (or exclude) each facility?
   What atmospheric factors influenced your decision?

3. **Uncertainty Assessment:**
   - What's your confidence level (High/Medium/Low)? Why?
   - What data limitations affect your analysis?
   - Is wind data quality adequate for attribution?

4. **Data Quality Concerns:**
   - Are there red flags in wind timing, confidence, or measurement quality?
   - Could temporal offset between wind and satellite data affect attribution?

5. **Recommendation:**
   - Single source: Recommend inspection of specific facility
   - Multiple contributors (3+ upwind): Recommend cluster investigation
   - Unclear: Recommend expanding monitoring or investigating alternative sources

**OUTPUT FORMAT:**
MOST LIKELY SOURCE: [Name or "No clear source"]
CONFIDENCE: [High/Medium/Low] ([%])
REASONING: [Your atmospheric analysis]
CONCERNS: [Data quality issues or limitations]
RECOMMENDATION: [Specific action]

{lang_instruction}
Keep analysis under 300 words, professional tone."""

            # Prepare content for AI (text + optional image)
            content_parts = []
            image_included = False

            # Add map image if provided (for vision analysis)
            if map_image_path and os.path.exists(map_image_path):
                try:
                    logger.info(f"🖼️ VISION MODE ENABLED: Loading map image from {map_image_path}")
                    if self.use_vertex:
                        # Vertex AI format
                        image_part = Image.load_from_file(map_image_path)
                        content_parts.append(image_part)
                        logger.info("Map image loaded successfully (Vertex AI format)")
                    else:
                        # Standard Gemini API format
                        from PIL import Image as PILImage
                        img = PILImage.open(map_image_path)
                        content_parts.append(img)
                        logger.info("Map image loaded successfully (Standard API format)")

                    image_included = True

                    # Add vision-specific instructions
                    prompt += """

**VISUAL ANALYSIS (Map Image Provided):**
The map shows:
- Pollution heatmap (green=low, yellow/orange=moderate, red/purple=high)
- Hotspot marker (pollution maximum)
- Blue arrow (wind direction)
- Factory markers (red=high priority, blue=other)

**ADDITIONAL VISION TASKS:**
1. Plume morphology: Does the pollution pattern suggest a point source or diffuse/area source?
2. Directional gradient: Does pollution intensity decrease in a direction that points to a specific facility?
3. Visual wind verification: Does the plume shape align with stated wind direction?
4. Spatial anomalies: Any unexpected patterns (e.g., pollution hotspot far from all facilities)?
5. Confidence verification: Does visual evidence support or contradict numerical attribution?

Integrate visual insights into your reasoning section."""

                    logger.info(f"✅ Map image added to {'Vertex AI' if self.use_vertex else 'Gemini'} vision analysis - Gemini 3 will perform spatial analysis")
                except Exception as img_err:
                    logger.error(f"❌ VISION MODE FAILED: Could not load map image for vision analysis: {img_err}")
                    logger.info("Falling back to TEXT-ONLY mode")
            elif map_image_path:
                logger.warning(f"⚠️ Map image path provided but file does not exist: {map_image_path}")
                logger.info("Using TEXT-ONLY mode")
            else:
                logger.info("📝 TEXT-ONLY MODE: No map image provided, using numerical data only")

            # Add text prompt (Vertex AI expects prompt first, then image)
            if self.use_vertex:
                content_parts.insert(0, prompt)
            else:
                content_parts.append(prompt)

            logger.info(f"Sending to Gemini 3 Pro ({'Vertex AI' if self.use_vertex else 'API'}) - Vision: {image_included}")

            config_params = {
                'temperature': 0.5,
                'max_output_tokens': 900,
                'top_p': 0.95
            }
            if image_included:
                config_params['media_resolution'] = 'high'

            response = self.model.generate_content(
                content_parts,
                generation_config=config_params
            )

            analysis = response.text

            mode = "vision + reasoning" if image_included else "reasoning"
            logger.info(f"✅ Gemini analysis complete ({mode})")

            return analysis
            
        except Exception as e:
            logger.error(f"Gemini AI analysis failed: {e}")
            return self._rule_based_analysis(violation_data)
    
    def _rule_based_analysis(self, violation_data: Dict) -> str:
        """Fallback rule-based analysis when AI is unavailable"""
        factories = violation_data.get('nearby_factories', [])
        lang = get_current_language()
        is_ar = lang == 'ar'

        if not factories:
            if is_ar:
                return "لم يتم العثور على مصانع بالقرب من بؤرة التلوث. قد يكون المصدر خارج المنطقة المراقبة أو مصدر متنقل."
            return "No factories found near pollution hotspot. Source may be outside monitored area or mobile source."

        gas = violation_data['gas']
        gas_name = violation_data.get('gas_name', gas)
        wind_dir = violation_data.get('wind', {}).get('direction_cardinal', 'Unknown')
        wind_deg = violation_data.get('wind', {}).get('direction_deg', 0)

        upwind_emitters = [f for f in factories if gas in f['emissions'] and f.get('likely_upwind', False)]
        all_emitters = [f for f in factories if gas in f['emissions']]
        non_emitters = [f for f in factories[:5] if gas not in f['emissions']]

        candidates = upwind_emitters if upwind_emitters else all_emitters

        wind_confidence = violation_data.get('wind', {}).get('confidence', 0)
        wind_source = violation_data.get('wind', {}).get('source_label', 'unknown')

        if candidates:
            top = candidates[0]
            top_confidence = top.get('confidence', 0)
            top_is_upwind = top.get('likely_upwind', False)
            MIN_CONFIDENCE_THRESHOLD = 40.0

            if not top_is_upwind and top_confidence < MIN_CONFIDENCE_THRESHOLD:
                if is_ar:
                    analysis = f"⚠️ لم يتم تحديد مصدر واضح\n"
                    analysis += f"{'='*50}\n\n"
                    analysis += f"السبب: لا توجد مصانع متوافقة مع اتجاه الرياح.\n\n"
                    analysis += f"ظروف الرياح:\n"
                    analysis += f"  - اتجاه الرياح: {wind_deg:.0f}° ({wind_dir})\n"
                    analysis += f"  - سرعة الرياح: {violation_data.get('wind', {}).get('speed_ms', 0):.1f} م/ث\n"
                    analysis += f"  - جودة بيانات الرياح: {wind_confidence:.0f}% ({wind_source})\n"
                    wind_time = violation_data.get('wind', {}).get('timestamp_ksa', 'N/A')
                    sat_time = violation_data.get('timestamp_ksa', 'N/A')
                    time_offset = violation_data.get('wind', {}).get('time_offset_hours', 'N/A')
                    analysis += f"  - وقت قياس الرياح: {wind_time}\n"
                    analysis += f"  - وقت رصد القمر الصناعي: {sat_time}\n"
                    analysis += f"  - الفارق الزمني: {time_offset:.1f} ساعات\n\n" if isinstance(time_offset, (int, float)) else f"  - الفارق الزمني: {time_offset}\n\n"

                    analysis += f"تحليل المصانع القريبة:\n"
                    analysis += f"  - {len(candidates)} مصانع تنتج {gas_name}\n"
                    analysis += f"  - أقرب مُنتِج: {top['name']} ({top['distance_km']:.1f} كم)\n"
                    analysis += f"  - ومع ذلك، هذا المصنع ليس في اتجاه الرياح (الثقة: {top_confidence:.0f}%)\n"
                    analysis += f"  - الرياح لا تهب من أي مصنع معروف ينتج {gas_name} نحو البؤرة\n\n"

                    analysis += f"التفسيرات المحتملة:\n"
                    analysis += f"  - مصدر خارج المنطقة المراقبة (مثلاً: بعد 20 كم)\n"
                    analysis += f"  - مصادر متنقلة (سفن، مركبات، نقل صناعي)\n"
                    analysis += f"  - عدم يقين اتجاه الرياح (ثقة الرياح: {wind_confidence:.0f}%)\n"
                    analysis += f"  - نقل التلوث بعيد المدى من مصادر بعيدة\n"
                    analysis += f"  - قاعدة بيانات انبعاثات المصانع غير مكتملة\n\n"

                    analysis += f"التوصية: توسيع نطاق المراقبة أو التحقيق في المصادر المتنقلة/البعيدة المحتملة.\n\n"

                    analysis += f"مرجع - المُنتِجون القريبون (ليسوا في اتجاه الرياح):\n"
                    for i, factory in enumerate(candidates[:5], 1):
                        analysis += f"  {i}. {factory['name']} - {factory['distance_km']:.1f} كم، ثقة {factory['confidence']:.0f}%\n"
                else:
                    analysis = f"⚠️ NO CLEAR SOURCE IDENTIFIED\n"
                    analysis += f"{'='*50}\n\n"
                    analysis += f"REASON: No factories are aligned with the wind direction.\n\n"
                    analysis += f"WIND CONDITIONS:\n"
                    analysis += f"  - Wind Direction: {wind_deg:.0f}° ({wind_dir})\n"
                    analysis += f"  - Wind Speed: {violation_data.get('wind', {}).get('speed_ms', 0):.1f} m/s\n"
                    analysis += f"  - Wind Data Quality: {wind_confidence:.0f}% ({wind_source})\n"
                    wind_time = violation_data.get('wind', {}).get('timestamp_ksa', 'N/A')
                    sat_time = violation_data.get('timestamp_ksa', 'N/A')
                    time_offset = violation_data.get('wind', {}).get('time_offset_hours', 'N/A')
                    analysis += f"  - Wind measurement time: {wind_time}\n"
                    analysis += f"  - Satellite observation time: {sat_time}\n"
                    analysis += f"  - Time offset: {time_offset:.1f} hours\n\n" if isinstance(time_offset, (int, float)) else f"  - Time offset: {time_offset}\n\n"

                    analysis += f"NEARBY FACTORIES ANALYSIS:\n"
                    analysis += f"  - {len(candidates)} factories produce {gas_name}\n"
                    analysis += f"  - Closest emitter: {top['name']} ({top['distance_km']:.1f} km)\n"
                    analysis += f"  - However, this factory is NOT upwind (confidence: {top_confidence:.0f}%)\n"
                    analysis += f"  - Wind does not blow from any known {gas_name}-producing factory toward the hotspot\n\n"

                    analysis += f"POSSIBLE EXPLANATIONS:\n"
                    analysis += f"  - Source outside monitored area (e.g., beyond 20km radius)\n"
                    analysis += f"  - Mobile sources (ships, vehicles, industrial transport)\n"
                    analysis += f"  - Wind direction uncertainty (wind confidence: {wind_confidence:.0f}%)\n"
                    analysis += f"  - Long-range pollution transport from distant sources\n"
                    analysis += f"  - Incomplete factory emissions database\n\n"

                    analysis += f"RECOMMENDATION: Expand monitoring radius or investigate potential mobile/distant sources.\n\n"

                    # List all nearby emitters for reference
                    analysis += f"REFERENCE - NEARBY EMITTERS (not upwind):\n"
                    for i, factory in enumerate(candidates[:5], 1):
                        analysis += f"  {i}. {factory['name']} - {factory['distance_km']:.1f} km, {factory['confidence']:.0f}% confidence\n"

                return analysis

            # Build justified analysis (only if confidence is sufficient)
            if is_ar:
                analysis = f"المصدر الأكثر احتمالاً: {top['name']} ({top['type']})\n"
                analysis += f"{'='*50}\n\n"

                analysis += "التبرير:\n"

                # 1. Emission matching
                analysis += f"✓ تطابق الانبعاثات: المصنع ينتج {gas_name} (الغاز المكتشف: {gas_name})\n"

                # 2. Wind direction
                analysis += f"\nظروف الرياح:\n"
                analysis += f"  - اتجاه الرياح: {wind_deg:.0f}° ({wind_dir})\n"
                analysis += f"  - سرعة الرياح: {violation_data.get('wind', {}).get('speed_ms', 0):.1f} م/ث\n"
                analysis += f"  - جودة بيانات الرياح: {wind_confidence:.0f}% ({wind_source})\n"
                wind_time = violation_data.get('wind', {}).get('timestamp_ksa', 'N/A')
                sat_time = violation_data.get('timestamp_ksa', 'N/A')
                time_offset = violation_data.get('wind', {}).get('time_offset_hours', 'N/A')
                analysis += f"  - وقت قياس الرياح: {wind_time}\n"
                analysis += f"  - وقت رصد القمر الصناعي: {sat_time}\n"
                analysis += f"  - الفارق الزمني: {time_offset:.1f} ساعات\n\n" if isinstance(time_offset, (int, float)) else f"  - الفارق الزمني: {time_offset}\n\n"

                if top.get('likely_upwind', False):
                    analysis += f"✓ توافق الرياح: المصنع في اتجاه الرياح من البؤرة\n"
                    analysis += f"  - الرياح تهب من المصنع إلى بؤرة التلوث\n"
                    analysis += f"  - الاتجاه من المصنع: {top.get('bearing_to_hotspot', 0):.0f}°\n"
                    analysis += f"  - ثقة التوافق: {top['confidence']:.0f}%\n"
                else:
                    analysis += f"⚠️ توافق الرياح: المصنع ليس في اتجاه الرياح المثالي\n"
                    analysis += f"  - الاتجاه من المصنع: {top.get('bearing_to_hotspot', 0):.0f}°\n"
                    analysis += f"  - عدم تطابق اتجاه الرياح (تم اختياره لتطابق الانبعاثات + القرب)\n"
                    analysis += f"  - ثقة التوافق: {top['confidence']:.0f}%\n"

                # 3. Distance
                analysis += f"✓ المسافة: {top['distance_km']:.1f} كم من البؤرة\n\n"

                # Exclusion reasoning
                if non_emitters:
                    analysis += f"المصانع المستبعدة:\n"
                    for f in non_emitters[:2]:
                        analysis += f"✗ {f['name']}: لا ينتج {gas_name}\n"

                # Similar distance comparison
                similar_distance_emitters = [
                    f for f in all_emitters
                    if f['name'] != top['name']
                    and abs(f['distance_km'] - top['distance_km']) < 2.0
                ]

                if similar_distance_emitters:
                    analysis += f"\nمقارنة مع مصانع على مسافة مماثلة:\n"
                    for f in similar_distance_emitters[:2]:
                        analysis += f"✗ {f['name']} ({f['distance_km']:.1f} كم):\n"
                        analysis += f"  - توافق الرياح: {f['confidence']:.0f}% مقابل {top['confidence']:.0f}%\n"
                        analysis += f"  - الاتجاه: {f.get('bearing_to_hotspot', 0):.0f}° مقابل {top.get('bearing_to_hotspot', 0):.0f}°\n"
                        if f['confidence'] < top['confidence']:
                            analysis += f"  - تم اختيار الأساسي بسبب توافق أفضل مع الرياح\n"
                        else:
                            analysis += f"  - مُدرج كمصدر بديل\n"

                other_downwind = [f for f in all_emitters
                                if not f.get('likely_upwind', False)
                                and f['name'] != top['name']
                                and f not in similar_distance_emitters]
                if other_downwind:
                    analysis += f"✗ {len(other_downwind)} مُنتِج(ون) آخر(ون): أبعد أو توافق ضعيف مع الرياح\n"

                analysis += "\n"

                # Overall confidence
                if wind_confidence < 50:
                    analysis += f"⚠️ الثقة الإجمالية: متوسطة-منخفضة (عدم يقين بيانات الرياح)\n\n"
                elif top.get('likely_upwind', False) and top['distance_km'] < 5:
                    analysis += f"✓ الثقة الإجمالية: عالية (في اتجاه الرياح + قريب + تطابق الانبعاثات)\n\n"
                else:
                    analysis += f"✓ الثقة الإجمالية: متوسطة (تم تأكيد تطابق الانبعاثات)\n\n"

                analysis += f"التوصية: فحص فوري لضوابط انبعاثات {top['name']} والتغييرات التشغيلية الأخيرة."

                if len(candidates) > 1:
                    analysis += f"\n\nالمصادر البديلة: {', '.join([f['name'] for f in candidates[1:3]])}"
            else:
                analysis = f"MOST LIKELY SOURCE: {top['name']} ({top['type']})\n"
                analysis += f"{'='*50}\n\n"

                analysis += "JUSTIFICATION:\n"

                # 1. Emission matching
                analysis += f"✓ Emission Match: Factory produces {gas_name} (detected gas: {gas_name})\n"

                # 2. Wind direction (ALWAYS show wind details)
                analysis += f"\nWIND CONDITIONS:\n"
                analysis += f"  - Wind Direction: {wind_deg:.0f}° ({wind_dir})\n"
                analysis += f"  - Wind Speed: {violation_data.get('wind', {}).get('speed_ms', 0):.1f} m/s\n"
                analysis += f"  - Wind Data Quality: {wind_confidence:.0f}% ({wind_source})\n"
                wind_time = violation_data.get('wind', {}).get('timestamp_ksa', 'N/A')
                sat_time = violation_data.get('timestamp_ksa', 'N/A')
                time_offset = violation_data.get('wind', {}).get('time_offset_hours', 'N/A')
                analysis += f"  - Wind measurement time: {wind_time}\n"
                analysis += f"  - Satellite observation time: {sat_time}\n"
                analysis += f"  - Time offset: {time_offset:.1f} hours\n\n" if isinstance(time_offset, (int, float)) else f"  - Time offset: {time_offset}\n\n"

                if top.get('likely_upwind', False):
                    analysis += f"✓ Wind Alignment: Factory IS upwind of hotspot\n"
                    analysis += f"  - Wind blows FROM factory TO pollution hotspot\n"
                    analysis += f"  - Bearing from factory: {top.get('bearing_to_hotspot', 0):.0f}°\n"
                    analysis += f"  - Alignment confidence: {top['confidence']:.0f}%\n"
                else:
                    analysis += f"⚠️ Wind Alignment: Factory NOT ideally upwind\n"
                    analysis += f"  - Bearing from factory: {top.get('bearing_to_hotspot', 0):.0f}°\n"
                    analysis += f"  - Wind bearing mismatch (selected for emission match + proximity)\n"
                    analysis += f"  - Alignment confidence: {top['confidence']:.0f}%\n"

                # 3. Distance
                analysis += f"✓ Distance: {top['distance_km']:.1f} km from hotspot\n\n"

                # Exclusion reasoning with details
                if non_emitters:
                    analysis += f"EXCLUDED FACTORIES:\n"
                    for f in non_emitters[:2]:
                        analysis += f"✗ {f['name']}: Does not produce {gas_name}\n"

                # Show comparison with other emitters at similar distances
                similar_distance_emitters = [
                    f for f in all_emitters
                    if f['name'] != top['name']
                    and abs(f['distance_km'] - top['distance_km']) < 2.0  # Within 2km distance
                ]

                if similar_distance_emitters:
                    analysis += f"\nCOMPARISON WITH SIMILAR-DISTANCE FACTORIES:\n"
                    for f in similar_distance_emitters[:2]:
                        analysis += f"✗ {f['name']} ({f['distance_km']:.1f} km):\n"
                        analysis += f"  - Wind alignment: {f['confidence']:.0f}% vs {top['confidence']:.0f}%\n"
                        analysis += f"  - Bearing: {f.get('bearing_to_hotspot', 0):.0f}° vs {top.get('bearing_to_hotspot', 0):.0f}°\n"
                        if f['confidence'] < top['confidence']:
                            analysis += f"  - Selected primary due to better wind alignment\n"
                        else:
                            analysis += f"  - Listed as alternative source\n"

                other_downwind = [f for f in all_emitters
                                if not f.get('likely_upwind', False)
                                and f['name'] != top['name']
                                and f not in similar_distance_emitters]
                if other_downwind:
                    analysis += f"✗ {len(other_downwind)} other emitter(s): Farther or poor wind alignment\n"

                analysis += "\n"

                # Overall confidence
                if wind_confidence < 50:
                    analysis += f"⚠️ OVERALL CONFIDENCE: MEDIUM-LOW (wind data uncertainty)\n\n"
                elif top.get('likely_upwind', False) and top['distance_km'] < 5:
                    analysis += f"✓ OVERALL CONFIDENCE: HIGH (upwind + close + emission match)\n\n"
                else:
                    analysis += f"✓ OVERALL CONFIDENCE: MEDIUM (emission match confirmed)\n\n"

                analysis += f"RECOMMENDATION: Immediate inspection of {top['name']} emission controls and recent operational changes."

                if len(candidates) > 1:
                    analysis += f"\n\nALTERNATIVE SOURCES: {', '.join([f['name'] for f in candidates[1:3]])}"
        else:
            # No factories produce this gas
            if is_ar:
                analysis = f"لم يتم تحديد مصدر واضح\n"
                analysis += f"{'='*50}\n\n"
                analysis += f"السبب: لا يُعرف أن أياً من المصانع القريبة ({len(factories)} مصنع) ينتج {gas_name}.\n\n"
                analysis += f"الغاز المكتشف: {gas_name}\n"
                analysis += f"المصانع القريبة: {', '.join([f['name'] for f in factories[:3]])}\n"
                analysis += f"انبعاثاتها: {', '.join([', '.join(f['emissions']) for f in factories[:3]])}\n\n"
                analysis += f"التفسيرات المحتملة:\n"
                analysis += f"- مصدر خارج المنطقة المراقبة\n"
                analysis += f"- مصادر متنقلة (مركبات، سفن)\n"
                analysis += f"- قاعدة بيانات انبعاثات المصانع غير مكتملة\n"
                analysis += f"- نقل بعيد المدى (الرياح: {wind_dir}، الثقة: {wind_confidence:.0f}%)"
            else:
                analysis = f"NO CLEAR SOURCE IDENTIFIED\n"
                analysis += f"{'='*50}\n\n"
                analysis += f"REASON: None of the {len(factories)} nearby factories are known to produce {gas_name}.\n\n"
                analysis += f"DETECTED GAS: {gas_name}\n"
                analysis += f"NEARBY FACTORIES: {', '.join([f['name'] for f in factories[:3]])}\n"
                analysis += f"THEIR EMISSIONS: {', '.join([', '.join(f['emissions']) for f in factories[:3]])}\n\n"
                analysis += f"POSSIBLE EXPLANATIONS:\n"
                analysis += f"- Source outside monitored area\n"
                analysis += f"- Mobile sources (vehicles, ships)\n"
                analysis += f"- Incomplete factory emission database\n"
                analysis += f"- Long-range transport (wind: {wind_dir}, confidence: {wind_confidence:.0f}%)"

        return analysis
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two points on Earth (km)"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 (degrees)"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _get_direction_relative_to_hotspot(bearing: float) -> str:
        """Convert bearing to cardinal direction relative to hotspot"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = int((bearing + 11.25) / 22.5) % 16
        return directions[index]
