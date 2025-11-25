"""
Map Visualizer Module

Creates interactive Folium maps with WHO threshold-normalized heatmaps
and industrial facility markers for pollution source attribution.

Features:
    - Health-based color scaling (WHO 2021 thresholds)
    - Wind direction arrows synchronized to satellite observation time
    - Factory markers with upwind/downwind status
    - Layer controls for satellite imagery overlay
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
from typing import Dict, List, Optional
import logging
import config
import os
import time

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class MapVisualizer:
    """Create interactive pollution maps with heatmaps and factory locations"""
    
    def create_pollution_map(self, gas_data: Dict, wind_data: Dict,
                           hotspot: Optional[Dict] = None,
                           factories: Optional[List[Dict]] = None,
                           violation: bool = False) -> folium.Map:
        """
        Create comprehensive pollution map
        
        Args:
            gas_data: Satellite gas data
            wind_data: Wind direction/speed data
            hotspot: Most intense pixel location
            factories: List of factories with analysis
            violation: Whether this is a violation event
            
        Returns:
            Folium map object
        """
        city = gas_data['city']
        gas = gas_data['gas']
        city_config = config.CITIES[city]
        
        # Create base map
        m = folium.Map(
            location=city_config['center'],
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add satellite layer option
        folium.TileLayer('Esri WorldImagery', name='Satellite').add_to(m)
        
        # Add heatmap if pixels available
        pixels = gas_data.get('pixels', [])
        if pixels:
            threshold_config = config.GAS_THRESHOLDS.get(gas, {})
            threshold = threshold_config.get('column_threshold')
            critical_threshold = threshold_config.get('critical_threshold')

            if threshold and critical_threshold:
                heat_data_normalized = []
                for p in pixels:
                    if p['value'] < threshold:
                        normalized = 0.5 * (p['value'] / threshold)
                    elif p['value'] < critical_threshold:
                        normalized = 0.5 + 0.3 * ((p['value'] - threshold) / (critical_threshold - threshold))
                    else:
                        normalized = 0.8 + 0.2 * min(1.0, (p['value'] - critical_threshold) / critical_threshold)
                    heat_data_normalized.append([p['lat'], p['lon'], min(1.0, normalized)])
                logger.info(f"WHO threshold normalization applied")
            else:
                values = [p['value'] for p in pixels]
                percentile_95 = np.percentile(values, 95) if len(values) > 1 else max(values)
                max_val = max(percentile_95, max(values) * 0.5)
                heat_data_normalized = [[p['lat'], p['lon'], min(1.0, p['value']/max_val)]
                                       for p in pixels]
                logger.info(f"Percentile-based normalization applied")

            HeatMap(
                heat_data_normalized,
                name=f'{gas} Concentration',
                radius=25,
                blur=30,
                min_opacity=0.3,
                max_zoom=18,
                gradient={
                    0.0: '#00E400',
                    0.2: '#92D050',
                    0.4: '#FFFF00',
                    0.5: '#FFC000',
                    0.6: '#FF7E00',
                    0.7: '#FF4500',
                    0.8: '#FF0000',
                    0.9: '#8F3F97',
                    1.0: '#7E0023'
                }
            ).add_to(m)

            logger.info(f"Heatmap created with {len(pixels)} pixels")
        
        # Add hotspot marker
        if hotspot:
            icon_color = 'red' if violation else 'orange'
            folium.Marker(
                location=[hotspot['lat'], hotspot['lon']],
                popup=folium.Popup(
                    f"<b>Maximum {gas} Concentration</b><br>"
                    f"Value: {hotspot['value']:.2f} {hotspot['unit']}<br>"
                    f"Location: {hotspot['lat']:.4f}, {hotspot['lon']:.4f}",
                    max_width=300
                ),
                tooltip=f"Peak {gas}: {hotspot['value']:.2f}",
                icon=folium.Icon(color=icon_color, icon='exclamation-triangle', prefix='fa')
            ).add_to(m)
            
            # Add wind arrow from hotspot (only for valid wind data with confidence > 0)
            wind_confidence = wind_data.get('confidence', 0)
            if (wind_data.get('success') and 
                wind_data.get('direction_deg') is not None and 
                wind_confidence > 0):
                self._add_wind_arrow(m, hotspot, wind_data)

        
        # Add factory locations
        if factories:
            factory_cluster = MarkerCluster(name='Factories').add_to(m)
            
            for factory in factories:
                # Determine marker color based on likelihood
                if factory.get('likely_upwind'):
                    color = 'red'
                    icon = 'industry'
                    priority = 'HIGH PRIORITY'
                else:
                    color = 'blue'
                    icon = 'building'
                    priority = 'Lower Priority'
                
                popup_html = f"""
                <div style='width: 250px'>
                    <h4>{factory['name']}</h4>
                    <b>Type:</b> {factory['type']}<br>
                    <b>Emissions:</b> {', '.join(factory['emissions'])}<br>
                    <b>Distance from hotspot:</b> {factory.get('distance_km', 'N/A'):.1f} km<br>
                    <b>Upwind likelihood:</b> {priority}<br>
                    <b>Confidence:</b> {factory.get('confidence', 0):.0f}%<br>
                    <b>Source:</b> {factory.get('source', 'Unknown')}
                </div>
                """
                
                folium.Marker(
                    location=factory['location'],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=factory['name'],
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(factory_cluster)
        
        threshold_config = config.GAS_THRESHOLDS.get(gas, {})
        threshold = threshold_config.get('column_threshold', 'N/A')
        critical = threshold_config.get('critical_threshold', 'N/A')
        threshold_str = f"{threshold:.1f}" if isinstance(threshold, (int, float)) else "N/A"
        critical_str = f"{critical:.1f}" if isinstance(critical, (int, float)) else "N/A"

        title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 500px; height: 140px;
                    background-color: white; border:2px solid grey;
                    z-index:9999; font-size:13px; padding: 10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin:0; color: #333;">{city} - {gas_data.get("gas_name", gas)} Satellite Monitor</h4>
        <p style="margin:5px 0; line-height:1.6;">
        <b>📅 Time (KSA):</b> {gas_data.get('timestamp_ksa', 'N/A')}<br>
        <b>📊 Mean:</b> {gas_data['statistics'].get('mean', 0):.2f} {gas_data['unit']} |
        <b>⚠️ Peak:</b> {gas_data['statistics'].get('max', 0):.2f} {gas_data['unit']}<br>
        <b>🏥 WHO Threshold:</b> {threshold_str} {gas_data['unit']} |
        <b>🚨 Critical:</b> {critical_str} {gas_data['unit']}<br>
        <small style="color: #666;">Data: Sentinel-5P TROPOMI | Resolution: ~7km | Source: NASA/ESA</small>
        </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 220px;
                    background-color: white; border:2px solid grey;
                    z-index:9999; font-size:11px; padding: 10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
        <h5 style="margin:0 0 8px 0; color: #333;">Air Quality Scale</h5>
        <div style="line-height:1.8;">
            <div><span style="display:inline-block; width:20px; height:12px; background:#00E400; border:1px solid #ccc;"></span> Good - Safe levels</div>
            <div><span style="display:inline-block; width:20px; height:12px; background:#FFFF00; border:1px solid #ccc;"></span> Moderate</div>
            <div><span style="display:inline-block; width:20px; height:12px; background:#FF7E00; border:1px solid #ccc;"></span> Unhealthy (Threshold)</div>
            <div><span style="display:inline-block; width:20px; height:12px; background:#FF0000; border:1px solid #ccc;"></span> Very Unhealthy</div>
            <div><span style="display:inline-block; width:20px; height:12px; background:#8F3F97; border:1px solid #ccc;"></span> Severe</div>
            <div><span style="display:inline-block; width:20px; height:12px; background:#7E0023; border:1px solid #ccc;"></span> Hazardous (Critical)</div>
        </div>
        <small style="color:#666; margin-top:5px; display:block;">Colors scaled to WHO health guidelines</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        logger.info(f"Map created for {city} - {gas}")
        return m
    
    def _add_wind_arrow(self, m: folium.Map, hotspot: Dict, wind_data: Dict):
        """Add wind direction arrow to map"""
        # Calculate arrow endpoint (5km in wind direction)
        wind_dir = wind_data['direction_deg']
        distance_km = 5
        
        # Convert to radians
        lat1 = np.radians(hotspot['lat'])
        lon1 = np.radians(hotspot['lon'])
        bearing = np.radians(wind_dir)
        
        # Earth radius
        R = 6371
        
        # Calculate endpoint
        lat2 = np.arcsin(np.sin(lat1) * np.cos(distance_km/R) +
                         np.cos(lat1) * np.sin(distance_km/R) * np.cos(bearing))
        lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance_km/R) * np.cos(lat1),
                                 np.cos(distance_km/R) - np.sin(lat1) * np.sin(lat2))
        
        lat2 = np.degrees(lat2)
        lon2 = np.degrees(lon2)
        
        # Add arrow line with wind info
        wind_speed = wind_data.get('speed_ms', 0)
        wind_dir = wind_data.get('direction_cardinal', 'N')
        confidence = wind_data.get('confidence', 0)
        source_label = wind_data.get('source_label', 'Unknown source')
        
        popup_text = f"""
        <b>Wind Data</b><br>
        Direction: {wind_dir} ({wind_data['direction_deg']:.0f}°)<br>
        Speed: {wind_speed:.1f} m/s<br>
        Confidence: {confidence:.0f}%<br>
        Source: {source_label}
        """

        folium.PolyLine(
            locations=[[hotspot['lat'], hotspot['lon']], [lat2, lon2]],
            color='blue',
            weight=4,
            opacity=0.8,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)
        
        # Add arrow head marker
        folium.Marker(
            location=[lat2, lon2],
            icon=folium.Icon(color='blue', icon='arrow-up', prefix='fa'),
            tooltip=f"Wind: {wind_dir} {wind_speed:.1f} m/s"
        ).add_to(m)

    def save_map_as_image(self, html_path: str, output_image_path: str,
                         width: int = 1200, height: int = 800) -> bool:
        """
        Convert HTML map to PNG image using selenium

        Args:
            html_path: Path to saved HTML map
            output_image_path: Path where to save PNG image
            width: Browser window width
            height: Browser window height

        Returns:
            True if successful, False otherwise
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait

            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--window-size={width},{height}')

            # Try to create driver (will use system Chrome/Chromedriver)
            driver = webdriver.Chrome(options=chrome_options)

            try:
                # Load the HTML file
                file_url = f'file:///{os.path.abspath(html_path)}'
                driver.get(file_url)

                # Wait for map to render (Folium uses JavaScript)
                time.sleep(2)

                # Take screenshot
                driver.save_screenshot(output_image_path)
                logger.info(f"Map image saved to {output_image_path}")
                return True

            finally:
                driver.quit()

        except ImportError:
            logger.warning("Selenium not installed. Install with: pip install selenium")
            return False
        except Exception as e:
            logger.warning(f"Could not create map image: {e}")
            logger.info("Vision analysis will be text-only. To enable map vision analysis, install Chrome and selenium.")
            return False
