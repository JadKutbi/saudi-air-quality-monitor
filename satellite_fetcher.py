"""
Satellite Data Fetcher - Retrieves real Sentinel-5P TROPOMI data
Uses Google Earth Engine API for near-real-time atmospheric measurements
"""

import ee
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, Tuple, Optional
import config
import requests
import os
from enhanced_wind_fetcher import EnhancedWindFetcher

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class SatelliteDataFetcher:
    """Fetch real satellite data from Sentinel-5P TROPOMI sensor"""
    
    def __init__(self):
        """Initialize Google Earth Engine and Enhanced Wind Fetcher"""
        self.ee_initialized = False

        try:
            # Try to get service account from Streamlit secrets first
            import streamlit as st
            if hasattr(st, 'secrets') and 'GEE_SERVICE_ACCOUNT' in st.secrets:
                # Use service account from secrets
                service_account = st.secrets['GEE_SERVICE_ACCOUNT']
                credentials = ee.ServiceAccountCredentials(
                    service_account,
                    key_data=st.secrets['GEE_PRIVATE_KEY']
                )
                ee.Initialize(credentials, project=config.GEE_PROJECT)
                logger.info(f"Google Earth Engine initialized with service account for project: {config.GEE_PROJECT}")
                self.ee_initialized = True
            else:
                # Fallback to default authentication (for local development)
                ee.Initialize(project=config.GEE_PROJECT)
                logger.info(f"Google Earth Engine initialized with default auth for project: {config.GEE_PROJECT}")
                self.ee_initialized = True

            # Test the connection
            self._test_ee_connection()

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize GEE: {error_msg}")

            # Provide specific error guidance
            if "401" in error_msg or "403" in error_msg:
                logger.error("Authentication failed. Check your service account credentials.")
            elif "Permission" in error_msg:
                logger.error("Permission denied. Ensure the service account has Earth Engine access.")
            elif "project" in error_msg.lower():
                logger.error(f"Project issue. Verify project ID: {config.GEE_PROJECT}")

            logger.info("For Streamlit Cloud: Add GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY to secrets")
            logger.info("For local: Run 'earthengine authenticate'")

            # Don't raise here, let the fetch methods handle it
            self.ee_initialized = False

        # Initialize enhanced wind fetcher with all API sources
        try:
            self.enhanced_wind_fetcher = EnhancedWindFetcher()
            logger.info("Enhanced wind fetcher initialized with multiple sources")
        except Exception as e:
            logger.warning(f"Wind fetcher initialization failed: {e}")
            self.enhanced_wind_fetcher = None

    def _test_ee_connection(self):
        """Test Earth Engine connection"""
        try:
            # Simple test to verify connection
            test = ee.Number(1).getInfo()
            logger.info("Earth Engine connection test successful")
        except Exception as e:
            logger.error(f"Earth Engine connection test failed: {e}")
            raise Exception(f"Cannot connect to Earth Engine: {e}")
    
    def fetch_gas_data(self, city: str, gas: str, days_back: int = 3) -> Dict:
        """
        Fetch gas concentration data for a specific city

        Args:
            city: City name (Yanbu, Jubail, Jazan)
            gas: Gas type (NO2, SO2, CO, O3, HCHO, CH4)
            days_back: Number of days to look back for data

        Returns:
            Dictionary with gas data, timestamp, and statistics
        """
        logger.info(f"Fetching {gas} data for {city}")

        # Check if Earth Engine is initialized
        if not self.ee_initialized:
            logger.error(f"Cannot fetch {gas} data - Earth Engine not initialized")
            return self._create_empty_response(city, gas, error="Earth Engine not initialized. Check authentication.")

        city_config = config.CITIES.get(city)
        gas_config = config.GAS_PRODUCTS.get(gas)

        if not city_config:
            logger.error(f"Unknown city: {city}")
            return self._create_empty_response(city, gas, error=f"Unknown city: {city}")

        if not gas_config:
            logger.error(f"Unknown gas: {gas}")
            return self._create_empty_response(city, gas, error=f"Unknown gas: {gas}")
        
        # Define area of interest
        bbox = city_config["bbox"]
        aoi = ee.Geometry.Rectangle(bbox)
        
        # Date range - Sentinel-5P data available from 2018-07-10
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # Load Sentinel-5P collection
            collection = ee.ImageCollection(gas_config["dataset"]) \
                .filterBounds(aoi) \
                .filterDate(start_date.strftime('%Y-%m-%d'), 
                           end_date.strftime('%Y-%m-%d')) \
                .select(gas_config["band"])
            
            # Get the most recent image
            image = collection.sort('system:time_start', False).first()
            
            # Check if image exists
            info = image.getInfo()
            if info is None:
                logger.warning(f"No recent {gas} data available for {city}")
                return self._create_empty_response(city, gas)
            
            # Get timestamp
            timestamp_ms = info['properties']['system:time_start']
            timestamp_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
            timestamp_ksa = timestamp_utc.astimezone(pytz.timezone(config.TIMEZONE))
            
            # Extract data as array
            band_data = image.select(gas_config["band"])
            
            # Get statistics
            stats = band_data.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.max(),
                    sharedInputs=True
                ).combine(
                    reducer2=ee.Reducer.min(),
                    sharedInputs=True
                ),
                geometry=aoi,
                scale=1000,
                maxPixels=1e9
            ).getInfo()
            
            # Get pixel data with coordinates
            lat_lon = ee.Image.pixelLonLat()
            combined = band_data.addBands(lat_lon)
            
            # Sample the region
            samples = combined.sample(
                region=aoi,
                scale=1000,
                geometries=True
            )
            
            # Convert to list
            sample_list = samples.toList(samples.size()).getInfo()
            
            # Extract pixel data
            pixels = []
            for sample in sample_list:
                props = sample['properties']
                if gas_config["band"] in props and props[gas_config["band"]] is not None:
                    # Get raw value in mol/m²
                    raw_value = props[gas_config["band"]]
                    
                    # Convert mol/m² to molecules/cm² then to display units
                    # Multiply by Avogadro's number divided by 10000 (m² to cm² conversion)
                    molecules_per_cm2 = raw_value * 6.02214e19  # mol/m² to molecules/cm²
                    
                    # Scale to display units
                    if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                        value = molecules_per_cm2 / 1e15  # Display as 10^15 molecules/cm²
                    elif gas == 'CO':
                        value = molecules_per_cm2 / 1e18  # Display as 10^18 molecules/cm²
                    elif gas == 'CH4':
                        value = raw_value  # Already in ppb
                    else:
                        value = molecules_per_cm2 / 1e15
                    
                    pixels.append({
                        'lat': props['latitude'],
                        'lon': props['longitude'],
                        'value': value
                    })

            
            logger.info(f"Retrieved {len(pixels)} valid pixels for {gas} in {city}")
            
            # Convert stats
            mean_val = stats.get(gas_config["band"] + '_mean')
            max_val = stats.get(gas_config["band"] + '_max')
            min_val = stats.get(gas_config["band"] + '_min')
            
            # Apply same conversion to statistics
            # Convert statistics using same formula
            if mean_val is not None:
                molecules_cm2 = mean_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    mean_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    mean_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    mean_val = mean_val
                
            if max_val is not None:
                molecules_cm2 = max_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    max_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    max_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    max_val = max_val
                    
            if min_val is not None:
                molecules_cm2 = min_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    min_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    min_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    min_val = min_val



            # Fetch wind data synchronized with satellite timestamp
            wind_data = self.fetch_wind_data(city, timestamp_utc)

            return {
                'success': True,
                'city': city,
                'gas': gas,
                'timestamp_utc': timestamp_utc,
                'timestamp_ksa': timestamp_ksa,
                'pixels': pixels,
                'statistics': {
                    'mean': mean_val,
                    'max': max_val,
                    'min': min_val,
                    'pixel_count': len(pixels)
                },
                'unit': gas_config['display_unit'],
                'bbox': bbox,
                'wind': wind_data  # Include wind data
            }
            
        except Exception as e:
            logger.error(f"Error fetching {gas} data for {city}: {e}")
            logger.debug(f"Date range: {start_date} to {end_date}")
            logger.debug(f"Dataset: {gas_config['dataset']}")
            logger.debug(f"Bounding box: {bbox}")
            return self._create_empty_response(city, gas, error=f"Sentinel-5P error: {str(e)}")

    
    def fetch_wind_data(self, city: str, target_time: Optional[datetime] = None) -> Dict:
        """
        Fetch wind data using enhanced multi-source approach with time synchronization.
        Automatically uses all configured APIs (Tomorrow.io, OpenWeatherMap, WeatherAPI)
        and picks the source with the closest time match to the satellite measurement.

        Args:
            city: City name
            target_time: Exact timestamp (UTC) of satellite measurement

        Returns:
            Dictionary with wind data, confidence score, and time difference
        """
        if target_time is None:
            target_time = datetime.now(pytz.UTC)
        else:
            target_time = target_time.astimezone(pytz.UTC)

        logger.info(f"Fetching wind data for {city} at {target_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Determine appropriate time tolerance based on data age
        hours_ago = abs((datetime.now(pytz.UTC) - target_time).total_seconds() / 3600)

        if hours_ago <= 3:
            # Recent data - strict tolerance
            max_tolerance = 30  # 30 minutes
        elif hours_ago <= 24:
            # Within a day - moderate tolerance
            max_tolerance = 45  # 45 minutes
        else:
            # Historical data - relaxed tolerance (hourly data sources)
            max_tolerance = 60  # 60 minutes (accept hourly data)

        logger.info(f"Using time tolerance of {max_tolerance} minutes (data is {hours_ago:.1f} hours old)")

        # Use enhanced fetcher with all 3 API sources
        wind_data = self.enhanced_wind_fetcher.fetch_wind_data(
            city=city,
            target_time=target_time,
            max_time_diff_minutes=max_tolerance
        )

        # Log the result
        if wind_data.get('confidence', 0) >= 70:
            logger.info(
                f"✅ High-confidence wind data: {wind_data['speed_ms']:.1f} m/s from "
                f"{wind_data['direction_cardinal']} ({wind_data['direction_deg']}°). "
                f"Time diff: {wind_data['time_difference_minutes']:.1f} min, "
                f"Confidence: {wind_data['confidence']}%, Source: {wind_data['source']}"
            )
        elif wind_data.get('confidence', 0) >= 50:
            logger.warning(
                f"⚠️ Moderate-confidence wind data: {wind_data['speed_ms']:.1f} m/s from "
                f"{wind_data['direction_cardinal']}. Time diff: {wind_data['time_difference_minutes']:.1f} min, "
                f"Confidence: {wind_data['confidence']}%, Source: {wind_data['source']}"
            )
        else:
            logger.error(
                f"❌ Low-confidence wind data: Time diff: {wind_data['time_difference_minutes']:.1f} min, "
                f"Confidence: {wind_data['confidence']}%. Factory attribution unreliable!"
            )

        # Convert to format expected by the rest of the system
        # Calculate wind components for compatibility
        import math
        direction_rad = math.radians(wind_data['direction_deg'])
        u_component = -wind_data['speed_ms'] * math.sin(direction_rad)  # East-West
        v_component = -wind_data['speed_ms'] * math.cos(direction_rad)  # North-South

        # Add timestamp information
        timestamp_ksa = wind_data['observation_time'].astimezone(pytz.timezone(config.TIMEZONE))
        time_offset_hours = (wind_data['observation_time'] - target_time).total_seconds() / 3600

        return {
            'success': wind_data.get('confidence', 0) > 0,
            'u_component': u_component,
            'v_component': v_component,
            'speed_ms': wind_data['speed_ms'],
            'direction_deg': wind_data['direction_deg'],
            'direction_cardinal': wind_data['direction_cardinal'],
            'timestamp_utc': wind_data['observation_time'],
            'timestamp_ksa': timestamp_ksa,
            'time_offset_hours': abs(time_offset_hours),
            'time_difference_minutes': wind_data['time_difference_minutes'],
            'confidence': wind_data['confidence'],
            'source': wind_data['source'],
            'source_label': wind_data.get('confidence_reason', ''),
            'warning': wind_data.get('warning', None)
        }

    def _fetch_wind_from_source(
        self,
        source_config: Dict,
        aoi: ee.Geometry,
        target_time: datetime,
        city: str
    ) -> Optional[Dict]:
        """Fetch wind from a specific configured source."""
        dataset = source_config["dataset"]
        u_band = source_config["u_component"]
        v_band = source_config["v_component"]
        scale = source_config.get("scale", 20000)
        search_windows = sorted(
            set(source_config.get("search_windows_hours", [3, 6, 12]))
        )
        forward_hours = source_config.get("forward_search_hours", 0)
        max_offset_hours = source_config.get(
            "max_time_offset_hours",
            search_windows[-1] if search_windows else 24
        )
        base_reliability = source_config.get("base_reliability", 1.0)
        label = source_config.get("label", dataset)
        sample_radius_km = source_config.get("sample_radius_km", 20)
        sample_geometry = aoi.buffer(sample_radius_km * 1000).bounds()

        target_ms = int(target_time.timestamp() * 1000)

        for window_hours in search_windows:
            start_time = target_time - timedelta(hours=window_hours)
            end_time = target_time + timedelta(
                hours=min(window_hours, forward_hours)
            )

            logger.info(
                "Attempting wind source %s for %s (window -%dh/+%dh from %s)",
                label,
                city,
                window_hours,
                min(window_hours, forward_hours),
                target_time.strftime('%Y-%m-%d %H:%M UTC')
            )

            collection = (
                ee.ImageCollection(dataset)
                .filterBounds(sample_geometry)
                .filterDate(
                    start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    end_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                )
                .select([u_band, v_band])
            )

            count = collection.size().getInfo()
            logger.info(
                "%s for %s: found %d images in window [%s to %s]",
                label,
                city,
                count,
                start_time.strftime('%Y-%m-%d %H:%M'),
                end_time.strftime('%Y-%m-%d %H:%M')
            )
            if count == 0:
                continue

            collection_with_diff = collection.map(
                lambda img: img.set(
                    'time_diff',
                    ee.Number(img.get('system:time_start')).subtract(target_ms).abs()
                )
            )

            closest_image = ee.Image(collection_with_diff.sort('time_diff').first())
            image_info = closest_image.getInfo()
            if not image_info:
                continue

            wind_timestamp_ms = image_info['properties']['system:time_start']
            wind_timestamp_utc = datetime.fromtimestamp(
                wind_timestamp_ms / 1000, tz=pytz.UTC
            )
            time_offset_hours = abs(
                (wind_timestamp_utc - target_time).total_seconds()
            ) / 3600

            if time_offset_hours > max_offset_hours:
                logger.debug(
                    "Discarding %s sample Δt=%.2fh (>%.2fh)",
                    label,
                    time_offset_hours,
                    max_offset_hours
                )
                continue

            sample_data = closest_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=sample_geometry,
                scale=scale,
                bestEffort=True,
                maxPixels=1e9
            ).getInfo()

            if not sample_data:
                continue

            u = sample_data.get(u_band)
            v = sample_data.get(v_band)

            if u is None or v is None:
                continue

            speed = float(np.sqrt(u**2 + v**2))
            direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            confidence, score_breakdown = self._compute_wind_confidence(
                speed=speed,
                time_offset_hours=time_offset_hours,
                base_reliability=base_reliability
            )

            logger.info(
                "Wind data retrieved (%s): %.1f m/s from %.0f° (Δt=%.2fh, confidence=%.0f%%)",
                label,
                speed,
                direction,
                time_offset_hours,
                confidence
            )

            ksa_time = wind_timestamp_utc.astimezone(
                pytz.timezone(config.TIMEZONE)
            )

            return {
                'success': True,
                'u_component': float(u),
                'v_component': float(v),
                'speed_ms': float(speed),
                'direction_deg': float(direction),
                'direction_cardinal': self._deg_to_cardinal(direction),
                'timestamp_utc': wind_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'timestamp_ksa': ksa_time.strftime('%Y-%m-%d %H:%M:%S KSA'),
                'time_offset_hours': float(time_offset_hours),
                'confidence': float(confidence),
                'confidence_breakdown': score_breakdown,
                'source': dataset,
                'source_id': source_config.get("id"),
                'source_label': label,
            }

        logger.debug("All windows exhausted for source %s", label)
        return None

    
    @staticmethod
    def _compute_wind_confidence(
        speed: float,
        time_offset_hours: float,
        base_reliability: float
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate confidence (0-100%) for wind data based on recency, speed, and source reliability."""
        if time_offset_hours <= 1:
            time_score = 100.0
        elif time_offset_hours <= 3:
            time_score = 85.0
        elif time_offset_hours <= 6:
            time_score = 70.0
        elif time_offset_hours <= 12:
            time_score = 50.0
        elif time_offset_hours <= 24:
            time_score = 30.0
        elif time_offset_hours <= 72:
            time_score = 20.0
        else:
            time_score = 10.0

        if speed < 0.5:
            speed_factor = 0.4
        elif speed < 1.5:
            speed_factor = 0.7
        else:
            speed_factor = 1.0

        reliability_factor = max(0.1, min(1.0, base_reliability))
        raw_score = time_score * speed_factor * reliability_factor
        confidence = float(max(5.0, min(100.0, raw_score)))

        breakdown = {
            'time_score': time_score,
            'speed_factor': speed_factor,
            'source_factor': reliability_factor,
            'raw_score': raw_score,
        }

        return confidence, breakdown

    def _fetch_wind_from_openweathermap(
        self,
        city: str,
        lat: float,
        lon: float,
        api_key: str,
        target_time: datetime
    ) -> Optional[Dict]:
        """
        Fetch real-time wind data from OpenWeatherMap API.
        
        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            api_key: OpenWeatherMap API key
            target_time: Target time for wind data
            
        Returns:
            Wind data dictionary or None if fetch fails
        """
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'wind' not in data:
                logger.warning("No wind data in OpenWeatherMap response for %s", city)
                return None
            
            wind_speed_ms = data['wind'].get('speed', 0.0)  # m/s
            wind_deg = data['wind'].get('deg', 0.0)  # degrees (where wind is coming FROM)
            
            # Calculate u and v components (meteorological convention)
            wind_rad = np.radians(wind_deg)
            u = -wind_speed_ms * np.sin(wind_rad)
            v = -wind_speed_ms * np.cos(wind_rad)
            
            # Get timestamp
            dt_unix = data.get('dt', int(datetime.now(pytz.UTC).timestamp()))
            wind_timestamp_utc = datetime.fromtimestamp(dt_unix, tz=pytz.UTC)
            wind_timestamp_ksa = wind_timestamp_utc.astimezone(
                pytz.timezone(config.TIMEZONE)
            )
            
            # Calculate time offset
            time_offset_hours = abs(
                (wind_timestamp_utc - target_time).total_seconds()
            ) / 3600
            
            # Compute confidence (OpenWeatherMap is very reliable for current conditions)
            if time_offset_hours <= 1:
                confidence = 95.0
            elif time_offset_hours <= 3:
                confidence = 85.0
            elif time_offset_hours <= 6:
                confidence = 70.0
            else:
                confidence = 50.0
            
            logger.info(
                "OpenWeatherMap wind for %s: %.1f m/s from %.0f° (Δt=%.2fh, confidence=%.0f%%)",
                city,
                wind_speed_ms,
                wind_deg,
                time_offset_hours,
                confidence
            )
            
            return {
                'success': True,
                'u_component': float(u),
                'v_component': float(v),
                'speed_ms': float(wind_speed_ms),
                'direction_deg': float(wind_deg),
                'direction_cardinal': self._deg_to_cardinal(wind_deg),
                'timestamp_utc': wind_timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'timestamp_ksa': wind_timestamp_ksa.strftime('%Y-%m-%d %H:%M:%S KSA'),
                'time_offset_hours': float(time_offset_hours),
                'confidence': float(confidence),
                'confidence_breakdown': {
                    'time_score': confidence,
                    'speed_factor': 1.0,
                    'source_factor': 0.95,
                    'raw_score': confidence,
                },
                'source': 'OpenWeatherMap API',
                'source_id': 'openweathermap',
                'source_label': 'OpenWeatherMap (real-time)',
            }
            
        except requests.exceptions.RequestException as req_err:
            logger.error("OpenWeatherMap API request failed for %s: %s", city, req_err)
            return None
        except Exception as e:
            logger.error("OpenWeatherMap wind processing failed for %s: %s", city, e)
            return None

    def _create_empty_response(self, city: str, gas: str, error: str = None) -> Dict:
        """Create empty response when no data available"""
        return {
            'success': False,
            'city': city,
            'gas': gas,
            'timestamp_utc': None,
            'timestamp_ksa': None,
            'pixels': [],
            'statistics': {'mean': None, 'max': None, 'min': None, 'pixel_count': 0},
            'unit': config.GAS_PRODUCTS[gas]['display_unit'],
            'error': error or 'No recent satellite data available'
        }
    
    @staticmethod
    def _deg_to_cardinal(degrees: float) -> str:
        """Convert degrees to cardinal direction"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((degrees + 11.25) / 22.5)
        return directions[idx % 16]
