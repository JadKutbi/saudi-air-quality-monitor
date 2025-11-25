"""
Satellite Data Fetcher Module

Retrieves Sentinel-5P TROPOMI atmospheric data from Google Earth Engine
with automatic day-by-day search for latest available measurements.

Capabilities:
    - Multi-gas monitoring (NO2, SO2, CO, HCHO, CH4)
    - Automatic temporal search (up to 30 days back)
    - Wind data synchronization from ERA5/GFS
    - Spatial statistics extraction
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
    """Fetch Sentinel-5P TROPOMI satellite data via Google Earth Engine."""

    def __init__(self):
        """Initialize Google Earth Engine connection and wind data fetcher."""
        self.ee_initialized = False

        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GEE_SERVICE_ACCOUNT' in st.secrets:
                service_account = st.secrets['GEE_SERVICE_ACCOUNT']
                credentials = ee.ServiceAccountCredentials(
                    service_account,
                    key_data=st.secrets['GEE_PRIVATE_KEY']
                )
                ee.Initialize(credentials, project=config.GEE_PROJECT)
                self.ee_initialized = True
            else:
                ee.Initialize(project=config.GEE_PROJECT)
                self.ee_initialized = True

            self._test_ee_connection()

        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            self.ee_initialized = False

        try:
            self.enhanced_wind_fetcher = EnhancedWindFetcher()
        except Exception as e:
            logger.warning(f"Wind fetcher initialization failed: {e}")
            self.enhanced_wind_fetcher = None

    def _test_ee_connection(self):
        """Verify Earth Engine connection is active."""
        try:
            ee.Number(1).getInfo()
        except Exception as e:
            raise Exception(f"Cannot connect to Earth Engine: {e}")

    def fetch_gas_data(self, city: str, gas: str, days_back: int = 3) -> Dict:
        """
        Retrieve the latest satellite measurements for a specific gas.

        Args:
            city: City name (must be defined in config.CITIES)
            gas: Gas identifier (must be defined in config.GAS_PRODUCTS)
            days_back: Initial days to search (auto-extends if needed)

        Returns:
            Dictionary with success status, statistics, pixels, and wind data
        """
        logger.info(f"Fetching {gas} data for {city}")

        if not self.ee_initialized:
            return self._create_empty_response(city, gas, error="Earth Engine not initialized")

        city_config = config.CITIES.get(city)
        gas_config = config.GAS_PRODUCTS.get(gas)

        if not city_config:
            return self._create_empty_response(city, gas, error=f"Unknown city: {city}")

        if not gas_config:
            return self._create_empty_response(city, gas, error=f"Unknown gas: {gas}")

        bbox = city_config["bbox"]
        aoi = ee.Geometry.Rectangle(bbox)

        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days_back)

        try:
            collection = ee.ImageCollection(gas_config["dataset"]) \
                .filterBounds(aoi) \
                .filterDate(start_date.strftime('%Y-%m-%d'),
                           end_date.strftime('%Y-%m-%d')) \
                .select(gas_config["band"])

            collection_count = collection.size().getInfo()
            logger.info(f"Found {collection_count} images for {gas} in {city}")

            if collection_count == 0:
                extended_start = end_date - timedelta(days=days_back * 3)
                collection = ee.ImageCollection(gas_config["dataset"]) \
                    .filterBounds(aoi) \
                    .filterDate(extended_start.strftime('%Y-%m-%d'),
                               end_date.strftime('%Y-%m-%d')) \
                    .select(gas_config["band"])

                collection_count = collection.size().getInfo()
                if collection_count == 0:
                    return self._create_empty_response(city, gas,
                        error=f"No satellite data in past {days_back * 3} days")

            image = None
            timestamp_utc = None
            timestamp_ksa = None
            max_days_to_search = 30
            days_searched = 0

            sorted_collection = collection.sort('system:time_start', False)
            image_list = sorted_collection.toList(collection_count)

            for image_index in range(collection_count):
                try:
                    test_image = ee.Image(image_list.get(image_index))
                    test_info = test_image.getInfo()

                    if test_info is None:
                        continue

                    timestamp_ms = test_info['properties']['system:time_start']
                    timestamp_utc_temp = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
                    timestamp_ksa_temp = timestamp_utc_temp.astimezone(pytz.timezone(config.TIMEZONE))
                    days_ago = (datetime.now(pytz.UTC) - timestamp_utc_temp).days

                    days_searched += 1
                    logger.info(f"Checking image {image_index + 1}/{collection_count} from {timestamp_ksa_temp.strftime('%Y-%m-%d %H:%M')} ({days_ago}d ago)")

                    if days_ago >= max_days_to_search:
                        break

                    band_data_test = test_image.select(gas_config["band"])
                    test_mean = None

                    for test_scale in [1113, 2000, 5000, 10000]:
                        stats_test = band_data_test.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=aoi,
                            scale=test_scale,
                            maxPixels=1e9,
                            bestEffort=True
                        ).getInfo()

                        test_mean = stats_test.get(gas_config["band"])

                        if test_mean is not None:
                            logger.info(f"Found data at scale {test_scale}m")
                            break

                    if test_mean is not None:
                        image = test_image
                        timestamp_utc = timestamp_utc_temp
                        timestamp_ksa = timestamp_ksa_temp
                        logger.info(f"Valid {gas} data from {timestamp_ksa.strftime('%Y-%m-%d %H:%M KSA')}")
                        break

                except Exception as e:
                    logger.debug(f"Error processing image {image_index}: {e}")
                    continue

            # If no valid data found after searching all days
            if image is None:
                logger.warning(f"No valid {gas} data found for {city} in past {days_searched} days (persistent cloud cover)")
                return self._create_empty_response(city, gas,
                    error=f"No valid measurements in past {days_searched} days (persistent cloud cover)")

            # Extract the pollution measurements from the single latest image
            band_data = image.select(gas_config["band"])

            # Calculate statistics WITHIN this single image across the city area:
            # - mean: average pollution across all pixels in the city
            # - max: hotspot (highest pollution pixel in the city)
            # - min: lowest pollution pixel in the city
            # NOTE: We're NOT averaging multiple images - this is one image's spatial statistics

            # Try multiple scales to handle sparse/cloud-affected data
            # Start with fine scale, progressively coarser if needed
            stats = None
            mean_val = None
            max_val = None
            min_val = None

            # CRITICAL FIX: Use Sentinel-5P native resolution (1113m) first
            # Official resolution from GEE documentation
            for scale in [1113, 2000, 5000, 10000]:
                stats = band_data.reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.max(),
                        sharedInputs=True
                    ).combine(
                        reducer2=ee.Reducer.min(),
                        sharedInputs=True
                    ).combine(
                        reducer2=ee.Reducer.count(),
                        sharedInputs=True
                    ),
                    geometry=aoi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()

                mean_val = stats.get(gas_config["band"] + '_mean')
                max_val = stats.get(gas_config["band"] + '_max')

                if mean_val is not None or max_val is not None:
                    logger.info(f"Got statistics at scale {scale}m")
                    break
            
            # Get pixel data with coordinates
            lat_lon = ee.Image.pixelLonLat()
            combined = band_data.addBands(lat_lon)
            
            # Sample the region at different scales to handle sparse data
            # Start with native Sentinel-5P resolution
            scales = [1113, 2000, 5000, 10000]
            sample_count = 0
            samples = None

            for scale in scales:
                try:
                    samples = combined.sample(
                        region=aoi,
                        scale=scale,
                        geometries=True,
                        numPixels=500
                    )
                    sample_count = samples.size().getInfo()
                    if sample_count > 0:
                        break
                except Exception:
                    continue

            pixels = []
            if sample_count > 0:
                sample_list = samples.toList(sample_count).getInfo()

                # Extract and convert pixel data to display units
                for sample in sample_list:
                    props = sample['properties']
                    if gas_config["band"] in props and props[gas_config["band"]] is not None:
                        raw_value = props[gas_config["band"]]
                        molecules_per_cm2 = raw_value * 6.02214e19

                        # Scale to appropriate display units
                        if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                            value = molecules_per_cm2 / 1e15
                        elif gas == 'CO':
                            value = molecules_per_cm2 / 1e18
                        elif gas == 'CH4':
                            value = raw_value
                        else:
                            value = molecules_per_cm2 / 1e15

                        pixels.append({
                            'lat': props['latitude'],
                            'lon': props['longitude'],
                            'value': value
                        })

            # Get min value and count from stats (already have mean and max from loop above)
            if stats:
                min_val = stats.get(gas_config["band"] + '_min')
                count_val = stats.get(gas_config["band"] + '_count', 0)

            # Final check - this should rarely fail now with multi-scale approach
            if mean_val is None and max_val is None and len(pixels) == 0:
                logger.warning(f"Selected day for {gas} had no usable data even with multi-scale processing")
                return self._create_empty_response(city, gas,
                    error=f"Data processing failed - insufficient cloud-free pixels")
            
            # Convert statistics to display units
            # Clamp all negative values to zero for cleaner display

            if mean_val is not None:
                molecules_cm2 = mean_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    mean_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    mean_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    mean_val = mean_val
                # Clamp negatives to zero
                mean_val = max(0.0, mean_val)

            if max_val is not None:
                molecules_cm2 = max_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    max_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    max_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    max_val = max_val
                # Clamp negatives to zero
                max_val = max(0.0, max_val)

            if min_val is not None:
                molecules_cm2 = min_val * 6.02214e19
                if gas in ['NO2', 'SO2', 'O3', 'HCHO']:
                    min_val = molecules_cm2 / 1e15
                elif gas == 'CO':
                    min_val = molecules_cm2 / 1e18
                elif gas == 'CH4':
                    min_val = min_val
                # Clamp negatives to zero
                min_val = max(0.0, min_val)



            # Get wind data that matches the satellite observation time
            # This ensures wind arrows point in the correct direction for this specific moment
            wind_data = {}
            if self.enhanced_wind_fetcher:
                try:
                    wind_data = self.fetch_wind_data(city, timestamp_utc)
                except Exception as e:
                    logger.warning(f"Could not fetch wind data: {e}")
                    wind_data = {'success': False, 'error': str(e)}

            # Check if we successfully got usable pollution data
            has_valid_data = (mean_val is not None or max_val is not None or len(pixels) > 0)

            if has_valid_data:
                # Calculate how many days ago this data is from
                days_old = (datetime.now(pytz.UTC) - timestamp_utc).days

                return {
                    'success': True,
                    'city': city,
                    'gas': gas,
                    'timestamp_utc': timestamp_utc,
                    'timestamp_ksa': timestamp_ksa,
                    'days_old': days_old,
                    'data_age_label': f"{days_old} day{'s' if days_old != 1 else ''} ago" if days_old > 0 else "today",
                    'pixels': pixels,
                    'statistics': {
                        'mean': mean_val if mean_val is not None else 0,
                        'max': max_val if max_val is not None else 0,
                        'min': min_val if min_val is not None else 0,
                        'pixel_count': len(pixels)
                    },
                    'unit': gas_config['display_unit'],
                    'bbox': bbox,
                    'wind': wind_data,
                    'data_quality': 'statistics_only' if len(pixels) == 0 else 'full'
                }
            else:
                logger.error(f"No valid data found for {gas} in {city}")
                return self._create_empty_response(city, gas,
                    error="No valid data after all attempts")

        except Exception as e:
            logger.error(f"Error fetching {gas} data for {city}: {e}")
            return self._create_empty_response(city, gas, error=f"Satellite data error: {str(e)}")

    
    def fetch_wind_data(self, city: str, target_time: Optional[datetime] = None) -> Dict:
        """
        Get wind conditions that match the satellite observation time.

        This function automatically searches through multiple weather data sources
        (Tomorrow.io, OpenWeatherMap, WeatherAPI) to find the wind reading closest
        in time to when the satellite passed overhead. This ensures accurate wind
        direction for tracing pollution back to its source.

        The function returns a confidence score showing how well the wind data
        matches the satellite timing.
        """
        if target_time is None:
            target_time = datetime.now(pytz.UTC)
        else:
            target_time = target_time.astimezone(pytz.UTC)

        # Determine appropriate time tolerance based on data age
        hours_ago = abs((datetime.now(pytz.UTC) - target_time).total_seconds() / 3600)

        if hours_ago <= 3:
            max_tolerance = 30  # Recent data - strict tolerance
        elif hours_ago <= 24:
            max_tolerance = 45  # Within a day - moderate tolerance
        else:
            max_tolerance = 60  # Historical data - relaxed tolerance

        # Use enhanced fetcher with all 3 API sources
        wind_data = self.enhanced_wind_fetcher.fetch_wind_data(
            city=city,
            target_time=target_time,
            max_time_diff_minutes=max_tolerance
        )

        # Log wind data confidence for monitoring
        if wind_data.get('confidence', 0) < 50:
            logger.warning(
                f"Low-confidence wind data for {city}: {wind_data.get('confidence', 0)}% "
                f"(Time diff: {wind_data.get('time_difference_minutes', 0):.0f} min)"
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
            end_time = target_time + timedelta(hours=min(window_hours, forward_hours))

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

            ksa_time = wind_timestamp_utc.astimezone(pytz.timezone(config.TIMEZONE))

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
