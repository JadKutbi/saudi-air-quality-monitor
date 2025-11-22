"""
Enhanced Wind Data Fetcher with Near-Perfect Time Synchronization
Multiple real-time sources for accurate wind data at exact pollution detection time
"""

import requests
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
from typing import Dict, Optional, List, Tuple
import json
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedWindFetcher:
    """
    Fetches wind data from multiple sources with time-synchronized accuracy
    Priority order (best to worst temporal accuracy):
    1. Local weather stations via Saudi NCMC (real-time)
    2. METAR airport data (hourly updates)
    3. OpenWeatherMap (real-time current + 1hr forecast)
    4. Tomorrow.io (1-minute temporal resolution)
    5. WeatherAPI.com (real-time + historical)
    6. ERA5-Land hourly reanalysis (5-day lag but hourly)
    7. NOAA GFS (6-hourly, interpolated)
    """

    def __init__(self):
        self.api_keys = {
            'openweather': os.getenv('OPENWEATHER_API_KEY'),
            'tomorrow': os.getenv('TOMORROW_IO_API_KEY'),
            'weatherapi': os.getenv('WEATHERAPI_KEY'),
            'meteomatics': os.getenv('METEOMATICS_USER'),  # user:password format
            'visualcrossing': os.getenv('VISUALCROSSING_KEY')
        }

        # Saudi Arabia weather station coordinates
        self.weather_stations = {
            'Yanbu': {
                'airport_code': 'OEYN',  # Prince Abdul Mohsin Bin Abdulaziz Airport
                'lat': 24.1442,
                'lon': 38.0634,
                'ncmc_station_id': '40430',  # Yanbu station ID
                'nearest_stations': [
                    {'id': '40430', 'name': 'Yanbu', 'distance_km': 0},
                    {'id': '40439', 'name': 'Rabigh', 'distance_km': 150},
                    {'id': '40375', 'name': 'Madinah', 'distance_km': 220}
                ]
            },
            'Jubail': {
                'airport_code': 'OEKK',  # King Fahd Airport nearby
                'lat': 27.0387,
                'lon': 49.6743,
                'ncmc_station_id': '40417',  # Jubail station ID
                'nearest_stations': [
                    {'id': '40417', 'name': 'Jubail', 'distance_km': 0},
                    {'id': '40420', 'name': 'Dhahran', 'distance_km': 80},
                    {'id': '40416', 'name': 'Dammam', 'distance_km': 90}
                ]
            },
            'Jazan': {
                'airport_code': 'OEGN',  # King Abdullah Airport
                'lat': 16.9011,
                'lon': 42.5858,
                'ncmc_station_id': '41140',  # Jazan station ID
                'nearest_stations': [
                    {'id': '41140', 'name': 'Gizan', 'distance_km': 0},
                    {'id': '41136', 'name': 'Sabya', 'distance_km': 30},
                    {'id': '41128', 'name': 'Abha', 'distance_km': 200}
                ]
            }
        }

    def fetch_wind_data(self, city: str, target_time: datetime,
                       max_time_diff_minutes: int = 30) -> Dict:
        """
        Fetch wind data with minimal time difference from target

        Args:
            city: City name
            target_time: Exact time of satellite measurement (UTC)
            max_time_diff_minutes: Maximum acceptable time difference

        Returns:
            Wind data with confidence score based on temporal accuracy
        """
        logger.info(f"Fetching wind for {city} at {target_time} UTC (max diff: {max_time_diff_minutes} min)")

        results = []

        # Determine time difference to pick best sources
        now = datetime.now(pytz.UTC)
        hours_ago = abs((now - target_time).total_seconds() / 3600)

        # Intelligently order sources based on time period
        if hours_ago <= 3:
            # Recent data - prioritize real-time sources
            sources = [
                ('openweather', self._fetch_openweather_realtime),
                ('tomorrow_io', self._fetch_tomorrow_io),
                ('metar', self._fetch_metar_wind),
                ('weatherapi', self._fetch_weatherapi_historical),
            ]
        elif hours_ago <= 24:
            # Within 24 hours - Tomorrow.io works great
            sources = [
                ('tomorrow_io', self._fetch_tomorrow_io),
                ('weatherapi', self._fetch_weatherapi_historical),
                ('openweather', self._fetch_openweather_realtime),
                ('metar', self._fetch_metar_wind),
            ]
        else:
            # Over 24 hours - use historical sources only
            sources = [
                ('weatherapi', self._fetch_weatherapi_historical),
                ('visualcrossing', self._fetch_visualcrossing),
                ('era5_hourly', self._fetch_era5_hourly),
                ('metar', self._fetch_metar_wind),
            ]

        # Add placeholder sources that aren't implemented yet
        sources.extend([
            ('saudi_ncmc', self._fetch_saudi_ncmc_wind),
            ('gfs_interpolated', self._fetch_gfs_interpolated)
        ])

        for source_name, fetch_func in sources:
            try:
                wind_data = fetch_func(city, target_time)
                if wind_data:
                    time_diff = abs((wind_data['observation_time'] - target_time).total_seconds() / 60)

                    if time_diff <= max_time_diff_minutes:
                        # Calculate confidence based on time difference
                        confidence = self._calculate_confidence(time_diff, source_name)

                        wind_data.update({
                            'source': source_name,
                            'time_difference_minutes': round(time_diff, 1),
                            'confidence': confidence,
                            'confidence_reason': self._get_confidence_reason(time_diff, source_name)
                        })

                        logger.info(f"✓ {source_name}: {time_diff:.1f} min difference, {confidence}% confidence")
                        results.append(wind_data)

                        # If we have perfect or near-perfect match, use it immediately
                        if time_diff <= 5:  # Within 5 minutes
                            return wind_data
                    else:
                        logger.debug(f"✗ {source_name}: {time_diff:.1f} min difference (too old)")

            except Exception as e:
                logger.debug(f"✗ {source_name} failed: {e}")

        # Return best result or interpolated data
        if results:
            # Sort by time difference
            results.sort(key=lambda x: x['time_difference_minutes'])
            best_result = results[0]

            # If we have multiple results, try to improve accuracy by interpolation
            if len(results) >= 2 and best_result['time_difference_minutes'] > 15:
                interpolated = self._interpolate_wind_data(results, target_time)
                if interpolated:
                    return interpolated

            return best_result

        # Fallback: return with low confidence
        logger.warning(f"No accurate wind data found for {city} at {target_time}")
        return self._create_fallback_wind_data(city, target_time)

    def _fetch_saudi_ncmc_wind(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        Fetch from Saudi National Center for Meteorology
        Real-time data from actual weather stations in Saudi Arabia
        """
        station = self.weather_stations[city]

        # NCMC provides real-time observations
        # This would require official API access - placeholder for integration
        # Contact: https://ncm.gov.sa

        # For now, return None to try next source
        # In production, this would connect to NCMC's API
        return None

    def _fetch_metar_wind(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        Fetch METAR data from nearby airports
        Updated hourly, very reliable
        """
        station = self.weather_stations[city]
        airport_code = station['airport_code']

        # METAR API endpoints
        endpoints = [
            f"https://aviationweather.gov/api/data/metar?ids={airport_code}&hours=3",
            f"https://api.weather.gov/stations/{airport_code}/observations/latest"
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    # Parse METAR data
                    data = response.text

                    # Extract wind info from METAR string
                    # Format: 31015KT (wind from 310° at 15 knots)
                    import re
                    wind_pattern = r'(\d{3})(\d{2,3})(G\d{2,3})?KT'
                    match = re.search(wind_pattern, data)

                    if match:
                        direction = int(match.group(1))
                        speed_knots = int(match.group(2))
                        speed_ms = speed_knots * 0.514444  # Convert to m/s

                        # METAR updates hourly
                        observation_time = target_time.replace(minute=0, second=0, microsecond=0)

                        return {
                            'direction_deg': direction,
                            'speed_ms': speed_ms,
                            'observation_time': observation_time,
                            'direction_cardinal': self._degrees_to_cardinal(direction)
                        }
            except:
                continue

        return None

    def _fetch_openweather_realtime(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        OpenWeatherMap - Real-time current conditions
        Updates every 10 minutes for major cities
        """
        if not self.api_keys['openweather']:
            return None

        station = self.weather_stations[city]

        # Check if target time is recent (within last 3 hours for current weather)
        now = datetime.now(pytz.UTC)
        time_diff_hours = abs((now - target_time).total_seconds() / 3600)

        if time_diff_hours <= 3:
            # Use current weather API for recent times
            url = (f"https://api.openweathermap.org/data/2.5/weather"
                   f"?lat={station['lat']}&lon={station['lon']}"
                   f"&appid={self.api_keys['openweather']}")

            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    wind = data.get('wind', {})

                    if wind:
                        # For current weather, use actual observation time
                        obs_timestamp = data.get('dt', int(now.timestamp()))
                        observation_time = datetime.fromtimestamp(obs_timestamp, pytz.UTC)

                        return {
                            'direction_deg': wind.get('deg', 0),
                            'speed_ms': wind.get('speed', 0),
                            'observation_time': observation_time,
                            'direction_cardinal': self._degrees_to_cardinal(wind.get('deg', 0))
                        }
            except Exception as e:
                logger.debug(f"OpenWeatherMap current weather failed: {e}")

        # For historical data, OpenWeatherMap requires paid subscription
        # Return None to try other sources
        return None

    def _fetch_tomorrow_io(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        Tomorrow.io (formerly ClimaCell) - 1-minute temporal resolution
        Best for precise historical data
        """
        if not self.api_keys['tomorrow']:
            return None

        station = self.weather_stations[city]

        # Tomorrow.io v4 API with proper formatting
        base_url = "https://api.tomorrow.io/v4/timelines"

        # Properly format the request
        headers = {
            'accept': 'application/json',
            'Accept-Encoding': 'gzip'
        }

        # Create time window around target
        start_time = (target_time - timedelta(minutes=30))
        end_time = (target_time + timedelta(minutes=30))

        # Format timestamps properly for Tomorrow.io (use Z format without timezone)
        start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        params = {
            'location': f"{station['lat']},{station['lon']}",
            'fields': 'windSpeed,windDirection',
            'timesteps': '1m',  # 1-minute resolution
            'startTime': start_str,
            'endTime': end_str,
            'apikey': self.api_keys['tomorrow']
        }

        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Parse the response
                if 'data' in data and 'timelines' in data['data']:
                    timelines = data['data']['timelines']
                    if timelines and len(timelines) > 0:
                        intervals = timelines[0].get('intervals', [])

                        if intervals:
                            # Find closest timestamp
                            closest = None
                            min_diff = float('inf')

                            for interval in intervals:
                                start_str = interval['startTime']
                                # Parse ISO format
                                if start_str.endswith('Z'):
                                    obs_time = datetime.fromisoformat(start_str[:-1] + '+00:00')
                                else:
                                    obs_time = datetime.fromisoformat(start_str)

                                diff = abs((obs_time - target_time).total_seconds())
                                if diff < min_diff:
                                    min_diff = diff
                                    closest = interval

                            if closest:
                                values = closest.get('values', {})
                                start_str = closest['startTime']
                                if start_str.endswith('Z'):
                                    obs_time = datetime.fromisoformat(start_str[:-1] + '+00:00')
                                else:
                                    obs_time = datetime.fromisoformat(start_str)

                                return {
                                    'direction_deg': values.get('windDirection', 0),
                                    'speed_ms': values.get('windSpeed', 0),
                                    'observation_time': obs_time,
                                    'direction_cardinal': self._degrees_to_cardinal(values.get('windDirection', 0))
                                }

            elif response.status_code == 429:
                logger.debug("Tomorrow.io rate limit exceeded")
            elif response.status_code == 400:
                logger.debug(f"Tomorrow.io bad request: {response.text}")
            else:
                logger.debug(f"Tomorrow.io returned status {response.status_code}")

        except Exception as e:
            logger.debug(f"Tomorrow.io error: {e}")

        return None

    def _fetch_weatherapi_historical(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        WeatherAPI.com - Good historical data with hourly resolution
        Free tier includes 7 days of history
        """
        if not self.api_keys['weatherapi']:
            return None

        station = self.weather_stations[city]

        # Check if date is within the last 7 days (free tier limit)
        now = datetime.now(pytz.UTC)
        days_ago = (now - target_time).days

        if days_ago > 7:
            logger.debug(f"WeatherAPI: Date {target_time} is beyond 7-day history limit")
            return None

        # WeatherAPI historical endpoint
        base_url = "http://api.weatherapi.com/v1/history.json"

        params = {
            'key': self.api_keys['weatherapi'],
            'q': f"{station['lat']},{station['lon']}",
            'dt': target_time.strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Parse forecast data
                forecast = data.get('forecast', {}).get('forecastday', [])
                if forecast and len(forecast) > 0:
                    hours = forecast[0].get('hour', [])

                    # Find closest hour
                    closest_hour = None
                    min_diff = float('inf')
                    target_hour = target_time.hour

                    for hour_data in hours:
                        # Use epoch time which is more reliable
                        epoch_time = hour_data.get('time_epoch')
                        if epoch_time:
                            try:
                                # Convert epoch to datetime
                                hour_time = datetime.fromtimestamp(epoch_time, pytz.UTC)

                                diff = abs((hour_time - target_time).total_seconds())
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_hour = hour_data
                                    closest_time = hour_time
                            except Exception as e:
                                logger.debug(f"Error parsing hour data: {e}")
                                continue

                    if closest_hour:
                        return {
                            'direction_deg': closest_hour.get('wind_degree', 0),
                            'speed_ms': closest_hour.get('wind_kph', 0) / 3.6,  # Convert kph to m/s
                            'observation_time': closest_time,
                            'direction_cardinal': closest_hour.get('wind_dir', 'N')
                        }

            elif response.status_code == 400:
                logger.debug(f"WeatherAPI bad request: {response.text}")
            else:
                logger.debug(f"WeatherAPI returned status {response.status_code}")

        except Exception as e:
            logger.debug(f"WeatherAPI error: {e}")

        return None

    def _fetch_visualcrossing(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        Visual Crossing - Excellent historical weather with hourly data
        Combines multiple sources for accuracy
        """
        if not self.api_keys['visualcrossing']:
            return None

        station = self.weather_stations[city]

        # Visual Crossing timeline API
        date_str = target_time.strftime('%Y-%m-%d')
        hour = target_time.hour

        url = (f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
               f"{station['lat']},{station['lon']}/{date_str}T{hour:02d}:00:00")

        params = {
            'key': self.api_keys['visualcrossing'],
            'include': 'hours',
            'elements': 'datetime,windspeed,winddir'
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()

                days = data.get('days', [])
                if days:
                    hours = days[0].get('hours', [])

                    # Find matching hour
                    for hour_data in hours:
                        if hour_data.get('datetime', '').startswith(f"{hour:02d}:"):
                            return {
                                'direction_deg': hour_data.get('winddir', 0),
                                'speed_ms': hour_data.get('windspeed', 0) * 0.44704,  # mph to m/s
                                'observation_time': target_time.replace(minute=0, second=0),
                                'direction_cardinal': self._degrees_to_cardinal(hour_data.get('winddir', 0))
                            }
        except:
            pass

        return None

    def _fetch_era5_hourly(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        ERA5-Land hourly reanalysis
        1-hour resolution but 5-day lag
        Most accurate historical data
        """
        # This would require CDS API setup
        # Provides hourly data with 5-day lag
        # Excellent for historical analysis
        return None

    def _fetch_gfs_interpolated(self, city: str, target_time: datetime) -> Optional[Dict]:
        """
        NOAA GFS with temporal interpolation
        6-hourly data interpolated to target time
        """
        station = self.weather_stations[city]

        # Find the two surrounding GFS times (0, 6, 12, 18 UTC)
        base_hour = (target_time.hour // 6) * 6
        time1 = target_time.replace(hour=base_hour, minute=0, second=0, microsecond=0)
        time2 = time1 + timedelta(hours=6)

        # Weight for interpolation
        weight = (target_time - time1).total_seconds() / 21600  # 6 hours in seconds

        # This is a simplified example
        # In production, fetch actual GFS data and interpolate
        return None

    def _interpolate_wind_data(self, results: List[Dict], target_time: datetime) -> Optional[Dict]:
        """
        Intelligently interpolate between multiple wind measurements
        """
        if len(results) < 2:
            return None

        # Find measurements before and after target time
        before = [r for r in results if r['observation_time'] <= target_time]
        after = [r for r in results if r['observation_time'] > target_time]

        if before and after:
            # Linear interpolation between closest before and after
            b = max(before, key=lambda x: x['observation_time'])
            a = min(after, key=lambda x: x['observation_time'])

            # Calculate interpolation weight
            total_diff = (a['observation_time'] - b['observation_time']).total_seconds()
            weight = (target_time - b['observation_time']).total_seconds() / total_diff

            # Interpolate wind direction (careful with circular values)
            dir1 = b['direction_deg']
            dir2 = a['direction_deg']

            # Handle circular interpolation
            if abs(dir2 - dir1) > 180:
                if dir2 > dir1:
                    dir1 += 360
                else:
                    dir2 += 360

            interp_dir = dir1 + (dir2 - dir1) * weight
            interp_dir = interp_dir % 360

            # Interpolate speed (simple linear)
            interp_speed = b['speed_ms'] + (a['speed_ms'] - b['speed_ms']) * weight

            return {
                'direction_deg': interp_dir,
                'speed_ms': interp_speed,
                'observation_time': target_time,
                'direction_cardinal': self._degrees_to_cardinal(interp_dir),
                'source': 'interpolated',
                'time_difference_minutes': 0,
                'confidence': 85,
                'confidence_reason': f"Interpolated between {b['source']} and {a['source']}"
            }

        return None

    def _calculate_confidence(self, time_diff_minutes: float, source: str) -> int:
        """
        Calculate confidence score based on temporal difference and source quality
        """
        # Base confidence by source
        source_confidence = {
            'saudi_ncmc': 100,      # Official Saudi weather stations
            'metar': 95,            # Airport observations
            'tomorrow_io': 95,      # 1-minute resolution
            'openweather': 90,      # 10-minute updates
            'weatherapi': 85,       # Hourly historical
            'visualcrossing': 85,   # Multiple sources
            'era5_hourly': 90,      # High quality reanalysis
            'gfs_interpolated': 70, # 6-hourly interpolated
            'interpolated': 85      # Interpolated data
        }.get(source, 50)

        # Reduce confidence based on time difference
        if time_diff_minutes <= 5:
            time_penalty = 0
        elif time_diff_minutes <= 15:
            time_penalty = 5
        elif time_diff_minutes <= 30:
            time_penalty = 10
        elif time_diff_minutes <= 60:
            time_penalty = 20
        else:
            time_penalty = 40

        confidence = max(10, source_confidence - time_penalty)

        return confidence

    def _get_confidence_reason(self, time_diff_minutes: float, source: str) -> str:
        """
        Explain confidence score
        """
        if time_diff_minutes <= 5:
            time_desc = "Nearly perfect temporal match"
        elif time_diff_minutes <= 15:
            time_desc = "Good temporal match"
        elif time_diff_minutes <= 30:
            time_desc = "Acceptable temporal match"
        elif time_diff_minutes <= 60:
            time_desc = "Moderate temporal difference"
        else:
            time_desc = "Significant temporal difference"

        source_desc = {
            'saudi_ncmc': "Official Saudi weather station",
            'metar': "Airport weather observation",
            'tomorrow_io': "High-resolution weather API",
            'openweather': "Real-time weather service",
            'weatherapi': "Historical weather data",
            'visualcrossing': "Multi-source weather data",
            'era5_hourly': "ERA5 reanalysis data",
            'gfs_interpolated': "GFS model interpolated",
            'interpolated': "Interpolated from multiple sources"
        }.get(source, "Alternative source")

        return f"{time_desc} ({time_diff_minutes:.1f} min). Source: {source_desc}"

    def _degrees_to_cardinal(self, degrees: float) -> str:
        """Convert degrees to cardinal direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]

    def _create_fallback_wind_data(self, city: str, target_time: datetime) -> Dict:
        """
        Create fallback wind data when no accurate source available
        """
        return {
            'direction_deg': 0,
            'speed_ms': 0,
            'observation_time': target_time,
            'direction_cardinal': 'UNKNOWN',
            'source': 'fallback',
            'time_difference_minutes': 999,
            'confidence': 0,
            'confidence_reason': 'No wind data available - factory attribution unreliable',
            'warning': 'Wind data unavailable - results should not be used for attribution'
        }

    @lru_cache(maxsize=128)
    def get_historical_wind_patterns(self, city: str, hour: int) -> Dict:
        """
        Get typical wind patterns for a specific hour of day
        Useful for validation and anomaly detection
        """
        # This would load historical patterns from database
        # For now, return placeholder
        return {
            'typical_direction': 315,  # NW prevailing winds in Saudi Arabia
            'typical_speed': 4.5,
            'direction_variance': 45,
            'speed_variance': 2.0
        }

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the enhanced wind fetcher
    fetcher = EnhancedWindFetcher()

    # Simulate satellite measurement time
    satellite_time = datetime.now(pytz.UTC) - timedelta(hours=3)  # 3 hours ago

    print("\n" + "="*60)
    print("ENHANCED WIND DATA FETCHING TEST")
    print("="*60)

    for city in ['Yanbu', 'Jubail', 'Jazan']:
        print(f"\n📍 Testing {city}")
        print(f"   Satellite time: {satellite_time}")

        wind = fetcher.fetch_wind_data(city, satellite_time, max_time_diff_minutes=30)

        print(f"   Result: {wind['source']}")
        print(f"   Time difference: {wind['time_difference_minutes']} minutes")
        print(f"   Confidence: {wind['confidence']}%")
        print(f"   Wind: {wind['speed_ms']:.1f} m/s from {wind['direction_cardinal']} ({wind['direction_deg']}°)")
        print(f"   Reason: {wind['confidence_reason']}")

        if wind['confidence'] < 50:
            print(f"   ⚠️ WARNING: Low confidence - attribution unreliable!")