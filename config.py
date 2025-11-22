"""
Configuration for pollution monitoring system
"""

import os
from datetime import datetime

# City configurations
CITIES = {
    "Yanbu": {
        "center": [24.0889, 38.0618],
        "bbox": [37.90, 23.85, 38.36, 24.25],  # EXPANDED: covers Yanbu Industrial City fully
        "radius_km": 25  # Increased radius
    },

    "Jubail": {
        "center": [27.0173, 49.6575],
        "bbox": [49.50, 26.90, 49.80, 27.15],  # EXPANDED bbox
        "radius_km": 25
    },

    "Jazan": {
        "center": [16.8892, 42.5511],
        "bbox": [42.4511, 16.7892, 42.6511, 16.9892],
        "radius_km": 15
    }
}

# Gas monitoring
GAS_PRODUCTS = {
    "NO2": {
        "name": "Nitrogen Dioxide",
        "dataset": "COPERNICUS/S5P/NRTI/L3_NO2",
        "band": "NO2_column_number_density",
        "unit": "mol/m²",
        "conversion_factor": 1e15,
        "display_unit": "10^15 mol/cm²"
    },
    "SO2": {
        "name": "Sulfur Dioxide",
        "dataset": "COPERNICUS/S5P/NRTI/L3_SO2",
        "band": "SO2_column_number_density",
        "unit": "mol/m²",
        "conversion_factor": 1e15,
        "display_unit": "10^15 mol/cm²"
    },
    "CO": {
        "name": "Carbon Monoxide",
        "dataset": "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m²",
        "conversion_factor": 1e18,
        "display_unit": "10^18 mol/cm²"
    },
    # O3 REMOVED - Measures total column including stratosphere, not suitable for pollution monitoring
    "HCHO": {
        "name": "Formaldehyde",
        "dataset": "COPERNICUS/S5P/NRTI/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "unit": "mol/m²",
        "conversion_factor": 1e15,
        "display_unit": "10^15 mol/cm²"
    },
    "CH4": {
        "name": "Methane",
        "dataset": "COPERNICUS/S5P/OFFL/L3_CH4",
        "band": "CH4_column_volume_mixing_ratio_dry_air",
        "unit": "ppb",
        "conversion_factor": 1,
        "display_unit": "ppb"
    }
}

# WHO 2021 thresholds
GAS_THRESHOLDS = {
    "NO2": {
        "annual_avg_ugm3": 10,
        "24h_avg_ugm3": 25,
        "1h_avg_ugm3": 200,
        "column_threshold": 10.0,    # 10^15 molecules/cm² (moderate pollution)
        "critical_threshold": 20.0,  # 10^15 molecules/cm² (high pollution)
        "unit": "10^15 mol/cm²",
        "source": "WHO Air Quality Guidelines 2021 + Sentinel-5P validation"
    },
    "SO2": {
        "24h_avg_ugm3": 40,
        "10min_avg_ugm3": 500,
        "column_threshold": 2.0,     # 10^15 molecules/cm² (industrial level) - CORRECTED
        "critical_threshold": 5.0,   # 10^15 molecules/cm² (heavy pollution) - CORRECTED
        "unit": "10^15 mol/cm²",
        "source": "WHO 2021 + Environmental validation - Fixed Nov 22"
    },
    "CO": {
        "24h_avg_mgm3": 4,
        "8h_avg_mgm3": 10,
        "1h_avg_mgm3": 35,
        "column_threshold": 3.5,     # 10^18 molecules/cm² (moderate - 0.058 mol/m²)
        "critical_threshold": 5.0,   # 10^18 molecules/cm² (high - 0.083 mol/m²)
        "unit": "10^18 mol/cm²",
        "source": "WHO/EPA Standards + Sentinel-5P typical values"
    },
    # O3 REMOVED - Total column measurements not comparable to WHO ground-level guidelines
    "HCHO": {
        "30min_avg_ugm3": 100,
        "column_threshold": 8.0,     # 10^15 molecules/cm² (elevated)
        "critical_threshold": 12.0,  # 10^15 molecules/cm² (high pollution)
        "unit": "10^15 mol/cm²",
        "source": "WHO Indoor Air Quality Guidelines + tropospheric studies"
    },
    "CH4": {
        "background_ppb": 1900,
        "column_threshold": 1950,
        "critical_threshold": 2100,
        "unit": "ppb",
        "source": "NOAA Global Monitoring Laboratory"
    }
}


# Factory database
FACTORIES = {
    "Yanbu": [
        {
            "name": "Yanbu Aramco Sinopec Refining Company (YASREF)",
            "location": [23.97180767441081, 38.27666340484693],
            "type": "Oil Refinery",
            "emissions": ["NO2", "SO2", "CO", "CH4", "HCHO"],
            "capacity": "400,000 bpd",
            "source": "Saudi Aramco / Sinopec JV",
            "verified": True
        },
        {
            "name": "Yanbu National Petrochemical Company (Yansab)",
            "location": [23.98755566320477, 38.26118777754422],
            "type": "Petrochemical",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Ethylene, polyethylene, polypropylene",
            "source": "SABIC 51% owned",
            "verified": True
        },
        {
            "name": "Saudi Aramco Mobil Refinery Company (SAMREF)",
            "location": [23.98170815715227, 38.23998762712382],
            "type": "Oil Refinery",
            "emissions": ["NO2", "SO2", "CO", "CH4", "HCHO"],
            "capacity": "Oil refinery operations",
            "source": "Saudi Aramco / ExxonMobil JV",
            "verified": True
        },
        {
            "name": "National Industrial Gaseous Company - GAS - SABIC Affiliate",
            "location": [23.99145021834566, 38.233966223430244],
            "type": "Industrial Gases",
            "emissions": ["NO2", "CO", "CH4"],
            "capacity": "Industrial gas production",
            "source": "SABIC Affiliate",
            "verified": True
        },
        {
            "name": "Farabi Yanbu Petrochemicals",
            "location": [23.997065895150783, 38.245319078310835],
            "type": "Petrochemical",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Petrochemical production",
            "source": "Farabi Petrochemicals",
            "verified": True
        },
        {
            "name": "WangKang Ceramic Factory",
            "location": [23.99454460117585, 38.27172544242537],
            "type": "Ceramic Manufacturing",
            "emissions": ["NO2", "SO2"],
            "capacity": "Ceramic production",
            "source": "WangKang Industrial",
            "verified": True
        },
        {
            "name": "KEMYEA YANBU FOR INDUSTRY LLC (KEMYAN)",
            "location": [23.991679434439412, 38.279816703638616],
            "type": "Chemical/Industrial",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Chemical manufacturing",
            "source": "Kemyan Industrial",
            "verified": True
        },
        {
            "name": "Elkhayyat Ceramic Factory",
            "location": [24.000389343563565, 38.278436798625506],
            "type": "Ceramic Manufacturing",
            "emissions": ["NO2", "SO2"],
            "capacity": "Ceramic production",
            "source": "Elkhayyat Industrial",
            "verified": True
        },
        {
            "name": "Saudi Aramco - Yanbu Refinery",
            "location": [23.958214308995846, 38.2922050960996],
            "type": "Oil Refinery",
            "emissions": ["NO2", "SO2", "CO", "CH4", "HCHO"],
            "capacity": "Oil refinery operations",
            "source": "Saudi Aramco",
            "verified": True
        },
        {
            "name": "LUBEREF (Saudi Aramco Base Oil Company)",
            "location": [23.94280933497991, 38.31558992864651],
            "type": "Lubricant Manufacturing",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Base oil production",
            "source": "Saudi Aramco subsidiary",
            "verified": True
        },
        {
            "name": "Yamamah Steel Plant",
            "location": [23.929242832773316, 38.337334853848496],
            "type": "Steel Manufacturing",
            "emissions": ["NO2", "SO2", "CO"],
            "capacity": "Steel production",
            "source": "Yamamah Industrial",
            "verified": True
        },
        {
            "name": "REVIVA GEMS - IWMC YANBU",
            "location": [23.936437388058888, 38.34763377503578],
            "type": "Industrial Waste Management",
            "emissions": ["NO2", "CO", "CH4"],
            "capacity": "Waste management facility",
            "source": "Reviva Environmental",
            "verified": True
        },
        {
            "name": "Saline Water Conversion Corporation Yanbu Medina",
            "location": [23.854613250074483, 38.386043774124936],
            "type": "Desalination/Power",
            "emissions": ["NO2", "SO2", "CO", "CH4"],
            "capacity": "Desalination and power generation",
            "source": "SWCC",
            "verified": True
        },
        {
            "name": "ASK GYPSUM",
            "location": [24.00966677237242, 38.26981308872577],
            "type": "Gypsum Manufacturing",
            "emissions": ["NO2", "SO2"],
            "capacity": "Gypsum production",
            "source": "ASK Industrial",
            "verified": True
        },
        {
            "name": "Lubrizol",
            "location": [23.957106616346582, 38.23656228492223],
            "type": "Chemical Manufacturing",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Specialty chemicals",
            "source": "Lubrizol Corporation",
            "verified": True
        },
        {
            "name": "Marafiq IWTP & SWTP",
            "location": [23.970008910654663, 38.22397924555842],
            "type": "Water Treatment Plant",
            "emissions": ["NO2", "CO"],
            "capacity": "Industrial water treatment",
            "source": "Marafiq Utilities",
            "verified": True
        },
        {
            "name": "MARAFIQ (The Power and Utility company for Jubail & Yanbu)",
            "location": [23.905797209940143, 38.323333590205266],
            "type": "Power/Desalination",
            "emissions": ["NO2", "SO2", "CO", "CH4"],
            "capacity": "2,750 MW + desalination",
            "source": "Marafiq - Main utility provider",
            "verified": True
        },
    ],


    "Jubail": [
        {
            "name": "SABIC Petrochemicals Complex (Jubail)",
            "location": [27.0456, 49.5867],  # Verified: 27°02'44.2"N 49°35'12.1"E - JIC1
            "type": "Petrochemical",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Multiple facilities",
            "source": "SABIC - Jubail Industrial City",
            "verified": True
        },
        {
            "name": "Saudi Aramco Jubail Refinery (SAMREF)",
            "location": [27.0069, 49.6589],  # Verified: 27°00'24.8"N 49°39'32.0"E - Coastal
            "type": "Oil Refinery",
            "emissions": ["NO2", "SO2", "CO", "CH4", "HCHO"],
            "capacity": "550,000 bpd",
            "source": "Saudi Aramco / ExxonMobil JV",
            "verified": True
        },
        {
            "name": "KEMYA (Saudi Methanol Company)",
            "location": [27.0567, 49.5734],  # Verified: 27°03'24.1"N 49°34'24.2"E
            "type": "Methanol",
            "emissions": ["NO2", "CO", "CH4", "HCHO"],
            "capacity": "Methanol production",
            "source": "SABIC / Celanese JV",
            "verified": True
        },
        {
            "name": "Saudi Iron & Steel Company (Hadeed)",
            "location": [27.0289, 49.6201],  # Verified: 27°01'44.0"N 49°37'12.4"E
            "type": "Steel",
            "emissions": ["NO2", "SO2", "CO"],
            "capacity": "6 million tons/year",
            "source": "SABIC subsidiary",
            "verified": True
        },
        {
            "name": "Ma'aden Phosphate Company (Jubail)",
            "location": [27.0623, 49.5923],  # Verified: 27°03'44.3"N 49°35'32.3"E
            "type": "Fertilizer",
            "emissions": ["NO2", "SO2"],
            "capacity": "Phosphate fertilizer",
            "source": "Ma'aden / SABIC JV",
            "verified": True
        },
        {
            "name": "Air Products Industrial Gases Hub",
            "location": [27.0512, 49.6123],  # Verified: 27°03'04.3"N 49°36'44.3"E
            "type": "Industrial Gases",
            "emissions": ["NO2", "CO", "CH4"],
            "capacity": "World's largest hydrogen plant",
            "source": "Air Products",
            "verified": True
        },
        {
            "name": "Saudi Kayan Petrochemical Company",
            "location": [27.0428, 49.5978],  # Added: Major petrochemical facility
            "type": "Petrochemical",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Integrated petrochemical complex",
            "source": "Saudi Kayan (SABIC affiliate)",
            "verified": True
        }
    ],
    "Jazan": [
        {
            "name": "Jazan Refinery & Terminal",
            "location": [16.7089, 42.6734],  # Verified: 16°42'32.0"N 42°40'24.2"E
            "type": "Oil Refinery",
            "emissions": ["NO2", "SO2", "CO", "CH4", "HCHO"],
            "capacity": "400,000 bpd",
            "source": "Saudi Aramco (Operational since 2021)",
            "verified": True
        },
        {
            "name": "Jazan IGCC Power Plant",
            "location": [16.7156, 42.6689],  # Verified: 16°42'56.2"N 42°40'08.0"E
            "type": "Power Generation",
            "emissions": ["NO2", "SO2", "CO"],
            "capacity": "3,800 MW (gasification)",
            "source": "Saudi Aramco Power Company",
            "verified": True
        },
        {
            "name": "Jazan Economic City Industrial Zone",
            "location": [16.8892, 42.5511],  # Verified: 16°53'21.1"N 42°33'04.0"E
            "type": "Mixed Industrial",
            "emissions": ["NO2", "SO2", "CO", "HCHO", "CH4"],
            "capacity": "Various light industries",
            "source": "Jazan Economic City Authority",
            "verified": True
        },
        {
            "name": "Jazan Port Industrial Complex",
            "location": [16.8750, 42.5833],  # Added: Port industrial area
            "type": "Port / Logistics",
            "emissions": ["NO2", "SO2", "CO"],
            "capacity": "Port operations and logistics",
            "source": "Saudi Ports Authority",
            "verified": True
        }
    ]
}

# Wind data configuration
# Wind data sources (ordered by priority)
# 1. OpenWeatherMap API (real-time, configured via .env)
# 2. ERA5 Reanalysis (historical, from Google Earth Engine)

WIND_SOURCES = [
    {
        "id": "era5_land_hourly",
        "label": "ECMWF ERA5-Land Hourly (5-day lag, BEST ACCURACY)",
        "dataset": "ECMWF/ERA5_LAND/HOURLY",
        "u_component": "u_component_of_wind_10m",
        "v_component": "v_component_of_wind_10m",
        "scale": 11132,  # ERA5-Land resolution: ~11km (higher than GFS)
        "search_windows_hours": [1, 2, 3, 6, 12, 24, 48, 72],  # Aggressive hourly search (1-3 days back)
        "forward_search_hours": 0,  # Only look backwards
        "max_time_offset_hours": 72,  # Accept data up to 3 days old
        "base_reliability": 0.95,  # ERA5 is highly reliable (reanalysis data)
        "sample_radius_km": 30,  # Sample area around city
    },
    {
        "id": "noaa_gfs",
        "label": "NOAA GFS (6-hourly, FREE, fallback)",
        "dataset": "NOAA/GFS0P25",
        "u_component": "u_component_of_wind_10m_above_ground",
        "v_component": "v_component_of_wind_10m_above_ground",
        "scale": 27830,  # GFS resolution: 0.25° (~28km)
        "search_windows_hours": [1, 3, 6, 12, 24, 48],  # Search up to 2 days back
        "forward_search_hours": 0,  # Only look backwards
        "max_time_offset_hours": 48,  # Accept data up to 2 days old
        "base_reliability": 0.92,  # NOAA GFS is highly reliable (operational forecast/analysis)
        "sample_radius_km": 40,  # Sample area around city
    },
    {
        "id": "era5_daily",
        "label": "ECMWF ERA5 Daily (fallback)",
        "dataset": "ECMWF/ERA5/DAILY",
        "u_component": "u_component_of_wind_10m",
        "v_component": "v_component_of_wind_10m",
        "scale": 27830,  # ERA5 Daily resolution: ~28km
        "search_windows_hours": [24, 72],  # Daily data, search 1-3 days back
        "forward_search_hours": 0,
        "max_time_offset_hours": 72,
        "base_reliability": 0.90,  # Slightly lower (daily aggregation)
        "sample_radius_km": 50,
    },
]

# Priority order (for Earth Engine sources):
# 1. ERA5-Land Hourly (BEST: hourly data, 5-day lag, 11km resolution) ⭐ PRIMARY
# 2. NOAA GFS (6-hourly, FREE, same-day availability)
# 3. ERA5 Daily (fallback, 5-day lag)
# Note: OpenWeatherMap API (real-time) is tried as final fallback if all Earth Engine sources fail

WIND_DEFAULTS = {
    "speed_ms": 2.0,
    "direction_deg": 0.0,
    "confidence": 0.0,
    "cardinal": "N",
}

LIVE_VIOLATION_TTL_HOURS = 12
LIVE_VIOLATIONS_MAX_ENTRIES = 50

# System configuration
SCAN_INTERVAL_HOURS = 12
SCAN_INTERVAL_MINUTES = SCAN_INTERVAL_HOURS * 60
TIMEZONE = "Asia/Riyadh"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIOLATION_DIR = os.path.join(BASE_DIR, "violations")
LIVE_VIOLATIONS_FILE = os.path.join(LOG_DIR, "live_violations.json")

for directory in [DATA_DIR, LOG_DIR, VIOLATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Notification configuration
def _env_bool(name: str, default: bool = False) -> bool:
    """Parse environment variable into boolean flag."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated environment variable into list."""
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


NOTIFICATION_CONFIG = {
    "email": {
        "enabled": _env_bool("EMAIL_NOTIFICATIONS_ENABLED", False),
        "smtp_server": os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("EMAIL_SMTP_PORT", "587")),
        "sender_email": os.getenv("EMAIL_SENDER_ADDRESS", "jadkutbi@gmail.com"),
        "sender_password": os.getenv("EMAIL_SENDER_PASSWORD", "@jK6833800"),
        "recipients": _env_list("EMAIL_RECIPIENTS", ["jadkutbi@gmail.com"]),
    },
    "webhook": {
        "enabled": _env_bool("WEBHOOK_NOTIFICATIONS_ENABLED", False),
        "url": os.getenv("WEBHOOK_URL", "https://your-webhook-url.com/notify"),
    },
    "telegram": {
        "enabled": _env_bool("TELEGRAM_NOTIFICATIONS_ENABLED", False),
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    },
}

GEE_PROJECT = 'ee-jadkutbi'
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
