"""
Test script to debug Earth Engine data availability
Run this locally or in a notebook to test data fetching
"""

import ee
import datetime
from datetime import timedelta
import pytz

# Initialize Earth Engine
try:
    ee.Initialize(project='rcjyenviroment')
    print("✅ Earth Engine initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    exit(1)

def test_data_availability(city_name, city_bbox, gas_name, dataset, band, days_back=7):
    """Test if data is available for a specific city and gas"""
    print(f"\n📍 Testing {gas_name} for {city_name}")
    print(f"Dataset: {dataset}")
    print(f"Band: {band}")
    print(f"Bounding box: {city_bbox}")

    # Define area of interest
    aoi = ee.Geometry.Rectangle(city_bbox)

    # Date range
    end_date = datetime.datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=days_back)

    print(f"Date range: {start_date.date()} to {end_date.date()}")

    try:
        # Load collection
        collection = ee.ImageCollection(dataset) \
            .filterBounds(aoi) \
            .filterDate(start_date.strftime('%Y-%m-%d'),
                       end_date.strftime('%Y-%m-%d')) \
            .select(band)

        # Get collection size
        count = collection.size().getInfo()
        print(f"Found {count} images")

        if count > 0:
            # Get first image info
            first_image = collection.first()
            image_info = first_image.getInfo()

            if image_info:
                # Get date
                timestamp_ms = image_info['properties']['system:time_start']
                img_date = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.UTC)
                print(f"First image date: {img_date}")

                # Get statistics
                stats = first_image.select(band).reduceRegion(
                    reducer=ee.Reducer.mean().combine(
                        reducer2=ee.Reducer.max(),
                        sharedInputs=True
                    ),
                    geometry=aoi,
                    scale=5000,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()

                mean_val = stats.get(f"{band}_mean")
                max_val = stats.get(f"{band}_max")

                print(f"Mean value: {mean_val}")
                print(f"Max value: {max_val}")

                if mean_val is None and max_val is None:
                    print("⚠️ No valid pixels (cloud cover or no data)")
                else:
                    print("✅ Data available!")
            else:
                print("❌ Could not get image info")
        else:
            print("❌ No images found in date range")

            # Try extended range
            print(f"\nTrying extended range ({days_back * 3} days)...")
            extended_start = end_date - timedelta(days=days_back * 3)

            collection = ee.ImageCollection(dataset) \
                .filterBounds(aoi) \
                .filterDate(extended_start.strftime('%Y-%m-%d'),
                           end_date.strftime('%Y-%m-%d')) \
                .select(band)

            count = collection.size().getInfo()
            print(f"Extended search found {count} images")

    except Exception as e:
        print(f"❌ Error: {e}")

# Test cities
cities = {
    "Yanbu": [37.90, 23.85, 38.36, 24.25],
    "Jubail": [49.50, 26.90, 49.80, 27.15],
    "Jazan": [42.4511, 16.7892, 42.6511, 16.9892]
}

# Test gases
gases = {
    "NO2": {
        "dataset": "COPERNICUS/S5P/NRTI/L3_NO2",
        "band": "NO2_column_number_density"
    },
    "SO2": {
        "dataset": "COPERNICUS/S5P/NRTI/L3_SO2",
        "band": "SO2_column_number_density"
    },
    "CO": {
        "dataset": "COPERNICUS/S5P/NRTI/L3_CO",
        "band": "CO_column_number_density"
    }
}

# Run tests
for city_name, bbox in cities.items():
    for gas_name, gas_config in gases.items():
        test_data_availability(
            city_name,
            bbox,
            gas_name,
            gas_config['dataset'],
            gas_config['band'],
            days_back=7
        )
    break  # Test just first city for now

print("\n" + "="*50)
print("Testing complete!")
print("\nIf no data is found:")
print("1. Check if the bounding box coordinates are correct")
print("2. Try a longer date range")
print("3. Check for regional data gaps")
print("4. Verify the dataset and band names are correct")