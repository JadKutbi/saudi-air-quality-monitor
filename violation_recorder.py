"""
Violation Recorder - Records and manages violation history using Google Cloud Firestore
Maps are stored as HTML content directly in Firestore documents
Falls back to local storage if Firestore is not available
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
import config
import pytz

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Try to import Firestore
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    logger.warning("google-cloud-firestore not installed. Using local storage only.")


class ViolationRecorder:
    """Records violations with persistent storage using Google Cloud Firestore"""

    def __init__(self, violations_dir: str = "violations"):
        """
        Initialize ViolationRecorder with Firestore (primary) or local storage (fallback)

        Args:
            violations_dir: Local directory for fallback storage
        """
        self.violations_dir = violations_dir
        self.maps_dir = os.path.join(violations_dir, "maps")
        self.records_file = os.path.join(violations_dir, "violation_records.json")

        self.db = None
        self.collection_name = "violations"
        self.use_firestore = False
        self.writable = False

        # Try to initialize Firestore first
        if FIRESTORE_AVAILABLE:
            self._init_firestore()

        # Fallback to local storage if Firestore not available
        if not self.use_firestore:
            self._init_local_storage()

        logger.info(f"ViolationRecorder initialized. Firestore: {self.use_firestore}, Writable: {self.writable}")

    def _init_firestore(self):
        """Initialize Google Cloud Firestore connection"""
        try:
            # Try to get credentials from Streamlit secrets or environment
            import streamlit as st

            # Check for service account credentials
            if hasattr(st, 'secrets') and 'GEE_SERVICE_ACCOUNT' in st.secrets:
                # Use the same service account as Earth Engine
                from google.oauth2 import service_account

                # Build credentials dict from secrets
                credentials_dict = {
                    "type": "service_account",
                    "project_id": st.secrets.get("GEE_PROJECT_ID", config.GEE_PROJECT),
                    "private_key_id": st.secrets.get("GEE_PRIVATE_KEY_ID", ""),
                    "private_key": st.secrets.get("GEE_PRIVATE_KEY", "").replace("\\n", "\n"),
                    "client_email": st.secrets.get("GEE_SERVICE_ACCOUNT", ""),
                    "client_id": st.secrets.get("GEE_CLIENT_ID", ""),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }

                credentials = service_account.Credentials.from_service_account_info(
                    credentials_dict,
                    scopes=["https://www.googleapis.com/auth/datastore"]
                )

                project_id = st.secrets.get("GEE_PROJECT_ID", config.GEE_PROJECT)
                self.db = firestore.Client(project=project_id, credentials=credentials)
                logger.info(f"Firestore initialized with service account for project: {project_id}")

            else:
                # Try default credentials (works locally with gcloud auth)
                self.db = firestore.Client(project=config.GEE_PROJECT)
                logger.info(f"Firestore initialized with default credentials for project: {config.GEE_PROJECT}")

            # Test connection by trying to access the collection
            test_ref = self.db.collection(self.collection_name).limit(1)
            list(test_ref.stream())  # This will raise an exception if not authorized

            self.use_firestore = True
            self.writable = True
            logger.info("Firestore connection test successful")

        except Exception as e:
            logger.warning(f"Firestore initialization failed: {e}")
            logger.info("Falling back to local storage")
            self.use_firestore = False

    def _init_local_storage(self):
        """Initialize local file storage as fallback"""
        try:
            os.makedirs(self.violations_dir, exist_ok=True)
            os.makedirs(self.maps_dir, exist_ok=True)
            logger.info(f"Local storage initialized: {os.path.abspath(self.violations_dir)}")

            # Test write access
            test_file = os.path.join(self.violations_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            self.writable = True
            logger.info("Local storage is writable")
        except Exception as e:
            logger.error(f"Local storage initialization failed: {e}")
            self.writable = False

    def violation_exists(self, city: str, gas: str, satellite_timestamp: str) -> Optional[str]:
        """
        Check if a violation with the same city, gas, and satellite timestamp already exists.

        Args:
            city: City name
            gas: Gas type (e.g., 'NO2', 'SO2')
            satellite_timestamp: The satellite observation timestamp

        Returns:
            Violation ID if exists, None otherwise
        """
        try:
            # Get recent violations for this city and gas
            violations = self.get_all_violations(city=city, gas=gas, limit=100)

            for v in violations:
                # Check satellite_timestamp field (new format) or timestamp_ksa (old format fallback)
                stored_sat_ts = v.get('satellite_timestamp', v.get('timestamp_ksa', ''))
                if stored_sat_ts == satellite_timestamp:
                    logger.info(f"Violation already exists: {v.get('id')} for {gas} at {satellite_timestamp}")
                    return v.get('id')

            return None
        except Exception as e:
            logger.error(f"Error checking for existing violation: {e}")
            return None

    def save_violation(self, violation_data: Dict, analysis: str,
                      map_html_path: Optional[str] = None) -> Optional[str]:
        """
        Save a violation record with all details

        Args:
            violation_data: Violation information dictionary
            analysis: AI analysis text
            map_html_path: Path to saved HTML map (not stored in Firestore)

        Returns:
            Violation ID (timestamp-based) or None if failed, or existing ID if duplicate
        """
        if not self.writable:
            logger.error("Cannot save violation: storage is not writable")
            return None

        try:
            city = violation_data.get('city', '')
            gas = violation_data.get('gas', '')
            satellite_timestamp = violation_data.get('timestamp_ksa', '')

            # Check if this violation already exists
            existing_id = self.violation_exists(city, gas, satellite_timestamp)
            if existing_id:
                logger.info(f"Skipping duplicate violation: {existing_id}")
                return existing_id  # Return existing ID instead of None

            logger.info(f"Saving new violation for {gas} in {city}")

            # Generate unique violation ID based on save timestamp
            ksa_tz = pytz.timezone(config.TIMEZONE)
            now = datetime.now(ksa_tz)
            violation_id = now.strftime("%Y%m%d_%H%M%S")
            gas = violation_data['gas']
            city = violation_data['city']

            # Create full ID: YYYYMMDD_HHMMSS_CITY_GAS
            full_id = f"{violation_id}_{city}_{gas}"

            # Get satellite observation timestamp (the unique identifier for this violation)
            satellite_timestamp = violation_data.get('timestamp_ksa', 'N/A')

            # Prepare record (Firestore-compatible - no complex objects)
            record = {
                'id': full_id,
                'timestamp': now.isoformat(),  # When saved
                'timestamp_ksa': now.strftime("%Y-%m-%d %H:%M:%S KSA"),  # When saved (display)
                'satellite_timestamp': satellite_timestamp,  # Satellite observation time (unique key)
                'city': city,
                'gas': gas,
                'gas_name': violation_data['gas_name'],
                'max_value': float(violation_data['max_value']),
                'threshold': float(violation_data['threshold']),
                'unit': violation_data['unit'],
                'severity': violation_data['severity'],
                'percentage_over': float(violation_data['percentage_over']),
                'ai_analysis': analysis,
            }

            # Add hotspot data if available
            if violation_data.get('hotspot'):
                hotspot = violation_data['hotspot']
                record['hotspot'] = {
                    'lat': float(hotspot.get('lat', 0)),
                    'lon': float(hotspot.get('lon', 0)),
                    'value': float(hotspot.get('value', 0)),
                    'gas': hotspot.get('gas', ''),
                    'unit': hotspot.get('unit', '')
                }

            # Add wind data if available
            if violation_data.get('wind'):
                wind = violation_data['wind']
                record['wind'] = {
                    'success': wind.get('success', False),
                    'direction_deg': float(wind.get('direction_deg', 0)) if wind.get('direction_deg') else 0,
                    'direction_cardinal': wind.get('direction_cardinal', 'N'),
                    'speed_ms': float(wind.get('speed_ms', 0)) if wind.get('speed_ms') else 0,
                    'confidence': float(wind.get('confidence', 0)) if wind.get('confidence') else 0,
                    'source_label': wind.get('source_label', ''),
                    'timestamp_ksa': str(wind.get('timestamp_ksa', '')) if wind.get('timestamp_ksa') else '',
                }

            # Add simplified factory data (top 5 only, essential fields)
            if violation_data.get('nearby_factories'):
                factories_simplified = []
                for f in violation_data['nearby_factories'][:5]:
                    factories_simplified.append({
                        'name': f.get('name', ''),
                        'type': f.get('type', ''),
                        'distance_km': float(f.get('distance_km', 0)),
                        'confidence': float(f.get('confidence', 0)),
                        'likely_upwind': bool(f.get('likely_upwind', False)),
                        'emissions': f.get('emissions', [])
                    })
                record['nearby_factories'] = factories_simplified

            # Save to appropriate storage
            if self.use_firestore:
                return self._save_to_firestore(full_id, record, map_html_path)
            else:
                return self._save_to_local(full_id, record, map_html_path)

        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _save_to_firestore(self, full_id: str, record: Dict,
                           map_html_path: Optional[str] = None) -> Optional[str]:
        """Save record to Firestore and upload map to Cloud Storage"""
        try:
            # Upload map to Cloud Storage if available
            # Store map HTML content directly in Firestore
            if map_html_path and os.path.exists(map_html_path):
                try:
                    with open(map_html_path, 'r', encoding='utf-8') as f:
                        map_html_content = f.read()
                    # Only store if under 900KB (Firestore doc limit is 1MB)
                    if len(map_html_content) < 900000:
                        record['map_html'] = map_html_content
                        logger.info(f"Map HTML stored in Firestore ({len(map_html_content)} bytes)")
                    else:
                        logger.warning(f"Map HTML too large ({len(map_html_content)} bytes), skipping")
                except Exception as html_err:
                    logger.warning(f"Could not read map HTML: {html_err}")

            doc_ref = self.db.collection(self.collection_name).document(full_id)
            doc_ref.set(record)
            logger.info(f"Violation saved to Firestore: {full_id}")
            return full_id
        except Exception as e:
            logger.error(f"Firestore save failed: {e}")
            return None

    def _save_to_local(self, full_id: str, record: Dict,
                       map_html_path: Optional[str] = None) -> Optional[str]:
        """Save record to local JSON file"""
        try:
            # Handle map files for local storage
            if map_html_path and os.path.exists(map_html_path):
                import shutil
                map_html_filename = f"{full_id}_map.html"
                map_html_dest = os.path.join(self.maps_dir, map_html_filename)
                shutil.copy(map_html_path, map_html_dest)
                record['map_html'] = map_html_filename

            # Load existing records and append
            records = self._load_local_records()
            records.append(record)
            self._save_local_records(records)

            logger.info(f"Violation saved locally: {full_id}")
            return full_id
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return None

    def get_all_violations(self, city: Optional[str] = None,
                          gas: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict]:
        """
        Get violation history with optional filtering

        Args:
            city: Filter by city (optional)
            gas: Filter by gas type (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of violation records (newest first)
        """
        if self.use_firestore:
            return self._get_from_firestore(city, gas, limit)
        else:
            return self._get_from_local(city, gas, limit)

    def _get_from_firestore(self, city: Optional[str] = None,
                           gas: Optional[str] = None,
                           limit: Optional[int] = None) -> List[Dict]:
        """Get records from Firestore"""
        try:
            logger.info(f"Fetching violations from Firestore (city={city}, gas={gas}, limit={limit})")

            # Get all documents and filter in Python to avoid index requirements
            collection_ref = self.db.collection(self.collection_name)
            docs = list(collection_ref.stream())

            logger.info(f"Firestore returned {len(docs)} documents")

            records = []
            for doc in docs:
                try:
                    data = doc.to_dict()
                    if data:
                        records.append(data)
                except Exception as doc_err:
                    logger.error(f"Error parsing document {doc.id}: {doc_err}")

            logger.info(f"Parsed {len(records)} records from Firestore")

            # Filter by city if specified
            if city:
                before_filter = len(records)
                records = [r for r in records if r.get('city') == city]
                logger.info(f"Filtered by city '{city}': {before_filter} -> {len(records)}")

            # Filter by gas if specified
            if gas:
                before_filter = len(records)
                records = [r for r in records if r.get('gas') == gas]
                logger.info(f"Filtered by gas '{gas}': {before_filter} -> {len(records)}")

            # Sort by timestamp (newest first)
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            # Apply limit if specified
            if limit:
                records = records[:limit]

            logger.info(f"Returning {len(records)} violations from Firestore")
            return records

        except Exception as e:
            logger.error(f"Firestore query failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _get_from_local(self, city: Optional[str] = None,
                       gas: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict]:
        """Get records from local storage"""
        records = self._load_local_records()

        # Filter by city if specified
        if city:
            records = [r for r in records if r.get('city') == city]

        # Filter by gas if specified
        if gas:
            records = [r for r in records if r.get('gas') == gas]

        # Sort by timestamp (newest first)
        records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Apply limit if specified
        if limit:
            records = records[:limit]

        return records

    def get_violation_by_id(self, violation_id: str) -> Optional[Dict]:
        """Get a specific violation by ID"""
        if self.use_firestore:
            try:
                doc_ref = self.db.collection(self.collection_name).document(violation_id)
                doc = doc_ref.get()
                if doc.exists:
                    return doc.to_dict()
                return None
            except Exception as e:
                logger.error(f"Firestore get failed: {e}")
                return None
        else:
            records = self._load_local_records()
            for record in records:
                if record.get('id') == violation_id:
                    return record
            return None

    def get_violation_map_path(self, violation_id: str,
                              image: bool = True) -> Optional[str]:
        """
        Get path to violation map (local storage only)
        Note: Map files are not stored in Firestore
        """
        if self.use_firestore:
            # Firestore doesn't store map files
            return None

        record = self.get_violation_by_id(violation_id)
        if not record:
            return None

        if image:
            filename = record.get('map_image')
        else:
            filename = record.get('map_html')

        if filename:
            return os.path.join(self.maps_dir, filename)
        return None

    def delete_violation(self, violation_id: str) -> bool:
        """Delete a violation record"""
        if self.use_firestore:
            try:
                doc_ref = self.db.collection(self.collection_name).document(violation_id)
                doc_ref.delete()
                logger.info(f"Deleted violation from Firestore: {violation_id}")
                return True
            except Exception as e:
                logger.error(f"Firestore delete failed: {e}")
                return False
        else:
            try:
                records = self._load_local_records()
                record = self.get_violation_by_id(violation_id)

                if not record:
                    return False

                # Delete map files
                if record.get('map_html'):
                    html_path = os.path.join(self.maps_dir, record['map_html'])
                    if os.path.exists(html_path):
                        os.remove(html_path)

                if record.get('map_image'):
                    img_path = os.path.join(self.maps_dir, record['map_image'])
                    if os.path.exists(img_path):
                        os.remove(img_path)

                # Remove record from list
                records = [r for r in records if r.get('id') != violation_id]
                self._save_local_records(records)

                logger.info(f"Deleted violation locally: {violation_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete violation {violation_id}: {e}")
                return False

    def get_statistics(self, city: Optional[str] = None) -> Dict:
        """Get violation statistics"""
        records = self.get_all_violations(city=city)

        if not records:
            return {
                'total_violations': 0,
                'by_gas': {},
                'by_severity': {},
                'by_city': {},
                'storage_type': 'firestore' if self.use_firestore else 'local'
            }

        stats = {
            'total_violations': len(records),
            'by_gas': {},
            'by_severity': {},
            'by_city': {},
            'date_range': {
                'oldest': records[-1].get('timestamp_ksa', 'Unknown'),
                'newest': records[0].get('timestamp_ksa', 'Unknown')
            },
            'storage_type': 'firestore' if self.use_firestore else 'local'
        }

        # Count by gas
        for record in records:
            gas = record.get('gas', 'Unknown')
            stats['by_gas'][gas] = stats['by_gas'].get(gas, 0) + 1

        # Count by severity
        for record in records:
            severity = record.get('severity', 'Unknown')
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        # Count by city
        for record in records:
            city_name = record.get('city', 'Unknown')
            stats['by_city'][city_name] = stats['by_city'].get(city_name, 0) + 1

        return stats

    def _load_local_records(self) -> List[Dict]:
        """Load violation records from local JSON file"""
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load local records: {e}")
                return []
        return []

    def _save_local_records(self, records: List[Dict]):
        """Save violation records to local JSON file"""
        try:
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save local records: {e}")

    def get_storage_info(self) -> Dict:
        """Get information about current storage configuration"""
        return {
            'use_firestore': self.use_firestore,
            'writable': self.writable,
            'firestore_available': FIRESTORE_AVAILABLE,
            'collection_name': self.collection_name if self.use_firestore else None,
            'local_path': os.path.abspath(self.violations_dir) if not self.use_firestore else None,
            'project_id': config.GEE_PROJECT if self.use_firestore else None
        }
