"""
Violation Recorder - Records and manages violation history using Google Cloud Firestore
Maps are stored in Google Cloud Storage
Falls back to local storage if cloud services are not available
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

# Try to import Cloud Storage
try:
    from google.cloud import storage as cloud_storage
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False
    logger.warning("google-cloud-storage not installed. Map storage disabled.")


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
        self.storage_client = None
        self.bucket = None
        self.bucket_name = f"{config.GEE_PROJECT}-violation-maps"
        self.collection_name = "violations"
        self.use_firestore = False
        self.use_cloud_storage = False
        self.writable = False

        # Try to initialize Firestore first
        if FIRESTORE_AVAILABLE:
            self._init_firestore()

        # Try to initialize Cloud Storage for maps
        if CLOUD_STORAGE_AVAILABLE and self.use_firestore:
            self._init_cloud_storage()

        # Fallback to local storage if Firestore not available
        if not self.use_firestore:
            self._init_local_storage()

        logger.info(f"ViolationRecorder initialized. Firestore: {self.use_firestore}, CloudStorage: {self.use_cloud_storage}, Writable: {self.writable}")

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

    def _init_cloud_storage(self):
        """Initialize Google Cloud Storage for map files"""
        try:
            import streamlit as st
            from google.oauth2 import service_account

            # Check for service account credentials
            if hasattr(st, 'secrets') and 'GEE_SERVICE_ACCOUNT' in st.secrets:
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
                    scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
                )

                self.storage_client = cloud_storage.Client(
                    project=st.secrets.get("GEE_PROJECT_ID", config.GEE_PROJECT),
                    credentials=credentials
                )
            else:
                self.storage_client = cloud_storage.Client(project=config.GEE_PROJECT)

            # Try to get or create bucket
            try:
                self.bucket = self.storage_client.get_bucket(self.bucket_name)
                logger.info(f"Cloud Storage bucket found: {self.bucket_name}")
            except Exception:
                # Bucket doesn't exist, try to create it
                try:
                    self.bucket = self.storage_client.create_bucket(
                        self.bucket_name,
                        location="europe-west1"  # Same as Firestore
                    )
                    logger.info(f"Cloud Storage bucket created: {self.bucket_name}")
                except Exception as create_err:
                    logger.warning(f"Could not create bucket: {create_err}")
                    logger.info("Map storage will be disabled")
                    return

            self.use_cloud_storage = True
            logger.info("Cloud Storage initialized successfully")

        except Exception as e:
            logger.warning(f"Cloud Storage initialization failed: {e}")
            self.use_cloud_storage = False

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

    def save_violation(self, violation_data: Dict, analysis: str,
                      map_html_path: Optional[str] = None) -> Optional[str]:
        """
        Save a violation record with all details

        Args:
            violation_data: Violation information dictionary
            analysis: AI analysis text
            map_html_path: Path to saved HTML map (not stored in Firestore)

        Returns:
            Violation ID (timestamp-based) or None if failed
        """
        if not self.writable:
            logger.error("Cannot save violation: storage is not writable")
            return None

        try:
            logger.info(f"Saving violation for {violation_data.get('gas')} in {violation_data.get('city')}")

            # Generate unique violation ID based on timestamp
            ksa_tz = pytz.timezone(config.TIMEZONE)
            now = datetime.now(ksa_tz)
            violation_id = now.strftime("%Y%m%d_%H%M%S")
            gas = violation_data['gas']
            city = violation_data['city']

            # Create full ID: YYYYMMDD_HHMMSS_CITY_GAS
            full_id = f"{violation_id}_{city}_{gas}"

            # Prepare record (Firestore-compatible - no complex objects)
            record = {
                'id': full_id,
                'timestamp': now.isoformat(),
                'timestamp_ksa': now.strftime("%Y-%m-%d %H:%M:%S KSA"),
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
            if map_html_path and os.path.exists(map_html_path) and self.use_cloud_storage:
                map_url = self._upload_map_to_cloud_storage(full_id, map_html_path)
                if map_url:
                    record['map_url'] = map_url
                    logger.info(f"Map uploaded to Cloud Storage: {map_url}")

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

    def _upload_map_to_cloud_storage(self, full_id: str, map_html_path: str) -> Optional[str]:
        """Upload map HTML file to Google Cloud Storage"""
        try:
            if not self.bucket:
                logger.warning("Cloud Storage bucket not available")
                return None

            # Create blob name
            blob_name = f"maps/{full_id}_map.html"
            blob = self.bucket.blob(blob_name)

            # Upload the file
            blob.upload_from_filename(map_html_path, content_type='text/html')

            # Make publicly readable
            blob.make_public()

            # Get public URL
            public_url = blob.public_url
            logger.info(f"Map uploaded to Cloud Storage: {public_url}")
            return public_url

        except Exception as e:
            logger.error(f"Failed to upload map to Cloud Storage: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            'use_cloud_storage': self.use_cloud_storage,
            'writable': self.writable,
            'firestore_available': FIRESTORE_AVAILABLE,
            'cloud_storage_available': CLOUD_STORAGE_AVAILABLE,
            'collection_name': self.collection_name if self.use_firestore else None,
            'bucket_name': self.bucket_name if self.use_cloud_storage else None,
            'local_path': os.path.abspath(self.violations_dir) if not self.use_firestore else None,
            'project_id': config.GEE_PROJECT if self.use_firestore else None
        }
