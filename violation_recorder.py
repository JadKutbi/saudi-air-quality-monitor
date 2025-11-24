"""
Violation Recorder - Records and manages violation history with heatmaps
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

class ViolationRecorder:
    """Records violations with maps and enables historical viewing"""

    def __init__(self, violations_dir: str = "violations"):
        self.violations_dir = violations_dir
        self.maps_dir = os.path.join(violations_dir, "maps")
        self.records_file = os.path.join(violations_dir, "violation_records.json")

        # Create directories if they don't exist
        os.makedirs(self.violations_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)

        logger.info(f"ViolationRecorder initialized: {violations_dir}")

    def save_violation(self, violation_data: Dict, analysis: str,
                      map_html_path: Optional[str] = None) -> str:
        """
        Save a violation record with all details

        Args:
            violation_data: Violation information dictionary
            analysis: AI analysis text
            map_html_path: Path to saved HTML map (will be converted to PNG)

        Returns:
            Violation ID (timestamp-based)
        """
        try:
            # Generate unique violation ID based on timestamp
            ksa_tz = pytz.timezone(config.TIMEZONE)
            now = datetime.now(ksa_tz)
            violation_id = now.strftime("%Y%m%d_%H%M%S")
            gas = violation_data['gas']
            city = violation_data['city']

            # Create full ID: YYYYMMDD_HHMMSS_CITY_GAS
            full_id = f"{violation_id}_{city}_{gas}"

            # Prepare record
            record = {
                'id': full_id,
                'timestamp': now.isoformat(),
                'timestamp_ksa': now.strftime("%Y-%m-%d %H:%M:%S KSA"),
                'city': city,
                'gas': gas,
                'gas_name': violation_data['gas_name'],
                'max_value': violation_data['max_value'],
                'threshold': violation_data['threshold'],
                'unit': violation_data['unit'],
                'severity': violation_data['severity'],
                'percentage_over': violation_data['percentage_over'],
                'hotspot': violation_data.get('hotspot'),
                'wind': violation_data.get('wind'),
                'nearby_factories': violation_data.get('nearby_factories', []),
                'ai_analysis': analysis,
                'map_image': None,
                'map_html': None
            }

            # Save map as image if HTML provided
            if map_html_path and os.path.exists(map_html_path):
                # Copy HTML to violations folder
                map_html_filename = f"{full_id}_map.html"
                map_html_dest = os.path.join(self.maps_dir, map_html_filename)

                import shutil
                shutil.copy(map_html_path, map_html_dest)
                record['map_html'] = map_html_filename

                # Try to convert to PNG
                map_png_filename = f"{full_id}_map.png"
                map_png_path = os.path.join(self.maps_dir, map_png_filename)

                # Use visualizer to convert HTML to PNG
                from visualizer import MapVisualizer
                visualizer = MapVisualizer()

                if visualizer.save_map_as_image(map_html_dest, map_png_path,
                                               width=1400, height=900):
                    record['map_image'] = map_png_filename
                    logger.info(f"Saved violation map image: {map_png_filename}")

            # Load existing records
            records = self._load_records()

            # Add new record
            records.append(record)

            # Save updated records
            self._save_records(records)

            logger.info(f"Violation recorded: {full_id}")
            return full_id

        except Exception as e:
            logger.error(f"Failed to save violation: {e}")
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
        records = self._load_records()

        # Filter by city if specified
        if city:
            records = [r for r in records if r['city'] == city]

        # Filter by gas if specified
        if gas:
            records = [r for r in records if r['gas'] == gas]

        # Sort by timestamp (newest first)
        records.sort(key=lambda x: x['timestamp'], reverse=True)

        # Apply limit if specified
        if limit:
            records = records[:limit]

        return records

    def get_violation_by_id(self, violation_id: str) -> Optional[Dict]:
        """Get a specific violation by ID"""
        records = self._load_records()
        for record in records:
            if record['id'] == violation_id:
                return record
        return None

    def get_violation_map_path(self, violation_id: str,
                              image: bool = True) -> Optional[str]:
        """
        Get path to violation map (HTML or PNG)

        Args:
            violation_id: Violation ID
            image: If True, return PNG path; if False, return HTML path

        Returns:
            Full path to map file, or None if not found
        """
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
        """Delete a violation record and its associated files"""
        try:
            records = self._load_records()
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
            records = [r for r in records if r['id'] != violation_id]
            self._save_records(records)

            logger.info(f"Deleted violation: {violation_id}")
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
                'by_city': {}
            }

        stats = {
            'total_violations': len(records),
            'by_gas': {},
            'by_severity': {},
            'by_city': {},
            'date_range': {
                'oldest': records[-1]['timestamp_ksa'],
                'newest': records[0]['timestamp_ksa']
            }
        }

        # Count by gas
        for record in records:
            gas = record['gas']
            stats['by_gas'][gas] = stats['by_gas'].get(gas, 0) + 1

        # Count by severity
        for record in records:
            severity = record['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

        # Count by city
        for record in records:
            city = record['city']
            stats['by_city'][city] = stats['by_city'].get(city, 0) + 1

        return stats

    def _load_records(self) -> List[Dict]:
        """Load violation records from JSON file"""
        if os.path.exists(self.records_file):
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load records: {e}")
                return []
        return []

    def _save_records(self, records: List[Dict]):
        """Save violation records to JSON file"""
        try:
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save records: {e}")
