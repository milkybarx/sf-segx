"""NASA CCMC DONKI API Client with local caching and data normalization."""
import os
import json
import time
import urllib3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("donki_client")

# Disable insecure warnings if users use self-signed proxies
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DONKIClient:
    """Client for querying the NASA Space Weather DONKI API with caching and normalization."""
    
    def __init__(self, base_url: str = "https://api.nasa.gov/DONKI", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        # Use env variable NASA_API_KEY if available, else fall back to the provided key or "DEMO_KEY"
        self.api_key = api_key or os.environ.get("NASA_API_KEY", "DEMO_KEY")
        
        # Local paths
        self.data_dir = Path("data/donki")
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with retries and exponential backoff
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _query_api(self, endpoint: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """Directly query the API with standard dates and key."""
        url = self._get_api_url(endpoint)
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key
        }
        
        logger.info(f"Querying DONKI {endpoint} from {start_date} to {end_date}...")
        for attempt in range(3):
            try:
                # Add reasonable timeout of 30s for both connect and read
                response = self.session.get(url, params=params, timeout=(10, 30), verify=True)
                
                if response.status_code == 429:
                    logger.warning("DONKI API Rate limit exceeded. Sleeping 5 seconds...")
                    time.sleep(5)
                    continue
                    
                response.raise_for_status()
                
                # Verify JSON payload
                data = response.json()
                if not isinstance(data, list):
                    if isinstance(data, dict) and "error" in data:
                        logger.error(f"DONKI API Error response: {data}")
                        return None
                    # If it's a single dict, wrap in a list
                    data = [data]
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"DONKI query timeout (attempt {attempt+1}/3). Retrying...")
                time.sleep(2)
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error querying DONKI {endpoint}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error querying DONKI {endpoint}: {e}")
                if attempt == 2:
                    return None
                time.sleep(2)
        return None

    def _get_cached_raw_filename(self, endpoint: str, start_date: str, end_date: str) -> Path:
        return self.raw_dir / f"{endpoint.lower()}_{start_date}_{end_date}.json"

    def _fetch_with_caching(self, endpoint: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch DONKI data, utilizing local raw JSON files as a cache layer."""
        # Clean dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Max window size is 30 days per DONKI query guidelines, but NASA servers often hang on 30 days.
        # Reducing to 7 days to prevent hanging.
        max_chunk = timedelta(days=7)
        
        all_data = []
        current_start = start_dt
        
        while current_start <= end_dt:
            current_end = min(current_start + max_chunk, end_dt)
            curr_start_str = current_start.strftime("%Y-%m-%d")
            curr_end_str = current_end.strftime("%Y-%m-%d")
            
            cache_file = self._get_cached_raw_filename(endpoint, curr_start_str, curr_end_str)
            
            if cache_file.exists():
                logger.info(f"Loading {endpoint} from local cache: {cache_file.name}")
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                    if isinstance(chunk_data, list):
                        all_data.extend(chunk_data)
                except Exception as e:
                    logger.error(f"Error reading cache file {cache_file}: {e}")
                    # If corrupt, re-fetch
                    chunk_data = self._query_api(endpoint, curr_start_str, curr_end_str)
                    if chunk_data is not None:
                        all_data.extend(chunk_data)
                        self._save_raw_cache(cache_file, chunk_data)
            else:
                chunk_data = self._query_api(endpoint, curr_start_str, curr_end_str)
                if chunk_data is not None:
                    all_data.extend(chunk_data)
                    self._save_raw_cache(cache_file, chunk_data)
                
            current_start = current_end + timedelta(days=1)
            
        return all_data

    def _save_raw_cache(self, filename: Path, data: List[Dict[str, Any]]):
        if data is None:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved raw responses to cache: {filename.name}")
        except Exception as e:
            logger.error(f"Failed to write cache file {filename}: {e}")

    def get_flares(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch solar flares (FLR) and normalize to flr.csv."""
        raw_data = self._fetch_with_caching("FLR", start_date, end_date)
        normalized = []
        for item in raw_data:
            normalized.append({
                "flare_id": item.get("flrID") or item.get("flareID"),
                "start_time": item.get("beginTime"),
                "peak_time": item.get("peakTime"),
                "end_time": item.get("endTime"),
                "class_type": item.get("classType"),
                "source_location": item.get("sourceLocation"),
                "active_region": item.get("activeRegionNum"),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("flr.csv", normalized, keys=["flare_id"])
        return normalized

    def get_cmes(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch Coronal Mass Ejections (CME) and normalize to cme.csv."""
        raw_data = self._fetch_with_caching("CME", start_date, end_date)
        normalized = []
        for item in raw_data:
            normalized.append({
                "cme_id": item.get("activityID"),
                "start_time": item.get("startTime"),
                "source_location": item.get("sourceLocation"),
                "active_region": item.get("activeRegionNum"),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("cme.csv", normalized, keys=["cme_id"])
        return normalized

    def get_cme_analyses(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch CME Analysis data and normalize to cme_analysis.csv."""
        # CME Analysis is retrieved either directly (if supported) or nested inside CME events
        # Let's do both to ensure safety. First, check direct query:
        raw_data = self._fetch_with_caching("CMEAnalysis", start_date, end_date)
        
        # If CMEAnalysis returned empty or is not fully supported as a standalone date-search in DONKI,
        # we can extract it from CME events in the same time frame!
        extracted = []
        
        # Helper to extract analyses from a CME or CMEAnalysis object
        def parse_analysis_item(a: Dict[str, Any], cme_id: Optional[str] = None) -> Dict[str, Any]:
            return {
                "analysis_id": a.get("time21_5", "") + "_" + (cme_id or "unknown"),
                "cme_id": cme_id,
                "time21_5": a.get("time21_5"),
                "latitude": a.get("latitude"),
                "longitude": a.get("longitude"),
                "half_angle": a.get("halfAngle"),
                "speed": a.get("speed"),
                "type": a.get("type"),
                "is_most_accurate": a.get("isMostAccurate"),
                "linked_events": json.dumps(a.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            }

        # Parse standalone CMEAnalysis if present
        for item in raw_data:
            #Standalones might be formatted differently or contain a link to cme
            cme_id = item.get("associatedCME") or item.get("activityID")
            extracted.append(parse_analysis_item(item, cme_id))
            
        # Also parse from CME records in this window to guarantee coverage
        cme_raw = self._fetch_with_caching("CME", start_date, end_date)
        for cme in cme_raw:
            cme_id = cme.get("activityID")
            analyses = cme.get("cmeAnalyses", []) or []
            for a in analyses:
                extracted.append(parse_analysis_item(a, cme_id))
                
        # Deduplicate by analysis_id
        df = pd.DataFrame(extracted)
        if not df.empty:
            df = df.drop_duplicates(subset=["analysis_id"])
            extracted = df.to_dict(orient="records")
            
        self._update_csv("cme_analysis.csv", extracted, keys=["analysis_id"])
        return extracted

    def get_seps(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch Solar Energetic Particle (SEP) events and normalize to sep.csv."""
        raw_data = self._fetch_with_caching("SEP", start_date, end_date)
        normalized = []
        for item in raw_data:
            normalized.append({
                "sep_id": item.get("activityID"),
                "event_time": item.get("eventTime"),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("sep.csv", normalized, keys=["sep_id"])
        return normalized

    def get_ips(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch Interplanetary Shocks (IPS) and normalize to ips.csv."""
        raw_data = self._fetch_with_caching("IPS", start_date, end_date)
        normalized = []
        for item in raw_data:
            normalized.append({
                "ips_id": item.get("activityID"),
                "event_time": item.get("eventTime"),
                "location": item.get("location"),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("ips.csv", normalized, keys=["ips_id"])
        return normalized

    def get_gst(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch Geomagnetic Storms (GST) and normalize to gst.csv."""
        raw_data = self._fetch_with_caching("GST", start_date, end_date)
        normalized = []
        for item in raw_data:
            # Extract Kp values if present
            kp_list = item.get("allKp", []) or []
            kp_max = max([float(k.get("kpRating", 0)) for k in kp_list], default=0.0)
            
            normalized.append({
                "gst_id": item.get("activityID"),
                "start_time": item.get("startTime"),
                "kp_max": kp_max,
                "kp_ratings": json.dumps(kp_list),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("gst.csv", normalized, keys=["gst_id"])
        return normalized

    def get_rbe(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch Radiation Belt Enhancements (RBE) and normalize to rbe.csv."""
        raw_data = self._fetch_with_caching("RBE", start_date, end_date)
        normalized = []
        for item in raw_data:
            normalized.append({
                "rbe_id": item.get("activityID"),
                "start_time": item.get("startTime"),
                "linked_events": json.dumps(item.get("linkedEvents", [])),
                "last_updated": datetime.now().isoformat()
            })
        self._update_csv("rbe.csv", normalized, keys=["rbe_id"])
        return normalized

    def _update_csv(self, filename: str, records: List[Dict[str, Any]], keys: List[str]):
        """Append or update records in a CSV file, avoiding duplicates by keys."""
        if not records:
            return
            
        filepath = self.data_dir / filename
        new_df = pd.DataFrame(records)
        
        if filepath.exists():
            try:
                old_df = pd.read_csv(filepath)
                # Combine old and new, dropping duplicates, keeping the newest record
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=keys, keep="last")
                combined.to_csv(filepath, index=False)
            except Exception as e:
                logger.error(f"Error updating CSV {filename}: {e}")
                new_df.to_csv(filepath, index=False)
        else:
            try:
                new_df.to_csv(filepath, index=False)
                logger.info(f"Created new CSV file: {filepath}")
            except Exception as e:
                logger.error(f"Error creating CSV {filename}: {e}")
