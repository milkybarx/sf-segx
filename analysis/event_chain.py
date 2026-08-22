"""DONKI Event Chain Extraction and Normalization."""
import json
import logging
from typing import Dict, List, Any, Optional
from .donki.donki_client import DONKIClient

logger = logging.getLogger("event_chain")

class EventChainExtractor:
    """Extracts and normalizes the full DONKI event chain (FLR -> CME -> SEP/GST)."""
    
    def __init__(self, donki_client: Optional[DONKIClient] = None):
        self.client = donki_client or DONKIClient()

    def _find_linked_event(self, linked_events: List[Dict], activity_id_prefix: str) -> Optional[str]:
        """Find a linked event ID matching the prefix (e.g. 'CME', 'SEP')."""
        if not linked_events:
            return None
        for evt in linked_events:
            evt_id = evt.get("activityID", "")
            if activity_id_prefix in evt_id:
                return evt_id
        return None

    def build_chain(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Build the full event chains for a date range."""
        flares = self.client.get_flares(start_date, end_date)
        cmes = {c["cme_id"]: c for c in self.client.get_cmes(start_date, end_date)}
        cme_analyses = self.client.get_cme_analyses(start_date, end_date)
        seps = {s["sep_id"]: s for s in self.client.get_seps(start_date, end_date)}
        gsts = {g["gst_id"]: g for g in self.client.get_gst(start_date, end_date)}
        
        # Group analyses by CME ID
        analyses_by_cme = {}
        for a in cme_analyses:
            cid = a.get("cme_id")
            if cid:
                if cid not in analyses_by_cme:
                    analyses_by_cme[cid] = []
                analyses_by_cme[cid].append(a)
                
        chains = []
        for flr in flares:
            flr_id = flr["flare_id"]
            # DONKI linkedEvents can be string (JSON) or list
            links = flr.get("linked_events")
            if isinstance(links, str):
                try:
                    links = json.loads(links)
                except json.JSONDecodeError:
                    links = []
            
            # Find associated CME
            cme_id = self._find_linked_event(links, "CME")
            cme = cmes.get(cme_id) if cme_id else None
            
            # Find associated SEP from flare links or CME links
            sep_id = self._find_linked_event(links, "SEP")
            if not sep_id and cme:
                cme_links = cme.get("linked_events", [])
                if isinstance(cme_links, str):
                    try:
                        cme_links = json.loads(cme_links)
                    except:
                        cme_links = []
                sep_id = self._find_linked_event(cme_links, "SEP")
            
            # Find associated GST from CME links
            gst_id = None
            if cme:
                cme_links = cme.get("linked_events", [])
                if isinstance(cme_links, str):
                    try:
                        cme_links = json.loads(cme_links)
                    except:
                        cme_links = []
                gst_id = self._find_linked_event(cme_links, "GST")
                
            # Get best CMEAnalysis
            best_analysis = None
            nasa_model_impacts = []
            if cme_id and cme_id in analyses_by_cme:
                # Prefer most accurate
                ans = analyses_by_cme[cme_id]
                best_analysis = next((a for a in ans if a.get("is_most_accurate")), ans[0] if ans else None)
                
                # Check for WSA-ENLIL impacts in DONKI analysis linked events / ENLIL data
                # Typically DONKI stores ENLIL simulations, but we will mock extraction if not directly in CMEAnalysis
                # Note: real extraction might need self.client.get_wsa_enlil()
                pass
                
            chains.append({
                "flare_id": flr_id,
                "cme_id": cme_id,
                "cme_speed": best_analysis.get("speed") if best_analysis else None,
                "cme_latitude": best_analysis.get("latitude") if best_analysis else None,
                "cme_longitude": best_analysis.get("longitude") if best_analysis else None,
                "cme_half_angle": best_analysis.get("half_angle") if best_analysis else None,
                "cme_type": best_analysis.get("type") if best_analysis else None,
                "sep_id": sep_id,
                "sep_observed": "SEP_OBSERVED" if sep_id else "SEP_NOT_OBSERVED",
                "gst_id": gst_id,
                "gst_observed": "OBSERVED" if gst_id else "NOT_OBSERVED"
            })
            
        return chains
