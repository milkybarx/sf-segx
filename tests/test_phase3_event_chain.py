import pytest
import numpy as np
from analysis.event_chain import EventChainExtractor
from analysis.cme_geometry import CMEGeometryModel, angular_difference
from analysis.spacecraft_catalog import SpacecraftCatalog
from analysis.space_weather_risk import SpaceWeatherRiskAnalyzer
from analysis.satellite_risk import SatelliteRiskAnalyzer

def test_angular_separation():
    # Test identical points
    assert np.isclose(angular_difference(0.0, 0.0, 0.0, 0.0), 0.0)
    # Test 90 degrees apart
    assert np.isclose(angular_difference(0.0, 0.0, 90.0, 0.0), 90.0)
    # Test 180 degrees apart
    assert np.isclose(angular_difference(0.0, -90.0, 0.0, 90.0), 180.0)

def test_cme_cone_classification():
    geom = CMEGeometryModel(flank_margin=15.0)
    # Inside cone
    exposure, sep = geom.evaluate_exposure(cme_lat=0.0, cme_lon=0.0, cme_half_angle=30.0, target_lat=10.0, target_lon=10.0)
    assert exposure == "INSIDE_CONE"
    
    # Near flank
    exposure, sep = geom.evaluate_exposure(cme_lat=0.0, cme_lon=0.0, cme_half_angle=30.0, target_lat=0.0, target_lon=40.0)
    assert exposure == "NEAR_FLANK"
    
    # Outside
    exposure, sep = geom.evaluate_exposure(cme_lat=0.0, cme_lon=0.0, cme_half_angle=30.0, target_lat=0.0, target_lon=50.0)
    assert exposure == "OUTSIDE"
    
    # Unknown/missing data
    exposure, sep = geom.evaluate_exposure(cme_lat=None, cme_lon=0.0, cme_half_angle=30.0, target_lat=0.0, target_lon=50.0)
    assert exposure == "UNKNOWN"

def test_arrival_estimation():
    geom = CMEGeometryModel()
    # 1000 km/s -> ~1e8 km / 1000 = 1e5 seconds -> ~41 hours
    hrs, method = geom.estimate_arrival_time(1000.0, 1.0)
    assert method == "OUR_GEOMETRIC_ESTIMATE"
    assert 40 < hrs < 45
    
    hrs, method = geom.estimate_arrival_time(None, 1.0)
    assert hrs is None
    assert method == "UNKNOWN"

def test_spacecraft_position_provenance():
    cat = SpacecraftCatalog()
    # Mocking SSCWeb/fallback priority
    pos = cat.get_spacecraft_position("SOHO")
    assert pos["position_source"] == "STATIC_ORBIT_APPROXIMATION"
    
    pos = cat.get_spacecraft_position("UNKNOWN_SAT")
    assert pos["position_source"] == "STATIC_ORBIT_APPROXIMATION" # Default fallback for now, could be UNKNOWN

def test_cme_exposure_calculation():
    cat = SpacecraftCatalog()
    exp = cat.calculate_cme_exposure("SOHO", 0.0, 0.0, 40.0, 1000.0, "2023-01-01T00:00:00Z", nasa_model_impact=None)
    assert exp["exposure_type"] == "INSIDE_CONE"
    assert exp["calculation_method"] == "OUR_GEOMETRIC_ESTIMATE"
    assert exp["event_source"] == "DONKI_OBSERVED"
    
    # Test NASA model precedence
    exp2 = cat.calculate_cme_exposure("SOHO", 90.0, 90.0, 10.0, 1000.0, "2023-01-01T00:00:00Z", 
                                      nasa_model_impact={"spacecraft_id": "SOHO", "arrival_time": "2023-01-03"})
    assert exp2["exposure_type"] == "INSIDE_CONE"
    assert exp2["calculation_method"] == "NASA_MODEL"

def test_risk_score_bounds():
    sw = SpaceWeatherRiskAnalyzer()
    env = sw.evaluate_component_risks(
        flare={"class_type": "X9.0"},
        cme_analysis={"speed": 3000, "is_most_accurate": True},
        sep={"activityID": "SEP"},
        gst={"kp_max": 9.0},
        rbe={"activityID": "RBE"}
    )
    for k, v in env["component_scores"].items():
        assert 0.0 <= v <= 100.0
        
    sat = SatelliteRiskAnalyzer()
    exposure = {"exposure_type": "INSIDE_CONE", "trajectory_source": "STATIC", "calculation_method": "OUR", "event_source": "DONKI"}
    sys_risks = sat.evaluate_subsystem_risks(exposure, env)
    
    for k, v in sys_risks["subsystem_scores"].items():
        assert 0.0 <= v <= 100.0
        
    assert sys_risks["overall_risk_score"] <= 100.0
    assert sys_risks["provenance"]["calculation_method"] == "OUR"

def test_event_chain_extraction(mocker):
    extractor = EventChainExtractor()
    mocker.patch.object(extractor.client, 'get_flares', return_value=[
        {"flare_id": "FLR1", "linked_events": [{"activityID": "CME1"}]}
    ])
    mocker.patch.object(extractor.client, 'get_cmes', return_value=[
        {"cme_id": "CME1", "linked_events": [{"activityID": "SEP1"}, {"activityID": "GST1"}]}
    ])
    mocker.patch.object(extractor.client, 'get_cme_analyses', return_value=[
        {"cme_id": "CME1", "speed": 1500, "is_most_accurate": True, "latitude": 10, "longitude": 20, "half_angle": 45}
    ])
    mocker.patch.object(extractor.client, 'get_seps', return_value=[{"sep_id": "SEP1"}])
    mocker.patch.object(extractor.client, 'get_gst', return_value=[{"gst_id": "GST1"}])
    
    chains = extractor.build_chain("2020-01-01", "2020-01-02")
    assert len(chains) == 1
    c = chains[0]
    assert c["flare_id"] == "FLR1"
    assert c["cme_id"] == "CME1"
    assert c["cme_speed"] == 1500
    assert c["sep_id"] == "SEP1"
    assert c["sep_observed"] == "SEP_OBSERVED"
    assert c["gst_id"] == "GST1"
    assert c["gst_observed"] == "OBSERVED"
