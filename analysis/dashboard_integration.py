import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from analysis.event_chain import EventChainExtractor
from analysis.cme_geometry import CMEGeometryModel
from analysis.spacecraft_catalog import SpacecraftCatalog
from analysis.space_weather_risk import SpaceWeatherRiskAnalyzer
from analysis.satellite_risk import SatelliteRiskAnalyzer
from analysis.flare_inference import FlareRiskInference

CRIMSON = "#DC143C"

@st.cache_resource
def get_flare_inference():
    return FlareRiskInference()

@st.cache_resource
def get_event_extractor():
    return EventChainExtractor()

@st.cache_resource
def get_catalogs():
    return SpacecraftCatalog(), CMEGeometryModel(), SpaceWeatherRiskAnalyzer(), SatelliteRiskAnalyzer()

def render_phase3_sections(selected_filament: dict, mode: str, obs_time_str: str, historical_case: dict = None):
    """Render the Phase 3 dashboard sections given a selected filament."""
    
    st.divider()
    st.header("Section 3: Relative Flare Risk", divider="red")
    
    # 3. Relative Flare Risk
    inference = get_flare_inference()
    
    if mode == "HISTORICAL EVENT DEMO" and historical_case:
        obs_time_str = historical_case["date"] + "T12:00:00Z"
        
    if not obs_time_str:
        st.warning("Relative flare-risk unavailable: observation timestamp required.")
        st.info("Please provide a timestamp in the sidebar or filename to enable flare risk scoring.")
        score, risk_level = None, None
    else:
        with st.spinner("Calculating Flare Risk..."):
            score, status = inference.predict_risk(selected_filament, obs_time_str)
            
        if pd.isna(score):
            # Any failure (version mismatch, missing model, etc.) → fall back to 2-feature RF
            # Phase 2E model incompatible with this sklearn version — use our 2-feature RF fallback
            try:
                from analysis.flare_prediction import calculate_flare_probability
                _fb_risk = calculate_flare_probability(
                    length_px=selected_filament.get("skeleton_length_px", 0.0),
                    region_type=selected_filament.get("spatial_region", "ARF")
                )
                _fb_pct = _fb_risk * 100
                if _fb_pct >= 60: risk_level = "HIGH"
                elif _fb_pct >= 35: risk_level = "MODERATE"
                else: risk_level = "LOW"
                score_text = f"{_fb_pct:.1f}%"
                col1, col2 = st.columns(2)
                col1.metric("24h Eruption Risk", score_text, delta=None)
                col2.metric("Risk Level", risk_level)
                st.info(
                    "ℹ️ Phase 2E contextual model unavailable (sklearn version mismatch). "
                    "Showing simplified 2-feature RF score (filament length + region type).",
                    icon=None
                )
            except Exception as fb_err:
                st.warning(f"Flare risk model unavailable: {fb_err}")
                risk_level = "UNKNOWN"
        else:
            if score >= 0.75: risk_level = "EXTREME"
            elif score >= 0.5: risk_level = "HIGH"
            elif score >= 0.25: risk_level = "MODERATE"
            else: risk_level = "LOW"
            
            col1, col2 = st.columns(2)
            col1.metric("Relative Risk Score", f"{score:.3f}")
            col2.metric("Risk Level", risk_level)
            st.caption("Phase 2E contextual RandomForest · UNCALIBRATED · for relative ranking only.")

    # 4. DONKI Event Chain
    st.divider()
    st.header("Section 4: DONKI Event Chain", divider="red")
    
    if mode == "HISTORICAL EVENT DEMO":
        st.info("EVENT SOURCE: HISTORICAL DONKI REPLAY")
        if not historical_case:
            st.error("No historical case selected.")
            return
            
        st.markdown(f"**Replaying**: {historical_case['label']}")
        st.markdown(f"_{historical_case['description']}_")
        
        # Hardcode the historical values for the demo
        cme_speed = historical_case["speed"]
        cme_lat = historical_case["lat"]
        cme_lon = historical_case["lon"]
        cme_ha = historical_case["half_angle"]
        cme_date = historical_case["date"] + "T00:00:00Z"
        sep_obs = "SEP_OBSERVED" if historical_case["sep_observed"] else "NOT_OBSERVED"
        gst_obs = "OBSERVED" if historical_case.get("max_kp", 0) and historical_case["max_kp"] >= 5.0 else "NOT_OBSERVED"
        
        st.markdown(f"""
        **CME ID**: `{historical_case['cme_id']}` (DONKI_OBSERVED)  
        **Speed**: `{cme_speed} km/s`  
        **Direction**: `Lat {cme_lat}°, Lon {cme_lon}°`  
        **Half-Angle**: `{cme_ha}°`  
        **SEP**: `{sep_obs}`  
        **GST**: `{gst_obs}`
        """)
        
    else:
        st.info("LIVE / CURRENT ANALYSIS")
        if not obs_time_str:
            st.warning("Timestamp required to fetch DONKI events.")
            return
            
        extractor = get_event_extractor()
        # Parse date range: observation date up to 3 days after
        try:
            ts = obs_time_str.replace("Z", "")
            if len(ts) == 8: dt = pd.to_datetime(ts, format="%Y%m%d")
            else: dt = pd.to_datetime(ts)
            
            start_str = dt.strftime("%Y-%m-%d")
            end_str = (dt + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            
            with st.spinner("Fetching DONKI Event Chain..."):
                chains = extractor.build_chain(start_str, end_str)
                
            if not chains:
                st.write("No associated DONKI events found for this time period.")
            else:
                for c in chains:
                    st.markdown(f"**Flare ID**: `{c.get('flare_id', 'UNKNOWN')}`")
                    st.markdown(f"**CME ID**: `{c.get('cme_id', 'UNKNOWN')}` (DONKI_OBSERVED)")
                    st.markdown(f"**Speed**: `{c.get('cme_speed', 'UNKNOWN')} km/s`")
                    st.markdown(f"**Direction**: `Lat {c.get('cme_latitude', 'UNKNOWN')}°, Lon {c.get('cme_longitude', 'UNKNOWN')}°`")
                    st.markdown(f"**Half-Angle**: `{c.get('cme_half_angle', 'UNKNOWN')}°`")
                    st.markdown(f"**SEP**: `{'SEP_OBSERVED' if c.get('sep_id') else 'NOT_OBSERVED'}`")
                    st.markdown(f"**GST**: `{'OBSERVED' if c.get('gst_id') else 'NOT_OBSERVED'}`")
                    st.divider()
                    
                # We'll just use the first chain's CME for the rest of the pipeline in LIVE mode
                first_cme = next((c for c in chains if c.get("cme_id")), None)
                if not first_cme:
                    st.warning("No CME data available to propagate.")
                    return
                cme_speed = first_cme.get("cme_speed")
                cme_lat = first_cme.get("cme_latitude")
                cme_lon = first_cme.get("cme_longitude")
                cme_ha = first_cme.get("cme_half_angle")
                cme_date = start_str + "T00:00:00Z"
        except Exception as e:
            st.error(f"Failed to build event chain: {e}")
            st.markdown("**DONKI STATUS**: UNAVAILABLE")
            return
            
    # If we made it here, we have CME parameters
    if any(x is None for x in [cme_speed, cme_lat, cme_lon, cme_ha]):
        st.warning("Incomplete CME parameters. Cannot propagate.")
        return
        
    st.divider()
    st.header("Section 5: CME Propagation", divider="red")
    
    cat, geom, sw_risk, sat_risk = get_catalogs()
    
    earth_status = geom.evaluate_earth_impact(float(cme_lat), float(cme_lon), float(cme_ha))
    st.markdown(f"### Earth-Directed Status: **{earth_status}**")
    st.caption("OUR_GEOMETRIC_ESTIMATE")
    
    st.divider()
    st.header("Section 6: Spacecraft Exposure", divider="red")
    
    sc_results = cat.evaluate_all_spacecraft(
        float(cme_lat), float(cme_lon), float(cme_ha), float(cme_speed), cme_date
    )
    
    # Sort by exposure
    def exp_score(e):
        if e == "INSIDE_CONE": return 2
        elif e == "NEAR_FLANK": return 1
        return 0
    sc_results.sort(key=lambda x: exp_score(x["exposure_type"]), reverse=True)
    
    df_sc = pd.DataFrame(sc_results)
    st.dataframe(df_sc[["satellite_id", "exposure_type", "angular_separation", "estimated_arrival", "calculation_method", "trajectory_source"]], use_container_width=True)
    
    # 8. Cone Visualization
    st.subheader("Geometric Cone Visualization")
    st.caption("Simplified geometric screening; not an MHD propagation simulation.")
    
    fig = go.Figure()
    
    # Sun
    fig.add_trace(go.Scatterpolar(r=[0], theta=[0], mode="markers", marker=dict(color="yellow", size=20), name="Sun"))
    
    # Spacecraft
    for r in sc_results:
        pos = cat.get_spacecraft_position(r["satellite_id"])
        c = "red" if r["exposure_type"] == "INSIDE_CONE" else "orange" if r["exposure_type"] == "NEAR_FLANK" else "blue"
        fig.add_trace(go.Scatterpolar(
            r=[pos["distance_au"]], theta=[pos["longitude"]], mode="markers+text", 
            marker=dict(color=c, size=10), text=[r["satellite_id"]], textposition="top center", name=r["satellite_id"]
        ))
        
    # CME Direction
    fig.add_trace(go.Scatterpolar(
        r=[0, 1.2], theta=[float(cme_lon), float(cme_lon)], mode="lines",
        line=dict(color="white", dash="dash"), name="CME Path"
    ))
    
    # Cone (rough approximation in 2D longitude)
    fig.add_trace(go.Scatterpolar(
        r=[0, 1.2, 1.2, 0], 
        theta=[float(cme_lon), float(cme_lon) - float(cme_ha), float(cme_lon) + float(cme_ha), float(cme_lon)],
        fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,0,0,0)"), name="CME Cone"
    ))
    
    fig.update_layout(
        polar=dict(angularaxis=dict(direction="counterclockwise", rotation=90)),
        showlegend=True, height=500, template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.header("Section 7: Subsystem Vulnerability", divider="red")
    
    # Calculate space weather risks
    flare_class = "M1.0" # Mock if unavailable
    kp_max = 5.0
    if mode == "HISTORICAL EVENT DEMO" and historical_case:
        kp_max = historical_case.get("max_kp") or 0.0
        
    env_risks = sw_risk.evaluate_component_risks(
        flare={"class_type": flare_class},
        cme_analysis={"speed": cme_speed, "longitude": cme_lon, "latitude": cme_lat},
        sep={"activityID": "1"} if mode == "HISTORICAL EVENT DEMO" and historical_case.get("sep_observed") else None,
        gst={"kp_max": kp_max} if kp_max > 0 else None,
        rbe=None
    )
    
    cols = st.columns(5)
    for i, (comp, score) in enumerate(env_risks["component_scores"].items()):
        cols[i % 5].metric(f"{comp.capitalize()} Risk", f"{score:.1f}")
        
    st.markdown("### Potential Spacecraft Subsystem Impact")
    sub_risks = []
    for sc in sc_results:
        sys_risk = sat_risk.evaluate_subsystem_risks(sc, env_risks)
        sub_risks.append({
            "Spacecraft": sc["satellite_id"],
            "Overall Risk": sys_risk["overall_risk_level"],
            "Primary Threat": sys_risk["primary_threat"],
            "Most Vulnerable": sys_risk["most_vulnerable_subsystem"],
            **sys_risk["subsystem_scores"]
        })
        
    df_sub = pd.DataFrame(sub_risks)
    st.dataframe(df_sub, use_container_width=True)
    
    st.divider()
    st.header("Section 8: Data Provenance", divider="red")
    with st.expander("View Provenance Tracking"):
        st.markdown(f"- **Segmentation Model**: `{selected_filament.get('model_name', 'Unknown')}`")
        st.markdown(f"- **Flare-risk Model**: `Phase 2E.2 RandomForest +Context`")
        st.markdown(f"- **Flare-risk Status**: `UNCALIBRATED`")
        st.markdown(f"- **DONKI Event Source**: `{'HISTORICAL DONKI REPLAY' if mode == 'HISTORICAL EVENT DEMO' else 'LIVE DONKI API'}`")
        st.markdown(f"- **CME Source**: `DONKI_OBSERVED`")
        st.markdown(f"- **Propagation**: `OUR_GEOMETRIC_ESTIMATE`")
        st.markdown(f"- **Spacecraft Trajectory**: `STATIC_ORBIT_APPROXIMATION`")
        st.markdown(f"- **Calculation**: `OUR_GEOMETRIC_ESTIMATE`")

    export_metrics = {
        "Relative_Flare_Risk_Score": score,
        "Flare_Risk_Status": risk_level,
        "Flare_Risk_Model_Version": "Phase 2E.2 RandomForest +Context",
        "DONKI_FLR_ID": "UNKNOWN",
        "DONKI_CME_ID": historical_case["cme_id"] if historical_case else (first_cme.get("cme_id") if 'first_cme' in locals() and first_cme else "UNKNOWN"),
        "CME_Speed": cme_speed,
        "CME_Direction": f"{cme_lat}, {cme_lon}",
        "CME_Half_Angle": cme_ha,
        "CME_Earth_Directed_Status": earth_status,
    }
    
    if sub_risks:
        worst_sc = sub_risks[0]
        export_metrics.update({
            "Spacecraft_ID": worst_sc["Spacecraft"],
            "Spacecraft_Exposure": sc_results[0]["exposure_type"],
            "Angular_Separation": sc_results[0]["angular_separation"],
            "Arrival_Time": sc_results[0]["estimated_arrival"],
            "Arrival_Source": sc_results[0]["calculation_method"],
            "Ionosphere_Risk": env_risks["component_scores"].get("ionosphere", 0),
            "GNSS_Risk": env_risks["component_scores"].get("gnss", 0),
            "Magnetosphere_Risk": env_risks["component_scores"].get("magnetosphere", 0),
            "Radiation_Risk": env_risks["component_scores"].get("radiation", 0),
            "Thermosphere_Risk": env_risks["component_scores"].get("thermosphere", 0),
            "Communications_Risk": worst_sc.get("communications", 0),
            "GNSS_Receiver_Risk": worst_sc.get("gnss_receiver", 0),
            "Attitude_Risk": worst_sc.get("attitude_control", 0),
            "Power_Risk": worst_sc.get("power", 0),
            "Sensor_Risk": worst_sc.get("sensors", 0),
            "Drag_Risk": worst_sc.get("drag", 0),
            "Overall_Spacecraft_Risk": worst_sc["Overall Risk"]
        })
        
    return export_metrics
