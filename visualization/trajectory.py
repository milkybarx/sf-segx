import numpy as np
import plotly.graph_objects as go

def calculate_cme_trajectory(image_width: int, centroid_x: float):
    """
    Estimates the solar longitude of a filament based on its X-coordinate on the solar disk.
    Assumes the image is a full-disk solar image where width roughly equals the solar diameter.
    
    Args:
        image_width: Width of the image in pixels.
        centroid_x: The X coordinate of the filament's centroid.
        
    Returns:
        theta: The estimated solar longitude in radians (0 means center/Earth-directed).
    """
    center_x = image_width / 2.0
    # Normalize X to [-1, 1], where 0 is center, -1 is left limb, 1 is right limb
    nx = (centroid_x - center_x) / (center_x)
    
    # Clip to valid range for arcsin to avoid math errors if centroid is slightly off disk
    nx = np.clip(nx, -1.0, 1.0)
    
    # Estimate longitude (theta)
    theta = np.arcsin(nx)
    return theta

def plot_earth_impact_trajectory(image_width: int, filament: dict) -> go.Figure:
    """
    Generates a high-quality, aerospace-style top-down Plotly visualization 
    of the inner solar system, showing a potential Coronal Mass Ejection 
    (CME) trajectory originating from a filament.
    """
    centroid_x = filament.get("centroid", {}).get("x", image_width / 2.0)
    theta = calculate_cme_trajectory(image_width, centroid_x)
    
    # CME parameters
    cme_half_width = np.radians(25)  # 50-degree wide CME cone
    cme_radius = 1.35 # Extend past Earth (which is at 1.0 AU)
    
    # Check if Earth (at angle 0) is inside the CME cone
    is_earth_impact = abs(theta) <= cme_half_width
    
    fig = go.Figure()

    # --- 1. Plot Orbital Rings (0.5 AU, 1.0 AU) ---
    orbit_angles = np.linspace(0, 2 * np.pi, 150)
    for au in [0.5, 1.0]:
        dash_style = "dash" if au == 1.0 else "dot"
        opacity = 0.4 if au == 1.0 else 0.2
        fig.add_trace(go.Scatter(
            x=au * np.cos(orbit_angles),
            y=au * np.sin(orbit_angles),
            mode="lines",
            line=dict(color=f"rgba(255, 255, 255, {opacity})", width=1, dash=dash_style),
            name=f"{au} AU Orbit",
            hoverinfo="none",
            showlegend=(au == 1.0)
        ))

    # --- 2. Plot the CME Cone (Aerospace Gradient Style) ---
    # We layer 3 polygons to create a glowing gradient effect
    layers = [
        {"width_mult": 1.0, "alpha": 0.2, "line": False},
        {"width_mult": 0.7, "alpha": 0.35, "line": False},
        {"width_mult": 0.3, "alpha": 0.6, "line": True}
    ]
    
    base_color = "220, 20, 60" if is_earth_impact else "46, 204, 113"
    
    for layer in layers:
        w = cme_half_width * layer["width_mult"]
        arc_angles = np.linspace(theta - w, theta + w, 30)
        arc_x = cme_radius * np.cos(arc_angles)
        arc_y = cme_radius * np.sin(arc_angles)
        
        poly_x = [0.0] + list(arc_x) + [0.0]
        poly_y = [0.0] + list(arc_y) + [0.0]
        
        line_dict = dict(color=f"rgba({base_color}, 1.0)", width=2) if layer["line"] else dict(width=0)
        
        fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y,
            fill="toself",
            fillcolor=f"rgba({base_color}, {layer['alpha']})",
            line=line_dict,
            name="CME Trajectory" if layer["line"] else None,
            showlegend=layer["line"],
            hoverinfo="none"
        ))
    
    # --- 3. Plot the Sun with glowing corona ---
    # Corona layers
    for s_size, s_alpha in [(80, 0.1), (60, 0.3), (40, 1.0)]:
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(size=s_size, color=f"rgba(255, 165, 0, {s_alpha})"),
            name="Sun" if s_alpha == 1.0 else None,
            showlegend=(s_alpha == 1.0),
            hoverinfo="name" if s_alpha == 1.0 else "none"
        ))
    
    # --- 4. Plot Earth ---
    fig.add_trace(go.Scatter(
        x=[1.0], y=[0.0],
        mode="markers",
        marker=dict(size=18, color="#3498db", line=dict(color="#2ecc71", width=3)),
        name="Earth",
        hoverinfo="name"
    ))

    # --- 5. Layout styling ---
    fig.update_layout(
        title="Orbital Trajectory Projection",
        title_font=dict(size=18, family="sans-serif", color="#ffffff"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(color="#aaaaaa")
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450
    )
    
    return fig, is_earth_impact
