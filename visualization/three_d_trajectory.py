import numpy as np
import plotly.graph_objects as go
import cv2

def create_sphere(radius, cx, cy, cz, resolution=30):
    """Helper to create a 3D sphere mesh."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u, v = np.meshgrid(u, v)
    x = cx + radius * np.cos(u) * np.sin(v)
    y = cy + radius * np.sin(u) * np.sin(v)
    z = cz + radius * np.cos(v)
    return x, y, z

def calculate_dbm_position_at_time(area_km2, target_hours=None):
    """
    Kinematic Drag-Based Model (DBM)
    Calculates CME transit time to 1 AU and current distance based on aerodynamic drag.
    Returns (transit_days, current_distance_normalized, v0)
    """
    v0 = 400.0 + (area_km2 / 5000.0) * 600.0 
    v0 = np.clip(v0, 300, 2500) # km/s

    w = 400.0 # Ambient solar wind speed (km/s)
    gamma = 0.2e-7 # Drag parameter (km^-1)
    d_1AU = 1.5e8 # 1 AU in km
    
    dt = 3600 # 1 hour time steps
    v = v0
    r = 0.0
    t_seconds = 0
    
    target_seconds = target_hours * 3600.0 if target_hours is not None else float('inf')
    r_at_target = None
    
    # Numerical Euler integration
    while r < d_1AU and t_seconds < 10 * 86400:
        if r_at_target is None and t_seconds >= target_seconds:
            r_at_target = r
        a = -gamma * (v - w) * abs(v - w)
        v += a * dt
        r += v * dt
        t_seconds += dt
        
    transit_days = t_seconds / 86400.0
    
    if r_at_target is None:
        r_at_target = r  # Reached end of simulation or hit Earth
        
    # Normalize r to 3D space units (where 1 AU = 5.0 units)
    # The sun surface is at r=1.0, so the travel distance is 5.0 units.
    cme_dist_normalized = (r_at_target / d_1AU) * 5.0
    return transit_days, cme_dist_normalized, v0


def get_dynamic_traces(time_elapsed_hours, transit_time_days, cme_dist_normalized, fx, fy, fz, cme_color):
    """Generates the Earth and CME traces for a specific moment in time."""
    earth_orbital_velocity_deg_per_day = 0.9856
    
    # Calculate Earth's position at the given simulation time
    current_time_days = time_elapsed_hours / 24.0
    earth_offset_rad = np.radians(current_time_days * earth_orbital_velocity_deg_per_day)
    
    earth_dist = -6.0
    earth_x = earth_dist * np.cos(earth_offset_rad)
    earth_y = earth_dist * np.sin(earth_offset_rad)
    
    xe, ye, ze = create_sphere(0.3, earth_x, earth_y, 0, resolution=30)
    earth_trace = go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, "#2ecc71"], [1, "#3498db"]],
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
        lightposition=dict(x=0, y=0, z=0),
        hoverinfo="name",
        name="Earth"
    )
    
    # Particle Simulation for CME
    cme_length = max(0.1, cme_dist_normalized + 2.0)
    cme_radius = max(0.1, (cme_dist_normalized / 5.0) * 2.5)
    
    num_particles = 800
    np.random.seed(42) # Constant seed so particles don't jitter between frames
    
    z_rand = np.random.uniform(0, 1, num_particles)
    r_rand = np.sqrt(np.random.uniform(0, 1, num_particles)) * (cme_radius / cme_length) * (z_rand * cme_length)
    theta_rand = np.random.uniform(0, 2 * np.pi, num_particles)
    
    p_x = r_rand * np.cos(theta_rand)
    p_y = r_rand * np.sin(theta_rand)
    p_z = z_rand * cme_length
    
    vec_z = np.array([0, 0, 1])
    vec_target = np.array([fx, fy, fz])
    vec_target = vec_target / np.linalg.norm(vec_target)
    
    v_rot = np.cross(vec_z, vec_target)
    s = np.linalg.norm(v_rot)
    c = np.dot(vec_z, vec_target)
    
    if s > 1e-6:
        v_skew = np.array([[0, -v_rot[2], v_rot[1]], [v_rot[2], 0, -v_rot[0]], [-v_rot[1], v_rot[0], 0]])
        R = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s**2))
    else:
        R = np.eye(3) if c > 0 else -np.eye(3)
        
    pts = np.stack([p_x, p_y, p_z])
    pts_rot = R @ pts
    
    p_x = pts_rot[0] + fx
    p_y = pts_rot[1] + fy
    p_z = pts_rot[2] + fz
    
    cme_trace = go.Scatter3d(
        x=p_x, y=p_y, z=p_z,
        mode='markers',
        marker=dict(
            size=1.5,
            color=z_rand,
            colorscale=[[0, cme_color + "1.0)"], [1, "rgba(255, 255, 255, 0.5)"]],
            opacity=0.8
        ),
        hoverinfo="none",
        name="CME Particles"
    )
    
    return earth_trace, cme_trace


def plot_3d_trajectory(image: np.ndarray, mask: np.ndarray, filament: dict) -> tuple[go.Figure, bool]:
    """
    Plots the solar disk in 3D, maps the H-alpha image to the front hemisphere,
    and visualizes the predicted CME cone and Earth's trajectory using Plotly animation frames.
    """
    fig = go.Figure()
    all_traces = []
    
    resolution = 100
    x_sun, y_sun, z_sun = create_sphere(1.0, 0, 0, 0, resolution=resolution)
    
    start_idx = resolution // 4
    end_idx = 3 * resolution // 4
    front_width = end_idx - start_idx
    
    mask_resized = cv2.resize(mask, (front_width, resolution))
    img_resized = cv2.resize(image, (front_width, resolution))
    
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
    intensity_front = (img_resized.astype(np.float64) / 255.0) * 0.99
    surface_colors_front = np.where(mask_resized > 0, 1.0, intensity_front)
    surface_colors_front = np.fliplr(surface_colors_front)
    
    surface_colors = np.zeros((resolution, resolution))
    surface_colors[:, start_idx:end_idx] = surface_colors_front
    
    custom_colorscale = [
        [0.0, "rgb(255,100,0)"],
        [0.5, "rgb(255,150,0)"],
        [0.9, "rgb(255,220,50)"],
        [0.99, "rgb(255,255,200)"],
        [1.0, "rgb(0,255,255)"]
    ]
    
    # 1. SUN TRACE (Index 0)
    sun_trace = go.Surface(
        x=x_sun, y=y_sun, z=z_sun,
        surfacecolor=surface_colors,
        colorscale=custom_colorscale,
        cmin=0.0, cmax=1.0,
        showscale=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        lightposition=dict(x=0, y=0, z=0),
        hoverinfo="none",
        name="Sun"
    )
    all_traces.append(sun_trace)
    
    # Calculate physics
    physical_props = filament.get("physical") or {}
    area_val = physical_props.get("area_km2")
    area_km2 = float(area_val) if area_val is not None else 2000.0
    
    transit_time_days, _, _ = calculate_dbm_position_at_time(area_km2)
    max_hours = int(transit_time_days * 24)
    
    earth_orbital_velocity_deg_per_day = 0.9856
    final_impact_rad = np.radians(transit_time_days * earth_orbital_velocity_deg_per_day)
    
    # 2. EARTH ORBIT PATH TRACE (Index 1)
    earth_dist = -6.0
    orbit_angles = np.linspace(0, final_impact_rad, 20)
    orbit_trace = go.Scatter3d(
        x=earth_dist * np.cos(orbit_angles),
        y=earth_dist * np.sin(orbit_angles),
        z=np.zeros_like(orbit_angles),
        mode="lines",
        line=dict(color="rgba(52, 152, 219, 0.5)", width=2, dash="dot"),
        hoverinfo="none",
        showlegend=False
    )
    all_traces.append(orbit_trace)

    # 3-18. PARKER SPIRAL TRACES (Indices 2 to 17)
    omega_w = 0.2 
    r_vals = np.linspace(1.0, 8.0, 100)
    for phi0 in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        phi_vals = phi0 + omega_w * (r_vals - 1.0)
        x_spiral = r_vals * np.cos(phi_vals)
        y_spiral = r_vals * np.sin(phi_vals)
        z_spiral = np.zeros_like(r_vals)
        
        spiral_trace = go.Scatter3d(
            x=x_spiral, y=y_spiral, z=z_spiral,
            mode='lines',
            line=dict(color='rgba(200, 100, 255, 0.15)', width=1.5),
            hoverinfo='none',
            showlegend=False
        )
        all_traces.append(spiral_trace)
        
    # Filament vector for CME
    H, W = image.shape[:2]
    cx = filament.get("centroid", {}).get("x", W / 2.0)
    cy = filament.get("centroid", {}).get("y", H / 2.0)
    
    fil_theta = 3 * np.pi / 2 - (cx / W) * np.pi
    fil_phi = (cy / H) * np.pi
    fx = np.sin(fil_phi) * np.cos(fil_theta)
    fy = np.sin(fil_phi) * np.sin(fil_theta)
    fz = np.cos(fil_phi)
    
    future_earth_vec = np.array([-np.cos(final_impact_rad), -np.sin(final_impact_rad), 0.0])
    eruption_vec = np.array([fx, fy, fz])
    
    angle_rad = np.arccos(np.clip(np.dot(eruption_vec, future_earth_vec), -1.0, 1.0))
    is_impact = angle_rad <= np.radians(25)
    cme_color = "rgba(255, 50, 50, " if is_impact else "rgba(50, 255, 100, "

    # Calculate Frames
    num_frames = 20
    step_hours = max_hours / num_frames
    
    frames = []
    slider_steps = []
    
    for i in range(num_frames + 1):
        h = i * step_hours
        _, current_dist_norm, _ = calculate_dbm_position_at_time(area_km2, h)
        
        e_trace, c_trace = get_dynamic_traces(h, transit_time_days, current_dist_norm, fx, fy, fz, cme_color)
        
        frame = go.Frame(
            data=[e_trace, c_trace],
            name=f"frame{i}",
            traces=[18, 19] # Indices of traces to update
        )
        frames.append(frame)
        
        slider_steps.append({
            "args": [
                [f"frame{i}"],
                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
            ],
            "label": f"T+ {int(h)}h",
            "method": "animate"
        })
        
        # Initial traces (t=0)
        if i == 0:
            all_traces.append(e_trace) # Index 18
            all_traces.append(c_trace) # Index 19

    fig.add_traces(all_traces)
    fig.frames = frames

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.0, y=1.2, z=0.5)
    )
    
    fig.update_layout(
        title=dict(text=f"Simulation Time: T+ 0h", font=dict(color='white', size=16), x=0.05, y=0.95),
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, range=[-8, 2]),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False, range=[-5, 5]),
            camera=camera,
            aspectmode='cube'
        ),
        paper_bgcolor='#1a1a24',
        plot_bgcolor='#1a1a24',
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        height=650,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16, "color": "white"},
                "prefix": "Time: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "linear"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps
        }]
    )
    
    return fig, is_impact
