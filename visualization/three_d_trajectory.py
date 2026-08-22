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

def plot_3d_trajectory(image: np.ndarray, mask: np.ndarray, filament: dict) -> tuple[go.Figure, bool]:
    """
    Plots the solar disk in 3D, maps the H-alpha image to the front hemisphere,
    and visualizes the predicted CME cone and Earth's trajectory.
    """
    resolution = 100
    x_sun, y_sun, z_sun = create_sphere(1.0, 0, 0, 0, resolution=resolution)
    
    # 1. Map the 2D image ONLY to the front hemisphere facing Earth
    # Earth is at -X, so the front hemisphere is theta from pi/2 to 3pi/2
    start_idx = resolution // 4
    end_idx = 3 * resolution // 4
    front_width = end_idx - start_idx
    
    # Resize image to cover ONLY the front hemisphere
    mask_resized = cv2.resize(mask, (front_width, resolution))
    img_resized = cv2.resize(image, (front_width, resolution))
    
    if len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
    intensity_front = (img_resized.astype(np.float64) / 255.0) * 0.99
    surface_colors_front = np.where(mask_resized > 0, 1.0, intensity_front)
    
    # Flip the image horizontally so the left edge of the image maps to the left side of the sphere (-Y)
    surface_colors_front = np.fliplr(surface_colors_front)
    
    # The rest of the Sun (back hemisphere) defaults to 0.0 (baseline bright orange)
    surface_colors = np.zeros((resolution, resolution))
    surface_colors[:, start_idx:end_idx] = surface_colors_front
    
    # Custom colorscale: Make the baseline extremely bright so shadows are invisible
    custom_colorscale = [
        [0.0, "rgb(255,100,0)"],   # Bright orange baseline
        [0.5, "rgb(255,150,0)"],
        [0.9, "rgb(255,220,50)"],
        [0.99, "rgb(255,255,200)"],
        [1.0, "rgb(0,255,255)"]    # Cyan for filaments
    ]
    
    fig = go.Figure()
    
    # 1. SUN (Emissive)
    fig.add_trace(go.Surface(
        x=x_sun, y=y_sun, z=z_sun,
        surfacecolor=surface_colors,
        colorscale=custom_colorscale,
        cmin=0.0, cmax=1.0,
        showscale=False,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
        lightposition=dict(x=0, y=0, z=0),
        hoverinfo="none",
        name="Sun"
    ))
    
    # 2. EARTH (A real 3D sphere) - Account for orbital motion
    # A typical CME takes ~2 days to reach Earth. Earth moves ~0.986 degrees/day.
    transit_time_days = 2.0
    earth_orbital_velocity_deg_per_day = 0.9856
    earth_offset_rad = np.radians(transit_time_days * earth_orbital_velocity_deg_per_day)
    
    earth_dist = -6.0
    # Original Earth was at [-6, 0, 0] (angle pi).
    # Moving counter-clockwise by earth_offset_rad:
    earth_x = earth_dist * np.cos(earth_offset_rad)
    earth_y = earth_dist * np.sin(earth_offset_rad)
    
    xe, ye, ze = create_sphere(0.3, earth_x, earth_y, 0, resolution=30)
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, "#2ecc71"], [1, "#3498db"]],
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.2),
        lightposition=dict(x=0, y=0, z=0), # Light comes from the sun
        hoverinfo="name",
        name="Earth (At Impact)"
    ))
    
    # Draw a faint orbital path leading to Earth to show its movement
    orbit_angles = np.linspace(0, earth_offset_rad, 20)
    fig.add_trace(go.Scatter3d(
        x=earth_dist * np.cos(orbit_angles),
        y=earth_dist * np.sin(orbit_angles),
        z=np.zeros_like(orbit_angles),
        mode="lines",
        line=dict(color="rgba(52, 152, 219, 0.5)", width=2, dash="dot"),
        hoverinfo="none",
        showlegend=False
    ))

    # 3. CME ERUPTION MESH
    H, W = image.shape[:2]
    cx = filament.get("centroid", {}).get("x", W / 2.0)
    cy = filament.get("centroid", {}).get("y", H / 2.0)
    
    # Calculate filament position on the front hemisphere
    # The image is mapped from theta = 3pi/2 (left, -Y) to pi/2 (right, +Y)
    fil_theta = 3 * np.pi / 2 - (cx / W) * np.pi
    fil_phi = (cy / H) * np.pi
    
    fx = np.sin(fil_phi) * np.cos(fil_theta)
    fy = np.sin(fil_phi) * np.sin(fil_theta)
    fz = np.cos(fil_phi)
    
    # Future Earth unit vector for impact detection
    future_earth_vec = np.array([-np.cos(earth_offset_rad), -np.sin(earth_offset_rad), 0.0])
    eruption_vec = np.array([fx, fy, fz])
    
    angle_rad = np.arccos(np.clip(np.dot(eruption_vec, future_earth_vec), -1.0, 1.0))
    is_impact = angle_rad <= np.radians(25)
    cme_color = "rgba(255, 50, 50, " if is_impact else "rgba(50, 255, 100, "
    
    # Create a translucent 3D cone mesh for the CME instead of a stick
    cme_length = 7.0
    cme_radius = 2.5
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, 1, 15)
    u, v = np.meshgrid(u, v)
    
    # Standard cone along Z axis
    cone_x = cme_radius * v * np.cos(u)
    cone_y = cme_radius * v * np.sin(u)
    cone_z = cme_length * v
    
    # Rotate cone to align with filament vector [fx, fy, fz]
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
        
    cone_pts = np.stack([cone_x.flatten(), cone_y.flatten(), cone_z.flatten()])
    cone_rot = R @ cone_pts
    
    cone_x = cone_rot[0].reshape(cone_x.shape)
    cone_y = cone_rot[1].reshape(cone_y.shape)
    cone_z = cone_rot[2].reshape(cone_z.shape)
    
    # Shift to start at the sun's surface
    cone_x += fx
    cone_y += fy
    cone_z += fz
    
    fig.add_trace(go.Surface(
        x=cone_x, y=cone_y, z=cone_z,
        colorscale=[[0, cme_color + "0.1)"], [1, cme_color + "0.4)"]],
        showscale=False,
        hoverinfo="none",
        name="CME Cloud"
    ))
    
    # Core particle beam inside the CME
    np.random.seed(123)
    for _ in range(15):
        noise = np.random.normal(0, 0.1, 3)
        end_pt = vec_target * cme_length + noise * cme_length
        fig.add_trace(go.Scatter3d(
            x=[fx, end_pt[0]], y=[fy, end_pt[1]], z=[fz, end_pt[2]],
            mode="lines",
            line=dict(color=cme_color + "0.8)", width=np.random.randint(1, 4)),
            hoverinfo="none",
            showlegend=False
        ))

    # Layout configuration
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.0, y=1.2, z=0.5)
    )
    
    fig.update_layout(
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
        height=650
    )
    
    return fig, is_impact
