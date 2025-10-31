import numpy as np
import json

def matrix_to_json_string(matrix):
    """Convert numpy matrix to the JSON string format"""
    rows = []
    for i in range(4):
        row_str = ' '.join(map(str, matrix[i]))
        rows.append(f"[{row_str}]")
    return ' '.join(rows) + ' '

def create_pan_camera(position, pan_angle_degrees):
    """
    Create a camera matrix with pan rotation (around Z-axis)
    cam11: Pan right
    """
    angle = np.radians(pan_angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    matrix = np.array([
        [cos_a, -sin_a, 0, position[0]],
        [sin_a,  cos_a, 0, position[1]],
        [0,      0,     1, position[2]],
        [0,      0,     0, 1]
    ])
    return matrix

def create_dolly_camera(start_pos, end_pos, t):
    """
    Create a camera moving forward/backward (dolly)
    cam12: Dolly zoom (moving forward)
    """
    pos = start_pos + t * (end_pos - start_pos)
    # Identity rotation (camera faces forward)
    matrix = np.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])
    return matrix

def create_oval_camera(center, radius_x, radius_y, angle_degrees, height):
    """
    Create a camera moving in an oval path
    cam13: Oval movement to the right
    """
    angle = np.radians(angle_degrees)
    
    # Oval position
    x = center[0] + radius_x * np.cos(angle)
    y = center[1] + radius_y * np.sin(angle)
    z = height
    
    # Calculate direction tangent to the oval for smooth rotation
    # Tangent angle accounts for elliptical path
    tangent_angle = np.arctan2(radius_x * np.cos(angle), -radius_y * np.sin(angle))
    
    # Camera rotation to look along the movement direction
    cos_a = np.cos(tangent_angle)
    sin_a = np.sin(tangent_angle)
    
    matrix = np.array([
        [cos_a, -sin_a, 0, x],
        [sin_a,  cos_a, 0, y],
        [0,      0,     1, z],
        [0,      0,     0, 1]
    ])
    return matrix

def generate_all_cameras(num_frames=100):
    """Generate all three camera trajectories"""
    
    # Starting position (same for all)
    base_position = np.array([3390.0, 1380.0, 240.0])
    
    camera_data = {}
    
    for frame in range(num_frames):
        t = frame / (num_frames - 1)  # Normalized time [0, 1]
        
        frame_data = {}
        
        # ============================================
        # CAM11: Pan Right (0° to 45°)
        # ============================================
        pan_angle = 0 + t * 45  # 45 degree pan
        cam11_matrix = create_pan_camera(base_position, pan_angle)
        cam11_transposed = cam11_matrix.T
        frame_data["cam11"] = matrix_to_json_string(cam11_transposed)
        
        # ============================================
        # CAM12: Dolly Zoom (move forward 500 units)
        # ============================================
        start_pos = base_position.copy()
        end_pos = base_position + np.array([500, 0, 0])  # Move forward in X
        cam12_matrix = create_dolly_camera(start_pos, end_pos, t)
        cam12_transposed = cam12_matrix.T
        frame_data["cam12"] = matrix_to_json_string(cam12_transposed)
        
        # ============================================
        # CAM13: Oval Right (circular path)
        # ============================================
        center = base_position[:2]  # [x, y] center of oval
        radius_x = 300  # Horizontal radius
        radius_y = 200  # Vertical radius (smaller = more oval)
        angle = t * 360  # Full circle
        height = base_position[2]
        
        cam13_matrix = create_oval_camera(center, radius_x, radius_y, angle, height)
        cam13_transposed = cam13_matrix.T
        frame_data["cam13"] = matrix_to_json_string(cam13_transposed)
        
        # Add to main data structure
        camera_data[f"frame{frame}"] = frame_data
    
    return camera_data

# Generate and save
print("Generating camera trajectories...")
camera_data = generate_all_cameras(num_frames=81)

with open('multi_camera_trajectories.json', 'w') as f:
    json.dump(camera_data, f, indent=4)

print("  - cam11: Pan right (45°)")
print("  - cam12: Dolly zoom (forward 500 units)")
print("  - cam13: Oval movement (300x200 oval)")
print("\nSaved to: multi_camera_trajectories.json")