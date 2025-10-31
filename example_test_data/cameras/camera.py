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
    """
    pos = start_pos + t * (end_pos - start_pos)
    matrix = np.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])
    return matrix

def create_pan_and_dolly_camera(start_pos, end_pos, pan_angle_degrees, t):
    """
    Create a camera that pans AND moves simultaneously
    """
    # Interpolate position
    pos = start_pos + t * (end_pos - start_pos)
    
    # Pan rotation
    angle = np.radians(pan_angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    matrix = np.array([
        [cos_a, -sin_a, 0, pos[0]],
        [sin_a,  cos_a, 0, pos[1]],
        [0,      0,     1, pos[2]],
        [0,      0,     0, 1]
    ])
    return matrix

def generate_all_cameras(num_frames=81):
    """Generate all three camera trajectories"""
    
    # Starting position (same for all)
    base_position = np.array([3390.0, 1380.0, 240.0])
    
    camera_data = {}
    
    for frame in range(num_frames):
        t = frame / (num_frames - 1)  # Normalized time [0, 1]
        
        frame_data = {}
        
        # ============================================
        # CAM14: Pan Right (0° to 45°) then Pan Left back to 0°
        # ============================================
        if t <= 0.5:
            # First half: pan right from 0° to 45°
            pan_angle = t * 2 * 45  # 0 to 45 degrees
        else:
            # Second half: pan left from 45° back to 0°
            pan_angle = (1 - (t - 0.5) * 2) * 45  # 45 back to 0 degrees
        
        cam14_matrix = create_pan_camera(base_position, pan_angle)
        cam14_transposed = cam14_matrix.T
        frame_data["cam14"] = matrix_to_json_string(cam14_transposed)
        
        # ============================================
        # CAM15: Pan Right (0° to 45°) AND Dolly Forward simultaneously
        # ============================================
        pan_angle = t * 45  # 0 to 45 degrees
        start_pos = base_position.copy()
        end_pos = base_position + np.array([500, 0, 0])  # Move forward 500 units in X
        
        cam15_matrix = create_pan_and_dolly_camera(start_pos, end_pos, pan_angle, t)
        cam15_transposed = cam15_matrix.T
        frame_data["cam15"] = matrix_to_json_string(cam15_transposed)
        
        # ============================================
        # CAM16: Pan Right (first half) THEN Dolly Forward (second half)
        # ============================================
        if t <= 0.5:
            # First half: pan right from 0° to 45°, stay in place
            pan_angle = t * 2 * 45  # 0 to 45 degrees
            cam16_matrix = create_pan_camera(base_position, pan_angle)
        else:
            # Second half: keep 45° pan, move forward
            pan_angle = 45  # Stay at 45 degrees
            t_dolly = (t - 0.5) * 2  # Remap to [0, 1] for second half
            start_pos = base_position.copy()
            end_pos = base_position + np.array([500, 0, 0])
            cam16_matrix = create_pan_and_dolly_camera(start_pos, end_pos, pan_angle, t_dolly)
        
        cam16_transposed = cam16_matrix.T
        frame_data["cam16"] = matrix_to_json_string(cam16_transposed)
        
        # Add frame to camera data
        camera_data[f"frame{frame}"] = frame_data
    
    return camera_data

# Generate and save
print("Generating camera trajectories...")
camera_data = generate_all_cameras(num_frames=81)

with open('v3_camera_trajectories.json', 'w') as f:
    json.dump(camera_data, f, indent=4)

