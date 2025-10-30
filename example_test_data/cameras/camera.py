# in this file, we will create a sequnce of camera trajectory

import numpy as np
import json

def create_pan_camera(position, pan_angle_degrees):
    """
    Create a camera matrix with pan rotation (around Z-axis)
    
    Args:
        position: [x, y, z] camera position
        pan_angle_degrees: rotation angle in degrees (positive = pan right)
    """
    angle = np.radians(pan_angle_degrees)
    
    # Rotation matrix around Z-axis (pan)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Build 4x4 matrix
    matrix = np.array([
        [cos_a, -sin_a, 0, position[0]],
        [sin_a,  cos_a, 0, position[1]],
        [0,      0,     1, position[2]],
        [0,      0,     0, 1]
    ])
    
    return matrix

def matrix_to_json_string(matrix):
    """Convert numpy matrix to the JSON string format"""
    rows = []
    for i in range(4):
        row_str = ' '.join(map(str, matrix[i]))
        rows.append(f"[{row_str}]")
    return ' '.join(rows) + ' '

# Generate custom pan sequence
num_frames = 81
start_angle = 0
end_angle = 45  # Pan 45 degrees to the right
position = [3390, 1380, 240]

camera_data = {}
for frame in range(num_frames):
    # Interpolate angle
    t = frame / (num_frames - 1)
    angle = start_angle + t * (end_angle - start_angle)
    
    # Create camera matrix
    cam_matrix = create_pan_camera(position, angle)
    
    # Transpose back to row format (inverse of what the code does)
    cam_matrix_transposed = cam_matrix.T
    
    # Convert to JSON string format
    cam_string = matrix_to_json_string(cam_matrix_transposed)
    
    camera_data[f"frame{frame}"] = {
        "cam11": cam_string
    }

# Save to JSON
with open('custom_camera_pan.json', 'w') as f:
    json.dump(camera_data, f, indent=4)

print(f"Generated {num_frames} frames with {end_angle}° pan")