

#   The JSON stores 4×4 transformation matrices in w2c format with a specific coordinate system that requires:
#  - Column reordering from [x, y, z, w] to [y, z, x, w]
#   - Sign flip on one axis
#  - Final transformation to convert to camera-to-world (c2w) format for visualization