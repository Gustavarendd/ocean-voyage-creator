"""Configuration parameters for ocean routing."""

# Image dimensions
IMAGE_WIDTH = 2700  # Width of the currents image
IMAGE_HEIGHT = 938  # Height of the currents image

# Latitude and longitude bounds for the currents data
LAT_MIN = -60.0  # 60°S
LAT_MAX = 65.0   # 65°N
LON_MIN = -180.0
LON_MAX = 180.0

# Navigation parameters
SHIP_SPEED_KN = 10.0  # ship speed in knots
RADIUS = 2  # Increase to allow larger steps
EXPLORATION_ANGLES = 32  # Number of angles to explore

# Coastal parameters
COASTAL_BUFFER_NM = 12  # Coastal buffer in nautical miles

# Route coordinates
ROUTE_COORDS = [
    #(32, 32), (36, -5),  # Port-Said -> Gibraltar
    #(10, -80), (40, -73.5),  # Panama Canal -> New York
    (29.5, -88), (40, -73),  # New Orleans -> New York
    #(25.5, -80), (40, -73.5),  # Miami -> New York
    #(25.5, 122.5), (58, -148),  # Taiwan -> Alaska
    #(38.7, -70.9), (30.37, -80.134)
]

# Current effect parameters
MIN_SPEED = 0.01  # Minimum speed to avoid division by zero
ADVERSE_CURRENT_FACTOR = 0.8  # Speed reduction when moving against current
FAVORABLE_CURRENT_BONUS = 0.9  # Time bonus for favorable currents
CURRENT_BIAS_FACTOR = 1.0  # Bias for seeking favorable currents
