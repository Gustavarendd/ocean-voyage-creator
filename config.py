"""Configuration parameters for ocean routing."""

# Image dimensions
from shapely import Polygon


IMAGE_WIDTH = 21600 * 2  # Width of the currents image
IMAGE_HEIGHT = 10800 * 2  # Height of the currents image

# Latitude and longitude bounds for the currents data
LAT_MIN = -90.0  # 60°S
LAT_MAX = 90.0   # 65°N
LON_MIN = -180.0
LON_MAX = 180.0

# Navigation parameters
RADIUS = 1  # Increase to allow larger steps
EXPLORATION_ANGLES = 90  # Number of angles to explore

# Current effect parameters
MIN_SPEED = 0.01  # Minimum speed to avoid division by zero

# Coastal parameters
COASTAL_BUFFER_NM = 0  # Coastal buffer in nautical miles

# Route coordinates
ROUTE_COORDS = [
    # (54.1, 8.5), (36, -6),  # Hamburg -> Gibraltar 
    # (32, 32), (36, -4),  # Port-Said -> Gibraltar
    #(10, -80), (40, -73.5),  # Panama Canal -> New York
    # (29.5, -88), (40, -73),  # New Orleans -> New York
    #(25.5, -80), (40, -73.5),  # Miami -> New York
    #(25.5, 122.5), (58, -148),  # Taiwan -> Alaska
    # (52.12, 3.5), (40, -73.5),  # Rotterdam -> New York
    # (52.12, 3.5), (41.1, -9),  # Rotterdam -> Porto
    (60.0, 26.0), (51.5, -8.0) # St. Petersburg -> Cork
   
    # (57.5, 11.2), (51.5, -8.0)  # Guthenburg -> Cork
    # (64.83805979142075, 39.87918005420811), (41.1, -9) # Murmansk -> Porto
    #(30, 170), (30, -170) # cross dateline test
]

# Critical regions to keep open when applying the land mask buffer
# These regions are defined by (xMin, xMax, yMin, yMax) pixel coordinates
CRITICAL_REGIONS = [
    # (1358, 1363, 102, 108), # English Channel
    # (2896, 2915, 898, 903) # Strait of Gibraltar
    ]

# Define Traffic separation area where predefined wp are, as a polygon (example coordinates)
TSS_AREA = Polygon([
    (-5.5, 50), ( 1, 51), (1.5, 52), (3.5, 51.5), (-0.5, 49), (-5, 48.5)
])

# Define predefined waypoints for the TS area
PREDEFINED_SEGMENTS = {
    "channel_w_to_e": [
        (49.7500, -5.0000),   # SW of Land's End
        (49.8333, -4.0000),   # Casquets TSS
        (49.9167, -2.5000),
        (49.8333, -1.5000),   # North of Channel Islands
        (50.0000,  0.5000),   # Western Dover Strait
        (50.9167,  1.1667),   # Enter Dover Strait Eastbound TSS
        (51.0000,  1.7500),
        (51.0833,  2.5000),   # Exit to North Sea
    ],
    "channel_e_to_w": [
        (51.0833,  2.5000),   # Start in North Sea
        (51.0000,  1.7500),   # Dover Strait Westbound TSS
        (50.9167,  1.1667),
        (50.8333,  0.5000),   # Near Dungeness
        (50.5000, -1.5000),   # South of Isle of Wight
        (49.7500, -2.5000),   # Casquets TSS
        (49.7500, -4.0000),
        (49.7500, -5.0000),   # Exit Channel West
    ]
}

