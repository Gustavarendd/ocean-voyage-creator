

# Function that splits separation_lanes_only.geojson separate files depending on their tss_flow_cardinal property

def split_tss_by_direction():
    import json
    from collections import defaultdict

    input_geojson = "TSS/separation_lanes_only.geojson"

    # Load the GeoJSON file
    with open(input_geojson, 'r') as f:
        data = json.load(f)

    # Dictionary to hold features by direction
    direction_dict = defaultdict(list)

    # Iterate through features and group by tss_flow_cardinal
    for feature in data['features']:
        direction = feature['properties'].get('tss_flow_cardinal', 'unknown')
        direction_dict[direction].append(feature)


    # Write out separate GeoJSON files for each direction
    for direction, features in direction_dict.items():
        output_data = {
            "type": "FeatureCollection",
            "features": features
        }
        output_file = f"TSS_Lanes_{direction}.geojson"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

def get_tss_waypoints_near_position(wp, direction_of_travel, max_distance_meters=50000):
    """
    Get TSS waypoints if the given waypoint is within specified distance of the start of a TSS.
    
    Args:
        wp: [lat, lon] - waypoint to check
        direction_of_travel: float - direction in degrees (0-360)
        max_distance_meters: float - maximum distance in meters to search for TSS (default: 10000)
    
    Returns:
        Dict with 'waypoints' and 'properties' if TSS found within range, None otherwise
    """
    import json
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.distance import nm_distance
    
    # Validate inputs
    if not wp or len(wp) != 2:
        print("Error: Waypoint must be [lat, lon]")
        return None
    
    if not (-90 <= wp[0] <= 90) or not (-180 <= wp[1] <= 180):
        print("Error: Invalid waypoint coordinates")
        return None
    
    if not (0 <= direction_of_travel <= 360):
        print("Error: Direction must be between 0 and 360 degrees")
        return None
    
    # Direction to cardinal mapping
    cardinal_direction = ""
    if (direction_of_travel >= 0 and direction_of_travel < 11.25) or (direction_of_travel >= 348.75 and direction_of_travel <= 360):
        cardinal_direction = "N"
    elif direction_of_travel >= 11.25 and direction_of_travel < 33.75:
        cardinal_direction = "NNE"
    elif direction_of_travel >= 33.75 and direction_of_travel < 56.25:
        cardinal_direction = "NE"
    elif direction_of_travel >= 56.25 and direction_of_travel < 78.75:
        cardinal_direction = "ENE"
    elif direction_of_travel >= 78.75 and direction_of_travel < 101.25:
        cardinal_direction = "E"
    elif direction_of_travel >= 101.25 and direction_of_travel < 123.75:
        cardinal_direction = "ESE"
    elif direction_of_travel >= 123.75 and direction_of_travel < 146.25:
        cardinal_direction = "SE"
    elif direction_of_travel >= 146.25 and direction_of_travel < 168.75:
        cardinal_direction = "SSE"
    elif direction_of_travel >= 168.75 and direction_of_travel < 191.25:
        cardinal_direction = "S"
    elif direction_of_travel >= 191.25 and direction_of_travel < 213.75:
        cardinal_direction = "SSW"
    elif direction_of_travel >= 213.75 and direction_of_travel < 236.25:
        cardinal_direction = "SW"
    elif direction_of_travel >= 236.25 and direction_of_travel < 258.75:
        cardinal_direction = "WSW"
    elif direction_of_travel >= 258.75 and direction_of_travel < 281.25:
        cardinal_direction = "W"
    elif direction_of_travel >= 281.25 and direction_of_travel < 303.75:
        cardinal_direction = "WNW"
    elif direction_of_travel >= 303.75 and direction_of_travel < 326.25:
        cardinal_direction = "NW"
    elif direction_of_travel >= 326.25 and direction_of_travel < 348.75:
        cardinal_direction = "NNW"
    else:
        cardinal_direction = "unknown"
    
    
    
    # Convert meters to nautical miles for distance calculation
    max_distance_nm = max_distance_meters / 1852
    
    # Determine which directions to check (main direction + two adjacent)
    directions_to_check = [cardinal_direction]
    
    # Add adjacent directions for all cardinal directions
    # direction_adjacency = {
    #     "N": ["NW", "NNW", "NNE", "NE"],
    #     "NNE": ["NNW", "N", "NE", "ENE"],
    #     "NE": ["N", "NNE", "ENE", "E"],
    #     "ENE": ["NNE", "NE", "E", "ESE"],
    #     "E": ["NE", "ENE", "ESE", "SE"],
    #     "ESE": ["ENE", "E", "SE", "SSE"],
    #     "SE": ["E", "ESE", "SSE", "S"],
    #     "SSE": ["ESE", "SE", "S", "SSW"],
    #     "S": ["SE", "SSE", "SSW", "SW"],
    #     "SSW": ["SSE", "S", "SW", "WSW"],
    #     "SW": ["S", "SSW", "WSW", "W"],
    #     "WSW": ["SSW", "SW", "W", "WNW"],
    #     "W": ["SW", "WSW", "WNW", "NW"],
    #     "WNW": ["WSW", "W", "NW", "NNW"],
    #     "NW": ["W", "WNW", "NNW", "N"],
    #     "NNW": ["WNW", "NW", "N", "NNE"],
    # }
    direction_adjacency = {
        "N": [ "NNW", "NNE"],
        "NNE": [ "N", "NE"],
        "NE": [ "NNE", "ENE"],
        "ENE": [ "NE", "E"],
        "E": [ "ENE", "ESE"],
        "ESE": [ "E", "SE"],
        "SE": [ "ESE", "SSE"],
        "SSE": [ "SE", "S"],
        "S": ["SSE", "SSW"],
        "SSW": [ "S", "SW"],
        "SW": [ "SSW", "WSW"],
        "WSW": [ "SW", "W"],
        "W": [ "WSW", "WNW"],
        "WNW": [ "W", "NW"],
        "NW": [ "WNW", "NNW"],
        "NNW": [ "NW", "N"],
    }
    
    if cardinal_direction in direction_adjacency:
        directions_to_check.extend(direction_adjacency[cardinal_direction])
    
    # Check each TSS lane to see if the waypoint is within the specified distance of the start
    closest_tss = None
    closest_distance = float('inf')
    
    for direction in directions_to_check:
        # Load the appropriate GeoJSON file based on cardinal direction
        geojson_file = f"TSS_by_direction/TSS_Lanes_{direction}.geojson"
        
        if not os.path.exists(geojson_file):
            print(f"No TSS file found for direction {direction}")
            continue
        
        try:
            with open(geojson_file, 'r') as f:
                tss_data = json.load(f)
        except Exception as e:
            print(f"Error loading TSS file {geojson_file}: {e}")
            continue
        
        for feature in tss_data['features']:
            if feature['geometry']['type'] == 'LineString':
                coordinates = feature['geometry']['coordinates']
                if len(coordinates) > 0:
                    # Get the start point of the TSS lane
                    start_point = coordinates[0]  # [lon, lat]
                    start_lat, start_lon = start_point[1], start_point[0]
                    
                    # Calculate distance from waypoint to start of TSS
                    distance_nm = nm_distance(wp[0], wp[1], start_lat, start_lon)
                    
                    if distance_nm <= max_distance_nm and distance_nm < closest_distance:
                        closest_distance = distance_nm
                        
                        # Prepare the waypoints (coordinates) of this TSS
                        waypoints = []
                        for coord in coordinates:
                            waypoints.append([coord[1], coord[0]])  # Convert [lon, lat] to [lat, lon]
                        
                        closest_tss = {
                            'waypoints': waypoints,
                            'properties': feature.get('properties', {}),
                            'distance_nm': distance_nm,
                            'distance_m': distance_nm * 1852
                        }
    
    if closest_tss:
        print(f"Found TSS within {closest_tss['distance_nm']:.2f} nm ({closest_tss['distance_m']:.0f} m) of waypoint")
        print(f"TSS properties: {closest_tss['properties']}")
        return closest_tss
    else:
        return None


# Example usage:
if __name__ == "__main__":
    # Test with a sample waypoint and direction
    test_wp = [52, 4]  # [lat, lon] - near a TSS in the North Sea
    test_direction = 355.71  # North direction
    
    result = get_tss_waypoints_near_position(test_wp, test_direction)
    if result:
        print(f"TSS waypoints found: {len(result['waypoints'])} points")
        for i, wp in enumerate(result['waypoints']):
            print(f"  WP {i+1}: {wp}")
        print(f"Distance: {result['distance_m']:.0f} meters")
    else:
        print("No TSS found near the test waypoint")
    


