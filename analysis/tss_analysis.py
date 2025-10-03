"""TSS analysis and export utilities."""

from utils.coordinates import pixel_to_latlon
from utils.distance import nm_distance
import csv


def print_tss_segments(path, in_tss_lane):
    """Print detailed analysis of TSS segments in the route.
    
    Args:
        path: List of (x, y) pixel coordinates
        in_tss_lane: List of booleans indicating TSS lane status for each point
    """
    if not path or not in_tss_lane or len(path) != len(in_tss_lane):
        print("Invalid path or TSS lane data")
        return
    
    print("\n" + "="*80)
    print("TSS LANE SEGMENT ANALYSIS")
    print("="*80)
    
    segments = []
    current_segment = None
    
    for i, (point, in_lane) in enumerate(zip(path, in_tss_lane)):
        if in_lane:
            if current_segment is None:
                # Start new TSS segment
                current_segment = {
                    'start_idx': i,
                    'end_idx': i,
                    'start_point': point,
                    'end_point': point
                }
            else:
                # Continue current segment
                current_segment['end_idx'] = i
                current_segment['end_point'] = point
        else:
            if current_segment is not None:
                # End current segment
                segments.append(current_segment)
                current_segment = None
    
    # Don't forget last segment if it ends in TSS
    if current_segment is not None:
        segments.append(current_segment)
    
    if not segments:
        print("No TSS lane segments found in route")
        return
    
    print(f"\nFound {len(segments)} TSS lane segment(s):\n")
    
    total_tss_distance = 0.0
    
    for idx, segment in enumerate(segments, 1):
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        start_point = segment['start_point']
        end_point = segment['end_point']
        
        # Convert to lat/lon
        start_lat, start_lon = pixel_to_latlon(start_point[0], start_point[1])
        end_lat, end_lon = pixel_to_latlon(end_point[0], end_point[1])
        
        # Calculate segment distance
        segment_distance = 0.0
        for i in range(start_idx, end_idx):
            p1 = path[i]
            p2 = path[i + 1]
            lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
            lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
            segment_distance += nm_distance(lat1, lon1, lat2, lon2)
        
        total_tss_distance += segment_distance
        
        print(f"Segment {idx}:")
        print(f"  Waypoints: {start_idx} to {end_idx} ({end_idx - start_idx + 1} points)")
        print(f"  Start: ({start_lat:.4f}°, {start_lon:.4f}°)")
        print(f"  End:   ({end_lat:.4f}°, {end_lon:.4f}°)")
        print(f"  Distance: {segment_distance:.2f} nm")
        print()
    
    # Calculate total route distance
    total_distance = 0.0
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        lat1, lon1 = pixel_to_latlon(p1[0], p1[1])
        lat2, lon2 = pixel_to_latlon(p2[0], p2[1])
        total_distance += nm_distance(lat1, lon1, lat2, lon2)
    
    print("-" * 80)
    print(f"Total route distance: {total_distance:.2f} nm")
    print(f"Distance in TSS lanes: {total_tss_distance:.2f} nm ({100*total_tss_distance/total_distance:.1f}%)")
    print(f"Distance outside TSS: {total_distance - total_tss_distance:.2f} nm ({100*(1-total_tss_distance/total_distance):.1f}%)")
    print("="*80 + "\n")


def export_tss_analysis(path, in_tss_lane, output_file):
    """Export TSS analysis to CSV file.
    
    Args:
        path: List of (x, y) pixel coordinates
        in_tss_lane: List of booleans indicating TSS lane status for each point
        output_file: Path to output CSV file
    """
    if not path or not in_tss_lane or len(path) != len(in_tss_lane):
        print(f"Cannot export TSS analysis: invalid path or TSS lane data")
        return
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['waypoint_index', 'latitude', 'longitude', 'in_tss_lane', 
                           'distance_from_start_nm', 'segment_distance_nm'])
            
            cumulative_distance = 0.0
            prev_lat, prev_lon = None, None
            
            for i, (point, in_lane) in enumerate(zip(path, in_tss_lane)):
                lat, lon = pixel_to_latlon(point[0], point[1])
                
                segment_distance = 0.0
                if i > 0 and prev_lat is not None:
                    segment_distance = nm_distance(prev_lat, prev_lon, lat, lon)
                    cumulative_distance += segment_distance
                
                writer.writerow([
                    i,
                    f"{lat:.6f}",
                    f"{lon:.6f}",
                    "Yes" if in_lane else "No",
                    f"{cumulative_distance:.2f}",
                    f"{segment_distance:.2f}"
                ])
                
                prev_lat, prev_lon = lat, lon
        
        print(f"TSS analysis exported to {output_file}")
    
    except Exception as e:
        print(f"Error exporting TSS analysis: {e}")
