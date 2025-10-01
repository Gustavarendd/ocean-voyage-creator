import csv
from pathlib import Path
import sys

def csv_to_linestring_feature(
    csv_path: str = "../CSV_features/waypoints.csv",
    osm_id: str = "1",
    z_order: int = 0,
    other_tags: str = r"\"seamark:type\"=>\"area_to_avoid\"",
    allow_single_point: bool = True,
    start_and_end_same: bool = True,
):
    """
    Read a CSV with columns LAT,LON (case-insensitive) and return a GeoJSON Feature (LineString).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Expected headers include LAT and LON (any case).
    osm_id : str
        Value for properties.osm_id.
    z_order : int
        Value for properties.z_order.
    other_tags : str
        Value for properties.other_tags.
    allow_single_point : bool
        If the CSV has only one row, duplicate the point so the LineString is valid.

    Returns
    -------
    dict
        A GeoJSON Feature dictionary with geometry.type = "LineString".
    """
    # Resolve path relative to this file if not absolute
    original_input = csv_path
    if not Path(csv_path).is_absolute():
        csv_path = str((Path(__file__).parent / csv_path).resolve())

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV not found. Tried: {csv_path}\n"
            f"Original argument: {original_input}\n"
            "Tip: Ensure the file exists, or pass an absolute path, e.g.\n"
            "python utils/create_feature.py /full/path/to/waypoints.csv"
        )

    coords = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize headers
        field_map = {name.lower().strip(): name for name in reader.fieldnames or []}
        if "lat" not in field_map or "lon" not in field_map:
            raise ValueError(
                f"CSV must contain 'LAT' and 'LON' headers; found: {reader.fieldnames}"
            )

        lat_key = field_map["lat"]
        lon_key = field_map["lon"]

        for row in reader:
            lat_str = (row.get(lat_key) or "").strip()
            lon_str = (row.get(lon_key) or "").strip()
            if not lat_str or not lon_str:
                continue  # skip blank or partial rows
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                # Skip rows with non-numeric values
                continue
            # GeoJSON uses [lon, lat] order
            coords.append([lon, lat])

    if not coords:
        raise ValueError("No valid coordinates found in CSV.")

    # Ensure a valid LineString (requires at least 2 positions)
    if len(coords) == 1:
        if allow_single_point:
            coords = coords * 2  # duplicate the single point
        else:
            raise ValueError("A LineString requires at least 2 coordinates.")
        
    if start_and_end_same and coords[0] != coords[-1]:
        coords.append(coords[0])  # close the loop if not already closed


    feature = {
        "type": "Feature",
        "properties": {
            "osm_id": str(osm_id),
            "z_order": int(z_order),
            "other_tags": other_tags,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }
    return feature


def _print_example_csv_hint():
    example = "LAT,LON\n52.1,4.3\n52.2,4.31\n"
    print("Example CSV (create if needed at CSV_features/waypoints.csv):\n" + example)


if __name__ == "__main__":
    # Allow optional CLI arg: path, else default
    cli_path = sys.argv[1] if len(sys.argv) > 1 else "../CSV_features/waypoints.csv"
    try:
        feature = csv_to_linestring_feature(csv_path=cli_path)
        print(feature)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        _print_example_csv_hint()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
