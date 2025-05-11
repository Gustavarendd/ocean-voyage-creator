# Ocean Router

Ocean Router is a Python-based tool for optimizing ship routes across oceans. It considers ocean currents, wave conditions, and coastal constraints to calculate the most efficient path between waypoints.

## Features

- **Currents and Wave Data Integration**: Extracts and processes ocean currents and wave data from input images.
- **Route Optimization**: Uses the A\* algorithm to find the optimal route, considering:
  - Ocean currents (U and V components).
  - Wave height, period, and direction.
  - Coastal buffer zones and critical water channels.
- **Direct vs Optimized Route Comparison**: Calculates and compares direct and optimized routes in terms of distance and time.
- **Visualization**: Plots the optimized and direct routes, along with ocean currents and water masks.
- **Export**: Saves the calculated routes to CSV files for further analysis.

## How It Works

1. **Input Data**:

   - Currents image (`currents_65N_60S_2700x938.png`): Encodes U and V components of ocean currents.
   - Land mask image (`land_mask_90N_90S_6000x3000.png`): Identifies land and water regions.
   - Wave image (`wave2.png`): Encodes wave height, period, and direction.

2. **Processing**:

   - The images are loaded and processed to extract currents, wave data, and a water mask.
   - A coastal buffer is applied to the water mask to account for navigational constraints.

3. **Route Calculation**:

   - Waypoints are converted from latitude/longitude to pixel coordinates.
   - The A\* algorithm calculates the optimized route, considering currents, waves, and coastal constraints.
   - Metrics such as distance and time are calculated for both direct and optimized routes.

4. **Output**:
   - The optimized and direct routes are exported to CSV files.
   - Visualizations of the routes and ocean data are displayed.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/ocean-router.git
   cd ocean-router
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the required input images in the `images/` directory.

## Usage

Run the main script:

```bash
python main.py
```

The script will:

- Calculate the optimized and direct routes.
- Export the routes to `exports/optimized_route.csv` and `exports/direct_route.csv`.
- Display visualizations of the routes and ocean data.

## Configuration

Modify the `config.py` file to adjust parameters such as:

- Image dimensions (`IMAGE_WIDTH`, `IMAGE_HEIGHT`).
- Latitude and longitude bounds (`LAT_MIN`, `LAT_MAX`, `LON_MIN`, `LON_MAX`).
- Ship characteristics (`SHIP_CONFIG`).
- Coastal buffer size (`COASTAL_BUFFER_NM`).
- Waypoints (`ROUTE_COORDS`).

## Example Output

- **Optimized Route**: A route that minimizes travel time by leveraging favorable currents and avoiding high wave heights.
- **Direct Route**: A straight-line route between waypoints, ignoring currents and waves.

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pillow
- Shapely
