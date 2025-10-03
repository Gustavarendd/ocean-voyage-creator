# Quick Start Guide

Get up and running with Ocean Router in 5 minutes!

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gustavarendd/ocean-voyage-creator.git
cd ocean-router

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Your First Route

### Step 1: Configure Your Route

Edit `config.py`:

```python
ROUTE_COORDS = [
    (51.5, -8.0),   # Start: Cork, Ireland
    (60.0, 26.0)    # End: St. Petersburg, Russia
]
```

### Step 2: Run the Router

```bash
python main.py
```

Expected output:
```
================================================================================
ROUTE REGION CALCULATION
================================================================================

Route waypoints: 2
  Start: (51.50°, -8.00°)
  End:   (60.00°, 26.00°)

Region bounds (with ±10° padding):
  Latitude:  41.50° to 70.00° (span: 28.50°)
  Longitude: -18.00° to 36.00° (span: 54.00°)
  Area:      2.1% of world (estimated)
  → Regional optimization: ~48× faster loading!
================================================================================

Loading cached buffered water mask...
Building TSS combined mask (cached)...
Using pixel search radius: 232 pixels
Using exploration angles: 120 (≈8.2 angles per pixel)

Calculating route...
A*: reached goal. expansions=15234 elapsed=23.45s

Route calculated successfully!
Total distance: 1847.32 nautical miles
TSS lane usage: 234/456 waypoints (51.3%)
```

### Step 3: View Results

Check `exports/` directory:
- `direct_route2.csv` - Your optimized route
- `tss_analysis.csv` - TSS compliance details

## Common Tasks

### Change Coastal Safety Buffer

In `config.py`:
```python
COASTAL_BUFFER_NM = 5  # Increase from 2 to 5 nautical miles
```

### Adjust Speed vs Precision

**For Speed** (quick routes):
```python
# In config.py
IMAGE_WIDTH = 21600      # Half resolution
IMAGE_HEIGHT = 10800

# In main.py, modify astar initialization:
exploration_angles = 90  # Fewer angles
heuristic_weight = 1.4   # More aggressive
```

**For Precision** (compliance-critical routes):
```python
# In config.py
IMAGE_WIDTH = 43200      # Full resolution
IMAGE_HEIGHT = 21600

# In main.py, modify astar initialization:
tss_cost_factor = 0.5    # Stronger TSS preference
exploration_angles = 150 # More angles
```

### Clear Cache After Changes

```bash
# Clear water mask cache
python cache_manager.py clear

# Clear TSS mask cache
python cache_manager.py clear-tss
```

## Example Routes

### Short Coastal Route
```python
ROUTE_COORDS = [
    (51.5, -8.0),    # Cork, Ireland
    (50.8, 1.2)      # Dover, UK
]
```
Expected: ~5 seconds, ~250 nm

### Trans-Atlantic Route
```python
ROUTE_COORDS = [
    (40.7, -73.5),   # New York, USA
    (51.5, -0.1)     # London, UK
]
```
Expected: ~60 seconds, ~3000 nm

### Mediterranean Route
```python
ROUTE_COORDS = [
    (36.0, -6.0),    # Gibraltar
    (32.0, 32.0)     # Port Said, Egypt
]
```
Expected: ~30 seconds, ~1800 nm

## Troubleshooting

### "No path found"
**Problem**: Waypoints might be on land or blocked by coastal buffer

**Solution**:
```python
# Reduce buffer temporarily
COASTAL_BUFFER_NM = 1

# Or check waypoints are in water
# The system will show: "Is navigable water: True/False"
```

### "Very slow (>2 minutes)"
**Problem**: Resolution too high or too many angles

**Solution**:
```python
# Reduce resolution
IMAGE_WIDTH = 21600
IMAGE_HEIGHT = 10800

# Reduce angles
exploration_angles = 90
```

### "Poor TSS compliance (<50%)"
**Problem**: TSS preference not strong enough

**Solution**:
```python
# Strengthen TSS preference
tss_cost_factor = 0.5  # Lower = stronger preference

# Widen TSS lanes
dilation_radius = 2    # Increase from 1
```

## Next Steps

1. **Read the full README** for detailed configuration options
2. **Check OPTIMIZATION_SUGGESTIONS.md** for performance tuning
3. **Explore the code** in `navigation/` for algorithm details
4. **Customize** for your specific vessel type and regulations

## Getting Help

- Check `README.md` for comprehensive documentation
- Review `OPTIMIZATION_SUGGESTIONS.md` for tuning guidance
- Look at `CHANGELOG.md` for recent changes
- Open an issue on GitHub for bugs or questions

## Tips for Success

✅ **DO:**
- Start with short routes to verify setup
- Enable caching for repeated runs
- Clear cache after changing resolution or buffers
- Monitor TSS compliance metrics
- Use provided benchmarks for comparison

❌ **DON'T:**
- Use very high resolution for first tests (slow!)
- Forget to clear cache after data changes
- Ignore TSS compliance warnings
- Set `COASTAL_BUFFER_NM = 0` (unsafe!)
- Modify cached files manually

---

**Happy Routing!** 🚢⚓

For questions: https://github.com/Gustavarendd/ocean-voyage-creator
