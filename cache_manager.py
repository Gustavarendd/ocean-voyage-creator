#!/usr/bin/env python3
"""Cache management utility for ocean router."""

from core.mask import clear_buffered_water_cache
import os, glob
import sys

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "clear":
            print("Clearing buffered water cache...")
            clear_buffered_water_cache()
            print("Cache cleared.")
            return
        if sys.argv[1] == "clear-tss":
            cache_dir = "cache"
            removed = 0
            for f in glob.glob(os.path.join(cache_dir, "tss_mask_*.npz")):
                try:
                    os.remove(f)
                    print(f"Removed {f}")
                    removed += 1
                except Exception as e:
                    print(f"Failed to remove {f}: {e}")
            if removed == 0:
                print("No TSS mask cache files found.")
            else:
                print(f"Removed {removed} TSS mask cache file(s).")
            return
    else:
        print("Ocean Router Cache Manager")
        print("Usage:")
        print("  python cache_manager.py clear        - Clear buffered water masks")
        print("  python cache_manager.py clear-tss    - Clear cached TSS masks")
        print("  python cache_manager.py              - Show this help")

if __name__ == "__main__":
    main()
