#!/usr/bin/env python3
"""Cache management utility for ocean router."""

from core.mask import clear_buffered_water_cache
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        print("Clearing buffered water cache...")
        clear_buffered_water_cache()
        print("Cache cleared.")
    else:
        print("Ocean Router Cache Manager")
        print("Usage:")
        print("  python cache_manager.py clear    - Clear all cached buffered water masks")
        print("  python cache_manager.py          - Show this help")

if __name__ == "__main__":
    main()
