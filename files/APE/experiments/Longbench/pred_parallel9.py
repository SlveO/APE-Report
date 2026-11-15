"""LongBench v3 entry: the shared, budgeted MAIN80 campaign.

Defaults and all three datasets are frozen by campaign.json / the data manifest.
T/S are passed by the shared worker through the public APE API. No short-ID
filtering or unbudgeted multi-GPU prediction loop remains here.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main80.campaign import main

if __name__ == "__main__":
    main()
