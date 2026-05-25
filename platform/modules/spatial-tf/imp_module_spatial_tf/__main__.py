"""Run the spatial-tf module:

    python -m imp_module_spatial_tf --station devstation
"""

import argparse
import os

from imp_sdk import run_module

from .module import TfModule


def main() -> None:
    ap = argparse.ArgumentParser(prog="imp-module-spatial-tf")
    ap.add_argument("--station", default=os.environ.get("IMP_STATION", "devstation"))
    args = ap.parse_args()
    run_module(TfModule(station=args.station))


if __name__ == "__main__":
    main()
