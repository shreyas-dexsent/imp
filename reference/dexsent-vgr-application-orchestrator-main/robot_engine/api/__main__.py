"""CLI entrypoint: python -m robot_engine.api"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Start the robot_engine FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "robot_engine.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
