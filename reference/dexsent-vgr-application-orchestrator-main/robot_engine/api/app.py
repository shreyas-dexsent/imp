from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from robot_engine.api.routers import (
    assets,
    cells,
    collision,
    contexts,
    core_math,
    frames,
    kinematics,
    motion,
    planning,
    sim,
    trajectory,
)

_TAGS = [
    {"name": "contexts", "description": "Session context lifecycle"},
    {"name": "cells", "description": "IMP orchestration cells, scene state, test cases, and live streams"},
    {"name": "core_math", "description": "Stateless math: transforms, rotations, Lie groups, interpolation"},
    {"name": "frames", "description": "Context-backed coordinate frame graph"},
    {"name": "assets", "description": "Robot/gripper/collision mesh loading"},
    {"name": "collision", "description": "Collision world, matrix, narrowphase, distance, continuous"},
    {"name": "kinematics", "description": "FK, Jacobian, IK (multiple backends)"},
    {"name": "planning", "description": "Sampling-based and deterministic path planners"},
    {"name": "sim", "description": "URDF-style simulation runtime: joint state and trajectory playback"},
    {"name": "trajectory", "description": "Cubic, quintic, trapezoidal, S-curve generation and validation"},
    {"name": "motion", "description": "High-level motion primitives: MoveJ, MoveL, approach/retreat, pick/place"},
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="robot_engine API",
        version="1.0.0",
        description=(
            "FastAPI service exposing the robot_engine algorithm modules. "
            "Session state (robot models, collision worlds, grasp libraries) lives "
            "in named contexts created via POST /api/v1/contexts."
        ),
        openapi_tags=_TAGS,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(contexts.router, prefix=prefix)
    app.include_router(cells.router, prefix=prefix)
    app.include_router(core_math.router, prefix=prefix)
    app.include_router(frames.router, prefix=prefix)
    app.include_router(assets.router, prefix=prefix)
    app.include_router(collision.router, prefix=prefix)
    app.include_router(kinematics.router, prefix=prefix)
    app.include_router(planning.router, prefix=prefix)
    app.include_router(sim.router, prefix=prefix)
    app.include_router(trajectory.router, prefix=prefix)
    app.include_router(motion.router, prefix=prefix)

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "service": "robot_engine"}

    @app.get("/ready", tags=["health"])
    def ready():
        return {"status": "ready", "service": "robot_engine"}

    return app
