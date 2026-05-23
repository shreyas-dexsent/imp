"""Implementation for `vision_engine.core.pipeline`."""

import time
from typing import Any, Dict, List

from vision_engine.core.logging import get_logger
from vision_engine.core.registry import get_module_class

log = get_logger("pipeline")


class VisionPipeline:
    def __init__(self, pipeline_cfg: List[Dict[str, Any]]):
        self.modules = []

        for m in pipeline_cfg:
            if not m.get("enabled", True):
                continue

            name = m["name"]
            params = m.get("params", {})

            cls = get_module_class(name)
            module = cls(name=name, params=params)

            self.modules.append(module)
            log.info(f"Loaded module: {name}")

    def run(self, frame_bundle) -> Dict[str, Any]:
        results = {}
        start = time.time()

        for module in self.modules:
            t0 = time.time()
            out = module.run(frame_bundle)
            results[module.name] = {
                "result": out,
                "latency_ms": (time.time() - t0) * 1000,
            }

        results["_pipeline_latency_ms"] = (time.time() - start) * 1000
        return results

    def stop(self):
        for m in self.modules:
            m.stop()
