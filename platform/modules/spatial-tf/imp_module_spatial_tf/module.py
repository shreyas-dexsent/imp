"""Bus-resident frame graph node.

Subscribes ``imp/<station>/tf`` (``TfEdge`` messages) and accumulates them
into a :class:`TfGraph`. Republishes a small status ``Scalar`` (number of
known frames) on ``imp/<station>/motion/tf/frames`` so an external observer
can see the graph is alive without re-running the lookup.

Any other module that needs frame composition can either run an in-process
``TfGraph`` (subscribe the same key directly) or, once the ``tf-lookup``
service lands in P8, call it as a queryable.
"""

from __future__ import annotations

from typing import Dict

from imp_sdk import Input, Module, Output, QosClass, keyexpr
from imp_sdk.schemas import imp_pb2

from .graph import TfGraph


class TfModule(Module):
    name = "spatial-tf"

    def __init__(self, station: str):
        self.station = station
        self.tf_key = keyexpr.tf(station)
        self.frames_key = keyexpr.motion(station, "tf", "frames")
        self.graph = TfGraph()

    def inputs(self):
        return [Input("edge", self.tf_key, imp_pb2.TfEdge)]

    def outputs(self):
        return [Output("frames", self.frames_key, imp_pb2.Scalar, QosClass.TELEMETRY)]

    def configure(self) -> None:
        # Nothing to load; the graph fills from the topic.
        pass

    def compute(self, latest: Dict[str, object]) -> Dict[str, object]:
        edge: imp_pb2.TfEdge = latest["edge"]  # type: ignore[assignment]
        if len(edge.matrix) != 16:
            return {}
        m = [edge.matrix[i * 4 : (i + 1) * 4] for i in range(4)]
        self.graph.add_edge(edge.parent_frame, edge.child_frame, m)
        return {
            "frames": imp_pb2.Scalar(
                header=imp_pb2.Header(schema="imp.Scalar/1"),
                value=float(len(self.graph.frames())),
            )
        }
