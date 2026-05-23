"""Implementation for `vision_engine.core.registry`."""

from typing import Dict, Type

from vision_engine.core.module_base import VisionModule
from vision_engine.modules.blob_detection.module import BlobDetectionModule
from vision_engine.modules.camera_preview.module import CameraPreviewModule
from vision_engine.modules.cuboid_pose_6d.module import CuboidPose6DModule
from vision_engine.modules.feature_matching.module import FeatureMatchingModule
from vision_engine.modules.megapose_bin_picking.module import MegaPoseBinPickingModule
from vision_engine.modules.opt_sift.module import OptSiftModule
from vision_engine.modules.object_proposals.module import ObjectProposalsModule
from vision_engine.modules.ppf_icp_bin_picking.module import PpfIcpBinPickingModule
from vision_engine.modules.tamplate_matching_sift.module import (
    TamplateMatchingSiftModule,
)

# Import modules here
from vision_engine.modules.template_matching.module import TemplateMatchingModule

MODULE_REGISTRY: Dict[str, Type[VisionModule]] = {
    "template_matching": TemplateMatchingModule,
    "tamplate_matching_sift": TamplateMatchingSiftModule,
    "opt_sift": OptSiftModule,
    "blob_detection": BlobDetectionModule,
    "camera_preview": CameraPreviewModule,
    "object_proposals": ObjectProposalsModule,
    "feature_matching": FeatureMatchingModule,
    "cuboid_pose_6d": CuboidPose6DModule,
    "megapose_bin_picking": MegaPoseBinPickingModule,
    "ppf_icp_bin_picking": PpfIcpBinPickingModule,
    # add more here
}


def get_module_class(name: str) -> Type[VisionModule]:
    if name not in MODULE_REGISTRY:
        raise ValueError(f"Unknown vision module: {name}")
    return MODULE_REGISTRY[name]
