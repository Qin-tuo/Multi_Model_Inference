# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream B2 networks for CPU export/reference, never application inference.

Only their inference entrypoints are assembled here. No learned layer, keypoint
head, feature dimension or trained weight is changed.
"""

from copy import deepcopy
import importlib.util
import runpy
import subprocess
import sys
import types

from project_paths import ROOT
from deploy import runtime


def verify_sources(entry):
    revision = entry.get("native_repository_revision")
    if revision:
        source = ROOT / entry["source_directory"]
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if head != revision or dirty:
            raise ValueError("Native reference checkout differs from its pin")
    for record in [entry["source_checkpoint"], *entry["native_sources"].values()]:
        path = ROOT / record["path"]
        if runtime.sha256(path) != record["sha256"]:
            raise ValueError("Upstream source differs: " + str(path))


def checkpoint(path):
    import numpy as np
    import torch

    # Upstream checkpoints also contain historical NumPy metric/meta arrays.
    with torch.serialization.safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.float64)),
            type(np.dtype(np.float32)),
            type(np.dtype(np.int64)),
            type(np.dtype(np.uint8)),
        ]
    ):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    state = loaded.get("state_dict", loaded)
    return {key.removeprefix("module."): value for key, value in state.items()}


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(role, entry):
    """Return the unmodified network and any upstream image/codec reference."""

    verify_sources(entry)
    state = checkpoint(ROOT / entry["source_checkpoint"]["path"])
    extras = {}
    if role == "person_reid":
        source = entry["native_sources"]["torchreid/models/osnet.py"]
        module = source_module(ROOT / source["path"], "b2_official_osnet")
        model = module.osnet_x1_0(
            num_classes=state["classifier.weight"].shape[0], pretrained=False
        )
    elif role == "face_identity":
        folder = ROOT / entry["source_directory"]
        sys.path.insert(0, str(folder))
        from nets.arcface import Arcface

        model = Arcface(backbone="mobilenetv1", pretrained=False, mode="predict")
        extras["image_utils"] = source_module(
            folder / "utils/utils.py", "b2_arcface_image_reference"
        )
    elif role == "body_hand_pose":
        model, extras = pose_model(entry)
    else:
        raise ValueError("No standard network factory for " + role)
    loaded = model.load_state_dict(state, strict=True)
    model = model.float().eval()
    return (
        model,
        extras,
        {
            "strict_state_keys": len(state),
            "missing_keys": loaded.missing_keys,
            "unexpected_keys": loaded.unexpected_keys,
            "source_checkpoint_sha256": entry["source_checkpoint"]["sha256"],
        },
    )


def pose_model(entry):
    """Use actual MMPose layers and transforms, skipping training aggregators."""
    import numpy as np
    import torch

    source = ROOT / entry["source_directory"]
    sys.path.insert(0, str(source))
    namespaces = (
        "mmpose.models",
        "mmpose.models.backbones",
        "mmpose.models.necks",
        "mmpose.models.heads",
        "mmpose.models.heads.coord_cls_heads",
        "mmpose.evaluation",
        "mmpose.codecs",
        "mmpose.datasets",
        "mmpose.datasets.transforms",
    )
    for name in namespaces:
        if name in sys.modules:
            raise RuntimeError("Pose native reference needs a fresh process")
        package = types.ModuleType(name)
        package.__path__ = [str(source / name.replace(".", "/"))]
        sys.modules[name] = package
    from mmpose.models.heads.base_head import BaseHead

    sys.modules["mmpose.models.heads"].BaseHead = BaseHead
    from mmpose.models.backbones.cspnext import CSPNeXt
    from mmpose.models.heads.coord_cls_heads.rtmcc_head import RTMCCHead
    from mmpose.models.heads.coord_cls_heads.rtmw_head import RTMWHead
    from mmpose.models.necks.cspnext_pafpn import CSPNeXtPAFPN
    from mmpose.models.data_preprocessors import PoseDataPreprocessor
    from mmpose.datasets.transforms.common_transforms import GetBBoxCenterScale
    from mmpose.datasets.transforms.topdown_transforms import TopdownAffine
    from mmpose.structures import PoseDataSample
    from mmpose.codecs.simcc_label import SimCCLabel

    config = deepcopy(runpy.run_path(str(ROOT / entry["config_file"]))["model"])

    def options(name):
        return {
            key: value
            for key, value in config[name].items()
            if key not in ("type", "_scope_")
        }

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CSPNeXt(**options("backbone"))
            if "neck" in config:
                self.neck = CSPNeXtPAFPN(**options("neck"))
            head = options("head")
            # Training loss and decoding have no inference parameters. Strict
            # state loading verifies that no learned state was omitted.
            head.update(loss=[], decoder=None)
            self.head = {"RTMCCHead": RTMCCHead, "RTMWHead": RTMWHead}[
                config["head"]["type"]
            ](**head)

        def forward(self, inputs):
            features = self.backbone(inputs)
            if hasattr(self, "neck"):
                features = self.neck(features)
            return self.head(features)

    model = Network()
    processor = PoseDataPreprocessor(**options("data_preprocessor"))
    codec = SimCCLabel(
        **{k: v for k, v in config["head"]["decoder"].items() if k != "type"}
    )
    bbox = GetBBoxCenterScale()
    input_size = tuple(config["head"]["input_size"])
    affine = TopdownAffine(input_size=input_size)

    def prepare(image, box):
        values = affine(bbox({"img": image, "bbox": np.array(box, np.float32)[None]}))
        tensor = torch.from_numpy(
            np.ascontiguousarray(values["img"].transpose(2, 0, 1))
        )
        normalized = processor(
            {"inputs": [tensor], "data_samples": [PoseDataSample()]}, training=False
        )["inputs"]
        return normalized, values["bbox_center"][0], values["bbox_scale"][0]

    return model, {"prepare": prepare, "codec": codec, "input_size": input_size}
