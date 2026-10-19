# SPDX-License-Identifier: Apache-2.0
"""Minimal, audited OMZ conversion repairs using NVIDIA ONNX GraphSurgeon.

These are representation/compatibility fixes, not precision or performance
tuning. Original IR/ONNX files remain intact; native-reference parity is required.
"""

from pathlib import Path


def normalize_omz(raw_path, source_xml):
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    import openvino as ov
    from onnx import numpy_helper

    raw_path = Path(raw_path)
    target = raw_path.with_name(raw_path.stem + "-trt.onnx")
    if target.exists():
        raise FileExistsError(target)
    model = onnx.load(raw_path)
    graph = gs.import_onnx(model)
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    native = ov.Core().read_model(str(source_xml))
    native_ops = {node.get_friendly_name(): node for node in native.get_ops()}
    graph_outputs = {tensor.name for tensor in graph.outputs}
    changes = []
    for node in graph.nodes:
        if node.op == "MaxPool" and len(node.outputs) > 1:
            if any(
                value.outputs or value.name in graph_outputs
                for value in node.outputs[1:]
            ):
                raise ValueError(
                    "Cannot discard consumed MaxPool indices: " + node.name
                )
            changes.append(
                {
                    "node": node.name,
                    "change": "remove unused optional MaxPool index outputs",
                    "removed": [v.name for v in node.outputs[1:]],
                }
            )
            node.outputs = node.outputs[:1]
        if node.op == "Slice" and len(node.inputs) >= 4 and node.inputs[3].is_empty():
            starts = constants.get(node.inputs[1].name)
            if starts is None or starts.ndim != 1:
                raise ValueError("Slice starts are not a reviewed constant vector")
            original = native_ops.get(node.name)
            if original is None or original.get_type_name() != "StridedSlice":
                raise ValueError("Missing original StridedSlice provenance")
            attributes = original.get_attributes()
            if any(
                any(attributes.get(key, []))
                for key in ("new_axis_mask", "shrink_axis_mask", "ellipsis_mask")
            ):
                raise ValueError("Unsupported StridedSlice axis-mask contract")
            rank = len(original.input_value(0).get_shape())
            if len(starts) > rank:
                raise ValueError("Invalid prefix Slice rank")
            # OpenVINO slices the specified leading dimensions and leaves the
            # remaining dimensions intact. Make those axes explicit in ONNX.
            axes = np.arange(len(starts), dtype=np.int64)
            node.inputs[3] = gs.Constant(node.name + "/explicit_axes", values=axes)
            changes.append(
                {
                    "node": node.name,
                    "change": "explicit leading Slice axes",
                    "axes": axes.tolist(),
                    "native_input_rank": rank,
                }
            )
    if not changes:
        return raw_path, []
    graph.cleanup().toposort()
    result = gs.export_onnx(graph)
    onnx.checker.check_model(result)
    onnx.save(result, str(target))
    return target, changes
