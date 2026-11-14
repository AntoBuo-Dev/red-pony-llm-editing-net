"""Read / write the policy as a plain ONNX MLP (Gemm + Elu), and evaluate it in numpy.

The graph is   obs[1,47] -> Gemm(mlp.0) -> Elu -> Gemm(mlp.2) -> Elu -> Gemm(mlp.4) -> Elu -> Gemm(mlp.6) -> actions[1,12]
Initializer names mlp.{0,2,4,6}.{weight,bias}; weight has shape (out, in) (Gemm transB=1).
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

LAYER_IDS = (0, 2, 4, 6)


def load_onnx_layers(path):
    m = onnx.load(path)
    init = {t.name: numpy_helper.to_array(t).copy() for t in m.graph.initializer}
    return [(init[f"mlp.{i}.weight"].astype(np.float32), init[f"mlp.{i}.bias"].astype(np.float32)) for i in LAYER_IDS]


def write_onnx_mlp(layers, path, doc=""):
    nodes, inits = [], []
    x = "obs"
    for k, (W, b) in enumerate(layers):
        i = LAYER_IDS[k]
        inits += [numpy_helper.from_array(np.asarray(W, np.float32), f"mlp.{i}.weight"),
                  numpy_helper.from_array(np.asarray(b, np.float32), f"mlp.{i}.bias")]
        y = "actions" if k == len(layers) - 1 else f"pre{k}"
        nodes.append(helper.make_node("Gemm", [x, f"mlp.{i}.weight", f"mlp.{i}.bias"], [y], transB=1, name=f"gemm{k}"))
        if k < len(layers) - 1:
            nodes.append(helper.make_node("Elu", [y], [f"h{k}"], alpha=1.0, name=f"elu{k}"))
            x = f"h{k}"
    n_in, n_out = layers[0][0].shape[1], layers[-1][0].shape[0]
    g = helper.make_graph(nodes, "policy", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", n_in])],
                          [helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["batch", n_out])], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="red-pony")
    model.ir_version = 8
    model.doc_string = doc
    onnx.checker.check_model(model)
    onnx.save(model, path)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NumpyPolicy:
    """MLP with ELU evaluated in numpy from a list of (W, b) (float64 internally)."""

    def __init__(self, layers):
        self.layers = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in layers]

    def __call__(self, obs, return_hidden=False):
        x = np.asarray(obs, np.float64)
        hs = []
        n = len(self.layers)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < n - 1:
                x = elu(x)
                hs.append(x)
        return (x, hs) if return_hidden else x

    def preacts(self, obs):
        """pre-activations of every layer (list of arrays)."""
        x = np.asarray(obs, np.float64)
        out = []
        for i, (W, b) in enumerate(self.layers):
            z = x @ W.T + b
            out.append(z)
            x = elu(z) if i < len(self.layers) - 1 else z
        return out
