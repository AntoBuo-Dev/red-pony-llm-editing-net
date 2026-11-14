#!/usr/bin/env python3
"""Copy the edited weights into the TorchScript export (run with a Python that has torch).

    python export_torchscript.py <source policy.pt> <edited policy dir>
"""
import hashlib, json, sys
import numpy as np, torch

src, out = sys.argv[1], sys.argv[2]
m = torch.jit.load(src, map_location="cpu")
w = np.load(out + "/weights.npz")
sd = m.state_dict()
assert set(sd) == set(w.files), (set(sd), set(w.files))
with torch.no_grad():
    for k, p in m.named_parameters(): p.copy_(torch.from_numpy(w[k]))
m.save(out + "/exported/policy.pt")
m2 = torch.jit.load(out + "/exported/policy.pt", map_location="cpu")
io = np.load(out + "/check_io.npz")
err = float((m2(torch.from_numpy(io["obs"])).detach().numpy() - io["actions"]).__abs__().max())
meta = json.load(open(out + "/weight_edit.json"))
meta["torchscript_sha256"] = hashlib.sha256(open(out + "/exported/policy.pt", "rb").read()).hexdigest()
meta["max_onnx_torchscript_error"] = err
json.dump(meta, open(out + "/weight_edit.json", "w"), indent=2)
print("TorchScript written, max |onnx - torchscript| =", err)
assert err < 1e-4
