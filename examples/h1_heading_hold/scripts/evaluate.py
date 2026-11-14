import numpy as np, pickle, sys, math
from multiprocessing import Pool
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_harness import *
DEPLOY = REPO + "/policies/h1_2_squat/2026-09-13_21-27-39/params/deploy.yaml"
CASES = [(vx,vy,h) for h in (0.98,0.85,0.74,0.71) for vx,vy in ((0.3,0),(0.5,0),(0.8,0),(1.0,0),(-0.3,0),(0,0.3),(0,-0.3))]
def load(which):
    if os.path.isdir(which): return NumpyPolicy(load_onnx_layers(which + "/exported/policy.onnx"))
    if which.endswith(".onnx"): return NumpyPolicy(load_onnx_layers(which))
    raise SystemExit("not a policy: " + which)
def _run(a):
    which,(vx,vy,h),T,kw = a
    r = run_segment(load(which), DEPLOY, vx=vx, vy=vy, height=h, T=T, record=True, **kw)
    tr = r.pop("traj"); act=r.pop("act"); r.pop("obs")
    t = np.arange(len(tr))*0.02; n0=int(6/0.02)
    # straightness: rms distance of the path from its own best-fit straight line (after 6 s)
    p = tr[n0:, :2]; pc = p-p.mean(0); u,s,vt = np.linalg.svd(pc, full_matrices=False)
    r["line_rms"] = float(s[1]/math.sqrt(len(p))); r["yaw_rate"] = float(np.degrees(np.polyfit(t[n0:], tr[n0:,2],1)[0]))
    r["dact"] = float(np.abs(np.diff(act[:, :21],axis=0)).mean()); r["wrist"]=float(np.abs(act[:,25:27]).max())
    return r
def table(which, T=30.0, kw={}):
    with Pool(14) as p: R = p.map(_run, [(which,c,T,kw) for c in CASES])
    print(f"== {which} {kw} (T={T}s)")
    print("   vx    vy    h |   fwd     lat   yaw_end  yaw_rate  line_rms  dact  |reg|max")
    for r in R: print(f" {r['vx']:5.2f} {r['vy']:5.2f} {r['height']:.2f} | {r['fwd']:6.2f} {r['lat']:7.3f} {r['yaw_deg']:7.2f}° {r['yaw_rate']:7.3f}°/s {100*r['line_rms']:6.1f}cm {r['dact']:.4f} {r['wrist']:.3f} {'FELL' if r['fell'] else ''}")
    print("   mean |yaw_end| %.2f°  mean |yaw_rate| %.3f°/s  mean line_rms %.1f cm  mean dact %.4f  falls %d"%(np.mean([abs(r['yaw_deg']) for r in R]), np.mean([abs(r['yaw_rate']) for r in R]), 100*np.mean([r['line_rms'] for r in R]), np.mean([r['dact'] for r in R]), sum(r['fell'] for r in R)))
    return R
if __name__=="__main__":
    for w in sys.argv[1:]: table(w)
