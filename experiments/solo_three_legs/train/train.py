#!/usr/bin/env python3
"""PPO training of the SOLO-12 walking policy (CPU MuJoCo physics, GPU PPO update).

    python train/train.py --out runs/solo_walk --iters 1500
    python train/train.py --out runs/solo_walk --iters 2000 --resume runs/solo_walk/model_1500.pt

Optional TRAINED 3-leg reference (fine-tune on a locked front-left leg):
    python train/train.py --out runs/solo_3leg_ft --fault locked FL --resume runs/solo_walk/model_final.pt --iters 600
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
from solo_env import SoloVecEnv, PRIV_DIM, RewardCfg  # noqa: E402
from solo_config import OBS_DIM  # noqa: E402
from ppo import ActorCritic, PPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nthread", type=int, default=16)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--fault", nargs=2, default=None, metavar=("KIND", "LEG"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_every", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SoloVecEnv(num_envs=a.num_envs, seed=a.seed, nthread=a.nthread, fault=tuple(a.fault) if a.fault else None)
    ac = ActorCritic(OBS_DIM, PRIV_DIM, 12).to(dev)
    start_it = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        ac.load_state_dict(ck["model"])
        start_it = ck.get("iter", 0) if not a.fault else 0
        if a.fault:  # fine-tuning: restart exploration noise at a moderate level
            with torch.no_grad():
                ac.log_std[:] = np.log(0.5)
    ppo = PPO(ac, a.num_envs, a.horizon, OBS_DIM, PRIV_DIM, 12, dev, lr=a.lr)
    json.dump(dict(vars(a), reward={k: v for k, v in vars(RewardCfg).items() if not k.startswith("_")}),
              open(os.path.join(a.out, "train_config.json"), "w"), indent=2)

    obs, priv = env.reset_all()
    to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
    obs_t, priv_t = to_t(obs), to_t(priv)
    log = open(os.path.join(a.out, "log.csv"), "a", newline="")
    w = csv.writer(log)
    header_done = start_it != 0
    ep_rets, ep_lens, falls = [], [], []
    t0 = time.time()
    steps = 0
    for it in range(start_it, start_it + a.iters):
        ti = time.time()
        rsum = 0.0
        terms_acc = {}
        for t in range(a.horizon):
            act = ppo.act_step(obs_t, priv_t)
            obs, priv, rew, done, info = env.step(act.cpu().numpy())
            rsum += rew.mean()
            for k, v in env.last_terms.items():
                terms_acc[k] = terms_acc.get(k, 0.0) + v / a.horizon
            tv = ppo.val[ppo.t] * to_t(info["timeout"].astype(np.float32))
            ppo.store(to_t(rew), to_t(done.astype(np.float32)), tv)
            obs_t, priv_t = to_t(obs), to_t(priv)
            if "ep_ret" in info:
                ep_rets += list(info["ep_ret"]); ep_lens += list(info["ep_len"]); falls += list(info["fell"])
        t_col = time.time() - ti
        st = ppo.update(obs_t, priv_t)
        steps += a.num_envs * a.horizon
        fps = a.num_envs * a.horizon / (time.time() - ti)
        if it % 10 == 0 or it == start_it + a.iters - 1:
            er = np.mean(ep_rets[-500:]) if ep_rets else 0.0
            el = np.mean(ep_lens[-500:]) * 0.02 if ep_lens else 0.0
            fr = np.mean(falls[-500:]) if falls else 0.0
            std = torch.exp(ac.log_std).mean().item()
            print(f"it {it:5d}  ret {er:7.3f}  len {el:5.1f}s  fall {fr:.2f}  r/step {rsum / a.horizon:.4f}  std {std:.3f}  "
                  f"lr {st['lr']:.1e}  kl {st['kl']:.4f}  fps {fps:7.0f} (collect {t_col:.2f}s)  "
                  f"lin {terms_acc.get('track_lin', 0):.3f} ang {terms_acc.get('track_ang', 0):.3f} gait {terms_acc.get('gait_contact', 0):.3f}",
                  flush=True)
            keys = sorted(terms_acc)
            if not header_done:
                w.writerow(["iter", "steps", "time_s", "mean_ep_ret", "mean_ep_len_s", "fall_rate", "mean_rew_step", "std", "lr",
                            "kl", "loss_v", "fps"] + keys)
                header_done = True
            w.writerow([it, steps, round(time.time() - t0, 1), er, el, fr, rsum / a.horizon, std, st["lr"], st["kl"], st["loss_v"], fps]
                       + [terms_acc[k] for k in keys])
            log.flush()
        if (it + 1) % a.save_every == 0 or it == start_it + a.iters - 1:
            ck = dict(model=ac.state_dict(), iter=it + 1, args=vars(a))
            torch.save(ck, os.path.join(a.out, f"model_{it + 1}.pt"))
            torch.save(ck, os.path.join(a.out, "model_last.pt"))
    torch.save(dict(model=ac.state_dict(), iter=start_it + a.iters, args=vars(a)), os.path.join(a.out, "model_final.pt"))
    env.close()
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
