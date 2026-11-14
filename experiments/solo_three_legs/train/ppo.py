"""Minimal PPO (rsl_rl style: clipped surrogate, GAE, adaptive learning rate from the KL)."""
import torch
import torch.nn as nn


def mlp(sizes, act=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim, actor_hidden=(512, 256, 128), critic_hidden=(512, 256, 128), init_std=1.0):
        super().__init__()
        self.actor = mlp([obs_dim, *actor_hidden, act_dim])
        self.critic = mlp([obs_dim + priv_dim, *critic_hidden, 1])
        self.log_std = nn.Parameter(torch.log(torch.ones(act_dim) * init_std))

    def dist(self, obs):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, torch.exp(self.log_std).expand_as(mu))

    def value(self, obs, priv):
        return self.critic(torch.cat([obs, priv], -1)).squeeze(-1)


class PPO:
    def __init__(self, ac, num_envs, horizon, obs_dim, priv_dim, act_dim, device, lr=1e-3, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=5, minibatches=4, ent_coef=0.005, vf_coef=1.0, max_grad=1.0, desired_kl=0.01):
        self.ac, self.device = ac, device
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.epochs, self.mb, self.ent, self.vf, self.max_grad, self.kl_t = epochs, minibatches, ent_coef, vf_coef, max_grad, desired_kl
        T, N = horizon, num_envs
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs, self.priv, self.act = z(T, N, obs_dim), z(T, N, priv_dim), z(T, N, act_dim)
        self.rew, self.done, self.val, self.logp = z(T, N), z(T, N), z(T, N), z(T, N)
        self.mu, self.sigma = z(T, N, act_dim), z(T, N, act_dim)
        self.t = 0

    @torch.no_grad()
    def act_step(self, obs, priv):
        d = self.ac.dist(obs)
        a = d.sample()
        self.obs[self.t], self.priv[self.t], self.act[self.t] = obs, priv, a
        self.val[self.t] = self.ac.value(obs, priv)
        self.logp[self.t] = d.log_prob(a).sum(-1)
        self.mu[self.t], self.sigma[self.t] = d.mean, d.stddev
        return a

    @torch.no_grad()
    def store(self, rew, done, timeout_value=None):
        if timeout_value is not None:
            rew = rew + self.gamma * timeout_value
        self.rew[self.t], self.done[self.t] = rew, done
        self.t += 1

    def update(self, last_obs, last_priv):
        with torch.no_grad():
            last_v = self.ac.value(last_obs, last_priv)
            adv = torch.zeros_like(self.rew)
            gae = 0
            for t in reversed(range(self.t)):
                nv = last_v if t == self.t - 1 else self.val[t + 1]
                nd = 1.0 - self.done[t]
                delta = self.rew[t] + self.gamma * nv * nd - self.val[t]
                gae = delta + self.gamma * self.lam * nd * gae
                adv[t] = gae
            ret = adv + self.val
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = self.t * self.obs.shape[1]
        f = lambda x: x[:self.t].reshape(B, *x.shape[2:])
        O, P, A, LP, R, ADV, V, MU, SG = map(f, (self.obs, self.priv, self.act, self.logp, ret, adv, self.val, self.mu, self.sigma))
        mbs = B // self.mb
        stats = dict(loss_pi=0.0, loss_v=0.0, kl=0.0, ent=0.0)
        n = 0
        for _ in range(self.epochs):
            perm = torch.randperm(B, device=self.device)
            for i in range(self.mb):
                idx = perm[i * mbs:(i + 1) * mbs]
                d = self.ac.dist(O[idx])
                logp = d.log_prob(A[idx]).sum(-1)
                ent = d.entropy().sum(-1).mean()
                with torch.no_grad():   # KL(old || new) for the adaptive learning rate
                    s_new, s_old = d.stddev, SG[idx]
                    kl = (torch.log(s_new / s_old) + (s_old ** 2 + (MU[idx] - d.mean) ** 2) / (2 * s_new ** 2) - 0.5).sum(-1).mean()
                    if kl > 2 * self.kl_t:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0 < kl < self.kl_t / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                ratio = torch.exp(logp - LP[idx])
                s1 = -ADV[idx] * ratio
                s2 = -ADV[idx] * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                loss_pi = torch.max(s1, s2).mean()
                v = self.ac.value(O[idx], P[idx])
                v_clip = V[idx] + (v - V[idx]).clamp(-self.clip, self.clip)
                loss_v = torch.max((v - R[idx]) ** 2, (v_clip - R[idx]) ** 2).mean()
                loss = loss_pi + self.vf * loss_v - self.ent * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
                self.opt.step()
                stats["loss_pi"] += loss_pi.item(); stats["loss_v"] += loss_v.item(); stats["kl"] += kl.item(); stats["ent"] += ent.item()
                n += 1
        self.t = 0
        return {k: v / n for k, v in stats.items()} | {"lr": self.lr}
