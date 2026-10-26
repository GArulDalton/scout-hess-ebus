import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_act=None):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        if out_act == 'tanh':
            self.W[-1] *= 0.1
        self.out_act = out_act
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = [x]
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            elif self.out_act == 'tanh':
                h = np.tanh(h)
            self.cache.append(h)
        return h

    def backward(self, gout):
        """returns grads list (W..., b...) and grad wrt input"""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        g = gout
        if self.out_act == 'tanh':
            g = g * (1 - self.cache[-1] ** 2)
        for i in reversed(range(len(self.W))):
            gW[i] = self.cache[i].T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                g = g * (self.cache[i] > 0)
        return gW + gb, g

    def adam(self, grads, lr, clip=10.0):
        self.t += 1
        n = np.sqrt(sum((g ** 2).sum() for g in grads))
        sc = min(1.0, clip / (n + 1e-8))
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            g = g * sc
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t); vh = v / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def copy_from(self, o, tau=1.0):
        for p, q in zip(self.params(), o.params()):
            p *= (1 - tau); p += tau * q


class Agent:
    """TD3 / DDPG family.
    cfg keys: algo ('td3'|'ddpg'), decomposed (bool: separate energy/aging critics),
    lagrangian (bool), w_age (fixed aging weight), proj_aware (bool), hid
    """

    def __init__(self, obs_dim, cfg, seed=0):
        self.cfg = cfg; rng = np.random.default_rng(seed); self.rng = rng
        h = cfg.get('hid', 64)
        self.actor = MLP([obs_dim, h, h, 1], rng, 'tanh'); self.actor_t = MLP([obs_dim, h, h, 1], rng, 'tanh')
        self.actor_t.copy_from(self.actor)
        ncomp = 2 if cfg['decomposed'] else 1
        ntwin = 2 if cfg['algo'] == 'td3' else 1
        self.Q = [[MLP([obs_dim + 1, h, h, 1], rng) for _ in range(ntwin)] for _ in range(ncomp)]
        self.Qt = [[MLP([obs_dim + 1, h, h, 1], rng) for _ in range(ntwin)] for _ in range(ncomp)]
        for c in range(ncomp):
            for j in range(ntwin):
                self.Qt[c][j].copy_from(self.Q[c][j])
        self.lam = 0.0
        self.gamma = cfg.get('gamma', 0.98); self.tau = 0.005
        self.it = 0
        cap = 200000; self.cap = cap; self.n = 0; self.ptr = 0
        self.S = np.zeros((cap, obs_dim)); self.A = np.zeros((cap, 1)); self.R = np.zeros((cap, 2))
        self.S2 = np.zeros((cap, obs_dim)); self.D = np.zeros((cap, 1))
        self.B = np.tile(np.array([[-1.0, 1.0, -1.0, 1.0]]), (cap, 1))   # feasible action bounds at s and s'

    def act(self, s, noise=0.0):
        a = self.actor.forward(s[None])[0, 0]
        if noise > 0:
            a += noise * self.rng.standard_normal()
        return float(np.clip(a, -1, 1))

    def store(self, s, a, re, ra, s2, d, bounds=None):
        i = self.ptr
        self.B[i] = bounds if bounds is not None else (-1, 1, -1, 1)
        self.S[i] = s; self.A[i] = a; self.R[i] = (re, ra); self.S2[i] = s2; self.D[i] = d
        self.ptr = (i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def weights(self):
        """weight of each reward component inside the actor objective"""
        w_age = self.cfg['w_age'] * (1.0 + self.lam)
        return (1.0, w_age)

    def update(self, bs=100):
        if self.n < 2000:
            return
        self.it += 1
        idx = self.rng.integers(0, self.n, bs)
        S, A, R, S2, D = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]
        LO, HI, LO2, HI2 = (self.B[idx, j:j + 1] for j in range(4))
        cfg = self.cfg
        a2 = self.actor_t.forward(S2)
        if cfg['algo'] == 'td3':
            a2 = a2 + np.clip(0.2 * self.rng.standard_normal(a2.shape), -0.5, 0.5)
        a2 = np.clip(a2, LO2, HI2)          # target action passes through the same projection
        X2 = np.hstack([S2, a2]); X = np.hstack([S, A])
        if cfg['decomposed']:
            comps = [R[:, :1], R[:, 1:2]]
        else:
            w = self.weights(); comps = [R[:, :1] * w[0] + R[:, 1:2] * w[1]]
        for c, r in enumerate(comps):
            qs = [q.forward(X2) for q in self.Qt[c]]
            # rewards are negative costs: pessimistic = min
            y = r + self.gamma * (1 - D) * np.minimum.reduce(qs)
            for q in self.Q[c]:
                out = q.forward(X)
                g, _ = q.backward(2 * (out - y) / bs)
                q.adam(g, 1e-3)
        if self.it % (2 if cfg['algo'] == 'td3' else 1) == 0:
            a_raw = self.actor.forward(S)
            a = np.clip(a_raw, LO, HI)       # critic only ever evaluated on feasible actions
            inside = ((a_raw >= LO) & (a_raw <= HI)).astype(float)
            Xa = np.hstack([S, a])
            gin_total = np.zeros_like(a)
            wts = self.weights() if cfg['decomposed'] else (1.0,)
            for c in range(len(self.Q)):
                q = self.Q[c][0]
                q.forward(Xa)
                _, gin = q.backward(-np.ones((bs, 1)) / bs * wts[c])
                gin_total += gin[:, -1:]
            gin_total = gin_total * inside + 2.0 * self.cfg.get('kappa', 1.0) * (a_raw - a) / bs
            ga, _ = self.actor.backward(gin_total)
            self.actor.adam(ga, 3e-4)
            self.actor_t.copy_from(self.actor, self.tau)
            for c in range(len(self.Q)):
                for j in range(len(self.Q[c])):
                    self.Qt[c][j].copy_from(self.Q[c][j], self.tau)
