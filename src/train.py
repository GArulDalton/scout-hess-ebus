import numpy as np, sys, json, time, pickle
from env import *
from agent import Agent

SCALE = 300.0
DPMAX = 60e3
METHODS = {
    'DDPG':        dict(algo='ddpg', decomposed=False, lagrangian=False, w_age=1.0, residual=False, preview=False, proj_aware=False, soc_pen=True),
    'TD3-E':       dict(algo='td3', decomposed=False, lagrangian=False, w_age=0.0, residual=False, preview=False, proj_aware=False, soc_pen=True),
    'TD3':         dict(algo='td3', decomposed=False, lagrangian=False, w_age=1.0, residual=False, preview=False, proj_aware=False, soc_pen=True),
    'SCOUT':       dict(algo='td3', decomposed=True, lagrangian=True, w_age=1.0, residual=True, preview=True, proj_aware=True, shaping=True),
    'SCOUT-noPrev': dict(algo='td3', decomposed=True, lagrangian=True, w_age=1.0, residual=True, preview=False, proj_aware=True, shaping=True),
    'SCOUT-noRes': dict(algo='td3', decomposed=True, lagrangian=True, w_age=1.0, residual=False, preview=True, proj_aware=True, shaping=True),
    'SCOUT-noLag': dict(algo='td3', decomposed=True, lagrangian=False, w_age=1.0, residual=True, preview=True, proj_aware=True, shaping=True),
    'SCOUT-noProj': dict(algo='td3', decomposed=True, lagrangian=True, w_age=1.0, residual=True, preview=True, proj_aware=False, shaping=True),
    'SCOUT-noShape': dict(algo='td3', decomposed=True, lagrangian=True, w_age=1.0, residual=True, preview=True, proj_aware=True, shaping=False),
}
LIFE_TARGET_KM = 250000.0
PREV_IDX = [7, 8, 9]


def obs_f(o, cfg):
    if not cfg['preview']:
        o = o.copy(); o[PREV_IDX] = 0.0
    return o


def phi(env):
    return (env.Esc - E_SC_MIN) / 3.6e6 * PRICE_E / CHG_EFF * ETA_C


def exo_aging(env, Pd):
    """Aging cost if the battery alone met the (exogenous) demand at ambient temperature and SOC 0.5.
    Depends only on exogenous quantities, so subtracting it shifts every return by a policy-independent constant."""
    Voc = ocv(0.5); R = r_pack(env.Tamb)
    Ib = (Voc - np.sqrt(max(Voc ** 2 - 4 * R * Pd, 0.0))) / (2 * R)
    Ic = np.clip(Ib, -I_BAT_CHG, I_BAT_DIS) / NP
    loss = EOL_LOSS * severity(abs(Ic) / QCELL, env.Tamb, 0.5) * abs(Ic) / 3600 / GAMMA_NOM
    return loss / EOL_LOSS * PACK_COST


def bounds(env, cfg):
    """Feasible interval of the normalised action at the current state."""
    if not cfg['proj_aware']:
        return (-1.0, 1.0)
    lo, hi = env.feasible_conv(env.P[env.k])
    base = env.rule_power() if cfg['residual'] else 0.0
    sc = DPMAX if cfg['residual'] else 200e3
    l = float(np.clip((lo - base) / sc, -1, 1)); h = float(np.clip((hi - base) / sc, -1, 1))
    return (min(l, h), max(l, h))


def command(env, a, cfg):
    return env.rule_power() + a * DPMAX if cfg['residual'] else a * 200e3


def run_episode(env, agent, cfg, noise=0.0, train=False):
    o = obs_f(env.reset(), cfg); done = False
    while not done:
        phi_prev = phi(env)
        Pd_k = env.P[env.k]
        a = agent.act(o, noise)
        bnd = bounds(env, cfg)
        cmd = command(env, a, cfg)
        o2, ec, ac, sh, done = env.step(cmd)
        o2 = obs_f(o2, cfg)
        if train:
            a_store = a
            pen = 0.0
            if cfg['proj_aware']:
                # store the executed (projected) action and penalise the projection distance
                base = env.log['Pconv'][-1]
                a_exec = (base - (cmd - a * DPMAX)) / DPMAX if cfg['residual'] else base / 200e3
                a_store = float(np.clip(a_exec, -1, 1))
                pen = 0.05 * sh / 200e3
            re = -SCALE * ec - pen
            if cfg.get('shaping'):
                # (i) remove the exogenous traction-demand energy (independent of the policy)
                re += SCALE * Pd_k / 3.6e6 * PRICE_E / CHG_EFF
                # (ii) potential-based shaping: SC energy valued at grid price (policy invariant)
                re += SCALE * (phi(env) - phi_prev)
            if cfg.get('soc_pen'):
                # conventional DRL-EMS reward term keeping the SC near a reference charge
                re -= SCALE * 0.004 * (env.sc_soc() - 0.75) ** 2
            ra = -SCALE * ac
            if cfg.get('shaping'):
                ra += SCALE * exo_aging(env, Pd_k)
            b2 = bnd if done else bounds(env, cfg)
            agent.store(o, a_store, re, ra, o2, float(done), (bnd[0], bnd[1], b2[0], b2[1]) if cfg['proj_aware'] else None)
            agent.update()
        o = o2
    return env.summary()


def train(name, seed, episodes=55, dur=1200, log=None):
    cfg = METHODS[name]; rng = np.random.default_rng(1000 + seed)
    agent = Agent(HESSEnv.OBS_DIM, cfg, seed)
    ema = 1.0; hist = []
    # held-out validation routes (seeds disjoint from training and test) for checkpoint selection
    VAL = [(random_kind(np.random.default_rng(5000 + i)), 5000 + i, Ta) for i, Ta in enumerate((25.0, 40.0, 32.0))]
    best = (np.inf, None)
    t0 = time.time()
    for ep in range(episodes):
        r = make_route(random_kind(rng), int(rng.integers(1e9)), dur)
        env = HESSEnv(r, T_amb=rng.uniform(20, 42), soc0=rng.uniform(0.4, 0.9), sc0=rng.uniform(0.5, 0.95))
        noise = max(0.05, 0.3 * (1 - ep / (0.7 * episodes)))
        s = run_episode(env, agent, cfg, noise, train=True)
        ratio = LIFE_TARGET_KM / s['life_km']
        ema = 0.8 * ema + 0.2 * ratio
        if cfg['lagrangian'] and ep >= 0.3 * episodes:
            agent.lam = float(np.clip(agent.lam + 0.2 * (ema - 1.0), 0.0, 4.0))
        if ep >= 0.6 * episodes and (ep % 3 == 0 or ep == episodes - 1):
            vc = np.mean([run_episode(HESSEnv(make_route(k, sd, 1200), T_amb=Ta), agent, cfg)['cost_per_km'] for k, sd, Ta in VAL])
            if vc < best[0]:
                best = (vc, [p.copy() for p in agent.actor.params()])
        hist.append(dict(ep=ep, cost=s['cost_per_km'], life=s['life_km'], kwh=s['kWh_per_km'], lam=agent.lam))
        if log:
            print(name, seed, ep, round(s['cost_per_km'], 4), int(s['life_km']), round(agent.lam, 3), round(time.time() - t0), file=log, flush=True)
    if best[1] is not None:
        for p, q in zip(agent.actor.params(), best[1]):
            p[...] = q
    return agent, hist


TEST = [(k, 777 + i, Ta) for i, k in enumerate(['urban', 'suburban', 'hilly']) for Ta in (25.0, 40.0)]


def evaluate(agent, cfg):
    out = {}
    for kind, sd, Ta in TEST:
        env = HESSEnv(make_route(kind, sd, 3600), T_amb=Ta)
        t0 = time.time()
        s = run_episode(env, agent, cfg)
        s['ms_per_step'] = (time.time() - t0) / env.N * 1000
        out[f'{kind}-{int(Ta)}'] = s
    return out


if __name__ == '__main__':
    name, seed = sys.argv[1], int(sys.argv[2])
    with open(f'../logs/{name}_{seed}.log', 'w') as lg:
        agent, hist = train(name, seed, log=lg)
    res = evaluate(agent, METHODS[name])
    pickle.dump(dict(hist=hist, res=res, actor=agent.actor), open(f'../results/{name}_{seed}.pkl', 'wb'))
    print('done', name, seed)
