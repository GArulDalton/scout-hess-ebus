"""Offline DP benchmark (full future knowledge) over (SC energy, cell temperature)."""
import numpy as np, pickle, time
from env import *

NE, NT, NA = 61, 13, 41


def stage(Esc, T, Pc_cmd, Pd, soc, Tamb):
    """Vectorised version of HESSEnv.step physics. Returns Esc', T', cost."""
    V = np.sqrt(2 * Esc / C_SC)
    Pint_dis = np.minimum(Esc - E_SC_MIN, V * I_SC_MAX)
    Pint_chg = np.minimum(E_SC_MAX - Esc, V * I_SC_MAX)
    hi = np.minimum(P_CONV_MAX, ETA_C * (Pint_dis - (Pint_dis / np.maximum(V, 1)) ** 2 * R_SC))
    lo = np.maximum(-P_CONV_MAX, -Pint_chg / ETA_C)
    Voc = ocv(soc); R = r_pack(T)
    Pb_max = Voc * I_BAT_DIS - I_BAT_DIS ** 2 * R
    Pb_min = -(Voc * I_BAT_CHG + I_BAT_CHG ** 2 * R)
    lo2 = np.maximum(lo, Pd - Pb_max); hi2 = np.minimum(hi, Pd - Pb_min)
    bad = lo2 > hi2
    lo2 = np.where(bad, lo, lo2); hi2 = np.where(bad, hi, hi2)
    Pc = np.clip(Pc_cmd, lo2, hi2)
    Pt = np.where(Pc > 0, Pc / ETA_C, Pc * ETA_C)
    Isc = (V - np.sqrt(np.maximum(V ** 2 - 4 * R_SC * Pt, 0))) / (2 * R_SC)
    E2 = np.clip(Esc - V * Isc, E_SC_MIN, E_SC_MAX)
    Pb = np.maximum(Pd - Pc, Pb_min)
    Ib = np.clip((Voc - np.sqrt(np.maximum(Voc ** 2 - 4 * R * Pb, 0))) / (2 * R), -I_BAT_CHG, I_BAT_DIS)
    Ic = Ib / NP; Rc = R * NP / NS
    T2 = T + (Ic ** 2 * Rc - HA_CELL * (T - Tamb)) / (M_CELL * CP_CELL)
    c = np.abs(Ic) / QCELL
    loss = EOL_LOSS * severity(c, T2, soc) * (np.abs(Ic) / 3600) / GAMMA_NOM
    cost = Voc * Ib / 3.6e6 * PRICE_E / CHG_EFF + loss / EOL_LOSS * PACK_COST
    return E2, T2, cost


def solve(route, Tamb, sc0=0.75, w_age=1.0):
    P = demand_power(route); N = len(P)
    # SOC trajectory estimate from a battery-only pass (SOC barely changes Voc for LFP)
    e = HESSEnv(route, Tamb); soc_tr = []
    for k in range(N):
        soc_tr.append(e.soc); e.step(e.rule_power())
    Eg = np.linspace(E_SC_MIN, E_SC_MAX, NE)
    Tg = np.linspace(Tamb, Tamb + 12, NT)
    Ag = np.linspace(-P_CONV_MAX, P_CONV_MAX, NA)
    EE, TT, AA = np.meshgrid(Eg, Tg, Ag, indexing='ij')
    E0 = 0.5 * C_SC * (V_SC_MIN + sc0 * (V_SC_MAX - V_SC_MIN)) ** 2
    Vt = (E0 - Eg) / 3.6e6 * PRICE_E / CHG_EFF / 0.9          # terminal correction
    V = np.repeat(Vt[:, None], NT, 1)
    pol = np.zeros((N, NE, NT), dtype=np.int8)
    from scipy.interpolate import RegularGridInterpolator
    for k in reversed(range(N)):
        E2, T2, c = stage(EE, TT, AA, P[k], soc_tr[k], Tamb)
        f = RegularGridInterpolator((Eg, Tg), V, bounds_error=False, fill_value=None)
        pts = np.stack([E2.ravel(), np.clip(T2, Tg[0], Tg[-1]).ravel()], 1)
        Q = c + f(pts).reshape(E2.shape)
        pol[k] = Q.argmin(2); V = Q.min(2)
    return Eg, Tg, Ag, pol


def simulate(route, Tamb, sol, sc0=0.75):
    Eg, Tg, Ag, pol = sol
    env = HESSEnv(route, Tamb, sc0=sc0); env.reset(); done = False
    while not done:
        i = int(np.clip(np.round((env.Esc - Eg[0]) / (Eg[1] - Eg[0])), 0, len(Eg) - 1))
        j = int(np.clip(np.round((env.T - Tg[0]) / (Tg[1] - Tg[0])), 0, len(Tg) - 1))
        _, _, _, _, done = env.step(Ag[pol[env.k, i, j]])
    return env.summary(), env


def run_rule(route, Tamb, kind):
    env = HESSEnv(route, Tamb); done = False
    while not done:
        if kind == 'BAT':
            cmd = 0.0
        elif kind == 'LPF':
            k = env.k; cmd = env.P[k] - env.Pf + 0.002 * (env.Esc - env.Esc0)
        else:  # RB-KE: LPF + kinetic-energy SC reference
            cmd = env.rule_power()
        _, _, _, _, done = env.step(cmd)
    return env.summary(), env


if __name__ == '__main__':
    from train import TEST
    out = {}
    for kind, sd, Ta in TEST:
        r = make_route(kind, sd, 3600)
        key = f'{kind}-{int(Ta)}'
        for rk in ['BAT', 'LPF', 'RB-KE']:
            out.setdefault(rk, {})[key] = run_rule(r, Ta, rk)[0]
        t0 = time.time(); sol = solve(r, Ta)
        out.setdefault('DP', {})[key] = simulate(r, Ta, sol)[0]
        print(key, 'DP time', round(time.time() - t0), out['DP'][key]['cost_per_km'], out['RB-KE'][key]['cost_per_km'], flush=True)
    pickle.dump(out, open('../results/baselines.pkl', 'wb'))
