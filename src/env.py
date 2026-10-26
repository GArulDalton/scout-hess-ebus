"""Battery supercapacitor electric bus simulator (1 Hz).
Physics: longitudinal bus dynamics, LFP pack (Rint + OCV + lumped thermal),
semi-empirical LFP cycle aging (Wang et al. 2011) turned into a severity factor,
supercapacitor bank (ideal C + ESR) behind a bidirectional DC/DC converter.
"""
import numpy as np

G = 9.81; RHO = 1.2
# ---------------- bus -----------------
BUS = dict(m0=13500.0, pax_m=70.0, Cd=0.70, A=8.0, Crr=0.008, rot=1.06,
           eta_dt=0.90, eta_rg=0.82, Pmot=250e3, Paux=8e3)
# ---------------- battery (LFP, 180s2p of 100 Ah cells) -----------------
NS, NP, QCELL = 180, 2, 100.0
R25 = 0.9e-3                       # cell ohmic resistance at 25 C
SOC_TAB = np.array([0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0])
OCV_TAB = np.array([2.90, 3.10, 3.20, 3.25, 3.27, 3.28, 3.29, 3.30, 3.31, 3.33, 3.35, 3.38, 3.45])
M_CELL, CP_CELL, HA_CELL = 2.0, 1100.0, 1.0   # kg, J/kgK, W/K
I_BAT_DIS, I_BAT_CHG = 2.5 * QCELL * NP, 1.0 * QCELL * NP  # pack current limits (A)
E_PACK_KWH = NS * NP * QCELL * 3.2 / 1000.0
# aging: control-oriented LFP cycle-life model (Suri and Onori, Energy 96 (2016) 644)
# Q_loss[%] = (alpha*SOC + beta) * exp((-Ea + eta*Ic)/(R*T)) * Ah^z, fitted on 2.3 Ah LFP cells
A_LO, B_LO, A_HI, B_HI = 2896.6, 7411.2, 2694.5, 6022.2
EA, ETA_I, Z, RGAS, Q_REF = 31500.0, 152.5, 0.57, 8.314, 2.3
# ---------------- supercapacitor (12 x 63 F / 125 V modules: 6s2p) --------
C_SC, R_SC, V_SC_MAX, V_SC_MIN = 21.0, 0.054, 750.0, 375.0
I_SC_MAX, ETA_C, P_CONV_MAX = 450.0, 0.95, 200e3
E_SC_MAX = 0.5 * C_SC * V_SC_MAX ** 2; E_SC_MIN = 0.5 * C_SC * V_SC_MIN ** 2
# ---------------- economics -----------------
PRICE_E = 0.12          # $/kWh grid electricity
CHG_EFF = 0.92          # charger efficiency
PACK_COST = 180.0 * E_PACK_KWH   # $ (pack replacement incl. labour)
EOL_LOSS = 20.0         # % capacity loss at end of life


def ocv(soc):
    return np.interp(soc, SOC_TAB, OCV_TAB) * NS


def r_pack(T):
    rc = R25 * np.exp(2500.0 * (1.0 / (T + 273.15) - 1.0 / 298.15))
    return rc * NS / NP


def gamma_ah(c, T, soc=0.5):
    """Cell Ah throughput (scaled to a QCELL cell) to EOL at C-rate c, temperature T (C), SOC."""
    pre = np.where(soc < 0.45, A_LO * soc + B_LO, A_HI * soc + B_HI)
    g = (EOL_LOSS / (pre * np.exp((-EA + ETA_I * c) / (RGAS * (T + 273.15))))) ** (1.0 / Z)
    return g * QCELL / Q_REF


GAMMA_SHAPE_NOM = gamma_ah(1.0, 25.0, 0.5)
EFC_NOM = 3500.0                        # equivalent full cycles to 80 % at 1C, 25 C (calibration)
GAMMA_NOM = 2.0 * EFC_NOM * QCELL       # cell Ah throughput to EOL at nominal stress


def severity(c, T, soc=0.5):
    return GAMMA_SHAPE_NOM / gamma_ah(c, T, soc)


# ---------------- route / driving-cycle generator -----------------
ROUTE_KINDS = {
    #            spacing m, vmax km/h range, signal prob, grade amp %, congestion prob, dwell s
    'urban':    dict(sp=330, vmax=(28, 42), sig=0.45, gr=0.8, cong=0.35, dw=(15, 40)),
    'suburban': dict(sp=720, vmax=(45, 62), sig=0.20, gr=1.2, cong=0.15, dw=(10, 30)),
    'hilly':    dict(sp=480, vmax=(32, 50), sig=0.25, gr=5.0, cong=0.20, dw=(12, 35)),
}


def random_kind(rng):
    """Training routes: parameters drawn from the union of the families."""
    return dict(sp=rng.uniform(280, 800), vmax=(rng.uniform(25, 40), rng.uniform(42, 62)),
                sig=rng.uniform(0.1, 0.5), gr=rng.uniform(0.3, 5.5), cong=rng.uniform(0.1, 0.4),
                dw=(rng.uniform(8, 15), rng.uniform(25, 40)))


def make_route(kind, seed, duration=1800):
    rng = np.random.default_rng(seed)
    p = ROUTE_KINDS[kind] if isinstance(kind, str) else kind
    # grade as a function of distance (smooth random hills)
    L = 60000.0
    xs = np.arange(0, L, 10.0)
    gr = np.zeros_like(xs)
    for k in range(4):
        lam = rng.uniform(300, 2500); ph = rng.uniform(0, 2 * np.pi)
        gr += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * xs / lam + ph)
    gr = gr / (np.abs(gr).max() + 1e-9) * p['gr'] / 100.0
    v, x, t = 0.0, 0.0, 0
    V, X, DSTOP, VLIM, MASS, ISSTOP = [], [], [], [], [], []
    pax = rng.integers(10, 50)
    while t < duration:
        seg = max(120.0, rng.lognormal(np.log(p['sp']), 0.35))
        stop_is_bus = True
        # optional intermediate signal stop
        subs = [seg]
        if rng.random() < p['sig']:
            f = rng.uniform(0.3, 0.7); subs = [seg * f, seg * (1 - f)]
        for si, s in enumerate(subs):
            target = rng.uniform(*p['vmax']) / 3.6
            a_acc = rng.uniform(0.7, 1.15); a_dec = rng.uniform(0.9, 1.35)
            cong = rng.random() < p['cong']
            rem = s
            while rem > 0.3 and t < duration:
                vt = target * (0.55 + 0.25 * np.sin(t / 9.0)) if cong else target * (1 + 0.04 * np.sin(t / 7.0))
                vbrake = np.sqrt(max(0.0, 2 * a_dec * max(rem - 1.0, 0.0)))
                vn = min(v + a_acc, vt, vbrake)
                vn = max(vn, max(v - 1.6, 0.0))
                if vn < 0.3 and rem > 2.0:
                    vn = min(1.5, rem)
                dx = 0.5 * (v + vn)
                rem -= dx; x += dx; v = vn
                V.append(v); X.append(x); DSTOP.append(max(rem, 0.0)); VLIM.append(target)
                MASS.append(BUS['m0'] + pax * BUS['pax_m']); ISSTOP.append(0); t += 1
            v = 0.0
            dwell = int(rng.uniform(*p['dw'])) if si == len(subs) - 1 else int(rng.uniform(10, 45))
            if si == len(subs) - 1:
                pax = int(np.clip(pax + rng.integers(-8, 9), 0, 80))
            for _ in range(dwell):
                if t >= duration:
                    break
                V.append(0.0); X.append(x); DSTOP.append(0.0); VLIM.append(target)
                MASS.append(BUS['m0'] + pax * BUS['pax_m']); ISSTOP.append(1); t += 1
    V = np.array(V[:duration]); X = np.array(X[:duration])
    acc = np.diff(np.concatenate([[0.0], V]))
    grade = np.interp(X, xs, gr)
    grade_ahead = np.interp(X + 200.0, xs, gr)
    return dict(v=V, a=acc, x=X, grade=grade, grade_ahead=grade_ahead, dstop=np.array(DSTOP[:duration]),
                vlim=np.array(VLIM[:duration]), mass=np.array(MASS[:duration]), stop=np.array(ISSTOP[:duration]))


def demand_power(r):
    m = r['mass']; v = r['v']; a = r['a']; th = np.arctan(r['grade'])
    F = BUS['rot'] * m * a + m * G * BUS['Crr'] * np.cos(th) * (v > 0.05) + 0.5 * RHO * BUS['Cd'] * BUS['A'] * v ** 2 \
        + m * G * np.sin(th) * (v > 0.05)
    Pw = np.clip(F * v, -BUS['Pmot'], BUS['Pmot'])
    Pe = np.where(Pw > 0, Pw / BUS['eta_dt'], Pw * BUS['eta_rg']) + BUS['Paux']
    return Pe


class HESSEnv:
    """One episode = one route. Action = converter power command (W, + = SC discharging to bus)."""
    OBS_DIM = 12

    def __init__(self, route, T_amb=30.0, soc0=0.8, sc0=0.75):
        self.r = route; self.P = demand_power(route); self.N = len(self.P)
        self.Tamb = T_amb; self.soc0 = soc0; self.sc0 = sc0
        self.reset()

    def reset(self):
        self.k = 0; self.soc = self.soc0; self.T = self.Tamb
        self.Esc = E_SC_MIN + self.sc0 ** 2 * 0  # placeholder replaced below
        self.Esc = 0.5 * C_SC * (V_SC_MIN + self.sc0 * (V_SC_MAX - V_SC_MIN)) ** 2
        self.Esc0 = self.Esc
        self.Pf = self.P[0]            # low pass filter state
        self.Pbat_prev = 0.0
        self.log = {k: [] for k in ['Pdem', 'Pbat', 'Pconv', 'Ibat', 'Vsc', 'T', 'soc', 'loss', 'Ebat', 'brake', 'shield']}
        self.cum = dict(Ebat_kWh=0.0, loss_pct=0.0, brake_kWh=0.0, viol=0)
        return self.obs()

    # ---- helpers ----
    def vsc(self, E=None):
        return np.sqrt(2 * (self.Esc if E is None else E) / C_SC)

    def sc_soc(self):
        return (self.vsc() - V_SC_MIN) / (V_SC_MAX - V_SC_MIN)

    def sc_ref_energy(self, v, m):
        """Kinetic-energy based SC target: keep room for the regen of the current speed."""
        Ekin = 0.5 * m * v ** 2 * BUS['eta_rg'] * ETA_C * 0.85
        return np.clip(E_SC_MAX - Ekin - 0.10 * (E_SC_MAX - E_SC_MIN), E_SC_MIN + 0.25 * (E_SC_MAX - E_SC_MIN), E_SC_MAX)

    def feasible_conv(self, Pd):
        """Converter power interval that respects SC energy/current and battery current limits."""
        V = self.vsc(); dt = 1.0
        # SC terminal power bounds from energy window and current
        Pint_dis = min((self.Esc - E_SC_MIN) / dt, V * I_SC_MAX)          # internal power available
        Pint_chg = min((E_SC_MAX - self.Esc) / dt, V * I_SC_MAX)
        Pc_hi = min(P_CONV_MAX, ETA_C * (Pint_dis - (Pint_dis / max(V, 1)) ** 2 * R_SC))
        Pc_lo = max(-P_CONV_MAX, -(Pint_chg / ETA_C))
        # battery limits: Pbat = Pd - Pconv
        Voc = ocv(self.soc); R = r_pack(self.T)
        Pb_max = Voc * I_BAT_DIS - I_BAT_DIS ** 2 * R
        Pb_min = -(Voc * I_BAT_CHG + I_BAT_CHG ** 2 * R)
        lo = max(Pc_lo, Pd - Pb_max); hi = min(Pc_hi, Pd - Pb_min)
        if lo > hi:  # infeasible: prefer SC limits (regen surplus goes to friction brakes)
            lo, hi = Pc_lo, Pc_hi
        return lo, hi

    def obs(self):
        k = min(self.k, self.N - 1); r = self.r
        v = r['v'][k]; m = r['mass'][k]
        Ekin = 0.5 * m * v ** 2 / 3.6e6
        return np.array([
            self.P[k] / 200e3, self.Pf / 200e3, self.Pbat_prev / 200e3,
            self.sc_soc() * 2 - 1, (self.soc - 0.5) * 2, (self.T - 35.0) / 10.0,
            v / 15.0, min(r['dstop'][k], 800.0) / 400.0 - 1, r['vlim'][k] / 15.0,
            r['grade_ahead'][k] * 20.0, Ekin / 0.6, (self.Tamb - 30.0) / 10.0], dtype=np.float64)

    def rule_power(self, tau=25.0, kp=0.004):
        """Baseline rule: low-pass frequency split + kinetic-energy SC reference tracking."""
        k = self.k; Pd = self.P[k]
        Phf = Pd - self.Pf
        Eref = self.sc_ref_energy(self.r['v'][k], self.r['mass'][k])
        return Phf + kp * (self.Esc - Eref)

    # ---- dynamics ----
    def step(self, Pconv_cmd):
        k = self.k; Pd = self.P[k]
        lo, hi = self.feasible_conv(Pd)
        Pc = float(np.clip(Pconv_cmd, lo, hi))
        shield = abs(Pc - Pconv_cmd)
        # SC side
        V = self.vsc()
        Pt = Pc / ETA_C if Pc > 0 else Pc * ETA_C      # SC terminal power
        disc = V ** 2 - 4 * R_SC * Pt
        Isc = (V - np.sqrt(max(disc, 0.0))) / (2 * R_SC)
        self.Esc = float(np.clip(self.Esc - V * Isc, E_SC_MIN, E_SC_MAX))
        # battery side
        Pb = Pd - Pc
        Voc = ocv(self.soc); R = r_pack(self.T)
        brake = 0.0
        Pb_min = -(Voc * I_BAT_CHG + I_BAT_CHG ** 2 * R)
        if Pb < Pb_min:                         # regen surplus -> friction brakes
            brake = Pb_min - Pb; Pb = Pb_min
        disc = Voc ** 2 - 4 * R * Pb
        Ib = (Voc - np.sqrt(max(disc, 0.0))) / (2 * R)
        Ib = float(np.clip(Ib, -I_BAT_CHG, I_BAT_DIS))
        self.soc -= Ib / (NP * QCELL * 3600.0)
        # thermal (cell level)
        Ic = Ib / NP; Rc = R * NP / NS
        q = Ic ** 2 * Rc
        self.T += (q - HA_CELL * (self.T - self.Tamb)) / (M_CELL * CP_CELL)
        # aging: severity-weighted Ah throughput
        c = abs(Ic) / QCELL
        dAh = abs(Ic) / 3600.0
        loss = EOL_LOSS * severity(c, self.T, self.soc) * dAh / GAMMA_NOM if dAh > 0 else 0.0
        Ebat = Voc * Ib / 3.6e6                 # chemical energy drawn (kWh)
        # bookkeeping
        self.cum['Ebat_kWh'] += Ebat; self.cum['loss_pct'] += loss; self.cum['brake_kWh'] += brake / 3.6e6
        self.cum['viol'] += int(shield > 1.0)
        for kk, vv in zip(['Pdem', 'Pbat', 'Pconv', 'Ibat', 'Vsc', 'T', 'soc', 'loss', 'Ebat', 'brake', 'shield'],
                          [Pd, Pb, Pc, Ib, self.vsc(), self.T, self.soc, loss, Ebat, brake, shield]):
            self.log[kk].append(vv)
        self.Pf += (Pd - self.Pf) / 25.0
        self.Pbat_prev = Pb
        self.k += 1
        done = self.k >= self.N
        e_cost = Ebat * PRICE_E / CHG_EFF
        a_cost = loss / EOL_LOSS * PACK_COST
        if done:   # terminal SC energy correction valued at grid price
            e_cost += (self.Esc0 - self.Esc) / 3.6e6 * PRICE_E / CHG_EFF / 0.9
        return self.obs(), e_cost, a_cost, shield, done

    def summary(self):
        km = self.r['x'][-1] / 1000.0
        dE_sc = (self.Esc0 - self.Esc) / 3.6e6
        Eeq = self.cum['Ebat_kWh'] + dE_sc / 0.9
        I = np.array(self.log['Ibat'])
        loss = self.cum['loss_pct']
        ecost = Eeq * PRICE_E / CHG_EFF; acost = loss / EOL_LOSS * PACK_COST
        return dict(km=km, kWh_per_km=Eeq / km, loss_pct_per_1000km=loss / km * 1000,
                    life_km=EOL_LOSS / (loss / km) if loss > 0 else np.inf,
                    Irms=np.sqrt(np.mean(I ** 2)), Ipk=np.abs(I).max(), Tmax=max(self.log['T']),
                    cost_per_km=(ecost + acost) / km, ecost_per_km=ecost / km, acost_per_km=acost / km,
                    brake_kWh=self.cum['brake_kWh'], viol=self.cum['viol'], sc_end=self.sc_soc())
