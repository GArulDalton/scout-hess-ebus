import pickle, glob, numpy as np, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import *
from dp import solve, simulate, run_rule

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Liberation Sans','DejaVu Sans'], 'figure.dpi': 600, 'savefig.dpi': 600})
B = pickle.load(open('../results/baselines.pkl', 'rb'))
CASES = list(B['DP'].keys())
runs = {}
for f in sorted(glob.glob('../results/*_*.pkl')):
    n, sd = f.split('/')[-1][:-4].rsplit('_', 1)
    runs.setdefault(n, []).append(pickle.load(open(f, 'rb')))
METS = ['cost_per_km', 'kWh_per_km', 'life_km', 'Irms', 'Tmax', 'ecost_per_km', 'acost_per_km', 'ms_per_step']


def agg(name):
    if name in B:
        return {c: {m: (B[name][c].get(m, np.nan), 0.0) for m in METS} for c in CASES}
    R = runs[name]
    return {c: {m: (np.mean([r['res'][c][m] for r in R]), np.std([r['res'][c][m] for r in R])) for m in METS} for c in CASES}


ORDER = ['BAT', 'LPF', 'RB-KE', 'DDPG', 'TD3-E', 'TD3', 'SCOUT', 'DP']
ABL = ['SCOUT', 'SCOUT-noPrev', 'SCOUT-noRes', 'SCOUT-noLag', 'SCOUT-noProj', 'SCOUT-noShape']
A = {n: agg(n) for n in ORDER + ABL if n in B or n in runs}
out = {'cases': CASES, 'nseeds': {n: len(runs.get(n, [])) for n in A}}
for n in A:
    out[n] = {c: {m: [float(A[n][c][m][0]), float(A[n][c][m][1])] for m in METS} for c in CASES}
# averages over cases
for n in A:
    out[n]['avg'] = {m: float(np.mean([A[n][c][m][0] for c in CASES])) for m in METS}
    if n in runs:
        per_seed = [np.mean([r['res'][c]['cost_per_km'] for c in CASES]) for r in runs[n]]
        out[n]['avg']['cost_std_seeds'] = float(np.std(per_seed))
json.dump(out, open('../results/summary.json', 'w'), indent=1)

# ---------- Fig: cost per case ----------
fig, ax = plt.subplots(figsize=(7, 2.8))
names = [n for n in ORDER if n in A]
w = 0.8 / len(names); x = np.arange(len(CASES))
for i, n in enumerate(names):
    m = [A[n][c]['cost_per_km'][0] for c in CASES]; s = [A[n][c]['cost_per_km'][1] for c in CASES]
    ax.bar(x + (i - len(names) / 2 + 0.5) * w, m, w, yerr=s, label=n, capsize=1.5, linewidth=0)
ax.set_xticks(x); ax.set_xticklabels([c.replace('-', chr(10)) + ' °C' for c in CASES])
ax.set_ylabel('Total cost (USD/km)'); ax.set_ylim(0.15, None); ax.legend(ncol=8, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)
plt.tight_layout(); plt.savefig('../figures/fig_cost.png'); plt.close()

# ---------- Fig: trade-off energy vs life ----------
fig, ax = plt.subplots(figsize=(3.4, 2.8))
for n in names:
    e = np.mean([A[n][c]['kWh_per_km'][0] for c in CASES]); l = np.mean([A[n][c]['life_km'][0] for c in CASES]) / 1e3
    ax.scatter(e, l, s=25); ax.annotate(n, (e, l), fontsize=6.5, xytext=(3, 2), textcoords='offset points')
ax.set_xlabel('Energy (kWh/km)'); ax.set_ylabel('Battery life (10³ km)')
plt.tight_layout(); plt.savefig('../figures/fig_tradeoff.png'); plt.close()

# ---------- Fig: learning curves ----------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for n in ['DDPG', 'TD3', 'TD3-E', 'SCOUT']:
    if n not in runs: continue
    H = np.array([[h['cost'] for h in r['hist']] for r in runs[n]])
    k = 5; Hs = np.array([np.convolve(h, np.ones(k) / k, mode='valid') for h in H])
    mu = Hs.mean(0); sd = Hs.std(0); ep = np.arange(len(mu)) + k
    ax.plot(ep, mu, label=n, lw=1); ax.fill_between(ep, mu - sd, mu + sd, alpha=0.2)
ax.set_xlabel('Training episode'); ax.set_ylabel('Episode cost (USD/km)'); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig('../figures/fig_learning.png'); plt.close()

# ---------- Fig: time series, urban 40 C ----------
r = make_route('urban', 777, 3600); Ta = 40.0
cfg = METHODS['SCOUT']; act = runs['SCOUT'][0]['actor']
env = HESSEnv(r, Ta); o = obs_f(env.reset(), cfg); done = False
while not done:
    a = float(np.clip(act.forward(o[None])[0, 0], -1, 1))
    o, _, _, _, done = env.step(command(env, a, cfg)); o = obs_f(o, cfg)
_, er = run_rule(r, Ta, 'RB-KE')
_, ed = simulate(r, Ta, solve(r, Ta))
t0, t1 = 600, 900; tt = np.arange(t0, t1)
fig, axs = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True)
axs[0].plot(tt, np.array(env.log['Pdem'][t0:t1]) / 1e3, 'k', lw=0.8); axs[0].set_ylabel('P_dem (kW)')
for e, lab in [(er, 'RB-KE'), (env, 'SCOUT'), (ed, 'DP')]:
    axs[1].plot(tt, e.log['Ibat'][t0:t1], lw=0.8, label=lab)
    axs[2].plot(tt, e.log['Vsc'][t0:t1], lw=0.8, label=lab)
axs[1].set_ylabel('Battery current (A)'); axs[2].set_ylabel('SC voltage (V)'); axs[2].set_xlabel('Time (s)')
axs[1].legend(ncol=3, fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig('../figures/fig_timeseries.png'); plt.close()
print(json.dumps({n: out[n]['avg'] for n in A}, indent=0))
