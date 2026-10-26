import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 8.5, 'savefig.dpi': 600})


def box(ax, x, y, w, h, text, fc='white', ls='-', lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec='black', lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontweight='bold' if bold else 'normal',
            linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, ls='-', lw=1.0, both=False, text=None, toff=(0, 0.12), fs=7.5):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='<|-|>' if both else '-|>', lw=lw, ls=ls,
                                                     color='black', shrinkA=0, shrinkB=0, mutation_scale=9))
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha='center', va='bottom',
                fontsize=fs, style='italic')


# ---------------- Fig. 1: HESS architecture ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.1))
ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis('off')
G1, G2 = '#d9d9d9', '#efefef'
box(ax, 0.2, 4.3, 2.5, 1.0, 'Depot charger\n(utility grid)', fc=G2)
box(ax, 3.4, 4.3, 2.5, 1.0, 'LFP battery pack\n180s2p, 115.2 kWh', fc=G1)
box(ax, 0.2, 1.3, 2.5, 1.0, 'Supercapacitor\nbank, 21 F,\n375 to 750 V', fc=G1)
box(ax, 3.4, 1.3, 2.5, 1.0, 'Bidirectional\nDC/DC converter', fc=G2)
# DC link
ax.plot([6.7, 6.7], [0.9, 5.8], color='black', lw=3.0)
ax.text(6.85, 0.75, 'DC link', ha='left', va='center', fontweight='bold')
box(ax, 7.6, 3.2, 2.2, 1.0, 'Traction\ninverter', fc=G2)
box(ax, 10.4, 3.2, 1.6, 1.0, 'Traction\nmotor', fc=G2)
box(ax, 12.5, 3.2, 1.3, 1.0, 'Wheels', fc=G2)
box(ax, 7.6, 1.3, 2.2, 1.0, 'Auxiliaries\n(HVAC, 8 kW)', fc=G2)
arrow(ax, (2.7, 4.8), (3.4, 4.8)); ax.text(3.05, 5.45, 'charging', ha='center', fontsize=7, style='italic')
arrow(ax, (5.9, 4.8), (6.7, 4.8), both=True); ax.text(6.0, 4.4, '$P_b$', fontsize=8.5)
arrow(ax, (2.7, 1.8), (3.4, 1.8), both=True)
arrow(ax, (5.9, 1.8), (6.7, 1.8), both=True); ax.text(6.0, 1.4, '$P_c$', fontsize=8.5)
arrow(ax, (6.7, 3.7), (7.6, 3.7), both=True, text='$P_{dem}$')
arrow(ax, (9.8, 3.7), (10.4, 3.7), both=True)
arrow(ax, (12.0, 3.7), (12.5, 3.7), both=True)
arrow(ax, (6.7, 1.8), (7.6, 1.8))
# controller
box(ax, 4.6, 6.75, 5.6, 1.15, 'SCOUT energy management controller\n(actor, base rule, feasibility projection)', fc='white', lw=1.4, bold=False)
box(ax, 10.6, 6.75, 3.2, 1.15, 'Route map and GPS\nstop distance, speed\nlimit, grade ahead', fc=G2)
box(ax, 0.2, 6.75, 3.6, 1.15, 'BMS and SC monitor\nSOC, cell temperature,\ncurrents, SC voltage', fc=G2)
arrow(ax, (10.6, 7.32), (10.2, 7.32), ls='--')
arrow(ax, (3.8, 7.32), (4.6, 7.32), ls='--')
ax.plot([6.3, 6.3, 4.65], [6.75, 2.85, 2.85], color='black', lw=1.0, ls='--')
arrow(ax, (4.65, 2.85), (4.65, 2.3), ls='--')
ax.text(5.2, 3.05, 'converter command', fontsize=7.5, style='italic', ha='center', va='bottom')
ax.annotate('', xy=(4.6, 7.0), xytext=(8.7, 4.2), arrowprops=dict(arrowstyle='-|>', ls='--', lw=1.0, color='black',
                                                                 mutation_scale=9, connectionstyle='arc3,rad=0.25'))
ax.text(8.95, 5.35, 'speed, demand', fontsize=7.5, style='italic')
ax.plot([0.25, 1.2], [0.35, 0.35], color='black', lw=1.0); ax.text(1.35, 0.35, 'power flow', va='center', fontsize=7.5)
ax.plot([3.2, 4.15], [0.35, 0.35], color='black', lw=1.0, ls='--'); ax.text(4.3, 0.35, 'signal', va='center', fontsize=7.5)
plt.tight_layout(); plt.savefig('../figures/fig_architecture.png'); plt.close()

# ---------------- Fig. 2: SCOUT framework ----------------
fig, ax = plt.subplots(figsize=(7.0, 4.9))
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis('off')
ax.text(0.2, 9.7, 'Online control loop (every 1 s)', fontweight='bold', va='top')
ax.add_patch(FancyBboxPatch((0.1, 5.35), 13.8, 4.55, boxstyle='round,pad=0.02', fc='none', ec='black', lw=0.8, ls=':'))
box(ax, 0.3, 7.4, 2.4, 1.3, 'State $s_k$\nvehicle, battery, SC\n+ route preview', fc=G2)
box(ax, 3.3, 7.4, 2.0, 1.3, 'Actor $\\mu_\\theta$\n12-64-64-1', fc=G1)
box(ax, 3.3, 5.6, 2.0, 1.2, 'Base rule $P^{rule}_c$\nfilter + kinetic\nenergy reference', fc=G2)
ax.add_patch(plt.Circle((6.15, 7.0), 0.28, fc='white', ec='black')); ax.text(6.15, 7.0, '+', ha='center', va='center', fontsize=11)
box(ax, 6.9, 6.35, 2.4, 1.3, 'Feasibility\nprojection\n$\\Pi_{[P_{lo},P_{hi}]}$', fc=G1)
box(ax, 9.9, 6.35, 1.9, 1.3, 'HESS plant\nbus, battery,\nSC', fc=G2)
box(ax, 12.1, 6.35, 1.7, 1.3, 'Costs\n$c^E_k$, $c^A_k$', fc=G2)
arrow(ax, (2.7, 8.05), (3.3, 8.05))
ax.annotate('', xy=(3.3, 6.2), xytext=(2.4, 7.4), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='black', mutation_scale=9))
arrow(ax, (5.3, 8.05), (6.0, 7.24), text='$a\\,\\Delta P_{max}$', toff=(0.35, 0.0))
arrow(ax, (5.3, 6.2), (6.0, 6.78))
arrow(ax, (6.43, 7.0), (6.9, 7.0))
arrow(ax, (9.3, 7.0), (9.9, 7.0), text='$P_c$')
arrow(ax, (11.8, 7.0), (12.1, 7.0))
ax.text(0.2, 4.95, 'Offline learning (training only)', fontweight='bold', va='top')
ax.add_patch(FancyBboxPatch((0.1, 0.1), 13.8, 4.95, boxstyle='round,pad=0.02', fc='none', ec='black', lw=0.8, ls=':'))
box(ax, 10.85, 2.75, 2.95, 1.8, 'Reward construction\nremove exogenous\ndemand and battery\nonly aging, credit\nSC energy $\\Phi$', fc=G2)
box(ax, 7.6, 2.9, 2.7, 1.5, 'Replay buffer\n$(s, a_{exec}, r^E, r^A, s\')$\n+ feasible bounds', fc=G2)
box(ax, 4.0, 3.35, 3.0, 1.05, 'Energy critics $Q^E_{1,2}$', fc=G1)
box(ax, 4.0, 2.1, 3.0, 1.05, 'Aging critics  $Q^{A}_{1,2}$', fc=G1)
box(ax, 0.3, 2.6, 3.1, 1.6, 'Actor update\nmax $Q^E + (1+\\lambda)Q^A$\nwith projected action', fc='white', lw=1.3)
box(ax, 0.3, 0.45, 3.1, 1.5, 'Lagrange multiplier $\\lambda$\ndual step on trip life\nversus target $L^*$', fc='white', lw=1.3)
box(ax, 4.0, 0.45, 3.0, 1.2, 'Trip fade rate\n$\\Sigma\\Delta Q / \\Sigma\\Delta x$', fc=G2)
ax.annotate('', xy=(12.35, 4.4), xytext=(12.95, 6.35), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='black', mutation_scale=9))
arrow(ax, (10.85, 3.65), (10.3, 3.65))
arrow(ax, (7.6, 3.85), (7.0, 3.85)); arrow(ax, (7.6, 2.65), (7.0, 2.65))
arrow(ax, (4.0, 3.85), (3.4, 3.6)); arrow(ax, (4.0, 2.65), (3.4, 3.0))
arrow(ax, (4.0, 1.05), (3.4, 1.05))
ax.annotate('', xy=(1.85, 2.6), xytext=(1.85, 1.95), arrowprops=dict(arrowstyle='-|>', lw=1.0, color='black', mutation_scale=9))
ax.text(1.95, 2.25, '$\\lambda$', fontsize=8)
ax.annotate('', xy=(4.3, 7.4), xytext=(1.2, 4.2), arrowprops=dict(arrowstyle='-|>', lw=1.0, ls='--', color='black',
                                                                 mutation_scale=9, connectionstyle='arc3,rad=-0.25'))
ax.text(1.5, 5.75, 'updated $\\theta$', fontsize=7.5, style='italic')
ax.plot([10.55, 10.55, 7.3], [6.35, 1.05, 1.05], color='black', lw=1.0, ls='--')
arrow(ax, (7.3, 1.05), (7.0, 1.05), ls='--')
ax.text(8.9, 1.15, 'per trip fade and distance', fontsize=7.5, style='italic', ha='center', va='bottom')
plt.tight_layout(); plt.savefig('../figures/fig_scout.png'); plt.close()
print('ok')
