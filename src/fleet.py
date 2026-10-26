"""Fleet-level grid and sustainability figures (Table 6 of the paper)."""
import json

FLEET, KM_PER_YEAR, YEARS, CHARGER_EFF, PACK_KWH = 300, 70000, 12, 0.92, 115.2
d = json.load(open('../results/summary.json'))
print(f"{'Method':8s} {'grid GWh/yr':>12s} {'pack lives/bus':>15s} {'capacity worn MWh':>18s} {'12-yr cost MUSD':>16s}")
for n in ['BAT', 'RB-KE', 'DDPG', 'TD3', 'SCOUT', 'DP']:
    a = d[n]['avg']
    grid = a['kWh_per_km'] / CHARGER_EFF * KM_PER_YEAR * FLEET / 1e6
    packs = KM_PER_YEAR * YEARS / a['life_km']
    worn = packs * PACK_KWH * FLEET / 1e3
    cost = a['cost_per_km'] * KM_PER_YEAR * YEARS * FLEET / 1e6
    print(f"{n:8s} {grid:12.1f} {packs:15.2f} {worn:18.1f} {cost:16.1f}")
