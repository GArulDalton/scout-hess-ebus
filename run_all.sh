#!/bin/bash
# Reproduce every result in the paper on one CPU core (about 1.5 h in total).
set -e
cd "$(dirname "$0")"
mkdir -p logs results figures
cd src
python dp.py                                   # rule baselines + offline DP  -> results/baselines.pkl
for s in 0 1 2; do for m in SCOUT TD3 TD3-E DDPG; do python train.py $m $s; done; done
for s in 0 1; do for m in SCOUT-noPrev SCOUT-noRes SCOUT-noLag SCOUT-noProj SCOUT-noShape; do python train.py $m $s; done; done
python analysis.py                             # Tables 3 to 5 (results/summary.json) and Figs. 3 to 6
python fleet.py                                # Table 6
python diagrams.py                             # Figs. 1 and 2
