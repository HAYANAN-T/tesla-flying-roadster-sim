"""Draw every simulation figure used in the article from results/ and fresh runs."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import physics as ph
from traffic_sim import run_episode, DT

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
C = {"naive": "#DC2626", "predictive": "#2563EB", "cooperative": "#16A34A"}
NAMES = {"naive": "No AI (hop into biggest gap now)", "predictive": "AI prediction",
         "cooperative": "AI prediction + car-to-car talk (V2V)"}

def save(fig, name, title, sub=None, src="Simulation by the author; code in the article's GitHub repository.", right=0.97):
    fig.text(0.03, 0.96, title, fontsize=20, fontweight="bold", va="top")
    if sub: fig.text(0.03, 0.905, sub, fontsize=13, color="#4B5563", va="top")
    fig.text(0.03, 0.015, src, fontsize=10.5, color="#6B7280")
    fig.subplots_adjust(top=0.80 if sub else 0.85, bottom=0.13, left=0.09, right=right)
    fig.savefig(f"images/{name}.png", dpi=200); plt.close(fig)

# 1. time-space diagram of one hop in moderate traffic (naive planner)
for seed in range(40):
    log = []
    r = run_episode(30, "naive", np.random.default_rng(seed), log=log, after=250)
    if r["outcome"] == "hopped":
        break
X = np.array([l[0] for l in log]); A = np.array([l[2] for l in log]); V = np.array([l[1] for l in log])
t = np.arange(len(X)) * DT - 5.0
air = np.where(~A[:, 0])[0]
x0, x1 = X[air[0] - 1, 0], X[air[-1] + 1, 0]
shift = 150 - x0                        # put take-off at 150 m on the plot
Xs = (X + shift) % 1000
fig, ax = plt.subplots(figsize=(12, 6.6))
for i in range(1, X.shape[1]):
    x = Xs[:, i].copy(); x[np.r_[False, np.abs(np.diff(x)) > 500]] = np.nan
    ax.plot(t, x, color="#9CA3AF", lw=1.2, zorder=1)
    braking = np.r_[np.diff(V[:, i]) / DT < -2.5, False]
    ax.scatter(t[braking], x[braking], color="#DC2626", s=10, zorder=2)
x = np.where(A[:, 0], Xs[:, 0], np.nan)
ax.plot(t, x, color="#F97316", lw=3.5, zorder=3, label="Hop-capable car")
ti, tf = t[air[0] - 1], t[air[-1] + 1]
ax.plot([ti, tf], [Xs[air[0] - 1, 0], Xs[air[-1] + 1, 0]], color="#F97316", lw=3.5, ls="--", zorder=3)
ax.annotate("Hop: 1.5 s in the air,\njumps over the cars ahead", (ti + 0.8, Xs[air[0] - 1, 0] + 25), xytext=(ti - 4.5, 520),
            arrowprops=dict(arrowstyle="->"), fontsize=12)
ax.scatter([], [], color="#DC2626", s=25, label="Other car braking hard (> 2.5 m/s²)")
ax.plot([], [], color="#9CA3AF", label="Other cars")
ax.set_xlim(-5, 25); ax.set_ylim(0, 750)
ax.set_xlabel("Time (seconds, 0 = take-off)"); ax.set_ylabel("Position along the road (m)")
ax.legend(loc="lower right", frameon=True)
save(fig, "sim1_time_space_hop", "Inside the simulation: one hop at 30 cars per km",
     "Each line is one car moving along the road over time. Steeper line = faster car.")

# 2. how often a safe hop is allowed
s = pd.read_csv("results/summary.csv")
fig, ax = plt.subplots(figsize=(12, 6.6))
for p in ["naive", "predictive", "cooperative"]:
    d = s[s.planner == p]
    ax.plot(d.density, 100 * d.hopped_share_of_valid, "-o", color=C[p], lw=3, ms=8, label=NAMES[p])
ax.set_xlabel("Traffic density (cars per km in the lane)"); ax.set_ylabel("Hops carried out (%)")
ax.set_ylim(-3, 105); ax.legend(frameon=False, loc="center right")
ax.axvspan(25, 45, color="#F3F4F6", zorder=0); ax.text(26, 60, "Busy road:\nAI finds almost\nno safe landing spot", fontsize=12, color="#374151")
save(fig, "sim2_safe_hops_vs_traffic", "When does the AI allow a hop?",
     "Share of hop requests carried out, out of 300 simulated attempts per point.")

# 3. landing headway and braking for the no-AI hop
d = s[s.planner == "naive"]
fig, ax1 = plt.subplots(figsize=(12, 6.6))
ax1.bar(d.density - 1.2, d.landing_thw_median_s, width=2.2, color="#F97316", label="Median gap after landing (seconds)")
ax1.axhline(1.0, color="#6B7280", ls="--"); ax1.text(14, 1.03, "1-second gap", color="#6B7280", fontsize=12)
ax1.set_ylabel("Seconds of gap after landing"); ax1.set_xlabel("Traffic density (cars per km in the lane)")
ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.bar(d.density + 1.2, d.hard_brakers_per_hop, width=2.2, color="#DC2626", label="Cars braking hard per hop")
ax2.set_ylabel("Cars braking harder than 4 m/s² per hop")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=True, loc="upper left", bbox_to_anchor=(0.0, 1.12), ncol=2)
ax1.set_ylim(0, 1.25); ax2.set_ylim(0, 2.5)
save(fig, "sim3_landing_gap_and_braking", "Hopping without AI: what it does to the cars around you",
     "As traffic gets denser, the landing gap shrinks and more drivers slam the brakes.", right=0.90)

# 4. physics: thrust decay and hover time
b = ph.full_throttle_burn()
fig, ax = plt.subplots(figsize=(12, 6.6))
ax.plot(b[:, 0], b[:, 1] / 1e3, color="#2563EB", lw=3, label="Thrust with all jets open (kN)")
for m, ls in [(1500, ":"), (1800, "--"), (2100, "-.")]:
    ax.axhline((m + 250) * 9.81 / 1e3, color="#6B7280", ls=ls, lw=1.5)
    ax.text(7.6, (m + 250) * 9.81 / 1e3 + 0.45, f"Weight of a {m:,} kg car + air", fontsize=11.5, color="#374151")
ax.set_xlim(0, 13); ax.set_xlabel("Seconds after the jets open"); ax.set_ylabel("Thrust (kilonewtons)")
ax.legend(frameon=False)
save(fig, "sim4_cold_gas_thrust", "Why it can only hop: the air tank empties in seconds",
     "Model of a 600 L tank with 250 kg of air (Tesla's reported figures), sized for 80 kN of starting thrust.",
     "Physics model by the author; code in the article's GitHub repository. Tank and thrust figures as reported from the Oct 15, 2026 reveal.")
print("ok", seed, r)
