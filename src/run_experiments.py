import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "src")
from traffic_sim import run_episode
import physics as ph

import os
rows = []
if os.path.exists("results/episodes.csv") and "--rerun" not in sys.argv:
    df = pd.read_csv("results/episodes.csv")
else:
  for density in [15, 20, 25, 30, 35, 40, 45]:
    for planner in ["naive", "predictive", "cooperative"]:
        rng = np.random.default_rng(1000 + density)
        for ep in range(300):
            r = run_episode(density, planner, rng)
            r.update(density=density, planner=planner, episode=ep)
            rows.append(r)
  df = pd.DataFrame(rows)
  df.to_csv("results/episodes.csv", index=False)
def summarise(d):
    h = d[d.outcome == "hopped"]
    return pd.Series({
        "attempts": len(d),
        "had_car_to_jump": (d.outcome != "no_gap").mean(),
        "hopped_share_of_valid": (d.outcome == "hopped").sum() / max((d.outcome != "no_gap").sum(), 1),
        "crash_per_1000_hops": 1000 * h.crash.astype(bool).sum() / max(len(h), 1),
        "landing_thw_median_s": h.min_thw.median(),
        "landing_thw_below_1s": (h.min_thw < 1.0).mean() if len(h) else np.nan,
        "landing_thw_below_0p6s": (h.min_thw < 0.6).mean() if len(h) else np.nan,
        "hard_brakers_per_hop": h.hard_brakers.mean(),
        "speed_dip_pct": (100 * (1 - h.min_speed / h.v_before)).median(),
        "road_covered_by_cars_pct": d.name[0] * 4.5 / 10,
    })
summ = df.groupby(["density", "planner"]).apply(summarise).reset_index()
summ.to_csv("results/summary.csv", index=False)
print(summ.round(3).to_string())

burn = ph.full_throttle_burn()
phys = {
    "ideal_gas_start_pressure_bar": ph.P0 / 1e5,
    "start_exhaust_velocity_m_s": ph.exhaust_velocity(*ph.tank_state(ph.M_AIR0)),
    "full_throttle_burn_s": float(burn[-1, 0]),
    "total_impulse_kNs": float(np.trapezoid(burn[:, 1], burn[:, 0]) / 1e3),
    "thrust_after_1s_kN": float(burn[np.searchsorted(burn[:, 0], 1.0), 1] / 1e3),
    "hover_s": {m: ph.max_hover_time(m)[0] for m in [1500, 1800, 2100]},
    "hop_2p5m_air_kg": float(ph.hop_air_cost(1800, 2.5, 5)[0]),
}
json.dump(phys, open("results/physics.json", "w"), indent=2, default=float)
print(phys)
