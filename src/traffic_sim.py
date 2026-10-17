"""Single-lane ring-road traffic simulation with one hop-capable car.

Cars follow the Intelligent Driver Model (IDM, Treiber et al. 2000).
At a random moment the hop-capable car jumps over the traffic ahead and
must land in a gap. Three ways of planning the hop are compared:

  naive        - picks the biggest gap it sees *now*, lands with 1.0 m error,
                 surrounding human drivers may startle and brake.
  predictive   - predicts where every car will be at touchdown, picks the
                 safest predicted gap, aborts if no gap is safe enough,
                 lands with 0.3 m error. Human drivers may still startle.
  cooperative  - predictive + V2V: the car that will be behind the landing
                 spot is asked to open the gap in advance, and nearby cars
                 are warned, so nobody startles.
"""
import numpy as np

DT, L_CAR = 0.1, 4.5
IDM = dict(a=1.2, b=2.0, s0=2.0, T=1.3, delta=4)


def idm_acc(v, gap, dv, v0):
    s_star = IDM["s0"] + np.maximum(0, v * IDM["T"] + v * dv / (2 * np.sqrt(IDM["a"] * IDM["b"])))
    return IDM["a"] * (1 - (v / v0) ** IDM["delta"] - (s_star / np.maximum(gap, 0.1)) ** 2)


class Ring:
    def __init__(self, density_per_km, length=1000.0, rng=None):
        self.rng = rng or np.random.default_rng()
        self.Lr = length
        n = max(3, int(density_per_km * length / 1000))
        self.x = np.sort(self.rng.uniform(0, length, n))
        self.x = np.linspace(0, length, n, endpoint=False) + self.rng.normal(0, 1.0, n)
        self.v = np.full(n, 10.0)
        self.v0 = self.rng.normal(30.0, 2.5, n).clip(22, 38)   # desired speeds (m/s)
        self.extra_brake = np.zeros(n)   # startle braking (m/s^2)
        self.brake_timer = np.zeros(n)
        self.active = np.ones(n, bool)   # False while the hop car is in the air
        self.log = None                  # set to a list to record (x, v, active) each step

    def leaders(self):
        idx = np.where(self.active)[0]
        order = idx[np.argsort(self.x[idx] % self.Lr)]
        lead = np.empty(len(self.x), int); lead[:] = -1
        lead[order] = np.roll(order, -1)
        return lead

    def gaps(self, lead):
        g = (self.x[lead] - self.x) % self.Lr - L_CAR
        return g

    def step(self):
        lead = self.leaders()
        act = self.active
        g = self.gaps(lead)
        a = idm_acc(self.v, g, self.v - self.v[lead], self.v0)
        a = a - self.extra_brake
        self.brake_timer -= DT
        self.extra_brake[self.brake_timer <= 0] = 0
        a = np.maximum(a, -9.0)
        self.v[act] = np.maximum(0, self.v[act] + a[act] * DT)
        self.x[act] = (self.x[act] + self.v[act] * DT) % self.Lr
        if self.log is not None:
            self.log.append((self.x.copy(), self.v.copy(), self.active.copy()))
        return a, g, lead


def run_episode(density, planner, rng, warmup=600, hop_time=1.5, max_advance=80.0,
                startle_p=0.35, after=80, reserve_time=4.0, log=None):
    road = Ring(density, rng=rng)
    for k in range(warmup):
        if k == warmup - 50:
            road.log = log           # record the last 5 s before the hop too
        road.step()
    v_free = road.v.mean()
    f = 0                                        # hop-capable car
    lead = road.leaders()
    # candidate gaps ahead of f, within reach
    order = [f]
    for _ in range(12):
        order.append(lead[order[-1]])
    v_f = road.v[f]
    land_err = 1.0 if planner == "naive" else 0.3
    best, best_score = None, -np.inf
    for k in range(1, len(order) - 1):
        back, front = order[k], order[k + 1]
        if back == f or front == f:
            continue
        rel = (road.x[back] - road.x[f]) % road.Lr
        if rel > max_advance:
            break
        if planner == "naive":
            gap = (road.x[front] - road.x[back]) % road.Lr - L_CAR
            score = gap
        else:   # predict positions at touchdown with current speeds
            xb = road.x[back] + road.v[back] * hop_time
            xf = road.x[front] + road.v[front] * hop_time
            gap = (xf - xb) % road.Lr - L_CAR
            # time headway each side if we land in the middle at the leader's speed
            half = (gap - L_CAR) / 2
            score = min(half / max(road.v[back], 1), half / max(road.v[front], 1))
        if score > best_score:
            best, best_score = (back, front), score
    if best is None:
        return dict(outcome="no_gap", v_before=v_free)
    back, front = best
    if planner == "cooperative" and best_score < 1.0:
        # V2V slot reservation: ask the car behind the chosen gap to ease off early
        for _ in range(int(reserve_time / DT)):
            road.extra_brake[back] = 1.0; road.brake_timer[back] = DT * 1.5
            road.step()
        xb = road.x[back] + road.v[back] * hop_time
        xf = road.x[front] + road.v[front] * hop_time
        half = ((xf - xb) % road.Lr - 2 * L_CAR) / 2
        best_score = min(half / max(road.v[back], 1), half / max(road.v[front], 1))
    if planner != "naive" and best_score < 1.0:     # needs >=1 s headway both sides
        return dict(outcome="aborted", v_before=v_free)
    coop_back = back if planner == "cooperative" else None

    # take off: car leaves the lane
    road.active[f] = False
    x_take = road.x[f]
    # nearby human drivers may startle
    if planner != "cooperative":
        near = np.where(((x_take - road.x) % road.Lr < 60) & road.active)[0]
        hit = near[rng.random(len(near)) < startle_p]
        road.extra_brake[hit] = 3.0
        road.brake_timer[hit] = 1.5
    for _ in range(int(hop_time / DT)):
        if coop_back is not None:
            road.extra_brake[coop_back] = 1.0; road.brake_timer[coop_back] = DT * 1.5
        road.step()
    # touchdown in the middle of the actual gap now (plus control error)
    gap_mid = road.x[back] + ((road.x[front] - road.x[back]) % road.Lr) / 2
    road.x[f] = (gap_mid + rng.normal(0, land_err)) % road.Lr
    road.v[f] = road.v[front]
    road.active[f] = True
    # landing near a car can also startle humans
    if planner != "cooperative":
        near = np.where((((road.x[f] - road.x) % road.Lr) < 40) & road.active)[0]
        near = near[near != f]
        hit = near[rng.random(len(near)) < startle_p]
        road.extra_brake[hit] = 3.0; road.brake_timer[hit] = 1.5
    g_front = (road.x[front] - road.x[f]) % road.Lr - L_CAR
    g_back = (road.x[f] - road.x[back]) % road.Lr - L_CAR
    min_thw = min(g_front / max(road.v[f], 1), g_back / max(road.v[back], 1))
    crash = g_front < 0 or g_back < 0
    hard, min_speed = 0, np.inf
    for _ in range(after):
        a, g, _ = road.step()
        hard = max(hard, int((a < -4.0).sum()))
        min_speed = min(min_speed, road.v.min())
        if (g[road.active] < 0).any():
            crash = True
    return dict(outcome="hopped", crash=crash, min_thw=min_thw, hard_brakers=hard,
                min_speed=min_speed, v_before=v_free)
