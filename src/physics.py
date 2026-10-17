"""Cold-gas thruster physics for a 'flying' road car.

Models an adiabatic (isentropic) blowdown of a compressed-air tank through
choked nozzles, ideally expanded to sea-level pressure.

Calibrated to public figures from the Oct 15, 2026 Tesla Roadster reveal:
  - ~600 L of tank volume, ~250 kg of stored air (reported)
  - total thrust "above 80 kN" (reported)
Everything else (vehicle mass, nozzle efficiency) is an explicit assumption.
"""
import numpy as np

R, GAMMA, T0, PA, G = 287.0, 1.4, 300.0, 101_325.0, 9.81
V_TANK, M_AIR0 = 0.600, 250.0
P0 = M_AIR0 * R * T0 / V_TANK          # ideal-gas-equivalent start pressure
CHOKE = (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1)))
NOZZLE_EFF = 0.90                       # assumed velocity efficiency vs ideal


def tank_state(m):
    """Pressure and temperature left in the tank when m kg of air remains."""
    r = m / M_AIR0
    return P0 * r ** GAMMA, T0 * r ** (GAMMA - 1)


def exhaust_velocity(P, T):
    if P <= PA:
        return 0.0
    ve = np.sqrt(2 * GAMMA / (GAMMA - 1) * R * T * (1 - (PA / P) ** ((GAMMA - 1) / GAMMA)))
    return NOZZLE_EFF * ve


def _area(initial_thrust=80_000.0):
    P, T = tank_state(M_AIR0)
    mdot0 = initial_thrust / exhaust_velocity(P, T)
    return mdot0 / (P * np.sqrt(GAMMA / (R * T)) * CHOKE)


def full_throttle_burn(initial_thrust=80_000.0, dt=0.002):
    """Open all nozzles at once; throat area is sized for the reported initial thrust."""
    area = _area(initial_thrust)
    t, m, rows = 0.0, M_AIR0, []
    while m > 1.0:
        P, T = tank_state(m)
        ve = exhaust_velocity(P, T)
        if ve <= 0:
            break
        mdot = area * P * np.sqrt(GAMMA / (R * T)) * CHOKE
        rows.append((t, mdot * ve, m, P / 1e5))
        m -= mdot * dt
        t += dt
    return np.array(rows)  # t [s], thrust [N], air left [kg], pressure [bar]


def max_hover_time(vehicle_dry_mass, dt=0.002):
    """Throttle the jets to exactly hold the car's weight until the tank can't."""
    area, m, t = _area(), M_AIR0, 0.0
    while m > 1.0:
        P, T = tank_state(m)
        ve = exhaust_velocity(P, T)
        need = (vehicle_dry_mass + m) * G
        cap = area * P * np.sqrt(GAMMA / (R * T)) * CHOKE * ve   # max thrust still available
        if ve <= 0 or cap < need:
            break
        m -= need / ve * dt
        t += dt
    return t, M_AIR0 - m


def hop_air_cost(vehicle_dry_mass, height, forward_push=0.0):
    """Air (kg) for one hop: launch to `height`, cushion the landing, add a forward push.
    Uses the exhaust velocity of a fresh tank, so this is an optimistic lower bound."""
    v_up = np.sqrt(2 * G * height)
    impulse = (vehicle_dry_mass + M_AIR0) * (2 * v_up + forward_push)
    return impulse / exhaust_velocity(*tank_state(M_AIR0)), 2 * v_up / G
