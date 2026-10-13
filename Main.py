"""
Fault-Tolerant Small Wind Turbine Control Simulator
====================================================
- Simple PMSG + rotor dynamics model
- Perturb & Observe (P&O) MPPT algorithm
- Six-state finite state machine with fault detection:
  IDLE -> STARTUP -> MPPT -> RATED -> FAULT -> ESTOP

Author: Tas Noor
Date: June 2026
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# TURBINE / GENERATOR PARAMETERS
# ============================================================
J = 0.5                 # Rotor inertia (kg·m²)
B_friction = 0.01       # Viscous friction coefficient
K_t = 0.6               # Torque constant (Nm/A)
K_e = 0.6               # Back-EMF constant (V·s/rad)
R_gen = 1.5             # Generator internal resistance (Ω)
RATED_RPM = 900         # Rated rotor speed
OVERSPEED_RPM = 1100    # Overspeed fault threshold
MAX_CURRENT = 15.0      # Overcurrent fault threshold (A)
MAX_TEMP = 80.0         # Overtemperature fault threshold (°C)
DT = 0.05               # Simulation timestep (s)
T_END = 30.0            # Total simulation time (s)


# ============================================================
# WIND PROFILE
# ============================================================
def wind_profile(t):
    """
    Smoothly varying wind speed with a gust event.
    Base wind ~7 m/s, sinusoidal variation, plus a gust at t=15s.
    """
    base = 7.0 + 1.5 * np.sin(0.3 * t)
    gust = 4.0 * np.exp(-((t - 15.0) ** 2) / 4.0)
    return base + gust


# ============================================================
# STATE MACHINE + MPPT CONTROLLER
# ============================================================
class TurbineController:
    """
    Fault-tolerant controller for a small wind turbine.
    States: IDLE, STARTUP, MPPT, RATED, FAULT, ESTOP
    """

    def __init__(self):
        self.state = "IDLE"
        self.duty = 0.5              # DC/DC converter duty cycle command
        self.p_step = 0.01           # P&O perturbation step size
        self.prev_power = 0.0
        self.fault_reason = None

    def _check_faults(self, rpm, current, temp):
        """Return a fault string if any safety threshold is exceeded."""
        if rpm > OVERSPEED_RPM:
            return "OVERSPEED"
        if current > MAX_CURRENT:
            return "OVERCURRENT"
        if temp > MAX_TEMP:
            return "OVERTEMP"
        return None

    def update(self, rpm, power, current, temp):
        """Run one control cycle."""
        # Fault check overrides all normal operation
        fault = self._check_faults(rpm, current, temp)
        if fault:
            self.fault_reason = fault
            self.state = "FAULT"
            self.duty = 0.0          # Fail-safe: cut power
            return

        if self.state == "IDLE":
            if rpm > 100:
                self.state = "STARTUP"

        elif self.state == "STARTUP":
            self.duty = 0.5
            if rpm > 400:
                self.state = "MPPT"
                self.prev_power = power

        elif self.state == "MPPT":
            # Perturb & Observe
            if power > self.prev_power:
                self.duty += self.p_step
            else:
                self.duty -= self.p_step
            self.duty = np.clip(self.duty, 0.1, 0.95)
            self.prev_power = power

            if rpm > RATED_RPM:
                self.state = "RATED"

        elif self.state == "RATED":
            # Back off duty to hold rated RPM
            self.duty = np.clip(self.duty - 0.005, 0.1, 0.95)
            if rpm < RATED_RPM - 50:
                self.state = "MPPT"

        elif self.state == "FAULT":
            # Latch in FAULT until rotor slows down
            if rpm < 200:
                self.state = "IDLE"
                self.fault_reason = None


# ============================================================
# SIMULATION
# ============================================================
def simulate():
    t_array = np.arange(0, T_END, DT)
    omega = 0.0
    temp = 25.0
    controller = TurbineController()

    log = {
        "t": [], "rpm": [], "power": [], "duty": [],
        "state": [], "wind": [], "current": [], "temp": []
    }

    for t in t_array:
        v_wind = wind_profile(t)

        # Aerodynamic torque (simplified)
        T_aero = 0.5 * 1.225 * 0.8 * (v_wind ** 3) / max(omega, 1.0)
        T_aero = np.clip(T_aero, 0, 8)

        # Electrical torque (simplified buck behavior)
        T_elec = K_t * (omega * K_e / R_gen) * controller.duty

        # Rotor dynamics
        domega = (T_aero - T_elec - B_friction * omega) / J
        omega = max(omega + domega * DT, 0)

        # Derived quantities
        rpm = omega * 60 / (2 * np.pi)
        current = (omega * K_e / R_gen) * controller.duty
        power = current ** 2 * R_gen * 0.9    # rough electrical power

        # Simple thermal model
        temp += (current ** 2 * R_gen * 0.05 - 0.05 * (temp - 25)) * DT

        # Run controller
        controller.update(rpm, power, current, temp)

        # Log
        log["t"].append(t)
        log["rpm"].append(rpm)
        log["power"].append(power)
        log["duty"].append(controller.duty)
        log["state"].append(controller.state)
        log["wind"].append(v_wind)
        log["current"].append(current)
        log["temp"].append(temp)

    return log


# ============================================================
# PLOTTING
# ============================================================
def plot_results(log):
    fig, axs = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    # Wind speed
    axs[0].plot(log["t"], log["wind"], color="tab:blue")
    axs[0].set_ylabel("Wind (m/s)")
    axs[0].set_title("Fault-Tolerant Turbine Control Simulation")
    axs[0].grid(alpha=0.3)

    # RPM with thresholds
    axs[1].plot(log["t"], log["rpm"], color="tab:red")
    axs[1].axhline(RATED_RPM, ls="--", color="gray", label="Rated")
    axs[1].axhline(OVERSPEED_RPM, ls=":", color="black", label="Overspeed")
    axs[1].set_ylabel("RPM")
    axs[1].legend(loc="upper left", fontsize=8)
    axs[1].grid(alpha=0.3)

    # Power output
    axs[2].plot(log["t"], log["power"], color="tab:green")
    axs[2].set_ylabel("Power (W)")
    axs[2].grid(alpha=0.3)

    # Duty cycle
    axs[3].plot(log["t"], log["duty"], color="tab:orange")
    axs[3].set_ylabel("Duty cycle")
    axs[3].grid(alpha=0.3)

    # State timeline
    state_map = {"IDLE": 0, "STARTUP": 1, "MPPT": 2,
                 "RATED": 3, "FAULT": 4, "ESTOP": 5}
    state_nums = [state_map[s] for s in log["state"]]
    axs[4].plot(log["t"], state_nums, color="tab:purple", drawstyle="steps-post")
    axs[4].set_yticks(list(state_map.values()))
    axs[4].set_yticklabels(list(state_map.keys()), fontsize=8)
    axs[4].set_ylabel("State")
    axs[4].set_xlabel("Time (s)")
    axs[4].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("turbine_control_plot.png", dpi=150)
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    log = simulate()
    plot_results(log)
    print(f"Final state: {log['state'][-1]}")
    print(f"Peak RPM: {max(log['rpm']):.1f}")
    print(f"Peak power: {max(log['power']):.1f} W")