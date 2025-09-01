import numpy as np
import sys
from nanbu_babovsky import Nanbu_Babovsky_VHS_ShockWave, plot_shockwave_profile

def main():
    """
    Run the shockwave simulation with parameters matching the paper (Sec. 6.2).
    """
    print("Starting Nanbu-Babovsky VHS ShockWave simulation...")

    gamma = 2.0
    M = 3.0

    # Upstream/right (state 1) – this is the paper’s reference state
    rho_R = 1.0
    T_R   = 1.0
    u_R   = -M * np.sqrt(gamma * T_R)     # ≈ -4.2426

    # Rankine–Hugoniot for γ=2, using upstream Mach M:
    r = ((gamma + 1.0) * M**2) / ((gamma - 1.0) * M**2 + 2.0)         # ρ2/ρ1
    T2_over_T1 = ((2*gamma*M**2 - (gamma - 1.0)) * ((gamma - 1.0)*M**2 + 2.0)) / ((gamma + 1.0)**2 * M**2)

    # Downstream/left (state 2)
    rho_L = r * rho_R                      # ≈ 2.4545
    T_L   = T2_over_T1 * T_R               # ≈ 4.7531
    u_L   = (rho_R / rho_L) * u_R          # u2 = (ρ1/ρ2) u1  ≈ -1.73


    
    meanV_left  = u_L
    meanV_right = u_R
    T_x0 = T_L        # use left temperature where a single T is required
    T_y0 = T_L        # same in y (isotropic)

    # ----- Simulation / numerical params (aligned with paper) -----
    S         = 1.0
    L         = 15.0               # domain [-7.5, 7.5]
    num_cells = 50                 # paper often uses 50 space cells
    dx        = L / num_cells

    # ~500 particles per cell (paper): total N ≈ 50*500 = 25_000
    N   = 25_000
    dt  = 0.0025                    # Fig. 9 uses Δt = 0.025 for TRMC; good starting value for NB too
    n_tot = 8000                   # average over long window as in paper
    e   = 1.0                      # rarefied regime ε = 1.0
    mu  = 1.0
    alpha = 1.0

    
    density, temp, mean_velocity = Nanbu_Babovsky_VHS_ShockWave(
    N, dt, n_tot, e, mu, alpha,
    L, num_cells, S, dx,
    rho_L, u_L, T_L,
    rho_R, u_R, T_R   
    )

    print("Running simulation...")
    print("Simulation completed successfully!")
    print(f"Results shape: density={density.shape}, temp={temp.shape}, mean_velocity={mean_velocity.shape}")

    print("Creating plots...")
    fig = plot_shockwave_profile(density, temp, mean_velocity, num_cells, L)
    print("Plotting completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)