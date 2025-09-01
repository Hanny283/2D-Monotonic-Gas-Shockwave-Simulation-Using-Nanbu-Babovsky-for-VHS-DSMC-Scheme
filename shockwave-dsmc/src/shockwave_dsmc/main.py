import numpy as np
import sys
from nanbu_babovsky import Nanbu_Babovsky_VHS_ShockWave, plot_shockwave_profile

def main():
    """
    Run the shockwave simulation with parameters matching the paper (Sec. 6.2).
    """
    print("Starting Nanbu-Babovsky VHS ShockWave simulation...")

    # ----- Physical / model params -----
    gamma = 2.0      # 2D monoatomic gas in velocity space (paper)
    M     = 3.0      # upstream Mach number used in RH relations

    # Downstream (RIGHT) macrostates fixed by the paper:
    rho_R = 1.0
    T_R   = 1.0
    u_R   = -M * np.sqrt(gamma * T_R)   # flow to the left

    # Upstream (LEFT) macrostates from Rankine–Hugoniot (using upstream Mach = M)
    r = ((gamma + 1.0) * M**2) / ((gamma - 1.0) * M**2 + 2.0)                  # rho2/rho1
    T2_over_T1 = ((2*gamma*M**2 - (gamma - 1.0)) * ((gamma - 1.0)*M**2 + 2.0)) / ((gamma + 1.0)**2 * M**2)

    rho_L = rho_R / r
    T_L   = T_R / T2_over_T1
    u_L   = (rho_R / rho_L) * u_R   # continuity: rho_L*u_L = rho_R*u_R  -> u_L = r*u_R

    # For your code paths that still expect these names:
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
    dt  = 0.025                    # Fig. 9 uses Δt = 0.025 for TRMC; good starting value for NB too
    n_tot = 8000                   # average over long window as in paper
    e   = 1.0                      # rarefied regime ε = 1.0
    mu  = 1.0
    alpha = 1.0

    # >>> call your solver; if you've updated inflow_boundary to use left/right states, prefer that API:
    density, temp, mean_velocity = Nanbu_Babovsky_VHS_ShockWave(
        T_x0, T_y0, N, dt, n_tot, e, mu,
        rho_R,                # keep if a single 'rho' is still required somewhere; not used for inflow
        L, alpha, S, dx,
        meanV_left, meanV_right,  # these now carry u_L, u_R
        num_cells
    )
    print(f"Parameters:")
    print(f"  T_x0: {T_x0}")
    print(f"  T_y0: {T_y0}")
    print(f"  rho: {rho}")
    print(f"  mu: {mu}")
    print(f"  alpha: {alpha}")
    print(f"  S: {S}")
    print(f"  L: {L}")
    print(f"  num_cells: {num_cells}")
    print(f"  dx: {dx}")
    print(f"  N: {N}")
    print(f"  dt: {dt}")
    print(f"  n_tot: {n_tot}")
    print(f"  e: {e}")
    print(f"  M: {M}")
    print(f"  gamma: {gamma}")
    print(f"  ux: {ux}")
    print(f"  uy: {uy}")
    print(f"  meanV_left: {meanV_left}")
    print(f"  meanV_right: {meanV_right}")
    print()
    
    try:
        # Run the simulation
        print("Running simulation...")
        density, temp, mean_velocity = Nanbu_Babovsky_VHS_ShockWave(
            T_x0=T_x0,
            T_y0=T_y0,
            N=N,
            dt=dt,
            n_tot=n_tot,
            e=e,
            mu=mu,
            rho=rho,
            L=L,
            alpha=alpha,
            S=S,
            dx=dx,
            meanV_left=meanV_left,
            meanV_right=meanV_right,
            num_cells=num_cells
        )
        
        print("Simulation completed successfully!")
        print(f"Results shape:")
        print(f"  density: {density.shape}")
        print(f"  temp: {temp.shape}")
        print(f"  mean_velocity: {mean_velocity.shape}")
        print()
        
        # Plot the results
        print("Creating plots...")
        fig = plot_shockwave_profile(density, temp, mean_velocity, num_cells, L)
        
        print("Plotting completed!")
        
        # Optional: Save the figure
        # fig.savefig('shockwave_profile.png', dpi=300, bbox_inches='tight')
        # print("Figure saved as 'shockwave_profile.png'")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)