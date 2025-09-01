import numpy as np
import matplotlib.pyplot as plt
import helpers as hf
import inflow_bc as ib

def Nanbu_Babovsky_VHS_ShockWave(N, dt, n_tot, e, mu, alpha, L, num_cells, S, dx, rho_left, u_left, T_left, rho_right, u_right, T_right):


# Sampe velocities from a 2d maxwellian distribution
    velocities = hf.sample_velocities_from_maxwellian_2d(T_left, T_left, N)

# assign each velocity a position in the spatial domain (this means positions and velocities have the same length)
    positions = hf.assign_positions(velocities, L)

# discretize the spatial domain into cells
    cell_width = L / num_cells

#assign each particle position an index that corresponds to a cell 
    particle_cell_indices = np.floor(positions / cell_width).astype(int)

# set the index of the last cell to the last cell index
    particle_cell_indices[particle_cell_indices == num_cells] = num_cells - 1

# count the number of particles in each cell
    particles_per_cell = np.bincount(particle_cell_indices, minlength=num_cells)

# create a list of velocities for each cell
    cell_velocities = [velocities[particle_cell_indices == i] for i in range(num_cells)]
    cell_velocities = np.array(cell_velocities, dtype=object)
    
    cell_global_indices = [np.flatnonzero(particle_cell_indices == i) for i in range(num_cells)]

    density = np.zeros(num_cells)          # time accumulator of n per cell
    temp = np.zeros(num_cells)             # time accumulator of T per cell
    mean_velocity = np.zeros((num_cells, 2))  # time accumulator of u per cell
    averaging_count = 0
    t = 0.0

    w = rho_left * S * dx / (N / num_cells)



   

    for n in range(n_tot):


    # compute an upper bound cross section for each cell 
        upper_bound_cross_sections = np.array([
        hf.compute_upper_bound_cross_section(cell) if len(cell) else 0.0
        for cell in cell_velocities
        ])

        n_local = particles_per_cell / (S * dx)   # per-cell number density estimate

    # calculate the expected number of collisions for each cell 
        Nc = np.minimum(
            hf.Iround((particles_per_cell * n_local * dt * upper_bound_cross_sections) / (2*e)),
            particles_per_cell // 2
        )


    # gets indices of particles to collide 
        sampled_indices = hf.sample_particle_indices_to_collide(Nc, cell_velocities)

        indices_i_global = []
        indices_j_global = []

        for c in range(num_cells):
            idx_local = sampled_indices[c]             # the local indices sampled in cell c
            if len(idx_local) == 0:
                continue

            # pair them locally (get the single (n_pairs, 2) array)
            pairs_loc = hf.pair_particle_indices(idx_local)[0]   # (n_pairs, 2)
            indices_i_loc = pairs_loc[:, 0]                      # 1-D
            indices_j_loc = pairs_loc[:, 1]                      # 1-D

            # map to global (still 1-D)
            idx_map = cell_global_indices[c]
            indices_i_global.append(idx_map[indices_i_loc])
            indices_j_global.append(idx_map[indices_j_loc])


        indices_i = (np.concatenate(indices_i_global) if indices_i_global else np.array([], dtype=int)).ravel()
        indices_j = (np.concatenate(indices_j_global) if indices_j_global else np.array([], dtype=int)).ravel()


    #collision step 
        if len(indices_i) > 0 and len(indices_j) > 0:

            # --- FORCE 1-D SHAPES ---
            indices_i = np.asarray(indices_i).reshape(-1)
            indices_j = np.asarray(indices_j).reshape(-1)

            # relative speed per pair -> (M,)
            v_rel = velocities[indices_j] - velocities[indices_i]      # (M, 2)
            v_rel_mag = np.linalg.norm(v_rel, axis=1)                  # (M,)

            # per-pair sigma(v) -> (M,)
            sigma_ij = hf.ArraySigma_VHS(v_rel_mag).reshape(-1)        # (M,)

            # per-pair Sigma (cell upper bound) -> (M,)
            Sigma_pairs = upper_bound_cross_sections[
                particle_cell_indices[indices_i]
            ].reshape(-1)                                              # (M,)

            # acceptance mask -> (M,)
            u = np.random.rand(indices_i.size) * Sigma_pairs           # (M,)
            accept_condition = u < sigma_ij                            # (M,)

            # filter pairs
            indices_i = indices_i[accept_condition]
            indices_j = indices_j[accept_condition]

            if len(indices_i) > 0:
                velocities = hf.collide_particles(velocities, indices_i, indices_j)
            
        positions = hf.update_positions(positions, velocities, dt)

        positions, velocities = ib.inflow_boundary(
            positions, velocities, L,
            u_left, u_right, T_left,
            dt, S, dx, rho_left
        )


        #assign each particle position an index that corresponds to a cell 
        particle_cell_indices = np.floor(positions / cell_width).astype(int)

        # set the index of the last cell to the last cell index
        particle_cell_indices[particle_cell_indices == num_cells] = num_cells - 1

        # count the number of particles in each cell
        particles_per_cell = np.bincount(particle_cell_indices, minlength=num_cells)

        cell_velocities = [velocities[particle_cell_indices == i] for i in range(num_cells)]
        cell_velocities = np.array(cell_velocities, dtype=object)

        cell_global_indices = [np.flatnonzero(particle_cell_indices == i) for i in range(num_cells)]

        t += dt

        # start accumulating after t>5, and cap ~8000 samples (per the paper)
        if t > 5.0 and averaging_count < 8000:
            n_inst = np.zeros(num_cells)
            T_inst = np.zeros(num_cells)
            u_inst = np.zeros((num_cells, 2))

            # compute per-cell moments
            for c in range(num_cells):
                mask = (particle_cell_indices == c)
                Nc = np.count_nonzero(mask)
                if Nc == 0:
                    continue

                v_c = velocities[mask]                  # (Nc, 2)
                u_c = v_c.mean(axis=0)                  # (2,)
                cpec = v_c - u_c                        # (Nc, 2)
                T_c = 0.5 * np.mean(np.sum(cpec**2, axis=1))  # scalar

                # number density n = N_cell / (S * dx)
                n_inst[c] = (w * Nc) / (S * dx)
                T_inst[c] = T_c
                u_inst[c] = u_c

            # accumulate time averages
            density += n_inst
            temp += T_inst
            mean_velocity += u_inst
            averaging_count += 1



    if averaging_count > 0:
        density /= averaging_count
        temp /= averaging_count
        mean_velocity /= averaging_count


    return density, temp, mean_velocity
    

    
def plot_shockwave_profile(density, temp, mean_velocity, num_cells, L):
    """
    Plot the shockwave profile showing density, temperature, and mean velocity.
    Only plots the middle section from -4.5 to 4.5 of the full spatial domain.
    
    Parameters
    ----------
    density : array
        Density values for each cell (length: num_cells)
    temp : array
        Temperature values for each cell (length: num_cells)
    mean_velocity : array
        Mean velocity values for each cell (shape: num_cells x 2)
    num_cells : int
        Number of cells in the spatial domain
    L : float
        Total length of the spatial domain
    """
    # Calculate the range we want to plot: [-4.5, 4.5]
    plot_range = 9.0  # total range to plot
    
    # Calculate the start and end indices for the plotting range
    # We want the middle 9 units centered at 0
    start_pos = (L - plot_range) / 2  # This gives us the starting position
    end_pos = start_pos + plot_range   # This gives us the ending position
    
    # Convert positions to cell indices
    cell_width = L / num_cells
    start_idx = int(start_pos / cell_width)
    end_idx = int(end_pos / cell_width)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(num_cells, end_idx)
    
    # Extract the subset of data we want to plot
    density_subset = density[start_idx:end_idx]
    temp_subset = temp[start_idx:end_idx]
    mean_velocity_subset = mean_velocity[start_idx:end_idx]
    
    # Create x-coordinates for the plotting range
    x_coords = np.linspace(-4.5, 4.5, len(density_subset))
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot density
    ax1.plot(x_coords, density_subset, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Density')
    ax1.set_title('Shockwave Density Profile')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-4.5, 4.5)
    
    # Plot temperature
    ax2.plot(x_coords, temp_subset, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Shockwave Temperature Profile')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4.5, 4.5)
    
    # Plot mean velocity components
    ax3.plot(x_coords, mean_velocity_subset[:, 0], 'g-', linewidth=2, marker='^', markersize=4, label='v_x')
    ax3.plot(x_coords, mean_velocity_subset[:, 1], 'm-', linewidth=2, marker='v', markersize=4, label='v_y')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Mean Velocity')
    ax3.set_title('Shockwave Mean Velocity Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-4.5, 4.5)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig


        









                









    
    
    
    
    
    
        
    
