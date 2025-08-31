import helpers as hf 
import numpy as np
import inflow_bc as ib

def Nanbu_Babovsky_VHS_ShockWave(T_x0, T_y0, N, dt, n_tot, e, mu, rho, L, alpha, S, dx, meanV_left, meanV_right, num_cells):

   
# Sampe velocities from a 2d maxwellian distribution
    velocities = hf.sample_velocities_from_maxwellian_2d(T_x0, T_y0, N)

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
    cell_velocities = [velocities[particle_cell_indices == i] for i in range(1, num_cells+1)]
    cell_velocities = np.array(cell_velocities, dtype=object)
    
    cell_global_indices = [np.flatnonzero(particle_cell_indices == i) for i in range(1, num_cells+1)]

   

    for n in range(n_tot):

# compute an upper bound cross section for each cell 
        upper_bound_cross_sections = np.array([
        hf.compute_upper_bound_cross_section(cell) if len(cell) else 0.0
        for cell in cell_velocities
        ])

        
 # calculate the expected number of collisions for each cell 
        Nc_mask = hf.Iround((particles_per_cell * rho * dt * upper_bound_cross_sections) / (2*e))

# gets indices of particles to collide 
        sampled_indices = hf.sample_particle_indices_to_collide(Nc, cell_velocities)
    
        indices_i_global = []
        indices_j_global = []

        for c in range(num_cells):
            idx_local = sampled_indices[c]             # the local indices sampled in cell c
            if len(idx_local) == 0:
                continue

            # pair them locally
            pairs_loc = hf.pair_particle_indices(idx_local)
            indices_i_loc, indices_j_loc = hf.split_pairs(pairs_loc)

            # map to global
            idx_map = cell_global_indices[c]
            indices_i_global.append(idx_map[indices_i_loc])
            indices_j_global.append(idx_map[indices_j_loc])

        # flatten to global arrays
        indices_i = np.concatenate(indices_i_global) if indices_i_global else np.array([], dtype=int)
        indices_j = np.concatenate(indices_j_global) if indices_j_global else np.array([], dtype=int)

#collision step 
        if len(indices_i) > 0 and len(indices_j) > 0:

            v_rel = velocities[indices_j] - velocities[indices_i]
            v_rel_mag = np.linalg.norm(v_rel, axis=1)


            sigma_ij = hf.ArraySigma_VHS(v_rel_mag)

             # Apply acceptance-rejection method:
            # Accept collision if random number < sigma_ij/Sigma
            accept_condition = np.random.rand(len(indices_i)) * upper_bound_cross_sections < sigma_ij

            indices_i = indices_i[accept_condition]
            indices_j = indices_j[accept_condition]

            if len(indices_i) > 0:
                velocities = hf.collide_particles_2d(velocities, indices_i, indices_j)






            









 
 
 
 
 
 
     
 
