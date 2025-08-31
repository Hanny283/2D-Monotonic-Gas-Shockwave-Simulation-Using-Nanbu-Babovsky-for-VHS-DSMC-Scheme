import helpers as hf 
import numpy as np

def inflow_boundary(positions, velocities, L, meanV_left, meanV_right, Tx, dt, S, dx, rho):
    """ Updates particle positions and velocities using inflow boundary conditions. """

   # Compute expected number of incoming particles at left and right boundaries
    newExpectedLeft = hf.expected_new_particles_left(S, dx, dt, rho, meanV_left, Tx)
    newExpectedRight = hf.expected_new_particles_right(S, dx, dt, rho, meanV_right, Tx)

    # Track particles that left the domain
    particlesLost = np.logical_or(positions < 0, positions > L)
    Nlost = np.sum(particlesLost)  # Number of particles lost

    # Split new particles based on left/right inflow
    actualLeft = hf.Iround(newExpectedLeft * Nlost / (newExpectedLeft + newExpectedRight))
    actualRight = Nlost - actualLeft

    # Sample new velocities for inflow particles
    new_velocitiesLeft = hf.sample_from_flux_weighted_maxwellian_left(actualLeft, meanV_left, Tx)
    new_velocitiesRight = hf.sample_from_flux_weighted_maxwellian_right(actualRight, meanV_right, Tx)

    # Assign new positions
    new_positionsL = np.zeros(actualLeft) + new_velocitiesLeft[:, 0] * dt * np.random.rand(actualLeft)
    new_positionsR = np.full(actualRight, L) + new_velocitiesRight[:, 0] * dt * np.random.rand(actualRight)

    # Remove lost particles and add new ones
    positions = np.delete(positions, np.where(particlesLost))
    velocities = np.delete(velocities, np.where(particlesLost), axis=0)

    positions = np.concatenate((positions, new_positionsL, new_positionsR))
    velocities = np.concatenate((velocities, new_velocitiesLeft, new_velocitiesRight))

    return positions, velocities