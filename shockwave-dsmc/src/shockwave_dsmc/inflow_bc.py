import helpers as hf 
import numpy as np

def inflow_boundary(positions, velocities, L,
                    meanV_left, meanV_right, T_left, T_right,
                    dt, S, dx, rho_left, rho_right):
    """ Updates particle positions and velocities using inflow boundary conditions. """

    # expected new particles 
    newExpectedLeft  = hf.expected_new_particles_left(S, dx, dt, rho_left,  meanV_left,  T_left)
    newExpectedRight = hf.expected_new_particles_right(S, dx, dt, rho_right, meanV_right, T_right)

    # particles that left the domain
    particlesLost = np.logical_or(positions < 0, positions > L)
    Nlost = int(np.sum(particlesLost))

    # nothing to replace
    if Nlost == 0:
        return positions, velocities

    # constant-N split 
    denom = newExpectedLeft + newExpectedRight
    share_left = newExpectedLeft / (denom + 1e-30)
    actualLeft  = int(hf.Iround(np.array([share_left * Nlost]))[0])
    actualRight = int(Nlost - actualLeft)

    # sample inflow velocities
    new_velocitiesLeft  = hf.sample_from_flux_weighted_maxwellian_left(actualLeft,  meanV_left,  T_left)
    new_velocitiesRight = hf.sample_from_flux_weighted_maxwellian_right(actualRight, meanV_right, T_right)


    # place new particles: x* = b + vx* dt * ξ  (b=0 left, b=L right)
    if actualLeft > 0:
        new_positionsL = np.zeros(actualLeft) + new_velocitiesLeft[:, 0] * dt * np.random.rand(actualLeft)
    else:
        new_positionsL = np.empty(0, dtype=positions.dtype)

    if actualRight > 0:
        new_positionsR = np.full(actualRight, L) + new_velocitiesRight[:, 0] * dt * np.random.rand(actualRight)
    else:
        new_positionsR = np.empty(0, dtype=positions.dtype)

    # remove lost, then append replacements
    keep = ~particlesLost
    positions  = positions[keep]
    velocities = velocities[keep]

    positions  = np.concatenate((positions,  new_positionsL, new_positionsR))
    velocities = np.concatenate((velocities, new_velocitiesLeft, new_velocitiesRight))

    return positions, velocities