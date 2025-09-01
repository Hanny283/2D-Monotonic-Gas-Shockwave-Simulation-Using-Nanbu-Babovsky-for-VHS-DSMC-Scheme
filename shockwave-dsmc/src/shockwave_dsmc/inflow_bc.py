import helpers as hf 
import numpy as np

def inflow_boundary(positions, velocities, L, meanV_left, meanV_right, Tx, dt, S, dx, rho):
    """ Updates particle positions and velocities using inflow boundary conditions. """

    # expected new particles (Eq. 65) – already includes dx
    newExpectedLeft  = hf.expected_new_particles_left(S, dx, dt, rho, meanV_left,  Tx)
    newExpectedRight = hf.expected_new_particles_right(S, dx, dt, rho, meanV_right, Tx)

    # particles that left the domain
    particlesLost = np.logical_or(positions < 0, positions > L)
    Nlost = int(np.sum(particlesLost))

    # nothing to replace
    if Nlost == 0:
        return positions, velocities

    # constant-N split (paper p.71), robust to denom ≈ 0
    denom = newExpectedLeft + newExpectedRight
    share_left = newExpectedLeft / (denom + 1e-30)
    actualLeft  = int(hf.Iround(np.array([share_left * Nlost]))[0])
    actualRight = int(Nlost - actualLeft)

    # sample inflow velocities
    new_velocitiesLeft  = hf.sample_from_flux_weighted_maxwellian_left(actualLeft,  meanV_left,  Tx)
    new_velocitiesRight = hf.sample_from_flux_weighted_maxwellian_right(actualRight, meanV_right, Tx)

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