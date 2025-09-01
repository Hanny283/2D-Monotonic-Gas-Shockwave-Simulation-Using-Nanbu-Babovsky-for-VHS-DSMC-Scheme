import numpy as np 
from numpy import sqrt, exp, pi
from scipy.special import erf
from scipy.integrate import quad


def sample_velocities_from_maxwellian_2d(T_x0, T_y0, N):
    """
    Sample from a 2D Maxwellian distribution.

    Parameters
    ----------
    T_x0 : float
    The x-component of the temperature.

    T_y0 : float
    The y-component of the temperature.

    N : int 
    The number of velocities to draw
    Returns a numpy array of shape (N, 2) containing the sampled velocities.
    """

    v_x_samples = np.random.normal(0, np.sqrt(T_x0), N)
    v_y_samples = np.random.normal(0, np.sqrt(T_y0), N)

    return np.column_stack((v_x_samples, v_y_samples))

def assign_positions(velocities, L):
    """
    Assign positions to particles on a 1-dimensional spatial domain.

    Parameters
    ----------
    velocities : numpy array of particles velocities
    L: float 
    The length of the spatial domain

    Returns
    -------
    positions : numpy array of shape (N, 2)
        The positions of the particles.
    """
    num_particles = len(velocities)
    positions = np.random.uniform(0, L, num_particles)
    
    return positions

def compute_upper_bound_cross_section(velocities: np.ndarray) -> float:
    """
    velocities: (N, 2) array for one cell.
    Returns 2 * max_i ||v_i - mean(v)||.
    """
    if velocities.size == 0:
        return 0.0  # or np.nan, depending on how you want to treat empty cells
    v_mean = velocities.mean(axis=0)                         # shape (2,)
    delta_v = np.linalg.norm(velocities - v_mean, axis=1).max()
    return 2.0 * delta_v

def Iround(x):
    """
    Vectorized probabilistic rounding of an array of floats.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array of rounded integers.
    """
    lower = np.floor(x).astype(int)
    prob = x - lower
    random_numbers = np.random.rand(*x.shape)
    return lower + np.where(random_numbers < prob, 1, 0)

def compute_total_collisions_for_each_cell(particles_per_cell, rho, Sigma, delta_t, epsilon):
    """
    Computes the expected total number of collisions for a given number of particles, pressure, cross section, time step, and epsilon.

    Parameters
    ----------
    N : int (number of particles)
    p : float (rho)
    Sigma : float (upper bound of the cross section)
    delta_t : float (time step)
    epsilon : float 

    Returns
    -------
    total_collisions : int (expected total number of collisions)
    """
    return np.minimum(Iround((particles_per_cell * rho * delta_t * Sigma) / (2*epsilon)), particles_per_cell//2)

def collide_particles(velocities, indices_i, indices_j):
    """
    Collides M particles with velocities given by the arraysv_i and v_j.

    Parameters
    ----------
    velocities : numpy array of shape (N, 2)
    indices_i : numpy array of shape (M,)
    indices_j : numpy array of shape (M,)
    """
    v_i = velocities[indices_i]
    v_j = velocities[indices_j]

    v_rel = v_i - v_j                      
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True)  

    theta = np.random.uniform(0.0, 2.0*np.pi, size=len(indices_i))
    omega = np.column_stack((np.cos(theta), np.sin(theta))) 

    v_cm = 0.5 * (v_i + v_j)

    v_i_prime = v_cm + 0.5 * v_rel_mag * omega
    v_j_prime = v_cm - 0.5 * v_rel_mag * omega

    velocities[indices_i] = v_i_prime
    velocities[indices_j] = v_j_prime

    return velocities

def _phi(z):  # standard normal pdf
    return np.exp(-0.5*z*z)/np.sqrt(2*np.pi)

def _Phi(z):  # standard normal cdf
    return 0.5*(1.0 + erf(z/np.sqrt(2.0)))

def _sample_vx_incoming(N, u, T):
    """
    Sample x-velocities from the half-range flux-weighted Maxwellian:
        p(x) ∝ x * exp(-(x - u)^2 / (2T)),   x > 0
    via inverse transform + Newton on the closed-form CDF.
    """
    sig = np.sqrt(T)
    U = np.random.rand(N)

    if abs(u) < 1e-12:
        # Rayleigh (exact): x = sqrt(-2 T ln(1-U))
        return np.sqrt(-2.0*T*np.log1p(-U))

    a = u/sig
    Phi_a = _Phi(a)
    exp_a = np.exp(-0.5*a*a)

    # normalizer C = u*sig*sqrt(2π)*Phi(a) + sig^2*exp(-a^2/2)
    C = u*sig*np.sqrt(2*np.pi)*Phi_a + sig*sig*exp_a

    # good initial guess: shifted Rayleigh
    x = np.maximum(0.0, u) + sig*np.sqrt(-2.0*np.log1p(-U))

    # Newton iterations (3–6 is plenty)
    Phi_neg_a = _Phi(-a)
    for _ in range(6):
        z = (x - u)/sig
        Phi_z = _Phi(z)
        exp_z = np.exp(-0.5*z*z)

        # CDF G(x)
        G = (u*sig*np.sqrt(2*np.pi)*(Phi_z - Phi_neg_a) - sig*sig*(exp_z - exp_a)) / C
        # derivative G'(x)
        Gp = (x * exp_z) / C

        # Newton step
        x_new = x - (G - U)/Gp
        # keep in domain
        x = np.maximum(x_new, 0.0)

    return x

def sample_from_flux_weighted_maxwellian_left(N, meanV, T):
    """
    Incoming from the LEFT boundary → v_x > 0 with drift u = meanV.
    v_y is independent ~ N(0, T).
    """
    vx = _sample_vx_incoming(N, meanV, T)          # x > 0
    vy = np.random.normal(0.0, np.sqrt(T), N)
    return np.column_stack((vx, vy))

def sample_from_flux_weighted_maxwellian_right(N, meanV, T):
    """
    Incoming from the RIGHT boundary → v_x < 0.
    Equivalent to sampling x>0 with drift -u and then negating.
    """
    vx_pos = _sample_vx_incoming(N, -meanV, T)     # x > 0 with drift -u
    vx = -vx_pos                                   # make negative
    vy = np.random.normal(0.0, np.sqrt(T), N)
    return np.column_stack((vx, vy))

def expected_new_particles_left(S, dx, dt, rho_L, u_L, T_L):
    sigma = np.sqrt(T_L)
    a = u_L / sigma
    return S * dx * dt * rho_L * (u_L * _Phi(a) + sigma * _phi(a))

def expected_new_particles_right(S, dx, dt, rho_R, u_R, T_R):
    sigma = np.sqrt(T_R)
    a = u_R / sigma
    return S * dx * dt * rho_R * (sigma * _phi(a) - u_R * _Phi(-a))

def sample_particle_indices_to_collide(Nc, cell_velocities):
    """
    Returns: list of 1-D int arrays (per-cell).
    """
    Nc = np.asarray(Nc, dtype=int)
    if len(Nc) != len(cell_velocities):
        raise ValueError("Length of Nc must match number of cells in cell_velocities")

    sampled_indices = []
    for g, k in zip(cell_velocities, Nc):
        if k < 0:
            raise ValueError("k must be nonnegative")
        if 2 * k > g.shape[0]:
            raise ValueError(f"Cannot sample {2*k} from group with {g.shape[0]} rows")
        # always a 1-D int array (possibly empty if k == 0)
        idx = np.random.choice(g.shape[0], size=2 * k, replace=False)
        sampled_indices.append(idx.astype(int))
    return sampled_indices

def pair_particle_indices(sampled_indices):
    """
    Accepts:
      - list of 1-D arrays (preferred), OR
      - a single 1-D int array (treated as one cell)
    Returns:
      list of (n_pairs, 2) int arrays
    """
    # Normalize input to a list of 1-D arrays
    if isinstance(sampled_indices, np.ndarray) and sampled_indices.dtype != object:
        # If it's a flat 1-D int array, wrap as one cell
        if sampled_indices.ndim == 1 and np.issubdtype(sampled_indices.dtype, np.integer):
            arrays = [sampled_indices]
        else:
            arrays = list(sampled_indices)  # best effort
    else:
        arrays = list(sampled_indices)

    paired = []
    for idx in arrays:
        idx = np.asarray(idx)
        if idx.ndim == 0:            # scalar -> make it length-1
            idx = idx.reshape(1)
        if idx.size == 0:            # nothing to pair
            paired.append(idx.reshape(0, 2))
            continue
        if idx.size % 2 != 0:
            raise ValueError("Number of indices must be even to form pairs.")
        shuffled = np.random.permutation(idx)
        pairs = shuffled.reshape(-1, 2)
        paired.append(pairs.astype(int))
    return paired


def split_pairs(paired_indices):
    """
    Split paired indices into two arrays: first indices and second indices.

    Parameters
    ----------
    paired_indices : list of np.ndarray
        Each element is a 2D NumPy array of shape (num_pairs, 2).

    Returns
    -------
    first_indices : list of np.ndarray
        List of column vectors containing the first index from each pair.
    second_indices : list of np.ndarray
        List of column vectors containing the second index from each pair.
    """
    first, second = [], []
    for pairs in paired_indices:
        first.append(pairs[:, 0])   # 1-D
        second.append(pairs[:, 1])  # 1-D
    return first, second


def ArraySigma_VHS(v):
    Constant = 1.0
    alpha = 1
    return Constant * np.power(v, alpha)

def update_positions(positions, velocities, dt):
    # positions: (N,), velocities: (N,2)
    # advance only along x with vx
    return positions + velocities[:, 0] * dt