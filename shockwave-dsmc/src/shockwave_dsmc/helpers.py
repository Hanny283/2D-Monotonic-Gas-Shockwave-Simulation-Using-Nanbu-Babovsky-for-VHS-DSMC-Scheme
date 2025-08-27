import numpy as np 


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

def compute_upper_bound_cross_section(velocities):
    """
    Computes the upper bound of the cross section for each particle with velocity v_i in the array velocities in the shockwave profile simulation. Note that this
    function assumes that C_alpha = 1 and alpha = 1.

    Parameters
    ----------
    velocities : numpy array of particles velocities

    Returns
    -------
    upper_bound_cross_section : numpy array of shape (N,1)
    """
    delta_v  = np.max(np.linalg.norm(velocities - np.mean(velocities,axis = 0)))

    return 2 * delta_v

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