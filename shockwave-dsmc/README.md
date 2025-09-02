# 2D Monotonic Gas Shockwave Simulation Using Nanbu-Babovsky for VHS DSMC Scheme

A Direct Simulation Monte Carlo (DSMC) implementation for simulating 2D gas shockwave dynamics using the Nanbu-Babovsky collision algorithm with Variable Hard Sphere (VHS) cross-sections.

## Overview

This project implements a 1D spatial domain with 2D velocity space DSMC simulation to study shockwave formation and propagation in a monoatomic gas. The simulation uses:

- **Nanbu-Babovsky collision algorithm** for particle interactions
- **Variable Hard Sphere (VHS) model** for collision cross-sections
- **Flux-weighted Maxwellian sampling** for inflow boundary conditions
- **Rankine-Hugoniot relations** for shockwave physics

## Features

- Efficient flux-weighted Maxwellian velocity sampling for boundary conditions
- 1D spatial domain with 2D velocity space
- Inflow boundary conditions with proper particle injection
- Real-time collision detection and processing
- Comprehensive shockwave profile visualization
- Configurable simulation parameters

## Project Structure

```
shockwave-dsmc/
├── src/
│   └── shockwave_dsmc/
│       ├── main.py              # Main simulation script
│       ├── nanbu_babovsky.py    # Core DSMC algorithm
│       ├── helpers.py           # Utility functions and sampling
│       └── inflow_bc.py         # Boundary condition handling
├── data/
│   ├── figures/                 # Output plots
│   └── outputs/                 # Simulation data
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd shockwave-dsmc
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

### Running the Simulation

Navigate to the source directory and run the main script:

```bash
cd src/shockwave_dsmc
python main.py
```

### Simulation Parameters

The simulation can be configured by modifying parameters in `main.py`:

```python
# Physical parameters
T_x0 = 1.0          # Initial x-temperature
T_y0 = 1.0          # Initial y-temperature
rho = 1.0           # Density
mu = 1.0            # Molecular mass
alpha = 1.0         # VHS parameter
S = 1.0             # Cross-sectional area
L = 15.0            # Domain length
num_cells = 200     # Number of spatial cells

# Simulation parameters
N = 500             # Number of particles
dt = 0.01           # Time step
n_tot = 8000        # Total time steps
e = 1.0             # Energy parameter

# Boundary conditions
M = 3.0             # Mach number
meanV_left = 3.0 * np.sqrt(2)   # Left boundary velocity
meanV_right = -4.24             # Right boundary velocity
```

### Output

The simulation generates:

1. **Console output**: Progress updates and parameter information
2. **Plots**: Three subplots showing:
   - Density profile across the domain
   - Temperature profile
   - Mean velocity components (x and y)
3. **Data arrays**: `density`, `temp`, and `mean_velocity` for further analysis

## Key Functions

### Core Simulation
- `Nanbu_Babovsky_VHS_ShockWave()`: Main simulation function
- `sample_particle_indices_to_collide()`: Particle selection for collisions
- `collide_particles()`: Nanbu-Babovsky collision algorithm

### Boundary Conditions
- `sample_from_flux_weighted_maxwellian_left()`: Left boundary particle injection
- `sample_from_flux_weighted_maxwellian_right()`: Right boundary particle injection
- `inflow_boundary()`: Boundary condition management

### Visualization
- `plot_shockwave_profile()`: Generate shockwave profile plots

## Physics Implementation

### Flux-Weighted Maxwellian Sampling

The simulation uses analytical inverse sampling for flux-weighted Maxwellian distributions:

```python
# For left boundary (vx > 0)
f(vx) ∝ vx * exp(-(vx - meanV)²/(2T))

# For right boundary (vx < 0)  
f(vx) ∝ |vx| * exp(-(vx - meanV)²/(2T))
```

### Collision Algorithm

The Nanbu-Babovsky algorithm:
1. Samples particle pairs for potential collisions
2. Calculates collision probabilities using VHS cross-sections
3. Applies acceptance-rejection for collision events
4. Updates particle velocities post-collision

### Boundary Conditions

- **Left boundary**: Particles injected with positive x-velocity
- **Right boundary**: Particles injected with negative x-velocity
- **Outflow**: Particles leaving the domain are removed

## Results

The simulation produces shockwave profiles showing:
- **Density jump**: Characteristic increase across the shock
- **Temperature rise**: Heating due to compression
- **Velocity change**: Flow deceleration through the shock

## Customization

### Adding New Collision Models

Modify the `ArraySigma_VHS()` function in `helpers.py`:

```python
def ArraySigma_VHS(v):
    # Custom cross-section model
    return your_cross_section_function(v)
```

### Changing Boundary Conditions

Update the boundary condition functions in `inflow_bc.py` or modify the sampling functions in `helpers.py`.

### Visualization Options

Customize plots by modifying `plot_shockwave_profile()` in `nanbu_babovsky.py`.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all files are in the same directory
2. **Plot not showing**: Check matplotlib backend settings
3. **Memory issues**: Reduce `N` (number of particles) or `n_tot` (time steps)

### Performance Tips

- Use fewer particles for faster testing
- Reduce time steps for quick visualization
- Increase `num_cells` for higher spatial resolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## References

- Nanbu, K. (1980). "Theoretical basis of the direct simulation Monte Carlo method"
- Babovsky, H. (1989). "On a simulation scheme for the Boltzmann equation"
- Bird, G.A. (1994). "Molecular Gas Dynamics and the Direct Simulation of Gas Flows"

## Contact

[Add your contact information here]
