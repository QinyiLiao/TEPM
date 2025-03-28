# TEPM: Tensorial Elastoplastic Model

This repository contains a C++ implementation of a tensorial elastoplastic model for studying glass-forming liquids.

## Overview

The model simulates a 2D lattice of sites, each with local stress tensors and yielding planes. It implements a tensorial elastoplastic model with Monte Carlo dynamics and proper Eshelby kernel propagation.

## Features

- Tensorial stress representation (σxx, σxy)
- Proper Eshelby kernel for elastic propagation
- Monte Carlo dynamics with thermal activation
- Customizable system size and temperature
- Automatic caching of Eshelby kernels for efficiency
- Data output for analysis

## Requirements

- C++17 compatible compiler (GCC, Clang, etc.)
- Standard libraries (no external dependencies)

## Building

To compile the code:

```bash
make
```

For a debug build with optimization disabled:

```bash
make debug
```

For a profiling build:

```bash
make profile
```

## Usage

Run the simulation with:

```bash
./tensorial_model [options]
```

Options:
- `-L <size>`: System size (L×L lattice), default: 16
- `-T <temp>`: Temperature, default: 0.03
- `-steps <num>`: Number of Monte Carlo steps, default: 10000
- `-h, --help`: Show help message

Example:
```bash
./tensorial_model -L 32 -T 0.04 -steps 100000
```

## Output

The simulation produces two output files:
1. `config_L<size>_T<temp>.dat`: Full system configuration (positions, stresses, angles)
2. `stats_L<size>_T<temp>.dat`: Statistical information about the simulation

## Code Structure

- `tensorial_model.h/cpp`: Base class implementing the tensorial elastoplastic model
- `mc_tensorial_model.h/cpp`: Monte Carlo implementation inheriting from the base class
- `main.cpp`: Main program that runs the simulation

## Physics Background

The model simulates a 2D lattice where each site has:
- A stress tensor (σxx, σxy)
- A randomly oriented yielding plane angle θ
- A distance to yield that determines plastic event probability

Key elements include:
- Eshelby kernel for stress propagation after plastic events
- Metropolis Monte Carlo for thermal activation
- Force balance maintained throughout the simulation

## References

This implementation is inspired by:

1. Ozawa, M., Biroli, G. (2023). Elasticity, Facilitation, and Dynamic Heterogeneity in Glass-Forming Liquids. *Physical Review Letters*, 130, 138201. https://link.aps.org/doi/10.1103/PhysRevLett.130.138201

2. Tahaei, A., Biroli, G., Ozawa, M., Popovic, M., Wyart, M. (2023). Scaling Description of Dynamical Heterogeneity and Avalanches of Relaxation in Glass-Forming Liquids. *Physical Review X*, 13, 031034. https://doi.org/10.1103/PhysRevX.13.031034

## License

This code is provided under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.
