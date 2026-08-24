# TEPM: Tensorial Elastoplastic Model

This repository contains a C++ implementation of a tensorial elastoplastic model for studying glass-forming liquids.

## Overview

The model simulates a 2D lattice of sites, each with local stress tensors and yielding planes. Sites relax by plastic events, and the resulting stress redistribution is propagated with the Eshelby kernel. It implements the tensorial model of Ref. 2, with three interchangeable dynamics: continuous-time Gillespie, Monte Carlo, and extremal dynamics at vanishing temperature.

## Features

- Tensorial stress representation (σxx, σxy) with a randomly oriented yielding plane per site
- Eshelby kernel for elastic propagation, cached to disk and reused between runs
- Three drivers: Gillespie (`edmd`), Monte Carlo (`mc`), and extremal dynamics (`extremal`) at T = 0⁺
- Persistence correlation function ⟨π(t)⟩ and four-point correlation function χ₄(t), averaged over physical-time origins
- Avalanche statistics and energy gaps for the T = 0⁺ critical point
- Selectable stress-drop and kernel conventions, so the published equations and their corrected forms can both be run — see *Implementation notes*
- Zero macroscopic stress enforced throughout
- Configurations can be saved and reloaded, so a state equilibrated under one set of conditions can be used as the initial state of another run

## Requirements

- A C++17 compiler
- `make`

No external libraries.

## Building

```bash
make            # release build, -O3
make debug      # debug build
make clean
```

## Usage

```bash
./tensorial_model [options]
```

Principal options:

- `-L <size>`: system size (L×L lattice)
- `-T <temp>`: temperature
- `-algo <name>`: `edmd` (Gillespie, default), `mc`, or `extremal` (T = 0⁺)
- `-steps <num>`: equilibration events for `edmd`, attempted moves for `mc`, measurement steps for `extremal`
- `-seed <uint32>`: RNG seed; generated and printed if omitted
- `-tag <name>`: suffix appended to output filenames

Measurement:

- `-analyze <n>`: average ⟨π(t)⟩ and χ₄(t) over `n` physical-time origins, and solve ⟨π(τ_α)⟩ = 1/2. Available for both `edmd` and `mc`
- `-pxsamples <n>`: samples of P(x) on a physical-time grid after `-analyze`; 0 skips
- `-tau`, `-taureps <n>`: single-origin half-persistence times. These follow events rather than uniform physical time, so they are diagnostics rather than the relaxation time above

Extremal dynamics:

- `-x0 <list>`: comma-separated avalanche thresholds
- `-transient <n>`: steps discarded before measuring
- `-stablex0 <x>`: sample stable-state P(x) and energy gaps at threshold `x`
- `-avalblocks <n>`: number of completion-time moment blocks

Conventions — see *Implementation notes*:

- `-drop <name>`: `aligned` (default), `paper`, or `papersgn`
- `-kernel <name>`: `paper` (default) or `proj`
- `-select <name>`: `rate` (default) or `literal`
- `-kamp <factor>`: scale the off-site kernel; a diagnostic, not a physical parameter

Other:

- `-loadconfig <file>`: start from a saved configuration instead of a random state. The file must describe the same lattice size; its own temperature and drop rule are not applied, so set those explicitly
- `-h`, `--help`: full option list

Example:

```bash
./tensorial_model -L 32 -T 0.04 -analyze 500 -seed 12345
```

Sizes, temperatures and run lengths are left to the user; nothing here assumes particular values.

## Output

Filenames carry the lattice size, temperature and any `-tag` suffix. Headers record the parameters each file was produced with.

- `config_*.dat`: full configuration — stresses and yielding-plane angles
- `stats_*.dat`: run parameters and summary statistics
- `dynamics_*.dat`: `t`, ⟨π(t)⟩, χ₄(t), written by `-analyze`
- `dynamics_origins_*.dat`: the per-origin persistence curves behind them, for block error estimates
- `px_*.dat`: P(x), the distribution of the distance to yield
- `aval_*.dat`, `avalblocks_*.dat`, `xdist_*.dat`, `egap_*.dat`: extremal-dynamics avalanche sizes, moment blocks, distributions and energy gaps
- `eshelby_kernel_L<size>*.dat`: cached kernel, reused automatically

## Code Structure

- `tensorial_model.h/cpp`: base class — lattice, Eshelby kernel, stress drops, persistence, the physical-time estimators, configuration I/O
- `edmd_tensorial_model.h/cpp`: Gillespie driver. Continuous time, one plastic event per step, site chosen with probability r_i/Σr
- `mc_tensorial_model.h/cpp`: Monte Carlo driver. Uniform site choice, acceptance exp(−E/T), time counted in sweeps including rejected attempts
- `extremal_tensorial_model.h/cpp`: extremal dynamics at T = 0⁺ — relax the weakest site, one unit of time per event
- `main.cpp`: command line, run orchestration, output

## Physics Background

Each site carries a stress tensor (σxx, σxy) and a randomly oriented yielding plane θ. The distance to yield is

    x = 1 − |σ · e|,    e = (sin θ, −cos θ)

and the local activation barrier is E(x) = x^(3/2). A site relaxes at rate exp(−E/T), or immediately when x ≤ 0. On relaxation the stress at that site drops, the yielding plane is reoriented at random, and the change is propagated to every other site through the Eshelby kernel. The macroscopic stress is projected out, as the model requires.

At vanishing temperature the dynamics becomes extremal: the site with the smallest barrier always relaxes next. This limit has a critical point, and the avalanche statistics and energy gaps produced by `-algo extremal` are the observables that characterise it.

## Implementation notes

Some equations as printed require care. This code makes each choice explicit rather than silently picking one, and exposes the alternatives as flags so they can be run and compared directly.

**Stress drop (`-drop`).** Eq. (A2)/(A3) of Ref. 2 applies a separate `sgn()` to each stress component. That leaves the drop not aligned with the yield normal, and its overall sign is not guaranteed to reduce |σ·e|. Run literally (`-drop paper`) the stress field diverges — easily reproduced, and the reason the mode is kept. Two repaired forms are provided: `papersgn` keeps the published direction but forces the overall sign so that an event always lowers |σ·e|, and `aligned`, the default, takes the drop along the yield normal with the single sign of σ·e, so the distance to yield after the event is exactly the drawn residual.

**Kernel convention (`-kernel`).** The mixed Fourier mode q_x q_y in Eq. (A6) can be written as printed, `2 sin(2π n_x/L) sin(2π n_y/L)`, which is the default; or as `4 sin(π n_x/L) sin(π n_y/L)`, the signed square root consistent with q_α² = 2 − 2cos(2π n_α/L). On even lattices the latter gives a real transform with negative rank-two Nyquist modes.

**Clock construction (`-select`).** The Gillespie step can be built either by drawing a waiting time for every site and taking the earliest (`literal`), or equivalently and more cheaply from the total rate (`rate`, the default). These agree. A third construction sometimes seen — choose the site by rate, then draw the waiting time from that site's own exponential without conditioning — is *not* equivalent: it overestimates the time per event by a factor equal to the number of sites. It is not implemented here.

**Driver agreement.** The Monte Carlo and Gillespie drivers integrate the same master equation by different means, and have been checked against each other on identical kernels: the relaxation time and the four-point function agree. Monte Carlo is a rejection sampler, so its cost per plastic event grows as the acceptance rate exp(−E/T) falls, and at low temperature the Gillespie driver is the practical choice.

## References

1. Ozawa, M., Biroli, G. (2023). Elasticity, Facilitation, and Dynamic Heterogeneity in Glass-Forming Liquids. *Physical Review Letters*, 130, 138201. https://link.aps.org/doi/10.1103/PhysRevLett.130.138201

2. Tahaei, A., Biroli, G., Ozawa, M., Popovic, M., Wyart, M. (2023). Scaling Description of Dynamical Heterogeneity and Avalanches of Relaxation in Glass-Forming Liquids. *Physical Review X*, 13, 031034. https://doi.org/10.1103/PhysRevX.13.031034

## License

See `LICENSE`.

## Contributing

Issues and pull requests are welcome.
