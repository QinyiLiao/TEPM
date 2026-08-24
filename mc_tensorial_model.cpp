#include "mc_tensorial_model.h"

// Constructor
MCTensorialModel::MCTensorialModel(int size, double temperature)
    : TensorialModel(size, temperature),
      MC_steps(0),
      accepted_steps(0) {
}

// One Monte Carlo attempt
int MCTensorialModel::runMCStep() {
    int N = L * L;

    // Time is measured in sweeps, rejected attempts included
    time += 1.0 / N;
    
    // Choose a site randomly
    int site = static_cast<int>(N * uniform_dist(rng));
    
    // Calculate distance to yield
    double x = calculateDistanceToYield(site);
    
    // Check if plastic event occurs
    bool plastic_event = false;
    
    if (x <= 0.0) {
        // System has yielded
        plastic_event = true;
    } else {
        // Metropolis criterion
        double energy = std::pow(x, 1.5);
        double p_accept = std::exp(-energy / T);
        
        if (uniform_dist(rng) < p_accept) {
            plastic_event = true;
        }
    }
    
    // If plastic event occurs, calculate stress drop and propagate
    if (plastic_event) {
        // Draw random stress drop from exponential distribution
        double z = drawResidualStress();

        // Calculate stress drop components (Eq. 26)
        double delta_sigma_xx, delta_sigma_xy;
        computeStressDrop(site, x, z, delta_sigma_xx, delta_sigma_xy);

        // Apply stress drop and propagate
        applyStressDrop(site, delta_sigma_xx, delta_sigma_xy);

        recordEvent(site);
        accepted_steps++;
    }

    MC_steps++;
    return plastic_event ? site : -1;
}

// Attempt moves until one is accepted or the clock would pass target_time.
//
// The guard is on the attempt that is about to be made, not the one just made:
// starting an attempt that would carry the clock past the target would sample
// dynamics beyond the requested window. When no attempt fits, the clock is set
// to the target exactly, matching the Gillespie driver, so origins land on the
// intended physical-time grid in both drivers.
int MCTensorialModel::stepUntil(double target_time) {
    const double dt = 1.0 / (L * L);
    while (time + dt <= target_time) {
        int site = runMCStep();
        if (site >= 0) return site;
    }
    time = target_time;
    return -1;
}

// Run simulation for specified number of steps
//
// Progress is reported on a logarithmic schedule rather than ten linear points.
// Relaxation from a random start is a decaying transient, so linear reporting
// spends nine of its ten lines on the equilibrated tail and none on the part
// that is actually moving. Alongside the cumulative acceptance -- which is an
// average over history and lags badly -- each line carries the acceptance over
// the window since the previous line and the stress rms, which together show
// whether the run has actually reached a steady state. Output only: none of
// this touches the RNG stream or any saved quantity.
void MCTensorialModel::runSimulation(long long int num_steps) {
    std::cout << "Starting MC simulation with L = " << L << ", T = " << T << std::endl;

    const int num_reports = 40;
    std::vector<long long int> schedule;
    for (int r = 0; r <= num_reports; r++) {
        double f = static_cast<double>(r) / num_reports;
        long long int s = static_cast<long long int>(
            std::pow(static_cast<double>(num_steps), f));
        if (schedule.empty() || s > schedule.back()) schedule.push_back(s);
    }
    if (schedule.back() < num_steps) schedule.push_back(num_steps);

    size_t next_report = 0;
    long long int last_step = 0, last_accepted = 0;

    std::cout << "# step  cum_accept  window_accept  stress_rms" << std::endl;
    for (long long int step = 0; step < num_steps; step++) {
        runMCStep();

        if (next_report < schedule.size() && step + 1 >= schedule[next_report]) {
            const long long int dstep = MC_steps - last_step;
            const long long int dacc = accepted_steps - last_accepted;
            std::cout << "Step " << MC_steps << "/" << num_steps
                      << " (Acceptance rate: "
                      << static_cast<double>(accepted_steps) / MC_steps
                      << ", window: "
                      << (dstep > 0 ? static_cast<double>(dacc) / dstep : 0.0)
                      << ", events/site: "
                      << static_cast<double>(accepted_steps) / (L * L)
                      << ", rms: " << stressRms() << ")" << std::endl;
            last_step = MC_steps;
            last_accepted = accepted_steps;
            while (next_report < schedule.size()
                   && schedule[next_report] <= step + 1) next_report++;
        }
    }

    std::cout << "Simulation completed. Total MC steps: " << MC_steps << std::endl;
    std::cout << "Accepted steps: " << accepted_steps 
              << " (Rate: " << static_cast<double>(accepted_steps) / MC_steps << ")" << std::endl;
}

// Override saveStatistics to include MC-specific information
void MCTensorialModel::saveStatistics(const std::string& filename) {
    // First call the base class implementation
    TensorialModel::saveStatistics(filename);
    
    // Then append MC-specific statistics
    std::ofstream file(filename, std::ios_base::app);  // Open in append mode
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for appending." << std::endl;
        return;
    }
    
    file << "# MC steps: " << MC_steps << "\n";
    file << "# Accepted steps: " << accepted_steps << "\n";
    file << "# Acceptance rate: " << getAcceptanceRate() << "\n";
    
    file.close();
    std::cout << "MC statistics appended to " << filename << std::endl;
}