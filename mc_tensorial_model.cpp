#include "mc_tensorial_model.h"

// Constructor
MCTensorialModel::MCTensorialModel(int size, double temperature)
    : TensorialModel(size, temperature),
      MC_steps(0),
      accepted_steps(0) {
}

// Run one Monte Carlo step
void MCTensorialModel::runMCStep() {
    int N = L * L;
    
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
        double z = exp_dist(rng);
        
        // Calculate stress drop components (Eq. 26)
        double delta_sigma_xx = (z - x) * sin(theta[site]) * (sigma_xx[site] > 0 ? 1.0 : -1.0);
        double delta_sigma_xy = (z - x) * cos(theta[site]) * (sigma_xy[site] > 0 ? 1.0 : -1.0);
        
        // Apply stress drop and propagate
        applyStressDrop(site, delta_sigma_xx, delta_sigma_xy);
        
        accepted_steps++;
    }
    
    MC_steps++;
}

// Run simulation for specified number of steps
void MCTensorialModel::runSimulation(long long int num_steps) {
    std::cout << "Starting MC simulation with L = " << L << ", T = " << T << std::endl;
    
    long long int print_interval = num_steps / 10;
    if (print_interval < 1) print_interval = 1;
    
    for (long long int step = 0; step < num_steps; step++) {
        runMCStep();
        
        if (step % print_interval == 0) {
            double acceptance_rate = static_cast<double>(accepted_steps) / MC_steps;
            std::cout << "Step " << step << "/" << num_steps 
                     << " (Acceptance rate: " << acceptance_rate << ")" << std::endl;
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