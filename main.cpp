#include "tensorial_model.h"
#include "mc_tensorial_model.h"
#include <iostream>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
    // Default parameters
    int L = 16;               // System size
    double T = 0.03;           // Temperature
    long long int steps = 10000;  // Number of MC steps

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-L" && i + 1 < argc) {
            L = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        } else if (arg == "-steps" && i + 1 < argc) {
            steps = std::stoll(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -L <size>      System size (default: 32)\n"
                      << "  -T <temp>      Temperature (default: 0.02)\n"
                      << "  -steps <num>   Number of MC steps (default: 10000)\n"
                      << "  -h, --help     Show this help message\n";
            return 0;
        }
    }

    std::cout << "Starting tensorial elastoplastic model simulation\n";
    std::cout << "System size: " << L << "x" << L << "\n";
    std::cout << "Temperature: " << T << "\n";
    std::cout << "MC steps: " << steps << "\n";

    // Create and initialize the model
    auto start_time = std::chrono::high_resolution_clock::now();

    MCTensorialModel model(L, T);
    std::cout << "Initializing system..." << std::endl;
    model.initialize();

    // Run simulation
    std::cout << "Running simulation..." << std::endl;
    model.runSimulation(steps);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "Simulation completed in " << duration << " seconds." << std::endl;

    // Save results
    std::string config_file = "config_L" + std::to_string(L) + "_T" + std::to_string(T) + ".dat";
    std::string stats_file = "stats_L" + std::to_string(L) + "_T" + std::to_string(T) + ".dat";

    model.saveConfiguration(config_file);
    model.saveStatistics(stats_file);

    return 0;
}
