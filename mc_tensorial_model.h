#ifndef MC_TENSORIAL_MODEL_H
#define MC_TENSORIAL_MODEL_H

#include "tensorial_model.h"

// Monte Carlo implementation of the tensorial model
class MCTensorialModel : public TensorialModel {
private:
    // Simulation statistics
    long long int MC_steps;
    long long int accepted_steps;
    
public:
    // Constructor
    MCTensorialModel(int size, double temperature);
    
    // Monte Carlo specific methods
    void runMCStep();
    void runSimulation(long long int num_steps);
    
    // Getters for MC statistics
    long long int getMCSteps() const { return MC_steps; }
    long long int getAcceptedSteps() const { return accepted_steps; }
    double getAcceptanceRate() const { 
        return MC_steps > 0 ? static_cast<double>(accepted_steps) / MC_steps : 0.0; 
    }
    
    // Override base class methods if needed
    void saveStatistics(const std::string& filename);
};

#endif // MC_TENSORIAL_MODEL_H