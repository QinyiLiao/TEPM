#ifndef TENSORIAL_MODEL_H
#define TENSORIAL_MODEL_H

#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>

// Base class for the tensorial elastoplastic model
class TensorialModel {
protected:
    // System parameters
    int L;                      // System size (L x L lattice)
    double T;                   // Temperature
    double sigma_Y;             // Yield stress (set to 1.0)
    double mu;                  // Shear modulus (set to 1.0)
    double K;                   // Energy barrier constant (set to 1.0)
    double z0;                  // Stress drop parameter (set to 1.0)
    
    // System state
    std::vector<double> sigma_xx;  // xx component of stress tensor
    std::vector<double> sigma_xy;  // xy component of stress tensor
    std::vector<double> theta;     // Angles for yielding planes
    
    // Eshelby kernels in real space
    std::vector<std::vector<double>> G_xx_xx;  // G_xx,xx(r_i - r_j)
    std::vector<std::vector<double>> G_xy_xy;  // G_xy,xy(r_i - r_j)
    std::vector<std::vector<double>> G_xx_xy;  // G_xx,xy(r_i - r_j)
    
    // Random number generators
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::exponential_distribution<double> exp_dist;
    
    // Helper methods
    void calculateEshelbyKernels();
    bool loadEshelbyKernels(const std::string& filename);
    void saveEshelbyKernels(const std::string& filename) const;
    
public:
    // Constructor
    TensorialModel(int size, double temperature);
    virtual ~TensorialModel() = default;
    
    // Initialization
    virtual void initialize();
    
    // Core functionality
    double calculateDistanceToYield(int site);
    void applyStressDrop(int site, double delta_sigma_xx, double delta_sigma_xy);
    
    // Measurement and analysis
    void saveConfiguration(const std::string& filename);
    void saveStatistics(const std::string& filename);
    virtual void measureCorrelations();
    
    // Getters and setters
    double getTemperature() const { return T; }
    void setTemperature(double temperature) { T = temperature; }
    int getSystemSize() const { return L; }
};

#endif // TENSORIAL_MODEL_H
