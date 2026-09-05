#ifndef TENSORIAL_MODEL_H
#define TENSORIAL_MODEL_H

#include <vector>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>

// Which form of the plastic stress drop to use.
//   Aligned: the drop is along the yield normal e = (sin th, -cos th) with the
//            single sign of sigma.e. This is the tensorial counterpart of the
//            scalar reset, the corrected rule, and the default.
//   Paper:   Eq. (A2)/(A3) of the PRX, verbatim
//            -- a separate sgn() per component and a leading minus.
//   PaperSigned: the paper's *direction* (sin th sgn s_xx, cos th sgn s_xy) but
//            with its projection oriented by the single sign of sigma.e. This
//            separates the two departures documented in readme.md: it repairs
//            the overall sign but keeps the component-wise direction.
enum class DropRule { Aligned, Paper, PaperSigned };

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
    std::vector<double> sin_theta; // Cached because only one angle changes per event
    std::vector<double> cos_theta;
    
    // Eshelby kernels in real space
    std::vector<std::vector<double>> G_xx_xx;  // G_xx,xx(r_i - r_j)
    std::vector<std::vector<double>> G_xy_xy;  // G_xy,xy(r_i - r_j)
    std::vector<std::vector<double>> G_xx_xy;  // G_xx,xy(r_i - r_j)
    
    // Dynamics state shared by every driver, so that persistence and
    // tau_alpha are measured identically no matter which one is running.
    double time;                    // continuous time (edmd) or sweeps (mc)
    long long int events;           // plastic events so far

    // Persistence: which sites have relaxed since the last origin
    std::vector<char> relaxed;
    long long int relaxed_count;
    double persistence_origin;

    DropRule drop_rule;

    // Diagnostic scale on the off-site kernel. applyStressDrop() never touches
    // G(r = 0), so scaling the stored arrays scales facilitation and nothing
    // else. 1.0 is the model; other values are a probe, not a physical choice.
    double kernel_amplitude;

    // Lattice convention for the mixed Fourier mode q_x q_y in Eq. (A6).
    // false: the paper's printed 2 sin(2pi nx/L) sin(2pi ny/L).
    // true:  4 sin(pi nx/L) sin(pi ny/L), the signed square root consistent
    //        with q_alpha^2 = 2 - 2 cos(2pi n_alpha/L). On even lattices its
    //        real transform has negative rank-two Nyquist modes; see readme.md.
    bool projector_kernel;

    // Random number generators
    std::uint32_t random_seed;
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::exponential_distribution<double> exp_dist;
    
    // Helper methods
    void calculateEshelbyKernels();
    bool loadEshelbyKernels(const std::string& filename);
    bool saveEshelbyKernels(const std::string& filename) const;

    // Stress drop of a plastic event, along the normal of the yielding plane.
    // See the sign discussion in readme.md: the paper's Eq. (A2)/(A3) apply a
    // separate sgn() to each component, which leaves the drop unaligned with
    // the yield normal and makes the dynamics diverge.
    void computeStressDrop(int site, double x, double z,
                           double& delta_sigma_xx, double& delta_sigma_xy) const;

    // Remove the macroscopic (q = 0) stress, which the model requires to vanish
    void projectOutMeanStress();

    // Book-keeping every driver must do when a site undergoes a plastic event
    void recordEvent(int site);
    
public:
    // Constructor
    TensorialModel(int size, double temperature);
    virtual ~TensorialModel() = default;
    
    // Initialization
    virtual void initialize();
    
    // Core functionality
    double calculateDistanceToYield(int site) const;
    void applyStressDrop(int site, double delta_sigma_xx, double delta_sigma_xy);

    // Relaxation rate of a site: 1/tau_0 when unstable (x <= 0), otherwise the
    // thermally activated rate exp(-E(x)/T)/tau_0 with E(x) = x^(3/2)
    double siteRate(int site);

    // One elementary move of whatever dynamics the subclass implements: a
    // plastic event for the Gillespie driver, a single attempt for Monte Carlo.
    virtual void advance() = 0;

    // Advance until the next plastic event, but no further than target_time.
    // Returns the site that relaxed, or -1 if no event occurred before the
    // target, in which case time is left exactly at target_time.
    //
    // This is the one primitive the paper-style analysis needs from a driver.
    // For the Gillespie driver an event is one step; for Monte Carlo it is the
    // next accepted attempt, with rejected attempts still advancing the clock.
    virtual int stepUntil(double target_time) = 0;

    // Run until the simulation time has advanced exactly by duration.
    virtual void runUntil(double duration);

    // Persistence measurement: reset the clock, then query the fraction of
    // sites that have not yet relaxed
    void resetPersistence();
    double persistence() const;
    double persistenceTime() const { return time - persistence_origin; }

    // Diagnostic single-origin half-persistence time. This is not the paper's
    // root of the physical-time-origin-averaged persistence curve. max_moves is
    // counted in whatever advance() does, so events for edmd, attempts for mc.
    double measureRelaxationTime(long long int max_moves);

    // Diagnostic successive half-times. For EDMD these origins follow events
    // and therefore sample the embedded jump chain, not uniform physical time.
    std::vector<double> measureRelaxationTimes(int reps, long long int max_moves);

    // Root-mean-square stress, used to detect a diverging run
    double stressRms() const;

    // Draw the reset-depth variate z from p(z) = exp(-z/z0)/z0
    double drawResidualStress() { return exp_dist(rng); }
    
    // Replace the randomly drawn initial state with one read back from a file
    // written by saveConfiguration(), so a configuration equilibrated at one
    // temperature can be used as the starting point of another run.
    //
    // Call this *after* initialize(). initialize() is left untouched: it still
    // builds the kernel and draws its random state, and that draw is not
    // suppressed, so the RNG stream sits at the same position it would for any
    // other run with the same seed. Only the state it produced is overwritten.
    //
    // saveConfiguration() writes six significant digits, so the state restored
    // here is not bit-identical to the one saved. Returns false, leaving the
    // state untouched, if the file cannot be read or does not describe exactly
    // this lattice.
    bool loadConfiguration(const std::string& filename);

    // Persistence correlation function <pi(t)> and the four-point correlation
    // function chi_4(t) = N(<pi^2(t)> - <pi(t)>^2), averaged over physical-time
    // origins. Origins lie on a regular physical-time grid and event times are
    // recorded once for all overlapping observation windows, avoiding both
    // event-chain sampling bias and redundant simulation.
    //
    // Driver-independent: it needs only stepUntil() and runUntil(), so both the
    // Gillespie and Monte Carlo drivers get the paper's estimator from the same
    // code rather than from two implementations that could drift apart.
    void measureDynamics(int num_origins, const std::vector<double>& times,
                         double origin_spacing,
                         std::vector<double>& pi_mean, std::vector<double>& chi4,
                         std::vector<std::vector<double>>* pi_by_origin = nullptr);

    // Distribution P(x) sampled at regular physical-time intervals.
    void measurePx(int num_samples, double sample_interval,
                   std::vector<double>& px);

    // Measurement and analysis
    // Return false if the requested scientific output cannot be written.
    bool saveConfiguration(const std::string& filename);
    bool saveStatistics(const std::string& filename);
    virtual void measureCorrelations();
    
    // Getters and setters
    // Must be called before initialize(), which builds or loads the kernel
    void setProjectorKernel(bool on) { projector_kernel = on; }
    void setDropRule(DropRule rule) { drop_rule = rule; }
    // Must be called before initialize(), which applies it to the kernel
    void setKernelAmplitude(double a) { kernel_amplitude = a; }
    void setRandomSeed(std::uint32_t seed) {
        random_seed = seed;
        rng.seed(seed);
        uniform_dist.reset();
        exp_dist.reset();
    }
    std::uint32_t getRandomSeed() const { return random_seed; }
    double getTime() const { return time; }
    long long int getEvents() const { return events; }
    bool usesProjectorKernel() const { return projector_kernel; }
    double getTemperature() const { return T; }
    void setTemperature(double temperature) { T = temperature; }
    int getSystemSize() const { return L; }
};

#endif // TENSORIAL_MODEL_H
