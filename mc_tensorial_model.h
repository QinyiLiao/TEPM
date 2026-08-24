#ifndef MC_TENSORIAL_MODEL_H
#define MC_TENSORIAL_MODEL_H

#include "tensorial_model.h"

// Monte Carlo implementation: pick a site uniformly, relax it outright if it
// is already unstable and otherwise with probability exp(-E(x)/T), and count
// time in sweeps -- attempts divided by N, rejected attempts included.
//
// This is the same dynamics as the Gillespie driver, sampled by rejection, so
// it is useful as an independent check of that driver on identical Eshelby
// kernels. It is only usable at the warm end of the range: the acceptance rate
// falls like exp(-E/T), so below T ~ 0.03 essentially every attempt is wasted.
class MCTensorialModel : public TensorialModel {
private:
    long long int MC_steps;        // attempts, accepted or not
    long long int accepted_steps;

public:
    MCTensorialModel(int size, double temperature);

    // One attempt. Advances time by 1/N sweeps whether or not it is accepted.
    // Returns the site that relaxed, or -1 if the attempt was rejected.
    int runMCStep();
    void advance() override { runMCStep(); }

    // Attempt moves until one is accepted, but stop rather than begin an
    // attempt that would carry the clock past target_time. Rejected attempts
    // advance the clock exactly as they do in a normal run, so the time this
    // reports is the same sweep count the dynamics would have reached anyway.
    int stepUntil(double target_time) override;

    void runSimulation(long long int num_steps);

    long long int getMCSteps() const { return MC_steps; }
    long long int getAcceptedSteps() const { return accepted_steps; }
    double getAcceptanceRate() const {
        return MC_steps > 0 ? static_cast<double>(accepted_steps) / MC_steps : 0.0;
    }

    void saveStatistics(const std::string& filename);
};

#endif // MC_TENSORIAL_MODEL_H
