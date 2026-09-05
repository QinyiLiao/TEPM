#ifndef MC_TENSORIAL_MODEL_H
#define MC_TENSORIAL_MODEL_H

#include "tensorial_model.h"

// Monte Carlo implementation: pick a site uniformly, relax it outright if it
// is already unstable and otherwise with probability exp(-E(x)/T), and count
// time in sweeps -- attempts divided by N, rejected attempts included.
//
// This is the same dynamics as the Gillespie driver, sampled by rejection, so
// it is useful as an independent check of that driver on identical Eshelby
// kernels. The optional exact skipping mode draws the geometrically distributed
// number of attempts to the next acceptance and the accepted site with weight
// r_i. It changes RNG consumption, but not the MC transition probabilities,
// attempt count, or discrete-time clock.
class MCTensorialModel : public TensorialModel {
private:
    long long int MC_steps;        // attempts, accepted or not
    long long int accepted_steps;
    // Remaining sweep time to the next attempt. Observation calls can stop
    // between attempts; retaining the unused fraction makes splitting a time
    // interval into multiple runUntil() calls leave the trajectory unchanged.
    double time_to_next_attempt;
    bool rejection_skipping;

    // A skipped accepted attempt is sampled once and retained across
    // observation boundaries. attempts_to_event includes the accepted attempt.
    bool pending_schedule;
    long long int attempts_to_event;
    int pending_site;
    std::vector<double> cumulative;

    void executeEvent(int site);
    int runUniformAttempt();
    void scheduleAcceptedAttempt();
    int runSkippedToAttempt(long long int target_steps);
    int stepUntilSkipped(double target_time);
    bool runSimulationSkipped(long long int num_steps);

public:
    MCTensorialModel(int size, double temperature);

    // One attempt. Normally advances time by 1/N sweeps; after an observation
    // boundary it first consumes the retained fractional interval.
    // Returns the site that relaxed, or -1 if the attempt was rejected.
    int runMCStep();
    void advance() override { runMCStep(); }

    // Attempt moves until one is accepted, but stop rather than begin an
    // attempt that would carry the clock past target_time. Rejected attempts
    // advance the clock exactly as they do in a normal run, so the time this
    // reports is the same sweep count the dynamics would have reached anyway.
    int stepUntil(double target_time) override;

    // False means the stress divergence guard fired.
    bool runSimulation(long long int num_steps);

    // These shadow the base diagnostics so their max_moves budget continues to
    // count attempts while exact skipping avoids iterating over every reject.
    double measureRelaxationTime(long long int max_moves);
    std::vector<double> measureRelaxationTimes(
        int reps, long long int max_moves);

    // Call before dynamics begin. The default remains literal uniform
    // rejection so historical fixed-seed trajectories remain reproducible.
    void setRejectionSkipping(bool on) {
        if (rejection_skipping != on) {
            rejection_skipping = on;
            pending_schedule = false;
        }
    }
    bool usesRejectionSkipping() const { return rejection_skipping; }

    long long int getMCSteps() const { return MC_steps; }
    long long int getAcceptedSteps() const { return accepted_steps; }
    double getAcceptanceRate() const {
        return MC_steps > 0 ? static_cast<double>(accepted_steps) / MC_steps : 0.0;
    }

    bool saveStatistics(const std::string& filename);
};

#endif // MC_TENSORIAL_MODEL_H
