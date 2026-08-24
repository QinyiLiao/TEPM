#ifndef EDMD_TENSORIAL_MODEL_H
#define EDMD_TENSORIAL_MODEL_H

#include "tensorial_model.h"

// Gillespie ("EDMD-like") implementation of the tensorial model, which is the
// algorithm used in Tahaei et al., PRX 13, 031034 (2023), Appendix A.
//
// Every step relaxes exactly one site, chosen with probability r_i / sum(r),
// and advances continuous time by an exponential deviate of mean 1/sum(r).
// Unstable sites (x <= 0) carry the maximal rate 1/tau_0. Stable sites retain
// their thermal rates and therefore still compete with unstable sites, exactly
// as in the paper's continuous-time Markov process.
class EDMDTensorialModel : public TensorialModel {
private:
    std::vector<double> cumulative;  // cumulative rates, for site selection

    // Site selection. Literal is the corrected independent-clock construction:
    // draw a waiting time for every site and take the earliest. Rate is the
    // equivalent, cheaper construction based on the total rate. A third
    // construction sometimes seen -- rate-weighted site choice followed by an
    // unconditioned draw from that site's own exponential -- is not equivalent
    // and is not implemented here; see readme.md.
    bool literal_selection;

    // Pick the site that relaxes next and the time until it does
    int selectSiteByRate(double& dt);
    int selectSiteLiteral(double& dt);

    // Apply an already selected event, without changing the clock.
    void executeEvent(int site);

    // Execute the next event only if it occurs at or before target_time.
    // Returns its site, or -1 after advancing exactly to target_time without
    // an event. Discarding a crossing exponential draw is exact by memorylessness.
    int stepUntil(double target_time) override;

public:
    EDMDTensorialModel(int size, double temperature);

    void setLiteralSelection(bool on) { literal_selection = on; }

    // Advance by a single plastic event; returns the site that relaxed
    int step();
    void advance() override { step(); }

    // Run a fixed number of plastic events; false means the divergence guard fired.
    bool runEvents(long long int num_events);

    // measureDynamics(), measurePx() and runUntil() now live on TensorialModel:
    // they need only stepUntil(), so both drivers share one implementation.

    void saveStatistics(const std::string& filename);
};

#endif // EDMD_TENSORIAL_MODEL_H
