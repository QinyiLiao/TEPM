#include "edmd_tensorial_model.h"
#include <algorithm>
#include <limits>

// Constructor
EDMDTensorialModel::EDMDTensorialModel(int size, double temperature)
    : TensorialModel(size, temperature),
      literal_selection(false) {

    cumulative.resize(size * size, 0.0);
}

// Site selection by cumulative rate: the site relaxes with probability
// r_i / sum(r), and the waiting time is exponential with mean 1 / sum(r).
// This is the standard Gillespie construction and the process used in the paper.
int EDMDTensorialModel::selectSiteByRate(double& dt) {
    int N = L * L;

    double total_rate = 0.0;
    for (int i = 0; i < N; i++) {
        total_rate += siteRate(i);
        cumulative[i] = total_rate;
    }

    double u = uniform_dist(rng);
    if (u <= 0.0) u = 1e-300;
    dt = -std::log(u) / total_rate;

    double target = uniform_dist(rng) * total_rate;
    int site = static_cast<int>(
        std::lower_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin());
    if (site >= N) site = N - 1;

    return site;
}

// Correct independent-clock construction: give every site an exponential
// waiting time of mean 1/r_i and take the earliest. This is statistically
// identical to selectSiteByRate but uses N times as many random numbers, so it
// exists as a diagnostic rather than for quantitative production runs.
//
// A cheaper-looking replacement for this step -- keep the rate-weighted choice
// of site, then advance the clock by an unconditioned draw from that site's own
// exponential -- is not equivalent: it overestimates the time per event by
// exactly N. Neither branch here does that.
int EDMDTensorialModel::selectSiteLiteral(double& dt) {
    int N = L * L;

    double earliest = std::numeric_limits<double>::infinity();
    int site = 0;

    for (int i = 0; i < N; i++) {
        double rate = siteRate(i);
        if (rate <= 0.0) continue;

        double u = uniform_dist(rng);
        if (u <= 0.0) u = 1e-300;
        double t = -std::log(u) / rate;

        if (t < earliest) {
            earliest = t;
            site = i;
        }
    }

    dt = earliest;
    return site;
}

void EDMDTensorialModel::executeEvent(int site) {
    // Plastic event: drop the stress and propagate it through the kernel
    double x = calculateDistanceToYield(site);
    double z = drawResidualStress();

    double delta_sigma_xx, delta_sigma_xy;
    computeStressDrop(site, x, z, delta_sigma_xx, delta_sigma_xy);
    applyStressDrop(site, delta_sigma_xx, delta_sigma_xy);

    recordEvent(site);
}

// One Gillespie step
int EDMDTensorialModel::step() {
    double dt = 0.0;
    int site = literal_selection ? selectSiteLiteral(dt) : selectSiteByRate(dt);
    time += dt;
    executeEvent(site);

    return site;
}

int EDMDTensorialModel::stepUntil(double target_time) {
    if (target_time <= time) return -1;

    double dt = 0.0;
    int site = literal_selection ? selectSiteLiteral(dt) : selectSiteByRate(dt);
    if (dt > target_time - time) {
        // No event occurs before the requested observation time. Exponential
        // waiting times are memoryless, so the unused event can be discarded.
        time = target_time;
        return -1;
    }

    time += dt;
    executeEvent(site);
    return site;
}

// Run a fixed number of plastic events
bool EDMDTensorialModel::runEvents(long long int num_events) {
    long long int report_interval = num_events / 10;
    if (report_interval < 1) report_interval = 1;
    const long long int guard_interval = std::min(report_interval, 1000LL);

    for (long long int e = 0; e < num_events; e++) {
        step();

        const bool report = (e % report_interval == 0);
        const bool guard = (e % guard_interval == 0);
        if (report || guard) {
            double rms = stressRms();
            if (report) {
                std::cout << "Event " << e << "/" << num_events
                          << " (t = " << time << ", rms = " << rms << ")" << std::endl;
            }

            // The paper's stress-drop rule can send the stress field off to
            // infinity. Stop rather than grind through the whole budget.
            if (guard && (!std::isfinite(rms) || rms > 1e3)) {
                std::cout << "Stress field is diverging (rms = " << rms
                          << "); stopping." << std::endl;
                return false;
            }
        }
    }

    return true;
}

// Statistics, with the EDMD-specific quantities appended
void EDMDTensorialModel::saveStatistics(const std::string& filename) {
    TensorialModel::saveStatistics(filename);

    std::ofstream file(filename, std::ios_base::app);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for appending." << std::endl;
        return;
    }

    file << "# Algorithm: edmd\n";
    file << "# EDMD selection: " << (literal_selection ? "literal" : "rate") << "\n";
    file << "# Plastic events: " << events << "\n";
    file << "# Simulation time: " << time << "\n";

    file.close();
    std::cout << "EDMD statistics appended to " << filename << std::endl;
}
