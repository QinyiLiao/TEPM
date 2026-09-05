#include "mc_tensorial_model.h"
#include <algorithm>
#include <limits>

// Constructor
MCTensorialModel::MCTensorialModel(int size, double temperature)
    : TensorialModel(size, temperature),
      MC_steps(0),
      accepted_steps(0),
      time_to_next_attempt(1.0 / (size * size)),
      rejection_skipping(false),
      pending_schedule(false),
      attempts_to_event(0),
      pending_site(-1),
      cumulative(size * size, 0.0) {
}

void MCTensorialModel::executeEvent(int site) {
    const double x = calculateDistanceToYield(site);
    const double z = drawResidualStress();

    double delta_sigma_xx, delta_sigma_xy;
    computeStressDrop(site, x, z, delta_sigma_xx, delta_sigma_xy);
    applyStressDrop(site, delta_sigma_xx, delta_sigma_xy);

    recordEvent(site);
    accepted_steps++;
}

// One literal uniform-rejection Monte Carlo attempt.
int MCTensorialModel::runUniformAttempt() {
    const int N = L * L;

    // Time is measured in sweeps, rejected attempts included. A preceding
    // observation may have stopped partway to this attempt.
    time += time_to_next_attempt;
    time_to_next_attempt = 1.0 / N;

    const int site = static_cast<int>(N * uniform_dist(rng));
    const double x = calculateDistanceToYield(site);

    bool plastic_event = false;
    if (x <= 0.0) {
        plastic_event = true;
    } else {
        // Keep the historical literal sampler byte-reproducible.
        const double p_accept = std::exp(-std::pow(x, 1.5) / T);
        if (uniform_dist(rng) < p_accept) {
            plastic_event = true;
        }
    }

    if (plastic_event) executeEvent(site);

    MC_steps++;
    return plastic_event ? site : -1;
}

// Draw the next accepted MC attempt without materializing its rejected
// predecessors. In a fixed state, one attempt succeeds with
//
//   p = (1/N) sum_i r_i.
//
// The attempt index is therefore geometric(p), and conditional on success the
// site has probability r_i/sum(r). This is exactly the uniform rejection
// sampler in distribution; unlike Gillespie, its waiting time remains discrete.
void MCTensorialModel::scheduleAcceptedAttempt() {
    const int N = L * L;
    double total_rate = 0.0;
    for (int site = 0; site < N; site++) {
        const double x = calculateDistanceToYield(site);
        const double rate = (x <= 0.0)
                          ? 1.0
                          : std::exp(-std::pow(x, 1.5) / T);
        total_rate += rate;
        cumulative[site] = total_rate;
    }

    pending_schedule = true;
    if (total_rate <= 0.0) {
        attempts_to_event = std::numeric_limits<long long int>::max();
        pending_site = -1;
        return;
    }

    const double probability =
        std::min(1.0, total_rate / static_cast<double>(N));
    if (probability >= 1.0) {
        attempts_to_event = 1;
    } else {
        double u = uniform_dist(rng);
        if (u <= 0.0) u = std::numeric_limits<double>::min();
        const long double failures = std::floor(
            std::log(static_cast<long double>(u))
            / std::log1p(-static_cast<long double>(probability)));
        const long double maximum_failures =
            static_cast<long double>(
                std::numeric_limits<long long int>::max() - 1);
        if (failures > maximum_failures) {
            // No event can be represented before the signed attempt counter
            // reaches its hard limit. Do not manufacture one at LLONG_MAX.
            attempts_to_event = std::numeric_limits<long long int>::max();
            pending_site = -1;
            return;
        }
        attempts_to_event = static_cast<long long int>(failures) + 1;
    }

    double target = uniform_dist(rng) * total_rate;
    if (target >= total_rate) {
        target = std::nextafter(total_rate, 0.0);
    }
    pending_site = static_cast<int>(
        std::upper_bound(cumulative.begin(), cumulative.end(), target)
        - cumulative.begin());
    if (pending_site >= N) {
        pending_site = static_cast<int>(
            std::lower_bound(cumulative.begin(), cumulative.end(), total_rate)
            - cumulative.begin());
    }
}

// Advance through attempts up to an absolute attempt counter, stopping at the
// first accepted one. A pending event beyond the boundary is retained, making
// fixed-seed trajectories invariant to how an attempt interval is partitioned.
int MCTensorialModel::runSkippedToAttempt(long long int target_steps) {
    if (target_steps <= MC_steps) return -1;
    if (!pending_schedule) scheduleAcceptedAttempt();

    const long long int available = target_steps - MC_steps;
    const double interval = 1.0 / (L * L);
    if (pending_site < 0 || attempts_to_event > available) {
        const long double elapsed =
            static_cast<long double>(time_to_next_attempt)
            + static_cast<long double>(available - 1) * interval;
        time += static_cast<double>(elapsed);
        time_to_next_attempt = interval;
        MC_steps = target_steps;
        if (pending_site >= 0) attempts_to_event -= available;
        return -1;
    }

    const long long int attempts = attempts_to_event;
    const long double elapsed =
        static_cast<long double>(time_to_next_attempt)
        + static_cast<long double>(attempts - 1) * interval;
    time += static_cast<double>(elapsed);
    time_to_next_attempt = interval;
    MC_steps += attempts;

    const int site = pending_site;
    pending_schedule = false;
    pending_site = -1;
    attempts_to_event = 0;
    executeEvent(site);
    return site;
}

// One Monte Carlo attempt. Exact skipping still honors this one-attempt public
// contract; bulk paths call runSkippedToAttempt() with a distant boundary.
int MCTensorialModel::runMCStep() {
    if (!rejection_skipping) return runUniformAttempt();
    if (MC_steps == std::numeric_limits<long long int>::max()) return -1;
    return runSkippedToAttempt(MC_steps + 1);
}

// Literal rejection path: attempt moves until one is accepted or the clock
// would pass target_time.
int MCTensorialModel::stepUntil(double target_time) {
    if (rejection_skipping) return stepUntilSkipped(target_time);
    if (target_time <= time) return -1;

    while (time_to_next_attempt <= target_time - time) {
        const int site = runUniformAttempt();
        if (site >= 0) return site;
    }

    time_to_next_attempt -= target_time - time;
    time = target_time;
    return -1;
}

// Exact skipped path for a physical-time boundary. Whole attempts are consumed
// in one block; any fractional interval and the already sampled future event
// are retained for the next call.
int MCTensorialModel::stepUntilSkipped(double target_time) {
    if (target_time <= time) return -1;

    const long double available =
        static_cast<long double>(target_time)
        - static_cast<long double>(time);
    if (available < time_to_next_attempt) {
        time_to_next_attempt -= static_cast<double>(available);
        time = target_time;
        return -1;
    }

    // Match the double interval used by literal MC and by
    // time_to_next_attempt; do not introduce a second rounded clock unit.
    const double interval_double = 1.0 / (L * L);
    const long double interval =
        static_cast<long double>(interval_double);
    const long double additional =
        std::floor((available - time_to_next_attempt) / interval);
    long double attempts_fit_ld = additional + 1.0L;
    const long long int capacity =
        std::numeric_limits<long long int>::max() - MC_steps;
    if (attempts_fit_ld > static_cast<long double>(capacity)) {
        attempts_fit_ld = static_cast<long double>(capacity);
    }
    const long long int attempts_fit =
        static_cast<long long int>(attempts_fit_ld);
    if (attempts_fit <= 0) {
        time = target_time;
        return -1;
    }

    long double residual = available
        - (static_cast<long double>(time_to_next_attempt)
           + static_cast<long double>(attempts_fit - 1) * interval);
    if (residual < 0.0L) residual = 0.0L;
    if (residual >= interval) {
        residual = std::nextafter(interval, 0.0L);
    }

    const int site = runSkippedToAttempt(MC_steps + attempts_fit);
    if (site >= 0) return site;

    time = target_time;
    time_to_next_attempt = static_cast<double>(interval - residual);
    return -1;
}

// Shadow the base diagnostics so max_moves remains an attempt budget while the
// accelerated path jumps directly between accepted events.
double MCTensorialModel::measureRelaxationTime(long long int max_moves) {
    if (!rejection_skipping) {
        return TensorialModel::measureRelaxationTime(max_moves);
    }

    resetPersistence();
    const long long int capacity =
        std::numeric_limits<long long int>::max() - MC_steps;
    const long long int budget = std::min(max_moves, capacity);
    const long long int target = MC_steps + budget;
    while (MC_steps < target) {
        const int site = runSkippedToAttempt(target);
        if (site < 0) break;
        if (persistence() <= 0.5) {
            return time - persistence_origin;
        }
    }
    return -1.0;
}

std::vector<double> MCTensorialModel::measureRelaxationTimes(
        int reps, long long int max_moves) {
    if (!rejection_skipping) {
        return TensorialModel::measureRelaxationTimes(reps, max_moves);
    }

    std::vector<double> out;
    out.reserve(reps);
    for (int rep = 0; rep < reps; rep++) {
        const double value = measureRelaxationTime(max_moves);
        if (value > 0.0) out.push_back(value);
    }
    return out;
}

bool MCTensorialModel::runSimulationSkipped(long long int num_steps) {
    const int num_reports = 40;
    std::vector<long long int> schedule;
    for (int report = 0; report <= num_reports; report++) {
        const double fraction =
            static_cast<double>(report) / num_reports;
        const long long int step = (report == num_reports)
            ? num_steps
            : static_cast<long long int>(
                std::pow(static_cast<double>(num_steps), fraction));
        if (schedule.empty() || step > schedule.back()) {
            schedule.push_back(step);
        }
    }
    if (schedule.back() < num_steps) schedule.push_back(num_steps);

    if (num_steps > std::numeric_limits<long long int>::max() - MC_steps) {
        std::cout << "MC attempt counter would overflow; stopping." << std::endl;
        return false;
    }
    const long long int start_step = MC_steps;
    const long long int end_step = start_step + num_steps;
    std::size_t next_report = 0;
    long long int last_step = MC_steps;
    long long int last_accepted = accepted_steps;
    long long int last_guard = MC_steps;
    const long long int guard_interval = 1000;

    while (next_report < schedule.size()) {
        const long long int target = start_step + schedule[next_report];
        while (MC_steps < target) {
            const int site = runSkippedToAttempt(target);
            if (site < 0) break;

            if (MC_steps - last_guard >= guard_interval) {
                const double rms = stressRms();
                last_guard = MC_steps;
                if (!std::isfinite(rms) || rms > 1e3) {
                    std::cout << "Stress field is diverging (rms = " << rms
                              << "); stopping." << std::endl;
                    return false;
                }
            }
        }

        const double rms = stressRms();
        const long long int delta_steps = MC_steps - last_step;
        const long long int delta_accepted = accepted_steps - last_accepted;
        std::cout << "Step " << (MC_steps - start_step) << "/" << num_steps
                  << " (Acceptance rate: "
                  << static_cast<double>(accepted_steps) / MC_steps
                  << ", window: "
                  << (delta_steps > 0
                      ? static_cast<double>(delta_accepted) / delta_steps
                      : 0.0)
                  << ", events/site: "
                  << static_cast<double>(accepted_steps) / (L * L)
                  << ", rms: " << rms << ")" << std::endl;
        last_step = MC_steps;
        last_accepted = accepted_steps;
        next_report++;

        if (!std::isfinite(rms) || rms > 1e3) {
            std::cout << "Stress field is diverging (rms = " << rms
                      << "); stopping." << std::endl;
            return false;
        }
    }

    if (MC_steps != end_step) return false;
    std::cout << "Simulation completed. Total MC steps: " << MC_steps
              << std::endl;
    std::cout << "Accepted steps: " << accepted_steps
              << " (Rate: "
              << static_cast<double>(accepted_steps) / MC_steps << ")"
              << std::endl;
    return true;
}

// Run simulation for a fixed number of attempts. The default branch is kept
// byte-for-byte compatible in its RNG use with the historical implementation.
bool MCTensorialModel::runSimulation(long long int num_steps) {
    std::cout << "Starting MC simulation with L = " << L << ", T = " << T
              << std::endl;
    if (rejection_skipping) {
        std::cout << "MC mode: exact geometric rejection skipping; "
                  << "-steps still counts attempted moves." << std::endl;
        return runSimulationSkipped(num_steps);
    }

    const int num_reports = 40;
    std::vector<long long int> schedule;
    for (int r = 0; r <= num_reports; r++) {
        double f = static_cast<double>(r) / num_reports;
        long long int s = (r == num_reports)
                        ? num_steps
                        : static_cast<long long int>(
                            std::pow(static_cast<double>(num_steps), f));
        if (schedule.empty() || s > schedule.back()) schedule.push_back(s);
    }
    if (schedule.back() < num_steps) schedule.push_back(num_steps);

    size_t next_report = 0;
    long long int last_step = 0, last_accepted = 0;
    const long long int guard_interval = 1000;

    std::cout << "# step  cum_accept  window_accept  stress_rms" << std::endl;
    for (long long int step = 0; step < num_steps; step++) {
        runUniformAttempt();

        const bool report = next_report < schedule.size()
                         && step + 1 >= schedule[next_report];
        const bool check_stress = report
                               || (step + 1) % guard_interval == 0
                               || step + 1 == num_steps;
        double rms = 0.0;
        if (check_stress) rms = stressRms();

        if (report) {
            const long long int dstep = MC_steps - last_step;
            const long long int dacc = accepted_steps - last_accepted;
            std::cout << "Step " << MC_steps << "/" << num_steps
                      << " (Acceptance rate: "
                      << static_cast<double>(accepted_steps) / MC_steps
                      << ", window: "
                      << (dstep > 0 ? static_cast<double>(dacc) / dstep : 0.0)
                      << ", events/site: "
                      << static_cast<double>(accepted_steps) / (L * L)
                      << ", rms: " << rms << ")" << std::endl;
            last_step = MC_steps;
            last_accepted = accepted_steps;
            while (next_report < schedule.size()
                   && schedule[next_report] <= step + 1) next_report++;
        }

        if (check_stress && (!std::isfinite(rms) || rms > 1e3)) {
            std::cout << "Stress field is diverging (rms = " << rms
                      << "); stopping." << std::endl;
            return false;
        }
    }

    std::cout << "Simulation completed. Total MC steps: " << MC_steps
              << std::endl;
    std::cout << "Accepted steps: " << accepted_steps
              << " (Rate: "
              << static_cast<double>(accepted_steps) / MC_steps << ")"
              << std::endl;
    return true;
}

// Override saveStatistics to include MC-specific information.
bool MCTensorialModel::saveStatistics(const std::string& filename) {
    if (!TensorialModel::saveStatistics(filename)) return false;

    std::ofstream file(filename, std::ios_base::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for appending." << std::endl;
        return false;
    }

    file << "# Algorithm: mc\n";
    file << "# MC steps: " << MC_steps << "\n";
    file << "# Accepted steps: " << accepted_steps << "\n";
    file << "# Acceptance rate: " << getAcceptanceRate() << "\n";
    if (rejection_skipping) {
        file << "# MC rejection skipping: exact_geometric\n";
    }

    file.close();
    if (!file) {
        std::cerr << "Error: Failed while appending " << filename << "."
                  << std::endl;
        return false;
    }
    std::cout << "MC statistics appended to " << filename << std::endl;
    return true;
}
