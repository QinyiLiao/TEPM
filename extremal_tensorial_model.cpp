#include "extremal_tensorial_model.h"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <limits>

// The base class wants a temperature; extremal dynamics has none, and siteRate
// is never called, so 0 is safe here and would not be anywhere else.
ExtremalTensorialModel::ExtremalTensorialModel(int size)
    : TensorialModel(size, 0.0),
      last_x_min(0.0),
      last_x_second(0.0),
      max_measured_x_min(-std::numeric_limits<double>::infinity()),
      steps(0),
      hist_x_min(NUM_BINS, 0),
      hist_x_regular(NUM_BINS, 0),
      hist_x_stable(NUM_BINS, 0),
      regular_x_samples(0),
      stable_x_samples(0),
      stable_configuration_samples(0),
      stable_x0(std::numeric_limits<double>::quiet_NaN()),
      regular_px_every(0),
      avalanche_blocks(0),
      transient_steps_requested(0),
      transient_steps_completed(0),
      measurement_steps_requested(0),
      measurement_steps_completed(0),
      equilibration_completed(false),
      measurement_completed(false),
      divergence_detected(false),
      failure_stage("none") {
}

ExtremalTensorialModel::WeakestSites
ExtremalTensorialModel::findWeakestSites() const {
    int N = L * L;
    WeakestSites weakest{0,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
    for (int i = 0; i < N; i++) {
        double x = calculateDistanceToYield(i);
        if (x < weakest.x_min) {
            weakest.x_second = weakest.x_min;
            weakest.x_min = x;
            weakest.site = i;
        } else if (x < weakest.x_second) {
            weakest.x_second = x;
        }
    }
    return weakest;
}

int ExtremalTensorialModel::relaxWeakest(const WeakestSites& weakest) {
    last_x_min = weakest.x_min;
    last_x_second = weakest.x_second;

    double z = drawResidualStress();
    double delta_sigma_xx, delta_sigma_xy;
    computeStressDrop(weakest.site, weakest.x_min, z,
                      delta_sigma_xx, delta_sigma_xy);
    applyStressDrop(weakest.site, delta_sigma_xx, delta_sigma_xy);

    // No physical time at T = 0+; the step index is the clock
    time += 1.0;
    steps++;
    recordEvent(weakest.site);

    return weakest.site;
}

// One extremal step: find the weakest site and relax it.
int ExtremalTensorialModel::stepExtremal() {
    return relaxWeakest(findWeakestSites());
}

bool ExtremalTensorialModel::equilibrate(long long int num_steps) {
    transient_steps_requested = num_steps;
    transient_steps_completed = 0;
    equilibration_completed = false;
    divergence_detected = false;
    failure_stage = "none";

    long long int report = std::max(1LL, num_steps / 10);

    for (long long int s = 0; s < num_steps; s++) {
        stepExtremal();
        transient_steps_completed = s + 1;
        if (s % report == 0 || s + 1 == num_steps) {
            double rms = stressRms();
            std::cout << "Step " << s << "/" << num_steps
                      << " (x_min = " << last_x_min
                      << ", rms = " << rms << ")" << std::endl;
            if (!std::isfinite(rms) || rms > 1e3) {
                std::cout << "Stress field is diverging (rms = " << rms
                          << "); stopping." << std::endl;
                divergence_detected = true;
                failure_stage = "equilibration";
                return false;
            }
        }
    }
    equilibration_completed = true;
    return true;
}

void ExtremalTensorialModel::setThresholds(const std::vector<double>& x0s) {
    int N = L * L;
    thresholds.clear();
    thresholds.reserve(x0s.size());

    for (double x0 : x0s) {
        Threshold t;
        t.x0 = x0;
        t.boundary_seen = false;
        t.left_truncated_discarded = false;
        t.active = false;
        t.id = 0;
        t.S = 0;
        t.S_tilde = 0;
        t.left_truncated_events = 0;
        t.stamp.assign(N, -1);
        thresholds.push_back(std::move(t));
    }
}

void ExtremalTensorialModel::prepareAvalancheRun(
        long long int num_steps, int px_every, double stable_threshold,
        int num_blocks) {
    measurement_steps_requested = num_steps;
    measurement_steps_completed = 0;
    measurement_completed = false;
    regular_px_every = px_every;
    stable_x0 = stable_threshold;
    avalanche_blocks = num_blocks;
    for (Threshold& t : thresholds) {
        t.blocks.assign(avalanche_blocks, {0, 0.0L, 0.0L, 0.0L, 0.0L});
    }
}

void ExtremalTensorialModel::sampleStableConfiguration(
        const WeakestSites& weakest, long long measurement_step) {
    const double emin = weakest.x_min * std::sqrt(weakest.x_min);
    const double esecond = weakest.x_second * std::sqrt(weakest.x_second);
    stable_samples.push_back(
        {measurement_step, weakest.x_min, weakest.x_second, esecond - emin});
    stable_configuration_samples++;

    int N = L * L;
    for (int i = 0; i < N; i++) {
        double x = calculateDistanceToYield(i);
        if (x >= 0.0 && x <= sigma_Y) {
            const int bin = std::min(
                NUM_BINS - 1, static_cast<int>(x / sigma_Y * NUM_BINS));
            hist_x_stable[bin]++;
            stable_x_samples++;
        }
    }
}

long long ExtremalTensorialModel::blockBoundary(int block) const {
    const long long quotient = measurement_steps_requested / avalanche_blocks;
    const long long remainder = measurement_steps_requested % avalanche_blocks;
    return quotient * block + std::min<long long>(block, remainder);
}

int ExtremalTensorialModel::blockForStep(long long measurement_step) const {
    const long long quotient = measurement_steps_requested / avalanche_blocks;
    const long long remainder = measurement_steps_requested % avalanche_blocks;
    const long long longer_region = (quotient + 1) * remainder;
    if (measurement_step < longer_region) {
        return static_cast<int>(measurement_step / (quotient + 1));
    }
    return static_cast<int>(remainder
        + (measurement_step - longer_region) / quotient);
}

bool ExtremalTensorialModel::runAvalanches(
        long long int num_steps, int px_every, double stable_threshold,
        int num_blocks) {
    int N = L * L;
    prepareAvalancheRun(num_steps, px_every, stable_threshold, num_blocks);
    std::fill(hist_x_min.begin(), hist_x_min.end(), 0);
    std::fill(hist_x_regular.begin(), hist_x_regular.end(), 0);
    std::fill(hist_x_stable.begin(), hist_x_stable.end(), 0);
    regular_x_samples = 0;
    stable_x_samples = 0;
    stable_configuration_samples = 0;
    stable_samples.clear();
    max_measured_x_min = -std::numeric_limits<double>::infinity();

    // The last pre-event x_min observed during equilibration tells us whether
    // the production window starts outside an avalanche. If it was already
    // below a threshold, discard that left-censored run until its first upper
    // boundary. With no transient there is no preceding observation, so use
    // the same conservative discard.
    for (Threshold& t : thresholds) {
        t.active = false;
        t.id = 0;
        t.S = 0;
        t.S_tilde = 0;
        t.left_truncated_events = 0;
        std::fill(t.stamp.begin(), t.stamp.end(), -1);
        t.hist_S.clear();
        t.hist_S_tilde.clear();
        t.boundary_seen = transient_steps_completed > 0 && last_x_min >= t.x0;
        t.left_truncated_discarded =
            transient_steps_completed > 0 && last_x_min < t.x0;
    }

    long long int report = std::max(1LL, num_steps / 10);

    for (long long int s = 0; s < num_steps; s++) {
        const WeakestSites weakest = findWeakestSites();
        if (std::isfinite(stable_x0) && weakest.x_min > stable_x0
                && weakest.x_min >= 0.0
                && std::isfinite(weakest.x_second)
                && weakest.x_second >= weakest.x_min) {
            sampleStableConfiguration(weakest, s);
        }

        int site = relaxWeakest(weakest);
        measurement_steps_completed = s + 1;
        max_measured_x_min = std::max(max_measured_x_min, last_x_min);

        if (last_x_min >= 0.0 && last_x_min <= sigma_Y) {
            const int bin = std::min(
                NUM_BINS - 1,
                static_cast<int>(last_x_min / sigma_Y * NUM_BINS));
            hist_x_min[bin]++;
        }

        for (Threshold& t : thresholds) {
            if (!t.boundary_seen) {
                if (last_x_min >= t.x0) {
                    t.boundary_seen = true;
                } else {
                    t.left_truncated_discarded = true;
                    t.left_truncated_events++;
                }
                continue;
            }

            if (last_x_min < t.x0) {
                if (!t.active) {
                    t.active = true;
                    t.id++;
                    t.S = 0;
                    t.S_tilde = 0;
                }
                t.S++;
                if (t.stamp[site] != t.id) {
                    t.stamp[site] = t.id;
                    t.S_tilde++;
                }
            } else if (t.active) {
                t.active = false;
                t.hist_S[t.S]++;
                t.hist_S_tilde[t.S_tilde]++;
                const int block = blockForStep(s);
                BlockMoments& moments = t.blocks[block];
                const long double size = static_cast<long double>(t.S);
                const long double site_size = static_cast<long double>(t.S_tilde);
                moments.completed++;
                moments.sum_S2 += size * size;
                moments.sum_S3 += size * size * size;
                moments.sum_St2 += site_size * site_size;
                moments.sum_St3 += site_size * site_size * site_size;
            }
        }

        if (px_every > 0 && s % px_every == 0) {
            for (int i = 0; i < N; i++) {
                double x = calculateDistanceToYield(i);
                if (x >= 0.0 && x <= sigma_Y) {
                    const int bin = std::min(
                        NUM_BINS - 1,
                        static_cast<int>(x / sigma_Y * NUM_BINS));
                    hist_x_regular[bin]++;
                    regular_x_samples++;
                }
            }
        }

        if (s % report == 0 || s + 1 == num_steps) {
            double rms = stressRms();
            std::cout << "Step " << s << "/" << num_steps
                      << " (x_min = " << last_x_min << ", rms = " << rms << ")" << std::endl;
            if (!std::isfinite(rms) || rms > 1e3) {
                std::cout << "Stress field is diverging (rms = " << rms
                          << "); stopping." << std::endl;
                divergence_detected = true;
                failure_stage = "measurement";
                return false;
            }
        }
    }
    // Whatever avalanche is still open is left unrecorded on purpose: its
    // size is truncated by the end of the run and would bias the moments low.
    measurement_completed = true;
    if (std::isfinite(stable_x0) && stable_configuration_samples == 0) {
        failure_stage = "stable_sampling_no_configurations";
        return false;
    }
    return true;
}

void ExtremalTensorialModel::writeRunMetadata(std::ostream& file) const {
    const char* drop_name = (drop_rule == DropRule::Paper) ? "paper"
                          : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                 : "aligned";
    file << std::setprecision(17)
         << "# algorithm extremal\n"
         << "# L " << L << "\n"
         << "# seed " << random_seed << "\n"
         << "# kernel_mode " << (projector_kernel ? "proj" : "paper") << "\n"
         << "# drop_rule " << drop_name << "\n"
         << "# kernel_amplitude " << kernel_amplitude << "\n"
         << "# transient_steps_requested " << transient_steps_requested << "\n"
         << "# transient_steps_completed " << transient_steps_completed << "\n"
         << "# equilibration_completed " << (equilibration_completed ? 1 : 0) << "\n"
         << "# measurement_steps_requested " << measurement_steps_requested << "\n"
         << "# measurement_steps_completed " << measurement_steps_completed << "\n"
         << "# measurement_completed " << (measurement_completed ? 1 : 0) << "\n"
         << "# divergence_detected " << (divergence_detected ? 1 : 0) << "\n"
         << "# failure_stage " << failure_stage << "\n"
         << "# total_extremal_steps " << steps << "\n"
         << "# regular_px_every " << regular_px_every << "\n"
         << "# avalanche_blocks " << avalanche_blocks << "\n";
    if (measurement_steps_completed > 0) {
        file << "# max_measured_x_min " << max_measured_x_min << "\n";
    } else {
        file << "# max_measured_x_min unavailable\n";
    }
    if (std::isfinite(stable_x0)) {
        file << "# stable_x0 " << stable_x0 << "\n";
    } else {
        file << "# stable_x0 disabled\n";
    }
    file << "# stable_configuration_samples " << stable_configuration_samples << "\n";
}

// One file per threshold, holding the two size histograms. Keeping the full
// histogram rather than just the moments lets the analysis bootstrap
// S_c = <S^3>/<S^2> and plot P(S) without re-running anything.
void ExtremalTensorialModel::saveAvalanches(const std::string& prefix) const {
    for (const Threshold& t : thresholds) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.4f", t.x0);

        std::string filename = prefix + "_x0" + buf + ".dat";
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: could not open " << filename << std::endl;
            continue;
        }

        writeRunMetadata(file);
        file << "# x0 " << t.x0 << "\n";
        long long completed = 0;
        for (const auto& kv : t.hist_S) completed += kv.second;
        file << "# completed avalanches " << completed << "\n";
        file << "# left_truncated_discarded "
             << (t.left_truncated_discarded ? 1 : 0) << "\n";
        file << "# left_truncated_events " << t.left_truncated_events << "\n";
        const bool right_truncated = t.active
            || (!t.boundary_seen && t.left_truncated_discarded);
        file << "# right_truncated_discarded "
             << (right_truncated ? 1 : 0) << "\n";
        file << "# kind size count\n";
        for (const auto& kv : t.hist_S)       file << "S " << kv.first << " " << kv.second << "\n";
        for (const auto& kv : t.hist_S_tilde) file << "St " << kv.first << " " << kv.second << "\n";
        file.close();
    }
    std::cout << "Avalanche histograms saved to " << prefix << "_x0*.dat" << std::endl;
}

void ExtremalTensorialModel::saveAvalancheBlocks(const std::string& prefix) const {
    for (const Threshold& t : thresholds) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.4f", t.x0);

        std::string filename = prefix + "_x0" + buf + ".dat";
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: could not open " << filename << std::endl;
            continue;
        }

        writeRunMetadata(file);
        file << "# x0 " << t.x0 << "\n"
             << "# assignment completion_step\n"
             << "# block start_step end_step completed sum_S2 sum_S3 sum_St2 sum_St3\n"
             << std::setprecision(21);
        for (int b = 0; b < avalanche_blocks; b++) {
            const long long start = blockBoundary(b);
            const long long end = blockBoundary(b + 1);
            const BlockMoments& moments = t.blocks[b];
            file << b << " " << start << " " << end << " "
                 << moments.completed << " "
                 << moments.sum_S2 << " " << moments.sum_S3 << " "
                 << moments.sum_St2 << " " << moments.sum_St3 << "\n";
        }
        file.close();
    }
    std::cout << "Avalanche block moments saved to " << prefix << "_x0*.dat" << std::endl;
}

void ExtremalTensorialModel::saveDistributions(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << std::endl;
        return;
    }

    long long int min_total = 0;
    for (long long int c : hist_x_min) min_total += c;

    writeRunMetadata(file);
    file << "# regular_x_samples " << regular_x_samples << "\n";
    file << "# stable_x_samples " << stable_x_samples << "\n";
    file << "# histogram_support 0<=x<=sigma_Y; each column is normalized over its in-range samples\n";
    file << "# stable_sampling pre_event_configurations_with_x_min_strictly_above_stable_x0\n";
    file << "# x P(x_min) P(x_regular) P(x_stable)\n";
    for (int b = 0; b < NUM_BINS; b++) {
        double x = (b + 0.5) / NUM_BINS * sigma_Y;
        double p_min = min_total
            ? static_cast<double>(hist_x_min[b]) * NUM_BINS
                / (static_cast<double>(min_total) * sigma_Y)
            : 0.0;
        double p_regular = regular_x_samples
            ? static_cast<double>(hist_x_regular[b]) * NUM_BINS
                / (static_cast<double>(regular_x_samples) * sigma_Y)
            : 0.0;
        double p_stable = stable_x_samples
            ? static_cast<double>(hist_x_stable[b]) * NUM_BINS
                / (static_cast<double>(stable_x_samples) * sigma_Y)
            : 0.0;
        file << x << " " << p_min << " " << p_regular << " " << p_stable << "\n";
    }
    file.close();
    std::cout << "Distributions saved to " << filename << std::endl;
}

void ExtremalTensorialModel::saveEnergyGaps(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filename << std::endl;
        return;
    }

    writeRunMetadata(file);
    file << "# measurement_step x_min x_second E_second_minus_E_min\n";
    for (const StableSample& sample : stable_samples) {
        file << sample.measurement_step << " "
             << sample.x_min << " "
             << sample.x_second << " "
             << sample.energy_gap << "\n";
    }
    file.close();
    std::cout << "Stable-state energy gaps saved to " << filename << std::endl;
}
