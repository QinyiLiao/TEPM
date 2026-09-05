#ifndef EXTREMAL_TENSORIAL_MODEL_H
#define EXTREMAL_TENSORIAL_MODEL_H

#include "tensorial_model.h"
#include <map>
#include <string>

// Extremal dynamics at T = 0+, following Sec. IV and Appendix C of Tahaei et
// al., PRX 13, 031034 (2023).
//
// At vanishing temperature the site with the smallest distance to yield x has
// an overwhelmingly larger rate than any other, so the dynamics reduces to
// "always relax the weakest site". One step = one x_min. There is no physical
// time; the step index is the only clock.
//
// The trajectory of x_min is a succession of avalanches. Given a threshold
// x_0, an x_0-avalanche is a maximal run of consecutive steps with x_min < x_0.
// Its event-based size S is the number of steps in the run; its site-based size
// S~ is the number of distinct sites relaxed during it, so S~ <= S.
//
// Sizes are accumulated for a whole grid of thresholds in one pass, since each
// threshold only needs O(1) book-keeping per step.
class ExtremalTensorialModel : public TensorialModel {
public:
    static const int NUM_BINS = 200;   // bins for P(x_min) and P(x) on [0, sigma_Y]

private:
    struct BlockMoments {
        long long completed;
        long double sum_S2;
        long double sum_S3;
        long double sum_St2;
        long double sum_St3;
    };

    // One threshold's running avalanche and its size histograms
    struct Threshold {
        double x0;
        bool boundary_seen;       // false while discarding a left-truncated run
        bool left_truncated_discarded;
        bool active;              // currently inside an avalanche
        long long id;             // avalanche counter, used to stamp sites
        long long S;              // events in the current avalanche
        long long S_tilde;        // distinct sites in the current avalanche
        long long left_truncated_events;
        std::vector<long long> stamp;   // per site: id of its last avalanche
        std::map<long long, long long> hist_S;
        std::map<long long, long long> hist_S_tilde;
        std::vector<BlockMoments> blocks;
    };

    struct WeakestSites {
        int site;
        double x_min;
        double x_second;
    };

    struct StableSample {
        long long measurement_step;
        double x_min;
        double x_second;
        double energy_gap;
    };

    std::vector<Threshold> thresholds;
    double last_x_min;
    double last_x_second;
    double max_measured_x_min;
    long long int steps;

    // Histograms over the measured trajectory. The regular P(x) is retained as
    // a diagnostic; the stable P(x) is the paper-comparable distribution,
    // sampled only from pre-event configurations with x_min > stable_x0.
    std::vector<long long> hist_x_min;
    std::vector<long long> hist_x_regular;
    std::vector<long long> hist_x_stable;
    long long int regular_x_samples;
    long long int stable_x_samples;
    long long int stable_configuration_samples;
    std::vector<StableSample> stable_samples;

    double stable_x0;
    int regular_px_every;
    int avalanche_blocks;
    long long int transient_steps_requested;
    long long int transient_steps_completed;
    long long int measurement_steps_requested;
    long long int measurement_steps_completed;
    bool equilibration_completed;
    bool measurement_completed;
    bool divergence_detected;
    std::string failure_stage;

    WeakestSites findWeakestSites() const;
    int relaxWeakest(const WeakestSites& weakest);
    void sampleStableConfiguration(const WeakestSites& weakest,
                                   long long measurement_step);
    long long blockBoundary(int block) const;
    int blockForStep(long long measurement_step) const;
    void writeRunMetadata(std::ostream& file) const;

public:
    explicit ExtremalTensorialModel(int size);

    // Relax the weakest site. Returns that site; last_x_min holds its x.
    int stepExtremal();
    void advance() override { stepExtremal(); }

    // Extremal dynamics has no thermal clock: every event costs exactly one
    // unit of time (see the `time += 1.0` in stepExtremal's callee). Provided
    // so the class is concrete; the paper's finite-temperature estimators are
    // not meaningful at T = 0+ and are never invoked on this driver.
    int stepUntil(double target_time) override {
        if (time + 1.0 > target_time) {
            time = target_time;
            return -1;
        }
        return stepExtremal();
    }

    double lastXmin() const { return last_x_min; }
    double lastXsecond() const { return last_x_second; }
    long long int getSteps() const { return steps; }

    // Run without measuring, to pass the initial transient
    bool equilibrate(long long int num_steps);

    // Set the thresholds to accumulate avalanches for, clearing any histograms
    void setThresholds(const std::vector<double>& x0s);

    // Record the requested measurement before equilibration so a failed run
    // can still overwrite same-tag outputs with complete failure metadata.
    void prepareAvalancheRun(long long int num_steps, int regular_px_every,
                             double stable_threshold, int num_blocks);

    // Run num_steps, accumulating avalanche sizes for every threshold. The
    // first and last truncated avalanche for each threshold are discarded.
    // regular_px_every retains the historical regular-step P(x) diagnostic.
    // When stable_threshold is finite, paper-comparable P(x) and energy gaps
    // are sampled from pre-event configurations with x_min > stable_threshold.
    bool runAvalanches(long long int num_steps, int regular_px_every,
                       double stable_threshold, int num_blocks);

    bool saveAvalanches(const std::string& prefix) const;
    bool saveAvalancheBlocks(const std::string& prefix) const;
    bool saveDistributions(const std::string& filename) const;
    bool saveEnergyGaps(const std::string& filename) const;
    bool divergenceDetected() const { return divergence_detected; }
    const std::string& failureStage() const { return failure_stage; }
};

#endif // EXTREMAL_TENSORIAL_MODEL_H
