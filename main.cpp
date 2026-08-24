#include "tensorial_model.h"
#include "mc_tensorial_model.h"
#include "edmd_tensorial_model.h"
#include "extremal_tensorial_model.h"
#include <algorithm>
#include <sstream>
#include <vector>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <string>

namespace {

double persistenceHalfTime(const std::vector<double>& times,
                           const std::vector<double>& pi_mean) {
    for (std::size_t k = 1; k < times.size(); k++) {
        if (pi_mean[k - 1] >= 0.5 && pi_mean[k] <= 0.5
                                     && pi_mean[k - 1] > pi_mean[k]) {
            const double fraction = (pi_mean[k - 1] - 0.5)
                                  / (pi_mean[k - 1] - pi_mean[k]);
            const double log_tau = std::log(times[k - 1])
                                 + fraction * (std::log(times[k])
                                             - std::log(times[k - 1]));
            return std::exp(log_tau);
        }
    }
    return -1.0;
}

std::vector<double> logarithmicTimeGrid(double center, int count) {
    std::vector<double> times(count);
    const double lo = std::log(center * 1e-3);
    const double hi = std::log(center * 2e1);
    for (int k = 0; k < count; k++) {
        times[k] = std::exp(lo + (hi - lo) * k / (count - 1));
    }
    return times;
}

std::string commaSeparated(const std::vector<double>& values) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i > 0) out << ",";
        out << values[i];
    }
    return out.str();
}

std::string thresholdFilenameKey(double x0) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << x0;
    return out.str();
}

}  // namespace


// ---------------------------------------------------------------------------
// The paper's adaptive physical-time analysis: pilot half-time, a preliminary
// pass to set the time scale, then the production <pi(t)> / chi_4(t) / P(x)
// measurement, with the dynamics and per-origin files written out.
//
// A template rather than a TensorialModel& on purpose. saveStatistics() is
// shadowed, not virtual, in both drivers, so a base reference would silently
// write the base version and drop the driver-specific block. Templating keeps
// the call static, which also means the EDMD instantiation is the same code
// that was inlined here before.
//
// Returns 0 to continue, or the exit code main should return.
// ---------------------------------------------------------------------------
template <class Model>
static int runPaperAnalysis(Model& model,
                            const char* algorithm,
                            int L, double T, std::uint32_t random_seed,
                            bool projector_kernel, DropRule drop_rule,
                            bool literal_select, double kernel_amplitude,
                            long long int steps, int analyze_origins,
                            double analysis_center, int px_samples,
                            bool measure_tau,
                            long long int pilot_move_budget,
                            const std::string& suffix,
                            const std::string& config_file,
                            const std::string& stats_file) {
    const std::string dyn_file = "dynamics" + suffix + ".dat";
    const std::string origins_file = "dynamics_origins" + suffix + ".dat";
    double tau = -1.0;
    if (measure_tau || analyze_origins > 0) {
        std::cout << "Measuring a single-origin pilot half-time..." << std::endl;
        // The budget is counted in whatever advance() does -- plastic events
        // for Gillespie, ATTEMPTS for Monte Carlo -- so the caller supplies it.
        // Passing the Gillespie figure to Monte Carlo buys 400*L*L attempts,
        // which at an acceptance of 1e-5 is a fraction of one event and can
        // never halve the persistence.
        tau = model.measureRelaxationTime(pilot_move_budget);
        if (tau > 0.0) {
            std::cout << "pilot_half_time = " << tau << std::endl;
        } else {
            std::cout << "pilot half-time not reached within the event budget" << std::endl;
        }
    }

    if (analyze_origins > 0 && tau <= 0.0) {
        std::cerr << "Paper-style analysis failed because the pilot "
                  << "half-time was not reached.\n";
        std::ofstream df(dyn_file, std::ios::trunc);
        df << "# algorithm " << algorithm << "\n"
           << "# L " << L << "\n"
           << "# T " << T << "\n"
           << "# seed " << random_seed << "\n"
           << "# analysis_scale_method adaptive_physical_time_v1\n"
           << "# analysis_failure pilot_half_time_not_reached\n"
           << "# production_bracketed 0\n"
           << "# tau_alpha_timeavg -1\n";
        std::ofstream of(origins_file, std::ios::trunc);
        of << "# algorithm " << algorithm << "\n"
           << "# L " << L << "\n"
           << "# T " << T << "\n"
           << "# seed " << random_seed << "\n"
           << "# analysis_scale_method adaptive_physical_time_v1\n"
           << "# analysis_failure pilot_half_time_not_reached\n"
           << "# production_bracketed 0\n";
        model.saveConfiguration(config_file);
        model.saveStatistics(stats_file);
        return 3;
    }

    if (analyze_origins > 0 && tau > 0.0) {
        const int num_times = 60;
        const int preliminary_origins = std::min(50, analyze_origins);
        const int max_preliminary_attempts = 8;
        const int max_production_attempts = 4;
        std::vector<double> preliminary_times;
        std::vector<double> preliminary_pi, preliminary_chi4;
        std::vector<double> preliminary_centers, preliminary_endpoint_pi;
        double preliminary_center = tau;
        double preliminary_tau = -1.0;

        for (int attempt = 0; attempt < max_preliminary_attempts; attempt++) {
            preliminary_times = logarithmicTimeGrid(preliminary_center,
                                                    num_times);
            std::cout << "Refining the physical-time scale over "
                      << preliminary_origins << " preliminary origins"
                      << " (attempt " << (attempt + 1) << "/"
                      << max_preliminary_attempts << ", center = "
                      << preliminary_center << ")..." << std::endl;
            model.measureDynamics(preliminary_origins, preliminary_times,
                                  preliminary_center, preliminary_pi,
                                  preliminary_chi4);
            preliminary_centers.push_back(preliminary_center);
            preliminary_endpoint_pi.push_back(preliminary_pi.back());
            preliminary_tau = persistenceHalfTime(preliminary_times,
                                                 preliminary_pi);
            if (preliminary_tau > 0.0) break;

            if (preliminary_pi.back() > 0.5) {
                preliminary_center *= 20.0;
            } else if (preliminary_pi.front() < 0.5) {
                preliminary_center *= 1e-3;
            } else {
                break;
            }
            if (!std::isfinite(preliminary_center)
                || preliminary_center <= 0.0) {
                break;
            }
        }

        if (preliminary_tau <= 0.0) {
            std::cerr << "Adaptive preliminary analysis failed to bracket "
                      << "tau_alpha after " << preliminary_centers.size()
                      << " attempts; centers="
                      << commaSeparated(preliminary_centers)
                      << ", endpoint_pi="
                      << commaSeparated(preliminary_endpoint_pi) << "\n";
            std::ofstream df(dyn_file, std::ios::trunc);
            df << "# algorithm " << algorithm << "\n"
               << "# L " << L << "\n"
               << "# T " << T << "\n"
               << "# seed " << random_seed << "\n"
               << "# analysis_scale_method adaptive_physical_time_v1\n"
               << "# analysis_failure preliminary_scale_unbracketed\n"
               << "# pilot_half_time " << tau << "\n"
               << "# preliminary_physical_time_origins "
               << preliminary_origins << "\n"
               << "# preliminary_attempts "
               << preliminary_centers.size() << "\n"
               << "# preliminary_centers "
               << commaSeparated(preliminary_centers) << "\n"
               << "# preliminary_endpoint_pi "
               << commaSeparated(preliminary_endpoint_pi) << "\n"
               << "# production_bracketed 0\n"
               << "# tau_alpha_timeavg -1\n";
            std::ofstream of(origins_file, std::ios::trunc);
            of << "# algorithm " << algorithm << "\n"
               << "# L " << L << "\n"
               << "# T " << T << "\n"
               << "# seed " << random_seed << "\n"
               << "# analysis_scale_method adaptive_physical_time_v1\n"
               << "# analysis_failure preliminary_scale_unbracketed\n"
               << "# production_bracketed 0\n";
            model.saveConfiguration(config_file);
            model.saveStatistics(stats_file);
            return 3;
        }
        std::cout << "preliminary_tau_alpha_timeavg = " << preliminary_tau
                  << std::endl;

        // Center the production grid and origin spacing on a physical-time
        // estimate rather than on the noisier post-event pilot half-time.
        // If a finite-size intermittent trajectory still misses the
        // crossing, discard that production attempt and expand the scale.
        std::vector<double> times;
        std::vector<double> pi_mean, chi4;
        std::vector<std::vector<double>> pi_by_origin;
        std::vector<double> production_centers, production_endpoint_pi;
        double production_center = (analysis_center > 0.0)
                                 ? analysis_center : preliminary_tau;
        if (analysis_center > 0.0) {
            std::cout << "Using requested shared production center = "
                      << analysis_center << std::endl;
        }
        double tau_timeavg = -1.0;
        for (int attempt = 0; attempt < max_production_attempts; attempt++) {
            times = logarithmicTimeGrid(production_center, num_times);
            std::cout << "Measuring <pi(t)> and chi_4(t) over "
                      << analyze_origins << " origins (attempt "
                      << (attempt + 1) << "/" << max_production_attempts
                      << ", center = " << production_center << ")..."
                      << std::endl;
            model.measureDynamics(analyze_origins, times, production_center,
                                  pi_mean, chi4, &pi_by_origin);
            production_centers.push_back(production_center);
            production_endpoint_pi.push_back(pi_mean.back());
            tau_timeavg = persistenceHalfTime(times, pi_mean);
            if (tau_timeavg > 0.0) break;

            if (pi_mean.back() > 0.5) {
                production_center *= 20.0;
            } else if (pi_mean.front() < 0.5) {
                production_center *= 1e-3;
            } else {
                break;
            }
            if (!std::isfinite(production_center)
                || production_center <= 0.0) {
                break;
            }
        }

        const double origin_spacing = production_centers.back();
        const bool production_bracketed = tau_timeavg > 0.0;
        if (tau_timeavg > 0.0) {
            std::cout << "tau_alpha_timeavg = " << tau_timeavg << std::endl;
        } else {
            std::cerr << "Adaptive production analysis failed to bracket "
                      << "tau_alpha after " << production_centers.size()
                      << " attempts; centers="
                      << commaSeparated(production_centers)
                      << ", endpoint_pi="
                      << commaSeparated(production_endpoint_pi) << "\n";
        }

        std::ofstream df(dyn_file);
        df << "# algorithm " << algorithm << "\n";
        df << "# L " << L << "\n";
        df << "# T " << T << "\n";
        df << "# seed " << random_seed << "\n";
        df << "# kernel " << (projector_kernel ? "proj" : "paper") << "\n";
        df << "# drop " << ((drop_rule == DropRule::Paper) ? "paper"
                              : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                     : "aligned") << "\n";
        df << "# selection " << (literal_select ? "literal" : "rate") << "\n";
        df << "# kernel_amplitude " << kernel_amplitude << "\n";
        df << "# equilibration_events " << steps << "\n";
        df << "# analysis_scale_method adaptive_physical_time_v1\n";
        df << "# production_center_override " << analysis_center << "\n";
        df << "# physical_time_origins " << analyze_origins << "\n";
        df << "# origin_spacing " << origin_spacing << "\n";
        df << "# pilot_half_time " << tau << "\n";
        df << "# preliminary_physical_time_origins " << preliminary_origins << "\n";
        df << "# preliminary_attempts " << preliminary_centers.size() << "\n";
        df << "# preliminary_centers "
           << commaSeparated(preliminary_centers) << "\n";
        df << "# preliminary_endpoint_pi "
           << commaSeparated(preliminary_endpoint_pi) << "\n";
        df << "# preliminary_tau_alpha_timeavg " << preliminary_tau << "\n";
        df << "# production_attempts " << production_centers.size() << "\n";
        df << "# production_centers "
           << commaSeparated(production_centers) << "\n";
        df << "# production_endpoint_pi "
           << commaSeparated(production_endpoint_pi) << "\n";
        df << "# discarded_production_origins "
           << analyze_origins * (production_centers.size() - 1) << "\n";
        df << "# production_bracketed " << (production_bracketed ? 1 : 0)
           << "\n";
        df << "# tau_alpha_timeavg " << tau_timeavg << "\n";
        df << "# t  pi(t)  chi4(t)\n";
        for (int k = 0; k < num_times; k++) {
            df << times[k] << " " << pi_mean[k] << " " << chi4[k] << "\n";
        }
        df.close();
        std::cout << "Dynamics saved to " << dyn_file << std::endl;

        std::ofstream of(origins_file);
        of << "# algorithm " << algorithm << "\n";
        of << "# L " << L << "\n";
        of << "# T " << T << "\n";
        of << "# seed " << random_seed << "\n";
        of << "# kernel " << (projector_kernel ? "proj" : "paper") << "\n";
        of << "# drop " << ((drop_rule == DropRule::Paper) ? "paper"
                              : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                     : "aligned") << "\n";
        of << "# selection " << (literal_select ? "literal" : "rate") << "\n";
        of << "# kernel_amplitude " << kernel_amplitude << "\n";
        of << "# equilibration_events " << steps << "\n";
        of << "# analysis_scale_method adaptive_physical_time_v1\n";
        of << "# production_center_override " << analysis_center << "\n";
        of << "# physical_time_origins " << analyze_origins << "\n";
        of << "# origin_spacing " << origin_spacing << "\n";
        of << "# pilot_half_time " << tau << "\n";
        of << "# preliminary_physical_time_origins "
           << preliminary_origins << "\n";
        of << "# preliminary_attempts " << preliminary_centers.size() << "\n";
        of << "# preliminary_centers "
           << commaSeparated(preliminary_centers) << "\n";
        of << "# preliminary_endpoint_pi "
           << commaSeparated(preliminary_endpoint_pi) << "\n";
        of << "# preliminary_tau_alpha_timeavg " << preliminary_tau << "\n";
        of << "# production_attempts " << production_centers.size() << "\n";
        of << "# production_centers "
           << commaSeparated(production_centers) << "\n";
        of << "# production_endpoint_pi "
           << commaSeparated(production_endpoint_pi) << "\n";
        of << "# discarded_production_origins "
           << analyze_origins * (production_centers.size() - 1) << "\n";
        of << "# production_bracketed " << (production_bracketed ? 1 : 0)
           << "\n";
        of << "# tau_alpha_timeavg " << tau_timeavg << "\n";
        of << "# columns origin_index t pi(t)\n";
        for (int origin = 0; origin < analyze_origins; origin++) {
            for (int k = 0; k < num_times; k++) {
                of << origin << " " << times[k] << " "
                   << pi_by_origin[origin][k] << "\n";
            }
        }
        of.close();
        std::cout << "Per-origin persistence saved to " << origins_file << std::endl;

        if (px_samples > 0 && production_bracketed) {
            std::cout << "Measuring P(x)..." << std::endl;
            std::vector<double> px(100);
            const double px_interval = 0.1 * (tau_timeavg > 0.0
                                            ? tau_timeavg : preliminary_tau);
            model.measurePx(px_samples, px_interval, px);

            std::string px_file = "px" + suffix + ".dat";
            std::ofstream pf(px_file);
            pf << "# algorithm " << algorithm << "\n";
            pf << "# L " << L << "\n";
            pf << "# T " << T << "\n";
            pf << "# seed " << random_seed << "\n";
            pf << "# kernel " << (projector_kernel ? "proj" : "paper") << "\n";
            pf << "# drop " << ((drop_rule == DropRule::Paper) ? "paper"
                                  : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                         : "aligned") << "\n";
            pf << "# selection " << (literal_select ? "literal" : "rate") << "\n";
            pf << "# kernel_amplitude " << kernel_amplitude << "\n";
            pf << "# equilibration_events " << steps << "\n";
            pf << "# physical_time_samples " << px_samples << "\n";
            pf << "# sample_interval " << px_interval << "\n";
            pf << "# x  P(x)\n";
            for (size_t b = 0; b < px.size(); b++) {
                pf << (b + 0.5) / px.size() << " " << px[b] << "\n";
            }
            pf.close();
            std::cout << "P(x) saved to " << px_file << std::endl;
        } else if (px_samples == 0) {
            std::cout << "P(x) sampling skipped (-pxsamples 0)." << std::endl;
        } else {
            std::cout << "P(x) sampling skipped because tau_alpha was not "
                      << "bracketed." << std::endl;
        }

        if (!production_bracketed) {
            model.saveConfiguration(config_file);
            model.saveStatistics(stats_file);
            return 3;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    // Default parameters
    int L = 32;                        // System size
    double T = 0.05;                   // Temperature
    long long int steps = -1;          // MC steps, or equilibration events for EDMD
    std::string algo = "edmd";         // "edmd" (Gillespie, as in the paper) or "mc"
    bool measure_tau = false;          // Diagnostic single-origin half-time
    int tau_reps = 0;                  // Repeat the diagnostic post-event half-time
    int analyze_origins = 0;           // Time origins for <pi(t)>, chi_4(t), P(x)
    int px_samples = 2000;              // Physical-time P(x) samples after analysis
    double analysis_center = -1.0;      // Optional shared production time scale
    bool projector_kernel = false;     // Lattice q_x q_y convention, see readme.md
    DropRule drop_rule = DropRule::Aligned;   // Stress drop convention
    bool literal_select = false;       // Correct independent exponential clocks
    std::string tag;                   // Suffix for the output filenames
    double kernel_amplitude = 1.0;     // Diagnostic scale on the off-site kernel
    std::string load_config;           // Initial state read back from a config file
    std::string x0_list;               // Avalanche thresholds for -algo extremal
    long long int transient = -1;      // Extremal steps discarded before measuring
    double stable_x0 = std::numeric_limits<double>::quiet_NaN();
    int avalanche_blocks = 100;        // Completion-time blocks for uncertainty estimates
    std::uint32_t random_seed = 0;      // RNG seed, generated once if not supplied
    bool seed_supplied = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-L" && i + 1 < argc) {
            L = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        } else if (arg == "-steps" && i + 1 < argc) {
            steps = std::stoll(argv[++i]);
        } else if (arg == "-algo" && i + 1 < argc) {
            algo = argv[++i];
        } else if (arg == "-tau") {
            measure_tau = true;
        } else if (arg == "-taureps" && i + 1 < argc) {
            tau_reps = std::stoi(argv[++i]);
        } else if (arg == "-analyze" && i + 1 < argc) {
            analyze_origins = std::stoi(argv[++i]);
        } else if (arg == "-pxsamples" && i + 1 < argc) {
            px_samples = std::stoi(argv[++i]);
            if (px_samples < 0) {
                std::cerr << "-pxsamples must be non-negative\n";
                return 1;
            }
        } else if (arg == "-analysiscenter" && i + 1 < argc) {
            analysis_center = std::stod(argv[++i]);
            if (!std::isfinite(analysis_center) || analysis_center <= 0.0) {
                std::cerr << "-analysiscenter must be finite and positive\n";
                return 1;
            }
        } else if (arg == "-kernel" && i + 1 < argc) {
            std::string k = argv[++i];
            projector_kernel = (k == "proj");
            if (!projector_kernel && k != "paper") {
                std::cerr << "Unknown -kernel " << k << " (use paper or proj)\n";
                return 1;
            }
        } else if (arg == "-drop" && i + 1 < argc) {
            std::string d = argv[++i];
            if (d == "aligned")          drop_rule = DropRule::Aligned;
            else if (d == "paper")       drop_rule = DropRule::Paper;
            else if (d == "papersgn")    drop_rule = DropRule::PaperSigned;
            else {
                std::cerr << "Unknown -drop " << d
                          << " (use aligned, paper or papersgn)\n";
                return 1;
            }
        } else if (arg == "-select" && i + 1 < argc) {
            std::string sel = argv[++i];
            literal_select = (sel == "literal");
            if (!literal_select && sel != "rate") {
                std::cerr << "Unknown -select " << sel << " (use rate or literal)\n";
                return 1;
            }
        } else if (arg == "-kamp" && i + 1 < argc) {
            kernel_amplitude = std::stod(argv[++i]);
        } else if (arg == "-loadconfig" && i + 1 < argc) {
            load_config = argv[++i];
        } else if (arg == "-x0" && i + 1 < argc) {
            x0_list = argv[++i];
        } else if (arg == "-transient" && i + 1 < argc) {
            transient = std::stoll(argv[++i]);
        } else if (arg == "-stablex0" && i + 1 < argc) {
            stable_x0 = std::stod(argv[++i]);
            if (!std::isfinite(stable_x0) || stable_x0 < 0.0 || stable_x0 > 1.0) {
                std::cerr << "-stablex0 must be finite and in [0, 1]\n";
                return 1;
            }
        } else if (arg == "-avalblocks" && i + 1 < argc) {
            avalanche_blocks = std::stoi(argv[++i]);
            if (avalanche_blocks <= 0 || avalanche_blocks > 10000) {
                std::cerr << "-avalblocks must be in [1, 10000]\n";
                return 1;
            }
        } else if (arg == "-tag" && i + 1 < argc) {
            tag = argv[++i];
        } else if (arg == "-seed" && i + 1 < argc) {
            unsigned long long parsed_seed = std::stoull(argv[++i]);
            if (parsed_seed > std::numeric_limits<std::uint32_t>::max()) {
                std::cerr << "Seed must be in [0, "
                          << std::numeric_limits<std::uint32_t>::max() << "]\n";
                return 1;
            }
            random_seed = static_cast<std::uint32_t>(parsed_seed);
            seed_supplied = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -L <size>      System size (default: 32)\n"
                      << "  -T <temp>      Temperature (default: 0.05)\n"
                      << "  -steps <num>   EDMD equilibration events (default 2000*L*L), MC steps,\n"
                      << "                 or extremal measurement steps (default 2000*L*L)\n"
                      << "  -algo <name>   edmd (Gillespie, default), mc, or extremal (T = 0+)\n"
                      << "  -kamp <factor> Scale the off-site kernel (diagnostic; default 1)\n"
                      << "  -x0 <list>     extremal: comma-separated avalanche thresholds\n"
                      << "  -transient <n> extremal: steps discarded before measuring (default 8000*L*L)\n"
                      << "  -stablex0 <x> extremal: sample paper-style stable P(x) and energy gaps\n"
                      << "  -avalblocks <n> extremal: completion-time moment blocks (default 100)\n"
                      << "  -tau           Diagnostic single-origin half-persistence time\n"
                      << "  -taureps <n>   Repeat diagnostic post-event half-times\n"
                      << "  -analyze <n>   Paper-style physical-time <pi(t)>, chi_4(t), P(x)\n"
                      << "  -pxsamples <n> Physical-time P(x) samples after -analyze (default 2000; 0 skips)\n"
                      << "  -analysiscenter <t> Initial production time scale (diagnostic)\n"
                      << "  -kernel <name> paper (Eq. A6 as printed, default) or proj\n"
                      << "  -drop <name>   aligned (corrected, default), paper (Eq. A2/A3\n"
                      << "                 verbatim) or papersgn (their direction, sign repaired)\n"
                      << "  -select <name> rate (default) or literal (independent clocks)\n"
                      << "  -seed <uint32> RNG seed (generated and printed if omitted)\n"
                      << "  -loadconfig <file>  Start from a saved configuration instead of a\n"
                      << "                 random state, e.g. to quench a finite-temperature\n"
                      << "                 configuration to T=0+ with -algo extremal. The file\n"
                      << "                 must describe the same L; its own header temperature\n"
                      << "                 and drop rule are not applied, so set those with -T\n"
                      << "                 and -drop. Six-digit round-tripping means the state\n"
                      << "                 is restored to that precision, not exactly.\n"
                      << "  -tag <name>    Suffix appended to the output filenames\n"
                      << "  -h, --help     Show this help message\n";
            return 0;
        }
    }

    if (algo != "edmd" && algo != "mc" && algo != "extremal") {
        std::cerr << "Unknown -algo " << algo << " (use edmd, mc, or extremal)\n";
        return 1;
    }
    // L*L is stored in int throughout the model, and extremal dynamics needs
    // at least two sites to define the second-smallest distance to yield.
    if (L < 2 || L > 46340) {
        std::cerr << "-L must be in [2, 46340]\n";
        return 1;
    }
    if (transient < -1) {
        std::cerr << "-transient must be non-negative\n";
        return 1;
    }

    // Equilibration has to be measured per site, not in absolute events.
    // tau_alpha drifts down by a factor of a few between ~5 and ~500 events per
    // site, and the drift is larger the colder it gets, so too short an
    // equilibration biases the Arrhenius slope upwards. 2000 is comfortably past
    // where it flattens; the MC driver keeps the old fixed default.
    if (steps < 0) {
        if (algo == "mc")            steps = 100000;
        else if (algo == "extremal") steps = 2000LL * L * L;
        else                         steps = 2000LL * L * L;
    }
    if (algo == "extremal" && steps <= 0) {
        std::cerr << "Extremal -steps must be positive\n";
        return 1;
    }

    if (std::isfinite(stable_x0) && algo != "extremal") {
        std::cerr << "-stablex0 is available only with -algo extremal\n";
        return 1;
    }

    if (!seed_supplied) {
        std::random_device rd;
        random_seed = rd();
    }

    std::cout << "Starting tensorial elastoplastic model simulation\n";
    std::cout << "Algorithm: " << algo << "\n";
    std::cout << "Kernel: "
              << (projector_kernel ? "proj (non-amplifying diagnostic)" : "paper (Eq. A6)")
              << "\n";
    const char* drop_name = (drop_rule == DropRule::Paper)      ? "paper (Eq. A2/A3 verbatim)"
                          : (drop_rule == DropRule::PaperSigned) ? "papersgn (paper direction)"
                                                                 : "aligned (corrected)";
    std::cout << "Stress drop: " << drop_name << "\n";
    std::cout << "System size: " << L << "x" << L << "\n";
    if (algo == "extremal") {
        std::cout << "Temperature: 0+ (extremal; -T is ignored)\n";
    } else {
        std::cout << "Temperature: " << T << "\n";
    }
    std::cout << "RNG seed: " << random_seed << "\n";
    std::cout << "Steps: " << steps
              << " (" << (static_cast<double>(steps) / (L * L)) << " per site)\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::string suffix = "_L" + std::to_string(L)
                       + (algo == "extremal" ? "" : "_T" + std::to_string(T)) + tag;
    std::string config_file = "config" + suffix + ".dat";
    std::string stats_file = "stats" + suffix + ".dat";

    if (algo == "extremal") {
        // Accumulate a broad passive threshold grid by default, wide enough to
        // bracket the critical point without assuming where it lies. The grid
        // is passive: it changes what is recorded, not the extremal trajectory.
        std::vector<double> x0s;
        if (x0_list.empty()) {
            for (int k = 0; k <= 16; k++) {
                x0s.push_back(0.30 + 0.025 * k);
            }
        } else {
            if (x0_list.front() == ',' || x0_list.back() == ','
                    || x0_list.find(",,") != std::string::npos) {
                std::cerr << "-x0 must be a non-empty comma-separated list\n";
                return 1;
            }
            std::stringstream ss(x0_list);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (item.empty()) {
                    std::cerr << "-x0 must be a non-empty comma-separated list\n";
                    return 1;
                }
                try {
                    std::size_t used = 0;
                    const double x0 = std::stod(item, &used);
                    if (used != item.size()) {
                        std::cerr << "Invalid -x0 value: " << item << "\n";
                        return 1;
                    }
                    x0s.push_back(x0);
                } catch (const std::exception&) {
                    std::cerr << "Invalid -x0 value: " << item << "\n";
                    return 1;
                }
            }
        }
        if (x0s.empty()) {
            std::cerr << "-x0 must contain at least one threshold\n";
            return 1;
        }
        std::set<std::string> x0_filename_keys;
        for (double x0 : x0s) {
            if (!std::isfinite(x0) || x0 < 0.0 || x0 > 1.0) {
                std::cerr << "Each -x0 threshold must be finite and in [0, 1]\n";
                return 1;
            }
            const std::string key = thresholdFilenameKey(x0);
            if (!x0_filename_keys.insert(key).second) {
                std::cerr << "-x0 thresholds collide at four-decimal output key "
                          << key << "\n";
                return 1;
            }
        }

        ExtremalTensorialModel model(L);
        model.setRandomSeed(random_seed);
        model.setProjectorKernel(projector_kernel);
        model.setDropRule(drop_rule);
        model.setKernelAmplitude(kernel_amplitude);
        std::cout << "Initializing system..." << std::endl;
        model.initialize();
        if (!load_config.empty()) {
            if (!model.loadConfiguration(load_config)) {
                std::cerr << "Failed to read the initial configuration; "
                          << "refusing to run from a random state instead.\n";
                return 1;
            }
            std::cout << "Initial state loaded from " << load_config << std::endl;
        }

        if (transient < 0) transient = 8000LL * L * L;
        const int extremal_px_every = 200;
        model.setThresholds(x0s);
        model.prepareAvalancheRun(
            steps, extremal_px_every, stable_x0, avalanche_blocks);
        std::cout << "Discarding transient of " << transient << " steps ("
                  << (static_cast<double>(transient) / (L * L)) << " per site)..." << std::endl;
        if (!model.equilibrate(transient)) {
            std::cerr << "Extremal equilibration failed the divergence guard; "
                      << "measurements were skipped.\n";
            model.saveAvalanches("aval" + suffix);
            model.saveAvalancheBlocks("avalblocks" + suffix);
            model.saveDistributions("xdist" + suffix + ".dat");
            if (std::isfinite(stable_x0)) {
                model.saveEnergyGaps("egap" + suffix + ".dat");
            }
            model.saveConfiguration(config_file);
            model.saveStatistics(stats_file);
            return 2;
        }

        std::cout << "Accumulating avalanches over " << x0s.size()
                  << " thresholds..." << std::endl;
        const bool measurement_ok =
            model.runAvalanches(
                steps, extremal_px_every, stable_x0, avalanche_blocks);

        model.saveAvalanches("aval" + suffix);
        model.saveAvalancheBlocks("avalblocks" + suffix);
        model.saveDistributions("xdist" + suffix + ".dat");
        if (std::isfinite(stable_x0)) {
            model.saveEnergyGaps("egap" + suffix + ".dat");
        }
        model.saveConfiguration(config_file);
        model.saveStatistics(stats_file);
        if (!measurement_ok) {
            if (model.divergenceDetected()) {
                std::cerr << "Extremal measurement failed the divergence guard; "
                          << "partial outputs are marked measurement_completed 0.\n";
                return 2;
            }
            std::cerr << "Extremal measurement completed, but the requested "
                      << "stable-state sample was empty (failure_stage "
                      << model.failureStage() << ").\n";
            return 3;
        }
    } else if (algo == "mc") {
        MCTensorialModel model(L, T);
        model.setRandomSeed(random_seed);
        model.setProjectorKernel(projector_kernel);
        model.setDropRule(drop_rule);
        model.setKernelAmplitude(kernel_amplitude);
        std::cout << "Initializing system..." << std::endl;
        model.initialize();
        if (!load_config.empty()) {
            if (!model.loadConfiguration(load_config)) {
                std::cerr << "Failed to read the initial configuration; "
                          << "refusing to run from a random state instead.\n";
                return 1;
            }
            std::cout << "Initial state loaded from " << load_config << std::endl;
        }

        std::cout << "Running simulation..." << std::endl;
        model.runSimulation(steps);

        // tau_alpha in sweeps. The budget has to be in attempts, and the
        // acceptance rate is what makes this expensive at low temperature.
        if (measure_tau || tau_reps > 0) {
            int reps = (tau_reps > 0) ? tau_reps : 1;
            std::cout << "Measuring tau_alpha over " << reps << " origins..." << std::endl;
            std::vector<double> taus = model.measureRelaxationTimes(reps, steps);
            for (double t : taus) std::cout << "tau_rep " << t << "\n";
            std::cout << "tau_reps_completed " << taus.size() << "/" << reps << std::endl;
        }

        // The paper's physical-time estimator, shared with the Gillespie
        // driver. measure_tau is passed as false because the block above has
        // already served -tau for this driver; the pilot inside is taken
        // regardless whenever -analyze is requested.
        //
        // Note that "# equilibration_events" in the output files carries the
        // -steps value, which for Monte Carlo counts ATTEMPTS rather than
        // plastic events. The "# algorithm" line distinguishes the two.
        if (analyze_origins > 0) {
            // The Gillespie pilot budget is 400*L*L plastic events. The Monte
            // Carlo equivalent is that many events converted to attempts by the
            // measured acceptance -- not the same number, which would be a
            // fraction of one event at low temperature. It is only a cap: the
            // pilot returns as soon as the persistence halves.
            const double acceptance = model.getAcceptanceRate();
            const long long int pilot_budget =
                (acceptance > 0.0)
                    ? static_cast<long long int>(400.0 * L * L / acceptance)
                    : steps;
            std::cout << "MC pilot budget " << pilot_budget
                      << " attempts (acceptance " << acceptance << ")"
                      << std::endl;
            const int rc = runPaperAnalysis(model, "mc", L, T, random_seed,
                                            projector_kernel, drop_rule,
                                            literal_select, kernel_amplitude,
                                            steps, analyze_origins,
                                            analysis_center, px_samples,
                                            false, pilot_budget,
                                            suffix, config_file, stats_file);
            if (rc != 0) return rc;
        }

        model.saveConfiguration(config_file);
        model.saveStatistics(stats_file);
    } else {
        EDMDTensorialModel model(L, T);
        model.setRandomSeed(random_seed);
        model.setProjectorKernel(projector_kernel);
        model.setLiteralSelection(literal_select);
        model.setDropRule(drop_rule);
        model.setKernelAmplitude(kernel_amplitude);
        std::cout << "Initializing system..." << std::endl;
        model.initialize();
        if (!load_config.empty()) {
            if (!model.loadConfiguration(load_config)) {
                std::cerr << "Failed to read the initial configuration; "
                          << "refusing to run from a random state instead.\n";
                return 1;
            }
            std::cout << "Initial state loaded from " << load_config << std::endl;
        }

        // Equilibrate, then discard the transient as the paper does
        std::cout << "Equilibrating..." << std::endl;
        if (!model.runEvents(steps)) {
            std::cerr << "EDMD equilibration failed the divergence guard; "
                      << "measurements were skipped.\n";
            model.saveConfiguration(config_file);
            model.saveStatistics(stats_file);
            return 2;
        }

        if (tau_reps > 0) {
            std::cout << "Measuring diagnostic post-event half-times over "
                      << tau_reps << " origins..." << std::endl;
            std::vector<double> taus = model.measureRelaxationTimes(tau_reps, 400LL * L * L);
            for (double t : taus) std::cout << "tau_rep " << t << "\n";
            std::cout << "tau_reps_completed " << taus.size() << "/" << tau_reps << std::endl;
        }

        {
            const int rc = runPaperAnalysis(model, "edmd", L, T, random_seed,
                                            projector_kernel, drop_rule,
                                            literal_select, kernel_amplitude,
                                            steps, analyze_origins,
                                            analysis_center, px_samples,
                                            measure_tau, 400LL * L * L,
                                            suffix, config_file, stats_file);
            if (rc != 0) return rc;
        }

        model.saveConfiguration(config_file);
        model.saveStatistics(stats_file);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "Simulation completed in " << duration << " seconds." << std::endl;

    return 0;
}
