#include "tensorial_model.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace {

constexpr std::array<char, 8> KERNEL_CACHE_MAGIC = {'T', 'E', 'P', 'M', 'K', 'R', 'N', 'L'};
constexpr std::uint32_t KERNEL_CACHE_FORMAT_VERSION = 1;
constexpr std::uint32_t PAPER_KERNEL_FORMULA_VERSION = 1;
constexpr std::uint32_t PROJECTOR_KERNEL_FORMULA_VERSION = 1;
constexpr std::uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

std::uint32_t kernelFormulaVersion(bool projector_kernel) {
    return projector_kernel ? PROJECTOR_KERNEL_FORMULA_VERSION
                            : PAPER_KERNEL_FORMULA_VERSION;
}

std::uint64_t updateChecksum(std::uint64_t checksum, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; i++) {
        checksum ^= bytes[i];
        checksum *= FNV_PRIME;
    }
    return checksum;
}

std::uintmax_t expectedKernelCacheSize(int L) {
    constexpr std::uintmax_t header_bytes = KERNEL_CACHE_MAGIC.size()
        + 4 * sizeof(std::uint32_t);
    constexpr std::uintmax_t checksum_bytes = sizeof(std::uint64_t);
    return header_bytes + 3ULL * static_cast<std::uintmax_t>(L)
        * static_cast<std::uintmax_t>(L) * sizeof(double) + checksum_bytes;
}

}  // namespace

// Constructor
TensorialModel::TensorialModel(int size, double temperature)
    : L(size),
      T(temperature),
      sigma_Y(1.0),
      mu(1.0),
      K(1.0),
      z0(1.0),
      time(0.0),
      events(0),
      relaxed_count(0),
      persistence_origin(0.0),
      drop_rule(DropRule::Aligned),
      kernel_amplitude(1.0),
      projector_kernel(false),
      random_seed(0) {

    // Initialize random number generators
    std::random_device rd;
    random_seed = rd();
    rng.seed(random_seed);
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
    exp_dist = std::exponential_distribution<double>(1.0 / z0);

    // Allocate memory for system state
    int N = L * L;
    sigma_xx.resize(N, 0.0);
    sigma_xy.resize(N, 0.0);
    theta.resize(N, 0.0);
    sin_theta.resize(N, 0.0);
    cos_theta.resize(N, 1.0);

    // Allocate memory for Eshelby kernels
    G_xx_xx.resize(L, std::vector<double>(L, 0.0));
    G_xy_xy.resize(L, std::vector<double>(L, 0.0));
    G_xx_xy.resize(L, std::vector<double>(L, 0.0));

    relaxed.resize(N, 0);
}

// Book-keeping every driver does when a site undergoes a plastic event
void TensorialModel::recordEvent(int site) {
    events++;
    if (!relaxed[site]) {
        relaxed[site] = 1;
        relaxed_count++;
    }
}

void TensorialModel::resetPersistence() {
    std::fill(relaxed.begin(), relaxed.end(), 0);
    relaxed_count = 0;
    persistence_origin = time;
}

double TensorialModel::persistence() const {
    return 1.0 - static_cast<double>(relaxed_count) / (L * L);
}

// Diagnostic half-persistence time from one origin. The paper's tau_alpha is
// instead obtained from a physical-time-origin-averaged persistence curve.
double TensorialModel::measureRelaxationTime(long long int max_moves) {
    resetPersistence();

    for (long long int m = 0; m < max_moves; m++) {
        advance();
        if (persistence() <= 0.5) {
            return time - persistence_origin;
        }
    }

    return -1.0;  // did not relax within the budget
}

// Successive diagnostic half-times, one per post-move persistence origin.
std::vector<double> TensorialModel::measureRelaxationTimes(
        int reps, long long int max_moves) {
    std::vector<double> out;
    out.reserve(reps);

    for (int r = 0; r < reps; r++) {
        double t = measureRelaxationTime(max_moves);
        if (t > 0.0) out.push_back(t);
    }

    return out;
}

double TensorialModel::stressRms() const {
    int N = L * L;
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += sigma_xx[i] * sigma_xx[i] + sigma_xy[i] * sigma_xy[i];
    }
    return std::sqrt(sum / N);
}

// Save Eshelby kernels to file for future use
bool TensorialModel::saveEshelbyKernels(const std::string& filename) const {
    // Write a complete temporary file and rename it into place. A reader can
    // therefore never observe a partially written cache when jobs start in
    // parallel. The token only has to make concurrent temporary names distinct.
    std::random_device rd;
    const std::uint64_t token = (static_cast<std::uint64_t>(rd()) << 32)
                              ^ static_cast<std::uint64_t>(rd())
                              ^ static_cast<std::uint64_t>(
                                  std::chrono::high_resolution_clock::now()
                                      .time_since_epoch().count());
    const std::string temporary = filename + ".tmp." + std::to_string(token);
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open temporary kernel cache " << temporary
                  << " for writing." << std::endl;
        return false;
    }

    const std::uint32_t cache_version = KERNEL_CACHE_FORMAT_VERSION;
    const std::uint32_t formula_version = kernelFormulaVersion(projector_kernel);
    const std::uint32_t stored_L = static_cast<std::uint32_t>(L);
    const std::uint32_t stored_mode = projector_kernel ? 1U : 0U;

    file.write(KERNEL_CACHE_MAGIC.data(), KERNEL_CACHE_MAGIC.size());
    file.write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));
    file.write(reinterpret_cast<const char*>(&formula_version), sizeof(formula_version));
    file.write(reinterpret_cast<const char*>(&stored_L), sizeof(stored_L));
    file.write(reinterpret_cast<const char*>(&stored_mode), sizeof(stored_mode));

    // Write all kernel elements
    std::uint64_t checksum = FNV_OFFSET_BASIS;
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            const std::array<double, 3> values = {
                G_xx_xx[x][y], G_xy_xy[x][y], G_xx_xy[x][y]
            };
            file.write(reinterpret_cast<const char*>(values.data()),
                       values.size() * sizeof(double));
            checksum = updateChecksum(checksum, values.data(),
                                      values.size() * sizeof(double));
        }
    }
    file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    file.flush();

    const bool write_ok = file.good();
    file.close();
    if (!write_ok) {
        std::cerr << "Error: Failed while writing temporary kernel cache "
                  << temporary << std::endl;
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return false;
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary, filename, rename_error);
    if (rename_error) {
        std::cerr << "Error: Could not install kernel cache " << filename
                  << ": " << rename_error.message() << std::endl;
        std::error_code remove_error;
        std::filesystem::remove(temporary, remove_error);
        return false;
    }

    return true;
}

// Load Eshelby kernels from file
bool TensorialModel::loadEshelbyKernels(const std::string& filename) {
    std::error_code size_error;
    const std::uintmax_t actual_size = std::filesystem::file_size(filename, size_error);
    if (size_error) {
        std::cout << "Eshelby kernel file " << filename << " not found." << std::endl;
        return false;
    }

    const std::uintmax_t expected_size = expectedKernelCacheSize(L);
    if (actual_size != expected_size) {
        std::cout << "Kernel cache " << filename << " has obsolete or invalid size "
                  << actual_size << " (expected " << expected_size << ")." << std::endl;
        return false;
    }

    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Eshelby kernel file " << filename << " not found." << std::endl;
        return false;
    }

    std::array<char, KERNEL_CACHE_MAGIC.size()> magic{};
    std::uint32_t cache_version = 0;
    std::uint32_t formula_version = 0;
    std::uint32_t stored_L = 0;
    std::uint32_t stored_mode = 0;
    file.read(magic.data(), magic.size());
    file.read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
    file.read(reinterpret_cast<char*>(&formula_version), sizeof(formula_version));
    file.read(reinterpret_cast<char*>(&stored_L), sizeof(stored_L));
    file.read(reinterpret_cast<char*>(&stored_mode), sizeof(stored_mode));

    const std::uint32_t expected_mode = projector_kernel ? 1U : 0U;
    if (!file || magic != KERNEL_CACHE_MAGIC
              || cache_version != KERNEL_CACHE_FORMAT_VERSION
              || formula_version != kernelFormulaVersion(projector_kernel)
              || stored_L != static_cast<std::uint32_t>(L)
              || stored_mode != expected_mode) {
        std::cout << "Kernel cache " << filename
                  << " has an incompatible header." << std::endl;
        return false;
    }

    // Read into temporary arrays so a failed cache never leaves a partially
    // initialized kernel behind.
    auto loaded_xx_xx = G_xx_xx;
    auto loaded_xy_xy = G_xy_xy;
    auto loaded_xx_xy = G_xx_xy;
    std::uint64_t checksum = FNV_OFFSET_BASIS;
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            std::array<double, 3> values{};
            file.read(reinterpret_cast<char*>(values.data()),
                      values.size() * sizeof(double));
            if (!file || !std::isfinite(values[0]) || !std::isfinite(values[1])
                      || !std::isfinite(values[2])) {
                std::cout << "Kernel cache " << filename
                          << " contains truncated or non-finite data." << std::endl;
                return false;
            }
            checksum = updateChecksum(checksum, values.data(),
                                      values.size() * sizeof(double));
            loaded_xx_xx[x][y] = values[0];
            loaded_xy_xy[x][y] = values[1];
            loaded_xx_xy[x][y] = values[2];
        }
    }

    std::uint64_t stored_checksum = 0;
    file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(stored_checksum));
    if (!file || stored_checksum != checksum) {
        std::cout << "Kernel cache " << filename << " failed its checksum." << std::endl;
        return false;
    }

    file.close();
    G_xx_xx = std::move(loaded_xx_xx);
    G_xy_xy = std::move(loaded_xy_xy);
    G_xx_xy = std::move(loaded_xx_xy);
    std::cout << "Eshelby kernels loaded from " << filename << std::endl;
    return true;
}


// Initialize the system
void TensorialModel::initialize() {
    int N = L * L;

    // First, try to load Eshelby kernels from file or calculate them
    std::string kernel_filename = "eshelby_kernel_L" + std::to_string(L)
                                + (projector_kernel ? "_proj" : "") + ".dat";
    if (!loadEshelbyKernels(kernel_filename)) {
        // If loading fails, calculate and save for future use
        std::cout << "Calculating Eshelby kernels..." << std::endl;
        calculateEshelbyKernels();
        if (saveEshelbyKernels(kernel_filename)) {
            std::cout << "Eshelby kernels saved to " << kernel_filename << std::endl;
        }
    }

    // Scale the off-site kernel, if asked. Applied after the cache is read so
    // that one cache file per L serves every amplitude.
    if (kernel_amplitude != 1.0) {
        for (int x = 0; x < L; x++) {
            for (int y = 0; y < L; y++) {
                G_xx_xx[x][y] *= kernel_amplitude;
                G_xy_xy[x][y] *= kernel_amplitude;
                G_xx_xy[x][y] *= kernel_amplitude;
            }
        }
        std::cout << "Off-site kernel scaled by " << kernel_amplitude << std::endl;
    }

    // Initialize angles uniformly in [0, 2π)
    for (int i = 0; i < N; i++) {
        theta[i] = 2.0 * M_PI * uniform_dist(rng);
        sin_theta[i] = std::sin(theta[i]);
        cos_theta[i] = std::cos(theta[i]);
    }

    // Draw the local stresses at random, as in Appendix A of the paper. The
    // transient is discarded anyway, so only the force balance below matters.
    std::normal_distribution<double> normal_dist(0.0, 0.3);
    for (int i = 0; i < N; i++) {
        sigma_xx[i] = normal_dist(rng);
        sigma_xy[i] = normal_dist(rng);
    }

    // Force balance: the macroscopic stress of a quiescent liquid is zero
    projectOutMeanStress();
}

// Remove the q = 0 component of the stress field
void TensorialModel::projectOutMeanStress() {
    int N = L * L;

    double sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < N; i++) {
        sum_xx += sigma_xx[i];
        sum_xy += sigma_xy[i];
    }

    double adjust_xx = sum_xx / N;
    double adjust_xy = sum_xy / N;
    for (int i = 0; i < N; i++) {
        sigma_xx[i] -= adjust_xx;
        sigma_xy[i] -= adjust_xy;
    }
}

// Relaxation rate of a site (tau_0 = 1), with E(x) = x^(3/2)
double TensorialModel::siteRate(int site) {
    double x = calculateDistanceToYield(site);
    if (x <= 0.0) {
        return 1.0;  // already unstable: relaxes at the maximal rate 1/tau_0
    }
    return std::exp(-x * std::sqrt(x) / T);  // x^(3/2), avoiding pow in the hot loop
}

// Stress drop of a plastic event. The yielding plane has normal
// e = [sin θ, -cos θ]; the drop is taken along that normal, with the single
// sign of σ·e, so that the distance to yield after the event is exactly z.
void TensorialModel::computeStressDrop(int site, double x, double z,
                                       double& delta_sigma_xx,
                                       double& delta_sigma_xy) const {
    if (drop_rule == DropRule::PaperSigned) {
        // The paper's drop direction, but never pointing outwards. The drop is
        // still not (z - x) along the yield normal, so the distance to yield
        // after the event is not z -- that is what the direction costs.
        double sgn_xx = (sigma_xx[site] >= 0.0) ? 1.0 : -1.0;
        double sgn_xy = (sigma_xy[site] >= 0.0) ? 1.0 : -1.0;

        double ux = sin_theta[site] * sgn_xx;
        double uy = cos_theta[site] * sgn_xy;

        double ex = sin_theta[site];
        double ey = -cos_theta[site];

        double projection = sigma_xx[site] * ex + sigma_xy[site] * ey;
        double along = ux * ex + uy * ey;           // how much of u lies along e
        double sign = ((projection >= 0.0) == (along >= 0.0)) ? 1.0 : -1.0;

        delta_sigma_xx = (z - x) * sign * ux;
        delta_sigma_xy = (z - x) * sign * uy;
        return;
    }

    if (drop_rule == DropRule::Paper) {
        // PRX Eq. (A2)/(A3), verbatim:
        //   d_sigma_xx = -(z - x) sin(theta) sgn(sigma_xx)
        //   d_sigma_xy = -(z - x) cos(theta) sgn(sigma_xy)
        // applied by the caller as sigma -> sigma - d_sigma. Note the second
        // component carries +cos(theta), where the yield normal used for x in
        // Eq. (43) is e = (sin theta, -cos theta).
        double sgn_xx = (sigma_xx[site] >= 0.0) ? 1.0 : -1.0;
        double sgn_xy = (sigma_xy[site] >= 0.0) ? 1.0 : -1.0;

        delta_sigma_xx = -(z - x) * sin_theta[site] * sgn_xx;
        delta_sigma_xy = -(z - x) * cos_theta[site] * sgn_xy;
        return;
    }

    double ex = sin_theta[site];
    double ey = -cos_theta[site];

    double projection = sigma_xx[site] * ex + sigma_xy[site] * ey;
    double sign = (projection >= 0.0) ? 1.0 : -1.0;

    delta_sigma_xx = (z - x) * sign * ex;
    delta_sigma_xy = (z - x) * sign * ey;
}

// Calculate Eshelby kernels in real space
void TensorialModel::calculateEshelbyKernels() {
    // We need to calculate the Eshelby kernels in Fourier space first
    // Then transform them to real space

    std::vector<std::vector<std::complex<double>>> G_xx_xx_q(L, std::vector<std::complex<double>>(L));
    std::vector<std::vector<std::complex<double>>> G_xy_xy_q(L, std::vector<std::complex<double>>(L));
    std::vector<std::vector<std::complex<double>>> G_xx_xy_q(L, std::vector<std::complex<double>>(L));

    // Calculate kernels in Fourier space
    // Note: (mx, my) are storage indices in [0, L); (nx, ny) are the physical
    // wavevector integers in [-L/2+1, L/2] and must never be used as indices.
    for (int mx = 0; mx < L; mx++) {
        int nx = mx - L/2 + 1;
        for (int my = 0; my < L; my++) {
            int ny = my - L/2 + 1;
            // Skip q = 0 (set to 0 as per Eq. 17)
            if (nx == 0 && ny == 0) {
                G_xx_xx_q[mx][my] = 0.0;
                G_xy_xy_q[mx][my] = 0.0;
                G_xx_xy_q[mx][my] = 0.0;
                continue;
            }

            // Calculate q vectors according to Eq. 12
            double qx2 = 2.0 - 2.0 * cos(2.0 * M_PI * nx / L);
            double qy2 = 2.0 - 2.0 * cos(2.0 * M_PI * ny / L);
            // q_x q_y. The paper prints 2 sin(2pi nx/L) sin(2pi ny/L), which is
            // not the square root of the q_x^2, q_y^2 it prints on the same line:
            // q_alpha^2 = 2 - 2 cos(2pi n/L) = (2 sin(pi n/L))^2. The mismatch is
            // a factor 2 cos(pi nx/L) cos(pi ny/L), so the printed form is twice
            // too large at long wavelength. It costs the kernel the rank-one
            // property Eq. (A4)-(A6) have in the continuum: G(q) picks up a
            // positive eigenvalue, i.e. a channel in which a plastic event
            // amplifies stress rather than relaxing it.
            // For even L the signed square root is not Hermitian on the
            // self-conjugate Nyquist lines. Taking the real transform below
            // zeros those mixed modes, leaving a non-amplifying but rank-two
            // response there. This diagnostic convention is documented in
            // readme.md; the paper mode is unaffected.
            double qxqy = projector_kernel
                ? 4.0 * sin(M_PI * nx / L) * sin(M_PI * ny / L)
                : 2.0 * sin(2.0 * M_PI * nx / L) * sin(2.0 * M_PI * ny / L);
            double q2 = qx2 + qy2;

            // Avoid division by zero
            if (q2 < 1e-10) {
                G_xx_xx_q[mx][my] = 0.0;
                G_xy_xy_q[mx][my] = 0.0;
                G_xx_xy_q[mx][my] = 0.0;
                continue;
            }

            // Calculate kernels using Eq. 16
            G_xx_xx_q[mx][my] = -std::pow((qx2 - qy2), 2) / std::pow(q2, 2);
            G_xy_xy_q[mx][my] = -4.0 * qx2 * qy2 / std::pow(q2, 2);
            G_xx_xy_q[mx][my] = -2.0 * qxqy * (qx2 - qy2) / std::pow(q2, 2);
        }
    }

    // Transform to real space using Eq. 15
    double normalization = 1.0 / (L * L);
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            std::complex<double> sum_xx_xx(0.0, 0.0);
            std::complex<double> sum_xy_xy(0.0, 0.0);
            std::complex<double> sum_xx_xy(0.0, 0.0);

            for (int mx = 0; mx < L; mx++) {
                for (int my = 0; my < L; my++) {
                    int nx = mx - L/2 + 1;
                    int ny = my - L/2 + 1;
                    double phase = 2.0 * M_PI * (nx * x + ny * y) / L;
                    std::complex<double> exp_factor(cos(phase), sin(phase));

                    sum_xx_xx += G_xx_xx_q[mx][my] * exp_factor;
                    sum_xy_xy += G_xy_xy_q[mx][my] * exp_factor;
                    sum_xx_xy += G_xx_xy_q[mx][my] * exp_factor;
                }
            }

            G_xx_xx[x][y] = normalization * sum_xx_xx.real();
            G_xy_xy[x][y] = normalization * sum_xy_xy.real();
            G_xx_xy[x][y] = normalization * sum_xx_xy.real();
        }
    }
}

// Calculate distance to yield for a given site
double TensorialModel::calculateDistanceToYield(int site) const {
    double sx = sigma_xx[site];
    double sy = sigma_xy[site];
    // Unit vector perpendicular to the yielding plane
    double ex = sin_theta[site];
    double ey = -cos_theta[site];

    // Calculate distance using Eq. 25
    double dot_product = sx * ex + sy * ey;
    double distance = 2.0 * sigma_Y - std::max(std::abs(dot_product - sigma_Y), std::abs(dot_product + sigma_Y));

    return distance;
}

// Apply stress drop and propagate to all sites
void TensorialModel::applyStressDrop(int site, double delta_sigma_xx, double delta_sigma_xy) {
    int N = L * L;

    // Apply stress drop to the chosen site
    sigma_xx[site] -= delta_sigma_xx;
    sigma_xy[site] -= delta_sigma_xy;

    // Calculate position of the site
    int site_x = site % L;
    int site_y = site / L;

    // Propagate stress to all other sites using Eshelby kernel
    for (int j = 0; j < N; j++) {
        if (j == site) continue;

        int j_x = j % L;
        int j_y = j / L;

        // Calculate relative position with periodic boundary conditions
        int dx = (j_x - site_x + L ) % L;
        int dy = (j_y - site_y + L ) % L;

        // Apply Eshelby propagation
        sigma_xx[j] += G_xx_xx[dx][dy] * delta_sigma_xx + G_xx_xy[dx][dy] * delta_sigma_xy;
        sigma_xy[j] += G_xx_xy[dx][dy] * delta_sigma_xx + G_xy_xy[dx][dy] * delta_sigma_xy;
    }

    // Generate new random angle for the site
    theta[site] = 2.0 * M_PI * uniform_dist(rng);
    sin_theta[site] = std::sin(theta[site]);
    cos_theta[site] = std::cos(theta[site]);

    // The site is given the full drop while the kernel only redistributes
    // sum_{r != 0} G(r) = -G(0) of it, so each event leaks a little
    // macroscopic stress. Project it out to keep the total stress at zero.
    projectOutMeanStress();
}

// Save system configuration to file
bool TensorialModel::loadConfiguration(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open configuration " << filename << std::endl;
        return false;
    }

    const int N = L * L;
    std::vector<double> new_xx(N), new_xy(N), new_theta(N);
    std::vector<char> seen(N, 0);
    int rows = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        int x, y;
        double sxx, sxy, th;
        if (!(in >> x >> y >> sxx >> sxy >> th)) {
            std::cerr << "Error: malformed row in " << filename << ": " << line << std::endl;
            return false;
        }
        if (x < 0 || x >= L || y < 0 || y >= L) {
            std::cerr << "Error: site (" << x << "," << y << ") lies outside L="
                      << L << " in " << filename << std::endl;
            return false;
        }
        // saveConfiguration() writes x = i % L, y = i / L.
        const int i = y * L + x;
        if (seen[i]) {
            std::cerr << "Error: site (" << x << "," << y << ") appears twice in "
                      << filename << std::endl;
            return false;
        }
        seen[i] = 1;
        new_xx[i] = sxx;
        new_xy[i] = sxy;
        new_theta[i] = th;
        rows++;
    }

    if (rows != N) {
        std::cerr << "Error: " << filename << " describes " << rows
                  << " sites, expected " << N << std::endl;
        return false;
    }

    for (int i = 0; i < N; i++) {
        sigma_xx[i] = new_xx[i];
        sigma_xy[i] = new_xy[i];
        theta[i] = new_theta[i];
        sin_theta[i] = std::sin(theta[i]);
        cos_theta[i] = std::cos(theta[i]);
    }

    // Six-digit round-tripping leaves a small residual mean stress. Force
    // balance is an invariant that applyStressDrop() relies on, so restore it
    // exactly as initialize() does for a freshly drawn state.
    projectOutMeanStress();
    return true;
}

void TensorialModel::saveConfiguration(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    const char* drop_name = (drop_rule == DropRule::Paper) ? "paper"
                          : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                 : "aligned";
    file << "# System size: " << L << "x" << L << "\n";
    file << "# Temperature: " << T << "\n";
    file << "# RNG seed: " << random_seed << "\n";
    file << "# Kernel mode: " << (projector_kernel ? "proj" : "paper") << "\n";
    file << "# Stress drop: " << drop_name << "\n";
    file << "# Kernel amplitude: " << kernel_amplitude << "\n";
    file << "# x y sigma_xx sigma_xy theta\n";

    int N = L * L;
    for (int i = 0; i < N; i++) {
        int x = i % L;
        int y = i / L;

        file << x << " " << y << " "
             << sigma_xx[i] << " " << sigma_xy[i] << " " << theta[i] << "\n";
    }

    file.close();
    std::cout << "Configuration saved to " << filename << std::endl;
}

// Save simulation statistics to file
void TensorialModel::saveStatistics(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Calculate various statistics
    double avg_sigma_xx = 0.0, avg_sigma_xy = 0.0;
    double var_sigma_xx = 0.0, var_sigma_xy = 0.0;
    int N = L * L;

    for (int i = 0; i < N; i++) {
        avg_sigma_xx += sigma_xx[i];
        avg_sigma_xy += sigma_xy[i];
    }

    avg_sigma_xx /= N;
    avg_sigma_xy /= N;

    for (int i = 0; i < N; i++) {
        var_sigma_xx += std::pow(sigma_xx[i] - avg_sigma_xx, 2);
        var_sigma_xy += std::pow(sigma_xy[i] - avg_sigma_xy, 2);
    }

    var_sigma_xx /= N;
    var_sigma_xy /= N;

    const char* drop_name = (drop_rule == DropRule::Paper) ? "paper"
                          : (drop_rule == DropRule::PaperSigned) ? "papersgn"
                                                                 : "aligned";

    // Write statistics to file
    file << "# System size: " << L << "x" << L << "\n";
    file << "# Temperature: " << T << "\n";
    file << "# RNG seed: " << random_seed << "\n";
    file << "# Kernel mode: " << (projector_kernel ? "proj" : "paper") << "\n";
    file << "# Stress drop: " << drop_name << "\n";
    file << "# Kernel amplitude: " << kernel_amplitude << "\n";
    file << "# Average sigma_xx: " << avg_sigma_xx << "\n";
    file << "# Average sigma_xy: " << avg_sigma_xy << "\n";
    file << "# Variance sigma_xx: " << var_sigma_xx << "\n";
    file << "# Variance sigma_xy: " << var_sigma_xy << "\n";

    file.close();
    std::cout << "Statistics saved to " << filename << std::endl;
}

// Measure spatial correlations of stress
void TensorialModel::measureCorrelations() {
    // This is a placeholder for more complex correlation measurements
    // which would be implemented based on specific research questions
    std::cout << "Correlation measurement not yet implemented." << std::endl;
}

// ---------------------------------------------------------------------------
// The paper's physical-time estimators.
//
// These were written for the Gillespie driver and lived on EDMDTensorialModel.
// They depend on nothing specific to it: stepUntil() and runUntil() are the
// only primitives they use, so they are defined here and both drivers share
// one implementation rather than two that could drift apart.
// ---------------------------------------------------------------------------

// Run until the simulation time has advanced exactly by duration
void TensorialModel::runUntil(double duration) {
    const double target = time + duration;
    while (stepUntil(target) >= 0) {}
}

// <pi(t)> and chi_4(t), averaged over regular physical-time origins. Origins
// can be correlated; statistical errors require a block analysis.
//
// For each origin we record the time at which every site first relaxes; the
// persistence at time t is then just the fraction of sites whose first
// relaxation lies beyond t, which can be evaluated on any grid afterwards.
void TensorialModel::measureDynamics(int num_origins,
                                 const std::vector<double>& times,
                                 double origin_spacing,
                                 std::vector<double>& pi_mean,
                                 std::vector<double>& chi4,
                                 std::vector<std::vector<double>>* pi_by_origin) {
    int N = L * L;
    size_t nt = times.size();
    double t_max = times.back();

    pi_mean.assign(nt, 0.0);
    std::vector<double> pi_sq(nt, 0.0);
    if (pi_by_origin != nullptr) {
        pi_by_origin->assign(num_origins, std::vector<double>(nt, 0.0));
    }

    // Equilibration ends immediately after an event, which is a sample of the
    // embedded jump chain rather than a physical-time sample. Advance one full
    // observation window and stop exactly between events before taking the
    // first origin.
    runUntil(t_max);

    // Record one trajectory for all overlapping origins. A fixed physical-time
    // grid is a time average; overlap changes covariance, not its expectation.
    const double first_origin = time;
    const double last_origin = first_origin + (num_origins - 1) * origin_spacing;
    const double trajectory_end = last_origin + t_max;
    std::vector<std::vector<double>> event_times(N);
    while (true) {
        int site = stepUntil(trajectory_end);
        if (site < 0) break;
        event_times[site].push_back(time);
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> first_relax(N);
    for (int origin = 0; origin < num_origins; origin++) {
        const double t0 = first_origin + origin * origin_spacing;
        for (int i = 0; i < N; i++) {
            const auto next = std::upper_bound(event_times[i].begin(),
                                       event_times[i].end(), t0);
            first_relax[i] = (next == event_times[i].end())
                           ? infinity : *next - t0;
        }

        // pi(t) for this origin, then accumulate the first two moments
        for (size_t k = 0; k < nt; k++) {
            int still = 0;
            for (int i = 0; i < N; i++) {
                if (first_relax[i] > times[k]) still++;
            }
            double pi = static_cast<double>(still) / N;
            pi_mean[k] += pi;
            pi_sq[k] += pi * pi;
            if (pi_by_origin != nullptr) {
                (*pi_by_origin)[origin][k] = pi;
            }
        }

        if ((origin + 1) % 20 == 0) {
            std::cout << "  origin " << (origin + 1) << "/" << num_origins << std::endl;
        }
    }

    chi4.assign(nt, 0.0);
    for (size_t k = 0; k < nt; k++) {
        pi_mean[k] /= num_origins;
        pi_sq[k] /= num_origins;
        chi4[k] = N * (pi_sq[k] - pi_mean[k] * pi_mean[k]);
    }
}

// Steady-state distribution of the distance to yield, on [0, sigma_Y], sampled
// on a physical-time grid rather than every fixed number of events.
void TensorialModel::measurePx(int num_samples, double sample_interval,
                               std::vector<double>& px) {
    int N = L * L;
    int bins = static_cast<int>(px.size());
    std::fill(px.begin(), px.end(), 0.0);

    double total = 0.0;
    for (int sample = 0; sample < num_samples; sample++) {
        runUntil(sample_interval);

        for (int i = 0; i < N; i++) {
            double x = calculateDistanceToYield(i);
            if (x >= 0.0 && x < sigma_Y) {
                px[static_cast<int>(x / sigma_Y * bins)] += 1.0;
                total += 1.0;
            }
        }

        if ((sample + 1) % 200 == 0) {
            std::cout << "  P(x) sample " << (sample + 1) << "/"
                      << num_samples << std::endl;
        }
    }

    // normalize to a probability density
    if (total > 0.0) {
        for (int b = 0; b < bins; b++) {
            px[b] *= bins / (total * sigma_Y);
        }
    }
}
