#include "tensorial_model.h"

// Constructor
TensorialModel::TensorialModel(int size, double temperature)
    : L(size),
      T(temperature),
      sigma_Y(1.0),
      mu(1.0),
      K(1.0),
      z0(1.0) {

    // Initialize random number generators
    std::random_device rd;
    rng.seed(rd());
    uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
    exp_dist = std::exponential_distribution<double>(1.0 / z0);

    // Allocate memory for system state
    int N = L * L;
    sigma_xx.resize(N, 0.0);
    sigma_xy.resize(N, 0.0);
    theta.resize(N, 0.0);

    // Allocate memory for Eshelby kernels
    G_xx_xx.resize(L, std::vector<double>(L, 0.0));
    G_xy_xy.resize(L, std::vector<double>(L, 0.0));
    G_xx_xy.resize(L, std::vector<double>(L, 0.0));
}

// Save Eshelby kernels to file for future use
void TensorialModel::saveEshelbyKernels(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
}

    // Write system size
    file.write(reinterpret_cast<const char*>(&L), sizeof(L));

    // Write all kernel elements
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            double g_xx_xx = G_xx_xx[x][y];
            double g_xy_xy = G_xy_xy[x][y];
            double g_xx_xy = G_xx_xy[x][y];

            file.write(reinterpret_cast<const char*>(&g_xx_xx), sizeof(g_xx_xx));
            file.write(reinterpret_cast<const char*>(&g_xy_xy), sizeof(g_xy_xy));
            file.write(reinterpret_cast<const char*>(&g_xx_xy), sizeof(g_xx_xy));
        }
    }

    file.close();
}

// Load Eshelby kernels from file
bool TensorialModel::loadEshelbyKernels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cout << "Eshelby kernel file " << filename << " not found." << std::endl;
        return false;
    }

    // Read system size
    int file_L;
    file.read(reinterpret_cast<char*>(&file_L), sizeof(file_L));

    // Check if the file's system size matches our system size
    if (file_L != L) {
        std::cout << "Mismatch in system size in kernel file: expected " << L
                  << ", found " << file_L << std::endl;
        file.close();
        return false;
    }

    // Read all kernel elements
    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            double g_xx_xx, g_xy_xy, g_xx_xy;

            file.read(reinterpret_cast<char*>(&g_xx_xx), sizeof(g_xx_xx));
            file.read(reinterpret_cast<char*>(&g_xy_xy), sizeof(g_xy_xy));
            file.read(reinterpret_cast<char*>(&g_xx_xy), sizeof(g_xx_xy));

            G_xx_xx[x][y] = g_xx_xx;
            G_xy_xy[x][y] = g_xy_xy;
            G_xx_xy[x][y] = g_xx_xy;
        }
    }

    file.close();
    std::cout << "Eshelby kernels loaded from " << filename << std::endl;
    return true;
}


// Initialize the system
void TensorialModel::initialize() {
    int N = L * L;

    // First, try to load Eshelby kernels from file or calculate them
    std::string kernel_filename = "eshelby_kernel_L" + std::to_string(L) + ".dat";
    if (!loadEshelbyKernels(kernel_filename)) {
        // If loading fails, calculate and save for future use
        std::cout << "Calculating Eshelby kernels..." << std::endl;
        calculateEshelbyKernels();
        saveEshelbyKernels(kernel_filename);
        std::cout << "Eshelby kernels saved to " << kernel_filename << std::endl;
    }

    // Initialize angles uniformly in [0, 2π)
    for (int i = 0; i < N; i++) {
        theta[i] = 2.0 * M_PI * uniform_dist(rng);
    }

    // Initialize stresses according to the equilibrium stress distribution
    // and ensuring force balance
    for (int i = 0; i < N; i++) {
        // Draw distance to yield from equilibrium distribution
        // p(x) ∝ exp[-(σ_Y - x)^(3/2)/T] with x < σ_Y
        // We'll use rejection sampling for simplicity
        double x_i;
        do {
            x_i = sigma_Y * uniform_dist(rng);  // x_i in [0, sigma_Y]
        } while (uniform_dist(rng) > std::exp(-std::pow(sigma_Y - x_i, 1.5) / T));

        // Set direction according to theta
        // [σ_xx,i, σ_xy,i] = ±(σ_Y - x_i)[sin θ_i, -cos θ_i]
        double sign = (uniform_dist(rng) < 0.5) ? 1.0 : -1.0;
        double scale = sign * (sigma_Y - x_i);

        sigma_xx[i] = scale * std::sin(theta[i]);
        sigma_xy[i] = -scale * std::cos(theta[i]);
    }

    // Force balance: ensure sum of stresses is zero
    double sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < N; i++) {
        sum_xx += sigma_xx[i];
        sum_xy += sigma_xy[i];
    }

    // Adjust each site to ensure zero sum
    double adjust_xx = sum_xx / N;
    double adjust_xy = sum_xy / N;
    for (int i = 0; i < N; i++) {
        sigma_xx[i] -= adjust_xx;
        sigma_xy[i] -= adjust_xy;
    }
}

// Calculate Eshelby kernels in real space
void TensorialModel::calculateEshelbyKernels() {
    // We need to calculate the Eshelby kernels in Fourier space first
    // Then transform them to real space

    std::vector<std::vector<std::complex<double>>> G_xx_xx_q(L, std::vector<std::complex<double>>(L));
    std::vector<std::vector<std::complex<double>>> G_xy_xy_q(L, std::vector<std::complex<double>>(L));
    std::vector<std::vector<std::complex<double>>> G_xx_xy_q(L, std::vector<std::complex<double>>(L));

    // Calculate kernels in Fourier space
    for (int mx = 0; mx < L; mx++) {
            int nx = mx - L/2 + 1;
        for (int my = 0; my < L; my++) {
            int ny = my - L/2 + 1;
            // Skip q = 0 (set to 0 as per Eq. 17)
            if (nx == 0 && ny == 0) {
                G_xx_xx_q[nx][ny] = 0.0;
                G_xy_xy_q[nx][ny] = 0.0;
                G_xx_xy_q[nx][ny] = 0.0;
                continue;
            }

            // Calculate q vectors according to Eq. 12
            double qx2 = 2.0 - 2.0 * cos(2.0 * M_PI * nx / L);
            double qy2 = 2.0 - 2.0 * cos(2.0 * M_PI * ny / L);
            double qxqy = 2.0 * sin(2.0 * M_PI * nx / L) * sin(2.0 * M_PI * ny / L);
            double q2 = qx2 + qy2;

            // Avoid division by zero
            if (q2 < 1e-10) {
                G_xx_xx_q[nx][ny] = 0.0;
                G_xy_xy_q[nx][ny] = 0.0;
                G_xx_xy_q[nx][ny] = 0.0;
                continue;
            }

            // Calculate kernels using Eq. 16
            G_xx_xx_q[nx][ny] = -std::pow((qx2 - qy2), 2) / std::pow(q2, 2);
            G_xy_xy_q[nx][ny] = -4.0 * qx2 * qy2 / std::pow(q2, 2);
            G_xx_xy_q[nx][ny] = -2.0 * qxqy * (qx2 - qy2) / std::pow(q2, 2);
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

                    sum_xx_xx += G_xx_xx_q[nx][ny] * exp_factor;
                    sum_xy_xy += G_xy_xy_q[nx][ny] * exp_factor;
                    sum_xx_xy += G_xx_xy_q[nx][ny] * exp_factor;
                }
            }

            G_xx_xx[x][y] = normalization * sum_xx_xx.real();
            G_xy_xy[x][y] = normalization * sum_xy_xy.real();
            G_xx_xy[x][y] = normalization * sum_xx_xy.real();
        }
    }
}

// Calculate distance to yield for a given site
double TensorialModel::calculateDistanceToYield(int site) {
    double sx = sigma_xx[site];
    double sy = sigma_xy[site];
    double th = theta[site];

    // Unit vector perpendicular to the yielding plane
    double ex = sin(th);
    double ey = -cos(th);

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
}

// Save system configuration to file
void TensorialModel::saveConfiguration(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

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

    // Write statistics to file
    file << "# System size: " << L << "x" << L << "\n";
    file << "# Temperature: " << T << "\n";
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
