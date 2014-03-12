#include <memory>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

#include <gflags/gflags.h>
#include <boost/chrono/chrono.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/process_cpu_clocks.hpp>

DEFINE_int64(num_features, 1000, "");
DEFINE_int64(num_iterations, 5, "");


std::vector<double> generateLogisticData() {
  auto result = std::vector<double>(FLAGS_num_features);
  std::mt19937 gen(
      boost::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (auto& r : result) {
    r = dist(gen);
  }

  std::sort(result.begin(), result.end());

  std::uniform_real_distribution<double> uniform;
  for (auto& r : result) {
    r = (uniform(gen) < r) ? 1.0 : 0.0;
  }
  return result;
}

void isotonicRegression(std::vector<double>* yp,
                        const std::vector<double>& weights) {
  auto& y = *yp;
  auto n = y.size();
  if (n <= 1) {
    return;
  }

  n -= 1;
  while (true) {
    size_t i = 0;
    bool pooled = false;
    while (i < n) {
      size_t k = i;

      while (k < n && y[k] >= y[k+1]) {
        k += 1;
      }

      if (y[i] != y[k]) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t j = i; j <= k; j++) {
          numerator += y[j] * weights[j];
          denominator += weights[j];
        }
        for (size_t j = i; j <= k; j++) {
          y[j] = numerator / denominator;
        }
        pooled = true;
      }
      i = k +1;
    }
    if (!pooled) {
      break;
    }
  }
}

void runIsotonicRegression() {
  const auto features = generateLogisticData();
  const auto weights = std::vector<double>(FLAGS_num_features, 1.0);

  auto iteration = [&]() {
    const auto start = boost::chrono::process_real_cpu_clock::now();
    auto scratch = features;
    isotonicRegression(&scratch, weights);
    const auto end = boost::chrono::process_real_cpu_clock::now();
    const auto delta = boost::chrono::duration_cast<boost::chrono::nanoseconds>(
        end - start);
    return std::chrono::nanoseconds(delta.count());
  };

  std::vector<std::chrono::nanoseconds> results(FLAGS_num_iterations);
  for (auto& elem : results) {
    elem = iteration();
  }

  double totalSec = 0.0;
  for (const auto& elem : results) {
    totalSec += static_cast<double>(elem.count()) / 1e9;
  }
  std::cout << "Average time: " << totalSec / (results.size()) << std::endl;
}

bool test() {
  std::vector<double> ys = {1, 41, 51, 1, 2, 5, 24};
  std::vector<double> ws = {1, 2, 3, 4, 5, 6, 7};
  std::vector<double> expected = {1.0, 13.95, 13.95, 13.95, 13.95, 13.95, 24};
  isotonicRegression(&ys, ws);
  for (size_t i = 0; i < ys.size(); i++) {
    if (abs(ys[i] - expected[i]) > 0.01) {
      std::cerr << "bad test" << std::endl;
      std::exit(1);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  test();
  runIsotonicRegression();
  return 0;
}
