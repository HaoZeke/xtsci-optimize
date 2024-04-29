#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>
#include <limits>
#include <random>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnoalias.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace minimize {
template <typename ScalarType> struct Particle {
  xt::xarray<ScalarType> position;
  xt::xarray<ScalarType> velocity;
  xt::xarray<ScalarType> best_position;
  ScalarType best_value;
};

template <typename ScalarType> class PSOptim {
private:
  std::vector<Particle<ScalarType>> swarm;
  xt::xarray<ScalarType> gbest_position; // global best position
  ScalarType gbest_value;                // global best value
  ScalarType prev_gbest_value;
  size_t num_particles;
  ScalarType inertia_weight;
  ScalarType cognitive_comp; // cognitive component
  ScalarType social_comp;    // social component
  std::mt19937 rng;          // Mersenne Twister random number generator
  OptimizeControl<ScalarType> m_control;

public:
  PSOptim(size_t num_particles = 10, ScalarType inertia = 0.5,
          ScalarType cognitive_comp = 1.5, ScalarType social_comp = 1.5,
          OptimizeControl<ScalarType> control = OptimizeControl<ScalarType>())
      : num_particles(num_particles), inertia_weight(inertia),
        cognitive_comp(cognitive_comp), social_comp(social_comp),
        m_control(control) {
    std::random_device rd;
    rng.seed(rd()); // Seed the generator
    prev_gbest_value = std::numeric_limits<ScalarType>::infinity();
  }

  void initialize_swarm(const FObjFunc &func,
                        const xt::xarray<ScalarType> &lower_bound,
                        const xt::xarray<ScalarType> &upper_bound) {
    for (size_t idx = 0; idx < num_particles; ++idx) {
      Particle<ScalarType> particle;

      // Randomly initialize position and velocity
      particle.position = random_position(lower_bound, upper_bound);
      particle.velocity = random_velocity(lower_bound, upper_bound);
      particle.best_position = particle.position;
      particle.best_value = func(particle.position);

      swarm.push_back(particle);

      // Update global best if needed
      if (idx == 0 || particle.best_value < gbest_value) {
        gbest_position = particle.best_position;
        gbest_value = particle.best_value;
      }
    }
  }

  void update_swarm(const FObjFunc &func,
                    const xt::xarray<ScalarType> &lower_bound,
                    const xt::xarray<ScalarType> &upper_bound) {
    size_t idx = 0;
    ScalarType Vmax = 0.5 * xt::linalg::norm(upper_bound - lower_bound);
    for (auto &particle : swarm) {
      // Update velocity
      particle.velocity =
          inertia_weight * particle.velocity +
          cognitive_comp * random_factor() *
              (particle.best_position - particle.position) +
          social_comp * random_factor() * (gbest_position - particle.position);

      // Velocity clamping
      particle.velocity = xt::clip(particle.velocity, -Vmax, Vmax);

      // Update position
      particle.position += particle.velocity;

      // Boundary conditions
      particle.position = xt::clip(particle.position, lower_bound, upper_bound);

      // Reflective boundary
      xt::xarray<bool> lower_violation = particle.position == lower_bound;
      xt::xarray<bool> upper_violation = particle.position == upper_bound;
      particle.velocity = xt::where(lower_violation || upper_violation,
                                    -particle.velocity, particle.velocity);

      // Evaluate new position
      ScalarType new_value = func(particle.position);
      if (m_control.verbose) {
        fmt::print("New position for {}: {}\n", idx, particle.position);
      }
      if (new_value < particle.best_value) {
        particle.best_value = new_value;
        particle.best_position = particle.position;
        // Update global best if needed
        if (new_value < gbest_value) {
          gbest_position = particle.best_position;
          gbest_value = new_value;
        }
      }
      idx++;
    }
  }

  xts::optimize::OptimizeResult<ScalarType>
  optimize(const FObjFunc &func, const xt::xarray<ScalarType> &lower_bound,
           const xt::xarray<ScalarType> &upper_bound) {
    initialize_swarm(func, lower_bound, upper_bound);

    size_t iteration = 0;
    size_t stagnant_iterations = 0;
    while (iteration < m_control.max_iterations) {
      if (m_control.verbose) {
        fmt::print("Iteration: {}\n", iteration);
        fmt::print("Best value: {}\n", gbest_value);
        fmt::print("Best position: {}\n", gbest_position);
      }
      update_swarm(func, lower_bound, upper_bound);
      ScalarType current_avg_velocity = compute_average_velocity();
      if (gbest_value == prev_gbest_value) {
        stagnant_iterations++;
      } else {
        stagnant_iterations = 0;
      }

      if (has_converged(stagnant_iterations, current_avg_velocity)) {
        break;
      }
      prev_gbest_value = gbest_value;

      ++iteration;
    }

    xts::optimize::OptimizeResult<ScalarType> result;
    result.x = gbest_position;
    result.fun = gbest_value;
    result.nit = iteration;
    result.nfev = func.evaluation_counts().function_evals;
    result.njev = func.evaluation_counts().gradient_evals;
    result.nhev = func.evaluation_counts().hessian_evals;
    return result;
  }

  ScalarType random_factor() {
    std::uniform_real_distribution<ScalarType> dist(0.0, 1.0);
    return dist(rng);
  }

  xt::xarray<ScalarType> random_position(const xt::xarray<ScalarType> &lower,
                                         const xt::xarray<ScalarType> &upper) {
    xt::xarray<ScalarType> position = xt::empty<ScalarType>(lower.shape());
    for (size_t idx = 0; idx < lower.size(); ++idx) {
      std::uniform_real_distribution<ScalarType> dist(lower(idx), upper(idx));
      position(idx) = dist(rng);
    }
    return position;
  }

  xt::xarray<ScalarType> random_velocity(const xt::xarray<ScalarType> &lower,
                                         const xt::xarray<ScalarType> &upper) {
    xt::xarray<ScalarType> velocity = xt::empty<ScalarType>(lower.shape());
    for (size_t idx = 0; idx < lower.size(); ++idx) {
      std::uniform_real_distribution<ScalarType> dist(
          -abs(upper(idx) - lower(idx)), abs(upper(idx) - lower(idx)));
      velocity(idx) = dist(rng);
    }
    return velocity;
  }

  bool has_converged(size_t stagnant_iterations, ScalarType avg_velocity) {
    if (stagnant_iterations > 50) {
      if (m_control.verbose) {
        fmt::print("Converged due to stagnant iterations: {}\n",
                   stagnant_iterations);
      }
      return true;
    }
    if (avg_velocity < 1e-8) {
      if (m_control.verbose) {
        fmt::print("Converged due to average velocity: {}\n", avg_velocity);
      }
      return true;
    }
    return false;
  }
  ScalarType compute_average_velocity() {
    ScalarType total_velocity = 0.0;

    for (const auto &particle : swarm) {
      ScalarType magnitude = xt::linalg::norm(particle.velocity, 2);
      total_velocity += magnitude;
    }

    return total_velocity / num_particles;
  }
};

} // namespace minimize
} // namespace optimize
} // namespace xts
