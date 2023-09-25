#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>
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
  size_t num_particles;
  ScalarType inertia_weight;
  ScalarType c1;    // cognitive component
  std::mt19937 rng; // Mersenne Twister random number generator

  ScalarType c2; // social component

public:
  PSOptim(size_t num_particles = 30, ScalarType inertia = 0.5,
          ScalarType c1 = 1.5, ScalarType c2 = 1.5)
      : num_particles(num_particles), inertia_weight(inertia), c1(c1), c2(c2) {
    std::random_device rd;
    rng.seed(rd()); // Seed the generator
  }

  void initialize_swarm(const ObjectiveFunction<ScalarType> &func,
                        const xt::xarray<ScalarType> &lower_bound,
                        const xt::xarray<ScalarType> &upper_bound) {
    for (size_t i = 0; i < num_particles; ++i) {
      Particle<ScalarType> particle;

      // Randomly initialize position and velocity
      particle.position = random_position(lower_bound, upper_bound);
      particle.velocity = random_velocity(lower_bound, upper_bound);
      particle.best_position = particle.position;
      particle.best_value = func(particle.position);

      swarm.push_back(particle);

      // Update global best if needed
      if (i == 0 || particle.best_value < gbest_value) {
        gbest_position = particle.best_position;
        gbest_value = particle.best_value;
      }
    }
  }

  void update_swarm(const ObjectiveFunction<ScalarType> &func) {
    for (auto &particle : swarm) {
      // Update velocity
      particle.velocity =
          inertia_weight * particle.velocity +
          c1 * random_factor() * (particle.best_position - particle.position) +
          c2 * random_factor() * (gbest_position - particle.position);

      // Update position
      particle.position += particle.velocity;

      // Evaluate new position
      ScalarType new_value = func(particle.position);
      if (new_value < particle.best_value) {
        particle.best_value = new_value;
        particle.best_position = particle.position;

        // Update global best if needed
        if (new_value < gbest_value) {
          gbest_position = particle.best_position;
          gbest_value = new_value;
        }
      }
    }
  }

  xts::optimize::OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const xt::xarray<ScalarType> &lower_bound,
           const xt::xarray<ScalarType> &upper_bound,
           const OptimizeControl<ScalarType> &control) {
    initialize_swarm(func, lower_bound, upper_bound);

    size_t iteration = 0;
    while (iteration < control.max_iterations) {
      update_swarm(func);
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
    for (size_t i = 0; i < lower.size(); ++i) {
      std::uniform_real_distribution<ScalarType> dist(lower(i), upper(i));
      position(i) = dist(rng);
    }
    return position;
  }

  xt::xarray<ScalarType> random_velocity(const xt::xarray<ScalarType> &lower,
                                         const xt::xarray<ScalarType> &upper) {
    xt::xarray<ScalarType> velocity = xt::empty<ScalarType>(lower.shape());
    for (size_t i = 0; i < lower.size(); ++i) {
      std::uniform_real_distribution<ScalarType> dist(-abs(upper(i) - lower(i)),
                                                      abs(upper(i) - lower(i)));
      velocity(i) = dist(rng);
    }
    return velocity;
  }
};

} // namespace minimize
} // namespace optimize
} // namespace xts
