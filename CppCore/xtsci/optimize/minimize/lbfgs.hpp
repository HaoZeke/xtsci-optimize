#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <deque>
#include <vector>
#include <utility>
// clang-format on

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts {
namespace optimize {
namespace minimize {

void printOptimizationStep(size_t step, const ScalarType &energy,
                           const ScalarType &fmax);

class LBFGSOptimizer : public AbstractOptimizer {
private:
  size_t m_corrections; // Number of corrections to store
  // Two-deque to store s and y, which represent differences in x and
  // gradient, respectively
  std::deque<ScalarVec> m_s_list, m_y_list;
  std::deque<ScalarType> m_rho_list;

public:
  explicit LBFGSOptimizer(SearchStrategy &strategy,
                          size_t mem_list = 2 /* Typically 5 to 20 */)
      : AbstractOptimizer(strategy) {
    m_corrections = mem_list;
  }

protected:
  void step(const Optimizable &optobj) override;

private:
  std::pair<ScalarVec, ScalarVec> get_grad_dir(const Optimizable &optobj,
                                               const SearchState &state) const;
  ScalarVec get_direction(const ScalarVec &gradient,
                          const std::deque<ScalarVec> &s_list,
                          const std::deque<ScalarVec> &y_list,
                          const std::deque<ScalarType> &rho_list) const;
};
} // namespace minimize
} // namespace optimize
} // namespace xts
