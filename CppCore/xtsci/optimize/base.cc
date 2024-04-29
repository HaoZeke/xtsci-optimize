// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtsci/optimize/base.hpp"

namespace xts::optimize {

bool AbstractOptimizer::converged(const SearchState &state) const {
  // std::cout << m_next->direction << std::endl;
  if (m_result.nit > 2) {
    return xt::linalg::norm(m_next->direction) < m_control.get().gtol;
  }
  return false;
}
} // namespace xts::optimize
