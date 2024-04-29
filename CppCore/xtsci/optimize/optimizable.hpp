#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xarray.hpp"

#include "xtsci/func/base.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts::optimize {
class Optimizable {
private:
  const ScalarVec m_cstate;

protected:
  const std::reference_wrapper<FObjFunc> m_func;

public:
  Optimizable(FObjFunc &func) : m_func{func} {}
  virtual ~Optimizable() = default;
  virtual ScalarType operator()(const ScalarVec &x) const = 0;
  virtual std::optional<ScalarVec> gradient(const ScalarVec &x) const = 0;
  virtual ScalarVec diff(const ScalarVec &_a, const ScalarVec &_b) const = 0;
  // Setters and Getters
  inline virtual ScalarVec getState() { return m_cstate; };
  virtual ScalarType getDOF() = 0;
  virtual void setState(ScalarVec x) = 0;
};
} // namespace xts::optimize
