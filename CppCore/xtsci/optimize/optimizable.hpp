#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xarray.hpp"

#include "xtsci/func/base.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts::optimize {
class Optimizable {
  // TODO(rg): Similar to the xtsci::func design, the idea is for derived
  // classes to implement methods like compute() and then public interfaces
  // automatically call set before ()
protected:
  mutable ScalarVec m_cstate;
  const std::reference_wrapper<FObjFunc> m_func;

public:
  explicit Optimizable(FObjFunc &func) : m_func{func} {}
  virtual ~Optimizable() = default;
  virtual ScalarType operator()(const ScalarVec &x) const = 0;
  virtual std::optional<ScalarVec> gradient(const ScalarVec &x) const = 0;
  virtual ScalarType
  directional_derivative(const ScalarVec &x,
                         const ScalarVec &direction) const = 0;
  virtual std::pair<ScalarVec, ScalarVec>
  grad_components(const ScalarVec &xpt, ScalarVec &direction,
                  bool is_normalized = false) const = 0;
  virtual ScalarVec diff(const ScalarVec &_a, const ScalarVec &_b) const = 0;
  // Setters and Getters
  inline virtual ScalarVec getState() { return m_cstate; }
  // virtual ScalarType getDOF() = 0; // TODO(rg): Is this necessary
  virtual void setState(ScalarVec x) = 0;
  // TODO(rg): Providing this for the metrics, should have an OptimizeResult
  // return instead
  inline const std::reference_wrapper<FObjFunc> get_fobj(void) const {
    return m_func;
  }
};

// This is the baseline for simple functions where everything "Just works"
// TOpt --> Trivially optimizable
class TOpt final : public Optimizable {
public:
  using Optimizable::Optimizable;
  inline ScalarType operator()(const ScalarVec &x) const override {
    return this->m_func.get()(x);
  }
  inline std::optional<ScalarVec> gradient(const ScalarVec &x) const override {
    return this->m_func.get().gradient(x);
  }
  inline ScalarType
  directional_derivative(const ScalarVec &x,
                         const ScalarVec &direction) const override {
    return this->m_func.get().directional_derivative(x, direction);
  }
  inline std::pair<ScalarVec, ScalarVec>
  grad_components(const ScalarVec &xpt, ScalarVec &direction,
                  bool is_normalized = false) const override {
    // TODO(rg): WTF, detemplate xts::func and fix this
    xt::xarray<ScalarType> dir = direction;
    return this->m_func.get().grad_components(xpt, dir, is_normalized);
  }
  inline ScalarVec diff(const ScalarVec &_a,
                        const ScalarVec &_b) const override {
    return (_a - _b);
  }
  inline void setState(ScalarVec x) override { this->m_cstate = x; }
};
} // namespace xts::optimize
