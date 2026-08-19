//! Line-search accept conditions (xtsci `linesearch/conditions/`).
//!
//! Formulas match HaoZeke/xtsci-optimize: Armijo, weak/strong curvature,
//! weak/strong Wolfe, Goldstein upper bound, and Goldstein.

/// Armijo sufficient decrease: `φ(α) <= φ(0) + c α φ'(0)`.
#[inline]
pub fn armijo(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    phi <= phi0 + c * alpha * dphi0
}

/// Weak curvature: `φ'(α) >= c2 φ'(0)`.
#[inline]
pub fn curvature(dphi: f64, dphi0: f64, c2: f64) -> bool {
    dphi >= c2 * dphi0
}

/// Strong curvature: `|φ'(α)| <= c2 |φ'(0)|`.
#[inline]
pub fn strong_curvature(dphi: f64, dphi0: f64, c2: f64) -> bool {
    dphi.abs() <= c2 * dphi0.abs()
}

/// Weak Wolfe: Armijo and weak curvature.
#[inline]
pub fn weak_wolfe(phi: f64, phi0: f64, alpha: f64, dphi: f64, dphi0: f64, c1: f64, c2: f64) -> bool {
    armijo(phi, phi0, alpha, dphi0, c1) && curvature(dphi, dphi0, c2)
}

/// Strong Wolfe: Armijo and strong curvature.
#[inline]
pub fn strong_wolfe(
    phi: f64,
    phi0: f64,
    alpha: f64,
    dphi: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> bool {
    armijo(phi, phi0, alpha, dphi0, c1) && strong_curvature(dphi, dphi0, c2)
}

/// Goldstein upper bound: `φ(α) <= φ(0) + (1 - c) α φ'(0)`.
///
/// `c` belongs in `(0, 0.5)` (xtsci `GoldsteinUpperBoundCondition`).
#[inline]
pub fn goldstein_upper(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    phi <= phi0 + (1.0 - c) * alpha * dphi0
}

/// Goldstein: Armijo and Goldstein upper bound.
#[inline]
pub fn goldstein(phi: f64, phi0: f64, alpha: f64, dphi0: f64, c: f64) -> bool {
    armijo(phi, phi0, alpha, dphi0, c) && goldstein_upper(phi, phi0, alpha, dphi0, c)
}
