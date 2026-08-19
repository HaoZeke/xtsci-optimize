//! Nonlinear CG: conjugacy coefficient and restart.

use ndarray::ArrayView1;

/// Restart policy.
pub mod restart;

/// Shared vectors for a β evaluation.
#[derive(Clone, Debug)]
pub struct ConjugacyContext<'a> {
    /// `g_k`.
    pub current_gradient: ArrayView1<'a, f64>,
    /// `g_{k-1}`.
    pub previous_gradient: ArrayView1<'a, f64>,
    /// `d_{k-1}` (search direction, already a descent direction).
    pub previous_direction: ArrayView1<'a, f64>,
}

/// Conjugacy coefficient β. Formulas are Nocedal and Wright chapter 5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Conjugacy {
    /// Fletcher-Reeves, NJWS 5.41a: `||g||^2 / ||g_old||^2`.
    FletcherReeves,
    /// Polak-Ribiere, NJWS 5.44: `g·(g - g_old) / ||g_old||^2`.
    PolakRibiere,
    /// Hestenes-Stiefel, NJWS 5.46: `g·y / (y·d_old)`.
    HestenesStiefel,
    /// Dai-Yuan (1999): `||g||^2 / (d_old·y)`.
    DaiYuan,
    /// Fletcher conjugate descent: `||g||^2 / (d_old·g_old)`.
    ConjugateDescent,
    /// Hager--Zhang (2005) eq. 1.3 / NJWS 5.50.
    HagerZhang,
    /// Liu--Storey (1991): `- g·y / (d_old·g_old)`.
    LiuStorey,
    /// Gilbert-Nocedal FR-PR hybrid, NJWS 5.50.
    FrPr,
}

/// Restart when conjugacy is lost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Restart {
    /// Never force β = 0.
    Never,
    /// NJWS 5.52: restart when `|g·g_old| / ||g_old||^2 >= threshold`.
    Njws {
        /// ν; book default is 0.1.
        threshold: f64,
    },
}

impl Restart {
    /// Default NJWS restart with ν = 0.1.
    pub fn njws() -> Self {
        Self::Njws { threshold: 0.1 }
    }

    /// True when the next step should reset to steepest descent.
    pub fn should_restart(&self, ctx: &ConjugacyContext<'_>) -> bool {
        match *self {
            Self::Never => false,
            Self::Njws { threshold } => {
                let gg_old = dot(ctx.previous_gradient, ctx.previous_gradient);
                if gg_old <= 0.0 {
                    return true;
                }
                let dev = dot(ctx.current_gradient, ctx.previous_gradient).abs() / gg_old;
                dev >= threshold
            }
        }
    }
}

impl Conjugacy {
    /// β for this method. Degenerate denominators return 0.
    pub fn beta(&self, ctx: &ConjugacyContext<'_>) -> f64 {
        let g = ctx.current_gradient;
        let gold = ctx.previous_gradient;
        let d = ctx.previous_direction;
        let gg = dot(g, g);
        let gg_old = dot(gold, gold);
        let y_g = dot(g, g) - dot(g, gold); // g · (g - gold)
        match *self {
            Self::FletcherReeves => div(gg, gg_old),
            Self::PolakRibiere => div(y_g, gg_old),
            Self::HestenesStiefel => {
                let y_d = dot(g, d) - dot(gold, d);
                div(y_g, y_d)
            }
            Self::DaiYuan => {
                let y_d = dot(g, d) - dot(gold, d);
                div(gg, y_d)
            }
            Self::ConjugateDescent => div(gg, dot(d, gold)),
            Self::HagerZhang => {
                let y_d = dot(g, d) - dot(gold, d);
                if y_d.abs() <= f64::EPSILON {
                    return 0.0;
                }
                let yy = gg + gg_old - 2.0 * dot(g, gold);
                let d_g = dot(d, g);
                y_g / y_d - 2.0 * yy * d_g / (y_d * y_d)
            }
            Self::LiuStorey => div(-y_g, dot(d, gold)),
            Self::FrPr => {
                let beta_pr = div(y_g, gg_old);
                let beta_fr = div(gg, gg_old);
                if beta_pr < -beta_fr {
                    -beta_fr
                } else if beta_pr.abs() <= beta_fr {
                    beta_pr
                } else {
                    beta_fr
                }
            }
        }
    }
}

fn dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn div(num: f64, den: f64) -> f64 {
    if den.abs() <= f64::EPSILON {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn textbook_betas_on_orthogonal_gradients() {
        let g = array![1.0, 0.0];
        let gold = array![0.0, 1.0];
        let d = array![-1.0, 0.0];
        let ctx = ConjugacyContext {
            current_gradient: g.view(),
            previous_gradient: gold.view(),
            previous_direction: d.view(),
        };
        assert_eq!(Conjugacy::FletcherReeves.beta(&ctx), 1.0);
        assert_eq!(Conjugacy::PolakRibiere.beta(&ctx), 1.0);
        // y = [1, -1], y·d = -1, g·y = 1, β_HS = -1
        assert_eq!(Conjugacy::HestenesStiefel.beta(&ctx), -1.0);
        // β_DY = 1 / (d·y) = 1 / -1 = -1
        assert_eq!(Conjugacy::DaiYuan.beta(&ctx), -1.0);
    }
}
