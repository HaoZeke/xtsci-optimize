//! Project translation and infinitesimal rotation out of a 3N vector.

use ndarray::{Array1, ArrayView1};

/// eOn `projectOutRotTrans`. No-op unless `x` is 3N with N >= 2.
pub(crate) fn project_out_rot_trans(vec: &mut Array1<f64>, pos: ArrayView1<f64>) {
    let n = pos.len();
    if n < 6 || n % 3 != 0 || vec.len() != n {
        return;
    }
    let nat = n / 3;
    let mut com = [0.0; 3];
    for i in 0..nat {
        com[0] += pos[3 * i];
        com[1] += pos[3 * i + 1];
        com[2] += pos[3 * i + 2];
    }
    let inv = 1.0 / nat as f64;
    com[0] *= inv;
    com[1] *= inv;
    com[2] *= inv;

    let mut basis = Vec::with_capacity(6);
    for d in 0..3 {
        let mut t = Array1::zeros(n);
        for j in 0..nat {
            t[3 * j + d] = 1.0;
        }
        basis.push(t);
    }
    let mut rx = Array1::zeros(n);
    let mut ry = Array1::zeros(n);
    let mut rz = Array1::zeros(n);
    for i in 0..nat {
        let x = pos[3 * i] - com[0];
        let y = pos[3 * i + 1] - com[1];
        let z = pos[3 * i + 2] - com[2];
        rx[3 * i + 1] = -z;
        rx[3 * i + 2] = y;
        ry[3 * i] = z;
        ry[3 * i + 2] = -x;
        rz[3 * i] = -y;
        rz[3 * i + 1] = x;
    }
    basis.push(rx);
    basis.push(ry);
    basis.push(rz);

    let mut ortho: Vec<Array1<f64>> = Vec::with_capacity(6);
    for v in basis {
        let mut u = v;
        for e in &ortho {
            let d = u.dot(e);
            u.scaled_add(-d, e);
        }
        let nrm = l2(&u);
        if nrm > 1.0e-9 {
            u /= nrm;
            ortho.push(u);
        }
    }
    for e in &ortho {
        let d = vec.dot(e);
        vec.scaled_add(-d, e);
    }
}

fn l2(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn drops_a_pure_translation() {
        let pos = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut v = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        project_out_rot_trans(&mut v, pos.view());
        assert!(l2(&v) < 1e-12);
    }
}
