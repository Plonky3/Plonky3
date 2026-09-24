//! The plan for moving between three-quadword elements and coordinate registers.
//!
//! `W` consecutive elements fill three registers.
//!
//! Coordinate `d` of element `k` sits at flat quadword `f = 3k + d`:
//!
//! ```text
//!     register    f / W
//!     position    f % W
//! ```
//!
//! For a fixed coordinate, the positions `(3k + d) % W` are all distinct.
//!
//! That holds because `3` is invertible modulo `W`, a power of two.
//!
//! So two blends gather a coordinate, and one permute puts it in element order.
//!
//! The same argument runs backwards for the scatter.

/// The blend masks and permute indices of both directions, for `W` quadwords per register.
pub(super) struct Plan<const W: usize> {
    /// Per coordinate: the position element `k` reads after the blend.
    pub(super) gather_index: [[usize; W]; 3],
    /// Per coordinate: the positions taken from the second and the third register.
    pub(super) gather_mask: [[u32; 2]; 3],
    /// Per coordinate: the element each position reads before the blend.
    pub(super) scatter_index: [[usize; W]; 3],
    /// Per register: the positions taken from the second and the third coordinate.
    pub(super) scatter_mask: [[u32; 2]; 3],
}

impl<const W: usize> Plan<W> {
    /// The plan, evaluated at compile time.
    pub(super) const NEW: Self = Self::new();

    /// Derives every index and mask from the slot formula.
    const fn new() -> Self {
        let mut plan = Self {
            gather_index: [[0; W]; 3],
            gather_mask: [[0; 2]; 3],
            scatter_index: [[0; W]; 3],
            scatter_mask: [[0; 2]; 3],
        };
        // Walk every quadword of the three registers once, in memory order.
        let mut f = 0;
        while f < 3 * W {
            let (k, d) = (f / 3, f % 3);
            let (r, p) = (f / W, f % W);

            // Gathering coordinate d: lane k reads position p, taken from register r.
            plan.gather_index[d][k] = p;
            if r > 0 {
                plan.gather_mask[d][r - 1] |= 1 << p;
            }

            // Scattering into register r: position p reads lane k of coordinate d.
            //
            // Distinct positions per coordinate make that lane the same for every register.
            plan.scatter_index[d][p] = k;
            if d > 0 {
                plan.scatter_mask[r][d - 1] |= 1 << p;
            }
            f += 1;
        }
        plan
    }
}

#[cfg(test)]
mod tests {
    use super::Plan;

    /// A gather followed by a scatter, on plain arrays, returns the input.
    fn round_trips<const W: usize>() {
        let plan = Plan::<W>::NEW;

        // Quadword f holds its own flat index, so every slot is recognisable.
        let registers: [[usize; W]; 3] =
            core::array::from_fn(|r| core::array::from_fn(|p| W * r + p));

        // Gather: blend by mask, then permute into element order.
        let coordinates: [[usize; W]; 3] = core::array::from_fn(|d| {
            let pick = |p: usize| {
                let [second, third] = plan.gather_mask[d];
                match (second >> p & 1, third >> p & 1) {
                    (0, 0) => registers[0][p],
                    (1, 0) => registers[1][p],
                    _ => registers[2][p],
                }
            };
            core::array::from_fn(|k| pick(plan.gather_index[d][k]))
        });
        for (d, row) in coordinates.iter().enumerate() {
            for (k, &quadword) in row.iter().enumerate() {
                assert_eq!(quadword, 3 * k + d, "gather d {d} k {k}");
            }
        }

        // Scatter: permute each coordinate, then blend by mask.
        let permuted: [[usize; W]; 3] = core::array::from_fn(|d| {
            core::array::from_fn(|p| coordinates[d][plan.scatter_index[d][p]])
        });
        for (r, (register, &[second, third])) in
            registers.iter().zip(&plan.scatter_mask).enumerate()
        {
            for (p, &quadword) in register.iter().enumerate() {
                let d = match (second >> p & 1, third >> p & 1) {
                    (0, 0) => 0,
                    (1, 0) => 1,
                    _ => 2,
                };
                assert_eq!(permuted[d][p], quadword, "scatter r {r} p {p}");
            }
        }
    }

    #[test]
    fn the_plans_round_trip_at_both_widths() {
        round_trips::<4>();
        round_trips::<8>();
    }
}
