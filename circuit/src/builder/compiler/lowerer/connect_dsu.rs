//! Disjoint-set forest with per-class witness allocation.

use hashbrown::{HashMap, HashSet};

use crate::types::{ExprId, WitnessAllocator, WitnessId};

/// Sparse disjoint-set forest for the expression connect mechanism.
///
/// Maintains equivalence classes over expression IDs declared via `connect(a, b)`.
/// Each class is mapped to a single shared witness slot, so that all members
/// of a class are assigned the same witness during lowering.
pub(super) struct ConnectDsu {
    /// Parent pointers for the union-find forest.
    ///
    /// An ID absent from this map is implicitly its own root.
    parents: HashMap<ExprId, ExprId>,
    /// Set of all expression IDs that participate in at least one connect pair.
    in_connect: HashSet<ExprId>,
    /// Maps each equivalence-class root to its allocated witness slot.
    ///
    /// Populated lazily on first allocation within a class.
    root_to_widx: HashMap<ExprId, WitnessId>,
}

impl ConnectDsu {
    /// Build a DSU from pending `connect(a, b)` pairs.
    ///
    /// Collects all participating IDs into the membership set, then unions each pair.
    pub fn from_connects(connects: &[(ExprId, ExprId)]) -> Self {
        // Collect every expression that appears in any connect pair.
        let in_connect = connects.iter().flat_map(|(a, b)| [*a, *b]).collect();
        let mut dsu = Self {
            parents: HashMap::new(),
            in_connect,
            root_to_widx: HashMap::new(),
        };
        // Merge each declared pair.
        for &(a, b) in connects {
            dsu.union(a, b);
        }
        dsu
    }

    /// Find the representative (root) of the class containing the given ID.
    ///
    /// Uses two-pass iterative path compression with zero heap allocation:
    /// - Pass 1: walk parent pointers to the root.
    /// - Pass 2: re-walk and point every node directly to the root.
    ///
    /// IDs never inserted into the parent map are implicitly self-roots.
    #[inline]
    pub fn find(&mut self, x: ExprId) -> ExprId {
        // Pass 1: chase to the root.
        let mut root = x;
        while let Some(&p) = self.parents.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // Pass 2: compress every node on the path to point directly at the root.
        let mut v = x;
        while let Some(&p) = self.parents.get(&v) {
            if p == root {
                break;
            }
            self.parents.insert(v, root);
            v = p;
        }
        root
    }

    /// Merge the equivalence classes of two expression IDs.
    ///
    /// The second ID's root is attached under the first ID's root.
    /// No-op if both already belong to the same class.
    #[inline]
    pub fn union(&mut self, a: ExprId, b: ExprId) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parents.insert(rb, ra);
        }
    }

    /// Allocate (or reuse) a witness slot for the given expression.
    ///
    /// - If the expression participates in a connect class, the class's shared
    ///   slot is returned (allocated on first access within that class).
    /// - Otherwise a fresh slot is allocated unconditionally.
    pub fn alloc_witness(&mut self, expr_id: ExprId, alloc: &mut WitnessAllocator) -> WitnessId {
        if self.in_connect.contains(&expr_id) {
            // Find the class representative via path-compressed lookup.
            let root = self.find(expr_id);
            // Reuse the class representative's slot, creating it on first access.
            *self
                .root_to_widx
                .entry(root)
                .or_insert_with(|| alloc.alloc())
        } else {
            // Not in any connect class — always allocate a fresh slot.
            alloc.alloc()
        }
    }

    /// Look up the witness slot for the class containing the given expression.
    ///
    /// Returns `None` if no witness has been allocated for that class yet.
    pub fn class_witness(&mut self, expr_id: ExprId) -> Option<WitnessId> {
        // Resolve to the class root, then look up the cached witness.
        let root = self.find(expr_id);
        self.root_to_widx.get(&root).copied()
    }

    /// Iterate over all expression IDs that participate in at least one connect.
    pub fn connected_exprs(&self) -> impl Iterator<Item = ExprId> + '_ {
        self.in_connect.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hashbrown::HashMap;
    use proptest::prelude::*;

    use super::*;

    fn bare_dsu() -> ConnectDsu {
        ConnectDsu {
            parents: HashMap::new(),
            in_connect: HashSet::new(),
            root_to_widx: HashMap::new(),
        }
    }

    #[test]
    fn test_dsu_utilities() {
        // An element absent from the parent map should be its own root.
        let mut dsu = bare_dsu();
        assert_eq!(dsu.find(ExprId(5)), ExprId(5));

        // Set up a 3-element chain: 1 -> 2 -> 3 (3 is root).
        dsu.parents.clear();
        dsu.parents.insert(ExprId(1), ExprId(2));
        dsu.parents.insert(ExprId(2), ExprId(3));
        dsu.parents.insert(ExprId(3), ExprId(3));
        // After find, element 1 should resolve to root 3.
        assert_eq!(dsu.find(ExprId(1)), ExprId(3));
        // Path compression should have pointed element 1 directly at root 3.
        assert_eq!(dsu.parents[&ExprId(1)], ExprId(3));

        // Union should merge two previously disjoint elements.
        dsu.parents.clear();
        dsu.union(ExprId(10), ExprId(20));
        assert_eq!(dsu.find(ExprId(10)), dsu.find(ExprId(20)));

        // Union of an element with itself is a no-op.
        dsu.parents.clear();
        dsu.union(ExprId(7), ExprId(7));
        assert_eq!(dsu.find(ExprId(7)), ExprId(7));
    }

    #[test]
    fn test_build_connect_dsu() {
        // No connect pairs should yield an empty parent map.
        let connects = vec![];
        let dsu = ConnectDsu::from_connects(&connects);
        assert!(dsu.parents.is_empty());

        // A single pair should put both elements in the same class.
        let connects = vec![(ExprId(1), ExprId(2))];
        let mut dsu = ConnectDsu::from_connects(&connects);
        assert_eq!(dsu.find(ExprId(1)), dsu.find(ExprId(2)));

        // Transitive chain: connecting 0-1, 1-2, 2-3 should unify all four.
        let connects = vec![
            (ExprId(0), ExprId(1)),
            (ExprId(1), ExprId(2)),
            (ExprId(2), ExprId(3)),
        ];
        let mut dsu = ConnectDsu::from_connects(&connects);
        let root = dsu.find(ExprId(0));
        assert_eq!(root, dsu.find(ExprId(1)));
        assert_eq!(root, dsu.find(ExprId(2)));
        assert_eq!(root, dsu.find(ExprId(3)));

        // Two disjoint pairs should form two separate classes.
        let connects = vec![(ExprId(0), ExprId(1)), (ExprId(2), ExprId(3))];
        let mut dsu = ConnectDsu::from_connects(&connects);
        let root01 = dsu.find(ExprId(0));
        assert_eq!(root01, dsu.find(ExprId(1)));
        let root23 = dsu.find(ExprId(2));
        assert_eq!(root23, dsu.find(ExprId(3)));
        // The two components must have different roots.
        assert_ne!(root01, root23);
    }

    #[test]
    fn test_alloc_witness_sharing() {
        // Connect elements 1, 2, 3 into one class.
        let connects = vec![(ExprId(1), ExprId(2)), (ExprId(2), ExprId(3))];
        let mut dsu = ConnectDsu::from_connects(&connects);
        let mut alloc = WitnessAllocator::new();

        // First allocation in the class creates a new witness.
        let w1 = dsu.alloc_witness(ExprId(1), &mut alloc);
        // Subsequent allocations for the same class reuse the witness.
        let w2 = dsu.alloc_witness(ExprId(2), &mut alloc);
        let w3 = dsu.alloc_witness(ExprId(3), &mut alloc);
        assert_eq!(w1, w2);
        assert_eq!(w1, w3);

        // An unconnected expression gets its own fresh witness.
        let w_other = dsu.alloc_witness(ExprId(99), &mut alloc);
        assert_ne!(w1, w_other);
    }

    #[test]
    fn test_class_witness_lookup() {
        let connects = vec![(ExprId(5), ExprId(6))];
        let mut dsu = ConnectDsu::from_connects(&connects);
        let mut alloc = WitnessAllocator::new();

        // Before any allocation, looking up the class witness returns nothing.
        assert!(dsu.class_witness(ExprId(5)).is_none());
        // Allocate a witness via one member of the class.
        let w = dsu.alloc_witness(ExprId(5), &mut alloc);
        // The other member should now resolve to the same witness.
        assert_eq!(dsu.class_witness(ExprId(6)), Some(w));
    }

    #[test]
    fn test_connected_exprs() {
        // Two connect pairs that form a transitive chain: 3-7-10.
        let connects = vec![(ExprId(3), ExprId(7)), (ExprId(7), ExprId(10))];
        let dsu = ConnectDsu::from_connects(&connects);
        let mut connected: Vec<ExprId> = dsu.connected_exprs().collect();
        connected.sort();
        // All three participating IDs should be reported.
        assert_eq!(connected, vec![ExprId(3), ExprId(7), ExprId(10)]);

        // Empty connects should yield no connected expressions.
        let dsu = ConnectDsu::from_connects(&[]);
        assert_eq!(dsu.connected_exprs().count(), 0);
    }

    #[test]
    fn test_dsu_stress_chain() {
        // Build a 1000-element linear chain: 0-1-2-...-999.
        let mut dsu = bare_dsu();
        for i in 0u32..999 {
            dsu.union(ExprId(i), ExprId(i + 1));
        }
        // All elements should share a single root after path compression.
        let root = dsu.find(ExprId(0));
        for i in 1u32..1000 {
            assert_eq!(dsu.find(ExprId(i)), root, "chain element {i} differs");
        }
    }

    #[test]
    fn test_dsu_stress_star() {
        // Build a star topology: element 0 is the hub, all others connect to it.
        let mut dsu = bare_dsu();
        for i in 1u32..1000 {
            dsu.union(ExprId(0), ExprId(i));
        }
        // All elements should resolve to the same root.
        let root = dsu.find(ExprId(0));
        for i in 1u32..1000 {
            assert_eq!(dsu.find(ExprId(i)), root, "star element {i} differs");
        }
    }

    #[test]
    fn test_dsu_stress_mixed() {
        // Build 100 disjoint 10-element chains.
        let mut dsu = bare_dsu();
        for chain in 0u32..100 {
            let base = chain * 10;
            for i in 0u32..9 {
                dsu.union(ExprId(base + i), ExprId(base + i + 1));
            }
        }
        // Merge all chains by connecting their heads to element 0.
        for chain in 1u32..100 {
            dsu.union(ExprId(0), ExprId(chain * 10));
        }
        // All 1000 elements should now share a single root.
        let root = dsu.find(ExprId(0));
        for i in 1u32..1000 {
            assert_eq!(dsu.find(ExprId(i)), root, "mixed element {i} differs");
        }
    }

    fn connections(max_id: u32) -> impl Strategy<Value = Vec<(ExprId, ExprId)>> {
        prop::collection::vec((0..max_id, 0..max_id), 0..20).prop_map(|pairs| {
            pairs
                .into_iter()
                .map(|(a, b)| (ExprId(a), ExprId(b)))
                .collect()
        })
    }

    proptest! {
        #[test]
        fn dsu_find_idempotent(connects in connections(50)) {
            let mut dsu = ConnectDsu::from_connects(&connects);

            for id in 0u32..50 {
                let root1 = dsu.find(ExprId(id));
                let root2 = dsu.find(ExprId(id));
                prop_assert_eq!(root1, root2, "dsu_find should be idempotent");
            }
        }

        #[test]
        fn dsu_union_transitivity(connects in connections(30)) {
            let mut dsu = ConnectDsu::from_connects(&connects);

            for (a, b) in &connects {
                let ra = dsu.find(*a);
                let rb = dsu.find(*b);
                prop_assert_eq!(ra, rb, "connected nodes should have same root");
            }
        }

        #[test]
        fn dsu_union_commutative(a in 0u32..100, b in 0u32..100) {
            let mut dsu1 = bare_dsu();
            let mut dsu2 = bare_dsu();

            dsu1.union(ExprId(a), ExprId(b));
            dsu2.union(ExprId(b), ExprId(a));

            let r1a = dsu1.find(ExprId(a));
            let r1b = dsu1.find(ExprId(b));
            let r2a = dsu2.find(ExprId(a));
            let r2b = dsu2.find(ExprId(b));

            prop_assert_eq!(r1a, r1b, "union should connect a and b");
            prop_assert_eq!(r2a, r2b, "union should connect b and a");
        }
    }
}
