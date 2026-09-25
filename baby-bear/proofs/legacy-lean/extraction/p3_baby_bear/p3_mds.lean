import Hax

-- copied from p3_mds
namespace p3_mds.util

/-- Given the first row of a circulant matrix, return the first column.
    Upstream (`mds/src/util.rs`): `col[0] = row[0]`; `col[i] = row[N - i]`
    for `i ∈ 1..N`. E.g. `[0, 1, 2, 3, 4, 5]` ↦ `[0, 5, 4, 3, 2, 1]`.

    FAITHFUL: this is the upstream permutation, not a placeholder. It must be
    total: the six `MATRIX_CIRC_MDS_*_COL` constants call this under
    `RustM.of_isOk _ (by rfl)`, so an `opaque` here is not an option — `isOk`
    cannot reduce through an opaque head, and the obligation is unprovable
    (an opaque `RustM` inhabitant could be `div` or `fail`).

    `-i` on `Fin N.toNat` is exactly upstream's index map: `Fin.neg` sends
    `i ↦ (N - i) % N`, i.e. `0 ↦ 0` and `i ↦ N - i` for `i ≥ 1`. Written this
    way rather than with a dependent `if` so the body stays plain `Nat`
    arithmetic and the six `rfl` obligations discharge by kernel reduction. -/
def first_row_to_first_col
    (N : usize)
    (T : Type)
    [_trait_constr_first_row_to_first_col_associated_type_i0 :
      core_models.marker.Copy.AssociatedTypes
      T]
    [_trait_constr_first_row_to_first_col_i0 : core_models.marker.Copy T ]
    (v : (RustArray T N)) :
    RustM (RustArray T N) :=
  pure (.ofVec (Vector.ofFn (fun i => v.toVec[-i])))

end p3_mds.util
