import Hax

-- copied from p3_mds
namespace p3_mds.util

/-- Given the first row of a circulant matrix, return the first column.
    Upstream (`mds/src/util.rs`): `col[0] = row[0]`; `col[i] = row[N - i]`
    for `i ∈ 1..N`. E.g. `[0, 1, 2, 3, 4, 5]` ↦ `[0, 5, 4, 3, 2, 1]`.
    ASSUMED — declared `opaque` (with a candidate body below, if needed). -/
opaque first_row_to_first_col
    (N : usize)
    (T : Type)
    [_trait_constr_first_row_to_first_col_associated_type_i0 :
      core_models.marker.Copy.AssociatedTypes
      T]
    [_trait_constr_first_row_to_first_col_i0 : core_models.marker.Copy T ]
    (v : (RustArray T N)) :
    RustM (RustArray T N)
--   pure (.ofVec (Vector.ofFn (fun i =>
--     if h : i.val = 0 then v.toVec[i]
--     else v.toVec.get ⟨N.toNat - i.val,
--       Nat.sub_lt (Nat.zero_lt_of_lt i.isLt) (Nat.pos_of_ne_zero h)⟩)))

end p3_mds.util
