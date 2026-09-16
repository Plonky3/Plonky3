import Hax

-- copied from p3_mds
namespace p3_mds.util

def first_row_to_first_col
    (N : usize)
    (T : Type)
    [_trait_constr_first_row_to_first_col_associated_type_i0 :
      core_models.marker.Copy.AssociatedTypes
      T]
    [_trait_constr_first_row_to_first_col_i0 : core_models.marker.Copy T ]
    (v : (RustArray T N)) :
    RustM (RustArray T N) := pure v

end p3_mds.util
