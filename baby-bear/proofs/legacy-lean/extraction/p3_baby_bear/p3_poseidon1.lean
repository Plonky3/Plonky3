import Hax

-- copied from p3_poseidon1
namespace p3_poseidon1

structure Poseidon1
  (F : Type)
  (FullRoundPerm : Type)
  (PartialRoundPerm : Type)
  (WIDTH : usize)
  (D : u64)
  where
  full_round_layer : FullRoundPerm
  partial_round_layer : PartialRoundPerm
  _phantom : (core_models.marker.PhantomData F)

structure Poseidon1Constants (F : Type) (WIDTH : usize) where
  rounds_f : usize
  rounds_p : usize
  mds_circ_col : (RustArray i64 WIDTH)
  round_constants : (alloc.vec.Vec (RustArray F WIDTH) alloc.alloc.Global)

opaque Impl_1.new
    (F : Type)
    (FullRoundPerm : Type)
    (PartialRoundPerm : Type)
    (WIDTH : usize)
    (D : u64)
    (raw : (Poseidon1Constants F (WIDTH))) :
    RustM (Poseidon1 F FullRoundPerm PartialRoundPerm (WIDTH) (D))

end p3_poseidon1
