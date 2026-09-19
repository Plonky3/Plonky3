import Hax

-- copied from p3_poseidon2
namespace p3_poseidon2

structure Poseidon2
  (F : Type)
  (ExternalPerm : Type)
  (InternalPerm : Type)
  (WIDTH : usize)
  (D : u64)
  where
  external_layer : ExternalPerm
  internal_layer : InternalPerm
  _phantom : (core_models.marker.PhantomData F)

structure ExternalLayerConstants (T : Type) (WIDTH : usize) where
  initial : (alloc.vec.Vec (RustArray T WIDTH) alloc.alloc.Global)
  terminal : (alloc.vec.Vec (RustArray T WIDTH) alloc.alloc.Global)

opaque Impl.new
    (F : Type)
    (ExternalPerm : Type)
    (InternalPerm : Type)
    (WIDTH : usize)
    (D : u64)
    (external_constants : (ExternalLayerConstants F (WIDTH)))
    (internal_constants : (alloc.vec.Vec F alloc.alloc.Global)) :
    RustM (Poseidon2 F ExternalPerm InternalPerm (WIDTH) (D))

end p3_poseidon2

namespace p3_poseidon2.external

opaque Impl_4.new
    (T : Type)
    (WIDTH : usize)
    (initial : (alloc.vec.Vec (RustArray T WIDTH) alloc.alloc.Global))
    (terminal : (alloc.vec.Vec (RustArray T WIDTH) alloc.alloc.Global)) :
    RustM (p3_poseidon2.ExternalLayerConstants T (WIDTH))

end p3_poseidon2.external
