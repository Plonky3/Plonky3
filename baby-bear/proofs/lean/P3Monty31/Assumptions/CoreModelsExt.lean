/-
# Gaps in hax-lean's `CoreModels`

Models of `core` items that the generated code references and that
`CoreModels` (hax-lean v0.3.27) does not define. Each one is a candidate to
upstream to hax-lean and then delete.
-/
import Aeneas
import CoreModels
open CoreModels Aeneas
open Aeneas.Std hiding namespace core alloc
open RustM ControlFlow Error

namespace CoreModels

/-- `impl<T: ?Sized> Clone for PhantomData<T>`, whose body is `Self`
(`library/core/src/marker.rs`). Referenced by the derived `Clone` for
`MontyField31`. -/
def core.marker.PhantomData.Insts.CoreCloneClone.clone {T : Type}
    (self : core.marker.PhantomData T) : RustM (core.marker.PhantomData T) :=
  ok self

/-- `impl<T: ?Sized + AsRef<U>, U: ?Sized> AsRef<U> for &T`, whose body is
`<T as AsRef<U>>::as_ref(*self)` (`library/core/src/convert/mod.rs`). aeneas
erases shared references, so `&T` is `T` and the impl is the inner one.
Referenced by baby-bear's `TwoAdicData<&'static [MontyField31<_>]>` impl. -/
def core.Shared0T.Insts.CoreConvertAsRef {T U : Type}
    (convertAsRefInst : core.convert.AsRef T U) : core.convert.AsRef T U :=
  { as_ref := convertAsRefInst.as_ref }

end CoreModels
