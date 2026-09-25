import Hax
import p3_baby_bear.p3_field
import p3_baby_bear.p3_monty_31
import p3_baby_bear.p3_mds
import p3_baby_bear.p3_poseidon1
import p3_baby_bear.p3_poseidon2
import p3_baby_bear.p3_symmetric
import p3_baby_bear.hax_ext

/-! Axiomatized interface to the Plonky3 crates `p3-baby-bear` depends on.

    Nothing here is extracted or proved: every declaration is an *assumption*
    about an upstream crate, written out only far enough to let the extracted
    `p3_baby_bear.lean` type-check. See `TCB.md`. -/
