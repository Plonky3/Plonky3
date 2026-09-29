-- [p3_baby_bear]: external functions.
-- HAND-WRITTEN, from hax's seed of Extraction/FunsExternal_Template.lean.
-- The generated code imports it by this module name. extract.sh fails if the
-- regenerated template declares different names than this file does.
--
-- EDITED: aeneas's template gives `fmt` a `&mut Formatter` back-function in
-- its result, but `CoreModels`' `core.fmt.Debug.fmt` returns only the updated
-- `Formatter`. The four signatures below are adjusted to CoreModels' shape.
-- `axiom` -> `opaque`: an opaque constant asserts only that
-- some inhabitant exists, so it adds nothing to a theorem's axiom footprint.
-- These are the bodies of four derived `Debug` impls, which extract.sh
-- makes opaque on purpose.
import Aeneas
import CoreModels
import P3BabyBear.Extraction.Types
open CoreModels Aeneas
open Aeneas.Std hiding namespace core alloc
open RustM ControlFlow Error
open Std.Do
set_option linter.dupNamespace false
set_option linter.hashCommand false
set_option linter.unusedVariables false
set_option linter.style.whitespace false
set_option linter.style.setOption false
set_option linter.style.longLine false

/- You can set the `maxHeartbeats` value with the `-max-heartbeats` CLI option -/
set_option maxHeartbeats 1000000

/- You can set the `maxRecDepth` value with the `-max-recdepth` CLI option -/
set_option maxRecDepth 2048
open p3_baby_bear

/-- [p3_baby_bear::baby_bear::{impl core::fmt::Debug for p3_baby_bear::baby_bear::BabyBearParameters}::fmt]:
    Source: 'baby-bear/src/baby_bear.rs', lines 11:31-11:36
    Visibility: public -/
opaque baby_bear.BabyBearParameters.Insts.CoreFmtDebug.fmt
  :
  baby_bear.BabyBearParameters → core.fmt.Formatter → RustM
    ((core.result.Result Unit core.fmt.Error) × core.fmt.Formatter)

/-- [p3_baby_bear::mds::{impl core::fmt::Debug for p3_baby_bear::mds::MDSBabyBearData}::fmt]:
    Source: 'baby-bear/src/mds.rs', lines 12:25-12:30
    Visibility: public -/
opaque mds.MDSBabyBearData.Insts.CoreFmtDebug.fmt
  :
  mds.MDSBabyBearData → core.fmt.Formatter → RustM ((core.result.Result
    Unit core.fmt.Error) × core.fmt.Formatter)

/-- [p3_baby_bear::poseidon1::{impl core::fmt::Debug for p3_baby_bear::poseidon1::BabyBearPoseidonParameters}::fmt]:
    Source: 'baby-bear/src/poseidon1.rs', lines 93:9-93:14
    Visibility: public -/
opaque poseidon1.BabyBearPoseidonParameters.Insts.CoreFmtDebug.fmt
  :
  poseidon1.BabyBearPoseidonParameters → core.fmt.Formatter → RustM
    ((core.result.Result Unit core.fmt.Error) × core.fmt.Formatter)

/-- [p3_baby_bear::poseidon2::{impl core::fmt::Debug for p3_baby_bear::poseidon2::BabyBearInternalLayerParameters}::fmt]:
    Source: 'baby-bear/src/poseidon2.rs', lines 392:9-392:14
    Visibility: public -/
opaque poseidon2.BabyBearInternalLayerParameters.Insts.CoreFmtDebug.fmt
  :
  poseidon2.BabyBearInternalLayerParameters → core.fmt.Formatter → RustM
    ((core.result.Result Unit core.fmt.Error) × core.fmt.Formatter)

