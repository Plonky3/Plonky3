-- [p3_baby_bear]: external types.
-- HAND-WRITTEN, from hax's seed of Extraction/TypesExternal_Template.lean.
-- The generated code imports it by this module name. extract.sh fails if the
-- regenerated template declares different names than this file does.
import Aeneas
import CoreModels
-- The dependency interface: the scoped extractions of p3-monty-31 and p3-mds
-- (generated/), plus the hand-written rest (assumptions/Interface/). `Interface.P3Monty31Missing`
-- imports `P3Monty31.Extraction` and the p3-field/p3-poseidon{1,2} files.
import P3Mds.Extraction
import Interface.P3Monty31Missing
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

