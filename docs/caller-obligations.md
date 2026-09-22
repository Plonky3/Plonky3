# Caller obligations

A reduction here is sound only under some conditions.

Most are checked, by a type or by a runtime error.

This page is for the rest: the conditions no crate can check from where it states them.

A caller has to discharge those itself, so read the page once before wiring a backend together.

## The order to prefer

```text
    a type carries it       ->  the compiler rejects the mistake
    a runtime check         ->  the code rejects the mistake, with a typed error
    this page               ->  neither is possible, and the reason is written down
```

Prose in a module doc is not one of the three.

An obligation stated only in a doc comment belongs in an entry below, or it belongs in code.

## Patterns to reuse

These are already in the tree, so reach for one before inventing a fourth.

| Shape | Pattern |
| --- | --- |
| A returned claim a caller could silently drop | Mark the type must-use, which makes a discard a compile error here |
| A value a prover could choose freely once a factor vanished | Reject the degenerate case, since a later absorb pins nothing |
| A security parameter supplied unchecked | Derive it from the object it has to match |
| A statement dimension outside the transcript | Absorb it as length-delimited bytes, never as field elements |
| A public accessor that drives the prover around its own checks | Make the internals private |

## Transcript ordering

A crate can absorb a value, but cannot ask a challenger what it already absorbed, or in what order.

So every rule of the form "bind this before drawing that" falls to the caller.

**Absorb the commitment, and bind the opening points, before opening at prescribed points.**
A challenger cannot be asked what it absorbed, so only a session token would close this.

**Bind the column heights before the sparse point is drawn, normally by committing.**
The jagged reduction has no commitment dependency, and is handed a point drawn before it seeds.

**Bind a caller-fixed opening point to the same sponge before handing it over.**
Such a point emits no transcript step today, so Fiat-Shamir cannot see it.

**Commit the columns a bus argument reads before the argument runs.**
The bus crate has no commitment dependency.

**Bind the width of an opaque observation through the instance label.**
The step records one scalar however many sponge units the value costs.

**Use only hint values the verifier re-derives or checks.**
Hint bytes never enter the sponge, so a prover may choose them after every challenge.

## Properties of the committed witness

A reduction sees field elements.

It cannot see which alphabet they came from, or what the surrounding constraints hold them to.

**Keep the constraint prime-field-valued for every witness the prover can commit to.**
The pinned fold has a large kernel over a wider alphabet, and patterns inside it are invisible.

The premise must come from the commitment alphabet, so a booleanity constraint will not do.

**Constrain every query count to the weight the lookup declares.**
That weight is the only input to the height bound, and one set too low lets a count wrap.

**Constrain exclusive-branch flags to be boolean and to sum to at most one.**
Two flags at once give a denominator the soundness argument never covers.

**Zero every entry past the message prefix before a padded encode.**
An implementation that skips the tail then produces a different codeword.

## Parameters whose other side is invisible

**Give the commitment a digest at least twice the security level in bits.**
The commitment trait reports no width, and the schedule is derived before any scheme exists.

**Make sure a rejection predicate can admit as many candidates as are asked of it.**
The predicate is an opaque closure, and a blanket bound would be wrong for one admitting repeats.

## Minimum non-degenerate shapes

A reduction with no rounds samples no challenge.

Sometimes that is a bug, sometimes the whole truth, and from outside they look identical:

```text
    degenerate  ->  no challenge is sampled AND nothing downstream pins the result
    trivial     ->  no challenge is sampled because there is nothing left to fold
```

Only the first deserves a floor.

A degenerate shape is how a separation test can pass against every transcript it is given.

A trivial shape is bracketed by the same checks a longer run is, so a floor only breaks it.

So the question to ask of a small shape is not whether it is small, but **what is left unpinned**.

Name the value nobody checks, and if you cannot name one, impose no floor.

| Reduction | Smallest shape | Verdict |
| --- | --- | --- |
| Generic-degree sumcheck | no rounds | **Trivial.** The summand is a constant, and the claimed sum is seeded before the surviving value is opened |
| Hiding sumcheck | no rounds | **Rejected.** The shape validator refuses it |
| Jagged reduction | live area of one | **Trivial.** One dense cell has no interior to test, and the claim names that cell |
| Binary PCS opening protocol | no claims | **Trivial.** A run that claims nothing proves nothing, and asserts nothing |
| Binary WHIR profile | no opened position | **Rejected.** Queries alone test proximity, so opening none accepts any codeword |
| Plain sumcheck rounds | no rounds | **Trivial**, as above. The surrounding protocol owns any floor |

## Known gaps

Recorded rather than fixed, and tracked against
[issue 2271](https://github.com/Plonky3/Plonky3/issues/2271).

- Prescribed-point opening returns a vector, and the must-use lint does not reach inside one.
- The rejecting and plain challenge draws record the same step, so no seed parts them.
- The layout verifier takes any field element where only the challenge it drew belongs.
- Ring switching, its bit-level variant, and the skip domain each admit a zero-round shape.
- The commitment trait reports no digest width, so the digest entry above cannot be checked.
- The multilinear commitment contract claims conformance tests that cover only the other trait.
