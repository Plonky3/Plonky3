# Plonky3 bus

Field-generic primitives for a direction-aware multiset bus.

The crate plans mixed-height bus layouts, materializes fingerprint factors, reduces their products with GKR, and prepares read-only memory checks.

Its debugger replays symbolic declarations and reports unmatched tuples with their source rows.

The standalone reduction returns unauthenticated terminal claims.

`p3-multi-stark` binds them to committed trace columns with a composition sumcheck and prescribed-point openings.

## The declaration surface

An AIR declares one tuple contribution by naming a channel, picking a side of the multiset equality, listing its payload expressions in slot order, and saying which rows contribute.

```rust,ignore
/// Every table that touches memory imports this one constant.
const MEMORY: BusName<'static> = BusName::new("memory");

builder.push_bus_interaction(
    MEMORY,
    BusDirection::Pull,
    [address, count, value],
    BusActivation::Boolean(is_load),
);
```

Nothing here knows what a machine puts on a channel.

Addresses, opcodes, program counters and registers belong to the repository that owns the instruction set.

What belongs here is a name, a side, a tuple and a row.

### Names are typed, and identity is the verifier's

A channel name is one to 64 bytes: an ASCII letter, then ASCII letters, digits, `_`, `.` or `-`.

The constructor is a `const fn`, so a machine names each channel once in a const item the compiler checks, and every table imports that constant.

A chip that mistypes the constant fails to resolve a name, rather than quietly opening a second channel whose tuples never match.

The alphabet is not tidiness.

The transcript separator length-prefixes every name, so arbitrary bytes would still encode injectively.

What they would not do is print injectively.

Two names that render the same way are still two multisets, and a diagnostic report shows them under one heading.

Restricting to ASCII rules out every such pair.

A name is not an identity.

Planning collects the names a statement declares, sorts them, and assigns each one a nonzero identity encoded into reserved slots of every tuple.

That assignment is a pure function of the symbolic declarations the verifier holds, so a prover cannot choose which channel its tuples land on.

Planning also rechecks every name it is handed, so a profile assembled by hand rather than through the builder is rejected before it reaches the transcript.

### Boundary flushes

A boundary flush happens once for a whole table rather than once per row: at row zero, or at the final row.

An initial state pushed once, and a final state pulled once, are its two uses.

The indicator is a backend row selector rather than a witness column.

Over the Boolean hypercube this bus is defined on it is zero or one on every row, so a boundary declaration owes no Booleanity constraint and the builder emits none.

A caller-supplied selector still costs one constraint of twice its degree.

The factor degree is the same either way.

Two consequences are worth knowing.

A table whose entire content is boundary declarations leaves the batched zerocheck nothing to prove and is refused, though any table a machine would really write has local constraints.

And a boundary declaration still occupies a block of its table's full height, whose other rows contribute the product identity, so it costs `2^k` leaves rather than one.

### Tuple widths

A channel's payload width is fixed by its first declaration, and every later declaration on that channel must agree.

Two chips disagreeing on a channel's arity is therefore a planning error that names the channel and both widths, rather than a multiset that silently never balances.

Widths are equalized across channels rather than per channel.

Writing `p` for the widest payload in the statement and `n` for the number of channels, every tuple is laid out as

```text
    [ payload 0 .. w )  [ zero w .. p )  [ identity bits p .. p + ceil(log2(n+1)) )  [ zero .. ]
```

padded to the next power of two, which is the fingerprint table the sampled point indexes.

A narrow channel therefore pays the widest channel's slots in table size, but not in evaluation cost.

One declaration's leaf factor is an inner product over its own payload width plus a constant the plan settles once.

What the padded width does cost is `log2` of itself in challenge coordinates.

A statement with a 40-slot instruction channel and three narrower ones lands on a 64-slot table and six coordinates.

### Diagnostics

The debugger replays declarations against concrete traces and reports the tuples that do not balance, with the bus, table, declaration and row behind each one.

It is development infrastructure.

No proving or verifying path reads it, and multiplicities are integers there rather than field elements, so equal counts cancel exactly.

It sits behind the default-on `diagnostics` feature, and a production build can turn it off without changing a proof, a transcript, or the optimized proving path.

## How this compares to the references

### leanVM

leanVM (`github.com/leanEthereum/leanVM`) is the closest reference for this shape: unordered rows, per-table degree-2 constraints, one shared bus balanced by a grand product.

**Channel naming.**
leanVM has one bus and names its channels by a constant field element placed at coordinate zero of the tuple, the generator powers `g^0`, `g^1` and `g^2`.

The constant is hand-written, and nothing checks that a table uses the right one.

Here the identity is assigned by the plan from the declared names, so a table cannot pick its own separator and two channels cannot collide.

We pay `ceil(log2(n+1))` slots for that where leanVM pays exactly one.

The difference only changes the padded width when the widest payload sits within those few slots of a power of two, so verifier-assigned identity is worth it.

**Per-channel arity.**
leanVM has none: one global assertion that the widest tuple fits the slot count, with shorter tuples implicitly zero-padded.

Its bytecode channel genuinely carries 9-slot and 11-slot tuples, sound only because the missing slots happen to be zero on both sides, which nothing in that codebase checks.

Our width agreement check is exactly the hole that leaves.

**Boundary flushes.**
This is where leanVM is ahead of us.

Its boundary is a block of log-height zero, one leaf in the product tree with all-constant coordinates, rather than a full-height block gated by a row selector.

Ours costs `2^k` leaves where `2^k - 1` of them are the identity.

leanVM pays for it elsewhere: a height-zero block has no row variables, so it is not settled by its table's sumcheck and instead keeps its own column claims at a prescribed point.

Adopting it here would mean a second opening path for boundary declarations, which is a change to the composition rather than to the declaration surface.

It is the right next step if boundary flushes ever sit on tall tables.

**Multiplicity.**
leanVM threads read counts as a multiplicative generator chain and proves every count nonzero with a third grand product, which keeps the push-side counter off the trace.

Our read-only memory argument uses the same generator-orbit idea.

Direction is structural in both: leanVM by which vector a flush lands in, here by explicit metadata, which is what keeps push and pull distinct in characteristic two where negation is the identity.

**Expression generality.**
A leanVM coordinate is a restricted form of degree at most two over the table's columns.

Ours is an arbitrary symbolic expression, priced through the composition degree rather than forbidden.

leanVM also has a free virtual index coordinate, and we have no equivalent, so a machine that wants a row index in a tuple commits a column for it.

### binius64

binius64 (`github.com/binius-zk/binius64`) has no bus, no channels and no flush tuples.

Its nearest construction is logUp\*, a single-column indexed lookup confined to the protocol internals with no frontend surface.

Tuple arity there is always one, and multi-column lookups are done by registering several lookers against one table and combining their claims at the call site.

There is nothing in it to borrow for tuple ergonomics.
