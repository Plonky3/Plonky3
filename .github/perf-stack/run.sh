#!/bin/bash
set -euo pipefail
repo_root=$(pwd)
bench_dir=/tmp/plonky3-stack-measurements
mkdir -p "$bench_dir/src" /tmp/stack-binaries
cp .github/perf-stack/main.rs "$bench_dir/src/main.rs"
write_manifest() {
cat > "$bench_dir/Cargo.toml" <<EOF
[package]
name="stack-measurements"
version="0.1.0"
edition="2024"
[workspace]
[features]
wide=["p3-binary-field/wide-poly"]
column=[]
accumulator=[]
basis=[]
[dependencies]
p3-binary-field={path="$repo_root/binary-field"}
p3-binary-dft={path="$repo_root/binary-dft"}
p3-field={path="$repo_root/field"}
[profile.release]
lto="thin"
codegen-units=1
EOF
}
export RUSTFLAGS='-C target-cpu=native'
git checkout -f perf/poly192-deferred-products
write_manifest
cargo build --release --manifest-path "$bench_dir/Cargo.toml" --features wide
cp "$bench_dir/target/release/stack-measurements" /tmp/stack-binaries/deferred
lscpu
for round in 1 2 3; do
 printf '\nROUND %s\n' "$round"
 /tmp/stack-binaries/deferred
done
