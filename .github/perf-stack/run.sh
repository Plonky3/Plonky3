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
for spec in 'before:eab7f0e3662500cde47cb0b6dad61afc99e3deac:' 'aes:perf/rijndael-neon-products:' 'neon:perf/poly64-neon-butterfly:' 'wide:perf/poly192-wide-packing:wide,basis' 'column:perf/poly64-column-dot:wide,basis,column'; do
 IFS=: read -r label revision features <<< "$spec"
 git checkout -f "$revision"
 write_manifest
 feature_args=()
 # The feature exists only in the last layers; omit its declaration on lower revisions.
 if [[ "$label" == before || "$label" == aes || "$label" == neon ]]; then
  sed -i '/wide=\[/d' "$bench_dir/Cargo.toml"
 fi
 if [[ -n "$features" ]]; then feature_args=(--features "$features"); fi
 cargo build --release --manifest-path "$bench_dir/Cargo.toml" "${feature_args[@]}"
 cp "$bench_dir/target/release/stack-measurements" "/tmp/stack-binaries/$label"
done
lscpu
for round in 1 2 3; do
 for label in before aes neon wide column; do
  printf '\nROUND %s REVISION %s\n' "$round" "$label"
  "/tmp/stack-binaries/$label"
 done
done
