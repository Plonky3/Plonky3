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
short=[]
fused=[]
power=[]
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
for spec in 'before:bad2eeb7' 'after:f045bc6e'; do
 IFS=: read -r label revision <<< "$spec"
 git checkout -f "$revision"
 write_manifest
 features=wide
 
 cargo build --release --manifest-path "$bench_dir/Cargo.toml" --features "$features"
 cp "$bench_dir/target/release/stack-measurements" "/tmp/stack-binaries/$label"
done
lscpu
for round in 1 2 3; do
 if [[ "$round" == 2 ]]; then labels=(after before); else labels=(before after); fi
 for label in "${labels[@]}"; do
  printf '\nROUND %s REVISION %s\n' "$round" "$label"
  "/tmp/stack-binaries/$label"
 done
done
