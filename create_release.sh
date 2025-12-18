#!/bin/bash

# Script to automate release generation for `Plonky3`.
#
# This script creates a release PR that, when merged, triggers CI to publish to crates.io.
#
# How it works:
# 1. `release-plz release-pr` analyzes commits since the last release
# 2. Determines version bumps based on conventional commits (respecting version_group for lock-step)
# 3. Generates changelogs using cliff.toml
# 4. Creates a PR with the "release" label containing all changes
# 5. When that PR is merged, CI runs `release-plz release` to publish to crates.io

set -euo pipefail

check_binary_installed() {
  local binary_name="$1"
  if ! command -v "$binary_name" &> /dev/null; then
    echo "Error: $binary_name is not installed."
    exit 1
  fi
}

if [ -z "${GIT_TOKEN:-}" ]; then
  echo "Error: GIT_TOKEN is not set. release-plz requires it to create PRs."
  exit 1
fi

check_binary_installed "release-plz"

# Ensure local main is up-to-date with remote
git fetch origin main
local_main=$(git rev-parse main)
remote_main=$(git rev-parse origin/main)

if [ "$local_main" != "$remote_main" ]; then
  echo "Error: Local 'main' is not up-to-date with 'origin/main'."
  echo "Please run: git checkout main && git pull"
  exit 1
fi

echo "Creating release PR..."
release-plz release-pr
