#!/bin/bash

# Script to automate release generation for `Plonky3`.
#
# This script creates a release PR that, when merged, triggers CI to publish to crates.io.
#
# Usage:
#   ./create_release.sh                # let release-plz compute the next version
#   ./create_release.sh 0.7.0-rc.1     # manually cut a release candidate
#
# How it works (no argument):
# 1. `release-plz release-pr` analyzes commits since the last release
# 2. Determines version bumps based on conventional commits (respecting version_group for lock-step)
# 3. Generates changelogs using cliff.toml
# 4. Creates a PR with the "release" label containing all changes
# 5. When that PR is merged, CI runs `release-plz release` to publish to crates.io
#
# Release candidates (rc argument):
# release-plz never derives an `-rc` version on its own, so an rc must be set by
# hand. Only rc versions may be passed here — stable bumps are always left to
# release-plz's automatic semver. The rc path edits the shared workspace version
# and opens the "release"-labelled PR directly; once published, plain reruns
# (no argument) advance rc.N -> rc.N+1 automatically.

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

rc_version="${1:-}"

# release-plz opens the release PR against the branch you run it on, so release
# from whichever branch is checked out: main, or a vX.Y.Z version/rc line.
branch=$(git symbolic-ref --short HEAD)

# Ensure the local release branch is up-to-date with its remote.
git fetch origin "$branch"
local_head=$(git rev-parse HEAD)
remote_head=$(git rev-parse "origin/$branch")

if [ "$local_head" != "$remote_head" ]; then
  echo "Error: Local '$branch' is not up-to-date with 'origin/$branch'."
  echo "Please run: git checkout $branch && git pull"
  exit 1
fi

if [ -n "$rc_version" ]; then
  if [[ "$rc_version" != *rc* ]]; then
    echo "Warning: '$rc_version' is not a release candidate (no 'rc')."
    echo "Only rc versions may be set manually; omit the argument and let"
    echo "release-plz compute stable versions automatically."
    exit 1
  fi

  check_binary_installed "gh"

  pr_branch="release-plz-$rc_version"
  echo "Cutting release candidate '$rc_version' from '$branch'..."

  # Every crate inherits `version.workspace = true`, so bumping the single
  # [workspace.package] version sets the whole workspace in lock-step.
  sed -i.bak -E "s/^version = \".*\"/version = \"$rc_version\"/" Cargo.toml
  rm -f Cargo.toml.bak

  git switch -c "$pr_branch"
  git commit -m "chore: release $rc_version" Cargo.toml
  git push -u origin "$pr_branch"

  GH_TOKEN="$GIT_TOKEN" gh pr create \
    --base "$branch" \
    --head "$pr_branch" \
    --title "chore: release $rc_version" \
    --body "Release candidate \`$rc_version\` (manually cut)." \
    --label release
  exit 0
fi

echo "Creating release PR for '$branch'..."
release-plz release-pr
