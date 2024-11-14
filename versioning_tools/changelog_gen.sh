#!/usr/bin/env bash

# Convenience script for prepending to the changelog after a version bump.
# Also has an additional check to prevent double appending a changelog for a version to `CHANGELOG`.
# 
# Intended to be used in the same PR that bumps the versions.

curr_dir=$(realpath "$(dirname "$0")")
changelog_path="${curr_dir}/../CHANGELOG"

latest_local_release_tag=$(git tag -l | grep -E 'v[0-9]+\.[0-9]+\.[0-9]+' | sort -r | head -n 1)
latest_local_release_tag_no_v=$(echo "$latest_local_release_tag" | sed -E 's/v(.*)/\1/')

# Check if there already is an entry in the changelog for this version.
if [ "$(grep -c "\[$latest_local_release_tag_no_v\]" < "$changelog_path")" -gt 0 ]; then
    echo "Version ${latest_local_release_tag} already has a entry in the changelog. Manually remove it before prepending to it with this script."
    exit 1
fi

git cliff -p "$changelog_path" --latest

echo "Changelog successfully updated for ${latest_local_release_tag}."
