#!/usr/bin/env bash

# Publish all subcrates to `crates.io` once we merged a version bump PR back to main.

# Note that the intended workflow of this script is:
# - Have a PR of successful run of `lockstep_version_bump.sh` merged back into `main`.
# - Run this script on the commit that got merged back into `main`.

# If a version bump occurred on this commit (highest version tag is present on this commit), then this will report that nothing has changed since the last bump (because we just bumped on this commit).
changed_res=$(cargo workspaces changed --error-on-empty)

if ! $?; then
    num_changed=$(echo "$changed_res" | wc -l)
    echo "${num_changed} crates have changed since the last release."
    echo "Do you want to publish a release now? (y/n)"

    read -r input
    if [ "$input" = "y" ]; then
        echo "Publishing to crates.io..."
    else
        exit 0
    fi
else
    echo "The latest version tag is not on this commit. Run \`lockstep_version_bump.sh\` to create a commit for publishing."
    exit 1
fi

# User green-lighted publishing.
# Because a failure during publishing could result in a desync between local and remote (and thus a big headache), we're going to do a dry run first in order to detect any errors during publishing.
if ! cargo workspaces publish --dry-run --no-git-push --allow-branch main; then
    echo "crates.io publishing dry run failed."
    exit 1
fi

# Publishing dry run succeeded. Do a real publish now.
