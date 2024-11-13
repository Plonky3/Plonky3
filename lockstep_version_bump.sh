#!/usr/bin/env bash

# Check if token is setup and display a message if not.

# We are currently incrementing versions with lockstep, which means that all subcrates will always be at the same version (eg. `0.x.0` is the crate version that should be used for all crates when we target this version).

# When we increment the version, we are always going to be following Semver of which ever package had the most "significant" bump. So for example, if one package had a major bump but all of the rest only had a patch bump, then all packages would receive a major bump.

# Check if a binary in installed and exit early if it's missing.
# $1 --> Binary name
function tool_installed_check () {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "\"${1}\" not found! Make sure it's installed before running this script."
        exit 1
    fi
}

# Tool install check.
tool_installed_check "cargo"
tool_installed_check "cargo-workspaces"
tool_installed_check "cargo-semver-checks"

# First we need to check if a version bump occurred that was never published. We don't want to accidentally bump a version and never publish it.
latest_local_release_tag=$(git tag -l | grep -E 'v[0-9]+\.[0-9]+\.[0-9]+' | sort -r | head -n 1)
all_local_subcrates_name_and_versions=$(cargo workspaces list -l)

all_local_subcrate_versions=$(echo "$all_local_subcrates_name_and_versions" | sed -E 's/.* v([0-9]+.[0-9]+.[0-9]+).*/\1/')
# First check that all subcrates are locksteped to the same version. If this isn't the case, then something is wrong and we should stop.
# Use `awk` to check that all crate versions are the same string.
if  [ "$(echo "$all_local_subcrate_versions" | uniq | wc -l)" -gt 1 ]; then
    echo "Something is wrong and all local subcrates are not on the same version!"
    echo "$all_local_subcrates_name_and_versions"
    echo "Aborting!"

    exit 1
fi

echo "$all_local_subcrates_name_and_versions"

# Now that we know that all local subcrates are locksteped to the same version, we need to also ensure that all published (remote) crates are on the same version.
for crate_name in $(cargo workspaces list)
do
    echo "Checking published version for ${crate_name}..."

    local_ver=$(echo "$all_local_subcrates_name_and_versions" | grep -E ".*$crate_name .*" | sed -E 's/.*([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
    published_ver=$(cargo search -q "$crate_name" | grep "$crate_name =" | sed -E 's/.* = "([0-9]+\.[0-9]+\.[0-9]+)".*/\1/')

    # Handle the case where this is a new crate that is not yet published.
    if [ "$published_ver" == "" ]; then
        echo "${crate_name} not yet published to crates.io. Will publish an initial version for it."
    elif [ ! "$local_ver" == "$published_ver" ]; then
        echo "The crate \"${crate_name}\" has a different published version (${published_ver}) than the current local version (${local_ver})."
        echo "This script relies that all sub-crates are bumped in lockstep, and if one crate does not match its remote, the script's core assumptions break down."
        echo "You're going to have to manually sync all desynced sub-crates to get their local versions to match the published version."
        echo "Aborting..."

        exit 1
    fi
done

# Now all local and published versions are currently synced. To get the most recent current version.
latest_published_version=$(cargo search -q uni-stark | sed -E 's/.* = "([0-9]+.[0-9]+.[0-9]+)".*/\1/')
major_version=$(echo "$latest_published_version" | sed -E "s/([0-9]+).*/\1/")

echo "Checking for the most significant semver change across all crates. This may take some time..."
# semver_check_out=$(cargo workspaces exec --no-bail cargo semver-checks 2>&1 | grep "Summary")
semver_check_out=$(cat semver_test_data.txt)

major_bumps_suggested=$(echo "$semver_check_out" | grep -ce "new major version")
minor_bumps_suggested=$(echo "$semver_check_out" | grep -ce "new minor version")
patch_bumps_suggested=$(echo "$semver_check_out" | grep -c "Summary no semver update required")

# Check if we would normally perform a `Major` bump but won't because the current `Major` version `0`. (https://semver.org/#spec-item-4) 
major_bump_suppressed=0

# Man... We sure love Bash here... Looks at this beautiful line below.
[ "$major_version" -eq 0 ] && [ "$major_bumps_suggested" -gt 0 ] && major_bump_suppressed=1

# Note: Because the rules for Semver are a bit different when major is `0`, we're going to override suggesting a `Major` change if the major version is `0`.
if [ "$major_bumps_suggested" -gt 0 ] && [ "$major_version" -gt 0 ]; then
    patch_bump_type="major"

# We want to downgrade a `Major` bump into a `minor` bump if the current `Major` version is `0`.
elif [ "$minor_bumps_suggested" -gt 0 ] || [ "$major_bump_suppressed" ]; then
    patch_bump_type="minor"
elif [ "$patch_bumps_suggested" -gt 0 ]; then
    patch_bump_type="patch"
else
    patch_bump_type="none"
fi

if [ $patch_bump_type == "none" ]; then 
    echo "No crates need to be bumped."
else
    # We can perform a bump.
    echo "${patch_bump_type} bump suggested. (Major: ${major_bumps_suggested}, Minor: ${minor_bumps_suggested}, Patch: ${patch_bumps_suggested})"

    if [ "$major_bump_suppressed" ]; then
        echo "Note: Even though there are breaking changes since the last release, because the current major version is \"0\", we are going to suggest a minor bump instead. (See https://semver.org/#spec-item-4)"
    fi

    echo "Proceed with a ${patch_bump_type} lockstep bump? (y/n)"
    read -r input

    if [ ! "$input" == "y" ]; then
        echo "Overriding semver suggested patch type. What kind of bump should be done instead? (\"major\" | \"minor\" | \"patch\")"
        read -r patch_bump_type

        case $patch_bump_type in
            major | minor | patch)
                # Valid input. Do nothing.
                ;; 

            *)
                echo "${patch_bump_type} not valid input! Exiting!"
                exit 1
                ;;
        esac
    fi

    # Now we have a valid bump type. Apply it.
    echo "Performing a ${patch_bump_type} bump..."
    cargo workspaces version -y --allow-branch main "${patch_bump_type}"
fi

# So at this point, we have either:
#   - Performed a version bump and we have a new lockstep version bump ready for publishing.
#   - No version bump was possible (which means that the workspace has no changes to publish).

# If a version bump occurred, then this will report that nothing has changed since the last bump (because we just bumped on this commit).
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
fi

# User green-lighted publishing.
# Because a failure during publishing could result in a desync between local and remote (and thus a big headache), we're going to do a dry run first in order to detect any errors during publishing.
if ! cargo workspaces publish --dry-run --no-git-push --allow-branch main; then
    echo "crates.io publishing dry run failed."
    exit 1
fi

# Publishing dry run succeeded. Do a real publish now.
