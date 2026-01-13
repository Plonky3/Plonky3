# Releasing Plonky3

This document describes how to create a new release of Plonky3.

## Prerequisites

- Install [release-plz](https://release-plz.dev/docs/usage/installation)
- Set the `GIT_TOKEN` environment variable with a GitHub token that has permission to create PRs

## Creating a Release

1. Ensure your local `main` branch is up-to-date with the remote:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Run the release script:
   ```bash
   ./create_release.sh
   ```

3. This creates a PR with:
   - Version bumps for all crates (in lock-step)
   - Updated changelogs based on conventional commits

4. Review and merge the PR. Once merged, CI automatically publishes all crates to crates.io.

## How It Works

- **Version grouping**: All crates share the same version via `version_group = "plonky3"` in `release-plz.toml`
- **Changelog generation**: Uses [git-cliff](https://git-cliff.org/) configured in `cliff.toml`
- **Version bump detection**: `release-plz` uses [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) to automatically detect the appropriate version bump:
  - Breaking API changes (removed/changed public items) → major bump
  - New public API additions → minor bump
  - Bug fixes, internal changes, docs → patch bump
- **Conventional commits** (optional): If used, commit prefixes like `feat!:` can also signal breaking changes

## Troubleshooting

### CI publish failed

If the publish step fails (e.g., crates.io outage), you can manually re-run the workflow:

1. Go to Actions → Release-plz
2. Click "Run workflow" on the main branch

### release-plz says "no changes to release"

This means there are no conventional commits since the last release tag. Ensure your commits follow the [conventional commits](https://www.conventionalcommits.org/) format.
