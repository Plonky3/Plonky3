# Versioning Tools

Because plonky3 has many, many subcrates, it was decided that it was worth putting in effort to automate the release process. The following three scripts when used together make this process significantly more streamlined and much less error-prone.

### But first, why not use existing release tools?

While the tooling available for automating Rust releases is fairly mature at this point, almost all of the tools assume that you are not releasing all of a project's sub-crates in lockstep. Specifically, "lockstep" here means that when a release occurs all individual subcrates will get a release for an identical semver version that matches the project's workspace version. For example, if we perform a `minor` bump and we're currently on `0.2.0`, all crates will be bumped to `0.3.0`.

## Intended Workflow

When it's decided that it's time to do a release, the following should occur:

- Create a PR that updates `CHANGELOG` and bumps all crate versions.
    - On a separate branch, run `lockstep_version_bump.sh` and follow the prompts. This will create a tagged commit and push the branch to remote. This script will create and publish a commit that bumps all crate versions based on how the latest changes affected Semver.
    - Once this is complete, also run `changelog_gen.sh`. This will search for all PR commits and prepend them to `CHANGELOG`. Also don't forget to commit this and push (will automate later).
- Once this PR is merged, we need to next publish this commit to `crates.io`.
    - From `main` on this version bump commit, run `lockstep_publish.sh`. This will do as much validation as possible before finally prompting you to publish all of the sub-crates.

And that's it! No more work is needed on your end.

## The Scripts

### `lockstep_version_bump.sh`

TODO

### `changelog_gen.sh`

TODO

### `lockstep_publish.sh`

TODO