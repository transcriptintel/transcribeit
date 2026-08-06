# Release process

TranscribeIt releases are prepared on `develop`, promoted to `main`, and
published by an immutable `vX.Y.Z` tag. The tag workflow builds the binary; no
local artifact is uploaded manually.

## Supported release archives

The release workflow builds locked Cargo dependencies and publishes:

- Linux x86_64 and arm64 (`tar.gz`);
- macOS Intel and Apple Silicon (`tar.gz`);
- Windows x86_64 (`zip`);
- `checksums-sha256.txt` covering every archive.

Every archive contains the platform binary, `README.md`, and `LICENSE`. FFmpeg
remains an external runtime prerequisite and is not bundled.

## Prepare on `develop`

1. Select the version using Semantic Versioning. Removing a released provider,
   CLI surface, or output contract requires a major release.
2. Update both `Cargo.toml` and the root-package entry in `Cargo.lock`.
3. Update `CHANGELOG.md`, keeping the release marked `Unreleased` until the tag
   is ready.
4. Run the local release gates:

   ```bash
   ./scripts/check.sh
   cargo audit --deny warnings
   cargo build --release --locked
   actionlint .github/workflows/*.yml
   ```

5. Push `develop` and verify both the CI and Security audit workflows. A queued,
   cancelled, or skipped run is not passing evidence.

## Promote and tag

1. Merge `develop` into `main` through the normal reviewed flow and wait for the
   `main` CI and Security audit workflows to pass.
2. Replace `Unreleased` in the changelog heading with the release date and commit
   that final release metadata through `develop` and `main`.
3. From the verified `main` commit, create and push an annotated tag:

   ```bash
   git tag -a v2.0.0 -m "TranscribeIt 2.0.0"
   git push origin v2.0.0
   ```

The release guard rejects tags that are not reachable from `main` or whose
version does not exactly match the Cargo package version. Do not move or reuse a
published tag after a failure; correct the problem and prepare the next version.

## Verify publication

After the Release workflow succeeds:

1. Confirm all five platform archives and `checksums-sha256.txt` are attached to
   the GitHub Release.
2. Verify at least one downloaded archive against the published checksum.
3. Run `transcribeit --version` and a local, non-secret smoke transcription from
   the downloaded archive on its target platform.
4. Review generated release notes against `CHANGELOG.md` and add only factual
   compatibility or migration clarification; never include credentials,
   transcript content, request identifiers, or local paths.
