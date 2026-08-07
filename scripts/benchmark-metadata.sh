#!/usr/bin/env bash
set -euo pipefail

fixture="${1:?usage: benchmark-metadata.sh FIXTURE}"
test -f "$fixture" || {
  echo "Benchmark fixture not found: $fixture" >&2
  exit 1
}
command -v jq >/dev/null || {
  echo "jq is required to emit benchmark metadata" >&2
  exit 1
}

case "$(uname -s)" in
  Darwin)
    cpu="$(sysctl -n machdep.cpu.brand_string)"
    logical_cores="$(sysctl -n hw.logicalcpu)"
    memory_bytes="$(sysctl -n hw.memsize)"
    ;;
  Linux)
    cpu="$(awk -F: '/model name/{sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    logical_cores="$(getconf _NPROCESSORS_ONLN)"
    memory_bytes="$(awk '/MemTotal/{print $2 * 1024; exit}' /proc/meminfo)"
    ;;
  *)
    cpu="unknown"
    logical_cores="unknown"
    memory_bytes="unknown"
    ;;
esac

fixture_bytes="$(wc -c < "$fixture" | tr -d ' ')"
fixture_sha256="$(shasum -a 256 "$fixture" | awk '{print $1}')"
git_commit="$(git rev-parse HEAD)"
if git diff --quiet && git diff --cached --quiet; then
  worktree_dirty=false
else
  worktree_dirty=true
fi

jq -n \
  --arg captured_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg git_commit "$git_commit" \
  --argjson worktree_dirty "$worktree_dirty" \
  --arg cpu "$cpu" \
  --arg logical_cores "$logical_cores" \
  --arg memory_bytes "$memory_bytes" \
  --arg os "$(uname -s)" \
  --arg kernel "$(uname -r)" \
  --arg architecture "$(uname -m)" \
  --arg rustc "$(rustc --version)" \
  --arg ffmpeg "$(ffmpeg -version | head -n 1)" \
  --arg fixture "$fixture" \
  --arg fixture_bytes "$fixture_bytes" \
  --arg fixture_sha256 "$fixture_sha256" \
  '{
    schema_version: "transcribeit.benchmark-environment.v1",
    captured_at_utc: $captured_at_utc,
    git: {commit: $git_commit, worktree_dirty: $worktree_dirty},
    machine: {
      cpu: $cpu,
      logical_cores: ($logical_cores | tonumber? // $logical_cores),
      memory_bytes: ($memory_bytes | tonumber? // $memory_bytes),
      os: $os,
      kernel: $kernel,
      architecture: $architecture
    },
    tools: {rustc: $rustc, ffmpeg: $ffmpeg},
    fixture: {
      path: $fixture,
      bytes: ($fixture_bytes | tonumber),
      sha256: $fixture_sha256
    }
  }'
