#!/usr/bin/env bun

import { appendFile, chmod, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";

const args = Bun.argv.slice(2);
const value = (name: string): string | undefined => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const outputDirectory = value("--output-dir");
const model = value("--model") ?? value("--remote-model") ?? "configured";
const provider = value("--provider") ?? "fake";
if (!outputDirectory) throw new Error("missing --output-dir");

if (process.env.FAKE_COUNTER_PATH) {
  await appendFile(process.env.FAKE_COUNTER_PATH, "1\n", { mode: 0o600 });
}
if (model === "fail") {
  console.error("synthetic HTTP 429 response");
  process.exit(7);
}
if (model === "fail-once" && process.env.FAKE_FAIL_ONCE_PATH) {
  const marker = Bun.file(process.env.FAKE_FAIL_ONCE_PATH);
  if (!(await marker.exists())) {
    await Bun.write(marker, "failed\n");
    console.error("synthetic HTTP 429 response");
    process.exit(7);
  }
}
if (model === "timeout") {
  const marker = process.env.FAKE_GRANDCHILD_MARKER_PATH;
  if (!marker) throw new Error("FAKE_GRANDCHILD_MARKER_PATH is required for timeout probes");
  const delay = Number(process.env.FAKE_GRANDCHILD_DELAY_MS ?? "1800");
  Bun.spawn(
    [
      process.execPath,
      "-e",
      `await Bun.sleep(${JSON.stringify(delay)}); await Bun.write(${JSON.stringify(marker)}, "survived\\n");`,
    ],
    { stdout: "ignore", stderr: "ignore" },
  );
  await Bun.sleep(30_000);
}
if (model === "dotenv-probe") {
  const dotenv = await nearestDotEnv(process.cwd());
  if (process.env.FAKE_PARENT_DOTENV_SECRET || dotenv.includes("FAKE_PARENT_DOTENV_SECRET=")) {
    console.error("unrequired parent dotenv value reached the child");
    process.exit(9);
  }
}
if (model === "stderr-spam") console.error("v".repeat(96 * 1024));

await mkdir(outputDirectory, { recursive: true, mode: 0o700 });
const textPath = join(outputDirectory, "fixture.txt");
const manifestPath = join(outputDirectory, "fixture.manifest.json");
await Bun.write(textPath, "synthetic transcript that must never enter the published result\n");
await chmod(textPath, 0o600);
if (model === "malformed-output") {
  await Bun.write(manifestPath, "{synthetic malformed provider manifest\n");
  await chmod(manifestPath, 0o600);
  process.exit(0);
}
await Bun.write(
  manifestPath,
  `${JSON.stringify({
    schema_version: "transcribeit.manifest.v2",
    capabilities: {
      segments: true,
      word_timestamps: false,
      speaker_labels: false,
      language_per_segment: false,
      emotion_per_segment: false,
      native_timestamps: false,
    },
    quality: {
      timing_source: "model_native",
      timing_reliable: false,
      timestamps_clamped: false,
      speaker_source: "unknown",
      warnings: ["synthetic warning"],
    },
    provider_metadata: {
      provider,
      data: provider === "apple-speech"
        ? {
            response: {
              locale: "en-US",
              on_device: true,
              apple_intelligence_available: true,
              asset_install_requested: false,
              asset_managed_by: "macos",
            },
          }
        : {
            request_id: "must-not-be-published",
            signed_url: "https://example.invalid/private",
          },
    },
  }, null, 2)}\n`,
);
await chmod(manifestPath, 0o600);

async function nearestDotEnv(start: string): Promise<string> {
  let directory = start;
  while (true) {
    const candidate = Bun.file(join(directory, ".env"));
    if (await candidate.exists()) return candidate.text();
    const parent = dirname(directory);
    if (parent === directory) return "";
    directory = parent;
  }
}
