#!/usr/bin/env bun

import { appendFile, chmod, mkdir } from "node:fs/promises";
import { join } from "node:path";

const args = Bun.argv.slice(2);
const value = (name: string): string | undefined => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};
const outputDirectory = value("--output-dir");
const model = value("--model") ?? value("--remote-model") ?? "configured";
if (!outputDirectory) throw new Error("missing --output-dir");

if (process.env.FAKE_COUNTER_PATH) {
  await appendFile(process.env.FAKE_COUNTER_PATH, "1\n", { mode: 0o600 });
}
if (model === "fail") {
  console.error("synthetic HTTP 429 response");
  process.exit(7);
}

await mkdir(outputDirectory, { recursive: true, mode: 0o700 });
const textPath = join(outputDirectory, "fixture.txt");
const manifestPath = join(outputDirectory, "fixture.manifest.json");
await Bun.write(textPath, "synthetic transcript that must never enter the published result\n");
await chmod(textPath, 0o600);
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
      timing_source: "none",
      timing_reliable: false,
      timestamps_clamped: false,
      speaker_source: "none",
      warnings: ["synthetic warning"],
    },
    provider_metadata: {
      provider: "fake",
      data: {
        request_id: "must-not-be-published",
        signed_url: "https://example.invalid/private",
      },
    },
  }, null, 2)}\n`,
);
await chmod(manifestPath, 0o600);
