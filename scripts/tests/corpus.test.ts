import { describe, expect, test } from "bun:test";
import { join } from "node:path";

import { createNoisyWav, validateManifest, validateScoring } from "../corpus";

const repository = join(import.meta.dir, "../..");

function floatWav(samples: number[]): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 4);
  const view = new DataView(bytes.buffer);
  const text = (offset: number, value: string) => {
    for (const [index, character] of [...value].entries()) bytes[offset + index] = character.charCodeAt(0);
  };
  text(0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 3, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 64000, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 32, true);
  text(36, "data");
  view.setUint32(40, samples.length * 4, true);
  for (const [index, sample] of samples.entries()) view.setFloat32(44 + index * 4, sample, true);
  return bytes;
}

describe("representative corpus", () => {
  test("validates the tracked manifest and separate scoring groups", async () => {
    const manifest = Bun.YAML.parse(
      await Bun.file(join(repository, "benchmarks/corpus/v1/manifest.yaml")).text(),
    );
    const scoring = Bun.YAML.parse(
      await Bun.file(join(repository, "benchmarks/corpus/v1/scoring.yaml")).text(),
    );
    expect(validateManifest(manifest).errors).toEqual([]);
    expect(validateScoring(scoring)).toEqual([]);
  });

  test("rejects paths that escape the ignored corpus root", async () => {
    const manifest = Bun.YAML.parse(
      await Bun.file(join(repository, "benchmarks/corpus/v1/manifest.yaml")).text(),
    ) as Record<string, any>;
    manifest.fixtures[0].audio.path = "../tracked-sensitive-audio.wav";
    expect(validateManifest(manifest).errors.some((error) => error.includes("audio.path"))).toBeTrue();
  });

  test("rejects domain terms absent from an inline reviewed reference", async () => {
    const manifest = Bun.YAML.parse(
      await Bun.file(join(repository, "benchmarks/corpus/v1/manifest.yaml")).text(),
    ) as Record<string, any>;
    manifest.fixtures[0].coverage.domain_terms.push("not in the transcript");
    expect(validateManifest(manifest).errors.some((error) => error.includes("domain term is absent"))).toBeTrue();
  });

  test("generates deterministic noise without changing the WAV envelope", () => {
    const source = floatWav([0.05, -0.05, 0.1, -0.1, 0.02, -0.02]);
    const first = createNoisyWav(source, 20260806, 10);
    const second = createNoisyWav(source, 20260806, 10);
    expect(first).toEqual(second);
    expect(first).not.toEqual(source);
    expect(first.slice(0, 44)).toEqual(source.slice(0, 44));
  });
});
