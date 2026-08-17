import { resolve } from "node:path";

import { repositoryRoot } from "./config";
import type { EvaluationRootBinding } from "./runner_paths";
import type { MatrixEntry } from "./types";

const systemEnvironmentNames = [
  "PATH",
  "TMPDIR",
  "TMP",
  "TEMP",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TZ",
  "SYSTEMROOT",
  "WINDIR",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "DYLD_LIBRARY_PATH",
  "DYLD_FRAMEWORK_PATH",
  "DYLD_FALLBACK_LIBRARY_PATH",
  "DYLD_FALLBACK_FRAMEWORK_PATH",
] as const;

export function childEnvironment(
  entry: MatrixEntry,
  evaluationRoot: EvaluationRootBinding,
): Record<string, string> {
  const environment: Record<string, string> = {};
  for (const name of systemEnvironmentNames) {
    const value = process.env[name];
    if (value !== undefined) environment[name] = value;
  }
  for (const name of entry.required_env) {
    const value = process.env[name];
    if (value === undefined) continue;
    environment[name] = name === "MODEL_CACHE_DIR"
      ? evaluationRoot?.absolutePath ?? resolve(repositoryRoot(), value)
      : value;
  }
  return environment;
}
