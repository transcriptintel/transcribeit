import { isMap } from "./state";

export const sha256Pattern = /^[a-f0-9]{64}$/;
export const gitHashPattern = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/;
export const safeIdPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
export const safeRevisionPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
export const safeLocalePattern = /^[A-Za-z0-9_-]{1,35}$/;
export const safeEnvironmentNamePattern = /^[A-Z][A-Z0-9_]*$/;

export function isIsoTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

export function isFiniteRange(value: unknown, minimum: number, maximum = Number.MAX_VALUE): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= minimum && value <= maximum;
}

export function isSafeIntegerRange(value: unknown, minimum: number, maximum = Number.MAX_SAFE_INTEGER): value is number {
  return Number.isSafeInteger(value) && Number(value) >= minimum && Number(value) <= maximum;
}

export function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

export function isNullableBoolean(value: unknown): value is boolean | null {
  return value === null || typeof value === "boolean";
}

export function isNullableFiniteRange(value: unknown, minimum = 0): value is number | null {
  return value === null || isFiniteRange(value, minimum);
}

export function isNullableSafeIntegerRange(value: unknown, minimum = 0): value is number | null {
  return value === null || isSafeIntegerRange(value, minimum);
}

export function hasOnlyStrings(value: unknown, allowEmpty = true): value is string[] {
  return Array.isArray(value) && (allowEmpty || value.length > 0) &&
    value.every((item) => typeof item === "string" && item.length > 0);
}

export function rejectUnexpected(
  value: Record<string, unknown>,
  allowed: readonly string[],
  label: string,
  errors: string[],
): void {
  const unexpected = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unexpected.length) errors.push(`${label} contains unsupported fields: ${unexpected.join(", ")}`);
}

export function requireMap(value: unknown, label: string, errors: string[]): Record<string, unknown> | undefined {
  if (isMap(value)) return value;
  errors.push(`${label} must be a mapping`);
  return undefined;
}

export function containsForbiddenString(value: unknown): boolean {
  if (typeof value === "string") {
    return (
      /file:\/\/|Bearer\s|-----BEGIN|[?&](?:token|key|signature)=/i.test(value) ||
      /(?:^|[\s"'=(])\/(?:Users|home|root|tmp|var|private|Volumes|Applications|Library|System|opt|etc|usr|mnt|srv|bin|sbin|dev|run|nix)\//.test(value) ||
      /(?:^|[\s"'=(])[A-Za-z]:[\\/]/.test(value) ||
      /(?:^|[\s"'=(])\\\\[^\\\s]+\\/.test(value) ||
      /\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{12,}\b/.test(value) ||
      /\bAIza[0-9A-Za-z_-]{20,}\b/.test(value) ||
      /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/.test(value)
    );
  }
  if (Array.isArray(value)) return value.some(containsForbiddenString);
  return isMap(value) && Object.values(value).some(containsForbiddenString);
}
