export type SupervisedProcessResult = {
  exitCode: number;
  stderr: string;
  wallMs: number;
  timedOut: boolean;
  stderrTruncated: boolean;
  interrupted: boolean;
  aborted: boolean;
};

type HandledSignal = "SIGINT" | "SIGTERM";
type ActiveProcess = { interrupt: () => void; abort: () => void };

const stderrHeadBytes = 32 * 1024;
const stderrTailBytes = 32 * 1024;

export class SubprocessSupervisor {
  readonly #active = new Map<number, ActiveProcess>();
  #interruptedSignal: HandledSignal | null = null;
  #internallyAborted = false;
  readonly #onSigint = (): void => this.cancel("SIGINT");
  readonly #onSigterm = (): void => this.cancel("SIGTERM");

  constructor() {
    process.on("SIGINT", this.#onSigint);
    process.on("SIGTERM", this.#onSigterm);
  }

  get interruptedSignal(): HandledSignal | null {
    return this.#interruptedSignal;
  }

  get internallyAborted(): boolean {
    return this.#internallyAborted;
  }

  cancel(signal: HandledSignal): void {
    this.#interruptedSignal ??= signal;
    for (const active of this.#active.values()) active.interrupt();
  }

  abort(): void {
    this.#internallyAborted = true;
    for (const active of this.#active.values()) active.abort();
  }

  dispose(): void {
    process.off("SIGINT", this.#onSigint);
    process.off("SIGTERM", this.#onSigterm);
  }

  async run(
    command: string[],
    options: { cwd: string; env: Record<string, string>; timeoutMs: number },
  ): Promise<SupervisedProcessResult> {
    if (this.#interruptedSignal) return interruptedWithoutSpawn();
    if (this.#internallyAborted) return abortedWithoutSpawn();
    const started = performance.now();
    const child = Bun.spawn(command, {
      cwd: options.cwd,
      env: options.env,
      stdout: "ignore",
      stderr: "pipe",
      detached: process.platform !== "win32",
    });
    let timedOut = false;
    let interrupted = false;
    let aborted = false;
    const kill = (): void => killProcessTree(child);
    this.#active.set(child.pid, {
      interrupt: () => {
        interrupted = true;
        kill();
      },
      abort: () => {
        aborted = true;
        kill();
      },
    });
    const stderrPromise = readBounded(child.stderr, stderrHeadBytes, stderrTailBytes);
    const timeout = setTimeout(() => {
      timedOut = true;
      kill();
    }, options.timeoutMs);
    let exitCode: number;
    try {
      exitCode = await child.exited;
    } finally {
      clearTimeout(timeout);
      this.#active.delete(child.pid);
    }
    const stderr = await stderrPromise;
    return {
      exitCode,
      stderr: stderr.text,
      wallMs: performance.now() - started,
      timedOut,
      stderrTruncated: stderr.truncated,
      interrupted,
      aborted,
    };
  }
}

function killProcessTree(child: { pid: number; kill(signal: number | NodeJS.Signals): void }): void {
  if (process.platform === "win32") {
    try {
      child.kill("SIGKILL");
    } catch {
      // The child may already have exited.
    }
    return;
  }
  try {
    process.kill(-child.pid, "SIGKILL");
  } catch {
    try {
      child.kill("SIGKILL");
    } catch {
      // The process group may already be gone.
    }
  }
}

async function readBounded(
  stream: ReadableStream<Uint8Array>,
  headLimit: number,
  tailLimit: number,
): Promise<{ text: string; truncated: boolean }> {
  const reader = stream.getReader();
  const headChunks: Uint8Array[] = [];
  let headBytes = 0;
  let tail = new Uint8Array();
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    const headRemaining = Math.max(0, headLimit - headBytes);
    if (headRemaining) {
      const chunk = value.byteLength > headRemaining ? value.subarray(0, headRemaining) : value;
      headChunks.push(chunk);
      headBytes += chunk.byteLength;
    }
    tail = appendTail(tail, value, tailLimit);
  }
  const head = new Uint8Array(headBytes);
  let offset = 0;
  for (const chunk of headChunks) {
    head.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const truncated = totalBytes > headLimit + tailLimit;
  if (!truncated) {
    if (totalBytes <= headBytes) return { text: new TextDecoder().decode(head), truncated: false };
    const overlap = headBytes + tail.byteLength - totalBytes;
    const output = new Uint8Array(totalBytes);
    output.set(head);
    output.set(tail.subarray(Math.max(0, overlap)), headBytes);
    return { text: new TextDecoder().decode(output), truncated: false };
  }
  return {
    text: `${new TextDecoder().decode(head)}\n...[stderr truncated]...\n${new TextDecoder().decode(tail)}`,
    truncated: true,
  };
}

function appendTail(current: Uint8Array, chunk: Uint8Array, limit: number): Uint8Array {
  if (chunk.byteLength >= limit) return chunk.slice(chunk.byteLength - limit);
  const retainedCurrent = Math.min(current.byteLength, limit - chunk.byteLength);
  const combined = new Uint8Array(retainedCurrent + chunk.byteLength);
  combined.set(current.subarray(current.byteLength - retainedCurrent));
  combined.set(chunk, retainedCurrent);
  return combined;
}

function interruptedWithoutSpawn(): SupervisedProcessResult {
  return {
    exitCode: -1,
    stderr: "",
    wallMs: 0,
    timedOut: false,
    stderrTruncated: false,
    interrupted: true,
    aborted: false,
  };
}

function abortedWithoutSpawn(): SupervisedProcessResult {
  return {
    exitCode: -1,
    stderr: "",
    wallMs: 0,
    timedOut: false,
    stderrTruncated: false,
    interrupted: false,
    aborted: true,
  };
}
