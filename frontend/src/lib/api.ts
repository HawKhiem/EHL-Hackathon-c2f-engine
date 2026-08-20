/**
 * Typed client for the FastAPI backend.
 *
 * Every call goes through `/api/...`, which the Vite dev server proxies to
 * the backend (see vite.config.ts). Add a method here rather than calling
 * fetch() from a component — it keeps types in one place and gives the
 * whole team one thing to grep for.
 *
 * The backend publishes an OpenAPI schema at http://127.0.0.1:8000/docs —
 * use it to keep these types honest.
 */

export const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Low-level fetch with JSON encoding and FastAPI error unwrapping.
 * Exported so feature-specific API modules can reuse it instead of
 * re-implementing error handling.
 */
export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    // FastAPI puts validation and HTTPException detail under `detail`.
    let body: unknown;
    let detail = res.statusText;
    try {
      body = await res.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = String((body as { detail: unknown }).detail);
      }
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(detail, res.status, body);
  }

  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const request = apiRequest;

// ---------------------------------------------------------------- types

export interface HealthResponse {
  status: "ok";
  llm_provider: string;
  llm_model: string;
  supabase_configured: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  content: string;
  model: string;
  /** True when the model declined the request; `content` explains why. */
  refused: boolean;
}

// ---------------------------------------------------------------- calls

export const api = {
  health: () => request<HealthResponse>("/health"),

  chat: (messages: ChatMessage[], system?: string) =>
    request<ChatResponse>("/llm/chat", {
      method: "POST",
      body: JSON.stringify({ messages, system }),
    }),

  /**
   * Server-sent-events stream of the assistant reply.
   *
   * `onToken` fires per text delta. Resolves when the stream ends; rejects on
   * transport failure. Pass an AbortSignal to let the user cancel.
   *
   *   const stop = new AbortController();
   *   await api.chatStream(msgs, (t) => setText((s) => s + t), { signal: stop.signal });
   */
  async chatStream(
    messages: ChatMessage[],
    onToken: (token: string) => void,
    opts: { system?: string; signal?: AbortSignal } = {},
  ): Promise<void> {
    const res = await fetch(`${BASE}/llm/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, system: opts.system }),
      signal: opts.signal,
    });

    if (!res.ok || !res.body) {
      throw new ApiError(`stream failed: ${res.statusText}`, res.status);
    }

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          try {
            const evt = JSON.parse(data) as
              | { type: "token"; text: string }
              | { type: "error"; message: string };
            if (evt.type === "token") onToken(evt.text);
            else throw new ApiError(evt.message, 500);
          } catch (err) {
            if (err instanceof ApiError) throw err;
            // Ignore a malformed frame rather than killing the stream.
          }
        }
      }
    }
  },
};
