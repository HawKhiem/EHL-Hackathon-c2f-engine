import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "@/lib/api";

/** Build a Response whose body streams `chunks` as separate network reads. */
function streamingResponse(chunks: string[], ok = true, status = 200): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return { ok, status, statusText: "OK", body } as unknown as Response;
}

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: async () => data,
  } as unknown as Response;
}

afterEach(() => vi.unstubAllGlobals());

describe("api.chatStream SSE parsing", () => {
  it("emits one token per data frame, in order", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse([
            'data: {"type":"token","text":"Hello"}\n\n',
            'data: {"type":"token","text":" world"}\n\n',
            "data: [DONE]\n\n",
          ]),
        ),
    );

    const tokens: string[] = [];
    await api.chatStream([{ role: "user", content: "hi" }], (t) => tokens.push(t));

    expect(tokens).toEqual(["Hello", " world"]);
  });

  it("reassembles a frame split across network reads", async () => {
    // The boundary falls mid-JSON, which naive per-chunk parsing would drop.
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse(['data: {"type":"token","te', 'xt":"split"}\n\n', "data: [DONE]\n\n"]),
        ),
    );

    const tokens: string[] = [];
    await api.chatStream([{ role: "user", content: "hi" }], (t) => tokens.push(t));

    expect(tokens).toEqual(["split"]);
  });

  it("handles several frames arriving in one read", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse([
            'data: {"type":"token","text":"a"}\n\ndata: {"type":"token","text":"b"}\n\n',
            "data: [DONE]\n\n",
          ]),
        ),
    );

    const tokens: string[] = [];
    await api.chatStream([{ role: "user", content: "hi" }], (t) => tokens.push(t));

    expect(tokens).toEqual(["a", "b"]);
  });

  it("stops at [DONE] and ignores anything after it", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse([
            'data: {"type":"token","text":"kept"}\n\n',
            "data: [DONE]\n\n",
            'data: {"type":"token","text":"ignored"}\n\n',
          ]),
        ),
    );

    const tokens: string[] = [];
    await api.chatStream([{ role: "user", content: "hi" }], (t) => tokens.push(t));

    expect(tokens).toEqual(["kept"]);
  });

  it("throws on an in-band error frame", async () => {
    // Errors after the first byte cannot be an HTTP status, so the backend
    // sends them as a frame. They must still surface as a thrown ApiError.
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamingResponse([
            'data: {"type":"token","text":"partial"}\n\n',
            'data: {"type":"error","message":"upstream exploded"}\n\n',
          ]),
        ),
    );

    const tokens: string[] = [];
    await expect(
      api.chatStream([{ role: "user", content: "hi" }], (t) => tokens.push(t)),
    ).rejects.toThrow("upstream exploded");

    expect(tokens).toEqual(["partial"]);
  });

  it("throws when the transport itself fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamingResponse([], false, 503)));

    await expect(api.chatStream([{ role: "user", content: "hi" }], () => {})).rejects.toThrow(
      ApiError,
    );
  });
});

describe("api error handling", () => {
  it("unwraps FastAPI's detail field", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Rate limit exceeded" }, false, 429)),
    );

    await expect(api.health()).rejects.toThrow("Rate limit exceeded");
  });

  it("exposes the status code on the thrown error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "nope" }, false, 429)));

    await expect(api.health()).rejects.toMatchObject({ status: 429 });
  });

  it("returns parsed JSON on success", async () => {
    const payload = {
      status: "ok",
      llm_provider: "anthropic",
      llm_model: "claude-opus-5",
      supabase_configured: true,
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(payload)));

    await expect(api.health()).resolves.toEqual(payload);
  });
});
