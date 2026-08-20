import { useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { api, type ChatMessage } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Minimal streaming chat against /llm/chat/stream. This is the reference
 * for how to consume the SSE endpoint — copy the pattern, then delete this
 * component once your real UI exists.
 */
export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    const text = draft.trim();
    if (!text || streaming) return;

    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...next, { role: "assistant", content: "" }]);
    setDraft("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await api.chatStream(
        next,
        (token) =>
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = {
              role: "assistant",
              content: copy[copy.length - 1].content + token,
            };
            return copy;
          }),
        { signal: controller.signal },
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            role: "assistant",
            content: `⚠ ${(err as Error).message}`,
          };
          return copy;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>LLM smoke test</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="max-h-80 space-y-3 overflow-y-auto">
          {messages.length === 0 && (
            <p className="text-muted-foreground text-sm">
              Send a message to confirm the LLM wrapper streams end to end.
            </p>
          )}
          {messages.map((m, i) => (
            <div key={i} className="text-sm">
              <span className="text-muted-foreground mr-2 font-mono text-xs uppercase">
                {m.role}
              </span>
              <span className="whitespace-pre-wrap">
                {m.content}
                {streaming && i === messages.length - 1 && (
                  <span className="bg-foreground ml-0.5 inline-block h-4 w-1.5 animate-pulse align-middle" />
                )}
              </span>
            </div>
          ))}
        </div>

        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask something…"
            disabled={streaming}
          />
          {streaming ? (
            <Button type="button" variant="outline" onClick={() => abortRef.current?.abort()}>
              <Square /> Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!draft.trim()}>
              <Send /> Send
            </Button>
          )}
        </form>
      </CardContent>
    </Card>
  );
}
