import { StatusBar } from "@/components/StatusBar";
import { ChatPanel } from "@/components/ChatPanel";

export default function App() {
  return (
    <main className="mx-auto max-w-3xl space-y-8 p-8">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">Hackathon scaffold</h1>
        <p className="text-muted-foreground text-sm">
          Vite + React + Tailwind + shadcn · FastAPI · Supabase. Read{" "}
          <code className="bg-muted rounded px-1 py-0.5 text-xs">CHALLENGE.md</code> for the
          brief, <code className="bg-muted rounded px-1 py-0.5 text-xs">AGENTS.md</code> for the
          architecture.
        </p>
        <StatusBar />
      </header>

      <ChatPanel />
    </main>
  );
}
