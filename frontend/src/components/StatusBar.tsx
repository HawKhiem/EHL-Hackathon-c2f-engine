import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { api, type HealthResponse } from "@/lib/api";
import { supabaseConfigured } from "@/lib/supabase";
import { Badge } from "@/components/ui/badge";

/**
 * Boot check. If any pill here is red, the stack is not wired up —
 * fix that before writing product code. Safe to delete once you ship.
 */
export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  const pill = (ok: boolean, label: string) => (
    <Badge variant={ok ? "success" : "destructive"} key={label}>
      {ok ? <CheckCircle2 /> : <XCircle />}
      {label}
    </Badge>
  );

  if (error) {
    return (
      <div className="flex flex-wrap items-center gap-2">
        {pill(false, `backend unreachable — ${error}`)}
      </div>
    );
  }

  if (!health) {
    return (
      <div className="text-muted-foreground flex items-center gap-2 text-sm">
        <Loader2 className="size-4 animate-spin" /> checking stack…
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {pill(true, "backend")}
      {pill(health.supabase_configured, "supabase")}
      {pill(Boolean(health.llm_model), `llm · ${health.llm_provider}`)}
      {pill(supabaseConfigured, "supabase (browser)")}
    </div>
  );
}
