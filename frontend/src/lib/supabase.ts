import { createClient } from "@supabase/supabase-js";

/**
 * Browser Supabase client.
 *
 * Uses the PUBLISHABLE key (`sb_publishable_...`), which connects as the
 * `anon` role and is therefore subject to Row Level Security. It is public by
 * design — Vite inlines every VITE_* var into the bundle.
 *
 * Never put the SECRET key (`sb_secret_...`) in frontend code: it bypasses RLS.
 * Server-side work belongs in backend/app/supabase_client.py.
 *
 * Note: publishable/secret keys replace the legacy anon/service_role JWTs,
 * which Supabase is deprecating by the end of 2026.
 */
const url = import.meta.env.VITE_SUPABASE_URL;
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

export const supabaseConfigured = Boolean(url && publishableKey);

if (!supabaseConfigured) {
  console.warn(
    "[supabase] VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY are unset — " +
      "run ./setup.sh, or copy the values from `supabase status` into .env",
  );
}

export const supabase = createClient(
  url ?? "http://127.0.0.1:54321",
  publishableKey ?? "sb_publishable_placeholder",
);
