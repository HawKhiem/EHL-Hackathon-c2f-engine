"""Server-side Supabase client.

Uses the SECRET key (`sb_secret_...`), which connects as `service_role` and
therefore BYPASSES Row Level Security. That is fine for trusted backend work
but means: never return raw rows to a caller you have not authorised, and
never send this key to the browser.

For user-scoped reads, prefer the frontend's publishable client (RLS applies)
or create a per-request client with the user's access token.
"""

from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    if not settings.supabase_configured:
        raise RuntimeError(
            "Supabase is not configured. Run ./setup.sh (or `supabase start`, "
            "then copy SUPABASE_URL and the secret key from `supabase status` "
            "into .env)."
        )
    return create_client(settings.supabase_url, settings.supabase_secret_key)
