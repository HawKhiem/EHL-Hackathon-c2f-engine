-- ============================================================
--  Base schema. Replace `notes` with your real domain tables.
--
--  Conventions worth keeping:
--    * RLS ON for every table exposed to the browser.
--    * Explicit policies — RLS with no policy denies everything.
--    * updated_at maintained by trigger, not by the client.
-- ============================================================

create extension if not exists "pgcrypto";

-- ---------- shared updated_at trigger ----------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- ---------- example table ----------
create table public.notes (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users (id) on delete cascade,
  title       text not null check (length(title) between 1 and 200),
  body        text not null default '',
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

comment on table public.notes is 'Example table — replace with your domain model.';

create index notes_user_id_created_at_idx
  on public.notes (user_id, created_at desc);

create trigger notes_set_updated_at
  before update on public.notes
  for each row execute function public.set_updated_at();

-- ---------- grants (REQUIRED - read this) ----------
-- RLS policies alone are NOT enough. A role needs table-level privileges
-- *and* a policy that lets the row through. Skip the grants and the API
-- returns `42501 permission denied for table ...` no matter how correct
-- your policies are.
--
-- Why this is not automatic: the default privileges on `public` are owned by
-- `supabase_admin`, but migrations run as `postgres`. Default ACLs only apply
-- to objects created by the role that owns them, so tables you create here
-- inherit nothing. Every new table needs its own grants.
--
-- Roles map to API keys like this:
--   anon          <- publishable key (sb_publishable_...)  browser, RLS applies
--   authenticated <- a signed-in user's JWT               browser, RLS applies
--   service_role  <- secret key (sb_secret_...)           backend, BYPASSES RLS
grant select on public.notes to anon;
grant select, insert, update, delete on public.notes to authenticated;
grant all on public.notes to service_role;

-- ---------- row level security ----------
alter table public.notes enable row level security;

-- Owner-only access. `(select auth.uid())` is wrapped in a subquery so
-- Postgres evaluates it once per statement instead of once per row.
create policy notes_select_own
  on public.notes for select
  to authenticated
  using ((select auth.uid()) = user_id);

create policy notes_insert_own
  on public.notes for insert
  to authenticated
  with check ((select auth.uid()) = user_id);

create policy notes_update_own
  on public.notes for update
  to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

create policy notes_delete_own
  on public.notes for delete
  to authenticated
  using ((select auth.uid()) = user_id);

-- Demo rows (user_id is null) are readable without auth so the scaffold
-- shows data before you build login. DELETE THIS POLICY before shipping
-- anything real.
create policy notes_select_demo_rows
  on public.notes for select
  to anon
  using (user_id is null);
