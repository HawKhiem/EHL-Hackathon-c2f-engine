-- Seed data for local development. Runs on `supabase db reset`.
-- Demo rows have a null user_id so they are visible pre-auth.

insert into public.notes (title, body) values
  ('Read CHALLENGE.md', 'Paste the real challenge brief in at kick-off.'),
  ('Delete the demo policy', 'supabase/migrations — notes_select_demo_rows is anon-readable.'),
  ('Ship something demoable', 'A narrow flow that works beats a broad one that does not.');
