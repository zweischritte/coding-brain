# Open Questions: Multi-User Memory Routing

1) Resolved: RLS policies in `openmemory/api/alembic/versions/add_rls_policies.py` blocked shared access.
   Added migration `openmemory/api/alembic/versions/disable_rls_for_shared_access.py` to disable RLS for `memories` and `apps`.

2) Resolved: skipped org_id backfill because only a single test memory exists.
   If needed, re-create the memory to set org_id on write, or provide a canonical org_id source for a targeted backfill.
