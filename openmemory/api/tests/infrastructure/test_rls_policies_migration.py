import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_migration_module():
    api_root = Path(__file__).resolve().parents[2]
    migration_path = api_root / "alembic" / "versions" / "disable_rls_for_shared_access.py"
    spec = importlib.util.spec_from_file_location("disable_rls_for_shared_access", migration_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _patch_op(monkeypatch, module, dialect_name):
    calls = []
    mock_bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect_name))
    monkeypatch.setattr(module.op, "get_bind", lambda: mock_bind)
    monkeypatch.setattr(module.op, "execute", lambda sql: calls.append(sql))
    return calls


def test_upgrade_disables_rls_for_postgres(monkeypatch):
    module = _load_migration_module()
    calls = _patch_op(monkeypatch, module, "postgresql")

    module.upgrade()

    assert "ALTER TABLE memories DISABLE ROW LEVEL SECURITY" in calls
    assert "ALTER TABLE apps DISABLE ROW LEVEL SECURITY" in calls
    assert "DROP POLICY IF EXISTS memories_tenant_select ON memories" in calls
    assert "DROP POLICY IF EXISTS apps_tenant_select ON apps" in calls


def test_upgrade_noops_for_non_postgres(monkeypatch):
    module = _load_migration_module()
    calls = _patch_op(monkeypatch, module, "sqlite")

    module.upgrade()

    assert calls == []


def test_downgrade_restores_policies_for_postgres(monkeypatch):
    module = _load_migration_module()
    calls = _patch_op(monkeypatch, module, "postgresql")

    module.downgrade()

    assert "ALTER TABLE memories ENABLE ROW LEVEL SECURITY" in calls
    assert "ALTER TABLE apps ENABLE ROW LEVEL SECURITY" in calls
    assert any("CREATE POLICY memories_tenant_select" in sql for sql in calls)
    assert any("CREATE POLICY apps_tenant_select" in sql for sql in calls)
