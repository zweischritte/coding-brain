"""Tests for AI governance controls.

This module tests the governance controls per section 5.8:
- Model selection auditing
- Data usage tracking
- Risk mitigation logging
- Policy compliance checks
"""

from datetime import datetime, timezone

import pytest

from openmemory.api.governance.controls import (
    GovernanceConfig,
    RiskLevel,
    RiskMitigation,
    ModelSelection,
    DataUsageRecord,
    GovernanceAuditLogger,
    create_governance_logger,
)


# ============================================================================
# RiskLevel Tests
# ============================================================================


class TestRiskLevel:
    """Tests for RiskLevel enumeration."""

    def test_risk_level_ordering(self):
        """Test risk levels are properly ordered."""
        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_risk_level_values(self):
        """Test risk level values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


# ============================================================================
# RiskMitigation Tests
# ============================================================================


class TestRiskMitigation:
    """Tests for RiskMitigation dataclass."""

    def test_mitigation_creation(self):
        """Test creating a risk mitigation record."""
        mitigation = RiskMitigation(
            risk_id="risk-001",
            risk_description="Potential data leakage through model output",
            risk_level=RiskLevel.HIGH,
            mitigation_action="Content filtering applied",
            mitigated_by="automated",
            mitigated_at=datetime.now(timezone.utc),
        )
        assert mitigation.risk_id == "risk-001"
        assert mitigation.risk_level == RiskLevel.HIGH
        assert mitigation.mitigation_action == "Content filtering applied"

    def test_mitigation_to_dict(self):
        """Test serializing mitigation to dictionary."""
        now = datetime.now(timezone.utc)
        mitigation = RiskMitigation(
            risk_id="risk-002",
            risk_description="PII exposure risk",
            risk_level=RiskLevel.CRITICAL,
            mitigation_action="PII redaction applied",
            mitigated_by="content_filter",
            mitigated_at=now,
        )
        d = mitigation.to_dict()
        assert d["risk_id"] == "risk-002"
        assert d["risk_level"] == "critical"
        assert "mitigation_action" in d


# ============================================================================
# ModelSelection Tests
# ============================================================================


class TestModelSelection:
    """Tests for ModelSelection dataclass."""

    def test_selection_creation(self):
        """Test creating a model selection record."""
        selection = ModelSelection(
            model_id="qwen3-embedding:8b",
            model_version="v1.0.0",
            purpose="code_embedding",
            selected_at=datetime.now(timezone.utc),
            selected_by="embedding_service",
        )
        assert selection.model_id == "qwen3-embedding:8b"
        assert selection.purpose == "code_embedding"

    def test_selection_with_rationale(self):
        """Test selection with rationale."""
        selection = ModelSelection(
            model_id="gemini-embedding-001",
            model_version="001",
            purpose="fallback_embedding",
            selected_at=datetime.now(timezone.utc),
            selected_by="policy_engine",
            rationale="Local model unavailable, using cloud fallback",
            policy_override=False,
        )
        assert selection.rationale is not None
        assert "fallback" in selection.rationale.lower()

    def test_selection_with_policy_override(self):
        """Test selection with policy override."""
        selection = ModelSelection(
            model_id="gpt-4",
            model_version="turbo",
            purpose="code_generation",
            selected_at=datetime.now(timezone.utc),
            selected_by="admin",
            policy_override=True,
            override_reason="Customer requested specific model",
        )
        assert selection.policy_override is True
        assert selection.override_reason is not None

    def test_selection_to_dict(self):
        """Test serializing selection to dictionary."""
        now = datetime.now(timezone.utc)
        selection = ModelSelection(
            model_id="nomic-embed-text",
            model_version="1.5",
            purpose="development_embedding",
            selected_at=now,
            selected_by="config",
        )
        d = selection.to_dict()
        assert d["model_id"] == "nomic-embed-text"
        assert d["purpose"] == "development_embedding"


# ============================================================================
# DataUsageRecord Tests
# ============================================================================


class TestDataUsageRecord:
    """Tests for DataUsageRecord dataclass."""

    def test_record_creation(self):
        """Test creating a data usage record."""
        record = DataUsageRecord(
            record_id="usage-001",
            data_type="code_snippet",
            purpose="embedding_generation",
            model_id="qwen3-embedding:8b",
            org_id="org-123",
            timestamp=datetime.now(timezone.utc),
        )
        assert record.record_id == "usage-001"
        assert record.data_type == "code_snippet"
        assert record.purpose == "embedding_generation"

    def test_record_with_consent(self):
        """Test record with consent tracking."""
        record = DataUsageRecord(
            record_id="usage-002",
            data_type="user_query",
            purpose="retrieval",
            model_id="gemini-embedding-001",
            org_id="org-456",
            timestamp=datetime.now(timezone.utc),
            consent_basis="legitimate_interest",
            data_residency="eu-west-1",
        )
        assert record.consent_basis == "legitimate_interest"
        assert record.data_residency == "eu-west-1"

    def test_record_with_retention(self):
        """Test record with retention policy."""
        record = DataUsageRecord(
            record_id="usage-003",
            data_type="feedback_event",
            purpose="quality_improvement",
            model_id="internal",
            org_id="org-789",
            timestamp=datetime.now(timezone.utc),
            retention_days=30,
        )
        assert record.retention_days == 30

    def test_record_to_dict(self):
        """Test serializing record to dictionary."""
        record = DataUsageRecord(
            record_id="usage-004",
            data_type="memory_content",
            purpose="semantic_search",
            model_id="qwen3-embedding:8b",
            org_id="org-000",
            timestamp=datetime.now(timezone.utc),
        )
        d = record.to_dict()
        assert d["record_id"] == "usage-004"
        assert d["data_type"] == "memory_content"


# ============================================================================
# GovernanceConfig Tests
# ============================================================================


class TestGovernanceConfig:
    """Tests for GovernanceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GovernanceConfig()
        assert config.enabled is True
        assert config.log_model_selection is True
        assert config.log_data_usage is True
        assert config.log_risk_mitigation is True
        assert config.require_policy_check is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GovernanceConfig(
            enabled=True,
            log_model_selection=True,
            log_data_usage=False,
            log_risk_mitigation=True,
            require_policy_check=False,
            allowed_models=["qwen3-embedding:8b", "nomic-embed-text"],
        )
        assert config.log_data_usage is False
        assert config.require_policy_check is False
        assert len(config.allowed_models) == 2


# ============================================================================
# GovernanceAuditLogger Tests
# ============================================================================


class TestGovernanceAuditLogger:
    """Tests for GovernanceAuditLogger."""

    @pytest.fixture
    def logger(self):
        """Create a logger instance."""
        return create_governance_logger()

    def test_log_model_selection(self, logger):
        """Test logging model selection."""
        selection = ModelSelection(
            model_id="qwen3-embedding:8b",
            model_version="v1.0.0",
            purpose="code_embedding",
            selected_at=datetime.now(timezone.utc),
            selected_by="embedding_service",
        )
        logger.log_model_selection(selection, user_id="user-123", org_id="org-456")

        # Verify event was logged
        events = logger.get_model_selection_events()
        assert len(events) >= 1
        assert events[-1].model_id == "qwen3-embedding:8b"

    def test_log_data_usage(self, logger):
        """Test logging data usage."""
        record = DataUsageRecord(
            record_id="usage-test",
            data_type="code_snippet",
            purpose="embedding",
            model_id="qwen3-embedding:8b",
            org_id="org-123",
            timestamp=datetime.now(timezone.utc),
        )
        logger.log_data_usage(record)

        events = logger.get_data_usage_events()
        assert len(events) >= 1

    def test_log_risk_mitigation(self, logger):
        """Test logging risk mitigation."""
        mitigation = RiskMitigation(
            risk_id="risk-test",
            risk_description="Potential sensitive data in output",
            risk_level=RiskLevel.HIGH,
            mitigation_action="Output sanitization applied",
            mitigated_by="content_filter",
            mitigated_at=datetime.now(timezone.utc),
        )
        logger.log_risk_mitigation(mitigation, user_id="user-789")

        events = logger.get_risk_events()
        assert len(events) >= 1
        assert events[-1].risk_level == RiskLevel.HIGH

    def test_log_policy_violation(self, logger):
        """Test logging policy violations."""
        logger.log_policy_violation(
            policy_id="policy-001",
            violation_type="model_not_allowed",
            details="Attempted to use non-whitelisted model",
            user_id="user-bad",
            org_id="org-xyz",
        )

        events = logger.get_policy_violation_events()
        assert len(events) >= 1

    def test_check_model_allowed(self, logger):
        """Test model allowlist checking."""
        # By default, all models should be allowed
        assert logger.is_model_allowed("qwen3-embedding:8b") is True
        assert logger.is_model_allowed("gemini-embedding-001") is True

    def test_check_model_not_allowed(self):
        """Test model allowlist with restricted config."""
        config = GovernanceConfig(
            allowed_models=["qwen3-embedding:8b", "nomic-embed-text"],
        )
        logger = GovernanceAuditLogger(config=config)

        assert logger.is_model_allowed("qwen3-embedding:8b") is True
        assert logger.is_model_allowed("nomic-embed-text") is True
        assert logger.is_model_allowed("gpt-4") is False

    def test_get_statistics(self, logger):
        """Test getting governance statistics."""
        # Log some events
        selection = ModelSelection(
            model_id="test-model",
            model_version="1.0",
            purpose="test",
            selected_at=datetime.now(timezone.utc),
            selected_by="test",
        )
        logger.log_model_selection(selection)

        stats = logger.get_statistics()
        assert "model_selections" in stats
        assert "data_usage_events" in stats
        assert "risk_mitigations" in stats
        assert "policy_violations" in stats

    def test_disabled_logging(self):
        """Test that logging can be disabled."""
        config = GovernanceConfig(enabled=False)
        logger = GovernanceAuditLogger(config=config)

        selection = ModelSelection(
            model_id="test-model",
            model_version="1.0",
            purpose="test",
            selected_at=datetime.now(timezone.utc),
            selected_by="test",
        )
        logger.log_model_selection(selection)

        # Should not log when disabled
        events = logger.get_model_selection_events()
        assert len(events) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestGovernanceIntegration:
    """Integration tests for governance controls."""

    def test_full_governance_workflow(self):
        """Test complete governance workflow."""
        logger = create_governance_logger()

        # Step 1: Model selection
        selection = ModelSelection(
            model_id="qwen3-embedding:8b",
            model_version="v1.0.0",
            purpose="code_embedding",
            selected_at=datetime.now(timezone.utc),
            selected_by="embedding_service",
            rationale="Best performance on CodeSearchNet benchmark",
        )
        logger.log_model_selection(selection, user_id="user-123", org_id="org-456")

        # Step 2: Data usage
        record = DataUsageRecord(
            record_id="usage-001",
            data_type="code_snippet",
            purpose="embedding_generation",
            model_id="qwen3-embedding:8b",
            org_id="org-456",
            timestamp=datetime.now(timezone.utc),
            consent_basis="legitimate_interest",
        )
        logger.log_data_usage(record)

        # Step 3: Risk mitigation during processing
        mitigation = RiskMitigation(
            risk_id="risk-001",
            risk_description="Detected potential secret in code",
            risk_level=RiskLevel.HIGH,
            mitigation_action="Secret quarantined and excluded from embedding",
            mitigated_by="secret_scanner",
            mitigated_at=datetime.now(timezone.utc),
        )
        logger.log_risk_mitigation(mitigation)

        # Verify all events are logged
        stats = logger.get_statistics()
        assert stats["model_selections"] >= 1
        assert stats["data_usage_events"] >= 1
        assert stats["risk_mitigations"] >= 1

    def test_iso42001_alignment(self):
        """Test alignment with ISO/IEC 42001 controls."""
        logger = create_governance_logger()

        # ISO 42001 requires tracking of:
        # 1. AI system purposes (via ModelSelection.purpose)
        # 2. Risk assessments (via RiskMitigation)
        # 3. Data governance (via DataUsageRecord)
        # 4. Accountability (via user_id tracking)

        # Log events covering these areas
        selection = ModelSelection(
            model_id="qwen3-embedding:8b",
            model_version="v1.0.0",
            purpose="semantic_code_search",  # Purpose documented
            selected_at=datetime.now(timezone.utc),
            selected_by="system",
            rationale="Selected for code understanding capabilities",
        )
        logger.log_model_selection(selection, user_id="admin-001")

        # All required fields should be present
        events = logger.get_model_selection_events()
        assert len(events) >= 1

        latest = events[-1]
        assert latest.purpose is not None
        assert latest.selected_by is not None
        assert latest.rationale is not None
