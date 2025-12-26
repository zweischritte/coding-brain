"""AI Governance controls.

This module implements AI governance controls per section 5.8:
- Controls aligned to AI governance standards (e.g., ISO/IEC 42001)
- Model selection tracking
- Data usage logging
- Risk mitigation tracking in audit logs

ISO/IEC 42001 alignment:
- 6.1.4: Risk assessment documentation
- 6.1.5: Risk treatment plans
- 7.2: Competence and awareness
- 7.4: Communication of AI system purposes
- 8.2: AI system requirements
- 9.1: Monitoring and measurement
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ============================================================================
# Enums
# ============================================================================


class RiskLevel(str, Enum):
    """Risk level for AI governance."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "RiskLevel") -> bool:
        """Compare risk levels."""
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "RiskLevel") -> bool:
        """Compare risk levels."""
        return self == other or self < other

    def __gt__(self, other: "RiskLevel") -> bool:
        """Compare risk levels."""
        return not self <= other

    def __ge__(self, other: "RiskLevel") -> bool:
        """Compare risk levels."""
        return self == other or self > other


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class RiskMitigation:
    """Record of a risk mitigation action.

    Per ISO 42001 6.1.5: Risk treatment plans must be documented.
    """

    risk_id: str
    risk_description: str
    risk_level: RiskLevel
    mitigation_action: str
    mitigated_by: str
    mitigated_at: datetime
    control_reference: str | None = None  # e.g., "ISO42001:6.1.5"
    verification_status: str | None = None
    residual_risk: RiskLevel | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "risk_id": self.risk_id,
            "risk_description": self.risk_description,
            "risk_level": self.risk_level.value,
            "mitigation_action": self.mitigation_action,
            "mitigated_by": self.mitigated_by,
            "mitigated_at": self.mitigated_at.isoformat(),
            "control_reference": self.control_reference,
            "verification_status": self.verification_status,
            "residual_risk": self.residual_risk.value if self.residual_risk else None,
        }


@dataclass
class ModelSelection:
    """Record of an AI model selection.

    Per ISO 42001 8.2: AI system requirements must be documented.
    """

    model_id: str
    model_version: str
    purpose: str
    selected_at: datetime
    selected_by: str
    rationale: str | None = None
    policy_override: bool = False
    override_reason: str | None = None
    performance_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "purpose": self.purpose,
            "selected_at": self.selected_at.isoformat(),
            "selected_by": self.selected_by,
            "rationale": self.rationale,
            "policy_override": self.policy_override,
            "override_reason": self.override_reason,
            "performance_metrics": self.performance_metrics,
        }


@dataclass
class DataUsageRecord:
    """Record of data usage for AI processing.

    Per ISO 42001 7.4: Data governance and communication requirements.
    """

    record_id: str
    data_type: str
    purpose: str
    model_id: str
    org_id: str
    timestamp: datetime
    consent_basis: str | None = None
    data_residency: str | None = None
    retention_days: int | None = None
    anonymized: bool = False
    processing_location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "data_type": self.data_type,
            "purpose": self.purpose,
            "model_id": self.model_id,
            "org_id": self.org_id,
            "timestamp": self.timestamp.isoformat(),
            "consent_basis": self.consent_basis,
            "data_residency": self.data_residency,
            "retention_days": self.retention_days,
            "anonymized": self.anonymized,
            "processing_location": self.processing_location,
        }


@dataclass
class PolicyViolation:
    """Record of a governance policy violation."""

    violation_id: str
    policy_id: str
    violation_type: str
    details: str
    timestamp: datetime
    user_id: str | None = None
    org_id: str | None = None
    action_taken: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_id": self.violation_id,
            "policy_id": self.policy_id,
            "violation_type": self.violation_type,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "org_id": self.org_id,
            "action_taken": self.action_taken,
        }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class GovernanceConfig:
    """Configuration for AI governance controls."""

    enabled: bool = True
    log_model_selection: bool = True
    log_data_usage: bool = True
    log_risk_mitigation: bool = True
    require_policy_check: bool = True
    allowed_models: list[str] = field(default_factory=list)  # Empty = allow all
    blocked_models: list[str] = field(default_factory=list)
    min_risk_level_for_audit: RiskLevel = RiskLevel.LOW


# ============================================================================
# Governance Audit Logger
# ============================================================================


class GovernanceAuditLogger:
    """Logger for AI governance events.

    Provides centralized logging for:
    - Model selection decisions
    - Data usage tracking
    - Risk mitigation actions
    - Policy violations
    """

    def __init__(self, config: GovernanceConfig | None = None):
        """Initialize the governance logger.

        Args:
            config: Governance configuration
        """
        self._config = config or GovernanceConfig()

        # In-memory storage for events (would be database in production)
        self._model_selections: list[ModelSelection] = []
        self._data_usage_records: list[DataUsageRecord] = []
        self._risk_mitigations: list[RiskMitigation] = []
        self._policy_violations: list[PolicyViolation] = []

    def log_model_selection(
        self,
        selection: ModelSelection,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> None:
        """Log a model selection event.

        Args:
            selection: The model selection record
            user_id: Optional user who triggered selection
            org_id: Optional organization context
        """
        if not self._config.enabled or not self._config.log_model_selection:
            return

        self._model_selections.append(selection)

    def log_data_usage(self, record: DataUsageRecord) -> None:
        """Log a data usage event.

        Args:
            record: The data usage record
        """
        if not self._config.enabled or not self._config.log_data_usage:
            return

        self._data_usage_records.append(record)

    def log_risk_mitigation(
        self,
        mitigation: RiskMitigation,
        user_id: str | None = None,
    ) -> None:
        """Log a risk mitigation action.

        Args:
            mitigation: The risk mitigation record
            user_id: Optional user who performed mitigation
        """
        if not self._config.enabled or not self._config.log_risk_mitigation:
            return

        if mitigation.risk_level < self._config.min_risk_level_for_audit:
            return

        self._risk_mitigations.append(mitigation)

    def log_policy_violation(
        self,
        policy_id: str,
        violation_type: str,
        details: str,
        user_id: str | None = None,
        org_id: str | None = None,
        action_taken: str | None = None,
    ) -> None:
        """Log a policy violation.

        Args:
            policy_id: ID of the violated policy
            violation_type: Type of violation
            details: Violation details
            user_id: User who triggered violation
            org_id: Organization context
            action_taken: Action taken in response
        """
        if not self._config.enabled:
            return

        import uuid

        violation = PolicyViolation(
            violation_id=str(uuid.uuid4()),
            policy_id=policy_id,
            violation_type=violation_type,
            details=details,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            org_id=org_id,
            action_taken=action_taken,
        )
        self._policy_violations.append(violation)

    def is_model_allowed(self, model_id: str) -> bool:
        """Check if a model is allowed by policy.

        Args:
            model_id: The model identifier

        Returns:
            True if model is allowed
        """
        # Check blocked list first
        if model_id in self._config.blocked_models:
            return False

        # If allowed list is empty, all models are allowed
        if not self._config.allowed_models:
            return True

        # Check allowed list
        return model_id in self._config.allowed_models

    def get_model_selection_events(self) -> list[ModelSelection]:
        """Get all model selection events.

        Returns:
            List of model selection records
        """
        return self._model_selections.copy()

    def get_data_usage_events(self) -> list[DataUsageRecord]:
        """Get all data usage events.

        Returns:
            List of data usage records
        """
        return self._data_usage_records.copy()

    def get_risk_events(self) -> list[RiskMitigation]:
        """Get all risk mitigation events.

        Returns:
            List of risk mitigation records
        """
        return self._risk_mitigations.copy()

    def get_policy_violation_events(self) -> list[PolicyViolation]:
        """Get all policy violation events.

        Returns:
            List of policy violation records
        """
        return self._policy_violations.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get governance statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "model_selections": len(self._model_selections),
            "data_usage_events": len(self._data_usage_records),
            "risk_mitigations": len(self._risk_mitigations),
            "policy_violations": len(self._policy_violations),
            "config_enabled": self._config.enabled,
        }

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate a compliance report.

        Returns:
            Compliance report dictionary
        """
        # Count events by type
        risk_by_level: dict[str, int] = {}
        for m in self._risk_mitigations:
            level = m.risk_level.value
            risk_by_level[level] = risk_by_level.get(level, 0) + 1

        models_used: dict[str, int] = {}
        for s in self._model_selections:
            models_used[s.model_id] = models_used.get(s.model_id, 0) + 1

        return {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_model_selections": len(self._model_selections),
                "total_data_usage_events": len(self._data_usage_records),
                "total_risk_mitigations": len(self._risk_mitigations),
                "total_policy_violations": len(self._policy_violations),
            },
            "risk_mitigations_by_level": risk_by_level,
            "models_used": models_used,
            "policy_override_count": sum(
                1 for s in self._model_selections if s.policy_override
            ),
        }


def create_governance_logger(
    config: GovernanceConfig | None = None,
) -> GovernanceAuditLogger:
    """Create a governance audit logger.

    Args:
        config: Optional governance configuration

    Returns:
        GovernanceAuditLogger instance
    """
    return GovernanceAuditLogger(config=config)
