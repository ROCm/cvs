"""SQLAlchemy database models for Fleet Manager."""

import os
from datetime import datetime
from typing import Generator
from enum import Enum as PyEnum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Index,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# DATABASE_URL must be set via environment variable (see .env.example).
# The fallback is intentionally non-functional to prevent accidental insecure defaults in production.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://fleet:change_this_password@localhost:5432/fleet_monitor")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class NodeStatus(str, PyEnum):
    """Node installation/connection status."""

    PENDING = "pending"
    CONNECTED = "connected"  # SSH connection verified, ready for installation
    INSTALLING = "installing"
    ACTIVE = "active"  # Exporters installed and running
    ERROR = "error"
    UNREACHABLE = "unreachable"


class MonitoringServer(Base):
    """A monitoring server configuration (Prometheus/Grafana/Loki)."""

    __tablename__ = "monitoring_servers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Monitoring server settings (where Prometheus/Loki/Grafana run)
    server_ip = Column(String(255), nullable=True)
    server_hostname = Column(String(255), nullable=True)

    # Service ports
    prometheus_port = Column(Integer, default=30090)
    loki_port = Column(Integer, default=30100)
    grafana_port = Column(Integer, default=30030)

    # Prometheus configuration
    prometheus_retention_time = Column(String(20), default="15d")
    prometheus_retention_size = Column(String(20), default="50GB")
    prometheus_scrape_interval = Column(String(10), default="15s")
    prometheus_storage_path = Column(
        String(512), nullable=True
    )  # Host path for TSDB data; defaults to ./data/prometheus

    # Loki configuration
    loki_retention_days = Column(Integer, default=7)

    # Grafana credentials
    grafana_admin_user = Column(String(255), default="admin")
    grafana_admin_password = Column(String(255), default="admin")

    # Remote monitoring setup options
    setup_monitoring_stack = Column(Boolean, default=False)
    ssh_user = Column(String(255), nullable=True)
    ssh_port = Column(Integer, default=22)
    ssh_auth_type = Column(String(20), default="password")
    ssh_key_path = Column(String(512), nullable=True)
    ssh_password = Column(String(512), nullable=True)

    # Jump host configuration
    use_jump_host = Column(Boolean, default=False)
    jump_host = Column(String(255), nullable=True)
    jump_port = Column(Integer, default=22)
    jump_user = Column(String(255), nullable=True)
    jump_auth_type = Column(String(20), default="key")
    jump_key_path = Column(String(512), nullable=True)
    jump_password = Column(String(512), nullable=True)
    remote_auth_type = Column(String(20), default="key")
    remote_key_path = Column(String(512), nullable=True)
    remote_password = Column(String(512), nullable=True)

    # Push gateway
    use_push_gateway = Column(Boolean, default=False)
    push_gateway_port = Column(Integer, default=9091)

    # Stack installation status
    stack_installed = Column(Boolean, default=False)
    last_install_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    node_groups = relationship("NodeGroup", back_populates="monitoring_server")
    control_node_groups = relationship("ControlNodeGroup", back_populates="monitoring_server")
    alert_rules = relationship("AlertRule", back_populates="monitoring_server", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<MonitoringServer(id={self.id}, name='{self.name}', server='{self.server_ip}')>"


class MetricGroup(Base):
    """A group of metrics to collect - can be associated with node groups."""

    __tablename__ = "metric_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # GPU metric categories (enabled/disabled)
    gpu_utilization = Column(Boolean, default=True)
    gpu_memory = Column(Boolean, default=True)
    gpu_temperature = Column(Boolean, default=True)
    gpu_power = Column(Boolean, default=True)
    gpu_fan = Column(Boolean, default=False)
    gpu_clocks = Column(Boolean, default=False)
    gpu_pcie = Column(Boolean, default=False)
    gpu_ecc = Column(Boolean, default=False)

    # Node exporter metrics
    node_cpu = Column(Boolean, default=True)
    node_memory = Column(Boolean, default=True)
    node_disk = Column(Boolean, default=True)
    node_network = Column(Boolean, default=False)

    # Log collection
    collect_logs = Column(Boolean, default=True)
    log_patterns = Column(JSON, default=list)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    node_groups = relationship("NodeGroup", back_populates="metric_group")

    def __repr__(self):
        return f"<MetricGroup(id={self.id}, name='{self.name}')>"


class NodeGroup(Base):
    """A group of GPU nodes to monitor together."""

    __tablename__ = "node_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Associated monitoring server
    monitoring_server_id = Column(Integer, ForeignKey("monitoring_servers.id"), nullable=True)

    # Associated metric group
    metric_group_id = Column(Integer, ForeignKey("metric_groups.id"), nullable=True)

    # SSH credentials for GPU nodes
    ssh_user = Column(String(255), nullable=False, default="root")
    ssh_port = Column(Integer, nullable=False, default=22)
    ssh_auth_type = Column(String(20), nullable=False, default="key")
    ssh_key_path = Column(String(512), nullable=True)
    ssh_password = Column(String(512), nullable=True)

    # Jump host configuration (optional)
    use_jump_host = Column(Boolean, default=False)
    jump_host = Column(String(255), nullable=True)
    jump_port = Column(Integer, nullable=True, default=22)
    jump_user = Column(String(255), nullable=True)
    jump_auth_type = Column(String(20), nullable=True, default="key")
    jump_key_path = Column(String(512), nullable=True)
    jump_password = Column(String(512), nullable=True)

    # Credentials for GPU nodes when using jump host
    remote_auth_type = Column(String(20), nullable=True, default="key")
    remote_key_path = Column(String(512), nullable=True)
    remote_password = Column(String(512), nullable=True)

    # Legacy field - kept for migration
    metric_config_id = Column(Integer, ForeignKey("metric_configs.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    nodes = relationship("Node", back_populates="node_group", cascade="all, delete-orphan")
    monitoring_server = relationship("MonitoringServer", back_populates="node_groups")
    metric_group = relationship("MetricGroup", back_populates="node_groups")
    metric_config = relationship("MetricConfig", back_populates="node_groups")

    def __repr__(self):
        return f"<NodeGroup(id={self.id}, name='{self.name}', nodes={len(self.nodes)})>"


class Node(Base):
    """A single GPU node in a node group."""

    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)
    node_group_id = Column(Integer, ForeignKey("node_groups.id"), nullable=False)

    # Node identification
    ip_address = Column(String(45), nullable=False)
    hostname = Column(String(255), nullable=True)

    # Status
    status = Column(String(20), default=NodeStatus.PENDING.value)
    status_message = Column(Text, nullable=True)
    last_seen = Column(DateTime, nullable=True)

    # GPU info (populated after installation)
    gpu_count = Column(Integer, nullable=True)
    gpu_model = Column(String(255), nullable=True)

    # Exporter ports
    gpu_exporter_port = Column(Integer, default=5000)
    node_exporter_port = Column(Integer, default=9100)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    node_group = relationship("NodeGroup", back_populates="nodes")
    installation_logs = relationship("InstallationLog", back_populates="node", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (Index("ix_nodes_group_ip", "node_group_id", "ip_address", unique=True),)

    def __repr__(self):
        return f"<Node(id={self.id}, ip='{self.ip_address}', status='{self.status}')>"


class MetricConfig(Base):
    """Legacy configuration for metrics - kept for backward compatibility."""

    __tablename__ = "metric_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False)

    # GPU metric categories
    fleet_health = Column(Boolean, default=True)
    thermal_power = Column(Boolean, default=True)
    pcie = Column(Boolean, default=False)
    xgmi = Column(Boolean, default=False)
    ecc = Column(Boolean, default=False)
    ras = Column(Boolean, default=False)
    utilization = Column(Boolean, default=True)
    memory = Column(Boolean, default=True)

    # Node exporter metrics
    cpu_metrics = Column(Boolean, default=True)
    memory_metrics = Column(Boolean, default=True)
    disk_metrics = Column(Boolean, default=True)
    network_metrics = Column(Boolean, default=False)

    # Log collection
    collect_dmesg = Column(Boolean, default=True)
    collect_journalctl = Column(Boolean, default=True)
    log_patterns = Column(JSON, default=list)

    # Additional custom metrics
    custom_metrics = Column(JSON, default=list)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    node_groups = relationship("NodeGroup", back_populates="metric_config")

    def __repr__(self):
        return f"<MetricConfig(id={self.id}, name='{self.name}')>"


class MonitoringConfig(Base):
    """Legacy global monitoring configuration - kept for migration."""

    __tablename__ = "monitoring_config"

    id = Column(Integer, primary_key=True, index=True)
    monitoring_server_ip = Column(String(255), nullable=True)
    monitoring_server_hostname = Column(String(255), nullable=True)
    prometheus_port = Column(Integer, default=30090)
    loki_port = Column(Integer, default=30100)
    grafana_port = Column(Integer, default=30030)
    prometheus_retention_time = Column(String(20), default="15d")
    prometheus_retention_size = Column(String(20), default="50GB")
    prometheus_scrape_interval = Column(String(10), default="15s")
    loki_retention_days = Column(Integer, default=7)
    grafana_admin_user = Column(String(255), default="admin")
    grafana_admin_password = Column(String(255), default="admin")
    setup_monitoring_stack = Column(Boolean, default=False)
    monitoring_ssh_user = Column(String(255), nullable=True)
    monitoring_ssh_port = Column(Integer, default=22)
    monitoring_ssh_auth_type = Column(String(20), default="password")
    monitoring_ssh_key_path = Column(String(512), nullable=True)
    monitoring_ssh_password = Column(String(512), nullable=True)
    monitoring_use_jump_host = Column(Boolean, default=False)
    monitoring_jump_host = Column(String(255), nullable=True)
    monitoring_jump_port = Column(Integer, default=22)
    monitoring_jump_user = Column(String(255), nullable=True)
    monitoring_jump_auth_type = Column(String(20), default="key")
    monitoring_jump_key_path = Column(String(512), nullable=True)
    monitoring_jump_password = Column(String(512), nullable=True)
    monitoring_remote_auth_type = Column(String(20), default="key")
    monitoring_remote_key_path = Column(String(512), nullable=True)
    monitoring_remote_password = Column(String(512), nullable=True)
    use_push_gateway = Column(Boolean, default=False)
    push_gateway_port = Column(Integer, default=9091)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<MonitoringConfig(id={self.id}, server='{self.monitoring_server_ip}')>"


class InstallationLog(Base):
    """Log of installation attempts on nodes."""

    __tablename__ = "installation_logs"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), nullable=False)

    # Installation details
    action = Column(String(50), nullable=False)
    component = Column(String(100), nullable=False)
    success = Column(Boolean, default=False)
    output = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    node = relationship("Node", back_populates="installation_logs")

    def __repr__(self):
        return f"<InstallationLog(id={self.id}, node_id={self.node_id}, action='{self.action}')>"


class ControlNodeGroup(Base):
    """A group of control plane nodes (Slurm head nodes or Kubernetes control plane)."""

    __tablename__ = "control_node_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Control plane type: "slurm" or "kubernetes"
    control_type = Column(String(20), nullable=False, default="slurm")

    # Associated monitoring server (reuses existing MonitoringServer)
    monitoring_server_id = Column(Integer, ForeignKey("monitoring_servers.id"), nullable=True)

    # Custom exporter port override (0 = use default: 9418 for slurm, 9419 for k8s)
    custom_exporter_port = Column(Integer, default=0)

    # SSH credentials for control nodes (direct connection)
    ssh_user = Column(String(255), nullable=False, default="root")
    ssh_port = Column(Integer, nullable=False, default=22)
    ssh_auth_type = Column(String(20), nullable=False, default="key")
    ssh_key_path = Column(String(512), nullable=True)
    ssh_password = Column(String(512), nullable=True)

    # Jump host configuration (optional)
    use_jump_host = Column(Boolean, default=False)
    jump_host = Column(String(255), nullable=True)
    jump_port = Column(Integer, nullable=True, default=22)
    jump_user = Column(String(255), nullable=True)
    jump_auth_type = Column(String(20), nullable=True, default="key")
    jump_key_path = Column(String(512), nullable=True)
    jump_password = Column(String(512), nullable=True)

    # Credentials for control nodes when using jump host
    remote_auth_type = Column(String(20), nullable=True, default="key")
    remote_key_path = Column(String(512), nullable=True)
    remote_password = Column(String(512), nullable=True)

    # Kubeconfig for Kubernetes control plane (only relevant when control_type="kubernetes")
    # kubeconfig_source: how the exporter finds the kubeconfig
    #   "auto"   — exporter auto-detects /etc/kubernetes/admin.conf or ~/.kube/config (default)
    #   "path"   — user provides the path on the K8s node; stored in kubeconfig_remote_path
    #   "upload" — user uploads a kubeconfig file; stored in kubeconfig_local_path on Fleet Manager,
    #              then pushed to the K8s node at /etc/k8s-cp-exporter/kubeconfig during install
    kubeconfig_source = Column(String(20), nullable=True, default="auto")
    kubeconfig_remote_path = Column(String(512), nullable=True)  # path ON the K8s node
    kubeconfig_local_path = Column(String(512), nullable=True)  # path on Fleet Manager server

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    nodes = relationship("ControlNode", back_populates="control_node_group", cascade="all, delete-orphan")
    monitoring_server = relationship("MonitoringServer", back_populates="control_node_groups")

    def __repr__(self):
        return f"<ControlNodeGroup(id={self.id}, name='{self.name}', type='{self.control_type}')>"


class ControlNode(Base):
    """A single control plane node in a control node group."""

    __tablename__ = "control_nodes"

    id = Column(Integer, primary_key=True, index=True)
    control_node_group_id = Column(Integer, ForeignKey("control_node_groups.id"), nullable=False)

    # Node identification
    ip_address = Column(String(45), nullable=False)
    hostname = Column(String(255), nullable=True)

    # Status
    status = Column(String(20), default=NodeStatus.PENDING.value)
    status_message = Column(Text, nullable=True)
    last_seen = Column(DateTime, nullable=True)

    # Role/type-specific info (k8s node role, slurm partition, etc.)
    role_info = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    control_node_group = relationship("ControlNodeGroup", back_populates="nodes")

    # Unique IP per control node group
    __table_args__ = (Index("ix_control_nodes_group_ip", "control_node_group_id", "ip_address", unique=True),)

    def __repr__(self):
        return f"<ControlNode(id={self.id}, ip='{self.ip_address}', status='{self.status}')>"


# ============================================
# Alert Configuration Models
# ============================================


class ContactPointType(str, PyEnum):
    """Contact point types supported by Grafana."""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "msteams"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    WEBHOOK = "webhook"
    DISCORD = "discord"


class AlertSeverity(str, PyEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRuleCategory(str, PyEnum):
    """Categories of alert rules."""

    NODE_HEALTH = "node_health"
    GPU_HARDWARE = "gpu_hardware"
    THERMAL = "thermal"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    LOGS = "logs"


class AlertContactPoint(Base):
    """A notification contact point for alerts (email, Slack, Teams, etc.)."""

    __tablename__ = "alert_contact_points"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Contact point type
    contact_type = Column(String(50), nullable=False)

    # Configuration stored as JSON (varies by type)
    # Email: {"addresses": ["a@b.com"], "single_email": false}
    # Slack: {"recipient": "#channel", "url": "webhook_url"}
    # Teams: {"url": "webhook_url"}
    # PagerDuty: {"integration_key": "xxx", "severity": "critical"}
    # Webhook: {"url": "https://...", "http_method": "POST"}
    settings = Column(JSON, nullable=False, default=dict)

    # Grafana sync status
    grafana_uid = Column(String(255), nullable=True)
    last_synced_at = Column(DateTime, nullable=True)
    sync_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    alert_rules = relationship("AlertRule", back_populates="contact_point")

    def __repr__(self):
        return f"<AlertContactPoint(id={self.id}, name='{self.name}', type='{self.contact_type}')>"


class AlertRuleTemplate(Base):
    """Predefined alert rule templates."""

    __tablename__ = "alert_rule_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False)

    # Whether this is a default (pre-checked) alert
    is_default = Column(Boolean, default=False)

    # Alert definition
    datasource_type = Column(String(20), default="prometheus")  # prometheus or loki
    query_expression = Column(Text, nullable=False)

    # Threshold configuration (JSON)
    # {"operator": ">", "value": 90, "for_duration": "5m"}
    default_threshold = Column(JSON, nullable=False, default=dict)

    # Alert metadata
    default_severity = Column(String(20), default="warning")
    summary_template = Column(Text, nullable=True)
    description_template = Column(Text, nullable=True)
    runbook_url = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AlertRuleTemplate(id={self.id}, name='{self.name}', default={self.is_default})>"


class AlertRule(Base):
    """A configured alert rule for a monitoring server."""

    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Link to monitoring server (required)
    monitoring_server_id = Column(Integer, ForeignKey("monitoring_servers.id"), nullable=False)

    # Link to template (optional - for predefined alerts)
    template_id = Column(Integer, ForeignKey("alert_rule_templates.id"), nullable=True)

    # Contact point for notifications (optional)
    contact_point_id = Column(Integer, ForeignKey("alert_contact_points.id"), nullable=True)

    # Node group filter (optional - null means all node groups)
    node_group_id = Column(Integer, ForeignKey("node_groups.id"), nullable=True)

    # Alert configuration
    enabled = Column(Boolean, default=True)
    severity = Column(String(20), default="warning")

    # Custom query (if not using template)
    datasource_type = Column(String(20), default="prometheus")
    query_expression = Column(Text, nullable=True)

    # Threshold configuration (overrides template default)
    threshold_config = Column(JSON, nullable=True)

    # Labels to add to alert
    labels = Column(JSON, default=dict)

    # Annotations
    summary = Column(Text, nullable=True)
    runbook_url = Column(String(512), nullable=True)

    # Grafana sync status
    grafana_uid = Column(String(255), nullable=True)
    grafana_folder_uid = Column(String(255), default="fleet-alerts")
    last_synced_at = Column(DateTime, nullable=True)
    sync_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint: one rule per template per monitoring server per node group
    __table_args__ = (
        Index(
            "ix_alert_rules_server_template_ng",
            "monitoring_server_id",
            "template_id",
            "node_group_id",
            unique=True,
        ),
    )

    # Relationships
    monitoring_server = relationship("MonitoringServer", back_populates="alert_rules")
    template = relationship("AlertRuleTemplate")
    contact_point = relationship("AlertContactPoint", back_populates="alert_rules")
    node_group = relationship("NodeGroup")

    def __repr__(self):
        return f"<AlertRule(id={self.id}, name='{self.name}', enabled={self.enabled})>"


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    import logging

    logger = logging.getLogger(__name__)

    # Check if we should reset the database (drop all tables and recreate)
    reset_db = os.environ.get("RESET_DATABASE", "").lower() in ("true", "1", "yes")
    if reset_db:
        logger.warning("RESET_DATABASE is set - dropping all tables!")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped")

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")

    # Apply schema migrations for columns added after initial table creation.
    # ALTER TABLE ... ADD COLUMN IF NOT EXISTS is idempotent and safe to run on every startup.
    with engine.connect() as conn:
        migrations = [
            "ALTER TABLE monitoring_servers ADD COLUMN IF NOT EXISTS prometheus_storage_path VARCHAR(512)",
            # Kubeconfig support for Kubernetes control node groups
            "ALTER TABLE control_node_groups ADD COLUMN IF NOT EXISTS kubeconfig_source VARCHAR(20) DEFAULT 'auto'",
            "ALTER TABLE control_node_groups ADD COLUMN IF NOT EXISTS kubeconfig_remote_path VARCHAR(512)",
            "ALTER TABLE control_node_groups ADD COLUMN IF NOT EXISTS kubeconfig_local_path VARCHAR(512)",
        ]
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                logger.warning(f"Migration skipped ({stmt!r}): {e}")

    logger.info("Schema migrations applied")

    # Create default metric group if not exists
    db = SessionLocal()
    try:
        default_metric_group = db.query(MetricGroup).filter(MetricGroup.name == "default").first()
        if not default_metric_group:
            default_metric_group = MetricGroup(
                name="default",
                description="Default metric collection - GPU utilization, memory, temperature, power, and basic node metrics",
                gpu_utilization=True,
                gpu_memory=True,
                gpu_temperature=True,
                gpu_power=True,
                gpu_fan=False,
                gpu_clocks=False,
                gpu_pcie=False,
                gpu_ecc=False,
                node_cpu=True,
                node_memory=True,
                node_disk=True,
                node_network=False,
                collect_logs=True,
                log_patterns=[
                    "ECC",
                    "UE",
                    "CE",
                    "RAS",
                    "ras_error",
                    "xgmi",
                    "XGMI.*error",
                    "GPU hang",
                    "gpu reset",
                    "amdgpu.*timeout",
                    "Hardware Error",
                    "MCE",
                    "machine check",
                    "thermal throttle",
                    "temperature",
                ],
            )
            db.add(default_metric_group)
            db.commit()
            logger.info("Created default metric group")

        # Create legacy default metric config if not exists
        default_config = db.query(MetricConfig).filter(MetricConfig.name == "default").first()
        if not default_config:
            default_config = MetricConfig(
                name="default",
                fleet_health=True,
                thermal_power=True,
                utilization=True,
                memory=True,
                cpu_metrics=True,
                memory_metrics=True,
                collect_dmesg=True,
                collect_journalctl=True,
                log_patterns=[
                    "ECC",
                    "UE",
                    "CE",
                    "RAS",
                    "ras_error",
                    "xgmi",
                    "XGMI.*error",
                    "GPU hang",
                    "gpu reset",
                    "amdgpu.*timeout",
                    "Hardware Error",
                    "MCE",
                    "machine check",
                    "thermal throttle",
                    "temperature",
                ],
            )
            db.add(default_config)
            db.commit()

        # Seed default alert rule templates if not exists
        _seed_alert_templates(db, logger)
    finally:
        db.close()


def _seed_alert_templates(db: Session, logger):
    """Seed default alert rule templates.

    This function will insert any missing templates, allowing new templates
    to be added to existing installations without resetting the database.
    """
    templates = [
        # === DEFAULT ALERTS (pre-checked) ===
        {
            "name": "node_unreachable",
            "display_name": "Node Unreachable",
            "description": "Alert when GPU nodes become unreachable",
            "category": AlertRuleCategory.NODE_HEALTH.value,
            "is_default": True,
            "datasource_type": "prometheus",
            "query_expression": 'up{job="amd_gpu_metrics"} == 0',
            "default_threshold": {"operator": "==", "value": 0, "for_duration": "5m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "GPU node {{ $labels.instance }} is unreachable",
            "description_template": "The GPU exporter on {{ $labels.hostname }} ({{ $labels.instance }}) has been down for more than 5 minutes.",
        },
        {
            "name": "ecc_uncorrectable_errors",
            "display_name": "ECC Uncorrectable Errors",
            "description": "Alert on uncorrectable double-bit ECC memory errors",
            "category": AlertRuleCategory.GPU_HARDWARE.value,
            "is_default": True,
            "datasource_type": "prometheus",
            "query_expression": "increase(gpu_ecc_uncorrect_total[5m]) > 0",
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "Uncorrectable ECC error on {{ $labels.hostname }} GPU{{ $labels.gpu }}",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} has experienced uncorrectable ECC memory errors. This indicates potential hardware failure.",
        },
        {
            "name": "gpu_reset_failed",
            "display_name": "GPU Reset Failure",
            "description": "Alert when GPU reset fails to recover the device",
            "category": AlertRuleCategory.GPU_HARDWARE.value,
            "is_default": True,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(amdgpu.*reset.*fail|amdgpu.*timeout|gpu.*hang.*recovery.*fail)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "GPU reset failure detected on {{ $labels.hostname }}",
            "description_template": "A GPU on {{ $labels.hostname }} has failed to recover from a reset. Manual intervention may be required.",
        },
        {
            "name": "aer_pcie_errors",
            "display_name": "PCIe AER Errors",
            "description": "Alert on Advanced Error Reporting / PCIe bus errors (GPU or RDMA NIC falling off bus)",
            "category": AlertRuleCategory.GPU_HARDWARE.value,
            "is_default": True,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(AER.*error|AER.*Uncorrected|pcieport.*error|pcie.*fatal|pcie.*correctable)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "PCIe AER error detected on {{ $labels.hostname }}",
            "description_template": "PCIe Advanced Error Reporting has detected errors on {{ $labels.hostname }}. GPU or RDMA NIC may have fallen off the bus.",
        },
        # === OPTIONAL ALERTS (user can enable) ===
        {
            "name": "gpu_temperature_high",
            "display_name": "High GPU Temperature",
            "description": "Alert when GPU junction temperature exceeds threshold for sustained period",
            "category": AlertRuleCategory.THERMAL.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "gpu_junction_temperature",
            "default_threshold": {"operator": ">", "value": 90, "for_duration": "5m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High temperature on {{ $labels.hostname }} GPU{{ $labels.gpu }}: {{ $value }}C",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} has been above threshold for more than 5 minutes.",
        },
        {
            "name": "gpu_temperature_critical",
            "display_name": "Critical GPU Temperature",
            "description": "Alert when GPU junction temperature is critically high",
            "category": AlertRuleCategory.THERMAL.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "gpu_junction_temperature",
            "default_threshold": {"operator": ">", "value": 100, "for_duration": "1m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "CRITICAL temperature on {{ $labels.hostname }} GPU{{ $labels.gpu }}: {{ $value }}C",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} is at critically high temperature and may throttle or shut down.",
        },
        {
            "name": "gpu_memory_high",
            "display_name": "High GPU Memory Usage",
            "description": "Alert when GPU memory usage exceeds threshold",
            "category": AlertRuleCategory.MEMORY.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "100 * gpu_used_vram / gpu_total_vram",
            "default_threshold": {"operator": ">", "value": 95, "for_duration": "10m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High memory usage on {{ $labels.hostname }} GPU{{ $labels.gpu }}",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} memory usage has been above 95% for more than 10 minutes.",
        },
        {
            "name": "rdma_link_down",
            "display_name": "RDMA Link Down",
            "description": "Alert when RDMA network link goes down",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "rdma_link_physical_state == 0",
            "default_threshold": {"operator": "==", "value": 0, "for_duration": "2m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "RDMA link down: {{ $labels.hostname }} {{ $labels.device }}",
            "description_template": "RDMA device {{ $labels.device }} on {{ $labels.hostname }} physical link is down.",
        },
        {
            "name": "rdma_link_flap",
            "display_name": "RDMA Link Flapping",
            "description": "Alert when RDMA link state changes frequently",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "changes(rdma_link_state[10m]) > 2",
            "default_threshold": {"operator": ">", "value": 2, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "RDMA link flapping: {{ $labels.hostname }} {{ $labels.device }}",
            "description_template": "RDMA device {{ $labels.device }} on {{ $labels.hostname }} has changed state multiple times in 10 minutes.",
        },
        {
            "name": "xgmi_errors",
            "display_name": "XGMI Link Errors",
            "description": "Alert on XGMI interconnect errors between GPUs",
            "category": AlertRuleCategory.GPU_HARDWARE.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(xgmi.*error|xgmi.*fail)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "XGMI error detected on {{ $labels.hostname }}",
            "description_template": "XGMI interconnect errors detected on {{ $labels.hostname }}. Multi-GPU workloads may be affected.",
        },
        {
            "name": "ecc_correctable_high",
            "display_name": "High ECC Correctable Errors",
            "description": "Alert when correctable ECC errors exceed threshold (may indicate memory degradation)",
            "category": AlertRuleCategory.GPU_HARDWARE.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "increase(gpu_ecc_correct_total[1h])",
            "default_threshold": {"operator": ">", "value": 100, "for_duration": "5m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High ECC correctable errors on {{ $labels.hostname }} GPU{{ $labels.gpu }}",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} has had over 100 correctable ECC errors in the past hour. This may indicate memory degradation.",
        },
        {
            "name": "gpu_power_high",
            "display_name": "High GPU Power Draw",
            "description": "Alert when GPU power exceeds expected limits",
            "category": AlertRuleCategory.THERMAL.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "gpu_power_usage",
            "default_threshold": {"operator": ">", "value": 700, "for_duration": "5m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High power draw on {{ $labels.hostname }} GPU{{ $labels.gpu }}: {{ $value }}W",
            "description_template": "GPU {{ $labels.gpu }} on {{ $labels.hostname }} is drawing more than expected power.",
        },
        {
            "name": "node_cpu_high",
            "display_name": "High Node CPU Usage",
            "description": "Alert when host CPU usage is sustained high",
            "category": AlertRuleCategory.NODE_HEALTH.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": '100 - (avg by (hostname) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            "default_threshold": {"operator": ">", "value": 90, "for_duration": "10m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High CPU usage on {{ $labels.hostname }}",
            "description_template": "Host CPU usage on {{ $labels.hostname }} has been above 90% for more than 10 minutes.",
        },
        {
            "name": "node_memory_high",
            "display_name": "High Node Memory Usage",
            "description": "Alert when host memory usage is critically high",
            "category": AlertRuleCategory.NODE_HEALTH.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
            "default_threshold": {"operator": ">", "value": 95, "for_duration": "5m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High memory usage on {{ $labels.hostname }}",
            "description_template": "Host memory usage on {{ $labels.hostname }} has been above 95% for more than 5 minutes. OOM killer may activate.",
        },
        {
            "name": "node_disk_high",
            "display_name": "High Disk Usage",
            "description": "Alert when disk space is running low",
            "category": AlertRuleCategory.NODE_HEALTH.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": '100 * (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|devtmpfs"} / node_filesystem_size_bytes{fstype!~"tmpfs|devtmpfs"})',
            "default_threshold": {"operator": ">", "value": 90, "for_duration": "10m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High disk usage on {{ $labels.hostname }}: {{ $labels.mountpoint }}",
            "description_template": "Filesystem {{ $labels.mountpoint }} on {{ $labels.hostname }} is above 90% capacity.",
        },
        # === STORAGE & FILESYSTEM ALERTS ===
        {
            "name": "storage_disk_errors",
            "display_name": "Disk I/O Errors",
            "description": "Alert on disk read/write errors, bad sectors, or I/O failures from dmesg",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(disk.*error|i/o error|read.*error.*sector|write.*error.*sector|bad.*sector|buffer i/o|blk_update_request)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "Disk I/O errors detected on {{ $labels.hostname }}",
            "description_template": "Disk errors have been logged on {{ $labels.hostname }}. Check dmesg for details - may indicate failing drive.",
        },
        {
            "name": "storage_nvme_errors",
            "display_name": "NVMe Drive Errors",
            "description": "Alert on NVMe drive errors, timeouts, or resets",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(nvme.*error|nvme.*fail|nvme.*timeout|nvme.*reset)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "NVMe errors detected on {{ $labels.hostname }}",
            "description_template": "NVMe drive errors on {{ $labels.hostname }}. May indicate drive health issues.",
        },
        {
            "name": "storage_filesystem_errors",
            "display_name": "Filesystem Errors",
            "description": "Alert on filesystem corruption, journal aborts, or remount read-only events",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(ext4.*error|xfs.*error|btrfs.*error|filesystem.*corrupt|remount.*read-only|journal.*abort)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "Filesystem error on {{ $labels.hostname }}",
            "description_template": "Filesystem errors detected on {{ $labels.hostname }}. Check for data corruption and consider fsck.",
        },
        {
            "name": "storage_raid_degraded",
            "display_name": "RAID/MD Degraded",
            "description": "Alert when software RAID (md) array becomes degraded or has errors",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(md.*error|raid.*error|md.*degraded|raid.*degraded|md.*fail)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "RAID array degraded on {{ $labels.hostname }}",
            "description_template": "Software RAID array on {{ $labels.hostname }} is degraded or has errors. Check mdadm status.",
        },
        {
            "name": "storage_disk_usage_critical",
            "display_name": "Critical Disk Usage (>95%)",
            "description": "Alert when filesystem usage exceeds 95%",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": '100 * (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|devtmpfs|overlay|squashfs"} / node_filesystem_size_bytes{fstype!~"tmpfs|devtmpfs|overlay|squashfs"})',
            "default_threshold": {"operator": ">", "value": 95, "for_duration": "5m"},
            "default_severity": AlertSeverity.CRITICAL.value,
            "summary_template": "Critical disk usage on {{ $labels.hostname }}: {{ $labels.mountpoint }} at {{ $value | printf \"%.1f\" }}%",
            "description_template": "Filesystem {{ $labels.mountpoint }} on {{ $labels.hostname }} is above 95% - may cause service failures.",
        },
        {
            "name": "storage_inode_usage_high",
            "display_name": "High Inode Usage",
            "description": "Alert when filesystem inode usage exceeds 90%",
            "category": AlertRuleCategory.STORAGE.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": '100 * (1 - node_filesystem_files_free{fstype!~"tmpfs|devtmpfs|overlay|squashfs"} / node_filesystem_files{fstype!~"tmpfs|devtmpfs|overlay|squashfs"})',
            "default_threshold": {"operator": ">", "value": 90, "for_duration": "10m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High inode usage on {{ $labels.hostname }}: {{ $labels.mountpoint }}",
            "description_template": "Inode usage on {{ $labels.mountpoint }} ({{ $labels.hostname }}) is above 90%. Cannot create new files when exhausted.",
        },
        # === RDMA/NETWORK ALERTS ===
        {
            "name": "rdma_nic_errors",
            "display_name": "RDMA/NIC Errors (Logs)",
            "description": "Alert on RDMA, RoCE, or NIC driver errors from dmesg (mlx5, bnxt, ainic)",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(rdma.*error|rdma.*fail|roce.*error|mlx5.*error|mlx5.*fail|bnxt.*error|bnxt.*fail|ainic.*error)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "RDMA/NIC errors on {{ $labels.hostname }}",
            "description_template": "RDMA or network interface errors detected on {{ $labels.hostname }}. Check dmesg for driver issues.",
        },
        {
            "name": "network_link_down_logs",
            "display_name": "Network Link Down (Logs)",
            "description": "Alert when network link down events appear in dmesg",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(link.*down|link is down|nic link is down|carrier lost|carrier off)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "Network link down event on {{ $labels.hostname }}",
            "description_template": "Network link down detected on {{ $labels.hostname }}. May affect RDMA or data traffic.",
        },
        {
            "name": "network_tx_timeout",
            "display_name": "Network TX Timeout",
            "description": "Alert on network transmit timeout or watchdog events",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "loki",
            "query_expression": '{job=~"dmesg|systemd-journal"} |~ "(?i)(tx.*timeout|transmit.*timeout|netdev.*watchdog)"',
            "default_threshold": {"operator": ">", "value": 0, "for_duration": "1m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "Network TX timeout on {{ $labels.hostname }}",
            "description_template": "Network transmit timeout on {{ $labels.hostname }}. May indicate NIC or driver issues.",
        },
        {
            "name": "rdma_high_error_rate",
            "display_name": "RDMA High Error Rate",
            "description": "Alert when RDMA packet error rate exceeds threshold",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "sum by (hostname, device) (rate(rdma_stat_seq_err_naks_rcvd[5m]) + rate(rdma_stat_oos_drop_count[5m]) + rate(rdma_stat_rx_roce_errors[5m]))",
            "default_threshold": {"operator": ">", "value": 1, "for_duration": "5m"},
            "default_severity": AlertSeverity.WARNING.value,
            "summary_template": "High RDMA error rate on {{ $labels.hostname }} {{ $labels.device }}",
            "description_template": "RDMA device {{ $labels.device }} on {{ $labels.hostname }} is experiencing elevated error rates.",
        },
        {
            "name": "rdma_high_cnp_rate",
            "display_name": "RDMA High Congestion (CNP)",
            "description": "Alert when RDMA Congestion Notification Packet rate is high",
            "category": AlertRuleCategory.NETWORK.value,
            "is_default": False,
            "datasource_type": "prometheus",
            "query_expression": "sum by (hostname) (rate(rdma_stat_tx_cnp_pkts[5m]) + rate(rdma_stat_rx_cnp_pkts[5m]))",
            "default_threshold": {"operator": ">", "value": 100, "for_duration": "5m"},
            "default_severity": AlertSeverity.INFO.value,
            "summary_template": "High RDMA congestion on {{ $labels.hostname }}",
            "description_template": "{{ $labels.hostname }} is experiencing high RDMA network congestion. May affect distributed training performance.",
        },
    ]

    # Get existing template names
    existing_names = {t.name for t in db.query(AlertRuleTemplate.name).all()}

    added_count = 0
    for tmpl_data in templates:
        if tmpl_data["name"] not in existing_names:
            template = AlertRuleTemplate(**tmpl_data)
            db.add(template)
            added_count += 1

    if added_count > 0:
        db.commit()
        logger.info(f"Added {added_count} new alert rule templates (total templates: {len(templates)})")
    else:
        logger.info(f"All {len(templates)} alert rule templates already exist")
