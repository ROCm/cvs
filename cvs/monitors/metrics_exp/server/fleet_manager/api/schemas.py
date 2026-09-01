"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NodeStatus(str, Enum):
    """Node status enumeration."""

    PENDING = "pending"
    CONNECTED = "connected"  # SSH verified, ready for install
    INSTALLING = "installing"
    ACTIVE = "active"  # Exporters installed
    ERROR = "error"
    UNREACHABLE = "unreachable"


# ============================================
# Metric Configuration Schemas
# ============================================


class MetricConfigBase(BaseModel):
    """Base metric configuration."""

    name: str = Field(..., min_length=1, max_length=255)

    # GPU metrics
    fleet_health: bool = True
    thermal_power: bool = True
    pcie: bool = False
    xgmi: bool = False
    ecc: bool = False
    ras: bool = False
    utilization: bool = True
    memory: bool = True

    # Node metrics
    cpu_metrics: bool = True
    memory_metrics: bool = True
    disk_metrics: bool = True
    network_metrics: bool = False

    # Log collection
    collect_dmesg: bool = True
    collect_journalctl: bool = True
    log_patterns: List[str] = Field(default_factory=list)

    # Custom metrics
    custom_metrics: List[str] = Field(default_factory=list)


class MetricConfigCreate(MetricConfigBase):
    """Schema for creating metric config."""

    pass


class MetricConfigResponse(MetricConfigBase):
    """Schema for metric config response."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Node Schemas
# ============================================


class NodeBase(BaseModel):
    """Base node schema."""

    ip_address: str
    hostname: Optional[str] = None
    gpu_exporter_port: int = 5000
    node_exporter_port: int = 9100


class NodeCreate(NodeBase):
    """Schema for creating a node."""

    pass


class NodeBulkCreate(BaseModel):
    """Schema for bulk creating nodes from IP list."""

    ip_addresses: List[str]
    gpu_exporter_port: int = 5000
    node_exporter_port: int = 9100

    @field_validator('ip_addresses')
    @classmethod
    def validate_ips(cls, v):
        if not v:
            raise ValueError('At least one IP address is required')
        return v


class NodeResponse(NodeBase):
    """Schema for node response."""

    id: int
    node_group_id: int
    status: NodeStatus
    status_message: Optional[str] = None
    last_seen: Optional[datetime] = None
    gpu_count: Optional[int] = None
    gpu_model: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NodeStatusUpdate(BaseModel):
    """Schema for updating node status."""

    status: NodeStatus
    status_message: Optional[str] = None


# ============================================
# Node Group Schemas
# ============================================


class NodeGroupBase(BaseModel):
    """Base node group schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # SSH credentials for GPU nodes (direct connection or via jump host)
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_auth_type: str = "key"  # "key" or "password"
    ssh_password: Optional[str] = None  # For direct connection with password

    # Jump host configuration
    use_jump_host: bool = False
    jump_host: Optional[str] = None
    jump_port: int = 22
    jump_user: Optional[str] = None
    jump_auth_type: str = "key"  # "key" or "password"
    jump_password: Optional[str] = None  # Password for jump host

    # GPU node access from jump host
    remote_auth_type: str = "key"  # "key" or "password"
    remote_key_path: Optional[str] = None  # Path to SSH key on the jump host
    remote_password: Optional[str] = None  # Password for GPU nodes (when using jump host)


class NodeGroupCreate(NodeGroupBase):
    """Schema for creating a node group."""

    metric_config_id: Optional[int] = None
    monitoring_server_id: Optional[int] = None
    metric_group_id: Optional[int] = None


class NodeGroupCreateWithNodes(NodeGroupCreate):
    """Schema for creating a node group with initial nodes."""

    ip_addresses: List[str] = Field(default_factory=list)


class NodeGroupResponse(NodeGroupBase):
    """Schema for node group response."""

    id: int
    ssh_key_path: Optional[str] = None
    jump_key_path: Optional[str] = None
    metric_config_id: Optional[int] = None
    monitoring_server_id: Optional[int] = None
    metric_group_id: Optional[int] = None
    node_count: int = 0
    active_nodes: int = 0
    created_at: datetime
    updated_at: datetime

    # Hide passwords in response
    ssh_password: Optional[str] = Field(None, exclude=True)
    jump_password: Optional[str] = Field(None, exclude=True)
    remote_password: Optional[str] = Field(None, exclude=True)

    # Show if password is set (without revealing it)
    has_ssh_password: bool = False
    has_jump_password: bool = False
    has_remote_password: bool = False

    class Config:
        from_attributes = True


class NodeGroupDetail(NodeGroupResponse):
    """Detailed node group response with nodes."""

    nodes: List[NodeResponse] = Field(default_factory=list)
    metric_config: Optional[MetricConfigResponse] = None


class NodeGroupUpdate(BaseModel):
    """Schema for updating a node group."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_auth_type: Optional[str] = None
    ssh_password: Optional[str] = None
    metric_config_id: Optional[int] = None
    monitoring_server_id: Optional[int] = None
    metric_group_id: Optional[int] = None
    use_jump_host: Optional[bool] = None
    jump_host: Optional[str] = None
    jump_port: Optional[int] = None
    jump_user: Optional[str] = None
    jump_auth_type: Optional[str] = None
    jump_password: Optional[str] = None
    remote_auth_type: Optional[str] = None
    remote_key_path: Optional[str] = None
    remote_password: Optional[str] = None


# ============================================
# SSH Key Upload Schema
# ============================================


class SSHKeyUpload(BaseModel):
    """Schema for SSH key upload response."""

    key_path: str
    message: str


# ============================================
# Installation Schemas
# ============================================


class InstallationRequest(BaseModel):
    """Schema for requesting installation on nodes."""

    node_ids: Optional[List[int]] = None  # If None, install on all pending nodes
    force: bool = False  # Force reinstall even if already installed
    parallel_limit: int = Field(default=10, ge=1, le=50)  # Max concurrent SSH connections


class InstallationStatus(BaseModel):
    """Schema for installation status response."""

    node_id: int
    ip_address: str
    status: str
    message: Optional[str] = None


class InstallationResponse(BaseModel):
    """Schema for installation response."""

    job_id: str
    total_nodes: int
    statuses: List[InstallationStatus]


class InstallationLogResponse(BaseModel):
    """Schema for installation log response."""

    id: int
    node_id: int
    action: str
    component: str
    success: bool
    output: Optional[str] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================
# Dashboard Schemas
# ============================================


class DashboardInfo(BaseModel):
    """Schema for dashboard information."""

    uid: str
    title: str
    url: str
    node_group_id: Optional[int] = None


class DashboardList(BaseModel):
    """Schema for list of dashboards."""

    dashboards: List[DashboardInfo]


# ============================================
# Health Check Schemas
# ============================================


class ServiceHealth(BaseModel):
    """Schema for individual service health."""

    name: str
    status: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    services: List[ServiceHealth]


# ============================================
# Statistics Schemas
# ============================================


class FleetStats(BaseModel):
    """Schema for fleet statistics."""

    total_node_groups: int
    total_nodes: int
    active_nodes: int
    pending_nodes: int
    error_nodes: int
    total_gpus: int


# ============================================
# Monitoring Configuration Schemas
# ============================================


class MonitoringConfigBase(BaseModel):
    """Base monitoring configuration schema."""

    # Monitoring server settings
    monitoring_server_ip: Optional[str] = None
    monitoring_server_hostname: Optional[str] = None

    # Service ports
    prometheus_port: int = 30090
    loki_port: int = 30100
    grafana_port: int = 30030

    # Grafana credentials
    grafana_admin_user: str = "admin"
    grafana_admin_password: Optional[str] = None  # Don't expose in responses by default

    # Prometheus settings
    prometheus_retention_time: str = "15d"
    prometheus_retention_size: str = "50GB"
    prometheus_scrape_interval: str = "15s"

    # Loki settings
    loki_retention_days: int = 7

    # Push gateway for isolated networks
    use_push_gateway: bool = False
    push_gateway_port: int = 9091


class MonitoringConfigCreate(MonitoringConfigBase):
    """Schema for creating/updating monitoring config."""

    # Remote server setup (optional - for installing stack on remote server)
    setup_monitoring_stack: bool = False
    monitoring_ssh_user: Optional[str] = None
    monitoring_ssh_port: int = 22
    monitoring_ssh_auth_type: str = "password"  # "key" or "password"
    monitoring_ssh_key_path: Optional[str] = None
    monitoring_ssh_password: Optional[str] = None
    # Jump host configuration
    monitoring_use_jump_host: bool = False
    monitoring_jump_host: Optional[str] = None
    monitoring_jump_port: int = 22
    monitoring_jump_user: Optional[str] = None
    monitoring_jump_auth_type: str = "key"  # "key" or "password"
    monitoring_jump_password: Optional[str] = None
    monitoring_remote_auth_type: str = "key"  # "key" or "password"
    monitoring_remote_key_path: Optional[str] = None
    monitoring_remote_password: Optional[str] = None


class MonitoringConfigResponse(MonitoringConfigBase):
    """Schema for monitoring config response."""

    id: int
    setup_monitoring_stack: bool = False
    monitoring_ssh_user: Optional[str] = None
    monitoring_ssh_port: int = 22
    monitoring_ssh_auth_type: str = "password"  # "key" or "password"
    has_monitoring_ssh_key: bool = False
    has_monitoring_ssh_password: bool = False
    # Jump host fields
    monitoring_use_jump_host: bool = False
    monitoring_jump_host: Optional[str] = None
    monitoring_jump_port: int = 22
    monitoring_jump_user: Optional[str] = None
    monitoring_jump_auth_type: str = "key"
    has_monitoring_jump_key: bool = False
    has_monitoring_jump_password: bool = False
    monitoring_remote_auth_type: str = "key"
    monitoring_remote_key_path: Optional[str] = None
    has_monitoring_remote_password: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MonitoringConfigUpdate(BaseModel):
    """Schema for partial update of monitoring config."""

    monitoring_server_ip: Optional[str] = None
    monitoring_server_hostname: Optional[str] = None
    prometheus_port: Optional[int] = None
    loki_port: Optional[int] = None
    grafana_port: Optional[int] = None
    grafana_admin_user: Optional[str] = None
    grafana_admin_password: Optional[str] = None
    prometheus_retention_time: Optional[str] = None
    prometheus_retention_size: Optional[str] = None
    prometheus_scrape_interval: Optional[str] = None
    loki_retention_days: Optional[int] = None
    use_push_gateway: Optional[bool] = None
    push_gateway_port: Optional[int] = None
    setup_monitoring_stack: Optional[bool] = None
    monitoring_ssh_user: Optional[str] = None
    monitoring_ssh_port: Optional[int] = None
    monitoring_ssh_auth_type: Optional[str] = None  # "key" or "password"
    monitoring_ssh_password: Optional[str] = None
    # Jump host fields
    monitoring_use_jump_host: Optional[bool] = None
    monitoring_jump_host: Optional[str] = None
    monitoring_jump_port: Optional[int] = None
    monitoring_jump_user: Optional[str] = None
    monitoring_jump_auth_type: Optional[str] = None
    monitoring_jump_password: Optional[str] = None
    monitoring_remote_auth_type: Optional[str] = None
    monitoring_remote_key_path: Optional[str] = None
    monitoring_remote_password: Optional[str] = None


# ============================================
# Control Node Schemas
# ============================================


class ControlNodeCreate(BaseModel):
    """Schema for creating a control node."""

    ip_address: str
    hostname: Optional[str] = None


class ControlNodeBulkCreate(BaseModel):
    """Schema for bulk creating control nodes from IP list."""

    ip_addresses: List[str]

    @field_validator('ip_addresses')
    @classmethod
    def validate_ips(cls, v):
        if not v:
            raise ValueError('At least one IP address is required')
        return v


class ControlNodeResponse(BaseModel):
    """Schema for control node response."""

    id: int
    control_node_group_id: int
    ip_address: str
    hostname: Optional[str] = None
    status: NodeStatus
    status_message: Optional[str] = None
    last_seen: Optional[datetime] = None
    role_info: Optional[dict] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Control Node Group Schemas
# ============================================


class ControlNodeGroupBase(BaseModel):
    """Base control node group schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # Control plane type
    control_type: str = "slurm"  # "slurm" or "kubernetes"

    # Custom exporter port (0 = use default: 9418 for slurm, 9419 for k8s)
    custom_exporter_port: int = 0

    # SSH credentials for control nodes
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_auth_type: str = "key"  # "key" or "password"
    ssh_password: Optional[str] = None

    # Jump host configuration
    use_jump_host: bool = False
    jump_host: Optional[str] = None
    jump_port: int = 22
    jump_user: Optional[str] = None
    jump_auth_type: str = "key"
    jump_password: Optional[str] = None

    # Control node access from jump host
    remote_auth_type: str = "key"
    remote_key_path: Optional[str] = None
    remote_password: Optional[str] = None

    # Kubeconfig source (only relevant for control_type="kubernetes")
    # "auto"   — exporter auto-detects /etc/kubernetes/admin.conf or ~/.kube/config
    # "path"   — provide the path on the K8s node (kubeconfig_remote_path)
    # "upload" — upload a kubeconfig file via the UI
    kubeconfig_source: str = "auto"
    kubeconfig_remote_path: Optional[str] = None  # path on the K8s node


class ControlNodeGroupCreate(ControlNodeGroupBase):
    """Schema for creating a control node group."""

    monitoring_server_id: Optional[int] = None


class ControlNodeGroupCreateWithNodes(ControlNodeGroupCreate):
    """Schema for creating a control node group with initial nodes."""

    ip_addresses: List[str] = Field(default_factory=list)


class ControlNodeGroupResponse(ControlNodeGroupBase):
    """Schema for control node group response."""

    id: int
    ssh_key_path: Optional[str] = None
    jump_key_path: Optional[str] = None
    monitoring_server_id: Optional[int] = None
    node_count: int = 0
    active_node_count: int = 0
    created_at: datetime
    updated_at: datetime

    # Hide passwords in response
    ssh_password: Optional[str] = Field(None, exclude=True)
    jump_password: Optional[str] = Field(None, exclude=True)
    remote_password: Optional[str] = Field(None, exclude=True)

    # Show if password is set (without revealing it)
    has_ssh_password: bool = False
    has_jump_password: bool = False
    has_remote_password: bool = False

    # Kubeconfig (never expose the file content path, just the source and node path)
    kubeconfig_source: str = "auto"
    kubeconfig_remote_path: Optional[str] = None
    has_kubeconfig_upload: bool = False  # True if a kubeconfig file has been uploaded

    class Config:
        from_attributes = True


class ControlNodeGroupDetail(ControlNodeGroupResponse):
    """Detailed control node group response with nodes."""

    nodes: List[ControlNodeResponse] = Field(default_factory=list)


class ControlNodeGroupUpdate(BaseModel):
    """Schema for updating a control node group."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    control_type: Optional[str] = None
    custom_exporter_port: Optional[int] = None
    ssh_user: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_auth_type: Optional[str] = None
    ssh_password: Optional[str] = None
    monitoring_server_id: Optional[int] = None
    use_jump_host: Optional[bool] = None
    jump_host: Optional[str] = None
    jump_port: Optional[int] = None
    jump_user: Optional[str] = None
    jump_auth_type: Optional[str] = None
    jump_password: Optional[str] = None
    remote_auth_type: Optional[str] = None
    remote_key_path: Optional[str] = None
    remote_password: Optional[str] = None
    kubeconfig_source: Optional[str] = None
    kubeconfig_remote_path: Optional[str] = None


# ============================================
# Alert Configuration Schemas
# ============================================


class ContactPointType(str, Enum):
    """Contact point types supported by Grafana."""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "msteams"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    WEBHOOK = "webhook"
    DISCORD = "discord"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRuleCategory(str, Enum):
    """Categories of alert rules."""

    NODE_HEALTH = "node_health"
    GPU_HARDWARE = "gpu_hardware"
    THERMAL = "thermal"
    MEMORY = "memory"
    NETWORK = "network"
    LOGS = "logs"


class ThresholdOperator(str, Enum):
    """Threshold comparison operators."""

    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="


class ThresholdConfig(BaseModel):
    """Threshold configuration for an alert rule."""

    operator: ThresholdOperator = ThresholdOperator.GT
    value: float
    for_duration: str = "5m"


# --- Contact Point Schemas ---


class AlertContactPointBase(BaseModel):
    """Base schema for alert contact points."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    contact_type: ContactPointType
    settings: dict = Field(default_factory=dict)


class AlertContactPointCreate(AlertContactPointBase):
    """Schema for creating a contact point."""

    pass


class AlertContactPointUpdate(BaseModel):
    """Schema for updating a contact point."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    contact_type: Optional[ContactPointType] = None
    settings: Optional[dict] = None


class AlertContactPointResponse(BaseModel):
    """Schema for contact point response."""

    id: int
    name: str
    description: Optional[str] = None
    contact_type: str
    # Settings with sensitive fields redacted
    settings: dict = Field(default_factory=dict)
    has_credentials: bool = False

    grafana_uid: Optional[str] = None
    last_synced_at: Optional[datetime] = None
    sync_error: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    alert_rule_count: int = 0

    class Config:
        from_attributes = True


# --- Alert Rule Template Schemas ---


class AlertRuleTemplateResponse(BaseModel):
    """Schema for alert rule template response."""

    id: int
    name: str
    display_name: str
    description: Optional[str] = None
    category: str
    is_default: bool

    datasource_type: str
    query_expression: str
    default_threshold: dict
    default_severity: str
    summary_template: Optional[str] = None
    description_template: Optional[str] = None
    runbook_url: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Alert Rule Schemas ---


class AlertRuleBase(BaseModel):
    """Base schema for alert rules."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.WARNING


class AlertRuleCreate(AlertRuleBase):
    """Schema for creating an alert rule."""

    monitoring_server_id: int
    template_id: Optional[int] = None
    contact_point_id: Optional[int] = None
    node_group_id: Optional[int] = None  # None = all node groups

    # Custom query (if not using template)
    datasource_type: str = "prometheus"
    query_expression: Optional[str] = None

    # Threshold (overrides template default)
    threshold_config: Optional[ThresholdConfig] = None

    # Custom labels and annotations
    labels: dict = Field(default_factory=dict)
    summary: Optional[str] = None
    runbook_url: Optional[str] = None


class AlertRuleUpdate(BaseModel):
    """Schema for updating an alert rule."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    enabled: Optional[bool] = None
    severity: Optional[AlertSeverity] = None
    contact_point_id: Optional[int] = None
    node_group_id: Optional[int] = None
    threshold_config: Optional[ThresholdConfig] = None
    labels: Optional[dict] = None
    summary: Optional[str] = None
    runbook_url: Optional[str] = None


class AlertRuleResponse(AlertRuleBase):
    """Schema for alert rule response."""

    id: int
    monitoring_server_id: int
    template_id: Optional[int] = None
    contact_point_id: Optional[int] = None
    node_group_id: Optional[int] = None

    datasource_type: str
    query_expression: Optional[str] = None
    threshold_config: Optional[dict] = None
    labels: dict = Field(default_factory=dict)
    summary: Optional[str] = None
    runbook_url: Optional[str] = None

    grafana_uid: Optional[str] = None
    grafana_folder_uid: str = "fleet-alerts"
    last_synced_at: Optional[datetime] = None
    sync_error: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    # Expanded relationship names for UI
    template_name: Optional[str] = None
    contact_point_name: Optional[str] = None
    node_group_name: Optional[str] = None
    monitoring_server_name: Optional[str] = None

    class Config:
        from_attributes = True


class BulkAlertRuleCreate(BaseModel):
    """Schema for creating multiple alert rules from templates."""

    monitoring_server_id: int
    contact_point_id: Optional[int] = None
    node_group_id: Optional[int] = None
    template_ids: List[int] = Field(..., min_length=1)


class AlertConfigurationSummary(BaseModel):
    """Summary of alert configuration for a monitoring server."""

    monitoring_server_id: int
    monitoring_server_name: str
    total_rules: int
    enabled_rules: int
    disabled_rules: int
    rules_with_sync_errors: int
    contact_points: List[str]
    last_sync_at: Optional[datetime] = None


class AlertSyncResponse(BaseModel):
    """Response for alert sync operations."""

    success: bool
    message: str
    synced_count: int = 0
    error_count: int = 0
    errors: List[str] = Field(default_factory=list)


class AlertTestResponse(BaseModel):
    """Response for contact point test."""

    success: bool
    message: str
