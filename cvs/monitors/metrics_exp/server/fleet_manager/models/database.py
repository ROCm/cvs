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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://fleet:fleet_secret@localhost:5432/fleet_monitor")

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
    finally:
        db.close()
