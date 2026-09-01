"""
Cluster file configuration schema.

Mirrors ``cvs/input/cluster_file/``. Used by ``cvs.schema.validate.validate_config_file``.
"""

import warnings
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClusterNodeConfig(BaseModel):
    """Schema for a single node entry in cluster.json node_dict."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields like bmc_ip, rack_id

    vpc_ip: str = Field(description="VPC IP or hostname for inter-node communication")
    bmc_ip: Optional[str] = Field(default=None, description="BMC IP for out-of-band management")


class HeadNodeConfig(BaseModel):
    """Schema for head_node_dict in cluster.json."""

    model_config = ConfigDict(extra="allow")

    mgmt_ip: str = Field(description="Management IP of head node")


class RackConfig(BaseModel):
    """
    Schema for a single rack entry inside the 'racks' block of cluster.json.

    A rack groups compute trays (referenced via node_dict rack_id) and the
    switch trays physically associated with that rack.
    """

    model_config = ConfigDict(extra="allow")

    platform: Optional[str] = Field(default=None, description="ARC platform name, e.g. 'HeliosP' or 'HeliosR'")
    arc_controller: Optional[str] = Field(
        default=None,
        description="IP of the ARC controller node. Defaults to first sorted node_dict entry with matching rack_id.",
    )
    switch_trays: List[str] = Field(
        default_factory=list,
        description="IPs of switch trays in this rack",
    )
    rmc: Optional[str] = Field(default=None, description="IP of the Rack Management Controller")


class RacksBlock(BaseModel):
    """
    Schema for the top-level 'racks' field in cluster.json.

    Holds optional global switch credentials and one RackConfig entry per rack
    (keyed by rack ID, e.g. 'rack-01'). Extra keys (rack IDs) are accepted via
    extra='allow' and retrieved via get_racks().

    Switch credentials are fleet-wide (homogeneous across all racks). Per-rack
    overrides are not supported in the current exec path; add them to RackConfig
    when that need arises.
    """

    model_config = ConfigDict(extra="allow")

    switch_ssh_user: Optional[str] = Field(
        default=None,
        description="SSH username for all switch trays in every rack.",
    )
    switch_ssh_password: Optional[str] = Field(
        default=None,
        description="SSH password for all switch trays. Ignored when switch_ssh_key_file is set.",
    )
    switch_ssh_key_file: Optional[str] = Field(
        default=None,
        description="Path to SSH private key for all switch trays. Takes priority over switch_ssh_password when set.",
    )

    def get_racks(self):
        """Return only the rack entries, excluding credential fields."""
        skip = {'switch_ssh_user', 'switch_ssh_password', 'switch_ssh_key_file'}
        result = {}
        for key, value in (self.__pydantic_extra__ or {}).items():
            if key not in skip and isinstance(value, dict):
                result[key] = RackConfig(**value)
        return result


class ClusterConfigFile(BaseModel):
    """
    Schema for cluster.json configuration file.

    Validates the cluster configuration before running benchmarks.
    Fails fast with clear error messages if required fields are missing.
    """

    model_config = ConfigDict(extra="allow")

    username: str = Field(description="SSH username for cluster nodes")
    priv_key_file: Optional[str] = Field(default=None, description="Path to SSH private key")
    password: Optional[str] = Field(default=None, description="SSH password (if not using key)")

    node_dict: Dict[str, ClusterNodeConfig] = Field(
        description="Dictionary mapping node hostname/IP to node configuration"
    )
    head_node_dict: Optional[HeadNodeConfig] = Field(default=None, description="Head node configuration")

    racks: Optional[RacksBlock] = Field(
        default=None,
        description=(
            "Rack topology block. Contains optional global switch credentials and one entry per rack "
            "(keyed by rack ID) listing switch_trays and platform."
        ),
    )
    rack_groups: Optional[RacksBlock] = Field(
        default=None,
        description="Deprecated alias for 'racks'. Use 'racks' instead.",
    )

    # Optional fields that may be present
    home_mount_dir_name: Optional[str] = Field(default="home")
    node_dir_name: Optional[str] = Field(default="root")

    @model_validator(mode='after')
    def validate_auth_method(self):
        """Ensure at least one authentication method is provided."""
        if not self.priv_key_file and not self.password:
            raise ValueError("Authentication required: provide either 'priv_key_file' or 'password' in cluster config")
        return self

    @model_validator(mode='after')
    def validate_nodes_exist(self):
        """Ensure at least one node is configured."""
        if not self.node_dict:
            raise ValueError("No nodes configured in 'node_dict' - at least one node is required")
        return self

    @model_validator(mode='after')
    def warn_rack_groups_deprecated(self):
        """Emit a deprecation warning when the old 'rack_groups' key is used."""
        if self.rack_groups is not None and self.racks is None:
            warnings.warn(
                "'rack_groups' in cluster.json is deprecated. Rename it to 'racks'.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    def get_racks_block(self):
        """Return the active racks block, preferring 'racks' over the deprecated 'rack_groups'."""
        return self.racks if self.racks is not None else self.rack_groups

    @field_validator('username')
    @classmethod
    def validate_username_not_placeholder(cls, v):
        """Check that username is not still a placeholder."""
        if '<changeme>' in v.lower():
            raise ValueError(
                "Username contains placeholder '<changeme>'. Please set a valid username in cluster config."
            )
        return v


__all__ = [
    "ClusterConfigFile",
    "ClusterNodeConfig",
    "HeadNodeConfig",
    "RackConfig",
    "RacksBlock",
]
