// Package inventory implements the inventory-first configuration gate: the
// single source of truth for the fleet (nodes + SSH access) that every tile
// builds on. No tile is usable until an inventory has been saved.
package inventory

import "time"

// AuthMethod is how the platform authenticates SSH to nodes.
type AuthMethod string

const (
	// AuthKey authenticates with an uploaded private key (default, recommended).
	AuthKey AuthMethod = "key"
	// AuthPassword authenticates with a password (less secure).
	AuthPassword AuthMethod = "password"
)

// NodeStatus is the last known probe result for a node. It is populated by the
// F2 connectivity probe; F1 leaves it empty.
type NodeStatus struct {
	Host        string    `json:"host"`
	Reachable   bool      `json:"reachable"`
	GPUType     string    `json:"gpu_type,omitempty"`
	GPUCount    int       `json:"gpu_count,omitempty"`
	ROCmVersion string    `json:"rocm_version,omitempty"`
	Error       string    `json:"error,omitempty"`
	CheckedAt   time.Time `json:"checked_at,omitempty"`
}

// JumpHost is an optional bastion used to reach nodes that are not directly
// routable from the container.
type JumpHost struct {
	Host         string     `json:"host"`
	Username     string     `json:"username"`
	AuthMethod   AuthMethod `json:"auth_method"`
	KeyName      string     `json:"key_name,omitempty"`
	NodeUsername string     `json:"node_username,omitempty"`
	NodeKeyFile  string     `json:"node_key_file,omitempty"`
}

// Inventory is the single fleet document shared by all tiles.
type Inventory struct {
	Nodes      []string   `json:"nodes"`
	Username   string     `json:"username"`
	AuthMethod AuthMethod `json:"auth_method"`
	// KeyName is the uploaded private-key filename (relative to the key dir).
	KeyName   string       `json:"key_name,omitempty"`
	JumpHost  *JumpHost    `json:"jump_host,omitempty"`
	Statuses  []NodeStatus `json:"statuses,omitempty"`
	UpdatedAt time.Time    `json:"updated_at"`
}
