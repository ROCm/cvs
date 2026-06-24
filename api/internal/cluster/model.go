// Package cluster implements the saved-cluster catalog (S3b): a global,
// persisted record of cluster_json files generated from subsets of the shared
// inventory. Generation is delegated to the authoritative `cvs generate
// cluster_json` CLI (a pure template render, no SSH); this package only selects
// the node subset, invokes the CLI, and tracks the result for reuse.
//
// A saved cluster overlaps Fleet's NodeGroup concept; the two are expected to
// converge under Postgres in a later slice.
package cluster

import "time"

// Cluster is one saved, reusable cluster_json file.
type Cluster struct {
	ID       string   `json:"id"`
	Name     string   `json:"name"`
	Nodes    []string `json:"nodes"`
	HeadNode string   `json:"head_node,omitempty"`
	// FilePath is the container-local path of the generated cluster_json on the
	// persistent data volume (e.g. /app/data/clusters/<id>.json).
	FilePath  string    `json:"file_path"`
	Source    string    `json:"source"` // "generated"
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
