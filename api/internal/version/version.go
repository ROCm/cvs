// Package version exposes build metadata for the daemon.
package version

// These are overridden at build time via -ldflags "-X ...".
var (
	// Version is the semantic version or git tag of the build.
	Version = "0.0.0-dev"
	// Commit is the git commit SHA of the build.
	Commit = "unknown"
	// BuildTime is the RFC3339 timestamp of the build.
	BuildTime = "unknown"
)
