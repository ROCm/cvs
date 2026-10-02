package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the cluster.yaml document (cluster: key) plus the node list.
type Config struct {
	NodesFile string        `yaml:"nodes_file"`
	SSH       SSHConfig     `yaml:"ssh"`
	Polling   PollingConfig `yaml:"polling"`
	Alerts    AlertsConfig  `yaml:"alerts"`
	Storage   StorageConfig `yaml:"storage"`
	RCCL      RCCLConfig    `yaml:"rccl"`
	HTTP      HTTPConfig    `yaml:"http"`
	Nodes     []string      `yaml:"-"`
	ConfigDir string        `yaml:"-"`
}

type HTTPConfig struct {
	CORSOrigins []string `yaml:"cors_origins"`
}

type SSHConfig struct {
	Username string         `yaml:"username"`
	KeyFile  string         `yaml:"key_file"`
	Password string         `yaml:"password,omitempty"`
	Timeout  int            `yaml:"timeout"`
	JumpHost JumpHostConfig `yaml:"jump_host"`
}

type JumpHostConfig struct {
	Enabled      bool   `yaml:"enabled"`
	Host         string `yaml:"host"`
	Username     string `yaml:"username"`
	Password     string `yaml:"password,omitempty"`
	KeyFile      string `yaml:"key_file"`
	NodeUsername string `yaml:"node_username"`
	NodeKeyFile  string `yaml:"node_key_file"`
}

type PollingConfig struct {
	Interval         int `yaml:"interval"`
	BatchSize        int `yaml:"batch_size"`
	StaggerDelay     int `yaml:"stagger_delay"`
	FailureThreshold int `yaml:"failure_threshold"`
}

type AlertsConfig struct {
	GPUTempThreshold float64 `yaml:"gpu_temp_threshold"`
}

type StorageConfig struct {
	Redis RedisConfig `yaml:"redis"`
}

type RedisConfig struct {
	URL                string `yaml:"url"`
	DB                 int    `yaml:"db"`
	Password           string `yaml:"password"`
	SnapshotMaxEntries int    `yaml:"snapshot_max_entries"`
	EventMaxEntries    int    `yaml:"event_max_entries"`
}

type RCCLConfig struct {
	RASPort               int             `yaml:"ras_port"`
	PollInterval          int             `yaml:"poll_interval"`
	CollectiveTimeoutSecs int             `yaml:"collective_timeout_secs"`
	RASMaxResponseBytes   int             `yaml:"ras_max_response_bytes"`
	DebugLogPath          string          `yaml:"debug_log_path"`
	Inspector             InspectorConfig `yaml:"inspector"`
}

type InspectorConfig struct {
	Enabled           bool   `yaml:"enabled"`
	Mode              string `yaml:"mode"`
	DumpDir           string `yaml:"dump_dir"`
	PollInterval      int    `yaml:"poll_interval"`
	MaxRecordsPerFile int    `yaml:"max_records_per_file"`
}

type fileRoot struct {
	Cluster Config `yaml:"cluster"`
}

func defaults() Config {
	return Config{
		NodesFile: "config/nodes.txt",
		SSH: SSHConfig{
			Username: "root",
			KeyFile:  "~/.ssh/id_rsa",
			Timeout:  30,
			JumpHost: JumpHostConfig{
				KeyFile:      "~/.ssh/id_rsa",
				NodeUsername: "root",
				NodeKeyFile:  "~/.ssh/id_rsa",
			},
		},
		Polling: PollingConfig{Interval: 60, BatchSize: 10, StaggerDelay: 2, FailureThreshold: 5},
		Alerts:  AlertsConfig{GPUTempThreshold: 85},
		Storage: StorageConfig{Redis: RedisConfig{URL: "redis://localhost:6379", SnapshotMaxEntries: 1000, EventMaxEntries: 10000}},
		RCCL: RCCLConfig{
			RASPort: 28028, PollInterval: 30, CollectiveTimeoutSecs: 10,
			RASMaxResponseBytes: 4 * 1024 * 1024,
			Inspector:           InspectorConfig{Mode: "file", PollInterval: 30, MaxRecordsPerFile: 100},
		},
		HTTP: HTTPConfig{CORSOrigins: []string{"http://localhost:3000", "http://localhost:5173"}},
	}
}

// Load reads cluster.yaml and nodes.txt. Passwords in YAML are accepted on
// cold start; the HTTP config API keeps subsequent passwords in memory only.
func Load(configDir string) (*Config, error) {
	cfg := defaults()
	cfg.ConfigDir = configDir

	yamlPath := filepath.Join(configDir, "cluster.yaml")
	if b, err := os.ReadFile(yamlPath); err == nil {
		var root fileRoot
		root.Cluster = cfg
		if err := yaml.Unmarshal(b, &root); err != nil {
			return nil, fmt.Errorf("parse %s: %w", yamlPath, err)
		}
		cfg = root.Cluster
		cfg.ConfigDir = configDir
	}

	if cfg.Polling.Interval <= 0 {
		cfg.Polling.Interval = 60
	}
	if cfg.Polling.FailureThreshold <= 0 {
		cfg.Polling.FailureThreshold = 5
	}
	if cfg.SSH.Timeout <= 0 {
		cfg.SSH.Timeout = 30
	}
	if cfg.RCCL.RASPort <= 0 {
		cfg.RCCL.RASPort = 28028
	}
	if cfg.RCCL.PollInterval <= 0 {
		cfg.RCCL.PollInterval = 30
	}
	if cfg.RCCL.CollectiveTimeoutSecs <= 0 {
		cfg.RCCL.CollectiveTimeoutSecs = 10
	}
	if cfg.RCCL.RASMaxResponseBytes <= 0 {
		cfg.RCCL.RASMaxResponseBytes = 4 * 1024 * 1024
	}
	if cfg.RCCL.RASMaxResponseBytes > 64*1024*1024 {
		cfg.RCCL.RASMaxResponseBytes = 64 * 1024 * 1024
	}
	if v := os.Getenv("CORS_ORIGINS"); v != "" {
		var origins []string
		for _, p := range strings.Split(v, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				origins = append(origins, p)
			}
		}
		if len(origins) > 0 {
			cfg.HTTP.CORSOrigins = origins
		}
	}

	nodes, err := loadNodes(configDir, cfg.NodesFile)
	if err != nil {
		return nil, err
	}
	cfg.Nodes = nodes
	return &cfg, nil
}

func loadNodes(configDir, nodesFile string) ([]string, error) {
	candidates := []string{
		filepath.Join(configDir, "nodes.txt"),
		nodesFile,
	}
	for _, p := range candidates {
		if p == "" {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var nodes []string
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			nodes = append(nodes, line)
		}
		if len(nodes) > 0 {
			return nodes, nil
		}
	}
	return []string{}, nil
}

// SaveNodes writes nodes.txt in the config directory.
func SaveNodes(configDir string, nodes []string) error {
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		return err
	}
	var b strings.Builder
	b.WriteString("# GPU Cluster Nodes\n# Auto-generated from web UI\n\n")
	for _, n := range nodes {
		n = strings.TrimSpace(n)
		if n == "" {
			continue
		}
		b.WriteString(n)
		b.WriteByte('\n')
	}
	return os.WriteFile(filepath.Join(configDir, "nodes.txt"), []byte(b.String()), 0o644)
}

// SaveYAML writes cluster.yaml without SSH passwords.
func SaveYAML(configDir string, cfg *Config) error {
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		return err
	}
	copy := *cfg
	copy.SSH.Password = ""
	copy.SSH.JumpHost.Password = ""
	root := fileRoot{Cluster: copy}
	b, err := yaml.Marshal(&root)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(configDir, "cluster.yaml"), b, 0o644)
}

// ExpandPath expands leading ~ to the current user's home directory.
func ExpandPath(p string) string {
	if p == "" {
		return p
	}
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return p
		}
		if p == "~" {
			return home
		}
		return home + p[1:]
	}
	return p
}

// ResolveConfigDir picks the Docker path, then a relative path.
func ResolveConfigDir() string {
	for _, p := range []string{"/app/config", "config", "../config"} {
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			abs, err := filepath.Abs(p)
			if err == nil {
				return abs
			}
			return p
		}
	}
	return "/app/config"
}

// NormalizeSSHKeyPath maps host ~/.ssh paths to /root/.ssh inside the container,
// matching the Python config API. Paths that look like they live on the jump
// host (contain "@") are left unchanged.
func NormalizeSSHKeyPath(path string) string {
	if path == "" {
		return path
	}
	if strings.Contains(path, "@") && strings.Contains(path, "/home/") {
		return path
	}
	if strings.HasPrefix(path, "/home/") && strings.Contains(path, "/.ssh/") {
		return "/root/.ssh/" + filepath.Base(path)
	}
	if strings.HasPrefix(path, "~/") {
		return "/root/" + strings.TrimPrefix(path, "~/")
	}
	return path
}
