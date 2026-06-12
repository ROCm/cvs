package testexec

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// ConfigRef points to one example config file for a suite.
type ConfigRef struct {
	Name   string `json:"name"`   // file name without extension
	Path   string `json:"path"`   // path relative to the config root
	Format string `json:"format"` // "json" | "yaml"
}

// configDirOverrides maps a suite's derived config subdir to its real one,
// for the cases where the test module path does not match the config tree.
// (e.g. test_aorta lives in cvs.tests.benchmark but its config is under aorta/.)
var configDirOverrides = map[string]string{
	"benchmark": "aorta",
}

// ConfigService resolves and loads example config files for suites. Suites map
// to a config subdirectory derived from their module path (with curated
// overrides); the files in that directory are the example configs.
type ConfigService struct {
	root string
}

// NewConfigService roots the service at the CVS config_file directory.
func NewConfigService(root string) *ConfigService {
	return &ConfigService{root: root}
}

// Root returns the configured config_file root.
func (c *ConfigService) Root() string { return c.root }

// LocateConfigDir resolves the CVS config_file directory. Precedence:
//  1. explicit (e.g. CVS_CONFIG_DIR) if it exists,
//  2. `python3 -c` lookup of the installed cvs package's input/config_file,
//  3. explicit value as-is (may not exist; callers degrade to empty results).
func LocateConfigDir(explicit, cvsBin string) string {
	if explicit != "" {
		if st, err := os.Stat(explicit); err == nil && st.IsDir() {
			return explicit
		}
	}
	if dir := pythonConfigDir(); dir != "" {
		return dir
	}
	return explicit
}

func pythonConfigDir() string {
	out, err := exec.Command("python3", "-c",
		"import cvs, os; print(os.path.join(os.path.dirname(cvs.__file__), 'input', 'config_file'))").Output()
	if err != nil {
		return ""
	}
	dir := strings.TrimSpace(string(out))
	if st, statErr := os.Stat(dir); statErr == nil && st.IsDir() {
		return dir
	}
	return ""
}

// configSubdir derives the config subdirectory for a suite from its module path
// (cvs.tests.inference.vllm -> inference/vllm), applying curated overrides.
func configSubdir(suite Suite) string {
	mod := suite.Module
	mod = strings.TrimPrefix(mod, "cvs.tests")
	mod = strings.TrimPrefix(mod, ".")
	sub := strings.ReplaceAll(mod, ".", "/")
	if override, ok := configDirOverrides[sub]; ok {
		return override
	}
	return sub
}

var configExts = map[string]string{".json": "json", ".yaml": "yaml", ".yml": "yaml"}

// Examples returns the example config files for a suite (empty if none).
func (c *ConfigService) Examples(suite Suite) ([]ConfigRef, error) {
	if c.root == "" {
		return []ConfigRef{}, nil
	}
	sub := configSubdir(suite)
	if sub == "" {
		return []ConfigRef{}, nil
	}
	dir := filepath.Join(c.root, sub)

	entries, err := os.ReadDir(dir)
	if err != nil {
		return []ConfigRef{}, nil // no config dir -> no examples
	}

	refs := []ConfigRef{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		format, ok := configExts[ext]
		if !ok {
			continue
		}
		rel := filepath.ToSlash(filepath.Join(sub, e.Name()))
		refs = append(refs, ConfigRef{
			Name:   strings.TrimSuffix(e.Name(), filepath.Ext(e.Name())),
			Path:   rel,
			Format: format,
		})
	}
	sort.Slice(refs, func(i, j int) bool { return refs[i].Path < refs[j].Path })
	return refs, nil
}

// Load reads and parses an example config. If exampleName is empty, the first
// example (alphabetical) is used. Returns the parsed value plus its ref.
func (c *ConfigService) Load(suite Suite, exampleName string) (any, ConfigRef, error) {
	examples, err := c.Examples(suite)
	if err != nil {
		return nil, ConfigRef{}, err
	}
	if len(examples) == 0 {
		return nil, ConfigRef{}, errNoConfig
	}

	ref := examples[0]
	if exampleName != "" {
		found := false
		for _, e := range examples {
			if e.Name == exampleName {
				ref, found = e, true
				break
			}
		}
		if !found {
			return nil, ConfigRef{}, fmt.Errorf("example %q not found for suite %q", exampleName, suite.Name)
		}
	}

	data, err := os.ReadFile(filepath.Join(c.root, filepath.FromSlash(ref.Path)))
	if err != nil {
		return nil, ConfigRef{}, fmt.Errorf("read config %s: %w", ref.Path, err)
	}

	value, err := parseConfig(data, ref.Format)
	if err != nil {
		return nil, ConfigRef{}, fmt.Errorf("parse config %s: %w", ref.Path, err)
	}
	return value, ref, nil
}

var errNoConfig = fmt.Errorf("no config for suite")

// IsNoConfig reports whether err indicates a suite has no example config.
func IsNoConfig(err error) bool { return err == errNoConfig }

func parseConfig(data []byte, format string) (any, error) {
	switch format {
	case "yaml":
		var v any
		if err := yaml.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return normalizeYAML(v), nil
	default:
		var v any
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return v, nil
	}
}

// normalizeYAML converts yaml.v3's map[string]interface{} / nested values into
// JSON-compatible structures (yaml.v3 already uses string keys for mappings).
func normalizeYAML(v any) any {
	switch t := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(t))
		for k, val := range t {
			m[k] = normalizeYAML(val)
		}
		return m
	case []any:
		s := make([]any, len(t))
		for i, val := range t {
			s[i] = normalizeYAML(val)
		}
		return s
	default:
		return v
	}
}
