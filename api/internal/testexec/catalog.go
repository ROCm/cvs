package testexec

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// knownTopologies are the recognized trailing tokens in a config filename.
// Anything else is treated as "no topology" (and folded back into the model),
// which keeps flat/odd filenames from producing bogus facets.
var knownTopologies = map[string]bool{
	"single":      true,
	"distributed": true,
}

// newConfigRef builds a ConfigRef from a config path relative to the config
// root, parsing the directory + filename into facets where the convention
// `category/framework/{platform}_{framework}_{model}_{topology}.ext` applies.
func newConfigRef(rel, format string) ConfigRef {
	ref := ConfigRef{
		Name:   strings.TrimSuffix(filepath.Base(rel), filepath.Ext(rel)),
		Path:   rel,
		Format: format,
	}
	parseFacets(&ref)
	return ref
}

// parseFacets fills the facet fields of ref from its path. It is deliberately
// forgiving: configs that do not match the convention (e.g. flat single-config
// dirs) simply get empty facets and behave as a single choice in the UI.
func parseFacets(ref *ConfigRef) {
	parts := strings.Split(ref.Path, "/")
	if len(parts) >= 2 {
		ref.Category = parts[0]
	}
	// A framework subdir only exists for nested catalogs:
	// category/framework/file. Flat configs (category/file) have no framework.
	if len(parts) >= 3 {
		ref.Framework = parts[len(parts)-2]
	}

	base := ref.Name
	tokens := strings.Split(base, "_")
	if ref.Framework == "" || len(tokens) < 2 {
		// Flat/odd config: no platform/model/topology decomposition.
		return
	}

	ref.Platform = tokens[0]

	remainder := strings.TrimPrefix(base, ref.Platform+"_")
	remainder = strings.TrimPrefix(remainder, ref.Framework+"_")

	if i := strings.LastIndex(remainder, "_"); i >= 0 {
		last := remainder[i+1:]
		if knownTopologies[last] {
			ref.Topology = last
			remainder = remainder[:i]
		}
	}
	ref.Model = remainder
}

// Facets is the set of distinct facet values across a collection of configs,
// each sorted, for populating UI dropdowns.
type Facets struct {
	Category  []string `json:"category"`
	Framework []string `json:"framework"`
	Platform  []string `json:"platform"`
	Model     []string `json:"model"`
	Topology  []string `json:"topology"`
}

// Catalog is the global view of every example config plus the distinct facet
// values, used by the faceted config picker.
type Catalog struct {
	Configs []ConfigRef `json:"configs"`
	Facets  Facets      `json:"facets"`
}

// Catalog walks the entire config root and returns every example config with
// its parsed facets, plus the distinct facet values. It never errors on a
// missing/empty root — it returns an empty catalog instead.
func (c *ConfigService) Catalog() (Catalog, error) {
	cat := Catalog{Configs: []ConfigRef{}}
	if c.root == "" {
		return cat, nil
	}

	err := filepath.WalkDir(c.root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries rather than aborting the walk
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		format, ok := configExts[ext]
		if !ok {
			return nil
		}
		rel, relErr := filepath.Rel(c.root, path)
		if relErr != nil {
			return nil
		}
		cat.Configs = append(cat.Configs, newConfigRef(filepath.ToSlash(rel), format))
		return nil
	})
	if err != nil {
		return cat, err
	}

	sort.Slice(cat.Configs, func(i, j int) bool { return cat.Configs[i].Path < cat.Configs[j].Path })
	cat.Facets = collectFacets(cat.Configs)
	return cat, nil
}

func collectFacets(configs []ConfigRef) Facets {
	cat := newStringSet()
	fw := newStringSet()
	plat := newStringSet()
	model := newStringSet()
	topo := newStringSet()
	for _, c := range configs {
		cat.add(c.Category)
		fw.add(c.Framework)
		plat.add(c.Platform)
		model.add(c.Model)
		topo.add(c.Topology)
	}
	return Facets{
		Category:  cat.sorted(),
		Framework: fw.sorted(),
		Platform:  plat.sorted(),
		Model:     model.sorted(),
		Topology:  topo.sorted(),
	}
}

type stringSet map[string]struct{}

func newStringSet() stringSet { return stringSet{} }

func (s stringSet) add(v string) {
	if v != "" {
		s[v] = struct{}{}
	}
}

func (s stringSet) sorted() []string {
	out := make([]string, 0, len(s))
	for v := range s {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
