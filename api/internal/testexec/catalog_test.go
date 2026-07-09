package testexec

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestParseFacets(t *testing.T) {
	cases := []struct {
		path string
		want ConfigRef
	}{
		{
			path: "training/jax/mi300x_jax_llama3_1_405b_distributed.json",
			want: ConfigRef{
				Category: "training", Framework: "jax", Platform: "mi300x",
				Model: "llama3_1_405b", Topology: "distributed",
			},
		},
		{
			path: "inference/pytorch_xdit/mi300x_pytorch_xdit_flux1_dev_single.json",
			want: ConfigRef{
				Category: "inference", Framework: "pytorch_xdit", Platform: "mi300x",
				Model: "flux1_dev", Topology: "single",
			},
		},
		{
			// Flat single-config dir: no framework, no facet decomposition.
			path: "rccl/rccl_config.json",
			want: ConfigRef{Category: "rccl"},
		},
		{
			// Two-segment health config: category only.
			path: "health/mi300_health_config.json",
			want: ConfigRef{Category: "health"},
		},
	}
	for _, tc := range cases {
		ref := ConfigRef{Name: trimExt(filepath.Base(tc.path)), Path: tc.path}
		parseFacets(&ref)
		got := ConfigRef{
			Category: ref.Category, Framework: ref.Framework, Platform: ref.Platform,
			Model: ref.Model, Topology: ref.Topology,
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseFacets(%q) facets = %+v, want %+v", tc.path, got, tc.want)
		}
	}
}

func trimExt(name string) string {
	return name[:len(name)-len(filepath.Ext(name))]
}

func TestCatalog(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "training", "jax", "mi300x_jax_llama3_1_70b_single.json"), `{}`)
	writeFile(t, filepath.Join(root, "training", "jax", "mi300x_jax_llama3_1_405b_distributed.json"), `{}`)
	writeFile(t, filepath.Join(root, "training", "megatron", "mi3xx_megatron_llama_distributed.json"), `{}`)
	writeFile(t, filepath.Join(root, "rccl", "rccl_config.json"), `{}`)
	writeFile(t, filepath.Join(root, "aorta", "aorta_benchmark.yaml"), "aorta: {}\n")
	writeFile(t, filepath.Join(root, "ignore.txt"), "not a config")

	cat, err := NewConfigService(root).Catalog()
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Configs) != 5 {
		t.Fatalf("expected 5 configs, got %d: %+v", len(cat.Configs), cat.Configs)
	}

	wantFacets := Facets{
		Category:  []string{"aorta", "rccl", "training"},
		Framework: []string{"jax", "megatron"},
		Platform:  []string{"mi300x", "mi3xx"},
		Model:     []string{"llama", "llama3_1_405b", "llama3_1_70b"},
		Topology:  []string{"distributed", "single"},
	}
	if !reflect.DeepEqual(cat.Facets, wantFacets) {
		t.Fatalf("facets = %+v, want %+v", cat.Facets, wantFacets)
	}
}

func TestCatalogEmptyRoot(t *testing.T) {
	cat, err := NewConfigService("").Catalog()
	if err != nil {
		t.Fatal(err)
	}
	if len(cat.Configs) != 0 {
		t.Fatalf("expected empty catalog, got %+v", cat.Configs)
	}
}
