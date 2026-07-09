package testexec

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestConfigSubdir(t *testing.T) {
	cases := map[string]string{
		"cvs.tests.rccl":           "rccl",
		"cvs.tests.inference.vllm": "inference/vllm",
		"cvs.tests.benchmark":      "aorta", // curated override
		"cvs.tests":                "",
	}
	for module, want := range cases {
		got := configSubdir(Suite{Module: module})
		if got != want {
			t.Errorf("configSubdir(%q) = %q, want %q", module, got, want)
		}
	}
}

func TestExamplesAndLoad(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "rccl", "rccl_config.json"), `{"rccl":{"mpi_pml":"auto"}}`)
	writeFile(t, filepath.Join(root, "aorta", "aorta_benchmark.yaml"), "aorta:\n  iters: 5\n")

	svc := NewConfigService(root)

	rccl := Suite{Name: "rccl_perf", Module: "cvs.tests.rccl"}
	examples, err := svc.Examples(rccl)
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != 1 || examples[0].Path != "rccl/rccl_config.json" || examples[0].Format != "json" {
		t.Fatalf("unexpected examples: %+v", examples)
	}

	// benchmark suite resolves to the aorta/ dir (override) and parses YAML.
	aorta := Suite{Name: "test_aorta", Module: "cvs.tests.benchmark"}
	val, ref, err := svc.Load(aorta, "")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Format != "yaml" || ref.Path != "aorta/aorta_benchmark.yaml" {
		t.Fatalf("unexpected ref: %+v", ref)
	}
	m, ok := val.(map[string]any)
	if !ok {
		t.Fatalf("expected map, got %T", val)
	}
	if _, ok := m["aorta"]; !ok {
		t.Fatalf("yaml not parsed into map: %+v", m)
	}
}

func TestLoadNoConfig(t *testing.T) {
	svc := NewConfigService(t.TempDir())
	_, _, err := svc.Load(Suite{Name: "conftest", Module: "cvs.tests"}, "")
	if !IsNoConfig(err) {
		t.Fatalf("want IsNoConfig, got %v", err)
	}
}

func TestInferSchemaGolden(t *testing.T) {
	value := map[string]any{
		"rccl": map[string]any{
			"_comment_mpi_pml": "MPI layer",
			"mpi_pml":          "auto",
			"no_of_nodes":      "2",
			"enabled":          true,
			"collectives":      []any{"all_reduce_perf", "all_gather_perf"},
		},
	}

	got := InferSchema(value)
	want := []Field{
		{
			Name: "rccl",
			Type: "object",
			Fields: []Field{
				{Name: "collectives", Type: "array", Item: &Field{Name: "", Type: "string"}},
				{Name: "enabled", Type: "boolean"},
				{Name: "mpi_pml", Type: "string", Description: "MPI layer"},
				{Name: "no_of_nodes", Type: "string"},
			},
		},
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("schema mismatch:\n got=%+v\nwant=%+v", got, want)
	}
}
