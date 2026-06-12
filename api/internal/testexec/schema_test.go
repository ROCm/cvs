package testexec

import "testing"

// findField returns the inferred field with the given name, or nil.
func findField(fields []Field, name string) *Field {
	for i := range fields {
		if fields[i].Name == name {
			return &fields[i]
		}
	}
	return nil
}

func TestInferSchemaFoldsCommentAndExample(t *testing.T) {
	cfg := map[string]any{
		"_comment_nccl_ib_hca": "RDMA HCA list",
		"_example_nccl_ib_hca": "bnxt_re0,bnxt_re1",
		"nccl_ib_hca":          "<changeme>",
		"plain":                "value",
	}
	fields := InferSchema(cfg)

	// Metadata keys must not surface as their own fields.
	for _, f := range fields {
		if f.Name == "_comment_nccl_ib_hca" || f.Name == "_example_nccl_ib_hca" {
			t.Fatalf("metadata key leaked as field: %q", f.Name)
		}
	}

	hca := findField(fields, "nccl_ib_hca")
	if hca == nil {
		t.Fatal("nccl_ib_hca field missing")
	}
	if hca.Description != "RDMA HCA list" {
		t.Errorf("description = %q, want %q", hca.Description, "RDMA HCA list")
	}
	if hca.Example != "bnxt_re0,bnxt_re1" {
		t.Errorf("example = %q, want %q", hca.Example, "bnxt_re0,bnxt_re1")
	}

	if plain := findField(fields, "plain"); plain == nil || plain.Example != "" {
		t.Errorf("plain field should have no example, got %+v", plain)
	}
}

func TestExampleStringNonString(t *testing.T) {
	cfg := map[string]any{
		"_example_ports": []any{float64(1), float64(2)},
		"ports":          "<changeme>",
	}
	fields := InferSchema(cfg)
	ports := findField(fields, "ports")
	if ports == nil {
		t.Fatal("ports field missing")
	}
	if ports.Example != "[1,2]" {
		t.Errorf("example = %q, want %q", ports.Example, "[1,2]")
	}
}
