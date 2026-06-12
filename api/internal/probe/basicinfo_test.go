package probe

import "testing"

func TestParseROCmVersion(t *testing.T) {
	cases := map[string]struct {
		in   string
		want string
	}{
		"list":           {`[{"tool":"AMDSMI","version":"26.2.0","rocm_version":"7.0.2"}]`, "7.0.2"},
		"object":         {`{"rocm_version":"6.4.1"}`, "6.4.1"},
		"warning prefix": {"WARNING: stuff\n[{\"rocm_version\":\"6.0.0\"}]", "6.0.0"},
		"na":             {`[{"rocm_version":"N/A"}]`, ""},
		"garbage":        {"command not found", ""},
		"empty":          {"", ""},
	}
	for name, c := range cases {
		if got := parseROCmVersion(c.in); got != c.want {
			t.Errorf("%s: parseROCmVersion = %q, want %q", name, got, c.want)
		}
	}
}

func TestParseProductName(t *testing.T) {
	in := `{"card0":{"Card Series":"AMD Instinct MI300X","Card Model":"0x74a1"},` +
		`"card1":{"Card Series":"AMD Instinct MI300X"},` +
		`"card2":{"Card Series":"AMD Instinct MI300X"}}`
	gpuType, count := parseProductName(in)
	if count != 3 {
		t.Fatalf("count = %d, want 3", count)
	}
	if gpuType != "AMD Instinct MI300X" {
		t.Fatalf("type = %q", gpuType)
	}
}

func TestParseProductName_ModelFallback(t *testing.T) {
	// No "Card Series"; should fall back to "Card Model".
	in := `{"card0":{"Card Model":"MI250"}}`
	gpuType, count := parseProductName(in)
	if count != 1 || gpuType != "MI250" {
		t.Fatalf("got type=%q count=%d", gpuType, count)
	}
}

func TestParseStaticCount(t *testing.T) {
	if c := parseStaticCount(`[{"gpu":0},{"gpu":1}]`); c != 2 {
		t.Errorf("list count = %d, want 2", c)
	}
	if c := parseStaticCount(`{"gpu_data":[{"gpu":0},{"gpu":1},{"gpu":2}]}`); c != 3 {
		t.Errorf("wrapped count = %d, want 3", c)
	}
	if c := parseStaticCount("nope"); c != 0 {
		t.Errorf("garbage count = %d, want 0", c)
	}
}

func TestExtractJSON(t *testing.T) {
	if got := extractJSON("WARNING: x\n[1,2]"); got != "[1,2]" {
		t.Errorf("array extract = %q", got)
	}
	if got := extractJSON("noise {\"a\":1}"); got != `{"a":1}` {
		t.Errorf("object extract = %q", got)
	}
	if got := extractJSON("no json here"); got != "" {
		t.Errorf("expected empty, got %q", got)
	}
}
