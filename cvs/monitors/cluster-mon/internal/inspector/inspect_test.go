package inspector

import "testing"

func TestMemoryPairBytes(t *testing.T) {
	used, total := memoryPair(map[string]any{
		"GPU Memory Used (B)":  "100",
		"GPU Memory Total (B)": "1000",
	})
	if used != 100 || total != 1000 {
		t.Fatalf("used=%d total=%d", used, total)
	}
}

func TestMemoryPairZeroUsedIsNotPercent(t *testing.T) {
	used, total := memoryPair(map[string]any{
		"GPU Memory Used (B)":          "0",
		"GPU Memory Total (B)":         "1000",
		"GPU Memory Allocated (VRAM%)": "75%",
	})
	if used != 0 || total != 1000 {
		t.Fatalf("used=%d total=%d", used, total)
	}
}

func TestMemoryPairPercentWithTotal(t *testing.T) {
	used, total := memoryPair(map[string]any{
		"GPU Memory Total (B)":         "1000",
		"GPU Memory Allocated (VRAM%)": "75%",
	})
	if used != 750 || total != 1000 {
		t.Fatalf("used=%d total=%d", used, total)
	}
}

func TestMemoryPairPercentWithoutTotal(t *testing.T) {
	used, total := memoryPair(map[string]any{
		"GPU Memory Allocated (VRAM%)": "75%",
	})
	if used != 0 || total != 0 {
		t.Fatalf("percent became bytes: used=%d total=%d", used, total)
	}
}
