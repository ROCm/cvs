package inspector

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseInspectorSample(t *testing.T) {
	p := filepath.Join("..", "..", "testdata", "inspector_sample.jsonl")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	recs := ParseLines(string(b), 0)
	if len(recs) < 4 {
		t.Fatalf("got %d records", len(recs))
	}
	snap := Aggregate(recs)
	if snap.AvgBusBWGbps == nil || *snap.AvgBusBWGbps <= 0 {
		t.Fatalf("avg bw missing: %+v", snap)
	}
	if snap.CollectiveBreakdown["AllReduce"] == 0 {
		t.Fatalf("breakdown=%v", snap.CollectiveBreakdown)
	}
}

func TestParseSkipsMalformed(t *testing.T) {
	recs := ParseLines("not json\n{\"header\":{}}\n", 0)
	if len(recs) != 0 {
		t.Fatalf("got %d", len(recs))
	}
}
