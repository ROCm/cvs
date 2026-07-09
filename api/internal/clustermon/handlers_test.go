package clustermon

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/ROCm/cvs/api/internal/inventory"
)

type fakeStore struct {
	inv inventory.Inventory
	ok  bool
}

func (f fakeStore) Get() (inventory.Inventory, bool, error) { return f.inv, f.ok, nil }
func (f fakeStore) Save(i inventory.Inventory) (inventory.Inventory, error) {
	return i, nil
}
func (f fakeStore) Clear() error { return nil }

func serve(t *testing.T, s inventory.Store) *httptest.Server {
	t.Helper()
	r := chi.NewRouter()
	NewHandler(s, nil, nil, nil, nil, nil).Routes(r)
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv
}

func TestNodes_MergesProbeStatus(t *testing.T) {
	checked := time.Now().UTC()
	store := fakeStore{ok: true, inv: inventory.Inventory{
		Nodes: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		Statuses: []inventory.NodeStatus{
			{Host: "10.0.0.1", Reachable: true, GPUType: "MI300X", GPUCount: 8, ROCmVersion: "6.2", CheckedAt: checked},
			{Host: "10.0.0.2", Reachable: false, Error: "dial timeout", CheckedAt: checked},
			// 10.0.0.3 was never probed.
		},
	}}
	srv := serve(t, store)

	resp, err := http.Get(srv.URL + "/clustermon/nodes")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var body struct {
		Nodes      []NodeView `json:"nodes"`
		Total      int        `json:"total"`
		Configured bool       `json:"configured"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if !body.Configured || body.Total != 3 || len(body.Nodes) != 3 {
		t.Fatalf("unexpected envelope: %+v", body)
	}
	// Order must follow inventory order.
	if body.Nodes[0].Host != "10.0.0.1" || !body.Nodes[0].Probed || !body.Nodes[0].Reachable ||
		body.Nodes[0].GPUType != "MI300X" || body.Nodes[0].GPUCount != 8 {
		t.Fatalf("node 0 wrong: %+v", body.Nodes[0])
	}
	if body.Nodes[1].Reachable || body.Nodes[1].Error != "dial timeout" || !body.Nodes[1].Probed {
		t.Fatalf("node 1 wrong: %+v", body.Nodes[1])
	}
	if body.Nodes[2].Probed || body.Nodes[2].Reachable {
		t.Fatalf("node 2 should be unprobed: %+v", body.Nodes[2])
	}
}

func TestNodes_NoInventory(t *testing.T) {
	srv := serve(t, fakeStore{ok: false})
	resp, err := http.Get(srv.URL + "/clustermon/nodes")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body struct {
		Total      int  `json:"total"`
		Configured bool `json:"configured"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Configured || body.Total != 0 {
		t.Fatalf("expected empty/unconfigured, got %+v", body)
	}
}
