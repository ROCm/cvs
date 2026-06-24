package probe

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"sync"

	"github.com/ROCm/cvs/api/pkg/pssh"
)

// Commands run over SSH to collect basic node info.
const (
	cmdROCmVersion = "amd-smi version --json"
	cmdProductName = "rocm-smi --loglevel error --showproductname --json"
)

// Info is the basic-info result for a single node.
type Info struct {
	GPUType     string
	GPUCount    int
	ROCmVersion string
	Err         string
}

// cardKeyRe matches rocm-smi per-GPU JSON keys like "card0", "card12".
var cardKeyRe = regexp.MustCompile(`^card\d+$`)

// Collect gathers basic info fleet-wide using two parallel pool.Exec calls.
func Collect(ctx context.Context, pool *pssh.Pool) map[string]Info {
	var versionRes, productRes map[string]pssh.Result
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); versionRes = pool.Exec(ctx, cmdROCmVersion) }()
	go func() { defer wg.Done(); productRes = pool.Exec(ctx, cmdProductName) }()
	wg.Wait()

	// Merge into Info per host; use versionRes as the host set (both commands
	// target the same reachable nodes, but versionRes is the primary source).
	out := make(map[string]Info, len(versionRes))
	for host, r := range versionRes {
		var info Info
		if r.Err == nil {
			info.ROCmVersion = parseROCmVersion(r.Output)
		}
		if pr, ok := productRes[host]; ok && pr.Err == nil {
			info.GPUType, info.GPUCount = parseProductName(pr.Output)
		}
		out[host] = info
	}
	// Also include hosts that appeared only in productRes (unlikely but safe).
	for host, pr := range productRes {
		if _, seen := out[host]; seen {
			continue
		}
		if pr.Err == nil {
			var info Info
			info.GPUType, info.GPUCount = parseProductName(pr.Output)
			out[host] = info
		}
	}
	return out
}

// extractJSON trims any leading non-JSON noise (e.g. "WARNING:" banners) so the
// payload can be unmarshalled, matching the Python preprocessing.
func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	obj := strings.IndexByte(s, '{')
	arr := strings.IndexByte(s, '[')
	start := -1
	switch {
	case obj < 0:
		start = arr
	case arr < 0:
		start = obj
	default:
		if obj < arr {
			start = obj
		} else {
			start = arr
		}
	}
	if start < 0 {
		return ""
	}
	return s[start:]
}

// parseROCmVersion reads rocm_version from `amd-smi version --json`, whose
// payload is a list of objects: [{"rocm_version":"7.0.2", ...}].
func parseROCmVersion(raw string) string {
	js := extractJSON(raw)
	if js == "" {
		return ""
	}
	var list []map[string]any
	if err := json.Unmarshal([]byte(js), &list); err == nil {
		for _, obj := range list {
			if v := stringField(obj, "rocm_version"); v != "" && v != "N/A" {
				return v
			}
		}
		return ""
	}
	// Some amd-smi builds emit a single object rather than a list.
	var obj map[string]any
	if err := json.Unmarshal([]byte(js), &obj); err == nil {
		if v := stringField(obj, "rocm_version"); v != "" && v != "N/A" {
			return v
		}
	}
	return ""
}

// parseProductName reads GPU model + count from `rocm-smi --showproductname
// --json`, an object keyed by card: {"card0":{"Card Series":"...", ...}, ...}.
func parseProductName(raw string) (gpuType string, count int) {
	js := extractJSON(raw)
	if js == "" {
		return "", 0
	}
	var obj map[string]map[string]any
	if err := json.Unmarshal([]byte(js), &obj); err != nil {
		return "", 0
	}
	// Stable order: card0, card1, ... for a deterministic first model.
	for i := 0; ; i++ {
		key := "card" + itoa(i)
		card, ok := obj[key]
		if !ok {
			break
		}
		count++
		if gpuType == "" {
			gpuType = firstNonEmpty(card,
				"Card Series", "Card Model", "Device Name", "Card SKU")
		}
	}
	// Fall back to counting any card-like keys if they were non-contiguous.
	if count == 0 {
		for k, card := range obj {
			if cardKeyRe.MatchString(k) {
				count++
				if gpuType == "" {
					gpuType = firstNonEmpty(card, "Card Series", "Card Model", "Device Name", "Card SKU")
				}
			}
		}
	}
	return gpuType, count
}

func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

func firstNonEmpty(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if v := stringField(m, k); v != "" && v != "N/A" {
			return v
		}
	}
	return ""
}

// itoa avoids a strconv import for small non-negative ints.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
