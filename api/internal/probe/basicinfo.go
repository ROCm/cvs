package probe

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"sync"
)

// Commands run over SSH to collect basic node info. These mirror the cluster-mon
// collectors: amd-smi is the primary source, with rocm-smi / sysfs fallbacks.
const (
	cmdROCmVersion     = "amd-smi version --json"
	cmdROCmVersionFall = "bash -c 'cat /opt/rocm*/.info/version 2>/dev/null | head -1'"
	cmdProductName     = "rocm-smi --loglevel error --showproductname --json"
	cmdStaticFall      = "amd-smi static --json"
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

// Collect gathers basic info for every host concurrently using r.
func Collect(ctx context.Context, r Runner, hosts []string) map[string]Info {
	out := make(map[string]Info, len(hosts))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, h := range hosts {
		wg.Add(1)
		go func(host string) {
			defer wg.Done()
			info := collectHost(ctx, r, host)
			mu.Lock()
			out[host] = info
			mu.Unlock()
		}(h)
	}
	wg.Wait()
	return out
}

func collectHost(ctx context.Context, r Runner, host string) Info {
	var info Info

	// ROCm version: amd-smi version --json -> [0].rocm_version, else sysfs.
	if v, err := r.Run(ctx, host, cmdROCmVersion); err == nil {
		info.ROCmVersion = parseROCmVersion(v)
	}
	if info.ROCmVersion == "" {
		if v, err := r.Run(ctx, host, cmdROCmVersionFall); err == nil {
			if s := strings.TrimSpace(v); s != "" && !looksLikeError(s) {
				info.ROCmVersion = s
			}
		}
	}

	// GPU type + count: rocm-smi --showproductname --json, else amd-smi static.
	if v, err := r.Run(ctx, host, cmdProductName); err == nil {
		gpuType, count := parseProductName(v)
		info.GPUType, info.GPUCount = gpuType, count
	}
	if info.GPUCount == 0 {
		if v, err := r.Run(ctx, host, cmdStaticFall); err == nil {
			if c := parseStaticCount(v); c > 0 {
				info.GPUCount = c
			}
		}
	}

	return info
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

func looksLikeError(s string) bool {
	u := strings.ToUpper(strings.TrimSpace(s))
	return strings.HasPrefix(u, "ERROR") || strings.HasPrefix(u, "ABORT") || strings.HasPrefix(u, "N/A")
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

// parseStaticCount counts GPUs in `amd-smi static --json`, which may be a list
// of GPU objects or an object wrapping {"gpu_data":[...]}.
func parseStaticCount(raw string) int {
	js := extractJSON(raw)
	if js == "" {
		return 0
	}
	var list []any
	if err := json.Unmarshal([]byte(js), &list); err == nil {
		return len(list)
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(js), &obj); err == nil {
		if gd, ok := obj["gpu_data"].([]any); ok {
			return len(gd)
		}
	}
	return 0
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
