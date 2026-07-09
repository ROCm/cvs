package clustermon

import (
	"context"
	"encoding/json"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/ROCm/cvs/api/pkg/pssh"
)

// NIC collector commands. These mirror the cluster-mon Python NIC collector
// verbatim so the parsed output stays at parity. Each runs once per host.
const (
	cmdRDMALink = "rdma link"
	cmdRDMAStat = `bash -c 'rdma statistic show --json 2>/dev/null || echo "{}"'`
	cmdIPAddr   = `bash -c 'ip addr show | grep -A 5 mtu --color=never'`
	cmdIPSLink  = "ip -s link show"
	cmdRDMARes  = "rdma res"
	// cmdLLDP collects LLDP neighbor data via lldpctl (lldpd must be running).
	// Uses sudo to match the cluster-mon reference implementation. Returns "{}"
	// when lldpd is absent so the error doesn't block the rest of NIC collection.
	cmdLLDP = `bash -c 'sudo lldpctl -f json 2>/dev/null || echo "{}"'`
)

// LLDPNeighbor is one LLDP peer seen on a given interface.
type LLDPNeighbor struct {
	Interface   string `json:"interface"`
	Chassis     string `json:"chassis"`
	ChassisID   string `json:"chassis_id,omitempty"`
	Port        string `json:"port"`
	Description string `json:"description,omitempty"`
	MgmtIP      string `json:"mgmt_ip,omitempty"`
}

// NodeLLDPData holds the LLDP neighbors for one node.
type NodeLLDPData struct {
	Host      string         `json:"host"`
	Neighbors []LLDPNeighbor `json:"neighbors,omitempty"`
	Error     string         `json:"error,omitempty"`
}

// CollectLLDP runs the lldpctl command fleet-wide via a single pool.Exec call
// and returns the per-host LLDP neighbor list. A missing lldpd returns an empty
// neighbors list (not an error) so the topology page can show a clean empty-state.
func CollectLLDP(ctx context.Context, pool *pssh.Pool) map[string]NodeLLDPData {
	results := pool.Exec(ctx, cmdLLDP)
	out := make(map[string]NodeLLDPData, len(results))
	for host, r := range results {
		nd := NodeLLDPData{Host: host}
		if r.Err != nil {
			nd.Error = r.Err.Error()
		} else {
			nd.Neighbors = parseLLDP(r.Output)
		}
		out[host] = nd
	}
	return out
}

// RDMALink is one RDMA device link (from `rdma link`).
type RDMALink struct {
	Device        string `json:"device"`
	State         string `json:"state"`
	PhysicalState string `json:"physical_state"`
	Netdev        string `json:"netdev"`
}

// RDMADeviceStats holds the numeric counters for one RDMA device (from
// `rdma statistic show --json`), keyed by counter name.
type RDMADeviceStats struct {
	Device string             `json:"device"`
	Stats  map[string]float64 `json:"stats"`
}

// RDMAResource holds resource counts (pd/cq/qp/...) for one device (`rdma res`).
type RDMAResource struct {
	Device string         `json:"device"`
	Values map[string]int `json:"values"`
}

// NICInterface is one network interface's addressing info (from `ip addr`).
type NICInterface struct {
	Name  string   `json:"name"`
	MTU   string   `json:"mtu,omitempty"`
	State string   `json:"state,omitempty"`
	MAC   string   `json:"mac_addr,omitempty"`
	IPv4  []string `json:"ipv4,omitempty"`
	IPv6  []string `json:"ipv6,omitempty"`
	Flags string   `json:"flags,omitempty"`
}

// EthStats holds per-interface RX/TX counters (from `ip -s link show`).
type EthStats struct {
	Iface string           `json:"iface"`
	Stats map[string]int64 `json:"stats"`
}

// NodeNICMetrics is the per-node NIC collection result.
type NodeNICMetrics struct {
	Host          string            `json:"host"`
	RDMALinks     []RDMALink        `json:"rdma_links,omitempty"`
	RDMAStats     []RDMADeviceStats `json:"rdma_stats,omitempty"`
	RDMAResources []RDMAResource    `json:"rdma_resources,omitempty"`
	Interfaces    []NICInterface    `json:"interfaces,omitempty"`
	EthStats      []EthStats        `json:"eth_stats,omitempty"`
	Error         string            `json:"error,omitempty"`
}

// CollectNIC runs the five NIC commands fleet-wide in parallel and merges the
// results per host. Each command is issued to all reachable nodes concurrently
// via pool.Exec; the five Exec calls themselves run in parallel goroutines.
func CollectNIC(ctx context.Context, pool *pssh.Pool) map[string]NodeNICMetrics {
	var (
		linkRes, statRes, addrRes, slinkRes, resRes map[string]pssh.Result
		wg                                          sync.WaitGroup
	)
	wg.Add(5)
	go func() { defer wg.Done(); linkRes = pool.Exec(ctx, cmdRDMALink) }()
	go func() { defer wg.Done(); statRes = pool.Exec(ctx, cmdRDMAStat) }()
	go func() { defer wg.Done(); addrRes = pool.Exec(ctx, cmdIPAddr) }()
	go func() { defer wg.Done(); slinkRes = pool.Exec(ctx, cmdIPSLink) }()
	go func() { defer wg.Done(); resRes = pool.Exec(ctx, cmdRDMARes) }()
	wg.Wait()

	out := make(map[string]NodeNICMetrics, len(linkRes))
	for host, r := range linkRes {
		m := NodeNICMetrics{Host: host}
		if r.Err != nil {
			m.Error = r.Err.Error()
			out[host] = m
			continue
		}
		m.RDMALinks = parseRDMALinks(r.Output)
		if sr, ok := statRes[host]; ok && sr.Err == nil {
			m.RDMAStats = parseRDMAStats(sr.Output)
		}
		if ar, ok := addrRes[host]; ok && ar.Err == nil {
			m.Interfaces = parseIPAddr(ar.Output)
		}
		if lr, ok := slinkRes[host]; ok && lr.Err == nil {
			m.EthStats = parseEthStats(lr.Output)
		}
		if rr, ok := resRes[host]; ok && rr.Err == nil {
			m.RDMAResources = parseRDMARes(rr.Output)
		}
		out[host] = m
	}
	return out
}

var (
	reRDMALink = regexp.MustCompile(`link\s+([\w_]+/\d+)\s+state\s+(\w+)\s+physical_state\s+(\w+)\s+netdev\s+([\w\-.]+)`)
	reIface    = regexp.MustCompile(`^\d+:\s+([\w.\-_/]+):\s+([<>,A-Z0-9_]+)`)
	reMTU      = regexp.MustCompile(`mtu (\d+)`)
	reState    = regexp.MustCompile(`state ([A-Z]+)`)
	reMAC      = regexp.MustCompile(`link/ether\s+([a-f0-9:]+)`)
	reInet     = regexp.MustCompile(`inet\s+([0-9./]+)`)
	reInet6    = regexp.MustCompile(`inet6\s+([a-f0-9:/]+)`)
	reResLine  = regexp.MustCompile(`(\d+):\s+([\w_]+):\s+(.+)`)
	reResKV    = regexp.MustCompile(`(\w+)\s+(\d+)`)
	reSLinkHdr = regexp.MustCompile(`^\d+:\s+(\S+):`)
)

// parseRDMALinks parses `rdma link` text lines like:
//
//	link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens1
func parseRDMALinks(out string) []RDMALink {
	var links []RDMALink
	for _, line := range strings.Split(out, "\n") {
		m := reRDMALink.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		links = append(links, RDMALink{
			Device:        m[1],
			State:         m[2],
			PhysicalState: m[3],
			Netdev:        m[4],
		})
	}
	sort.Slice(links, func(i, j int) bool { return links[i].Device < links[j].Device })
	return links
}

// rdmaProjection defines the RDMA counters that are always included in the
// snapshot regardless of value. Zero is meaningful here — it confirms the
// counter exists and the link is clean. Add an entry to expose a new counter
// to the UI without changing any other code.
var rdmaProjection = map[string]bool{
	// Traffic
	"rx_pkts":  true,
	"tx_pkts":  true,
	"rx_bytes": true,
	"tx_bytes": true,
	// RoCE errors
	"tx_roce_errors":   true,
	"rx_roce_errors":   true,
	"tx_roce_discards": true,
	"rx_roce_discards": true,
	"recoverable_errors": true,
	// Error detail (bnxt_re)
	"req_cqe_error":              true,
	"resp_cqe_error":             true,
	"req_remote_access_errors":   true,
	"resp_local_access_errors":   true,
	// Resources / QP state
	"active_qps":    true,
	"watermark_qps": true,
	// Retry
	"rnr_retry_count": true,
	"retry_count":     true,
}

// parseRDMAStats parses `rdma statistic show --json`, a list of objects each
// with an ifname/port plus numeric counters.
//
// Projection rules (see rdmaProjection):
//   - Required fields: always included, even if zero (zero confirms the link is clean).
//   - Non-required fields: included only if non-zero (anomaly detection).
//   - Metadata keys (ifname/port/ifindex): always dropped.
func parseRDMAStats(out string) []RDMADeviceStats {
	js := extractJSON(out)
	if js == "" {
		return nil
	}
	var list []map[string]any
	if err := json.Unmarshal([]byte(js), &list); err != nil {
		return nil
	}
	meta := map[string]bool{"ifname": true, "port": true, "ifindex": true}
	var stats []RDMADeviceStats
	for _, entry := range list {
		ifname, _ := entry["ifname"].(string)
		if ifname == "" {
			continue
		}
		dev := ifname
		if p, ok := entry["port"]; ok {
			dev = ifname + "/" + strings.TrimSuffix(strconv.FormatFloat(toFloat(p), 'f', -1, 64), ".0")
		}
		counters := make(map[string]float64, len(rdmaProjection))
		// Required fields: always present (zero is meaningful data).
		for k := range rdmaProjection {
			if v, ok := entry[k]; ok {
				counters[k] = toFloat(v)
			} else {
				counters[k] = 0
			}
		}
		// Non-required fields: only if non-zero (unexpected activity / anomaly).
		for k, v := range entry {
			if meta[k] || rdmaProjection[k] {
				continue
			}
			switch v.(type) {
			case float64, json.Number:
				if f := toFloat(v); f != 0 {
					counters[k] = f
				}
			}
		}
		stats = append(stats, RDMADeviceStats{Device: dev, Stats: counters})
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].Device < stats[j].Device })
	return stats
}

// parseIPAddr parses the `ip addr show | grep -A 5 mtu` output into per-interface
// addressing info.
func parseIPAddr(out string) []NICInterface {
	byName := map[string]*NICInterface{}
	var order []string
	var cur *NICInterface
	for _, line := range strings.Split(out, "\n") {
		if m := reIface.FindStringSubmatch(line); m != nil {
			name := m[1]
			ni := &NICInterface{Name: name, Flags: m[2], IPv4: []string{}, IPv6: []string{}}
			byName[name] = ni
			order = append(order, name)
			cur = ni
		}
		if cur == nil {
			continue
		}
		if m := reMTU.FindStringSubmatch(line); m != nil {
			cur.MTU = m[1]
		}
		if m := reState.FindStringSubmatch(line); m != nil {
			cur.State = m[1]
		}
		if m := reMAC.FindStringSubmatch(line); m != nil {
			cur.MAC = m[1]
		}
		if m := reInet.FindStringSubmatch(line); m != nil {
			cur.IPv4 = append(cur.IPv4, m[1])
		}
		if m := reInet6.FindStringSubmatch(line); m != nil {
			cur.IPv6 = append(cur.IPv6, m[1])
		}
	}
	ifaces := make([]NICInterface, 0, len(order))
	for _, n := range order {
		ifaces = append(ifaces, *byName[n])
	}
	return ifaces
}

// ethStatField pairs the column order of the RX/TX lines in `ip -s link`.
var (
	rxFields = []string{"rx_bytes", "rx_packets", "rx_errors", "rx_dropped", "rx_missed", "rx_mcast"}
	txFields = []string{"tx_bytes", "tx_packets", "tx_errors", "tx_dropped", "tx_carrier", "tx_collsns"}
)

// parseEthStats parses `ip -s link show` RX/TX counter blocks per interface,
// skipping loopback and veth devices (parity with the Python collector).
func parseEthStats(out string) []EthStats {
	var stats []EthStats
	lines := strings.Split(out, "\n")
	var cur *EthStats
	rxNext, txNext := false, false
	for _, line := range lines {
		if m := reSLinkHdr.FindStringSubmatch(line); m != nil {
			name := strings.Split(m[1], "@")[0]
			if name == "lo" || strings.HasPrefix(name, "veth") {
				cur = nil
				rxNext, txNext = false, false
				continue
			}
			stats = append(stats, EthStats{Iface: name, Stats: map[string]int64{}})
			cur = &stats[len(stats)-1]
			rxNext, txNext = false, false
			continue
		}
		if cur == nil {
			continue
		}
		if strings.Contains(line, "RX:") && strings.Contains(line, "bytes") {
			rxNext = true
			continue
		}
		if strings.Contains(line, "TX:") && strings.Contains(line, "bytes") {
			txNext = true
			continue
		}
		if rxNext {
			assignStats(cur, line, rxFields)
			rxNext = false
			continue
		}
		if txNext {
			assignStats(cur, line, txFields)
			txNext = false
			continue
		}
	}
	return stats
}

// parseLLDP parses `sudo lldpctl -f json` output (lldpd's JSON format).
//
// The lldpctl JSON structure for one host is:
//
//	{
//	  "lldp": {
//	    "interface": [
//	      { "enp28s0np0": {
//	          "chassis": { "switch-hostname": { "id": {"type":"mac","value":"..."}, "descr":"...", "mgmt-ip":"..." } },
//	          "port":    { "id": {"type":"ifname","value":"Eth1/1"}, "descr":"..." }
//	      }}
//	    ]
//	  }
//	}
//
// The interface field is an array of single-key objects (key = interface name).
// The chassis field is an object keyed by the neighbor hostname.
func parseLLDP(raw string) []LLDPNeighbor {
	js := extractJSON(raw)
	if js == "" {
		return nil
	}
	var top map[string]any
	if err := json.Unmarshal([]byte(js), &top); err != nil {
		return nil
	}
	lldpData, ok := top["lldp"].(map[string]any)
	if !ok {
		return nil
	}
	// interface: [{ifname: {chassis:{...}, port:{...}}}]
	ifaceArr, ok := lldpData["interface"].([]any)
	if !ok || len(ifaceArr) == 0 {
		return nil
	}

	var neighbors []LLDPNeighbor
	for _, entry := range ifaceArr {
		entryMap, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		// Each entry is a single-key map: {ifname -> ifdata}
		for ifname, ifdata := range entryMap {
			ifMap, ok := ifdata.(map[string]any)
			if !ok {
				continue
			}

			// chassis: {"hostname": {"id":{"type":"mac","value":"..."},"descr":"...","mgmt-ip":"..."}}
			chassisName, chassisID, mgmtIP, descr := "", "", "", ""
			if chassis, ok := ifMap["chassis"].(map[string]any); ok {
				for name, cdata := range chassis {
					chassisName = name
					if cd, ok := cdata.(map[string]any); ok {
						if id, ok := cd["id"].(map[string]any); ok {
							chassisID, _ = id["value"].(string)
						}
						descr, _ = cd["descr"].(string)
						// mgmt-ip can be a string or array
						switch v := cd["mgmt-ip"].(type) {
						case string:
							mgmtIP = v
						case []any:
							if len(v) > 0 {
								mgmtIP, _ = v[0].(string)
							}
						}
					}
					break // one chassis per interface
				}
			}

			// port: {"id":{"type":"ifname","value":"Eth1/1"}, "descr":"..."}
			portName := ""
			if port, ok := ifMap["port"].(map[string]any); ok {
				portName, _ = port["descr"].(string)
				if portName == "" {
					if id, ok := port["id"].(map[string]any); ok {
						portName, _ = id["value"].(string)
					}
				}
			}

			if chassisName == "" && ifname == "" {
				continue
			}
			neighbors = append(neighbors, LLDPNeighbor{
				Interface:   ifname,
				Chassis:     chassisName,
				ChassisID:   chassisID,
				Port:        portName,
				Description: descr,
				MgmtIP:      mgmtIP,
			})
		}
	}
	return neighbors
}

func assignStats(es *EthStats, line string, fields []string) {
	parts := strings.Fields(strings.TrimSpace(line))
	if len(parts) < len(fields) {
		return
	}
	for i, f := range fields {
		n, err := strconv.ParseInt(parts[i], 10, 64)
		if err != nil {
			return
		}
		es.Stats[f] = n
	}
}

// parseRDMARes parses `rdma res` lines like:
//
//	0: bnxt_re0: pd 1 cq 1 qp 1 cm_id 0 mr 0 ctx 0 srq 0
func parseRDMARes(out string) []RDMAResource {
	var res []RDMAResource
	for _, line := range strings.Split(out, "\n") {
		m := reResLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		values := map[string]int{}
		for _, kv := range reResKV.FindAllStringSubmatch(m[3], -1) {
			n, _ := strconv.Atoi(kv[2])
			values[kv[1]] = n
		}
		res = append(res, RDMAResource{Device: m[2], Values: values})
	}
	sort.Slice(res, func(i, j int) bool { return res[i].Device < res[j].Device })
	return res
}
