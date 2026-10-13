package collectors

import (
	"fmt"
	"strings"
	"time"
)

// PythonGPUPayload is the GPU object the existing React pages consume under
// latestMetrics.gpu (utilization / memory / temperature / power / pcie / xgmi / ras_errors).
func PythonGPUPayload(byHost map[string]NodeGPUMetrics) map[string]any {
	util := map[string]any{}
	mem := map[string]any{}
	temp := map[string]any{}
	power := map[string]any{}
	pcie := map[string]any{}
	pcieLink := map[string]any{}
	xgmi := map[string]any{}
	ras := map[string]any{}

	for host, n := range byHost {
		if n.Error != "" {
			errObj := map[string]any{"error": n.Error}
			util[host] = errObj
			mem[host] = errObj
			temp[host] = errObj
			continue
		}
		u, m, t, pwr, pc, xg := map[string]any{}, map[string]any{}, map[string]any{}, map[string]any{}, map[string]any{}, map[string]any{}
		var pcieGPU, rasGPU, xgmiGPU, pwrGPU []any
		for _, g := range n.GPUs {
			id := fmt.Sprintf("card%d", g.Index)
			u[id] = map[string]any{
				"GPU use (%)":  fmt.Sprintf("%v", g.UtilizationPct),
				"GFX Activity": fmt.Sprintf("%v", g.UtilizationPct),
			}
			m[id] = map[string]any{
				"VRAM Total Memory (B)":      int64(g.MemTotalMB * 1024 * 1024),
				"VRAM Total Used Memory (B)": int64(g.MemUsedMB * 1024 * 1024),
			}
			t[id] = map[string]any{
				"Temperature (Sensor edge) (C)":     fmt.Sprintf("%v", g.TempEdgeC),
				"Temperature (Sensor junction) (C)": fmt.Sprintf("%v", g.TempHotspotC),
				"Temperature (Sensor memory) (C)":   fmt.Sprintf("%v", g.TempMemC),
			}
			pwrFields := map[string]any{"socket_power": fmt.Sprintf("%v W", g.PowerW)}
			pwr[id] = pwrFields
			pwrGPU = append(pwrGPU, map[string]any{"gpu": g.Index, "power": pwrFields})
			if g.PCIe != nil {
				fields := map[string]any{
					"width":                g.PCIe.Width,
					"speed":                g.PCIe.Speed,
					"bandwidth":            g.PCIe.Bandwidth,
					"replay_count":         g.PCIe.ReplayCount,
					"l0_to_recovery_count": g.PCIe.L0ToRecoveryCount,
					"nak_sent_count":       g.PCIe.NakSentCount,
					"nak_received_count":   g.PCIe.NakReceivedCount,
					"nak_count":            g.PCIe.NakSentCount + g.PCIe.NakReceivedCount,
				}
				pc[id] = fields
				pcieGPU = append(pcieGPU, map[string]any{"gpu": g.Index, "pcie": fields})
			}
			if g.XGMI != nil {
				xgFields := map[string]any{"status": g.XGMI.Status, "error_count": g.XGMI.ErrorCount}
				xg[id] = xgFields
				xgmiGPU = append(xgmiGPU, map[string]any{"gpu": g.Index, "xgmi": xgFields, "xgmi_err": xgFields})
			}
			if g.ECC != nil {
				rasGPU = append(rasGPU, map[string]any{
					"gpu": g.Index,
					"ecc": map[string]any{
						"correctable":               g.ECC.Correctable,
						"uncorrectable":             g.ECC.Uncorrectable,
						"deferred":                  g.ECC.Deferred,
						"total_correctable":         g.ECC.Correctable,
						"total_uncorrectable":       g.ECC.Uncorrectable,
						"total_correctable_count":   g.ECC.Correctable,
						"total_uncorrectable_count": g.ECC.Uncorrectable,
					},
				})
			}
		}
		util[host], mem[host], temp[host] = u, m, t
		power[host] = withGPUData(pwr, pwrGPU)
		pcieLink[host] = pc
		pcieErr := map[string]any{}
		for k, v := range pc {
			pcieErr[k] = v
		}
		if len(pcieGPU) > 0 {
			pcieErr["gpu_data"] = pcieGPU
		}
		pcie[host] = pcieErr
		xgmi[host] = withGPUData(xg, xgmiGPU)
		ras[host] = map[string]any{"gpu_data": rasGPU}
	}

	return map[string]any{
		"utilization":      util,
		"memory":           mem,
		"temperature":      temp,
		"power":            power,
		"pcie":             pcie,
		"pcie_link_status": pcieLink,
		"xgmi":             xgmi,
		"ras_errors":       ras,
		"info":             power,
	}
}

func withGPUData(cards map[string]any, rows []any) map[string]any {
	out := map[string]any{}
	for k, v := range cards {
		out[k] = v
	}
	if len(rows) > 0 {
		out["gpu_data"] = rows
	}
	return out
}

// PythonNICPayload is latestMetrics.nic for the existing React pages.
func PythonNICPayload(byHost map[string]NodeNICMetrics, lldpRaw map[string]any) map[string]any {
	links := map[string]any{}
	stats := map[string]any{}
	res := map[string]any{}
	ip := map[string]any{}
	eth := map[string]any{}

	rdmaNetdevs := map[string]map[string]struct{}{}

	for host, n := range byHost {
		if n.Error != "" {
			errObj := map[string]any{"error": n.Error}
			links[host] = errObj
			continue
		}
		lm := map[string]any{}
		nd := map[string]struct{}{}
		for _, l := range n.RDMALinks {
			lm[l.Device] = map[string]any{
				"state":          l.State,
				"physical_state": l.PhysicalState,
				"netdev":         l.Netdev,
			}
			if l.Netdev != "" {
				nd[l.Netdev] = struct{}{}
			}
		}
		links[host] = lm
		rdmaNetdevs[host] = nd

		sm := map[string]any{}
		for _, s := range n.RDMAStats {
			sm[s.Device] = s.Stats
		}
		stats[host] = sm

		rm := map[string]any{}
		for _, r := range n.RDMAResources {
			rm[r.Device] = r.Values
		}
		res[host] = rm

		im := map[string]any{}
		for _, iface := range n.Interfaces {
			im[iface.Name] = map[string]any{
				"mtu":            iface.MTU,
				"state":          iface.State,
				"mac_addr":       iface.MAC,
				"ipv4":           iface.IPv4,
				"ipv4_addr_list": iface.IPv4,
				"ipv6":           iface.IPv6,
				"flags":          iface.Flags,
			}
		}
		ip[host] = im

		em := map[string]any{}
		for _, e := range n.EthStats {
			em[e.Iface] = e.Stats
		}
		eth[host] = em
	}

	return map[string]any{
		"rdma_links":     links,
		"rdma_stats":     stats,
		"rdma_resources": res,
		"ip_addr":        ip,
		"ethtool_stats":  eth,
		"lldp":           filterLLDP(lldpRaw, rdmaNetdevs),
	}
}

func filterLLDP(raw map[string]any, rdmaNetdevs map[string]map[string]struct{}) map[string]any {
	out := map[string]any{}
	if raw == nil {
		return out
	}
	for host, doc := range raw {
		allow := rdmaNetdevs[host]
		out[host] = filterLLDPDoc(doc, allow)
	}
	return out
}

func filterLLDPDoc(doc any, allow map[string]struct{}) any {
	if len(allow) == 0 {
		return map[string]any{}
	}
	m, ok := doc.(map[string]any)
	if !ok {
		return map[string]any{}
	}
	lldp, _ := m["lldp"].(map[string]any)
	if lldp == nil {
		lldp = m
	}
	iface := lldp["interface"]
	switch arr := iface.(type) {
	case []any:
		var kept []any
		for _, entry := range arr {
			em, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			filt := map[string]any{}
			for ifname, ifdata := range em {
				if _, ok := allow[ifname]; ok {
					filt[ifname] = ifdata
				}
			}
			if len(filt) > 0 {
				kept = append(kept, filt)
			}
		}
		if len(kept) == 0 {
			return map[string]any{}
		}
		return map[string]any{"lldp": map[string]any{"interface": kept}}
	case map[string]any:
		filt := map[string]any{}
		for ifname, ifdata := range arr {
			if _, ok := allow[ifname]; ok {
				filt[ifname] = ifdata
			}
		}
		if len(filt) == 0 {
			return map[string]any{}
		}
		return map[string]any{"lldp": map[string]any{"interface": filt}}
	default:
		return map[string]any{}
	}
}

// PythonLogsPayload matches GET /api/logs/dmesg.
func PythonLogsPayload(snap *LogsSnapshot) map[string]any {
	amd, dmesg, user := map[string]any{}, map[string]any{}, map[string]any{}
	ts := time.Now().UTC().Format(time.RFC3339)
	if snap == nil {
		return map[string]any{"timestamp": ts, "amd_logs": amd, "dmesg_errors": dmesg, "userspace_errors": user}
	}
	ts = snap.CollectedAt.Format(time.RFC3339)
	for _, n := range snap.Nodes {
		if n.Error != "" {
			amd[n.Host] = map[string]any{"error": n.Error}
			dmesg[n.Host] = map[string]any{"error": n.Error}
			user[n.Host] = map[string]any{"error": n.Error}
			continue
		}
		amd[n.Host] = n.AMDLogs
		dmesg[n.Host] = n.DmesgErrors
		user[n.Host] = n.UserspaceErrors
	}
	return map[string]any{
		"timestamp":        ts,
		"amd_logs":         amd,
		"dmesg_errors":     dmesg,
		"userspace_errors": user,
	}
}

// PythonSearchPayload matches GET /api/logs/search.
func PythonSearchPayload(res *SearchResponse) map[string]any {
	results := map[string]any{}
	if res == nil {
		return map[string]any{
			"timestamp":            time.Now().UTC().Format(time.RFC3339),
			"grep_command":         "",
			"results":              results,
			"total_nodes_searched": 0,
			"nodes_with_results":   0,
		}
	}
	for _, r := range res.Results {
		if strings.TrimSpace(r.Output) != "" {
			results[r.Host] = r.Output
		}
	}
	return map[string]any{
		"timestamp":            time.Now().UTC().Format(time.RFC3339),
		"grep_command":         res.GrepCommand,
		"results":              results,
		"total_nodes_searched": res.TotalNodes,
		"nodes_with_results":   res.NodesWithResults,
	}
}

// PythonGPUSoftwarePayload matches GET /api/software/gpu.
func PythonGPUSoftwarePayload(snap *GPUSoftwareSnapshot) map[string]any {
	rocm := map[string]any{}
	fw := map[string]any{}
	ts := time.Now().UTC().Format(time.RFC3339)
	if snap != nil {
		ts = snap.CollectedAt.Format(time.RFC3339)
		for _, n := range snap.Nodes {
			if n.Version != nil {
				rocm[n.Host] = n.Version
			} else {
				rocm[n.Host] = map[string]any{"rocm_version": "N/A", "amdgpu_version": "N/A"}
			}
			if n.Error != "" {
				fw[n.Host] = map[string]any{"error": n.Error}
				continue
			}
			rows := make([]map[string]any, 0, len(n.Firmware))
			for _, g := range n.Firmware {
				list := make([]map[string]any, 0, len(g.FWList))
				for _, e := range g.FWList {
					list = append(list, map[string]any{"fw_id": e.ID, "fw_version": e.Version})
				}
				rows = append(rows, map[string]any{"gpu": g.GPU, "fw_list": list})
			}
			fw[n.Host] = rows
		}
	}
	return map[string]any{"timestamp": ts, "rocm_version": rocm, "gpu_firmware": fw}
}

// PythonDevlinkPayload matches GET /api/software/nic/devlink.
func PythonDevlinkPayload(snap *DevlinkSnapshot) map[string]any {
	devlink := map[string]any{}
	ts := time.Now().UTC().Format(time.RFC3339)
	if snap != nil {
		ts = snap.CollectedAt.Format(time.RFC3339)
		for _, n := range snap.Nodes {
			if n.Error != "" {
				devlink[n.Host] = map[string]any{"error": n.Error}
				continue
			}
			devs := map[string]any{}
			for _, d := range n.Devices {
				key := "pci/" + d.PCIAddress
				devs[key] = d
			}
			devlink[n.Host] = devs
		}
	}
	return map[string]any{"timestamp": ts, "devlink": devlink}
}
