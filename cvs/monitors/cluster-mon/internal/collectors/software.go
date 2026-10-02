package collectors

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/ROCm/cvs/monitors/cluster-mon/internal/pssh"
)

// Software/on-demand collector tuning. These collectors are slower and change
// rarely, so results are cached for softwareTTL (parity with the cluster-mon
// Python `software_cache_ttl`) with a stale-on-error fallback.
const (
	// softwareTTL matches cluster-mon: cached responses are reused for 3 minutes.
	softwareTTL = 180 * time.Second
	// softwareCollectTimeout must cover a cold-cache fleet sweep: ceil(N/maxParallel)
	// waves × (SSH + devlink). With timeout 20 on devlink, large fleets need >90s.
	softwareCollectTimeout = 150 * time.Second
)

// Software collector commands (verbatim parity with cluster-mon Python).
const (
	cmdAMDSMIVersion  = "amd-smi version --json"
	cmdAMDSMIFirmware = "amd-smi firmware --json"
	// timeout 20 prevents a stalled NIC driver from blocking the entire sweep.
	cmdDevlinkInfo = `bash -c 'timeout 20 devlink dev info --json 2>/dev/null || echo "{}"'`
)

// GPUSoftwareInfo is the per-node ROCm/AMD-SMI/driver versions from
// `amd-smi version --json`.
type GPUSoftwareInfo struct {
	AMDSMITool     string `json:"amdsmi_tool"`
	AMDSMILibrary  string `json:"amdsmi_library"`
	ROCmVersion    string `json:"rocm_version"`
	AMDGPUVersion  string `json:"amdgpu_version"`
	AMDHSMPVersion string `json:"amd_hsmp_version"`
}

// FirmwareEntry is one firmware component on a GPU.
type FirmwareEntry struct {
	ID      string `json:"fw_id"`
	Version string `json:"fw_version"`
}

// GPUFirmware is one GPU's firmware component list.
type GPUFirmware struct {
	GPU    int             `json:"gpu"`
	FWList []FirmwareEntry `json:"fw_list"`
}

// NodeGPUSoftware is the per-node GPU software collection result.
type NodeGPUSoftware struct {
	Host     string           `json:"host"`
	Version  *GPUSoftwareInfo `json:"version,omitempty"`
	Firmware []GPUFirmware    `json:"firmware,omitempty"`
	Error    string           `json:"error,omitempty"`
}

// GPUSoftwareSnapshot is one fleet-wide GPU software sweep.
type GPUSoftwareSnapshot struct {
	CollectedAt time.Time         `json:"collected_at"`
	Nodes       []NodeGPUSoftware `json:"nodes"`
}

// DevlinkDevice is one NIC's normalized devlink info (vendor-agnostic).
type DevlinkDevice struct {
	PCIAddress   string `json:"pci_address"`
	Driver       string `json:"driver"`
	Vendor       string `json:"vendor"`
	SerialNumber string `json:"serial_number"`
	BoardSerial  string `json:"board_serial"`
	BoardID      string `json:"board_id"`
	ASICID       string `json:"asic_id"`
	ASICRev      string `json:"asic_rev"`
	FWVersion    string `json:"fw_version"`
	FWPSID       string `json:"fw_psid"`
	FWMgmt       string `json:"fw_mgmt"`
	FWMgmtAPI    string `json:"fw_mgmt_api"`
	FWCPLD       string `json:"fw_cpld"`
	FWHeartbeat  string `json:"fw_heartbeat"`
}

// NodeDevlink is the per-node devlink collection result.
type NodeDevlink struct {
	Host    string          `json:"host"`
	Devices []DevlinkDevice `json:"devices,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// DevlinkSnapshot is one fleet-wide NIC devlink sweep.
type DevlinkSnapshot struct {
	CollectedAt time.Time     `json:"collected_at"`
	Nodes       []NodeDevlink `json:"nodes"`
}

// SoftwareService collects the on-demand (software/firmware) collectors over the
// fleet singleton SSH pool, each behind a TTL cache with a stale-on-error fallback.
// Cold-cache HTTP calls wait for the first sweep (Python /api/software/* parity).
// The existing React pages fetch once on mount and cache the JSON, including an
// empty body — returning immediately would show blank until the user hits Refresh.
type SoftwareService struct {
	getPool func() *pssh.Pool
	logger  *slog.Logger
	ttl     time.Duration

	gpuMu         sync.Mutex
	gpuCache      *GPUSoftwareSnapshot
	gpuCollecting bool
	gpuWaiters    []chan error

	devlinkMu         sync.Mutex
	devlinkCache      *DevlinkSnapshot
	devlinkCollecting bool
	devlinkWaiters    []chan error
}

// NewSoftwareService wires the software collectors to the fleet singleton pool getter.
func NewSoftwareService(getPool func() *pssh.Pool, logger *slog.Logger) *SoftwareService {
	if logger == nil {
		logger = slog.Default()
	}
	return &SoftwareService{getPool: getPool, logger: logger, ttl: softwareTTL}
}

// GPUSoftware returns a fresh cached snapshot, or waits for a sweep when the
// cache is empty or older than softwareTTL.
func (s *SoftwareService) GPUSoftware(ctx context.Context) (*GPUSoftwareSnapshot, error) {
	if snap := s.gpuIfFresh(); snap != nil {
		return snap, nil
	}
	wait := s.kickGPUCollect()
	select {
	case <-ctx.Done():
		if snap := s.gpuCached(); snap != nil {
			return snap, nil
		}
		return nil, ctx.Err()
	case err := <-wait:
		if snap := s.gpuCached(); snap != nil {
			return snap, nil
		}
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("gpu software collection produced no data")
	}
}

func (s *SoftwareService) gpuIfFresh() *GPUSoftwareSnapshot {
	s.gpuMu.Lock()
	defer s.gpuMu.Unlock()
	if s.gpuCache != nil && time.Since(s.gpuCache.CollectedAt) < s.ttl {
		return s.gpuCache
	}
	return nil
}

func (s *SoftwareService) gpuCached() *GPUSoftwareSnapshot {
	s.gpuMu.Lock()
	defer s.gpuMu.Unlock()
	return s.gpuCache
}

func (s *SoftwareService) kickGPUCollect() <-chan error {
	ch := make(chan error, 1)
	s.gpuMu.Lock()
	if s.gpuCache != nil && time.Since(s.gpuCache.CollectedAt) < s.ttl {
		s.gpuMu.Unlock()
		ch <- nil
		return ch
	}
	s.gpuWaiters = append(s.gpuWaiters, ch)
	if s.gpuCollecting {
		s.gpuMu.Unlock()
		return ch
	}
	s.gpuCollecting = true
	s.gpuMu.Unlock()
	go s.runGPUCollect()
	return ch
}

func (s *SoftwareService) runGPUCollect() {
	ctx, cancel := context.WithTimeout(context.Background(), softwareCollectTimeout)
	defer cancel()
	snap, err := s.collectGPUSoftware(ctx)
	s.gpuMu.Lock()
	if err != nil {
		s.logger.Warn("gpu_software_collect_failed", "err", err)
	} else {
		s.gpuCache = snap
		s.logger.Info("gpu_software_collected", "nodes", len(snap.Nodes))
	}
	waiters := s.gpuWaiters
	s.gpuWaiters = nil
	s.gpuCollecting = false
	s.gpuMu.Unlock()
	for _, w := range waiters {
		w <- err
	}
}

func (s *SoftwareService) collectGPUSoftware(ctx context.Context) (*GPUSoftwareSnapshot, error) {
	pool := s.getPool()
	if pool == nil {
		return nil, fmt.Errorf("no SSH pool: inventory not configured")
	}

	var versionRes, firmwareRes map[string]pssh.Result
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); versionRes = pool.Exec(ctx, cmdAMDSMIVersion) }()
	go func() { defer wg.Done(); firmwareRes = pool.Exec(ctx, cmdAMDSMIFirmware) }()
	wg.Wait()

	nodes := make([]NodeGPUSoftware, 0, len(versionRes))
	for host, r := range versionRes {
		m := NodeGPUSoftware{Host: host}
		if r.Err != nil {
			m.Error = r.Err.Error()
		} else {
			m.Version = parseAMDSMIVersion(r.Output)
		}
		if fr, ok := firmwareRes[host]; ok && fr.Err == nil {
			m.Firmware = parseGPUFirmware(fr.Output)
		}
		nodes = append(nodes, m)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Host < nodes[j].Host })
	return &GPUSoftwareSnapshot{CollectedAt: time.Now().UTC(), Nodes: nodes}, nil
}

// parseAMDSMIVersion parses `amd-smi version --json`, a list whose first element
// carries the tool/library/rocm/amdgpu versions.
func parseAMDSMIVersion(raw string) *GPUSoftwareInfo {
	js := extractJSON(raw)
	if js == "" {
		return nil
	}
	var list []map[string]any
	if err := json.Unmarshal([]byte(js), &list); err != nil || len(list) == 0 {
		return nil
	}
	v := list[0]
	str := func(k string) string {
		if s, ok := v[k].(string); ok && s != "" {
			return s
		}
		return "N/A"
	}
	return &GPUSoftwareInfo{
		AMDSMITool:     str("version"),
		AMDSMILibrary:  str("amdsmi_library_version"),
		ROCmVersion:    str("rocm_version"),
		AMDGPUVersion:  str("amdgpu_version"),
		AMDHSMPVersion: str("amd_hsmp_driver_version"),
	}
}

// parseGPUFirmware parses `amd-smi firmware --json` into per-GPU component lists.
func parseGPUFirmware(raw string) []GPUFirmware {
	js := extractJSON(raw)
	if js == "" {
		return nil
	}
	objs, perr := gpuObjects(js)
	if perr != "" {
		return nil
	}
	fws := make([]GPUFirmware, 0, len(objs))
	for _, obj := range objs {
		g := GPUFirmware{GPU: intField(obj, "gpu")}
		if list, ok := obj["fw_list"].([]any); ok {
			for _, item := range list {
				e, ok := item.(map[string]any)
				if !ok {
					continue
				}
				id, _ := e["fw_id"].(string)
				ver, _ := e["fw_version"].(string)
				g.FWList = append(g.FWList, FirmwareEntry{ID: id, Version: ver})
			}
		}
		fws = append(fws, g)
	}
	sort.Slice(fws, func(i, j int) bool { return fws[i].GPU < fws[j].GPU })
	return fws
}

// NICDevlink returns a fresh cached snapshot, or waits for a sweep when the
// cache is empty or older than softwareTTL. Python's /nic/devlink always
// collects; we still cache for 180s so a remount of the same tab is instant.
func (s *SoftwareService) NICDevlink(ctx context.Context) (*DevlinkSnapshot, error) {
	if snap := s.devlinkIfFresh(); snap != nil {
		return snap, nil
	}
	wait := s.kickDevlinkCollect()
	select {
	case <-ctx.Done():
		if snap := s.devlinkCached(); snap != nil {
			return snap, nil
		}
		return nil, ctx.Err()
	case err := <-wait:
		if snap := s.devlinkCached(); snap != nil {
			return snap, nil
		}
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("nic devlink collection produced no data")
	}
}

func (s *SoftwareService) devlinkIfFresh() *DevlinkSnapshot {
	s.devlinkMu.Lock()
	defer s.devlinkMu.Unlock()
	if s.devlinkCache != nil && time.Since(s.devlinkCache.CollectedAt) < s.ttl {
		return s.devlinkCache
	}
	return nil
}

func (s *SoftwareService) devlinkCached() *DevlinkSnapshot {
	s.devlinkMu.Lock()
	defer s.devlinkMu.Unlock()
	return s.devlinkCache
}

func (s *SoftwareService) kickDevlinkCollect() <-chan error {
	ch := make(chan error, 1)
	s.devlinkMu.Lock()
	if s.devlinkCache != nil && time.Since(s.devlinkCache.CollectedAt) < s.ttl {
		s.devlinkMu.Unlock()
		ch <- nil
		return ch
	}
	s.devlinkWaiters = append(s.devlinkWaiters, ch)
	if s.devlinkCollecting {
		s.devlinkMu.Unlock()
		return ch
	}
	s.devlinkCollecting = true
	s.devlinkMu.Unlock()
	go s.runDevlinkCollect()
	return ch
}

func (s *SoftwareService) runDevlinkCollect() {
	ctx, cancel := context.WithTimeout(context.Background(), softwareCollectTimeout)
	defer cancel()
	snap, err := s.collectDevlink(ctx)
	s.devlinkMu.Lock()
	if err != nil {
		s.logger.Warn("nic_devlink_collect_failed", "err", err)
	} else {
		s.devlinkCache = snap
		s.logger.Info("nic_devlink_collected", "nodes", len(snap.Nodes))
	}
	waiters := s.devlinkWaiters
	s.devlinkWaiters = nil
	s.devlinkCollecting = false
	s.devlinkMu.Unlock()
	for _, w := range waiters {
		w <- err
	}
}

func (s *SoftwareService) collectDevlink(ctx context.Context) (*DevlinkSnapshot, error) {
	pool := s.getPool()
	if pool == nil {
		return nil, fmt.Errorf("no SSH pool: inventory not configured")
	}

	results := pool.Exec(ctx, cmdDevlinkInfo)
	nodes := make([]NodeDevlink, 0, len(results))
	for host, r := range results {
		m := NodeDevlink{Host: host}
		if r.Err != nil {
			m.Error = r.Err.Error()
		} else {
			m.Devices = parseDevlink(r.Output)
		}
		nodes = append(nodes, m)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Host < nodes[j].Host })
	return &DevlinkSnapshot{CollectedAt: time.Now().UTC(), Nodes: nodes}, nil
}

// parseDevlink parses `devlink dev info --json` ({"info": {"pci/...": {...}}})
// into normalized per-device records, mirroring the Python vendor mapping.
func parseDevlink(raw string) []DevlinkDevice {
	js := extractJSON(raw)
	if js == "" {
		return nil
	}
	var root struct {
		Info map[string]struct {
			Driver       string `json:"driver"`
			SerialNumber string `json:"serial_number"`
			BoardSerial  string `json:"board.serial_number"`
			Versions     struct {
				Fixed   map[string]string `json:"fixed"`
				Running map[string]string `json:"running"`
				Stored  map[string]string `json:"stored"`
			} `json:"versions"`
		} `json:"info"`
	}
	if err := json.Unmarshal([]byte(js), &root); err != nil {
		return nil
	}

	pick := func(m map[string]string, keys ...string) string {
		for _, k := range keys {
			if v, ok := m[k]; ok && v != "" {
				return v
			}
		}
		return "-"
	}

	devs := make([]DevlinkDevice, 0, len(root.Info))
	for pciDev, d := range root.Info {
		fixed, running := d.Versions.Fixed, d.Versions.Running
		dev := DevlinkDevice{
			PCIAddress:   trimPrefix(pciDev, "pci/"),
			Driver:       orDash(d.Driver),
			Vendor:       devlinkVendor(d.Driver),
			SerialNumber: orDash(d.SerialNumber),
			BoardSerial:  orDash(d.BoardSerial),
			BoardID:      pick(fixed, "board.id"),
			ASICID:       pick(fixed, "asic.id"),
			ASICRev:      pick(fixed, "asic.rev"),
			FWVersion:    pick(running, "fw", "fw.version", "fw.a35_fip_a"),
			FWPSID:       pick(fixed, "fw.psid"),
			FWMgmt:       pick(running, "fw.mgmt"),
			FWMgmtAPI:    pick(running, "fw.mgmt.api"),
			FWCPLD:       pick(running, "fw.cpld"),
			FWHeartbeat:  pick(running, "fw.heartbeat"),
		}
		devs = append(devs, dev)
	}
	sort.Slice(devs, func(i, j int) bool { return devs[i].PCIAddress < devs[j].PCIAddress })
	return devs
}

func devlinkVendor(driver string) string {
	switch driver {
	case "bnxt_en":
		return "Broadcom Thor2"
	case "mlx5_core":
		return "NVIDIA CX7"
	case "pds_core", "ionic":
		return "AMD AINIC"
	case "i40e":
		return "Intel"
	default:
		return "Unknown"
	}
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func trimPrefix(s, prefix string) string {
	if len(s) >= len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):]
	}
	return s
}
