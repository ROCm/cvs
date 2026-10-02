package rccl

import (
	"regexp"
	"strconv"
	"strings"
)

var (
	reNCCL    = regexp.MustCompile(`(?i)NCCL version\s+(\S+)`)
	reCUDA    = regexp.MustCompile(`(?i)CUDA runtime version:\s+(\d+)`)
	reCUDAD   = regexp.MustCompile(`(?i)CUDA driver version:\s+(\d+)`)
	reComm    = regexp.MustCompile(`(?i)Communicator\s+\(Hash:\s*(\S+)\)`)
	reRank    = regexp.MustCompile(`(?i)^\s*Rank\s+(\d+):`)
	reHost    = regexp.MustCompile(`(?i)Host:\s+(\S+)`)
	rePID     = regexp.MustCompile(`(?i)Pid:\s+(\d+)`)
	reCUDADev = regexp.MustCompile(`(?i)CUDA Dev:\s+(\d+)`)
	reNVML    = regexp.MustCompile(`(?i)NVML Dev:\s+(\d+)`)
	reColl    = regexp.MustCompile(`Collective\s+(\S+)\s+count:\s+(\d+)`)
	reInit    = regexp.MustCompile(`(?i)Init state:\s+(\d+)`)
	reAsync   = regexp.MustCompile(`(?i)Async error:\s+(\d+)`)
	reFinal   = regexp.MustCompile(`(?i)Finalize called:\s+(\S+)`)
	reDestroy = regexp.MustCompile(`(?i)Destroy flag:\s+(\S+)`)
	reAbort   = regexp.MustCompile(`(?i)Abort flag:\s+(\S+)`)
	reMissing = regexp.MustCompile(`(?i)Missing ranks:\s+(\d+)`)

	reRCCLVer       = regexp.MustCompile(`RCCL version (\S+)\s+compiled with ROCm`)
	reHIPDriver     = regexp.MustCompile(`HIP runtime version (\d+),\s*amdgpu driver version (\d+)`)
	reJobSummaryRow = regexp.MustCompile(`(?m)^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$`)
	reCommRow       = regexp.MustCompile(`(?m)^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+(?:-\d+)?)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$`)
	reDeadPeers     = regexp.MustCompile(`(?i)Dead peers?:\s*(.+)`)
	reConnRefused   = regexp.MustCompile(`(?i)Connection refused|Failed to connect|Connection reset by peer`)
)

func ParseText(raw string) Snapshot {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return emptySnap("no_job")
	}
	if reConnRefused.MatchString(raw) {
		return emptySnap("no_job")
	}
	if strings.HasPrefix(raw, "{") {
		return emptySnap("error")
	}
	if strings.Contains(raw, "Job summary") || reRCCLVer.MatchString(raw) {
		return parseRASTable(raw)
	}
	return parseNCCLDump(raw)
}

func parseRASTable(raw string) Snapshot {
	summary := parseRASJobSummary(raw)
	comms := parseRASCommunicators(raw)
	dead := parseDeadPeers(raw)
	errs := parseErrorsSection(raw)
	state := "healthy"
	if summary == nil && len(comms) == 0 {
		state = "no_job"
	} else if len(dead) > 0 || len(errs) > 0 {
		state = "degraded"
	} else {
		for _, c := range comms {
			if c.Health == "degraded" || c.MissingRanks > 0 {
				state = "degraded"
				break
			}
		}
	}
	if dead == nil {
		dead = []string{}
	}
	if errs == nil {
		errs = []string{}
	}
	return Snapshot{
		Timestamp:     emptySnap("").Timestamp,
		State:         state,
		JobSummary:    summary,
		Communicators: comms,
		Peers:         []any{},
		DeadPeers:     dead,
		Errors:        errs,
	}
}

func parseRASJobSummary(text string) *JobSummary {
	summary := &JobSummary{RCCLVersion: "unknown"}
	if m := reRCCLVer.FindStringSubmatch(text); len(m) > 1 {
		summary.RCCLVersion = m[1]
	}
	if m := reHIPDriver.FindStringSubmatch(text); len(m) > 2 {
		summary.HIPRuntimeVersion, _ = strconv.Atoi(m[1])
		summary.AMDGPUDriverVersion, _ = strconv.Atoi(m[2])
		summary.HIPRuntimeVersionStr = hipVersionStr(summary.HIPRuntimeVersion)
		summary.AMDGPUDriverVersionStr = hipVersionStr(summary.AMDGPUDriverVersion)
		summary.DriverRuntimeMismatch = summary.HIPRuntimeVersion != summary.AMDGPUDriverVersion
	}
	sec := rasSection(text, "Job summary")
	row := reJobSummaryRow.FindStringSubmatch(sec)
	if len(row) < 6 {
		if summary.RCCLVersion == "unknown" && summary.HIPRuntimeVersion == 0 {
			return nil
		}
		return summary
	}
	summary.TotalNodes, _ = strconv.Atoi(row[1])
	summary.TotalProcesses, _ = strconv.Atoi(row[4])
	summary.TotalGPUs, _ = strconv.Atoi(row[5])
	return summary
}

func parseRASCommunicators(text string) []Communicator {
	sec := rasSection(text, "Communicators")
	var comms []Communicator
	for _, row := range reCommRow.FindAllStringSubmatch(sec, -1) {
		if len(row) < 9 {
			continue
		}
		groupNum, _ := strconv.Atoi(row[1])
		commsInGroup, _ := strconv.Atoi(row[2])
		ranksPerComm, _ := strconv.Atoi(row[5])
		ranksInGroup, _ := strconv.Atoi(row[6])
		status := row[7]
		errors := row[8]
		health := "healthy"
		if errors != "OK" {
			health = "degraded"
		} else if status != "RUNNING" {
			health = "healthy"
		}
		total := ranksPerComm * commsInGroup
		missing := total - ranksInGroup
		if missing < 0 {
			missing = 0
		}
		comms = append(comms, Communicator{
			CommHash:        "group_" + strconv.Itoa(groupNum),
			TotalRanks:      total,
			RespondingRanks: ranksInGroup,
			MissingRanks:    missing,
			Ranks:           []Rank{},
			Health:          health,
		})
	}
	if comms == nil {
		return []Communicator{}
	}
	return comms
}

func parseDeadPeers(text string) []string {
	m := reDeadPeers.FindStringSubmatch(text)
	if len(m) < 2 {
		return []string{}
	}
	var out []string
	for _, p := range strings.Split(m[1], ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if out == nil {
		return []string{}
	}
	return out
}

func parseErrorsSection(text string) []string {
	content := rasSection(text, "Errors")
	var out []string
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	if out == nil {
		return []string{}
	}
	return out
}

func rasSection(text, header string) string {
	lines := strings.Split(text, "\n")
	start := -1
	for i, line := range lines {
		trim := strings.TrimSpace(line)
		if !strings.HasPrefix(trim, header) {
			continue
		}
		if i+1 < len(lines) && isEqualsUnderline(lines[i+1]) {
			start = i + 2
			break
		}
	}
	if start < 0 {
		return ""
	}
	var body []string
	for i := start; i < len(lines); i++ {
		if i+1 < len(lines) && isEqualsUnderline(lines[i+1]) && looksLikeSectionHeader(lines[i]) {
			break
		}
		body = append(body, lines[i])
	}
	return strings.Join(body, "\n")
}

func isEqualsUnderline(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) < 3 {
		return false
	}
	return strings.Trim(s, "=") == ""
}

func looksLikeSectionHeader(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	c := s[0]
	return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
}

func parseNCCLDump(raw string) Snapshot {
	lower := strings.ToLower(raw)
	if strings.Contains(lower, "no job") || strings.Contains(lower, "no nccl") {
		return emptySnap("no_job")
	}

	summary := &JobSummary{RCCLVersion: "unknown"}
	if m := reNCCL.FindStringSubmatch(raw); len(m) > 1 {
		summary.RCCLVersion = m[1]
	}
	if m := reCUDA.FindStringSubmatch(raw); len(m) > 1 {
		summary.HIPRuntimeVersion, _ = strconv.Atoi(m[1])
		summary.HIPRuntimeVersionStr = hipVersionStr(summary.HIPRuntimeVersion)
	}
	if m := reCUDAD.FindStringSubmatch(raw); len(m) > 1 {
		summary.AMDGPUDriverVersion, _ = strconv.Atoi(m[1])
		summary.AMDGPUDriverVersionStr = hipVersionStr(summary.AMDGPUDriverVersion)
	}
	summary.DriverRuntimeMismatch = summary.HIPRuntimeVersion != summary.AMDGPUDriverVersion

	var comms []Communicator
	lines := strings.Split(raw, "\n")
	var cur *Communicator
	var rank *Rank
	hosts := map[string]struct{}{}
	procs := map[string]struct{}{}

	flushRank := func() {
		if cur != nil && rank != nil {
			cur.Ranks = append(cur.Ranks, *rank)
		}
		rank = nil
	}
	flushComm := func() {
		flushRank()
		if cur != nil {
			cur.RespondingRanks = len(cur.Ranks)
			if cur.TotalRanks == 0 {
				cur.TotalRanks = len(cur.Ranks) + cur.MissingRanks
			}
			health := "healthy"
			if cur.MissingRanks > 0 {
				health = "degraded"
			}
			for _, r := range cur.Ranks {
				if r.Status.AsyncError != 0 || r.Status.AbortFlag || r.Status.InitState != 0 {
					health = "degraded"
					break
				}
			}
			cur.Health = health
			comms = append(comms, *cur)
		}
		cur = nil
	}

	for _, line := range lines {
		if m := reComm.FindStringSubmatch(line); len(m) > 1 {
			flushComm()
			cur = &Communicator{CommHash: m[1]}
			continue
		}
		if cur == nil {
			continue
		}
		if m := reMissing.FindStringSubmatch(line); len(m) > 1 {
			cur.MissingRanks, _ = strconv.Atoi(m[1])
			continue
		}
		if m := reRank.FindStringSubmatch(line); len(m) > 1 {
			flushRank()
			rn, _ := strconv.Atoi(m[1])
			rank = &Rank{CommRank: rn, CollOpCounts: map[string]int{}}
			continue
		}
		if rank == nil {
			continue
		}
		if m := reHost.FindStringSubmatch(line); len(m) > 1 {
			rank.NodeAddr = m[1]
			hosts[m[1]] = struct{}{}
		}
		if m := rePID.FindStringSubmatch(line); len(m) > 1 {
			rank.PID, _ = strconv.Atoi(m[1])
			if rank.NodeAddr != "" {
				procs[rank.NodeAddr+":"+m[1]] = struct{}{}
			}
		}
		if m := reCUDADev.FindStringSubmatch(line); len(m) > 1 {
			rank.CUDADev, _ = strconv.Atoi(m[1])
		}
		if m := reNVML.FindStringSubmatch(line); len(m) > 1 {
			rank.NVMLDev, _ = strconv.Atoi(m[1])
		}
		if m := reColl.FindStringSubmatch(line); len(m) > 2 {
			n, _ := strconv.Atoi(m[2])
			rank.CollOpCounts[m[1]] = n
		}
		if m := reInit.FindStringSubmatch(line); len(m) > 1 {
			rank.Status.InitState, _ = strconv.Atoi(m[1])
		}
		if m := reAsync.FindStringSubmatch(line); len(m) > 1 {
			rank.Status.AsyncError, _ = strconv.Atoi(m[1])
		}
		if m := reFinal.FindStringSubmatch(line); len(m) > 1 {
			rank.Status.FinalizeCalled = isTrue(m[1])
		}
		if m := reDestroy.FindStringSubmatch(line); len(m) > 1 {
			rank.Status.DestroyFlag = isTrue(m[1])
		}
		if m := reAbort.FindStringSubmatch(line); len(m) > 1 {
			rank.Status.AbortFlag = isTrue(m[1])
		}
	}
	flushComm()

	totalGPUs := 0
	for _, c := range comms {
		totalGPUs += c.TotalRanks
	}
	summary.TotalNodes = max(len(hosts), 1)
	summary.TotalProcesses = max(len(procs), 1)
	summary.TotalGPUs = totalGPUs
	state := "healthy"
	if len(comms) == 0 {
		state = "no_job"
	}
	for _, c := range comms {
		if c.Health == "degraded" {
			state = "degraded"
			break
		}
	}
	return Snapshot{
		Timestamp:     emptySnap("").Timestamp,
		State:         state,
		JobSummary:    summary,
		Communicators: comms,
		Peers:         []any{},
		DeadPeers:     []string{},
		Errors:        []string{},
	}
}

func isTrue(s string) bool {
	s = strings.ToLower(strings.TrimSpace(s))
	return s == "1" || s == "true" || s == "yes"
}
