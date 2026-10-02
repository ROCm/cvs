package rccl

// EventType is the timeline event for a job_state transition (Python TYPE_MAP).
func EventType(prev, curr string) string {
	if prev == curr {
		return ""
	}
	key := prev + "->" + curr
	if t, ok := stateEventMap[key]; ok {
		return t
	}
	return "state_change"
}

var stateEventMap = map[string]string{
	"no_job->healthy":       "job_start",
	"no_job->degraded":      "job_start_degraded",
	"no_job->unreachable":   "nodes_unreachable",
	"no_job->error":         "collector_error",
	"healthy->degraded":     "job_degraded",
	"healthy->no_job":       "job_end",
	"healthy->unreachable":  "node_unreachable",
	"healthy->error":        "collector_error",
	"degraded->healthy":     "job_recovered",
	"degraded->no_job":      "job_end",
	"degraded->unreachable": "node_unreachable",
	"degraded->error":       "collector_error",
	"unreachable->healthy":  "node_recovered",
	"unreachable->degraded": "node_recovered_degraded",
	"unreachable->no_job":   "job_end",
	"unreachable->error":    "collector_error",
	"error->healthy":        "job_start",
	"error->degraded":       "job_start_degraded",
	"error->no_job":         "job_end",
	"error->unreachable":    "node_unreachable",
}
