package collectors

import (
	"strings"
	"testing"
)

const sampleLspci = `
01:00.0 Ethernet controller: Example ConnectX-7
	LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported
	LnkSta: Speed 16GT/s, Width x8
02:00.0 Network controller: Example BCM
	LnkCap: Speed 8GT/s, Width x4
	LnkSta: Speed 8GT/s, Width x4
`

func TestNICPCIeCommandUsesEREAlternation(t *testing.T) {
	if strings.Contains(cmdNICPCIe, `\|`) {
		t.Fatalf("escaped pipe matches a literal: %s", cmdNICPCIe)
	}
	if !strings.Contains(cmdNICPCIe, "ethernet|network") {
		t.Fatalf("missing alternation: %s", cmdNICPCIe)
	}
}

func TestParseNICPCIe(t *testing.T) {
	devs := parseNICPCIe(sampleLspci)
	if len(devs) != 2 {
		t.Fatalf("devices=%d %v", len(devs), devs)
	}
	eth, _ := devs["01:00.0"].(map[string]any)
	if eth["link_speed_cap"] != "32GT/s" || eth["link_width_cap"] != "x16" {
		t.Fatalf("cap %v", eth)
	}
	if eth["link_speed_current"] != "16GT/s" || eth["link_width_current"] != "x8" {
		t.Fatalf("sta %v", eth)
	}
	if eth["pcie_gen"] != "Gen4" {
		t.Fatalf("gen %v", eth["pcie_gen"])
	}
	net, _ := devs["02:00.0"].(map[string]any)
	if net["pcie_gen"] != "Gen3" || net["link_speed_cap"] != "8GT/s" {
		t.Fatalf("net %v", net)
	}
	if len(parseNICPCIe("no devices here")) != 0 {
		t.Fatal("expected empty")
	}
}

func TestPCIeGenMapping(t *testing.T) {
	cases := map[string]string{"64GT/s": "Gen5", "32GT/s": "Gen5", "16GT/s": "Gen4", "5GT/s": "Gen2", "2.5GT/s": "Gen1", "": ""}
	for in, want := range cases {
		if got := pcieGen(in); got != want {
			t.Fatalf("%s -> %s, want %s", in, got, want)
		}
	}
}
