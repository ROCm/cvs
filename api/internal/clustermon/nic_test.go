package clustermon

import (
	"testing"
)

const sampleRDMALink = `link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev ens1
link mlx5_1/1 state DOWN physical_state DISABLED netdev ens2`

const sampleRDMAStat = `[
  {"ifname": "rdma0", "port": 1, "ifindex": 5, "rx_pkts": 123, "tx_pkts": 456, "out_of_sequence": 0},
  {"ifname": "rdma1", "port": 1, "rx_pkts": 7}
]`

const sampleIPAddr = `2: ens1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP
    link/ether aa:bb:cc:dd:ee:ff brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.10/24 scope global ens1
    inet6 fe80::1/64 scope link
3: ens2: <BROADCAST,MULTICAST> mtu 1500 state DOWN
    link/ether 11:22:33:44:55:66 brd ff:ff:ff:ff:ff:ff`

const sampleIPSLink = `1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536
    RX: bytes  packets  errors  dropped missed  mcast
    100        2        0       0       0       0
    TX: bytes  packets  errors  dropped carrier collsns
    100        2        0       0       0       0
2: ens1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000
    RX: bytes    packets  errors  dropped missed  mcast
    123456789    123456   1       2       0       10
    TX: bytes    packets  errors  dropped carrier collsns
    987654321    654321   0       0       0       0`

const sampleRDMARes = `0: bnxt_re0: pd 1 cq 2 qp 3 cm_id 0 mr 4 ctx 0 srq 0
1: bnxt_re1: pd 5 cq 6 qp 7`


func TestParseRDMALinks(t *testing.T) {
	links := parseRDMALinks(sampleRDMALink)
	if len(links) != 2 {
		t.Fatalf("want 2 links, got %d: %+v", len(links), links)
	}
	if links[0].Device != "mlx5_0/1" || links[0].State != "ACTIVE" ||
		links[0].PhysicalState != "LINK_UP" || links[0].Netdev != "ens1" {
		t.Fatalf("link 0 wrong: %+v", links[0])
	}
	if links[1].State != "DOWN" {
		t.Fatalf("link 1 wrong: %+v", links[1])
	}
}

func TestParseRDMAStats(t *testing.T) {
	stats := parseRDMAStats(sampleRDMAStat)
	if len(stats) != 2 {
		t.Fatalf("want 2 devices, got %d: %+v", len(stats), stats)
	}
	if stats[0].Device != "rdma0/1" {
		t.Fatalf("dev key wrong: %+v", stats[0])
	}
	// Metadata keys (ifname/port/ifindex) must be dropped.
	if _, ok := stats[0].Stats["ifindex"]; ok {
		t.Fatalf("metadata leaked into stats: %+v", stats[0].Stats)
	}
	// Required projection fields are always present (even if zero in the source).
	if stats[0].Stats["rx_pkts"] != 123 || stats[0].Stats["tx_pkts"] != 456 {
		t.Fatalf("required counters wrong: %+v", stats[0].Stats)
	}
	// tx_roce_errors is a required field and must be present even though it
	// wasn't in the input (defaults to 0, not absent).
	if _, ok := stats[0].Stats["tx_roce_errors"]; !ok {
		t.Fatalf("required field tx_roce_errors missing from stats: %+v", stats[0].Stats)
	}
	// out_of_sequence is NOT in the projection and is 0 — it must be dropped.
	if _, ok := stats[0].Stats["out_of_sequence"]; ok {
		t.Fatalf("non-required zero counter out_of_sequence should be dropped: %+v", stats[0].Stats)
	}
}

func TestParseIPAddr(t *testing.T) {
	ifaces := parseIPAddr(sampleIPAddr)
	if len(ifaces) != 2 {
		t.Fatalf("want 2 ifaces, got %d: %+v", len(ifaces), ifaces)
	}
	e1 := ifaces[0]
	if e1.Name != "ens1" || e1.MTU != "9000" || e1.State != "UP" || e1.MAC != "aa:bb:cc:dd:ee:ff" {
		t.Fatalf("ens1 wrong: %+v", e1)
	}
	if len(e1.IPv4) != 1 || e1.IPv4[0] != "192.168.1.10/24" {
		t.Fatalf("ens1 ipv4 wrong: %+v", e1.IPv4)
	}
	if len(e1.IPv6) != 1 || e1.IPv6[0] != "fe80::1/64" {
		t.Fatalf("ens1 ipv6 wrong: %+v", e1.IPv6)
	}
}

func TestParseEthStats_SkipsLoopback(t *testing.T) {
	stats := parseEthStats(sampleIPSLink)
	if len(stats) != 1 {
		t.Fatalf("want 1 iface (lo skipped), got %d: %+v", len(stats), stats)
	}
	s := stats[0]
	if s.Iface != "ens1" {
		t.Fatalf("iface wrong: %+v", s)
	}
	if s.Stats["rx_bytes"] != 123456789 || s.Stats["rx_errors"] != 1 || s.Stats["rx_dropped"] != 2 {
		t.Fatalf("rx wrong: %+v", s.Stats)
	}
	if s.Stats["tx_bytes"] != 987654321 || s.Stats["tx_packets"] != 654321 {
		t.Fatalf("tx wrong: %+v", s.Stats)
	}
}

func TestParseRDMARes(t *testing.T) {
	res := parseRDMARes(sampleRDMARes)
	if len(res) != 2 {
		t.Fatalf("want 2 devices, got %d: %+v", len(res), res)
	}
	if res[0].Device != "bnxt_re0" || res[0].Values["qp"] != 3 || res[0].Values["mr"] != 4 {
		t.Fatalf("res 0 wrong: %+v", res[0])
	}
}

