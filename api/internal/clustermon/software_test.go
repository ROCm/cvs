package clustermon

import "testing"

const sampleAMDSMIVersion = `[{
  "tool": "AMDSMI Tool",
  "version": "26.2.0+021c61fc",
  "amdsmi_library_version": "26.2.0",
  "rocm_version": "7.0.2",
  "amdgpu_version": "6.16.6",
  "amd_hsmp_driver_version": "N/A"
}]`

const sampleAMDSMIFirmware = `WARNING: banner
[
  {"gpu": 1, "fw_list": [{"fw_id": "VCN", "fw_version": "1.2.3"}]},
  {"gpu": 0, "fw_list": [{"fw_id": "SMU", "fw_version": "9.9"}, {"fw_id": "VCN", "fw_version": "1.2.3"}]}
]`

const sampleDevlink = `{"info": {
  "pci/0000:76:00.0": {
    "driver": "bnxt_en",
    "serial_number": "ABC123",
    "board.serial_number": "BRD999",
    "versions": {
      "fixed": {"board.id": "BID", "asic.id": "AID", "asic.rev": "1", "fw.psid": "PSID1"},
      "running": {"fw": "230.0.0", "fw.mgmt": "M1"},
      "stored": {"fw": "230.0.0"}
    }
  },
  "pci/0000:0a:00.0": {
    "driver": "mlx5_core",
    "versions": {"fixed": {}, "running": {"fw.version": "28.39.1002"}}
  }
}}`

func TestParseAMDSMIVersion(t *testing.T) {
	v := parseAMDSMIVersion(sampleAMDSMIVersion)
	if v == nil {
		t.Fatal("nil version")
	}
	if v.ROCmVersion != "7.0.2" || v.AMDGPUVersion != "6.16.6" || v.AMDSMILibrary != "26.2.0" {
		t.Fatalf("version wrong: %+v", v)
	}
	if v.AMDHSMPVersion != "N/A" {
		t.Fatalf("hsmp should pass through N/A: %+v", v)
	}
}

func TestParseGPUFirmware(t *testing.T) {
	fw := parseGPUFirmware(sampleAMDSMIFirmware)
	if len(fw) != 2 {
		t.Fatalf("want 2 gpus, got %d: %+v", len(fw), fw)
	}
	// Sorted by GPU index.
	if fw[0].GPU != 0 || fw[1].GPU != 1 {
		t.Fatalf("not sorted: %+v", fw)
	}
	if len(fw[0].FWList) != 2 || fw[0].FWList[0].ID != "SMU" || fw[0].FWList[0].Version != "9.9" {
		t.Fatalf("gpu0 fw wrong: %+v", fw[0])
	}
}

func TestParseDevlink(t *testing.T) {
	devs := parseDevlink(sampleDevlink)
	if len(devs) != 2 {
		t.Fatalf("want 2 devices, got %d: %+v", len(devs), devs)
	}
	// Sorted by PCI address: 0000:0a:00.0 before 0000:76:00.0.
	mlx := devs[0]
	if mlx.PCIAddress != "0000:0a:00.0" || mlx.Driver != "mlx5_core" || mlx.Vendor != "NVIDIA CX7" {
		t.Fatalf("mlx wrong: %+v", mlx)
	}
	if mlx.FWVersion != "28.39.1002" {
		t.Fatalf("mlx fw.version fallback wrong: %+v", mlx)
	}
	bnxt := devs[1]
	if bnxt.Vendor != "Broadcom Thor2" || bnxt.FWVersion != "230.0.0" || bnxt.FWPSID != "PSID1" {
		t.Fatalf("bnxt wrong: %+v", bnxt)
	}
	if bnxt.SerialNumber != "ABC123" || bnxt.BoardSerial != "BRD999" || bnxt.ASICID != "AID" {
		t.Fatalf("bnxt fields wrong: %+v", bnxt)
	}
	// Missing fields default to "-".
	if mlx.SerialNumber != "-" || mlx.FWPSID != "-" {
		t.Fatalf("mlx defaults wrong: %+v", mlx)
	}
}
