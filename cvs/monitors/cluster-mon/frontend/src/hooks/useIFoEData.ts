import { useState, useEffect, useCallback } from 'react'
import { api } from '@/services/api'

export interface SwitchOverview {
  platform_summary: Record<string, any>
  psu_status:       Record<string, any>[]
  fan_status:       Record<string, any>[]
  temperature:      Record<string, any>[]
  sw_version:       Record<string, string>
  system_status:    Record<string, string>
  docker_ps:        Record<string, any>[]
  docker_stats:     Record<string, any>[]
  memory:           Record<string, any>
}

export interface SwitchMetrics {
  interfaces:     Record<string, any>[]
  intf_counters:  Record<string, any>[]
  pfc_counters:   Record<string, any>[]
  queue_counters: Record<string, any>[]
  queue_wm:       Record<string, any>[]
}

export interface IFoEData {
  compute_devices:      Record<string, Record<string, string>[]>
  compute_ports:        Record<string, Record<string, string>[]>
  compute_port_stats:   Record<string, {
    mac:  Record<string, any>[]
    fec:  Record<string, any>[]
    ifcp: Record<string, any>[]
    pfc:  Record<string, any>[]
  }>
  switch_vlan:     Record<string, Record<string, Record<string, string>[]>>
  switch_mac:      Record<string, Record<string, Record<string, string>[]>>
  topology:        TopologyRow[]
  switch_overview: Record<string, SwitchOverview>
  switch_metrics:  Record<string, SwitchMetrics>
  last_updated:    string | null
  errors:          Record<string, string>
  state:           string
}

export interface TopologyRow {
  compute_tray:   string
  gpu_index:      string
  station_index:  string
  port_index:     string
  ifoe_interface: string
  compute_mac:    string
  link_status:    string
  speed:          string
  switch_tray:    string
  asic:           string
  switch_port:    string
  mapped:         boolean
}

export function useIFoEData(pollMs = 300_000) {
  const [data, setData]         = useState<IFoEData | null>(null)
  const [loading, setLoading]   = useState(false)
  const [refreshing, setRefreshing] = useState(false)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await api.getIFoEData() as IFoEData
      setData(resp)
    } catch {
      // leave stale data in place
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, pollMs)
    return () => clearInterval(id)
  }, [fetchData, pollMs])

  const triggerRefresh = async () => {
    setRefreshing(true)
    try {
      await api.refreshIFoEData()
      setTimeout(fetchData, 3000)
    } finally {
      setRefreshing(false)
    }
  }

  return { data, loading, refreshing, triggerRefresh }
}
