export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes === 0 || isNaN(bytes)) return '0 B'

  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))

  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

export function formatPercentage(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return '0.0%'
  return `${value.toFixed(1)}%`
}

export function formatTemperature(celsius: number | null | undefined): string {
  if (celsius === null || celsius === undefined || isNaN(celsius)) return '0.0°C'
  return `${celsius.toFixed(1)}°C`
}

export function formatPower(watts: number | null | undefined): string {
  if (watts === null || watts === undefined || isNaN(watts)) return '0.0W'
  return `${watts.toFixed(1)}W`
}

export function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString()
}

export function getStatusColor(value: number, threshold: number): string {
  if (value >= threshold) return 'text-red-500'
  if (value >= threshold * 0.8) return 'text-yellow-500'
  return 'text-green-500'
}

export function getStatusBgColor(value: number, threshold: number): string {
  if (value >= threshold) return 'bg-red-500'
  if (value >= threshold * 0.8) return 'bg-yellow-500'
  return 'bg-green-500'
}
