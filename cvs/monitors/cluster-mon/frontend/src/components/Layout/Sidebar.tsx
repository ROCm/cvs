import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, Settings, Cpu, Network, Activity,
  Share2, FileText, Clock, GitFork, Zap, HardDrive,
  ChevronDown, ChevronRight, Server,
} from 'lucide-react'
import { cn } from '@/utils/cn'

// ---------------------------------------------------------------------------
// Leaf nav item (NavLink)
// ---------------------------------------------------------------------------

function NavItem({
  name,
  href,
  icon: Icon,
  indent = 0,
}: {
  name: string
  href: string
  icon?: React.ElementType
  indent?: number
}) {
  return (
    <NavLink
      to={href}
      className={({ isActive }) =>
        cn(
          'flex items-center gap-3 py-2.5 rounded-lg transition-colors text-sm',
          indent === 0 && 'px-4',
          indent === 1 && 'pl-8 pr-4',
          indent === 2 && 'pl-12 pr-4',
          indent === 3 && 'pl-16 pr-4',
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-gray-300 hover:bg-gray-800 hover:text-white'
        )
      }
    >
      {Icon && <Icon className="h-4 w-4 shrink-0" />}
      <span className="font-medium">{name}</span>
    </NavLink>
  )
}

// ---------------------------------------------------------------------------
// Expandable section (toggle + children)
// ---------------------------------------------------------------------------

function NavSection({
  name,
  icon: Icon,
  childPrefixes,
  children,
  indent = 0,
}: {
  name: string
  icon?: React.ElementType
  childPrefixes: string[]
  children: React.ReactNode
  indent?: number
}) {
  const location = useLocation()
  const isChildActive = childPrefixes.some((p) => location.pathname.startsWith(p))
  const [open, setOpen] = useState(isChildActive)

  useEffect(() => {
    if (isChildActive) setOpen(true)
  }, [isChildActive])

  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'w-full flex items-center gap-3 py-2.5 rounded-lg transition-colors text-sm font-medium',
          indent === 0 && 'px-4',
          indent === 1 && 'pl-8 pr-4',
          indent === 2 && 'pl-12 pr-4',
          isChildActive
            ? 'text-blue-400'
            : 'text-gray-300 hover:bg-gray-800 hover:text-white'
        )}
      >
        {Icon && <Icon className="h-4 w-4 shrink-0" />}
        <span className="flex-1 text-left">{name}</span>
        {open
          ? <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          : <ChevronRight className="h-3.5 w-3.5 shrink-0" />}
      </button>
      {open && (
        <div className="mt-0.5 space-y-0.5">
          {children}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section label divider
// ---------------------------------------------------------------------------

function SectionLabel({ label }: { label: string }) {
  return (
    <div className="pt-4 pb-1 px-4">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
        {label}
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

export function Sidebar() {
  return (
    <div className="flex flex-col w-64 bg-gray-900 dark:bg-gray-950 text-white">
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-gray-800 dark:border-gray-800">
        <Activity className="h-8 w-8 text-blue-400" />
        <div>
          <h1 className="text-lg font-bold">GPU Cluster</h1>
          <p className="text-xs text-gray-400">Monitor</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-4 overflow-y-auto space-y-0.5">

        <NavItem name="Dashboard"     href="/"       icon={LayoutDashboard} />
        <NavItem name="Configuration" href="/config" icon={Settings} />

        {/* ── Compute ── */}
        <NavSection name="Compute" icon={Cpu} childPrefixes={['/compute']}>
          <NavSection name="CPUs" icon={Server} childPrefixes={['/compute/cpus']} indent={1}>
            <NavItem name="Summary" href="/compute/cpus/summary" indent={2} />
            <NavItem name="Metrics" href="/compute/cpus/metrics" indent={2} />
          </NavSection>

          <NavSection name="GPUs" icon={Cpu} childPrefixes={['/compute/gpus']} indent={1}>
            <NavItem name="Summary"    href="/compute/gpus/summary" indent={2} />
            <NavItem name="Metrics"    href="/compute/gpus/metrics" indent={2} />
            <NavItem name="IFoE Ports" href="/compute/gpus/ifoe"    indent={2} />
          </NavSection>
        </NavSection>

        {/* ── Networks ── */}
        <NavSection name="Networks" icon={Network} childPrefixes={['/networks']}>
          <NavSection name="Frontend" icon={Network} childPrefixes={['/networks/frontend']} indent={1}>
            <NavItem name="Health"  href="/networks/frontend/health"  indent={2} />
            <NavItem name="Metrics" href="/networks/frontend/metrics" indent={2} />
          </NavSection>

          <NavSection name="Backend" icon={Network} childPrefixes={['/networks/backend']} indent={1}>
            <NavSection name="Scale-out" icon={Network} childPrefixes={['/networks/backend/scale-out']} indent={2}>
              <NavItem name="Summary" href="/networks/backend/scale-out/summary" indent={3} />
              <NavItem name="Health"  href="/networks/backend/scale-out/health"  indent={3} />
              <NavItem name="Metrics" href="/networks/backend/scale-out/metrics" indent={3} />
            </NavSection>

            <NavSection name="Scale-up" icon={Network} childPrefixes={['/networks/backend/scale-up']} indent={2}>
              <NavItem name="Overview" href="/networks/backend/scale-up/overview" indent={3} />
              <NavItem name="Links"    href="/networks/backend/scale-up/links"    indent={3} />
              <NavItem name="Metrics"  href="/networks/backend/scale-up/metrics"  indent={3} />
            </NavSection>
          </NavSection>
        </NavSection>

        {/* ── Storage ── */}
        <NavSection name="Storage" icon={HardDrive} childPrefixes={['/storage']}>
          <NavItem name="Summary"        href="/storage/summary" indent={1} />
          <NavItem name="IO Metrics"     href="/storage/io"      indent={1} />
          <NavItem name="Cache & Memory" href="/storage/cache"   indent={1} />
        </NavSection>

        <NavItem name="Topology" href="/topology" icon={Share2} />
        <NavItem name="Logs"     href="/logs"     icon={FileText} />

        {/* RCCL */}
        <SectionLabel label="RCCL" />
        <NavItem name="RCCL Health"  href="/rccl-health"      icon={Activity} />
        <NavItem name="RAS Topology" href="/rccl-topology"    icon={GitFork} />
        <NavItem name="Timeline"     href="/rccl-timeline"    icon={Clock} />
        <NavItem name="Performance"  href="/rccl-performance" icon={Zap} />
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-800">
        <p className="text-xs text-gray-500">Version 0.1.0</p>
      </div>
    </div>
  )
}
