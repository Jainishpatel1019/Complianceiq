import { Routes, Route, NavLink } from 'react-router-dom'
import { Activity, BarChart2, Network, AlertTriangle, FlaskConical } from 'lucide-react'
import RegulatoryFeed from './pages/RegulatoryFeed'
import Reports from './pages/Reports'
import Causal from './pages/Causal'
import Graph from './pages/Graph'

const NAV = [
  { to: '/',        label: 'Regulatory Feed',  icon: Activity },
  { to: '/reports', label: 'Impact Reports',   icon: BarChart2 },
  { to: '/causal',  label: 'Causal Analysis',  icon: FlaskConical },
  { to: '/graph',   label: 'Knowledge Graph',  icon: Network },
  { to: '/alerts',  label: 'Alerts',           icon: AlertTriangle, soon: true },
]

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-gray-800">
          <span className="text-lg font-bold tracking-tight text-white">
            Compliance<span className="text-blue-400">IQ</span>
          </span>
          <p className="text-xs text-gray-500 mt-0.5">Regulatory Intelligence</p>
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-2 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon, soon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                  soon
                    ? 'text-gray-600 cursor-default pointer-events-none'
                    : isActive
                    ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }
            >
              <Icon size={15} />
              <span>{label}</span>
              {soon && (
                <span className="ml-auto text-xs text-gray-700 bg-gray-800 px-1.5 py-0.5 rounded">
                  Soon
                </span>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Phase badge */}
        <div className="px-4 py-3 border-t border-gray-800">
          <span className="text-xs text-gray-600">Phase 5 — Evaluation</span>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-950">
        <Routes>
          <Route path="/"        element={<RegulatoryFeed />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/causal"  element={<Causal />} />
          <Route path="/graph"   element={<Graph />} />
        </Routes>
      </main>
    </div>
  )
}
