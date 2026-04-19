/**
 * Regulatory Feed — Phase 2 dashboard page.
 *
 * Shows the regulatory change feed sorted by drift score descending,
 * with per-regulation CI bars, JSD significance badges, and a weekly
 * sparkline showing which regulations changed most this week.
 *
 * Phase 2 goal from the master doc:
 *   "You can show a chart of which regulations changed most this week."
 */
import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { RefreshCw, AlertTriangle, Filter } from 'lucide-react'
import { useApi } from '../hooks/useApi'
import ScoreBadge from '../components/ScoreBadge'
import CIBar from '../components/CIBar'

// ── Colour scale for drift bar chart ────────────────────────────────────────
function driftColour(score) {
  if (score >= 0.5) return '#ef4444'  // red
  if (score >= 0.2) return '#f59e0b'  // amber
  return '#22c55e'                     // green
}

// ── Tooltip for the bar chart ────────────────────────────────────────────────
function DriftTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 text-xs max-w-xs">
      <p className="text-gray-300 font-bold mb-1 truncate">{d.title}</p>
      <p className="text-gray-400">{d.agency}</p>
      <p className="text-blue-300 mt-1">
        Drift: <span className="font-mono">{d.drift_display}</span>
      </p>
      {d.jsd_significant && (
        <p className="text-yellow-400 mt-0.5">⚠ JSD significant (p={d.jsd_p_value?.toFixed(3)})</p>
      )}
    </div>
  )
}

// ── Section heatmap ──────────────────────────────────────────────────────────
function SectionHeatmap({ regulationId }) {
  const { data, loading } = useApi(`/change-scores/heatmap/${regulationId}`)

  if (loading) return <div className="text-gray-600 text-xs p-2">Loading heatmap…</div>
  if (!data?.sections?.length) return null

  return (
    <div className="mt-3 border-t border-gray-800 pt-3">
      <p className="text-xs text-gray-500 mb-2">Section heatmap</p>
      <div className="space-y-1.5">
        {data.sections.slice(0, 8).map((s, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-36 truncate">{s.section_title}</span>
            <div className="flex-1 h-2 rounded bg-gray-800 overflow-hidden">
              <div
                className="h-full rounded transition-all"
                style={{
                  width: `${Math.round(s.drift_score * 100)}%`,
                  backgroundColor: driftColour(s.drift_score),
                  opacity: 0.7,
                }}
              />
            </div>
            <span className="text-xs font-mono text-gray-400 w-8 text-right">
              {Math.round(s.drift_score * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Feed row ─────────────────────────────────────────────────────────────────
function FeedRow({ item, expanded, onToggle }) {
  return (
    <div
      className="border border-gray-800 rounded-lg bg-gray-900/50 hover:bg-gray-900 transition-colors cursor-pointer"
      onClick={onToggle}
    >
      <div className="p-4">
        {/* Top row: agency + badges */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-blue-400 font-mono">{item.document_number}</span>
              <span className="text-xs text-gray-600">·</span>
              <span className="text-xs text-gray-500">{item.agency}</span>
              {item.flagged_for_analysis && (
                <AlertTriangle size={12} className="text-yellow-500" />
              )}
            </div>
            <p className="text-sm text-gray-200 leading-snug truncate">{item.title}</p>
          </div>

          {/* Score badges */}
          <div className="flex flex-col items-end gap-1.5 shrink-0">
            <ScoreBadge score={item.drift_score} label="drift" />
            {item.jsd_significant && (
              <ScoreBadge score={item.jsd_score} label="jsd" />
            )}
          </div>
        </div>

        {/* CI bar */}
        <div className="mt-3">
          <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
            <span>Semantic drift  95% CI</span>
            <span className="font-mono text-gray-400">{item.drift_display}</span>
          </div>
          <CIBar score={item.drift_score} ciLow={item.drift_ci_low} ciHigh={item.drift_ci_high} />
        </div>

        {/* Three measure summary */}
        <div className="flex gap-4 mt-3 text-xs text-gray-600">
          <span>
            Drift <span className="font-mono text-gray-400">{item.drift_display}</span>
          </span>
          <span>
            JSD <span className={`font-mono ${item.jsd_significant ? 'text-yellow-400' : 'text-gray-400'}`}>
              {item.jsd_score ? (item.jsd_score * 100).toFixed(1) + '%' : '—'}
            </span>
          </span>
          <span>
            W2 <span className="font-mono text-gray-400">
              {item.wasserstein_score ? (item.wasserstein_score * 100).toFixed(1) + '%' : '—'}
            </span>
          </span>
          <span>
            Composite <span className="font-mono text-gray-400">
              {(item.composite_score * 100).toFixed(1)}%
            </span>
          </span>
        </div>
      </div>

      {expanded && <SectionHeatmap regulationId={item.regulation_id} />}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function RegulatoryFeed() {
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const [expandedId, setExpandedId]   = useState(null)

  const { data: scores, loading, error, refetch } = useApi('/change-scores', {
    limit: 50,
    flagged_only: flaggedOnly,
  })

  // Prepare chart data (top 15 by drift)
  const chartData = (scores ?? [])
    .slice(0, 15)
    .map(s => ({
      name: s.document_number,
      drift: +(s.drift_score ?? 0).toFixed(3),
      title: s.title,
      agency: s.agency,
      drift_display: s.drift_display,
      jsd_significant: s.jsd_significant,
      jsd_p_value: s.jsd_p_value,
    }))

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">Regulatory Feed</h1>
          <p className="text-sm text-gray-500 mt-0.5">
            Change scores across Federal Register + SEC rules — sorted by semantic drift
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setFlaggedOnly(f => !f)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs border transition-colors ${
              flaggedOnly
                ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40'
                : 'bg-gray-800 text-gray-400 border-gray-700 hover:text-gray-200'
            }`}
          >
            <Filter size={12} />
            Flagged only
          </button>
          <button
            onClick={refetch}
            className="p-1.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Bar chart — top 15 regulations by drift this week */}
      {chartData.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 mb-6">
          <p className="text-sm font-medium text-gray-300 mb-4">
            Top regulations by semantic drift
          </p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 40 }}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 9, fill: '#6b7280' }}
                angle={-45}
                textAnchor="end"
              />
              <YAxis
                tick={{ fontSize: 9, fill: '#6b7280' }}
                domain={[0, 1]}
                tickFormatter={v => `${Math.round(v * 100)}%`}
              />
              <Tooltip content={<DriftTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="drift" radius={[3, 3, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={driftColour(entry.drift)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-600 mt-2">
            Threshold line at 0.15 — above this triggers full agent analysis
          </p>
        </div>
      )}

      {/* Status line */}
      <div className="flex items-center gap-3 text-xs text-gray-500 mb-4">
        {loading && <span className="animate-pulse">Loading scores…</span>}
        {error && <span className="text-red-400">Error: {error}</span>}
        {!loading && !error && scores && (
          <>
            <span>{scores.length} regulations</span>
            <span>·</span>
            <span className="text-yellow-400">
              {scores.filter(s => s.flagged_for_analysis).length} flagged
            </span>
          </>
        )}
      </div>

      {/* Feed rows */}
      <div className="space-y-3">
        {(scores ?? []).map(item => (
          <FeedRow
            key={item.score_id}
            item={item}
            expanded={expandedId === item.score_id}
            onToggle={() => setExpandedId(id => id === item.score_id ? null : item.score_id)}
          />
        ))}
        {!loading && !error && scores?.length === 0 && (
          <div className="text-center py-16 text-gray-600">
            <p className="text-4xl mb-3">🔍</p>
            <p>No change scores yet.</p>
            <p className="text-xs mt-1">Run the <code className="text-blue-400">change_detection</code> DAG to compute scores.</p>
          </div>
        )}
      </div>
    </div>
  )
}
