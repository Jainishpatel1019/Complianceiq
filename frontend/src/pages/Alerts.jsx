/**
 * Alerts — Phase 5 dashboard page.
 *
 * Shows flagged regulations that require immediate compliance review,
 * grouped by severity and enriched with causal impact estimates.
 *
 * Data sources:
 *   GET /api/v1/change-scores?flagged_only=true&limit=100
 *   GET /api/v1/causal          (impact estimates per regulation)
 */
import { useState } from 'react'
import {
  AlertTriangle, TrendingUp, Shield, Clock,
  ChevronDown, ChevronUp, RefreshCw, Bell, BellOff,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'

// ── Helpers ───────────────────────────────────────────────────────────────────

function severityFromDrift(drift) {
  if (drift >= 0.55) return 'critical'
  if (drift >= 0.35) return 'high'
  if (drift >= 0.15) return 'medium'
  return 'low'
}

const SEVERITY_CONFIG = {
  critical: {
    label: 'Critical',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    badge: 'bg-red-500/20 text-red-300 border border-red-500/30',
    dot: 'bg-red-500',
    icon: '🔴',
    description: 'Immediate action required — regulatory change is statistically significant and exceeds all drift thresholds.',
  },
  high: {
    label: 'High',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    badge: 'bg-orange-500/20 text-orange-300 border border-orange-500/30',
    dot: 'bg-orange-500',
    icon: '🟠',
    description: 'Review within 48 hours — substantial semantic drift detected with statistically significant JSD.',
  },
  medium: {
    label: 'Medium',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    badge: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
    dot: 'bg-yellow-500',
    icon: '🟡',
    description: 'Monitor closely — moderate regulatory change, may affect specific business lines.',
  },
  low: {
    label: 'Low',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
    badge: 'bg-blue-500/20 text-blue-300 border border-blue-500/30',
    dot: 'bg-blue-500',
    icon: '🔵',
    description: 'Informational — flagged for tracking. No immediate compliance action needed.',
  },
}

function driftBar(score) {
  const pct = Math.min(100, Math.round((score ?? 0) * 100))
  const color = score >= 0.55 ? '#ef4444' : score >= 0.35 ? '#f97316' : score >= 0.15 ? '#eab308' : '#3b82f6'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded bg-gray-800 overflow-hidden">
        <div className="h-full rounded" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono text-gray-400 w-9 text-right">{pct}%</span>
    </div>
  )
}

function timeSince(isoString) {
  if (!isoString) return 'Unknown'
  const d = new Date(isoString)
  const diff = (Date.now() - d.getTime()) / 1000
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`
  return `${Math.round(diff / 86400)}d ago`
}

// ── Alert card ─────────────────────────────────────────────────────────────────

function AlertCard({ item, causalMap, dismissed, onDismiss }) {
  const [expanded, setExpanded] = useState(false)
  const severity = severityFromDrift(item.drift_score ?? 0)
  const cfg = SEVERITY_CONFIG[severity]
  const causal = causalMap?.[item.regulation_id]

  if (dismissed) return null

  return (
    <div className={`border rounded-xl ${cfg.bg} ${cfg.border} transition-all`}>
      {/* Header */}
      <div
        className="p-4 cursor-pointer"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-start justify-between gap-3">
          {/* Left: severity dot + meta */}
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className="mt-1.5 flex-shrink-0">
              <div className={`w-2.5 h-2.5 rounded-full ${cfg.dot} shadow-lg`} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <span className={`text-xs px-2 py-0.5 rounded font-medium ${cfg.badge}`}>
                  {cfg.label}
                </span>
                <span className="text-xs text-blue-400 font-mono">{item.document_number}</span>
                <span className="text-xs text-gray-600">·</span>
                <span className="text-xs text-gray-500">{item.agency}</span>
                <span className="text-xs text-gray-600">·</span>
                <span className="text-xs text-gray-600 flex items-center gap-1">
                  <Clock size={10} />
                  {timeSince(item.computed_at)}
                </span>
              </div>
              <p className="text-sm text-gray-200 leading-snug">{item.title}</p>

              {/* Plain-English summary */}
              <p className="text-xs text-gray-500 mt-1.5 leading-relaxed">
                {cfg.description}
              </p>
            </div>
          </div>

          {/* Right: drift score + expand */}
          <div className="flex flex-col items-end gap-2 flex-shrink-0">
            <div className="text-right">
              <div className="text-lg font-bold font-mono text-white leading-none">
                {Math.round((item.drift_score ?? 0) * 100)}%
              </div>
              <div className="text-xs text-gray-600">semantic drift</div>
            </div>
            {expanded ? (
              <ChevronUp size={14} className="text-gray-500" />
            ) : (
              <ChevronDown size={14} className="text-gray-500" />
            )}
          </div>
        </div>

        {/* Drift bar */}
        <div className="mt-3 pl-5">
          {driftBar(item.drift_score)}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-4 pb-4 pt-0 border-t border-gray-800/60">
          <div className="pt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {/* Metric tiles */}
            <MetricTile
              label="Drift score"
              value={item.drift_display ?? `${(item.drift_score ?? 0).toFixed(2)}`}
              sub="95% confidence interval"
              highlight={item.drift_score >= 0.35}
            />
            <MetricTile
              label="JSD"
              value={item.jsd_score != null ? `${(item.jsd_score * 100).toFixed(1)}%` : '—'}
              sub={item.jsd_significant ? 'Statistically significant (p<0.05)' : 'Not significant'}
              highlight={item.jsd_significant}
            />
            <MetricTile
              label="Wasserstein"
              value={item.wasserstein_score != null ? `${(item.wasserstein_score * 100).toFixed(1)}%` : '—'}
              sub="Distribution distance"
            />
            <MetricTile
              label="Composite"
              value={`${(item.composite_score * 100).toFixed(1)}%`}
              sub="Weighted aggregate score"
              highlight={item.composite_score >= 0.3}
            />
          </div>

          {/* Causal impact if available */}
          {causal && (
            <div className="mt-4 bg-gray-900/60 border border-gray-800 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp size={13} className="text-blue-400" />
                <span className="text-xs font-medium text-gray-300">Causal Impact Estimate</span>
                <span className="text-xs text-gray-600 ml-auto">{causal.method?.toUpperCase()}</span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                  <div className="text-sm font-mono font-bold text-white">
                    {(causal.att_estimate ?? causal.att) != null
                      ? ((causal.att_estimate ?? causal.att) * 100).toFixed(2) + '%'
                      : '—'}
                  </div>
                  <div className="text-xs text-gray-600">ATT estimate</div>
                </div>
                <div>
                  <div className="text-sm font-mono font-bold text-white">
                    {(causal.p_value ?? causal.placebo_p_value) != null
                      ? (causal.p_value ?? causal.placebo_p_value).toFixed(3)
                      : '—'}
                  </div>
                  <div className={`text-xs ${(causal.p_value ?? causal.placebo_p_value) < 0.05 ? 'text-yellow-400' : 'text-gray-600'}`}>
                    {(causal.p_value ?? causal.placebo_p_value) < 0.05 ? 'p-value (significant)' : 'p-value'}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-mono font-bold text-white">
                    {causal.ci_low_95 != null
                      ? `[${(causal.ci_low_95 * 100).toFixed(1)}%, ${(causal.ci_high_95 * 100).toFixed(1)}%]`
                      : '—'}
                  </div>
                  <div className="text-xs text-gray-600">95% CI</div>
                </div>
              </div>
              {causal.outcome_variable && (
                <p className="text-xs text-gray-600 mt-2">
                  Outcome variable: <span className="text-gray-400 font-mono">{causal.outcome_variable}</span>
                </p>
              )}
            </div>
          )}

          {/* Recommended actions */}
          <div className="mt-4 bg-gray-900/60 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Shield size={13} className="text-green-400" />
              <span className="text-xs font-medium text-gray-300">Recommended Actions</span>
            </div>
            <ul className="space-y-1">
              {recommendedActions(severity, item).map((action, i) => (
                <li key={i} className="text-xs text-gray-400 flex items-start gap-2">
                  <span className="text-gray-600 mt-0.5">•</span>
                  {action}
                </li>
              ))}
            </ul>
          </div>

          {/* Dismiss button */}
          <div className="mt-3 flex justify-end">
            <button
              onClick={(e) => { e.stopPropagation(); onDismiss(item.score_id) }}
              className="flex items-center gap-1.5 text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              <BellOff size={12} />
              Dismiss alert
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricTile({ label, value, sub, highlight }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-3 text-center">
      <div className={`text-base font-bold font-mono ${highlight ? 'text-yellow-300' : 'text-white'}`}>
        {value}
      </div>
      <div className="text-xs font-medium text-gray-400 mt-0.5">{label}</div>
      <div className="text-xs text-gray-600 mt-0.5 leading-tight">{sub}</div>
    </div>
  )
}

function recommendedActions(severity, item) {
  const base = [
    `Review the full text diff for ${item.document_number} with legal/compliance team.`,
    `Update internal compliance playbook to reflect changes in ${item.agency} guidance.`,
  ]
  if (severity === 'critical') return [
    'Escalate to Chief Compliance Officer immediately.',
    'Schedule emergency review meeting within 24 hours.',
    ...base,
    'Assess capital reserve implications and report to risk committee.',
  ]
  if (severity === 'high') return [
    'Assign to senior compliance analyst for priority review.',
    ...base,
    'Check affected business lines and notify relevant department heads.',
  ]
  if (severity === 'medium') return [
    ...base,
    'Add to weekly compliance review agenda.',
    'Confirm no impact on upcoming regulatory filings.',
  ]
  return [
    'Log for quarterly compliance review.',
    'Monitor for subsequent amendments to this regulation.',
  ]
}

// ── Summary stats bar ─────────────────────────────────────────────────────────

function StatsBar({ alerts }) {
  const counts = { critical: 0, high: 0, medium: 0, low: 0 }
  for (const a of alerts) counts[severityFromDrift(a.drift_score ?? 0)]++
  return (
    <div className="grid grid-cols-4 gap-3 mb-6">
      {Object.entries(counts).map(([sev, count]) => {
        const cfg = SEVERITY_CONFIG[sev]
        return (
          <div key={sev} className={`border rounded-xl p-4 text-center ${cfg.bg} ${cfg.border}`}>
            <div className="text-2xl font-bold text-white">{count}</div>
            <div className={`text-xs mt-1 px-2 py-0.5 rounded inline-block font-medium ${cfg.badge}`}>
              {cfg.label}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Alerts() {
  const [dismissed, setDismissed] = useState(new Set())
  const [filterSeverity, setFilterSeverity] = useState('all')

  const { data: scores, loading, error, refetch } = useApi('/change-scores', {
    flagged_only: true,
    limit: 100,
  })

  const { data: causalList } = useApi('/causal/estimates', { limit: 200 })

  // Build regulation_id → causal map
  const causalMap = {}
  for (const c of causalList ?? []) {
    if (!causalMap[c.regulation_id]) causalMap[c.regulation_id] = c
  }

  // Sort by drift desc
  const allAlerts = (scores ?? []).sort((a, b) => (b.drift_score ?? 0) - (a.drift_score ?? 0))

  const filteredAlerts = allAlerts.filter(a => {
    if (dismissed.has(a.score_id)) return false
    if (filterSeverity === 'all') return true
    return severityFromDrift(a.drift_score ?? 0) === filterSeverity
  })

  const activeCount = allAlerts.filter(a => !dismissed.has(a.score_id)).length

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-white">Compliance Alerts</h1>
            {activeCount > 0 && (
              <span className="bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded-full">
                {activeCount}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mt-0.5">
            Flagged regulations requiring compliance review — sorted by drift severity
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Bell size={16} className="text-yellow-400" />
          <button
            onClick={refetch}
            className="p-1.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700"
            title="Refresh"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Stats bar */}
      {!loading && !error && allAlerts.length > 0 && (
        <StatsBar alerts={allAlerts.filter(a => !dismissed.has(a.score_id))} />
      )}

      {/* Severity filter tabs */}
      <div className="flex items-center gap-2 mb-5">
        {['all', 'critical', 'high', 'medium', 'low'].map(sev => (
          <button
            key={sev}
            onClick={() => setFilterSeverity(sev)}
            className={`text-xs px-3 py-1.5 rounded-md border transition-colors capitalize ${
              filterSeverity === sev
                ? sev === 'all'
                  ? 'bg-blue-600/20 text-blue-300 border-blue-500/40'
                  : `${SEVERITY_CONFIG[sev]?.badge}`
                : 'bg-gray-900 text-gray-500 border-gray-800 hover:text-gray-300'
            }`}
          >
            {sev === 'all' ? 'All alerts' : SEVERITY_CONFIG[sev].label}
          </button>
        ))}
        {dismissed.size > 0 && (
          <button
            onClick={() => setDismissed(new Set())}
            className="ml-auto text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Restore {dismissed.size} dismissed
          </button>
        )}
      </div>

      {/* State messages */}
      {loading && (
        <div className="flex items-center justify-center py-16 text-gray-600">
          <RefreshCw size={16} className="animate-spin mr-2" />
          <span className="text-sm">Loading alerts…</span>
        </div>
      )}
      {error && (
        <div className="text-center py-12 text-red-400 text-sm">
          Failed to load alerts: {error}
        </div>
      )}
      {!loading && !error && filteredAlerts.length === 0 && (
        <div className="text-center py-16 text-gray-600">
          <p className="text-4xl mb-3">✅</p>
          <p className="text-sm">
            {allAlerts.length === 0
              ? 'No flagged regulations. Run the change detection pipeline to compute scores.'
              : filterSeverity !== 'all'
              ? `No ${filterSeverity} severity alerts.`
              : 'All alerts dismissed.'}
          </p>
          {allAlerts.length === 0 && (
            <p className="text-xs mt-1 text-gray-700">
              Alerts appear when <code className="text-blue-400">flagged_for_analysis = true</code> in the change score.
            </p>
          )}
        </div>
      )}

      {/* Alert cards */}
      <div className="space-y-3">
        {filteredAlerts.map(item => (
          <AlertCard
            key={item.score_id}
            item={item}
            causalMap={causalMap}
            dismissed={false}
            onDismiss={id => setDismissed(s => new Set([...s, id]))}
          />
        ))}
      </div>

      {/* Footer explainer */}
      {!loading && filteredAlerts.length > 0 && (
        <div className="mt-8 p-4 bg-gray-900 border border-gray-800 rounded-xl text-xs text-gray-600 leading-relaxed">
          <p className="font-medium text-gray-400 mb-1">How alerts are generated</p>
          <p>
            ComplianceIQ monitors regulatory changes across the Federal Register and SEC.
            When a new regulation version is published, the system computes three drift measures:
            <span className="text-gray-400"> semantic drift</span> (TF-IDF cosine distance between versions),
            <span className="text-gray-400"> Jensen-Shannon divergence</span> (topic distribution shift), and
            <span className="text-gray-400"> Wasserstein distance</span> (word distribution difference).
            Regulations exceeding the composite threshold of 0.15 are flagged and appear here.
            Causal impact estimates (ATT) quantify the downstream effect on capital requirements.
          </p>
        </div>
      )}
    </div>
  )
}
