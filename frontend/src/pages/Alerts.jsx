/**
 * Alerts — Compliance intelligence page.
 *
 * Shows flagged regulations with:
 *  - Severity tier (Critical / High / Medium / Low) derived from real drift scores
 *  - "Proof: What Changed" panel — actual before/after sentence diffs from the API
 *  - Plain-English "What this means" explanation per regulation type
 *  - Causal ATT estimate when available
 *  - Recommended compliance actions specific to the regulation type
 *
 * Data sources:
 *   GET /api/v1/change-scores?flagged_only=true&limit=100
 *   GET /api/v1/regulations/{id}/diff                      (per-alert, lazy)
 *   GET /api/v1/causal/estimates                           (causal ATT)
 */
import { useState, useEffect } from 'react'
import {
  AlertTriangle, TrendingUp, Shield, Clock, ChevronDown, ChevronUp,
  RefreshCw, Bell, BellOff, FileText, Eye, ArrowRight,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'

// ── Severity helpers ──────────────────────────────────────────────────────────

function severityFromDrift(drift) {
  if (drift >= 0.55) return 'critical'
  if (drift >= 0.35) return 'high'
  if (drift >= 0.15) return 'medium'
  return 'low'
}

const SEVERITY_CONFIG = {
  critical: {
    label: 'Critical',
    bg: 'bg-red-500/10', border: 'border-red-500/30',
    badge: 'bg-red-500/20 text-red-300 border border-red-500/30',
    dot: 'bg-red-500', pulse: 'animate-pulse',
    action: 'Escalate to Chief Compliance Officer immediately — within 24 hours.',
  },
  high: {
    label: 'High',
    bg: 'bg-orange-500/10', border: 'border-orange-500/30',
    badge: 'bg-orange-500/20 text-orange-300 border border-orange-500/30',
    dot: 'bg-orange-500', pulse: '',
    action: 'Assign to senior compliance analyst — review within 48 hours.',
  },
  medium: {
    label: 'Medium',
    bg: 'bg-yellow-500/10', border: 'border-yellow-500/30',
    badge: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30',
    dot: 'bg-yellow-400', pulse: '',
    action: 'Add to weekly compliance agenda — review within 2 weeks.',
  },
  low: {
    label: 'Low',
    bg: 'bg-blue-500/10', border: 'border-blue-500/30',
    badge: 'bg-blue-500/20 text-blue-300 border border-blue-500/30',
    dot: 'bg-blue-500', pulse: '',
    action: 'Log for quarterly compliance review.',
  },
}

// ── Diff loader hook (fetches lazily when alert is expanded) ──────────────────
function useDiff(regulationId, enabled) {
  const [diff, setDiff]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  useEffect(() => {
    if (!enabled || !regulationId || diff) return
    setLoading(true)
    fetch(`/api/v1/regulations/${regulationId}/diff`)
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(d => { setDiff(d); setLoading(false) })
      .catch(e => { setError(String(e)); setLoading(false) })
  }, [enabled, regulationId])

  return { diff, loading, error }
}

// ── Diff display component ────────────────────────────────────────────────────

function DiffPanel({ regulationId, enabled }) {
  const { diff, loading, error } = useDiff(regulationId, enabled)

  if (!enabled) return null
  if (loading) return (
    <div className="mt-4 border-t border-gray-800 pt-4">
      <div className="flex items-center gap-2 text-xs text-gray-500 animate-pulse">
        <RefreshCw size={12} className="animate-spin" />
        Loading text evidence…
      </div>
    </div>
  )
  if (error) return (
    <div className="mt-4 border-t border-gray-800 pt-4 text-xs text-red-400">
      Could not load diff: {error}
    </div>
  )
  if (!diff) return null

  const highChanges = (diff.changes || []).filter(c => c.significance === 'high').slice(0, 4)
  const otherChanges = (diff.changes || []).filter(c => c.significance !== 'high').slice(0, 3)
  const showChanges = [...highChanges, ...otherChanges]

  return (
    <div className="mt-4 border-t border-gray-800 pt-4">
      {/* Plain-English explanation */}
      {diff.plain_english && (
        <div className="mb-4 p-3 bg-blue-500/5 border border-blue-500/20 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">💡</span>
            <span className="text-xs font-semibold text-blue-300">Plain-English Explanation</span>
          </div>
          <p className="text-sm text-gray-300 leading-relaxed">{diff.plain_english}</p>
        </div>
      )}

      {/* Change statistics */}
      <div className="flex items-center gap-4 mb-3 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <FileText size={11} />
          <span className="font-mono text-gray-400">{diff.pct_changed}%</span> of document changed
        </span>
        <span>·</span>
        <span>
          <span className="text-green-400 font-mono">{diff.stats?.added ?? 0}</span> added ·{' '}
          <span className="text-red-400 font-mono">{diff.stats?.removed ?? 0}</span> removed ·{' '}
          <span className="text-yellow-400 font-mono">{diff.stats?.changed ?? 0}</span> revised
        </span>
        {(diff.word_count_delta ?? 0) !== 0 && (
          <>
            <span>·</span>
            <span>
              <span className={diff.word_count_delta > 0 ? 'text-green-400' : 'text-red-400'}>
                {diff.word_count_delta > 0 ? '+' : ''}{diff.word_count_delta}
              </span>{' '}words
            </span>
          </>
        )}
      </div>

      {/* Text evidence — before / after */}
      {showChanges.length > 0 ? (
        <div className="space-y-3">
          <p className="text-xs text-gray-500 flex items-center gap-1">
            <Eye size={11} />
            Proof of change — actual regulatory text:
          </p>
          {showChanges.map((change, i) => (
            <DiffBlock key={i} change={change} />
          ))}
        </div>
      ) : (
        <p className="text-xs text-gray-600">No sentence-level changes detected.</p>
      )}
    </div>
  )
}

function DiffBlock({ change }) {
  const [expanded, setExpanded] = useState(change.significance === 'high')
  const typeLabel = change.type === 'added' ? 'ADDED' : change.type === 'removed' ? 'REMOVED' : 'REVISED'
  const typeColor = change.type === 'added' ? 'text-green-400' : change.type === 'removed' ? 'text-red-400' : 'text-yellow-400'

  return (
    <div
      className="border border-gray-800 rounded-lg overflow-hidden cursor-pointer"
      onClick={() => setExpanded(e => !e)}
    >
      <div className="px-3 py-2 bg-gray-900/60 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold font-mono ${typeColor}`}>{typeLabel}</span>
          {change.has_numeric_change && (
            <span className="text-xs bg-yellow-500/10 text-yellow-300 border border-yellow-500/20 px-1.5 rounded">
              📊 numeric change
            </span>
          )}
          {change.numbers_before?.length > 0 && change.numbers_after?.length > 0 && (
            <span className="text-xs text-gray-500 font-mono">
              {change.numbers_before[0]}
              <ArrowRight size={10} className="inline mx-1" />
              <span className={change.type === 'removed' ? 'text-red-400' : 'text-green-400'}>
                {change.numbers_after[0]}
              </span>
            </span>
          )}
        </div>
        <span className="text-gray-700">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <div className="p-3 space-y-2">
          {change.old_text && (
            <div>
              <p className="text-xs text-red-400 font-semibold mb-1">BEFORE (v1):</p>
              <p className="text-xs text-gray-400 bg-red-500/5 border border-red-500/15 rounded p-2 leading-relaxed">
                {change.old_text}
              </p>
            </div>
          )}
          {change.new_text && (
            <div>
              <p className="text-xs text-green-400 font-semibold mb-1">AFTER (v2):</p>
              <p className="text-xs text-gray-300 bg-green-500/5 border border-green-500/15 rounded p-2 leading-relaxed">
                {change.new_text}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Compliance action recommendations ─────────────────────────────────────────

function complianceActions(severity, regType) {
  const base = {
    capital: [
      'Recalculate risk-weighted assets under the new capital ratio requirements.',
      'Model the impact on dividend and share buyback capacity.',
      'Submit updated capital plan to the board risk committee.',
    ],
    liquidity: [
      'Rerun LCR calculations under the updated outflow rate assumptions.',
      'Assess impact on HQLA buffer composition.',
      'Update liquidity contingency funding plan.',
    ],
    stress_testing: [
      'Incorporate new scenario parameters into the next DFAST cycle.',
      'Model the cyber-operational scenario against current resilience posture.',
      'Update stress testing governance documentation.',
    ],
    consumer_protection: [
      'Review and update customer communication workflows.',
      'Audit digital communication platforms for compliance gaps.',
      'Brief front-line staff on new consumer protection obligations.',
    ],
    trading: [
      'Identify all covered fund exposures that fall under the new definitions.',
      'Engage legal counsel to assess divestiture timelines.',
      'Update trading desk approvals and IMA documentation.',
    ],
    aml: [
      'Update KYC/CDD procedures to reflect the new beneficial ownership threshold.',
      'Re-screen existing customers against the new 10% ownership threshold.',
      'Notify compliance technology vendors of the regulatory change.',
    ],
  }

  const severity_extras = {
    critical: ['Escalate immediately to CCO and General Counsel.', 'Convene emergency risk committee meeting.'],
    high: ['Assign dedicated compliance analyst to this regulation.', 'Set hard deadline: 48-hour preliminary assessment.'],
    medium: ['Add to next weekly compliance team meeting agenda.'],
    low: ['Log in regulatory change management system.'],
  }

  const typeActions = base[regType] ?? base.consumer_protection
  const sevActions = severity_extras[severity] ?? []

  return [...sevActions, ...typeActions]
}

// ── Metric tile ───────────────────────────────────────────────────────────────

function MetricTile({ label, value, sub, highlight }) {
  return (
    <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-3 text-center">
      <div className={`text-base font-bold font-mono ${highlight ? 'text-yellow-300' : 'text-white'}`}>{value}</div>
      <div className="text-xs font-medium text-gray-400 mt-0.5">{label}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5 leading-tight">{sub}</div>}
    </div>
  )
}

// ── Alert card ────────────────────────────────────────────────────────────────

function AlertCard({ item, causalMap, regTypeMap, onDismiss }) {
  const [expanded, setExpanded]     = useState(false)
  const [showDiff, setShowDiff]     = useState(false)
  const [showActions, setShowActions] = useState(false)

  const severity = severityFromDrift(item.drift_score ?? 0)
  const cfg      = SEVERITY_CONFIG[severity]
  const causal   = causalMap?.[item.regulation_id]
  const regType  = regTypeMap?.[item.regulation_id] ?? 'consumer_protection'
  const actions  = complianceActions(severity, regType)

  function timeSince(iso) {
    if (!iso) return '—'
    const s = (Date.now() - new Date(iso).getTime()) / 1000
    if (s < 3600) return `${Math.round(s / 60)}m ago`
    if (s < 86400) return `${Math.round(s / 3600)}h ago`
    return `${Math.round(s / 86400)}d ago`
  }

  const pct = Math.min(100, Math.round((item.drift_score ?? 0) * 100))
  const barColor = severity === 'critical' ? '#ef4444' : severity === 'high' ? '#f97316' : severity === 'medium' ? '#eab308' : '#3b82f6'

  return (
    <div className={`border rounded-xl ${cfg.bg} ${cfg.border}`}>
      {/* ── Summary header ── */}
      <div className="p-4 cursor-pointer" onClick={() => setExpanded(e => !e)}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <div className={`mt-1.5 w-2.5 h-2.5 rounded-full flex-shrink-0 ${cfg.dot} ${cfg.pulse}`} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap mb-1">
                <span className={`text-xs px-2 py-0.5 rounded font-semibold ${cfg.badge}`}>{cfg.label}</span>
                <span className="text-xs text-blue-400 font-mono">{item.document_number}</span>
                <span className="text-xs text-gray-600">·</span>
                <span className="text-xs text-gray-500">{item.agency}</span>
                <span className="text-xs text-gray-600">·</span>
                <span className="text-xs text-gray-600 flex items-center gap-1">
                  <Clock size={10} /> {timeSince(item.computed_at)}
                </span>
              </div>
              <p className="text-sm text-gray-200 leading-snug font-medium">{item.title}</p>
              <p className="text-xs text-gray-500 mt-1">{cfg.action}</p>
            </div>
          </div>

          <div className="flex flex-col items-end gap-1 flex-shrink-0">
            <div className="text-right">
              <div className="text-xl font-bold font-mono text-white leading-none">{pct}%</div>
              <div className="text-xs text-gray-600">drift</div>
            </div>
            {expanded ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
          </div>
        </div>

        {/* Drift bar */}
        <div className="mt-3 pl-5">
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1.5 rounded bg-gray-800 overflow-hidden">
              <div className="h-full rounded" style={{ width: `${pct}%`, backgroundColor: barColor }} />
            </div>
            <span className="text-xs font-mono text-gray-500 w-9 text-right">{pct}%</span>
          </div>
        </div>
      </div>

      {/* ── Expanded detail ── */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-gray-800/60">
          {/* Metric tiles */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 mt-4">
            <MetricTile
              label="Drift score"
              value={item.drift_display ?? `${(item.drift_score ?? 0).toFixed(2)}`}
              sub="Bootstrap 95% CI"
              highlight={item.drift_score >= 0.35}
            />
            <MetricTile
              label="JSD"
              value={item.jsd_score != null ? `${(item.jsd_score * 100).toFixed(1)}%` : '—'}
              sub={item.jsd_significant ? '✓ Significant p<0.05' : 'Not significant'}
              highlight={item.jsd_significant}
            />
            <MetricTile
              label="Wasserstein"
              value={item.wasserstein_score != null ? `${(item.wasserstein_score * 100).toFixed(1)}%` : '—'}
              sub="Word distribution shift"
            />
            <MetricTile
              label="Composite"
              value={`${(item.composite_score * 100).toFixed(1)}%`}
              sub="0.5×drift + 0.3×JSD + 0.2×W2"
              highlight={item.composite_score >= 0.3}
            />
          </div>

          {/* Causal estimate */}
          {causal && (
            <div className="mt-4 bg-gray-900/60 border border-gray-800 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp size={13} className="text-blue-400" />
                <span className="text-xs font-semibold text-gray-300">Causal Impact Estimate</span>
                <span className="text-xs text-gray-600 ml-auto font-mono">{causal.method?.toUpperCase()}</span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                  <div className="text-sm font-mono font-bold text-white">
                    {(causal.att_estimate ?? causal.att) != null
                      ? `${((causal.att_estimate ?? causal.att) * 100).toFixed(2)}%` : '—'}
                  </div>
                  <div className="text-xs text-gray-600">ATT estimate</div>
                </div>
                <div>
                  <div className={`text-sm font-mono font-bold ${(causal.p_value ?? causal.placebo_p_value) < 0.05 ? 'text-yellow-300' : 'text-white'}`}>
                    {(causal.p_value ?? causal.placebo_p_value)?.toFixed(4) ?? '—'}
                  </div>
                  <div className="text-xs text-gray-600">
                    {(causal.p_value ?? causal.placebo_p_value) < 0.05 ? '✓ Significant' : 'p-value'}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-mono font-bold text-white">
                    {causal.ci_low_95 != null
                      ? `[${(causal.ci_low_95 * 100).toFixed(2)}%, ${(causal.ci_high_95 * 100).toFixed(2)}%]`
                      : '—'}
                  </div>
                  <div className="text-xs text-gray-600">95% CI</div>
                </div>
              </div>
              {causal.outcome_variable && (
                <p className="text-xs text-gray-600 mt-2">
                  Outcome: <span className="font-mono text-gray-400">{causal.outcome_variable.replace(/_/g, ' ')}</span>
                </p>
              )}
            </div>
          )}

          {/* Proof: What Changed */}
          <div className="mt-4">
            <button
              onClick={e => { e.stopPropagation(); setShowDiff(d => !d) }}
              className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded border transition-colors w-full ${
                showDiff
                  ? 'bg-purple-500/15 text-purple-300 border-purple-500/30'
                  : 'bg-gray-900 text-gray-400 border-gray-700 hover:text-gray-200'
              }`}
            >
              <FileText size={12} />
              {showDiff ? 'Hide' : 'Show'} proof — What actually changed in the text
              <span className="ml-auto text-gray-600">
                {showDiff ? '▲' : '▼'}
              </span>
            </button>
            <DiffPanel regulationId={item.regulation_id} enabled={showDiff} />
          </div>

          {/* Recommended actions */}
          <div className="mt-3">
            <button
              onClick={e => { e.stopPropagation(); setShowActions(a => !a) }}
              className="flex items-center gap-2 text-xs px-3 py-1.5 rounded border bg-gray-900 text-gray-400 border-gray-700 hover:text-gray-200 transition-colors w-full"
            >
              <Shield size={12} />
              {showActions ? 'Hide' : 'Show'} recommended compliance actions
              <span className="ml-auto">{showActions ? '▲' : '▼'}</span>
            </button>
            {showActions && (
              <div className="mt-2 bg-gray-900/60 border border-gray-800 rounded-lg p-3">
                <ul className="space-y-2">
                  {actions.map((action, i) => (
                    <li key={i} className="text-xs text-gray-400 flex items-start gap-2">
                      <span className="text-gray-600 mt-0.5 font-mono text-xs">{i + 1}.</span>
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Dismiss */}
          <div className="mt-3 flex justify-end">
            <button
              onClick={e => { e.stopPropagation(); onDismiss(item.score_id) }}
              className="flex items-center gap-1.5 text-xs text-gray-600 hover:text-gray-400 transition-colors"
            >
              <BellOff size={11} /> Dismiss alert
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Stats bar ─────────────────────────────────────────────────────────────────

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
            <span className={`text-xs mt-1 px-2 py-0.5 rounded inline-block font-semibold ${cfg.badge}`}>
              {cfg.label}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Alerts() {
  const [dismissed, setDismissed]       = useState(new Set())
  const [filterSeverity, setFilterSeverity] = useState('all')

  const { data: scores, loading, error, refetch } = useApi('/change-scores', {
    flagged_only: true,
    limit: 100,
  })
  const { data: regulations } = useApi('/regulations', { page_size: 100 })
  const { data: causalList }  = useApi('/causal/estimates', { limit: 200 })

  // Build regulation_id → regulation_type map for action recommendations
  const regTypeMap = {}
  for (const reg of regulations?.items ?? []) {
    regTypeMap[reg.id] = reg.regulation_type
  }

  // Build regulation_id → causal estimate map
  const causalMap = {}
  for (const c of causalList ?? []) {
    if (!causalMap[c.regulation_id]) causalMap[c.regulation_id] = c
  }

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
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-white">Compliance Alerts</h1>
            {activeCount > 0 && (
              <span className="bg-red-500 text-white text-xs font-bold px-2 py-0.5 rounded-full animate-pulse">
                {activeCount}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mt-0.5">
            Flagged regulations — sorted by semantic drift. Every alert shows proof of what changed.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Bell size={15} className="text-yellow-400" />
          <button
            onClick={refetch}
            className="p-1.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Stats */}
      {!loading && !error && allAlerts.length > 0 && (
        <StatsBar alerts={allAlerts.filter(a => !dismissed.has(a.score_id))} />
      )}

      {/* Severity filter */}
      <div className="flex items-center gap-2 mb-5 flex-wrap">
        {['all', 'critical', 'high', 'medium', 'low'].map(sev => (
          <button
            key={sev}
            onClick={() => setFilterSeverity(sev)}
            className={`text-xs px-3 py-1.5 rounded-md border transition-colors capitalize font-medium ${
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

      {/* State: loading / error / empty */}
      {loading && (
        <div className="flex items-center justify-center py-16 text-gray-600">
          <RefreshCw size={16} className="animate-spin mr-2" />
          <span className="text-sm">Loading alerts…</span>
        </div>
      )}
      {error && (
        <div className="text-center py-12 text-red-400 text-sm">Failed to load: {error}</div>
      )}
      {!loading && !error && filteredAlerts.length === 0 && (
        <div className="text-center py-16 text-gray-600">
          <p className="text-4xl mb-3">✅</p>
          <p className="text-sm">
            {allAlerts.length === 0
              ? 'No flagged regulations yet.'
              : filterSeverity !== 'all'
              ? `No ${filterSeverity} severity alerts.`
              : 'All alerts dismissed.'}
          </p>
        </div>
      )}

      {/* Alert cards */}
      <div className="space-y-3">
        {filteredAlerts.map(item => (
          <AlertCard
            key={item.score_id}
            item={item}
            causalMap={causalMap}
            regTypeMap={regTypeMap}
            onDismiss={id => setDismissed(s => new Set([...s, id]))}
          />
        ))}
      </div>

      {/* Footer — how alerts work */}
      {!loading && filteredAlerts.length > 0 && (
        <div className="mt-10 p-5 bg-gray-900 border border-gray-800 rounded-xl text-xs text-gray-600 leading-relaxed">
          <p className="font-semibold text-gray-400 mb-2">📐 How alert severity is determined</p>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-4 text-center">
            {[
              { sev: 'critical', range: '≥ 55% drift' },
              { sev: 'high',     range: '35–54% drift' },
              { sev: 'medium',   range: '15–34% drift' },
              { sev: 'low',      range: '< 15% drift' },
            ].map(({ sev, range }) => (
              <div key={sev} className={`border rounded-lg p-2 ${SEVERITY_CONFIG[sev].bg} ${SEVERITY_CONFIG[sev].border}`}>
                <span className={`text-xs font-semibold ${SEVERITY_CONFIG[sev].badge} px-2 py-0.5 rounded`}>
                  {SEVERITY_CONFIG[sev].label}
                </span>
                <p className="text-gray-500 mt-1 text-xs">{range}</p>
              </div>
            ))}
          </div>
          <p>
            <span className="text-gray-400">Semantic drift</span> measures how much the regulatory text changed
            in meaning (TF-IDF cosine distance with bootstrap 95% CI).{' '}
            <span className="text-gray-400">JSD</span> (Jensen-Shannon divergence) catches vocabulary structure changes.{' '}
            <span className="text-gray-400">Wasserstein distance</span> detects reorganisation — same words in different structure.
            The composite score weights all three: 50% drift + 30% JSD + 20% Wasserstein.
            Every alert includes the actual text that changed, so compliance teams have evidence, not just scores.
          </p>
        </div>
      )}
    </div>
  )
}
