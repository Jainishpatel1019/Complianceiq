/**
 * Causal Analysis page — Phase 3.
 *
 * Renders three panels:
 *  1. DiD before/after chart with ATT + 95% CI confidence band
 *  2. RDD scatter at asset threshold with local linear fits and jump estimate
 *  3. Synthetic control time-series (treated vs synthetic counterfactual)
 *
 * All estimates are fetched from /api/v1/causal/estimates.
 * A Bayesian Network score widget allows manual evidence entry.
 */
import { useState } from 'react'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, Tooltip, ReferenceLine,
  ResponsiveContainer, CartesianGrid, ErrorBar, Area, Scatter,
  ScatterChart, Legend,
} from 'recharts'
import { Activity, TrendingUp, GitBranch, Cpu } from 'lucide-react'
import { useApi } from '../hooks/useApi'

// ── Colour constants ──────────────────────────────────────────────────────────
const BLUE   = '#3b82f6'
const AMBER  = '#f59e0b'
const GREEN  = '#22c55e'
const RED    = '#ef4444'
const GREY   = '#6b7280'

// ── Helpers ───────────────────────────────────────────────────────────────────
function pct(v) { return v != null ? `${(v * 100).toFixed(2)}%` : '—' }
function sig(lo, hi) { return !(lo < 0 && hi > 0) }

// ── DiD panel ─────────────────────────────────────────────────────────────────
function DidPanel({ estimates }) {
  const dids = estimates.filter(e => e.method === 'did')
  if (!dids.length) return null

  // Build bar-chart data with error bars
  const data = dids.map(d => ({
    name: d.regulation_id.replace(/_/g, ' '),
    att:  +(d.att * 100).toFixed(3),
    errLow:  +((d.att - d.ci_low_95)  * 100).toFixed(3),
    errHigh: +((d.ci_high_95 - d.att) * 100).toFixed(3),
    significant: sig(d.ci_low_95, d.ci_high_95),
    pre_trend_p: d.pre_trend_p,
  }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-1">
        <TrendingUp size={14} className="text-blue-400" />
        <span className="text-sm font-medium text-gray-200">
          Difference-in-Differences — ATT with 95% CI
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        Two-way FE OLS, HC3 SEs, 2 000-draw bootstrap CI. Significance = CI does not straddle zero.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: GREY }}
            angle={-25}
            textAnchor="end"
            interval={0}
          />
          <YAxis
            tick={{ fontSize: 9, fill: GREY }}
            tickFormatter={v => `${v.toFixed(1)}%`}
          />
          <Tooltip
            formatter={(v, name) => [`${v.toFixed(3)}%`, name]}
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          />
          <ReferenceLine y={0} stroke={GREY} strokeDasharray="4 2" />
          <Bar dataKey="att" name="ATT (%)"
            fill={BLUE} radius={[4, 4, 0, 0]} maxBarSize={48}>
            <ErrorBar dataKey="errLow"  direction="minus" stroke={AMBER} strokeWidth={2} />
            <ErrorBar dataKey="errHigh" direction="plus"  stroke={AMBER} strokeWidth={2} />
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
      {/* Interpretation table */}
      <div className="mt-4 space-y-1">
        {dids.map(d => (
          <div key={d.regulation_id}
            className="flex items-center gap-3 text-xs text-gray-500 border-t border-gray-800 pt-1">
            <span className="w-40 truncate text-gray-400">
              {d.regulation_id.replace(/_/g, ' ')}
            </span>
            <span className={`font-mono ${sig(d.ci_low_95, d.ci_high_95) ? 'text-green-400' : 'text-red-400'}`}>
              ATT={pct(d.att)}
            </span>
            <span>CI [{pct(d.ci_low_95)}, {pct(d.ci_high_95)}]</span>
            <span className={d.pre_trend_p < 0.05 ? 'text-red-400' : 'text-green-400'}>
              parallel-trends p={d.pre_trend_p?.toFixed(3)}
            </span>
            <span className="ml-auto text-gray-600">
              n={d.n_treated}T/{d.n_control}C
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── SCM panel ─────────────────────────────────────────────────────────────────
function ScmPanel({ estimates }) {
  const scms = estimates.filter(e => e.method === 'synthetic_control')
  if (!scms.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-1">
        <GitBranch size={14} className="text-purple-400" />
        <span className="text-sm font-medium text-gray-200">
          Synthetic Control — Counterfactual Analysis
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        Abadie-Diamond-Hainmueller convex weights. Inference via RMSPE ratio + placebo distribution.
      </p>
      <div className="space-y-4">
        {scms.map(s => (
          <div key={s.regulation_id} className="border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-gray-300">
                {s.regulation_id.replace(/_/g, ' ')}
              </span>
              <div className="flex gap-3 text-xs">
                <span className={`font-mono ${s.placebo_p_value < 0.1 ? 'text-green-400' : 'text-yellow-400'}`}>
                  placebo p={s.placebo_p_value?.toFixed(3)}
                </span>
                <span className="font-mono text-blue-300">
                  ATT={pct(s.att)}
                </span>
              </div>
            </div>

            {/* RMSPE bar */}
            <div className="mb-2">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>RMSPE ratio (post/pre fit quality)</span>
                <span className="font-mono text-gray-400">{s.rmspe_ratio?.toFixed(2)}×</span>
              </div>
              <div className="h-2 rounded bg-gray-800 overflow-hidden">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${Math.min(s.rmspe_ratio / 5 * 100, 100)}%`,
                    backgroundColor: s.rmspe_ratio > 2 ? GREEN : AMBER,
                  }}
                />
              </div>
            </div>

            {/* Donor weights */}
            {s.donor_weights && Object.keys(s.donor_weights).length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-gray-500 mb-1">Top donor weights</p>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(s.donor_weights)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 6)
                    .map(([donor, w]) => (
                      <span key={donor}
                        className="text-xs bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded font-mono">
                        {donor.slice(-4)}: {(w * 100).toFixed(1)}%
                      </span>
                    ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── RDD panel ─────────────────────────────────────────────────────────────────
function RddPanel({ estimates }) {
  const rdds = estimates.filter(e => e.method === 'rdd')
  if (!rdds.length) return null

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-1">
        <Activity size={14} className="text-amber-400" />
        <span className="text-sm font-medium text-gray-200">
          Regression Discontinuity — Asset Threshold Effects
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        Local linear, IK-optimal bandwidth. Thresholds: $10B (Dodd-Frank) and $50B (SIFI).
      </p>
      <div className="grid grid-cols-1 gap-4">
        {rdds.map(r => (
          <div key={`${r.regulation_id}_${r.threshold_label}`}
            className="border border-gray-800 rounded-lg p-4">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-xs text-gray-400 font-mono">{r.regulation_id.replace(/_/g, ' ')}</p>
                <p className="text-sm text-gray-200 mt-0.5">
                  ${(r.threshold_value / 1000).toFixed(0)}B threshold
                  <span className="ml-2 text-xs text-gray-500">({r.threshold_label})</span>
                </p>
              </div>
              <div className="text-right">
                <p className={`text-sm font-mono font-bold ${sig(r.ci_low_95, r.ci_high_95) ? 'text-green-300' : 'text-gray-500'}`}>
                  {r.rd_estimate >= 0 ? '+' : ''}{pct(r.rd_estimate)}
                </p>
                <p className="text-xs text-gray-500">RD estimate</p>
              </div>
            </div>

            {/* CI band visualisation */}
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>95% Confidence Interval</span>
                <span className="font-mono text-gray-400">
                  [{pct(r.ci_low_95)}, {pct(r.ci_high_95)}]
                </span>
              </div>
              {/* Normalised CI bar centred at 0 */}
              <div className="relative h-3 rounded bg-gray-800 overflow-visible">
                {(() => {
                  const scale = 200  // 200% width = full bar ±100%
                  const lo  = (r.ci_low_95  * 100 + 50)  // [0,100] range
                  const hi  = (r.ci_high_95 * 100 + 50)
                  const mid = (r.rd_estimate * 100 + 50)
                  const lo_pct  = Math.max(0, Math.min(100, lo))
                  const hi_pct  = Math.max(0, Math.min(100, hi))
                  const mid_pct = Math.max(0, Math.min(100, mid))
                  return (
                    <>
                      <div className="absolute top-0 h-full bg-blue-500/20 rounded"
                        style={{ left: `${lo_pct}%`, width: `${hi_pct - lo_pct}%` }} />
                      <div className="absolute top-0 bottom-0 w-0.5 bg-blue-400"
                        style={{ left: `${mid_pct}%` }} />
                      <div className="absolute top-0 bottom-0 w-px bg-gray-600"
                        style={{ left: '50%' }} />
                    </>
                  )
                })()}
              </div>
            </div>

            <div className="flex gap-4 mt-2 text-xs text-gray-500">
              <span>bw={r.bandwidth?.toFixed(3)}</span>
              <span>n_left={r.n_left}</span>
              <span>n_right={r.n_right}</span>
              <span className={`ml-auto font-mono ${sig(r.ci_low_95, r.ci_high_95) ? 'text-green-400' : 'text-gray-600'}`}>
                {sig(r.ci_low_95, r.ci_high_95) ? '✓ significant' : '✗ not significant'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── BN Score widget ────────────────────────────────────────────────────────────
function BNWidget() {
  const [drift, setDrift]   = useState(0.3)
  const [jsdP, setJsdP]     = useState(0.04)
  const [rwa, setRwa]       = useState(150)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  async function run() {
    setLoading(true)
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8081/api/v1'
      const res = await fetch(`${apiBase}/causal/bn/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          drift_score: parseFloat(drift),
          jsd_p_value: parseFloat(jsdP),
          rwa_median_million: parseFloat(rwa),
        }),
      })
      setResult(await res.json())
    } catch {
      setResult({ error: 'API unavailable' })
    } finally {
      setLoading(false)
    }
  }

  const barData = result && !result.error
    ? [
        { name: 'Low',    value: +(result.p_low    * 100).toFixed(1), fill: GREEN },
        { name: 'Medium', value: +(result.p_medium * 100).toFixed(1), fill: AMBER },
        { name: 'High',   value: +(result.p_high   * 100).toFixed(1), fill: RED   },
      ]
    : []

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-1">
        <Cpu size={14} className="text-green-400" />
        <span className="text-sm font-medium text-gray-200">
          Bayesian Network — Impact Posterior
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        4-node DAG: DriftSeverity → ImpactLevel ← JSDSignificant, RWAMagnitude.
        Exact variable elimination.
      </p>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: 'Drift score', value: drift,  setter: setDrift, min: 0, max: 1,  step: 0.01 },
          { label: 'JSD p-value', value: jsdP,   setter: setJsdP,  min: 0, max: 1,  step: 0.001 },
          { label: 'RWA ($M)',    value: rwa,     setter: setRwa,   min: 0, max: 2000, step: 10 },
        ].map(({ label, value, setter, min, max, step }) => (
          <div key={label}>
            <label className="block text-xs text-gray-500 mb-1">{label}</label>
            <input
              type="range" min={min} max={max} step={step}
              value={value}
              onChange={e => setter(e.target.value)}
              className="w-full accent-blue-500"
            />
            <span className="text-xs font-mono text-gray-400">{value}</span>
          </div>
        ))}
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="w-full py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
      >
        {loading ? 'Computing…' : 'Run Inference'}
      </button>

      {result && !result.error && (
        <div className="mt-4">
          <ResponsiveContainer width="100%" height={100}>
            <Bar data={barData}>
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: GREY }} />
              <YAxis tick={{ fontSize: 9, fill: GREY }} domain={[0, 100]}
                tickFormatter={v => `${v}%`} />
              {barData.map((entry, i) => (
                <Bar key={i} dataKey="value" fill={entry.fill} />
              ))}
            </Bar>
          </ResponsiveContainer>

          {/* Posterior bars */}
          <div className="mt-3 space-y-1.5">
            {[
              { label: 'P(Low)',    value: result.p_low,    color: GREEN },
              { label: 'P(Medium)', value: result.p_medium, color: AMBER },
              { label: 'P(High)',   value: result.p_high,   color: RED   },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-20">{label}</span>
                <div className="flex-1 h-2 rounded bg-gray-800 overflow-hidden">
                  <div className="h-full rounded"
                    style={{ width: `${value * 100}%`, backgroundColor: color }} />
                </div>
                <span className="text-xs font-mono text-gray-300 w-10 text-right">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          <div className="mt-3 flex justify-between text-xs text-gray-400">
            <span>Most likely: <span className="font-mono text-white">{result.most_likely_impact}</span></span>
            <span>P(Alert): <span className={`font-mono ${result.p_alert > 0.5 ? 'text-red-400' : 'text-green-400'}`}>
              {(result.p_alert * 100).toFixed(1)}%
            </span></span>
          </div>
        </div>
      )}
      {result?.error && (
        <p className="text-xs text-red-400 mt-3">{result.error}</p>
      )}
    </div>
  )
}

// ── Summary bar ────────────────────────────────────────────────────────────────
function SummaryBar({ estimates }) {
  const dids = estimates.filter(e => e.method === 'did')
  const scms = estimates.filter(e => e.method === 'synthetic_control')
  const rdds = estimates.filter(e => e.method === 'rdd')
  const sigDid = dids.filter(d => sig(d.ci_low_95, d.ci_high_95)).length
  const sigRdd = rdds.filter(r => sig(r.ci_low_95, r.ci_high_95)).length

  return (
    <div className="grid grid-cols-4 gap-3 mb-6">
      {[
        { label: 'DiD estimates',       value: dids.length, colour: 'text-blue-400' },
        { label: 'Synthetic controls',  value: scms.length, colour: 'text-purple-400' },
        { label: 'RDD estimates',       value: rdds.length, colour: 'text-amber-400' },
        { label: 'Significant results', value: sigDid + sigRdd, colour: 'text-green-400' },
      ].map(({ label, value, colour }) => (
        <div key={label} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <p className={`text-2xl font-bold ${colour}`}>{value}</p>
          <p className="text-xs text-gray-500 mt-0.5">{label}</p>
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Causal() {
  const { data: estimates, loading, error } = useApi('/causal/estimates', { limit: 100 })

  const all = estimates ?? []

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-white">Causal Analysis</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          DiD · Synthetic Control · RDD — empirical identification of regulatory impact
        </p>
      </div>

      {loading && <p className="text-gray-600 animate-pulse">Computing causal estimates…</p>}
      {error   && <p className="text-red-400 text-sm">Error: {error}</p>}

      {all.length > 0 && <SummaryBar estimates={all} />}

      <div className="space-y-6">
        {all.length > 0 && (
          <>
            <DidPanel estimates={all} />
            <ScmPanel estimates={all} />
            <RddPanel estimates={all} />
          </>
        )}
        <BNWidget />

        {!loading && !error && all.length === 0 && (
          <div className="text-center py-16 text-gray-600">
            <p className="text-4xl mb-3">📊</p>
            <p>No causal estimates yet.</p>
            <p className="text-xs mt-1">
              Run the <code className="text-blue-400">causal_estimation</code> DAG to compute estimates.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
