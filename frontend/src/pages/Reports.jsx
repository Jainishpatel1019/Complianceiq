/**
 * Impact Reports page — shows agent-generated reports with Bayesian P(High)
 * scores and Basel III RWA estimates with Monte Carlo confidence intervals.
 */
import { useApi } from '../hooks/useApi'
import ScoreBadge from '../components/ScoreBadge'
import { AlertTriangle, TrendingUp } from 'lucide-react'

function ImpactBar({ pLow, pMedium, pHigh }) {
  const low  = Math.round((pLow    ?? 0) * 100)
  const med  = Math.round((pMedium ?? 0) * 100)
  const high = Math.round((pHigh   ?? 0) * 100)

  return (
    <div className="flex rounded overflow-hidden h-2 w-full gap-px">
      <div style={{ width: `${low}%`  }} className="bg-green-600"  title={`Low: ${low}%`}  />
      <div style={{ width: `${med}%`  }} className="bg-yellow-500" title={`Med: ${med}%`}  />
      <div style={{ width: `${high}%` }} className="bg-red-500"    title={`High: ${high}%`} />
    </div>
  )
}

function ReportCard({ report }) {
  const pH = report.impact_score?.p_high ?? 0
  const rwa = report.rwa_estimate

  return (
    <div className="border border-gray-800 rounded-xl bg-gray-900/50 p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-blue-400 font-mono">{report.document_number}</span>
            <span className="text-xs text-gray-600">{report.agency}</span>
            {pH >= 0.8 && <AlertTriangle size={13} className="text-red-400" />}
          </div>
          <p className="text-sm text-gray-200 leading-snug">{report.title}</p>
          {report.summary && (
            <p className="text-xs text-gray-500 mt-2 line-clamp-2">{report.summary}</p>
          )}
        </div>
        <ScoreBadge score={pH} label="P(High)" />
      </div>

      {/* Bayesian posterior bar */}
      <div className="mt-4">
        <div className="flex justify-between text-xs text-gray-600 mb-1">
          <span>Impact distribution</span>
          <span className="font-mono">
            L{Math.round((report.impact_score?.p_low ?? 0) * 100)}%
            M{Math.round((report.impact_score?.p_medium ?? 0) * 100)}%
            H{Math.round(pH * 100)}%
          </span>
        </div>
        <ImpactBar
          pLow={report.impact_score?.p_low}
          pMedium={report.impact_score?.p_medium}
          pHigh={pH}
        />
      </div>

      {/* RWA estimate */}
      {rwa?.median_million_usd != null && (
        <div className="mt-3 flex items-center gap-2 text-xs">
          <TrendingUp size={12} className="text-blue-400" />
          <span className="text-gray-400">ΔCapital:</span>
          <span className="font-mono text-blue-300">
            ${rwa.median_million_usd.toFixed(0)}M
          </span>
          <span className="text-gray-600">
            90% CI [${rwa.ci_low_90_million_usd?.toFixed(0)}M,
            ${rwa.ci_high_90_million_usd?.toFixed(0)}M]
          </span>
        </div>
      )}

      {/* Metadata */}
      <div className="mt-3 flex items-center gap-3 text-xs text-gray-600">
        <span>{report.reasoning_steps} reasoning steps</span>
        {report.alert_dispatched && (
          <span className="text-red-400">Alert dispatched</span>
        )}
        <span className="ml-auto">{new Date(report.created_at).toLocaleDateString()}</span>
      </div>
    </div>
  )
}

export default function Reports() {
  const { data: reports, loading, error } = useApi('/reports/high-impact', { threshold: 0.0 })

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-white">Impact Reports</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          LangGraph agent output — Bayesian P(High) + Basel III ΔCapital estimates
        </p>
      </div>

      {loading && <p className="text-gray-600 animate-pulse">Loading reports…</p>}
      {error   && <p className="text-red-400 text-sm">Error: {error}</p>}

      <div className="space-y-4">
        {(reports ?? []).map(r => (
          <ReportCard key={r.report_id} report={r} />
        ))}
        {!loading && !error && reports?.length === 0 && (
          <div className="text-center py-16 text-gray-600">
            <p className="text-4xl mb-3">📋</p>
            <p>No impact reports yet.</p>
            <p className="text-xs mt-1">Run the <code className="text-blue-400">impact_agent</code> DAG to generate reports.</p>
          </div>
        )}
      </div>
    </div>
  )
}
