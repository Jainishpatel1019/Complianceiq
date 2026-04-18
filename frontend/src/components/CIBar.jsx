/**
 * Visualises a score with its 95% confidence interval as a horizontal bar.
 * Every score in ComplianceIQ comes with a CI — never a bare number.
 *
 * Anatomy:
 *   ──────[===●===]──────
 *         ci_low  ci_high
 *              ^ point estimate
 */
export default function CIBar({ score, ciLow, ciHigh, max = 1 }) {
  const toPercent = (v) => `${Math.round(((v ?? 0) / max) * 100)}%`

  const barLeft  = toPercent(ciLow)
  const barWidth = toPercent((ciHigh ?? score) - (ciLow ?? score))
  const dotLeft  = toPercent(score)

  return (
    <div className="relative h-3 w-full rounded bg-gray-800">
      {/* CI range band */}
      <div
        className="absolute h-full rounded bg-blue-500/30"
        style={{ left: barLeft, width: barWidth }}
      />
      {/* Point estimate dot */}
      <div
        className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-blue-400"
        style={{ left: dotLeft }}
      />
    </div>
  )
}
