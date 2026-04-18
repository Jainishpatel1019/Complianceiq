/**
 * Colour-coded score badge: green (low) → yellow (medium) → red (high).
 * Used everywhere a drift/impact score needs visual encoding.
 */
import clsx from 'clsx'

export default function ScoreBadge({ score, label }) {
  const pct = Math.round((score ?? 0) * 100)
  const colour =
    pct >= 50 ? 'bg-red-500/20 text-red-300 border-red-500/40' :
    pct >= 20 ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40' :
                'bg-green-500/20 text-green-300 border-green-500/40'

  return (
    <span className={clsx('inline-flex items-center gap-1 px-2 py-0.5 rounded border text-xs font-mono', colour)}>
      {label && <span className="text-gray-500">{label}</span>}
      {pct}%
    </span>
  )
}
