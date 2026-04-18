/**
 * Knowledge Graph page — Phase 4.
 *
 * Renders a force-directed graph using D3 (via useEffect).
 *   - Nodes: sized by PageRank, coloured by community
 *   - Edges: opacity by weight, colour by edge type
 *   - Hover tooltip: regulation title, agency, PageRank, community
 *   - Click: shows ego-subgraph panel on the right
 *   - Agent trigger: "Run Impact Agent" button fires POST /graph/agent/:id
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { Network, RefreshCw, Play, Loader, AlertTriangle } from 'lucide-react'
import { useApi } from '../hooks/useApi'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8081/api/v1'

// ── Community colour palette (up to 12 communities) ──────────────────────────
const COMMUNITY_COLOURS = [
  '#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444',
  '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#a78bfa',
  '#14b8a6', '#fb7185',
]

function communityColour(id) {
  if (id < 0) return '#6b7280'
  return COMMUNITY_COLOURS[id % COMMUNITY_COLOURS.length]
}

// ── D3 force graph ─────────────────────────────────────────────────────────────
function ForceGraph({ nodes, edges, onNodeClick, selectedId }) {
  const svgRef = useRef(null)
  const simRef = useRef(null)

  useEffect(() => {
    if (!nodes.length || !svgRef.current) return

    // Dynamically import D3 to avoid SSR issues
    import('https://cdn.jsdelivr.net/npm/d3@7/+esm').then(d3 => {
      const svg = d3.select(svgRef.current)
      svg.selectAll('*').remove()

      const width  = svgRef.current.clientWidth  || 800
      const height = svgRef.current.clientHeight || 500

      const g = svg.append('g')

      // Zoom + pan
      svg.call(
        d3.zoom()
          .scaleExtent([0.2, 4])
          .on('zoom', e => g.attr('transform', e.transform))
      )

      // PageRank → node radius [6, 22]
      const prValues = nodes.map(n => n.pagerank || 0.001)
      const prScale = d3.scaleSqrt()
        .domain([Math.min(...prValues), Math.max(...prValues)])
        .range([6, 22])

      // Simulation
      const sim = d3.forceSimulation(nodes)
        .force('link',   d3.forceLink(edges)
          .id(d => d.regulation_id)
          .distance(d => 80 + (1 - d.weight) * 60))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide(d => prScale(d.pagerank || 0.001) + 4))

      simRef.current = sim

      // Edge type colour
      const edgeColour = { semantic: '#3b82f6', citation: '#8b5cf6',
                           amendment: '#f59e0b', shared_agency: '#374151' }

      // Draw edges
      const link = g.append('g')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('stroke', d => edgeColour[d.edge_type] || '#374151')
        .attr('stroke-opacity', d => 0.3 + d.weight * 0.5)
        .attr('stroke-width', d => 1 + d.weight * 2)

      // Draw nodes
      const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
        .attr('r', d => prScale(d.pagerank || 0.001))
        .attr('fill', d => communityColour(d.community))
        .attr('stroke', d => d.regulation_id === selectedId ? '#fff' : 'transparent')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', (_, d) => onNodeClick(d))
        .call(
          d3.drag()
            .on('start', (event, d) => {
              if (!event.active) sim.alphaTarget(0.3).restart()
              d.fx = d.x; d.fy = d.y
            })
            .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
            .on('end',  (event, d) => {
              if (!event.active) sim.alphaTarget(0)
              d.fx = null; d.fy = null
            })
        )

      // Labels (only for high-PageRank nodes)
      const maxPr = Math.max(...prValues)
      const label = g.append('g')
        .selectAll('text')
        .data(nodes.filter(n => (n.pagerank || 0) > maxPr * 0.3))
        .join('text')
        .text(d => d.agency)
        .attr('font-size', 8)
        .attr('fill', '#9ca3af')
        .attr('text-anchor', 'middle')
        .attr('dy', d => prScale(d.pagerank || 0.001) + 10)
        .style('pointer-events', 'none')

      sim.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y)
        node.attr('cx', d => d.x).attr('cy', d => d.y)
        label.attr('x', d => d.x).attr('y', d => d.y)
      })
    }).catch(() => {
      // D3 CDN not available — render placeholder
    })

    return () => { simRef.current?.stop() }
  }, [nodes, edges, selectedId])

  return (
    <svg
      ref={svgRef}
      className="w-full h-full"
      style={{ background: '#030712' }}
    />
  )
}

// ── Node detail panel ─────────────────────────────────────────────────────────
function NodeDetail({ node, onRunAgent }) {
  const [agentResult, setAgentResult] = useState(null)
  const [running, setRunning] = useState(false)

  async function runAgent() {
    setRunning(true)
    setAgentResult(null)
    try {
      const res = await fetch(`${API_BASE}/graph/agent/${node.regulation_id}`, {
        method: 'POST',
      })
      setAgentResult(await res.json())
    } catch {
      setAgentResult({ error: 'Agent unavailable' })
    } finally {
      setRunning(false)
    }
  }

  const pH = agentResult?.impact_score?.p_high
  const rwa = agentResult?.rwa_estimate

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Node info */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span
            className="w-3 h-3 rounded-full shrink-0"
            style={{ backgroundColor: communityColour(node.community) }}
          />
          <span className="text-xs font-mono text-blue-400">{node.document_number}</span>
        </div>
        <p className="text-sm text-gray-200 leading-snug">{node.title}</p>
        <p className="text-xs text-gray-500 mt-1">{node.agency} · {node.doc_type}</p>
      </div>

      {/* Graph metrics */}
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'PageRank',  value: node.pagerank?.toFixed(5) },
          { label: 'Community', value: `#${node.community}` },
          { label: 'In-degree',  value: node.in_degree },
          { label: 'Out-degree', value: node.out_degree },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-900 rounded p-2">
            <p className="text-xs text-gray-500">{label}</p>
            <p className="text-sm font-mono text-gray-200">{value}</p>
          </div>
        ))}
      </div>

      {/* Agent trigger */}
      <button
        onClick={runAgent}
        disabled={running}
        className="w-full flex items-center justify-center gap-2 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors disabled:opacity-50"
      >
        {running ? <Loader size={13} className="animate-spin" /> : <Play size={13} />}
        {running ? 'Running agent…' : 'Run Impact Agent'}
      </button>

      {/* Agent result */}
      {agentResult && !agentResult.error && (
        <div className="space-y-3">
          {/* Impact distribution */}
          <div>
            <p className="text-xs text-gray-500 mb-1.5">Bayesian Impact Posterior</p>
            {[
              { label: 'P(Low)',    value: agentResult.impact_score?.p_low    ?? 0, colour: '#22c55e' },
              { label: 'P(Medium)', value: agentResult.impact_score?.p_medium ?? 0, colour: '#f59e0b' },
              { label: 'P(High)',   value: agentResult.impact_score?.p_high   ?? 0, colour: '#ef4444' },
            ].map(({ label, value, colour }) => (
              <div key={label} className="flex items-center gap-2 mb-1">
                <span className="text-xs text-gray-500 w-18">{label}</span>
                <div className="flex-1 h-1.5 rounded bg-gray-800 overflow-hidden">
                  <div className="h-full rounded"
                    style={{ width: `${value * 100}%`, backgroundColor: colour }} />
                </div>
                <span className="text-xs font-mono text-gray-300 w-8 text-right">
                  {(value * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>

          {/* RWA */}
          {rwa?.median_million_usd != null && (
            <div className="bg-gray-900 rounded p-3">
              <p className="text-xs text-gray-500 mb-1">ΔCapital Requirement</p>
              <p className="text-lg font-mono text-blue-300">
                ${rwa.median_million_usd.toFixed(0)}M
              </p>
              <p className="text-xs text-gray-600 mt-0.5">
                90% CI [{rwa.ci_low_90_million_usd?.toFixed(0)}M,{' '}
                {rwa.ci_high_90_million_usd?.toFixed(0)}M]
              </p>
            </div>
          )}

          {/* Alert */}
          {agentResult.alert_dispatched && (
            <div className="flex items-center gap-1.5 text-xs text-red-400">
              <AlertTriangle size={12} />
              High-impact alert dispatched
            </div>
          )}

          {/* Summary */}
          {agentResult.summary && (
            <div>
              <p className="text-xs text-gray-500 mb-1">Agent Summary</p>
              <p className="text-xs text-gray-400 leading-relaxed">
                {agentResult.summary}
              </p>
            </div>
          )}

          {/* Steps */}
          <div>
            <p className="text-xs text-gray-600 mb-1">
              {agentResult.reasoning_steps} reasoning steps
            </p>
          </div>
        </div>
      )}

      {agentResult?.error && (
        <p className="text-xs text-red-400">{agentResult.error}</p>
      )}
    </div>
  )
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend({ communities }) {
  return (
    <div className="absolute bottom-4 left-4 bg-gray-900/90 border border-gray-800 rounded-lg p-3 text-xs">
      <p className="text-gray-500 mb-2 font-medium">Communities</p>
      <div className="space-y-1">
        {communities.slice(0, 6).map(c => (
          <div key={c.community_id} className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ backgroundColor: communityColour(c.community_id) }}
            />
            <span className="text-gray-400">
              C{c.community_id}: {c.agencies.slice(0, 2).join(', ')}
              <span className="text-gray-600 ml-1">({c.n_members})</span>
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 pt-2 border-t border-gray-800 space-y-1">
        <p className="text-gray-500 mb-1">Edge types</p>
        {[
          { colour: '#3b82f6', label: 'semantic' },
          { colour: '#f59e0b', label: 'amendment' },
          { colour: '#374151', label: 'shared agency' },
        ].map(({ colour, label }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-4 h-px" style={{ backgroundColor: colour, height: 2 }} />
            <span className="text-gray-500">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function Graph() {
  const [selectedNode, setSelectedNode] = useState(null)
  const { data: nodes,       loading: nLoad, refetch: refetchNodes }       = useApi('/graph/nodes', { limit: 200 })
  const { data: edges,       loading: eLoad }       = useApi('/graph/edges', { limit: 500 })
  const { data: summary,     loading: sLoad }       = useApi('/graph/snapshot')
  const { data: communities, loading: cLoad }       = useApi('/graph/communities')

  const loading = nLoad || eLoad || sLoad || cLoad
  const graphNodes = nodes ?? []
  const graphEdges = (edges ?? []).map(e => ({
    ...e,
    source: e.source_id,
    target: e.target_id,
  }))

  return (
    <div className="flex h-full overflow-hidden">
      {/* Graph canvas */}
      <div className="relative flex-1">
        {/* Header overlay */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-5 py-3 bg-gradient-to-b from-gray-950 to-transparent">
          <div>
            <h1 className="text-lg font-bold text-white">Knowledge Graph</h1>
            {summary && (
              <p className="text-xs text-gray-500">
                {summary.n_nodes} regulations · {summary.n_edges} edges ·{' '}
                {summary.n_communities} communities · modularity {summary.modularity?.toFixed(3)}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={refetchNodes}
              className="p-1.5 rounded bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-700"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-gray-600 animate-pulse">Building graph…</p>
          </div>
        )}

        {!loading && graphNodes.length > 0 && (
          <ForceGraph
            nodes={graphNodes}
            edges={graphEdges}
            onNodeClick={setSelectedNode}
            selectedId={selectedNode?.regulation_id}
          />
        )}

        {!loading && graphNodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600">
            <Network size={48} className="mb-3 opacity-30" />
            <p>No graph data yet.</p>
            <p className="text-xs mt-1">
              Run the <code className="text-blue-400">graph_update</code> DAG to build the graph.
            </p>
          </div>
        )}

        {communities?.length > 0 && (
          <Legend communities={communities} />
        )}
      </div>

      {/* Side panel */}
      <div className="w-72 shrink-0 border-l border-gray-800 bg-gray-950 flex flex-col">
        {selectedNode ? (
          <>
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
              <span className="text-sm font-medium text-gray-300">Node Detail</span>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-gray-600 hover:text-gray-300 text-xs"
              >
                ✕
              </button>
            </div>
            <NodeDetail node={selectedNode} />
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-600 p-6 text-center">
            <Network size={32} className="mb-3 opacity-30" />
            <p className="text-sm">Click a node to view details</p>
            <p className="text-xs mt-1">
              Run the Impact Agent to get Bayesian scores and ΔCapital estimates.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
