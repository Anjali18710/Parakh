import { useRef, useState } from 'react'
import type { Flow, FlowParty } from './api'
import { lakhs, shortTime } from './format'

// Colours live in CSS (--flow-in, --flow-out, --flow-other) so light and dark themes each get their own
// validated steps: money in is blue, money out is orange, and transfers that are not evidence are grey.

const W = 760
const ROW = 46
const NODE_W = 10
const LEFT_X = 196
const CENTER_X = 330
const CENTER_W = 100
const RIGHT_X = W - 196 - NODE_W
const MAX_BAND_TOTAL = 260 // tallest stack of bands at the centre node, in px

interface Band {
  party: FlowParty
  side: 'in' | 'out'
  nodeY: number
  thick: number
  centerTop: number
}

interface Tip {
  band: Band
  x: number
  y: number
}

function ribbon(x0: number, y0: number, x1: number, y1: number, t0: number, t1: number) {
  const xm = (x0 + x1) / 2
  return `M${x0},${y0} C${xm},${y0} ${xm},${y1} ${x1},${y1} L${x1},${y1 + t1} C${xm},${y1 + t1} ${xm},${y0 + t0} ${x0},${y0 + t0} Z`
}

interface Props {
  flow: Flow
  onPick: (txnIds: string[]) => void
  /** Accounts the agents are looking at right now; their bands pulse. */
  focus?: string[]
  /** Agents are still working on this case. */
  live?: boolean
}

export function FlowGraph({ flow, onPick, focus = [], live = false }: Props) {
  const [tip, setTip] = useState<Tip | null>(null)
  const box = useRef<HTMLDivElement>(null)
  const rows = Math.max(flow.inflows.length, flow.outflows.length, 3)
  const TOP = 30
  const H = rows * ROW + TOP + 8
  const biggest = Math.max(flow.total_in, flow.total_out, 1)
  const scale = Math.min(MAX_BAND_TOTAL, rows * ROW * 0.72) / biggest

  const layout = (parties: FlowParty[], side: 'in' | 'out') => {
    const total = parties.reduce((a, p) => a + p.total, 0)
    const stack = total * scale + Math.max(parties.length - 1, 0) * 2
    let cursor = TOP + (H - TOP) / 2 - stack / 2
    const offset = (rows - parties.length) * ROW / 2
    return parties.map((party, i): Band => {
      const thick = Math.max(2, party.total * scale)
      const band = { party, side, nodeY: TOP + offset + i * ROW + ROW / 2, thick, centerTop: cursor }
      cursor += thick + 2
      return band
    })
  }
  const bands = [...layout(flow.inflows, 'in'), ...layout(flow.outflows, 'out')]
  const mid = TOP + (H - TOP) / 2
  const centerTop = Math.min(...bands.map((b) => b.centerTop), mid - 22)
  const centerBottom = Math.max(...bands.map((b) => b.centerTop + b.thick), mid + 22)
  const anyCited = bands.some((b) => b.party.cited)

  const key = (b: Band) => `${b.side}:${b.party.account}`
  const label = (p: FlowParty) => `${lakhs(p.total)}, ${p.count} payment${p.count === 1 ? '' : 's'}`
  // Before there is a report nothing is cited yet, so every band keeps its direction colour.
  const tone = (b: Band) => (b.party.cited || !anyCited ? b.side : 'other')
  const hovered = tip ? key(tip.band) : null

  function show(b: Band, e: React.MouseEvent) {
    const r = box.current?.getBoundingClientRect()
    if (r) setTip({ band: b, x: e.clientX - r.left + (box.current?.scrollLeft ?? 0), y: e.clientY - r.top })
  }

  return (
    <figure className={`flow ${live ? 'live' : ''}`}>
      <div className="flow-legend" aria-hidden="true">
        <span><i className="sw in" />Money in</span>
        <span><i className="sw out" />Money out</span>
        {anyCited && <span><i className="sw other" />Not cited as evidence</span>}
      </div>
      <div className="flow-scroll" ref={box} onMouseLeave={() => setTip(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Money flowing into and out of ${flow.account}`}>
          {bands.map((b, i) => {
            const t = b.thick
            const nodeTop = b.nodeY - t / 2
            const d =
              b.side === 'in'
                ? ribbon(LEFT_X + NODE_W, nodeTop, CENTER_X, b.centerTop, t, t)
                : ribbon(CENTER_X + CENTER_W, b.centerTop, RIGHT_X, nodeTop, t, t)
            const k = key(b)
            const pickable = b.party.cited && b.party.txn_ids.length > 0
            const order = b.side === 'in' ? i : i - flow.inflows.length
            // Money arrives first, then leaves: inflows draw in, then outflows.
            const delay = (b.side === 'in' ? 0 : 650) + order * 45
            return (
              <g key={k}
                 className={[
                   'band', b.side, `t-${tone(b)}`,
                   hovered && hovered !== k ? 'dim' : '',
                   hovered === k ? 'hot' : '',
                   focus.includes(b.party.account) ? 'tracing' : '',
                   pickable ? 'clickable' : '',
                 ].join(' ')}
                 style={{ animationDelay: `${delay}ms` }}
                 onMouseMove={(e) => show(b, e)}
                 onClick={() => pickable && onPick(b.party.txn_ids)}>
                <path d={d} />
                <rect x={b.side === 'in' ? LEFT_X : RIGHT_X} y={nodeTop} width={NODE_W} height={t} rx={2} />
              </g>
            )
          })}

          {bands.map((b) => {
            const left = b.side === 'in'
            const x = left ? LEFT_X - 10 : RIGHT_X + NODE_W + 10
            return (
              <g key={`l-${key(b)}`} className={`flow-label ${hovered && hovered !== key(b) ? 'dim' : ''}`}>
                <text x={x} y={b.nodeY - 3} textAnchor={left ? 'end' : 'start'} className="acct">{b.party.account}</text>
                <text x={x} y={b.nodeY + 13} textAnchor={left ? 'end' : 'start'} className="amt">{label(b.party)}</text>
              </g>
            )
          })}

          <rect x={CENTER_X} y={centerTop - 6} width={CENTER_W} height={centerBottom - centerTop + 12} rx={6} className="hub" />
          <text x={CENTER_X + CENTER_W / 2} y={mid - 4} textAnchor="middle" className="hub-acct">{flow.account}</text>
          <text x={CENTER_X + CENTER_W / 2} y={mid + 13} textAnchor="middle" className="hub-sub">this account</text>

          {flow.inflows.length > 0 && <text x={LEFT_X + NODE_W} y={14} textAnchor="end" className="side">Money in</text>}
          {flow.outflows.length > 0 && <text x={RIGHT_X} y={14} className="side">Money out</text>}
        </svg>
        {tip && (
          <div className="flow-tip" style={{ left: tip.x, top: tip.y }} role="tooltip">
            <strong>{tip.band.side === 'in' ? 'From' : 'To'} {tip.band.party.account}</strong>
            <span>{label(tip.band.party)}</span>
            {tip.band.party.first && (
              <span>
                {shortTime(tip.band.party.first)}
                {tip.band.party.last !== tip.band.party.first ? ` to ${shortTime(tip.band.party.last)}` : ''}
              </span>
            )}
            <em>
              {tip.band.party.cited
                ? tip.band.party.txn_ids.length ? 'Cited as evidence. Click to see it.' : 'Cited as evidence'
                : 'Not cited as evidence'}
            </em>
          </div>
        )}
      </div>
      <figcaption>
        {live ? 'The agents are tracing the pulsing bands. ' : ''}
        Band width is proportional to the amount. Hover a band for details; click a coloured band to find its transactions.
      </figcaption>
    </figure>
  )
}
