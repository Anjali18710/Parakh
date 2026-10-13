import { useState } from 'react'
import type { Flow, FlowParty } from './api'
import { lakhs, shortTime } from './format'

// Validated pair (colour-blind safe, contrast checked): evidence flows in gold, other flows in blue.
const CITED = '#b8860b'
const OTHER = '#3f78b8'

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

function ribbon(x0: number, y0: number, x1: number, y1: number, t0: number, t1: number) {
  const xm = (x0 + x1) / 2
  return `M${x0},${y0} C${xm},${y0} ${xm},${y1} ${x1},${y1} L${x1},${y1 + t1} C${xm},${y1 + t1} ${xm},${y0 + t0} ${x0},${y0 + t0} Z`
}

export function FlowGraph({ flow, onPick }: { flow: Flow; onPick: (txnIds: string[]) => void }) {
  const [hover, setHover] = useState<string | null>(null)
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

  const key = (b: Band) => `${b.side}:${b.party.account}`
  const label = (p: FlowParty) =>
    `${lakhs(p.total)}, ${p.count} payment${p.count === 1 ? '' : 's'}`

  return (
    <figure className="flow">
      <div className="flow-legend" aria-hidden="true">
        <span><i style={{ background: CITED }} />Cited as evidence</span>
        <span><i style={{ background: OTHER }} />Other transfers in the same window</span>
      </div>
      <div className="flow-scroll">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Money flowing into and out of ${flow.account}`}>
        {bands.map((b) => {
          const t = b.thick
          const nodeTop = b.nodeY - t / 2
          const d =
            b.side === 'in'
              ? ribbon(LEFT_X + NODE_W, nodeTop, CENTER_X, b.centerTop, t, t)
              : ribbon(CENTER_X + CENTER_W, b.centerTop, RIGHT_X, nodeTop, t, t)
          const k = key(b)
          const dim = hover !== null && hover !== k
          const pickable = b.party.cited && b.party.txn_ids.length > 0
          return (
            <g key={k}
               className={`band ${dim ? 'dim' : ''} ${pickable ? 'clickable' : ''}`}
               onMouseEnter={() => setHover(k)}
               onMouseLeave={() => setHover(null)}
               onClick={() => pickable && onPick(b.party.txn_ids)}>
              <path d={d} fill={b.party.cited ? CITED : OTHER} />
              <rect x={b.side === 'in' ? LEFT_X : RIGHT_X} y={nodeTop} width={NODE_W} height={t} rx={2}
                    fill={b.party.cited ? CITED : OTHER} />
              <title>
                {`${b.side === 'in' ? 'From' : 'To'} ${b.party.account}: ${label(b.party)}` +
                  (b.party.first ? `, ${shortTime(b.party.first)} to ${shortTime(b.party.last)}` : '') +
                  (pickable ? '. Click to find it in the cited transactions.' : '')}
              </title>
            </g>
          )
        })}

        {bands.map((b) => {
          const left = b.side === 'in'
          const x = left ? LEFT_X - 10 : RIGHT_X + NODE_W + 10
          return (
            <g key={`l-${key(b)}`} className={`flow-label ${hover !== null && hover !== key(b) ? 'dim' : ''}`}>
              <text x={x} y={b.nodeY - 3} textAnchor={left ? 'end' : 'start'} className="acct">{b.party.account}</text>
              <text x={x} y={b.nodeY + 13} textAnchor={left ? 'end' : 'start'} className="amt">{label(b.party)}</text>
            </g>
          )
        })}

        <rect x={CENTER_X} y={centerTop - 6} width={CENTER_W} height={centerBottom - centerTop + 12} rx={6} className="hub" />
        <text x={CENTER_X + CENTER_W / 2} y={TOP + (H - TOP) / 2 - 4} textAnchor="middle" className="hub-acct">{flow.account}</text>
        <text x={CENTER_X + CENTER_W / 2} y={TOP + (H - TOP) / 2 + 13} textAnchor="middle" className="hub-sub">this account</text>

        {flow.inflows.length > 0 && <text x={LEFT_X + NODE_W} y={14} textAnchor="end" className="side">Money in</text>}
        {flow.outflows.length > 0 && <text x={RIGHT_X} y={14} className="side">Money out</text>}
      </svg>
      </div>
      <figcaption>
        Band width is proportional to the amount. Hover a band for details; click a gold band to find its transaction below.
      </figcaption>
    </figure>
  )
}
