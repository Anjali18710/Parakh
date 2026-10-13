import { useCallback, useEffect, useState } from 'react'
import { api, errorMessage, type CaseDetail, type Flow, type TimelineEvent, type Txn } from './api'
import { FlowGraph } from './FlowGraph'
import {
  AGENT_LABELS, SIGNAL_LABELS, STATUS_LABELS, TOOL_LABELS, hoursText, isWorking, lakhs, patternName, rupees, shortTime,
} from './format'

function describe(action: string): string {
  const m = action.match(/^called (\w+)$/)
  return m ? TOOL_LABELS[m[1]] ?? action : action
}

function detailText(detail: unknown): string | null {
  if (detail == null || detail === '') return null
  if (typeof detail === 'string') return detail.replace(/[*#|]+/g, ' ').replace(/\s{2,}/g, ' ').trim()
  if (Array.isArray(detail)) return detail.length ? detail.join('; ') : null
  if (typeof detail === 'object') {
    const entries = Object.entries(detail as Record<string, unknown>).filter(([k]) => k !== 'account')
    return entries.length
      ? entries.map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') || 'none' : v}`).join(', ')
      : null
  }
  return String(detail)
}

/** One line that tells the story of the agents' work. */
function trailStory(c: CaseDetail): string {
  const t = c.timeline
  const tools = t.filter((e) => e.action.startsWith('called ')).length
  const rewrites = t.filter((e) => e.action.startsWith('rewrote')).length
  const v = c.verification
  const parts = [`${tools} tool call${tools === 1 ? '' : 's'}`]
  if (c.pattern) parts.push(`matched ${patternName(c.pattern.pattern_id).toLowerCase()}`)
  if (rewrites) parts.push(`report rewritten ${rewrites} time${rewrites === 1 ? '' : 's'} after fact-checking`)
  if (v) parts.push(v.passed ? 'every claim verified' : `${v.claims_total - v.claims_verified} claim(s) still unverified`)
  return parts.join(', ')
}

function Trail({ events, live }: { events: TimelineEvent[]; live: boolean }) {
  return (
    <ol className={`trail ${live ? 'live' : ''}`}>
      {events.map((e, i) => {
        const d = detailText(e.detail)
        const latest = live && i === events.length - 1
        return (
          <li key={i} className={`a-${e.agent} ${latest ? 'latest' : ''}`}>
            <time>{shortTime(e.time)}</time>
            <div>
              <strong>{AGENT_LABELS[e.agent] ?? e.agent}</strong> {describe(e.action)}
              {d && <p className="clamp">{d}</p>}
            </div>
          </li>
        )
      })}
      {live && (
        <li className="next">
          <time />
          <div>
            <span className="pulse" aria-hidden="true" /> Working on the next step…
          </div>
        </li>
      )}
    </ol>
  )
}

const pct = (v: unknown) => `${Math.round(Number(v) * 100)}%`

/** The measured facts the pattern agent was given, so the analyst can check its reasoning. */
function ShapeFacts({ shape }: { shape: Record<string, unknown> }) {
  const sched = (shape.outflow_schedule ?? {}) as { median_gap_days?: number; gap_variation?: number | null }
  const regular = sched.gap_variation != null && sched.gap_variation < 0.5
  const facts = [
    `${shape.senders_in_window} sender${shape.senders_in_window === 1 ? '' : 's'}, ${shape.recipients_in_window} recipient${shape.recipients_in_window === 1 ? '' : 's'} in the window`,
    `${pct(shape.inflow_from_repeat_senders)} of money in from repeat senders`,
    `${pct(shape.largest_recipient_share)} of money out to the biggest recipient`,
    sched.median_gap_days != null
      ? `pays out every ${sched.median_gap_days} days, ${regular ? 'on a regular schedule' : 'irregularly'}`
      : 'too few payment days to judge a schedule',
  ]
  return (
    <div className="shape">
      <h4>Measured from the ledger</h4>
      <ul>{facts.map((f) => <li key={f}>{f}</li>)}</ul>
    </div>
  )
}

export function CaseFile({ caseId, onChanged }: { caseId: string; onChanged: () => void }) {
  const [c, setCase] = useState<CaseDetail | null>(null)
  const [txns, setTxns] = useState<Txn[]>([])
  const [flow, setFlow] = useState<Flow | null>(null)
  const [highlight, setHighlight] = useState<string[]>([])
  const [trailOpen, setTrailOpen] = useState(false)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await api.get<CaseDetail>(`/cases/${caseId}`)
      setCase(r.data)
      const [f, t] = await Promise.all([
        api.get<Flow>(`/cases/${caseId}/flow`),
        r.data.report ? api.get<Txn[]>(`/cases/${caseId}/transactions`) : Promise.resolve({ data: [] as Txn[] }),
      ])
      setFlow(f.data)
      setTxns(t.data)
    } catch (err) {
      setError(errorMessage(err))
    }
  }, [caseId])

  useEffect(() => {
    load()
  }, [load])

  const working = c ? isWorking(c.status) : false
  useEffect(() => {
    if (!working) return
    const id = window.setInterval(load, 2000)
    return () => window.clearInterval(id)
  }, [working, load])

  async function decide(action: 'approve' | 'reject' | 'reinvestigate') {
    if (action === 'reinvestigate' && !note.trim()) {
      setError('Write what the agents should look into, then ask them to dig further.')
      return
    }
    setBusy(true)
    try {
      await api.post(`/cases/${caseId}/decision`, { action, note: note.trim() || null })
      setNote('')
      setError(null)
      await load()
      onChanged()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  function showTxns(ids: string[]) {
    const present = ids.filter((id) => document.getElementById(`txn-${id}`))
    setHighlight(present)
    if (present[0]) document.getElementById(`txn-${present[0]}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  if (!c) {
    return (
      <section className="casefile">
        {error ? <p className="field-error pad">{error}</p> : <p className="loading">Opening case…</p>}
      </section>
    )
  }

  const r = c.report
  const v = c.verification
  const checks = r?.checks ?? []

  return (
    <section className="casefile" aria-live="polite">
      <header className="case-head">
        <div>
          <h2>{c.account}</h2>
          <p className={`status s-${c.status}`}>
            {working && <span className="pulse" aria-hidden="true" />}
            {STATUS_LABELS[c.status] ?? c.status}
          </p>
        </div>
        {r && (
          <div className={`verdict ${r.verdict}`}>
            <div className="score" aria-label={`Risk score ${r.risk_score} out of 100`}>
              {r.risk_score}
              <small>/100 risk</small>
            </div>
            <div>
              <strong>{r.verdict === 'suspicious' ? 'Suspicious' : 'Looks innocent'}</strong>
              <span>{patternName(r.pattern_id)}</span>
            </div>
          </div>
        )}
      </header>

      {flow && (flow.inflows.length > 0 || flow.outflows.length > 0) && (
        <dl className="strip">
          <div>
            <dt>Money in</dt>
            <dd>{lakhs(flow.total_in)}</dd>
            <span>from {flow.senders} account{flow.senders === 1 ? '' : 's'}</span>
          </div>
          <div>
            <dt>Money out</dt>
            <dd>{lakhs(flow.total_out)}</dd>
            <span>to {flow.recipients} account{flow.recipients === 1 ? '' : 's'}</span>
          </div>
          <div>
            <dt>Passed straight through</dt>
            <dd>{Math.round(flow.passed_through * 100)}%</dd>
            <span>of what came in</span>
          </div>
          <div>
            <dt>Time span</dt>
            <dd>{flow.span_hours ? hoursText(flow.span_hours) : '–'}</dd>
            <span>{r ? 'of the cited activity' : 'busiest period'}</span>
          </div>
          <div>
            <dt>Evidence</dt>
            <dd>{v ? `${v.claims_verified}/${v.claims_total}` : '–'}</dd>
            <span>{v ? 'claims verified' : 'waiting for report'}</span>
          </div>
        </dl>
      )}

      {c.status === 'failed' && <p className="banner bad">The agents stopped with an error: {c.error}</p>}

      <div className="case-body">
        <section>
          <h3>Why this account was flagged</h3>
          <ul className="signals">
            {c.alert.signals.length ? (
              c.alert.signals.map((s) => (
                <li key={s.name}>
                  <strong>{SIGNAL_LABELS[s.name] ?? s.name}</strong>
                  <span>{s.detail}</span>
                </li>
              ))
            ) : (
              <li>Opened by an analyst, not by a screening rule.</li>
            )}
          </ul>
          {c.triage && <p className="aside">Triage agent: {c.triage.reason}</p>}
        </section>

        {working && (
          <section className="at-work">
            <h3>Agents at work</h3>
            <Trail events={c.timeline} live />
          </section>
        )}

        {flow && (flow.inflows.length > 0 || flow.outflows.length > 0) && (
          <section>
            <h3>Where the money went</h3>
            <FlowGraph flow={flow} onPick={showTxns} />
          </section>
        )}

        {r && (
          <>
            <section>
              <h3>Findings</h3>
              <p className="summary">{r.summary}</p>
              <p className="action">
                <strong>Recommended:</strong> {r.recommended_action}
              </p>
            </section>

            <section>
              <div className="section-head">
                <h3>Evidence</h3>
                {v && (
                  <p className={`check ${v.passed ? 'ok' : 'bad'}`}>
                    {v.claims_verified} of {v.claims_total} claims match the transactions
                  </p>
                )}
              </div>
              <ol className="claims">
                {r.claims.map((cl, i) => {
                  const res = v?.claim_results.find((x) => x.index === i)
                  return (
                    <li key={i} className={res ? (res.ok ? 'ok' : 'bad') : ''}>
                      <span className="assay" aria-label={res?.ok ? 'Verified' : 'Not verified'} />
                      <div>
                        <p>{cl.statement}</p>
                        <div className="chips">
                          {cl.txn_ids.map((t) => (
                            <button key={t} className="chip" onClick={() => showTxns([t])}>
                              {t}
                            </button>
                          ))}
                        </div>
                        {res && !res.ok && <p className="problem">Fact-check: {res.problems.join('; ')}</p>}
                      </div>
                    </li>
                  )
                })}
              </ol>
              {checks.length > 0 && (
                <>
                  <h4>Also checked, nothing found</h4>
                  <ul className="checks">
                    {checks.map((ch, i) => {
                      const res = v?.check_results?.find((x) => x.index === i)
                      return (
                        <li key={i} className={res && !res.ok ? 'bad' : ''}>
                          <span>{ch.description}</span>
                          <small>
                            {res && !res.ok ? res.problems.join('; ') : `Investigator ${TOOL_LABELS[ch.tool] ?? ch.tool}`}
                          </small>
                        </li>
                      )
                    })}
                  </ul>
                </>
              )}
            </section>

            {txns.length > 0 && (
              <section>
                <h3>Transactions cited</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Id</th>
                        <th scope="col">When</th>
                        <th scope="col">From</th>
                        <th scope="col">To</th>
                        <th scope="col" className="num">Amount</th>
                        <th scope="col">Via</th>
                      </tr>
                    </thead>
                    <tbody>
                      {txns.map((t) => (
                        <tr key={t.txn_id} id={`txn-${t.txn_id}`} className={highlight.includes(t.txn_id) ? 'hl' : ''}>
                          <td>{t.txn_id}</td>
                          <td>{shortTime(t.timestamp)}</td>
                          <td className={t.from_account === c.account ? 'me' : ''}>{t.from_account}</td>
                          <td className={t.to_account === c.account ? 'me' : ''}>{t.to_account}</td>
                          <td className="num">{rupees(t.amount)}</td>
                          <td>{t.payment_format}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {c.pattern && (
              <section>
                <h3>Pattern match</h3>
                <p className="summary">
                  <strong>{patternName(c.pattern.pattern_id)}</strong>, {Math.round(c.pattern.confidence * 100)}% confident.{' '}
                  {c.pattern.reasoning}
                </p>
                {c.pattern.shape && <ShapeFacts shape={c.pattern.shape} />}
                {c.pattern.ruled_out.length > 0 && (
                  <p className="aside">Ruled out: {c.pattern.ruled_out.map(patternName).join(', ')}</p>
                )}
              </section>
            )}
          </>
        )}

        {!working && c.timeline.length > 0 && (
          <section>
            <div className="section-head">
              <h3>Agent trail</h3>
              <button className="link" aria-expanded={trailOpen} onClick={() => setTrailOpen((o) => !o)}>
                {trailOpen ? 'Hide steps' : `Show all ${c.timeline.length} steps`}
              </button>
            </div>
            <p className="story">{trailStory(c)}.</p>
            {trailOpen && <Trail events={c.timeline} live={false} />}
            {c.usage && (
              <p className="aside">
                {c.tool_calls ?? 0} tool calls, {c.usage.llm_calls} model calls,{' '}
                {(c.usage.input_tokens + c.usage.output_tokens).toLocaleString('en-IN')} tokens, about $
                {c.usage.est_cost_usd.toFixed(4)}, {c.duration_s}s
              </p>
            )}
          </section>
        )}
      </div>

      {c.status === 'awaiting_review' && r && (
        <footer className="decision">
          <label htmlFor="note">Your note</label>
          <textarea
            id="note"
            rows={2}
            placeholder="Optional for a decision. Required if you want the agents to dig further, e.g. check where the money came from."
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          {error && <p className="field-error">{error}</p>}
          <div className="buttons">
            <button className="primary" disabled={busy} onClick={() => decide('approve')}>
              Agree: {r.verdict === 'suspicious' ? 'mark suspicious' : 'close as innocent'}
            </button>
            <button disabled={busy} onClick={() => decide('reject')}>
              Overrule: {r.verdict === 'suspicious' ? 'close as innocent' : 'mark suspicious'}
            </button>
            <button disabled={busy} onClick={() => decide('reinvestigate')}>
              Ask agents to dig further
            </button>
          </div>
        </footer>
      )}
      {c.decision && c.status !== 'awaiting_review' && (
        <p className="decided">
          Analyst decision:{' '}
          {c.decision.action === 'approve'
            ? 'agreed with the agents'
            : c.decision.action === 'reject'
              ? 'overruled the agents'
              : 'asked for more investigation'}
          {c.decision.note ? `. Note: ${c.decision.note}` : ''}
        </p>
      )}
    </section>
  )
}
