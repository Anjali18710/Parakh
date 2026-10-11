import { useCallback, useEffect, useState } from 'react'
import { api, errorMessage, type CaseDetail, type Txn } from './api'
import { AGENT_LABELS, SIGNAL_LABELS, STATUS_LABELS, isWorking, patternName, rupees, shortTime } from './format'

const TOOL_LABELS: Record<string, string> = {
  account_profile: 'looked at the account profile',
  list_transactions: 'listed transactions',
  trace_money: 'followed the money',
  find_cycles: 'checked for loops',
  common_counterparties: 'checked for shared destinations',
  search_patterns: 'searched the pattern library',
}

function describe(action: string): string {
  const m = action.match(/^called (\w+)$/)
  return m ? TOOL_LABELS[m[1]] ?? action : action
}

function detailText(detail: unknown): string | null {
  if (detail == null || detail === '') return null
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.length ? detail.join('; ') : null
  if (typeof detail === 'object') {
    const entries = Object.entries(detail as Record<string, unknown>).filter(([k]) => k !== 'account')
    return entries.length ? entries.map(([k, v]) => `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') || 'none' : v}`).join(', ') : null
  }
  return String(detail)
}

export function CaseFile({ caseId, onChanged }: { caseId: string; onChanged: () => void }) {
  const [c, setCase] = useState<CaseDetail | null>(null)
  const [txns, setTxns] = useState<Txn[]>([])
  const [highlight, setHighlight] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await api.get<CaseDetail>(`/cases/${caseId}`)
      setCase(r.data)
      if (r.data.report) {
        const t = await api.get<Txn[]>(`/cases/${caseId}/transactions`)
        setTxns(t.data)
      }
    } catch (err) {
      setError(errorMessage(err))
    }
  }, [caseId])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!c || !isWorking(c.status)) return
    const id = window.setInterval(load, 2500)
    return () => window.clearInterval(id)
  }, [c, load])

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

  function showTxn(id: string) {
    setHighlight(id)
    document.getElementById(`txn-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  if (!c) return <section className="casefile">{error ? <p className="field-error">{error}</p> : <p className="loading">Opening case…</p>}</section>

  const r = c.report
  const v = c.verification
  const working = isWorking(c.status)

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

        {working && !r && (
          <section className="working">
            <p>The agents are investigating. Their steps appear in the trail below as they happen.</p>
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
                            <button key={t} className="chip" onClick={() => showTxn(t)}>
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
                        <tr key={t.txn_id} id={`txn-${t.txn_id}`} className={highlight === t.txn_id ? 'hl' : ''}>
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
                <p>
                  <strong>{patternName(c.pattern.pattern_id)}</strong>, {Math.round(c.pattern.confidence * 100)}% confident.{' '}
                  {c.pattern.reasoning}
                </p>
                {c.pattern.ruled_out.length > 0 && (
                  <p className="aside">Ruled out: {c.pattern.ruled_out.map(patternName).join(', ')}</p>
                )}
              </section>
            )}
          </>
        )}

        <section>
          <h3>Agent trail</h3>
          <ol className="trail">
            {c.timeline.map((e, i) => {
              const d = detailText(e.detail)
              return (
                <li key={i} className={`a-${e.agent}`}>
                  <time>{shortTime(e.time)}</time>
                  <div>
                    <strong>{AGENT_LABELS[e.agent] ?? e.agent}</strong> {describe(e.action)}
                    {d && <p>{d}</p>}
                  </div>
                </li>
              )
            })}
            {!c.timeline.length && <li className="none">Waiting for the agents to start.</li>}
          </ol>
          {c.usage && (
            <p className="aside">
              {c.tool_calls ?? 0} tool calls, {c.usage.llm_calls} model calls,{' '}
              {(c.usage.input_tokens + c.usage.output_tokens).toLocaleString('en-IN')} tokens, about $
              {c.usage.est_cost_usd.toFixed(4)}, {c.duration_s}s
            </p>
          )}
        </section>
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
          Analyst decision: {c.decision.action === 'approve' ? 'agreed with the agents' : c.decision.action === 'reject' ? 'overruled the agents' : 'asked for more investigation'}
          {c.decision.note ? `. Note: ${c.decision.note}` : ''}
        </p>
      )}
    </section>
  )
}
