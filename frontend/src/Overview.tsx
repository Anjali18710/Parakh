import { useState } from 'react'
import { api, errorMessage, type CaseDetail, type CaseSummary, type Stats } from './api'
import { count } from './format'
import { CountUp } from './motion'

const SAMPLE_ACCOUNT = 'AC387042' // a gather-scatter hub from the demo data

const STEPS = [
  ['Investigator', 'Follows the money with five tools, within a fixed budget of calls.'],
  ['Pattern', 'Matches the flow to a laundering typology. Code first rules out the ones its shape makes impossible.'],
  ['Report', 'Writes the case file. Every claim cites the transactions behind it.'],
  ['Fact-check', 'Python recomputes every amount and count. Failing reports go back for a rewrite.'],
  ['You', 'Agree, overrule, or send the agents back. Confirmed cases export an STR.'],
]

const SAFEGUARDS = [
  ['Prompt-injection defence', 'Payment remarks that try to instruct the AI are withheld from the agents and shown to you.'],
  ['Learns from your decisions', 'Past decisions on similar accounts are shown to the agents, but can never quietly clear a case.'],
  ['Survives restarts', 'Paused investigations are saved in MongoDB and resume where they stopped.'],
]

/** The case to show a first-time visitor: the strongest finished investigation. */
function sampleCase(cases: CaseSummary[]): CaseSummary | undefined {
  const rank = (c: CaseSummary) =>
    (c.status === 'confirmed_suspicious' ? 3 : c.status === 'awaiting_review' ? 2 : c.verdict ? 1 : 0) * 1000 +
    (c.risk_score ?? 0)
  return [...cases].sort((a, b) => rank(b) - rank(a))[0]
}

export function Overview({ stats, cases, onOpen }: {
  stats: Stats | null
  cases: CaseSummary[]
  onOpen: (id: string) => void
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const by = stats?.by_status ?? {}
  const sample = sampleCase(cases)

  async function investigateSample() {
    setBusy(true)
    try {
      const r = await api.post<CaseDetail>('/cases', { account: SAMPLE_ACCOUNT })
      onOpen(r.data._id)
    } catch (err) {
      setError(errorMessage(err))
      setBusy(false)
    }
  }

  const figures: [string, number | null, string][] = [
    ['Rule alerts', stats?.alerts ?? null, 'accounts flagged by screening'],
    ['Investigated', stats?.cases ?? null, 'cases opened by the agents'],
    ['Waiting for you', by.awaiting_review ?? 0, 'cases that need a decision'],
    ['Confirmed suspicious', (by.confirmed_suspicious ?? 0) + (by.auto_suspicious ?? 0), 'ready for an STR'],
    ['Cleared', (by.closed_not_suspicious ?? 0) + (by.auto_not_suspicious ?? 0), 'closed as innocent'],
  ]

  return (
    <section className="overview">
      <div className="intro">
        <h2>AI agents that investigate accounts flagged for money laundering</h2>
        <p>
          Screening rules flag accounts, and most are false alarms. Parakh's agents follow the money, match it to known
          laundering patterns and write a case file where every amount and count is checked against the transactions.
          You make the final call.
        </p>
        <div className="intro-actions">
          {sample ? (
            <button className="primary" onClick={() => onOpen(sample._id)}>Open a sample case</button>
          ) : (
            <button className="primary" onClick={investigateSample} disabled={busy}>
              {busy ? 'Opening…' : `Investigate a sample account (${SAMPLE_ACCOUNT})`}
            </button>
          )}
          <span className="hint">or press <b>Start investigation</b> above to let the triage agent pick alerts</span>
        </div>
        {error && <p className="field-error">{error}</p>}
      </div>

      <dl className="figures-grid">
        {figures.map(([label, value, note]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value == null ? '–' : <CountUp value={value} format={count} />}</dd>
            <span>{note}</span>
          </div>
        ))}
        <div>
          <dt>Claims fact-checked</dt>
          <dd>{stats?.claims_total ? `${stats.claims_verified}/${stats.claims_total}` : '–'}</dd>
          <span>verified against the transactions</span>
        </div>
      </dl>

      <h3>From alert to decision</h3>
      <ol className="steps">
        {STEPS.map(([name, text], i) => (
          <li key={name} style={{ animationDelay: `${120 + i * 70}ms` }}>
            <b>{i + 1}</b>
            <strong>{name}</strong>
            <span>{text}</span>
          </li>
        ))}
      </ol>

      <h3>Safeguards</h3>
      <ul className="safeguards">
        {SAFEGUARDS.map(([name, text]) => (
          <li key={name}>
            <strong>{name}</strong>
            <span>{text}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
