import { useEffect, useState } from 'react'
import { api, errorMessage } from './api'
import { patternName } from './format'

interface ModeResult {
  label: string
  cases: number
  errors: number
  tp: number
  fp: number
  fn: number
  tn: number
  precision: number
  recall: number
  f1: number
  innocent_cleared: number | null
  pattern_accuracy: number | null
  claims_total: number
  claims_verified: number
  claims_corrected: number
  claims_removed: number
  avg_tokens: number
  avg_cost_usd: number
  avg_seconds: number
  false_alarms_by_role: Record<string, number>
  missed_by_pattern: Record<string, number>
}

interface Summary {
  name: string
  data: string
  model: string
  sample_size: number
  laundering_in_sample: number
  modes: string[]
  updated_at: string
  complete: boolean
  screening: { alerts: number; laundering_accounts: number; laundering_accounts_alerted: number }
  results: Record<string, ModeResult>
}

const MODE_LABELS: Record<string, string> = {
  rules: 'Screening rules alone',
  single: 'One agent with every tool',
  no_guardrails: 'Agent team, agents decide everything',
  full: 'Agent team with shape rules and fact-check loop',
}

const MODE_ISOLATES: Record<string, string> = {
  rules: 'The baseline every bank already has.',
  single: 'Does splitting the work across specialised agents help?',
  no_guardrails: 'What the agents get right on their own, with no code checking them.',
  full: 'What the code-enforced shape rules and fact-check loop add on top.',
}

const pct = (v: number | null | undefined) => (v == null ? '–' : `${Math.round(v * 100)}%`)

function Bar({ value, strong }: { value: number | null | undefined; strong?: boolean }) {
  if (value == null) return <span className="muted">–</span>
  return (
    <span className="bar-cell">
      <span className="bar-track" aria-hidden="true">
        <span className={`bar-fill ${strong ? 'strong' : ''}`} style={{ width: `${Math.max(2, value * 100)}%` }} />
      </span>
      <span className="bar-value">{pct(value)}</span>
    </span>
  )
}

export function Evaluation() {
  const [runs, setRuns] = useState<Summary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState<string | null>(null)

  useEffect(() => {
    api
      .get<Summary[]>('/evaluations')
      .then((r) => {
        setRuns(r.data)
        if (r.data[0]) setName(r.data[0].name)
      })
      .catch((err) => setError(errorMessage(err)))
  }, [])

  if (error) return <section className="eval"><p className="field-error">{error}</p></section>
  if (!runs) return <section className="eval"><p className="loading">Loading evaluation…</p></section>
  if (!runs.length) {
    return (
      <section className="eval">
        <h2>No evaluation yet</h2>
        <p className="lead">
          Run the experiment from the backend folder. It saves after every case and resumes where it stopped, so it can
          run over several days on a free API tier.
        </p>
        <pre>python -m parakh.eval.run_eval --name demo --sample 30</pre>
      </section>
    )
  }

  const run = runs.find((r) => r.name === name) ?? runs[0]
  const pending = run.modes.filter((m) => !run.results[m])
  const rows = run.modes.filter((m) => run.results[m]).map((m) => [m, run.results[m]] as const)
  const cov = run.screening

  return (
    <section className="eval">
      <header className="eval-head">
        <div>
          <h2>How well does it work?</h2>
          <p className="lead">
            The same {run.sample_size} rule alerts ({run.laundering_in_sample} truly laundering, the rest innocent
            look-alikes) go through each setup, and every verdict is checked against the answer key in the data.
          </p>
        </div>
        {runs.length > 1 && (
          <label className="run-pick">
            Run
            <select value={run.name} onChange={(e) => setName(e.target.value)}>
              {runs.map((r) => (
                <option key={r.name} value={r.name}>{r.name}</option>
              ))}
            </select>
          </label>
        )}
      </header>

      <p className="eval-meta">
        Data <strong>{run.data}</strong>, model <strong>{run.model}</strong>, updated{' '}
        {new Date(run.updated_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}.{' '}
        {run.complete ? 'All setups finished.' : 'Still running: results so far.'}
      </p>

      <div className="coverage">
        <strong>{pct(cov.laundering_accounts_alerted / cov.laundering_accounts)}</strong>
        <p>
          of laundering accounts are flagged by the screening rules at all ({cov.laundering_accounts_alerted} of{' '}
          {cov.laundering_accounts}). The agents can only investigate what the rules flag, so this caps what any setup
          below can catch.
        </p>
      </div>

      <h3>Verdicts</h3>
      <div className="table-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th scope="col">Setup</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall</th>
              <th scope="col">Innocent cleared</th>
              <th scope="col">Pattern right</th>
              <th scope="col" className="num">Cases</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([mode, m]) => (
              <tr key={mode} className={mode === 'full' ? 'ours' : ''}>
                <th scope="row">
                  {m.label}
                  <small>{MODE_ISOLATES[mode]}</small>
                </th>
                <td><Bar value={m.precision} strong={mode === 'full'} /></td>
                <td><Bar value={m.recall} strong={mode === 'full'} /></td>
                <td><Bar value={m.innocent_cleared} strong={mode === 'full'} /></td>
                <td><Bar value={m.pattern_accuracy} strong={mode === 'full'} /></td>
                <td className="num">{m.cases}{m.errors ? ` (${m.errors} failed)` : ''}</td>
              </tr>
            ))}
            {pending.map((mode) => (
              <tr key={mode} className="pending">
                <th scope="row">
                  {MODE_LABELS[mode] ?? mode}
                  <small>{MODE_ISOLATES[mode]}</small>
                </th>
                <td colSpan={5}>Not run yet</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="aside">
        Precision: of the accounts called suspicious, how many really were. Recall: of the laundering accounts, how
        many were caught. Innocent cleared: of the innocent look-alikes, how many were correctly closed. Pattern right:
        of the laundering accounts caught, how many got the right typology.
      </p>

      <h3>Evidence and cost</h3>
      <div className="table-wrap">
        <table className="eval-table">
          <thead>
            <tr>
              <th scope="col">Setup</th>
              <th scope="col" className="num">Claims verified</th>
              <th scope="col" className="num">Corrected</th>
              <th scope="col" className="num">Removed</th>
              <th scope="col" className="num">Tokens per case</th>
              <th scope="col" className="num">Cost per case</th>
              <th scope="col" className="num">Time per case</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([mode, m]) => (
              <tr key={mode} className={mode === 'full' ? 'ours' : ''}>
                <th scope="row">{m.label}</th>
                <td className="num">{m.claims_total ? `${m.claims_verified} of ${m.claims_total}` : '–'}</td>
                <td className="num">{m.claims_total ? m.claims_corrected : '–'}</td>
                <td className="num">{m.claims_total ? m.claims_removed : '–'}</td>
                <td className="num">{m.avg_tokens ? m.avg_tokens.toLocaleString('en-IN') : '–'}</td>
                <td className="num">{m.avg_tokens ? `$${m.avg_cost_usd.toFixed(4)}` : '–'}</td>
                <td className="num">{m.avg_tokens ? `${m.avg_seconds}s` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Where each setup goes wrong</h3>
      <div className="mistakes">
        {rows.map(([mode, m]) => {
          const alarms = Object.entries(m.false_alarms_by_role).sort((a, b) => b[1] - a[1])
          const missed = Object.entries(m.missed_by_pattern).sort((a, b) => b[1] - a[1])
          return (
            <div key={mode} className="mistake">
              <h4>{m.label}</h4>
              <p>
                <span className="muted">False alarms:</span>{' '}
                {alarms.length ? alarms.map(([role, n]) => `${n} ${role}${n === 1 ? '' : 's'}`).join(', ') : 'none'}
              </p>
              <p>
                <span className="muted">Missed:</span>{' '}
                {missed.length ? missed.map(([p, n]) => `${n} ${patternName(p).toLowerCase()}`).join(', ') : 'none'}
              </p>
            </div>
          )
        })}
      </div>
    </section>
  )
}
