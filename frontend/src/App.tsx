import { useCallback, useEffect, useState } from 'react'
import { api, errorMessage, type CaseSummary, type Health, type Run, type Stats } from './api'
import { count } from './format'
import { Queue } from './Queue'
import { CaseFile } from './CaseFile'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [run, setRun] = useState<Run | null>(null)
  const [numCases, setNumCases] = useState(5)
  const [error, setError] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [h, s, c] = await Promise.all([
        api.get<Health>('/health'),
        api.get<Stats>('/stats'),
        api.get<CaseSummary[]>('/cases'),
      ])
      setHealth(h.data)
      setStats(s.data)
      setCases(c.data)
      setError(null)
    } catch (err) {
      setError(errorMessage(err))
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = window.setInterval(refresh, 4000)
    return () => window.clearInterval(id)
  }, [refresh])

  useEffect(() => {
    if (!run || run.status === 'done' || run.status === 'failed') return
    const id = window.setInterval(async () => {
      try {
        const r = await api.get<Run>(`/runs/${run._id}`)
        setRun(r.data)
      } catch {
        /* the main refresh shows connection errors */
      }
    }, 2500)
    return () => window.clearInterval(id)
  }, [run])

  useEffect(() => {
    if (!selected && cases.length) setSelected(cases[0]._id)
  }, [cases, selected])

  async function startRun() {
    setStarting(true)
    try {
      const r = await api.post<Run>('/runs', { max_cases: numCases })
      setRun(r.data)
      setError(null)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setStarting(false)
    }
  }

  const running = run && (run.status === 'triaging' || run.status === 'investigating')
  const runMessage = run
    ? {
        triaging: `Triage agent is reviewing ${count(run.alerts_screened)} alerts…`,
        investigating: `Agents are investigating ${run.case_ids.length} accounts…`,
        done: `Finished. ${run.case_ids.length} case${run.case_ids.length === 1 ? '' : 's'} opened.`,
        failed: `Investigation stopped: ${run.error ?? 'unknown error'}`,
      }[run.status]
    : null

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <svg className="mark" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M8 23 L24 9" />
            <path d="M10 25 L20 16" className="faint" />
          </svg>
          <div>
            <h1>Parakh</h1>
            <p>AI agents that investigate suspicious accounts</p>
          </div>
        </div>

        {stats && (
          <dl className="figures">
            <div>
              <dt>Transactions</dt>
              <dd>{count(stats.transactions)}</dd>
            </div>
            <div>
              <dt>Rule alerts</dt>
              <dd>{count(stats.alerts)}</dd>
            </div>
            <div>
              <dt>Claims fact-checked</dt>
              <dd>
                {stats.claims_total ? `${stats.claims_verified}/${stats.claims_total}` : '–'}
              </dd>
            </div>
            <div>
              <dt>Avg. time per case</dt>
              <dd>{stats.avg_duration_s != null ? `${stats.avg_duration_s}s` : '–'}</dd>
            </div>
          </dl>
        )}

        <form
          className="run"
          onSubmit={(e) => {
            e.preventDefault()
            startRun()
          }}
        >
          <label htmlFor="numCases">Investigate up to</label>
          <input
            id="numCases"
            type="number"
            min={1}
            max={10}
            value={numCases}
            onChange={(e) => setNumCases(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
          />
          <span>alerts</span>
          <button type="submit" className="primary" disabled={!!running || starting || health?.llm_configured === false}>
            {running ? 'Investigating…' : 'Start investigation'}
          </button>
        </form>
      </header>

      {(error || runMessage || health?.llm_configured === false) && (
        <div className={`banner ${error || run?.status === 'failed' ? 'bad' : ''}`} role="status">
          {error ??
            (health?.llm_configured === false
              ? 'The server has no LLM API key yet. Add GROQ_API_KEY to start investigations.'
              : runMessage)}
        </div>
      )}

      <main className="workspace">
        <Queue cases={cases} selected={selected} onSelect={setSelected} onOpened={(id) => { setSelected(id); refresh() }} />
        {selected ? (
          <CaseFile key={selected} caseId={selected} onChanged={refresh} />
        ) : (
          <section className="casefile empty">
            <h2>No cases yet</h2>
            <p>
              Screening rules have flagged {stats ? count(stats.alerts) : 'some'} accounts. Start an investigation and the
              triage agent will pick the alerts most worth a closer look.
            </p>
          </section>
        )}
      </main>
    </div>
  )
}
