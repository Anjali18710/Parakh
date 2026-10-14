import { useState } from 'react'
import { api, errorMessage, type CaseDetail, type CaseSummary } from './api'
import { STATUS_LABELS, isWorking, patternName } from './format'
import { CountUp } from './motion'

type Filter = 'review' | 'all'

interface Props {
  cases: CaseSummary[]
  selected: string | null
  onSelect: (id: string) => void
  onOpened: (id: string) => void
}

export function Queue({ cases, selected, onSelect, onOpened }: Props) {
  const [filter, setFilter] = useState<Filter>('all')
  const [account, setAccount] = useState('')
  const [openError, setOpenError] = useState<string | null>(null)
  const needsReview = cases.filter((c) => c.status === 'awaiting_review').length
  const shown = filter === 'review' ? cases.filter((c) => c.status === 'awaiting_review') : cases

  async function openCase(e: React.FormEvent) {
    e.preventDefault()
    if (!account.trim()) return
    try {
      const r = await api.post<CaseDetail>('/cases', { account: account.trim().toUpperCase() })
      setAccount('')
      setOpenError(null)
      onOpened(r.data._id)
    } catch (err) {
      setOpenError(errorMessage(err))
    }
  }

  return (
    <nav className="queue" aria-label="Cases">
      <div className="queue-head">
        <h2>Cases</h2>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={filter === 'all'} onClick={() => setFilter('all')}>
            All {cases.length}
          </button>
          <button role="tab" aria-selected={filter === 'review'} onClick={() => setFilter('review')}>
            Needs review {needsReview}
          </button>
        </div>
      </div>

      <ul>
        {shown.map((c) => (
          <li key={c._id}>
            <button className={`case-row ${c._id === selected ? 'current' : ''} ${isWorking(c.status) ? 'working' : ''}`} onClick={() => onSelect(c._id)}>
              <span className={`streak ${c.verdict ?? 'pending'}`} aria-hidden="true" />
              <span className="acct">{c.account}</span>
              <span className={`status s-${c.status}`}>
                {isWorking(c.status) && <span className="pulse" aria-hidden="true" />}
                {STATUS_LABELS[c.status] ?? c.status}
              </span>
              <span className="meta">
                {c.verdict
                  ? `${c.verdict === 'suspicious' ? 'Suspicious' : 'Looks innocent'}${c.pattern_id && c.pattern_id !== 'none' ? `, ${patternName(c.pattern_id)}` : ''}`
                  : `Rule score ${c.alert_score ?? '–'}`}
              </span>
              {c.risk_score != null && <span className={`risk ${c.verdict ?? ''}`}><CountUp value={c.risk_score} /></span>}
            </button>
          </li>
        ))}
        {!shown.length && (
          <li className="none">{filter === 'review' ? 'Nothing waiting for you.' : 'Cases appear here as agents open them.'}</li>
        )}
      </ul>

      <form className="open-case" onSubmit={openCase}>
        <label htmlFor="acct">Investigate a specific account</label>
        <div>
          <input id="acct" placeholder="e.g. AC387042" value={account} onChange={(e) => setAccount(e.target.value)} />
          <button type="submit">Open case</button>
        </div>
        {openError && <p className="field-error">{openError}</p>}
      </form>
    </nav>
  )
}
