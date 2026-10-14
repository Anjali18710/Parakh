import type { CaseDetail, TimelineEvent } from './api'
import { patternName } from './format'

type StageId = 'investigate' | 'pattern' | 'report' | 'check' | 'review'
type StageState = 'waiting' | 'active' | 'done' | 'failed'

interface Stage {
  id: StageId
  name: string
  state: StageState
  note: string
}

const ORDER: StageId[] = ['investigate', 'pattern', 'report', 'check', 'review']

function stageOf(e: TimelineEvent): StageId | null {
  if (e.agent === 'investigator' || e.agent === 'analyst' || e.agent === 'guard') return 'investigate'
  if (e.agent === 'pattern' || (e.agent === 'verifier' && e.action.includes('pattern'))) return 'pattern'
  if (e.agent === 'report') return 'report'
  if (e.agent === 'verifier') return 'check'
  if (e.agent === 'reviewer') return 'review'
  return null
}

/** Which stage the agents are on, worked out from the events they have emitted so far. */
function currentStage(events: TimelineEvent[], stages: StageId[]): StageId {
  const last = [...events].reverse().find((e) => stageOf(e))
  if (!last) return stages[0]
  const s = stageOf(last)!
  const next = (id: StageId) => stages[Math.min(stages.indexOf(id) + 1, stages.length - 1)]
  if (s === 'investigate') return last.action.startsWith('finished') || last.agent === 'guard' ? next('investigate') : 'investigate'
  if (s === 'pattern') return last.agent === 'pattern' ? next('pattern') : 'pattern' // a rejected pick means it is trying again
  if (s === 'report') return 'check'
  if (s === 'check') return last.action.includes('problem') ? 'report' : 'review' // sent back for a rewrite
  return s === 'review' && last.action.includes('reinvestigate') ? stages[0] : 'review'
}

export function stagesFor(c: CaseDetail): Stage[] {
  const single = c.mode === 'single'
  const ids = single ? ORDER.filter((s) => s !== 'pattern') : ORDER
  // After "dig further", only the latest round counts.
  const lastRestart = c.timeline.map((e) => e.action).lastIndexOf('decided: reinvestigate')
  const round = c.timeline.slice(lastRestart + 1)

  const tools = round.filter((e) => e.action.startsWith('called ') && (e.agent === 'investigator' || e.agent === 'analyst')).length
  const rewrites = round.filter((e) => e.action.startsWith('rewrote')).length
  const rejections = round.filter((e) => e.agent === 'verifier' && e.action.includes('pattern')).length
  const withheld = c.injections?.length ?? 0
  const v = c.verification

  let active: StageId | null = null
  let finishedAll = false
  if (c.status === 'queued') active = null
  else if (c.status === 'investigating' || c.status === 'failed') active = currentStage(round, ids)
  else if (c.status === 'awaiting_review' || c.status === 'closing') active = 'review'
  else finishedAll = true

  const notes: Record<StageId, string> = {
    investigate: [tools ? `${tools} tool call${tools === 1 ? '' : 's'}` : '', withheld ? `${withheld} remark${withheld === 1 ? '' : 's'} withheld` : '']
      .filter(Boolean).join(' · '),
    pattern: c.pattern ? patternName(c.pattern.pattern_id) : rejections ? `${rejections} pick${rejections === 1 ? '' : 's'} rejected` : '',
    report: c.report ? `${c.report.claims.length} claim${c.report.claims.length === 1 ? '' : 's'}${rewrites ? ` · rewritten ${rewrites}×` : ''}` : rewrites ? `rewrite ${rewrites}` : '',
    check: v ? `${v.claims_verified}/${v.claims_total} verified` : '',
    review:
      c.status === 'awaiting_review' ? 'your call'
        : c.status === 'closing' ? 'closing'
          : c.decision?.action === 'approve' ? 'agreed'
            : c.decision?.action === 'reject' ? 'overruled'
              : c.status.startsWith('auto_') ? 'not needed' : '',
  }
  const names: Record<StageId, string> = {
    investigate: single ? 'Single agent' : 'Investigator', pattern: 'Pattern', report: 'Report', check: 'Fact-check', review: 'You',
  }
  const activeIdx = active ? ids.indexOf(active) : -1
  return ids.map((id, i) => ({
    id,
    name: names[id],
    note: notes[id],
    state: finishedAll ? 'done'
      : c.status === 'failed' && id === active ? 'failed'
        : i < activeIdx ? 'done'
          : id === active ? 'active' : 'waiting',
  }))
}

const Tick = () => (
  <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5 L6.5 11.5 L12.5 4.5" /></svg>
)

/** The five stages of a case, lit up as the agents move through them. */
export function Pipeline({ c }: { c: CaseDetail }) {
  const stages = stagesFor(c)
  const rewrites = c.timeline.filter((e) => e.action.startsWith('rewrote')).length
  const current = stages.find((s) => s.state === 'active')
  return (
    <div className="pipeline-wrap">
      <ol className="pipeline" aria-label={current ? `Now: ${current.name}` : 'Case stages'}>
        {stages.map((s, i) => (
          <li key={s.id} className={`stage ${s.state} st-${s.id}`} aria-current={s.state === 'active' ? 'step' : undefined}>
            <span className="node" aria-hidden="true">
              {s.state === 'done' ? <Tick /> : s.state === 'failed' ? '!' : i + 1}
            </span>
            <span className="stage-name">{s.name}</span>
            <span className="stage-note">{s.note || ' '}</span>
            {s.id === 'check' && rewrites > 0 && (
              <span className="loop" title={`The fact-checker sent the report back ${rewrites} time${rewrites === 1 ? '' : 's'}`}>
                ↺ sent back {rewrites}×
              </span>
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}
