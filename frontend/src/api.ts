import axios from 'axios'

const baseURL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:8000'

export const api = axios.create({ baseURL: `${baseURL}/api`, timeout: 30000 })

export type CaseStatus =
  | 'queued'
  | 'investigating'
  | 'closing'
  | 'awaiting_review'
  | 'confirmed_suspicious'
  | 'closed_not_suspicious'
  | 'auto_suspicious'
  | 'auto_not_suspicious'
  | 'superseded'
  | 'failed'

export interface Health {
  status: string
  llm_provider: string
  model: string
  llm_configured: boolean
  storage: string
  verifier_enabled: boolean
}

export interface Stats {
  transactions: number
  accounts: number
  alerts: number
  cases: number
  by_status: Record<string, number>
  avg_cost_usd: number | null
  avg_duration_s: number | null
  claims_verified: number
  claims_total: number
}

export interface CaseSummary {
  _id: string
  account: string
  status: CaseStatus
  mode: string
  created_at: string
  updated_at: string
  verdict: 'suspicious' | 'not_suspicious' | null
  risk_score: number | null
  pattern_id: string | null
  claims_verified: number | null
  claims_total: number | null
  alert_score: number | null
  duration_s: number | null
  error: string | null
}

export interface Signal {
  name: string
  detail: string
  txn_ids?: string[]
}

export interface Claim {
  statement: string
  txn_ids: string[]
  accounts: string[]
  amount: number | null
  count: number | null
}

export interface Check {
  description: string
  tool: string
}

export interface Report {
  verdict: 'suspicious' | 'not_suspicious'
  risk_score: number
  pattern_id: string
  summary: string
  claims: Claim[]
  checks?: Check[]
  recommended_action: string
}

export interface Verification {
  passed: boolean
  claims_total: number
  claims_verified: number
  claim_results: { index: number; ok: boolean; problems: string[] }[]
  issues: string[]
  enforced: boolean
  checks_total?: number
  checks_verified?: number
  check_results?: { index: number; ok: boolean; problems: string[] }[]
}

export interface FlowParty {
  account: string
  total: number
  count: number
  cited: boolean
  txn_ids: string[]
  first: string
  last: string
  other?: boolean
}

export interface Flow {
  account: string
  window?: { start: string; end: string }
  total_in: number
  total_out: number
  passed_through: number
  senders: number
  recipients: number
  span_hours: number
  inflows: FlowParty[]
  outflows: FlowParty[]
}

export interface TimelineEvent {
  time: string
  agent: string
  action: string
  detail: unknown
}

export interface CaseDetail extends Omit<CaseSummary, 'verdict' | 'risk_score' | 'pattern_id' | 'claims_verified' | 'claims_total' | 'alert_score'> {
  alert: { account: string; score: number; signals: Signal[] }
  triage: { reason: string; priority: number } | null
  report: Report | null
  pattern: {
    pattern_id: string
    kind: string
    confidence: number
    reasoning: string
    ruled_out: string[]
    shape?: Record<string, unknown>
  } | null
  verification: Verification | null
  timeline: TimelineEvent[]
  investigation_summary: string | null
  tool_calls: number | null
  decision: { action: string; note: string | null; at: string } | null
  analyst_notes: string[]
  usage: { llm_calls: number; input_tokens: number; output_tokens: number; est_cost_usd: number } | null
}

export interface Txn {
  txn_id: string
  timestamp: string
  from_account: string
  to_account: string
  amount: number
  payment_format: string
}

export interface Run {
  _id: string
  status: 'triaging' | 'investigating' | 'done' | 'failed'
  case_ids: string[]
  error?: string
  alerts_screened: number
}

export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (!err.response) return 'Cannot reach the Parakh server. It may be starting up; try again in a minute.'
    return `Request failed (${err.response.status}).`
  }
  return String(err)
}
