const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 })

export const rupees = (n: number) => inr.format(n)

export const count = (n: number) => new Intl.NumberFormat('en-IN').format(n)

export function shortTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('en-IN', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false })
}

export const PATTERN_NAMES: Record<string, string> = {
  fan_out: 'Fan-out',
  fan_in: 'Fan-in',
  cycle: 'Round-trip loop',
  scatter_gather: 'Scatter-gather',
  gather_scatter: 'Gather-scatter hub',
  layering_chain: 'Layering chain',
  stack: 'Stacked layers',
  bipartite: 'Bipartite transfer',
  payroll: 'Payroll',
  wholesale: 'Wholesale trade',
  investment: 'Investment transfer',
  property: 'Property purchase',
  family: 'Family transfers',
  none: 'No clear pattern',
}

export const patternName = (id: string | null | undefined) => (id ? PATTERN_NAMES[id] ?? id : '')

export const STATUS_LABELS: Record<string, string> = {
  queued: 'Queued',
  investigating: 'Agents working',
  closing: 'Closing',
  awaiting_review: 'Needs your review',
  confirmed_suspicious: 'Confirmed suspicious',
  closed_not_suspicious: 'Closed, not suspicious',
  auto_suspicious: 'Suspicious',
  auto_not_suspicious: 'Not suspicious',
  superseded: 'Reopened as new case',
  failed: 'Failed',
}

export const SIGNAL_LABELS: Record<string, string> = {
  fan_out_burst: 'Paid many accounts in a day',
  fan_in_burst: 'Paid by many accounts in a day',
  rapid_pass_through: 'Money left almost as soon as it arrived',
  near_threshold: 'Payments just under ₹10 lakh',
  in_cycle: 'Part of a loop of transfers',
}

export const AGENT_LABELS: Record<string, string> = {
  investigator: 'Investigator',
  analyst: 'Single agent',
  pattern: 'Pattern agent',
  report: 'Report agent',
  verifier: 'Fact-checker',
  reviewer: 'Human analyst',
  system: 'System',
}

export const isWorking = (s: string) => s === 'queued' || s === 'investigating' || s === 'closing'
