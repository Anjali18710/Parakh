import { useEffect, useRef, useState } from 'react'

/** True when the person has asked their system for less motion. Checked live, so toggling it applies at once. */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/** Animate a number from its previous value to `target` (ease-out). Jumps straight there when motion is reduced. */
export function useCountUp(target: number, ms = 900): number {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0))
  const current = useRef(prefersReducedMotion() ? target : 0)
  useEffect(() => {
    if (prefersReducedMotion() || !Number.isFinite(target)) {
      current.current = target
      setValue(target)
      return
    }
    const from = current.current
    if (from === target) return
    const start = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const k = Math.min(1, (now - start) / ms)
      const eased = 1 - Math.pow(1 - k, 3)
      current.current = from + (target - from) * eased
      setValue(current.current)
      if (k < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return value
}

/** A number that counts up to its value. `format` receives the in-between values too. */
export function CountUp({ value, format = (n) => Math.round(n).toString(), ms }: {
  value: number
  format?: (n: number) => string
  ms?: number
}) {
  const v = useCountUp(value, ms)
  // Screen readers get the final value, not every frame.
  return (
    <>
      <span aria-hidden="true">{format(v)}</span>
      <span className="sr-only">{format(value)}</span>
    </>
  )
}

/** Risk score as a ring that fills to the score, coloured by verdict. */
export function RiskRing({ score, verdict }: { score: number; verdict: string }) {
  const v = useCountUp(score, 1100)
  const r = 26
  const len = 2 * Math.PI * r
  return (
    <div className={`ring ${verdict}`} role="img" aria-label={`Risk score ${score} out of 100`}>
      <svg viewBox="0 0 64 64" aria-hidden="true">
        <circle cx="32" cy="32" r={r} className="ring-track" />
        <circle cx="32" cy="32" r={r} className="ring-fill" strokeDasharray={len}
                strokeDashoffset={len * (1 - v / 100)} transform="rotate(-90 32 32)" />
      </svg>
      <span aria-hidden="true">{Math.round(v)}</span>
    </div>
  )
}
