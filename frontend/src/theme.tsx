import { useState } from 'react'

type Theme = 'light' | 'dark'
const KEY = 'parakh-theme'

function stored(): Theme | null {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null
  }
}

const systemTheme = (): Theme => (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')

/** Apply a saved choice before React renders, so the page never flashes the wrong theme. */
export function applySavedTheme() {
  const t = stored()
  if (t) document.documentElement.dataset.theme = t
}

/** Light/dark switch. Until it is used, the page follows the system setting. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => stored() ?? systemTheme())
  function flip() {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    try {
      localStorage.setItem(KEY, next)
    } catch {
      /* private mode: the choice lasts until reload */
    }
    setTheme(next)
  }
  const label = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
  return (
    <button className="theme-toggle" onClick={flip} aria-label={label} title={label}>
      {theme === 'dark' ? (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" /></svg>
      )}
    </button>
  )
}
