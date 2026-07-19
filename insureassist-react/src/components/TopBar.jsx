import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { PhoneOff } from 'lucide-react'
import ThemeToggle from './ThemeToggle'

function useElapsed(running) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (!running) { setSeconds(0); return }
    const start = Date.now()
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(id)
  }, [running])
  const m = String(Math.floor(seconds / 60)).padStart(2, '0')
  const s = String(seconds % 60).padStart(2, '0')
  return `${m}:${s}`
}

export default function TopBar({ running, onEndCall }) {
  const timer = useElapsed(running)

  return (
    <header className="flex items-center justify-between border-b border-slate-200/70 dark:border-white/10 bg-white/60 dark:bg-black/20 px-5 py-3 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 200, damping: 14 }}
          className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-teal-400 shadow-lg shadow-indigo-500/20"
        >
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
            <path d="M12 2 3 6v6c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6l-9-4Z" fill="white" fillOpacity="0.15" stroke="white" strokeWidth="1.4" />
            <path d="M9 12.5 11 14.5 15.5 9.5" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </motion.div>
        <div>
          <p className="font-display text-[15px] font-semibold leading-tight text-slate-900 dark:text-white">
            InsureAssist AI
          </p>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">Real-time assistant for your customer calls</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <span className="hidden sm:inline text-xs text-slate-500 dark:text-slate-400">Call ID #8f2c-91a</span>
        <span className="font-mono text-[13px] tabular-nums text-slate-700 dark:text-slate-200">{timer}</span>
        <ThemeToggle />
        <button
          onClick={onEndCall}
          className="flex items-center gap-1.5 rounded-lg border border-rose-300/50 dark:border-rose-500/30 bg-rose-50 dark:bg-rose-500/10
            px-3 py-1.5 text-xs font-semibold text-rose-500 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-500/20 transition-colors"
        >
          <PhoneOff size={13} /> End call
        </button>
      </div>
    </header>
  )
}
