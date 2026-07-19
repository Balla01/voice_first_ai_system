import { useState } from 'react'
import { Copy, Check } from 'lucide-react'
import { motion } from 'framer-motion'

export default function CopyButton({ text, label = 'Copy', className = '' }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1400)
    } catch {
      // clipboard API unavailable — silently ignore in this prototype
    }
  }

  return (
    <motion.button
      whileTap={{ scale: 0.94 }}
      onClick={handleCopy}
      aria-label={copied ? 'Copied to clipboard' : label}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition-colors
        ${copied
          ? 'border-emerald-400/40 text-emerald-500 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10'
          : 'border-slate-200 dark:border-white/10 text-slate-500 dark:text-slate-300 hover:text-slate-800 dark:hover:text-white hover:border-slate-400 dark:hover:border-white/30'}
        ${className}`}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
      {copied ? 'Copied' : label}
    </motion.button>
  )
}
