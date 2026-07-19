import { motion } from 'framer-motion'

export default function EmptyState({ icon: Icon, title, subtitle }) {
  return (
    <div className="m-auto flex max-w-[240px] flex-col items-center gap-3 py-10 text-center">
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-400/20 to-teal-400/20 dark:from-gold/10 dark:to-teal/10 border border-slate-200/60 dark:border-white/10"
      >
        <Icon size={26} className="text-indigo-400 dark:text-gold" aria-hidden="true" />
      </motion.div>
      <p className="text-sm font-medium text-slate-600 dark:text-slate-300">{title}</p>
      {subtitle && <p className="text-xs text-slate-400 dark:text-slate-500">{subtitle}</p>}
    </div>
  )
}
