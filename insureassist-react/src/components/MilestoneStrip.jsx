import { AnimatePresence, motion } from 'framer-motion'
import { Flag } from 'lucide-react'

export default function MilestoneStrip({ milestones }) {
  return (
    <div className="flex gap-2 overflow-x-auto border-b border-slate-200/70 dark:border-white/10 px-4 py-2.5 scrollbar-thin">
      {milestones.length === 0 && (
        <span className="flex items-center gap-1.5 py-1 text-[11px] text-slate-400 dark:text-slate-500">
          <Flag size={12} /> Call milestones will appear here
        </span>
      )}
      <AnimatePresence initial={false}>
        {milestones.map((m) => (
          <motion.span
            key={m.id}
            initial={{ opacity: 0, scale: 0.9, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className="flex flex-none items-center gap-1.5 whitespace-nowrap rounded-full border border-slate-200 dark:border-white/10
              bg-white/70 dark:bg-panel-2 px-3 py-1 text-[11px] text-slate-600 dark:text-slate-300"
          >
            <Flag size={11} className="text-indigo-400 dark:text-gold" />
            {m.text}
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  )
}
