import { AnimatePresence, motion } from 'framer-motion'
import { Check, Loader2, BrainCircuit } from 'lucide-react'

export default function ThinkingStepper({ thinking }) {
  return (
    <AnimatePresence>
      {thinking && (
        <motion.div
          initial={{ opacity: 0, height: 0, marginBottom: 0 }}
          animate={{ opacity: 1, height: 'auto', marginBottom: 14 }}
          exit={{ opacity: 0, height: 0, marginBottom: 0 }}
          transition={{ duration: 0.3 }}
          className="overflow-hidden rounded-xl border border-slate-200 dark:border-white/10 bg-white/60 dark:bg-panel-2 px-4 py-3"
        >
          <div className="mb-2.5 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
            <Loader2 size={12} className="animate-spin text-indigo-400 dark:text-gold" />
            {thinking.mode === 'update' ? 'Updating' : 'Working on it'}
          </div>
          <div className="flex flex-col gap-1.5">
            {thinking.steps.map((step, idx) => (
              <div
                key={idx}
                className={`flex items-center gap-2.5 text-[12.5px] transition-colors
                  ${step.status === 'pending' ? 'text-slate-400 dark:text-slate-500' : 'text-slate-700 dark:text-slate-200'}`}
              >
                <span
                  className={`flex h-4 w-4 flex-none items-center justify-center rounded-full border transition-colors
                    ${step.status === 'done' ? 'border-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-500'
                      : step.status === 'active' ? 'border-indigo-400 dark:border-gold' : 'border-slate-300 dark:border-slate-700'}`}
                >
                  {step.status === 'done' && <Check size={10} />}
                  {step.status === 'active' && <BrainCircuit size={10} className="text-indigo-400 dark:text-gold" />}
                </span>
                {step.label}
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
