import { memo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, CheckCircle2, Lightbulb, MessageCircleQuestion } from 'lucide-react'
import CopyButton from './CopyButton'

const AnswerCard = memo(function AnswerCard({ answer, onAskAI }) {
  const visibleText = answer.text.slice(0, answer.revealed)
  const isDone = answer.status === 'done'
  const isSuperseded = answer.status === 'superseded'

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 10, scale: 0.98 }}
      animate={{ opacity: isSuperseded ? 0.45 : 1, y: 0, scale: 1 }}
      transition={{ duration: 0.3 }}
      className={`rounded-2xl border bg-white/70 dark:bg-panel-2 shadow-md dark:shadow-glow overflow-hidden
        ${isSuperseded ? 'border-slate-200 dark:border-white/5 saturate-50' : 'border-slate-200 dark:border-white/10'}`}
    >
      <header className="flex items-center justify-between border-b border-slate-200/70 dark:border-white/10 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-400/20 to-purple-400/20 dark:from-gold/15 dark:to-gold/5 text-indigo-500 dark:text-gold">
            <Sparkles size={14} />
          </span>
          <div>
            <p className="font-display text-[13px] font-semibold text-slate-800 dark:text-white">Suggested response</p>
            <p className="flex items-center gap-1 text-[10.5px] font-medium text-emerald-500 dark:text-emerald-400">
              <CheckCircle2 size={11} /> Strong match
            </p>
          </div>
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide
            ${isSuperseded ? 'bg-rose-50 dark:bg-rose-500/10 text-rose-500 dark:text-rose-400'
              : isDone ? 'bg-slate-100 dark:bg-black/30 text-slate-500 dark:text-slate-400'
              : 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-500 dark:text-emerald-400'}`}
        >
          {isSuperseded ? 'updated' : isDone ? 'done' : 'live'}
        </span>
      </header>

      <div className="px-4 py-3 text-[13.5px] leading-relaxed text-slate-700 dark:text-slate-200">
        {visibleText}
        {!isDone && <span className="ml-0.5 inline-block h-3.5 w-[2px] animate-pulse bg-slate-400 align-middle" />}
      </div>

      <AnimatePresence>
        {isDone && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} transition={{ duration: 0.3 }}>
            <div className="px-4 pb-1">
              <p className="mb-1.5 flex items-center gap-1.5 text-[10.5px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                <Lightbulb size={11} /> Why this answer
              </p>
              <ul className="space-y-1">
                {answer.reasons.map((r, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-[12px] text-slate-500 dark:text-slate-400">
                    <CheckCircle2 size={12} className="mt-0.5 flex-none text-emerald-400" />
                    {r}
                  </li>
                ))}
              </ul>
            </div>

            {answer.nextQuestion && (
              <div className="mx-4 mb-3 mt-2.5 flex gap-2 rounded-xl border border-indigo-100 dark:border-blue-400/20 bg-indigo-50/70 dark:bg-blue-500/10 px-3 py-2.5">
                <MessageCircleQuestion size={15} className="mt-0.5 flex-none text-indigo-400 dark:text-blue-300" />
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-indigo-400 dark:text-blue-300">Next question to ask</p>
                  <p className="text-[12.5px] leading-relaxed text-slate-600 dark:text-slate-200">{answer.nextQuestion}</p>
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-200/70 dark:border-white/10 px-4 py-3">
              <div className="flex flex-wrap gap-1.5">
                {answer.sources.map((s) => (
                  <span key={s} className="rounded-md border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-black/20 px-2 py-0.5 text-[10.5px] text-slate-500 dark:text-slate-400">
                    {s}
                  </span>
                ))}
              </div>
              <div className="flex items-center gap-2">
                <CopyButton text={answer.text} />
                <button
                  onClick={() => onAskAI(answer)}
                  className="rounded-lg bg-indigo-500 dark:bg-gold px-3 py-1.5 text-[11px] font-semibold text-white dark:text-ink hover:brightness-105"
                >
                  Ask AI
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  )
})

export default AnswerCard
