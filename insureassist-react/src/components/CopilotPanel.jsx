import { AnimatePresence, motion } from 'framer-motion'
import { Bot, RotateCcw, Sparkles } from 'lucide-react'
import IntentGrid from './IntentGrid'
import ThinkingStepper from './ThinkingStepper'
import AnswerCard from './AnswerCard'
import EmptyState from './EmptyState'

export default function CopilotPanel({ state, onAskAI, onManualIntent, onReplay, running }) {
  const activeAnswer = state.answers.find((a) => a.id === state.activeAnswerId)
  // Prefer the live classifier's autoIntents (useLiveCall); fall back to the
  // answer's own intents field for the scripted simulation (useCallSimulation).
  const liveIntents = state.autoIntents && state.autoIntents.length ? state.autoIntents : activeAnswer?.intents || []
  const activeIntents = state.manualIntent ? [state.manualIntent] : liveIntents

  return (
    <section className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200/70 dark:border-white/10
      bg-white/60 dark:glass-dark shadow-lg dark:shadow-glow">
      <div className="flex-none border-b border-slate-200/70 dark:border-white/10 px-4 py-3">
        <h2 className="flex items-center gap-1.5 font-display text-sm font-semibold text-slate-800 dark:text-white">
          <Bot size={15} className="text-indigo-500 dark:text-gold" /> AI Copilot
        </h2>
        <p className="text-[11px] text-slate-500 dark:text-slate-400">Understands what the customer needs, live</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
        <IntentGrid activeKeys={activeIntents} manualIntent={state.manualIntent} onSelect={onManualIntent} />

        <div className="mt-5">
          <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
            Suggested answer
          </h3>

          <ThinkingStepper thinking={state.thinking} />

          <AnimatePresence>
            {state.banner && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mb-3 flex items-center gap-2 rounded-lg border border-amber-200 dark:border-amber-400/20 bg-amber-50 dark:bg-amber-500/10 px-3 py-2 text-[11.5px] text-amber-600 dark:text-amber-300"
              >
                <Sparkles size={13} className="animate-sparkle" /> {state.banner.text}
              </motion.div>
            )}
          </AnimatePresence>

          {state.answers.length === 0 && !state.thinking ? (
            <EmptyState
              icon={Bot}
              title="Suggestions will appear here"
              subtitle="Once the conversation starts, the copilot's answers show up in this feed."
            />
          ) : (
            <div className="flex flex-col gap-3">
              <AnimatePresence initial={false}>
                {state.answers.map((answer) => (
                  <AnswerCard key={answer.id} answer={answer} onAskAI={onAskAI} />
                ))}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-none items-center justify-between border-t border-slate-200/70 dark:border-white/10 px-4 py-3">
        <span className="text-[11px] text-slate-400 dark:text-slate-500">{state.progress}</span>
        <button
          onClick={onReplay}
          disabled={!running}
          className="flex items-center gap-1.5 rounded-lg bg-indigo-500 dark:bg-gold px-3 py-1.5 text-[11.5px] font-semibold
            text-white dark:text-ink disabled:opacity-40 disabled:cursor-not-allowed hover:brightness-105"
        >
          <RotateCcw size={13} /> Replay this call
        </button>
      </div>
    </section>
  )
}
