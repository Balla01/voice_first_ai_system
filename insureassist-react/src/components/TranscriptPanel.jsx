import { useEffect, useRef, memo } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Headphones, User, Bot } from 'lucide-react'
import MilestoneStrip from './MilestoneStrip'
import EmptyState from './EmptyState'

const TurnBubble = memo(function TurnBubble({ turn }) {
  const isAgent = turn.speaker === 'agent'
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex gap-2.5 ${isAgent ? 'flex-row-reverse' : ''}`}
    >
      <div
        className={`mt-0.5 flex h-7 w-7 flex-none items-center justify-center rounded-lg shadow-sm
          ${isAgent ? 'bg-gradient-to-br from-indigo-400 to-purple-400 text-white dark:from-blue-500 dark:to-blue-700 dark:text-blue-100'
                    : 'bg-gradient-to-br from-teal-400 to-emerald-400 text-white dark:from-teal-600 dark:to-teal-800 dark:text-teal-100'}`}
      >
        {isAgent ? <Bot size={14} /> : <User size={14} />}
      </div>
      <div className={`flex max-w-[82%] flex-col ${isAgent ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-3.5 py-2.5 text-[13.5px] leading-relaxed shadow-md
            ${isAgent
              ? 'rounded-tr-sm bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-blue-500/15 dark:to-blue-500/5 border border-indigo-100 dark:border-blue-400/20 text-slate-700 dark:text-slate-100 shadow-indigo-500/10'
              : 'rounded-tl-sm bg-gradient-to-br from-slate-100 to-teal-50/60 dark:from-panel-2 dark:to-panel border border-slate-200 dark:border-white/10 text-slate-700 dark:text-slate-100 shadow-teal-500/5'}
            ${turn.important ? 'ring-1 ring-gold/50 shadow-[0_0_0_1px_rgba(201,162,75,0.12)_inset]' : ''}`}
        >
          {turn.text}
        </div>
        <span className="mt-1 text-[10px] text-slate-400 dark:text-slate-500">
          {new Date(turn.time).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </motion.div>
  )
})

export default function TranscriptPanel({ turns, milestones }) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [turns])

  return (
    <section
      aria-label="Live conversation transcript"
      className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200/70 dark:border-white/10
        bg-white/60 dark:glass-dark shadow-lg dark:shadow-glow"
    >
      <div className="flex-none border-b border-slate-200/70 dark:border-white/10 px-4 py-3">
        <h2 className="font-display text-sm font-semibold text-slate-800 dark:text-white">Live Conversation</h2>
      </div>

      <MilestoneStrip milestones={milestones} />

      <div
        ref={scrollRef}
        role="log"
        aria-live="polite"
        aria-relevant="additions"
        className="flex flex-1 flex-col gap-3 overflow-y-auto p-4 scrollbar-thin"
      >
        {turns.length === 0 ? (
          <EmptyState
            icon={Headphones}
            title="Turn on your mic and customer audio"
            subtitle="The live conversation will appear here once both are on."
          />
        ) : (
          <AnimatePresence initial={false}>
            {turns.map((turn) => (
              <TurnBubble key={turn.id} turn={turn} />
            ))}
          </AnimatePresence>
        )}
      </div>
    </section>
  )
}
