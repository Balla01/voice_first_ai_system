import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Sparkles, RefreshCw, Bot, User, Mail, Loader2, CheckCircle2, ChevronDown } from 'lucide-react'
import Markdown from './Markdown'
import CopyButton from './CopyButton'
import EmptyState from './EmptyState'
import { SUGGESTED_PROMPTS } from '../data/script'

function TypingDots() {
  return (
    <div className="flex items-center gap-1 rounded-2xl rounded-tl-sm border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-panel-2 px-3.5 py-3 w-fit">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-slate-400 dark:bg-slate-500"
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
        />
      ))}
    </div>
  )
}

function formatTime(ms) {
  if (!ms) return ''
  return new Date(ms).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })
}

function buildAutoSummary(profileItems, latestAnswer) {
  const facts = (profileItems || []).map((p) => `- ${p.text}`).join('\n')
  const recommendation = latestAnswer ? `\n\nRecommendation discussed:\n${latestAnswer.text}` : ''
  return `Hi,\n\nThanks for the call today. Here's a quick summary of what we discussed:\n\n${facts || '- (details will populate once the call starts)'}${recommendation}\n\nLet me know if you have any questions.\n\nBest,\nYour InsureAssist agent`
}

// Email composer, folded in under Ask AI (not its own sidebar tab). Drafts
// itself from the live profile + latest suggestion, and sends via the
// backend's Gmail integration (onSendEmail, from useLiveCall).
function EmailComposer({ profileItems, latestAnswer, onSendEmail }) {
  const [open, setOpen] = useState(false)
  const [to, setTo] = useState('')
  const [subject, setSubject] = useState('Your plan — summary & next steps')
  const [message, setMessage] = useState('')
  const [status, setStatus] = useState('idle') // idle | sending | sent | error
  const [errorText, setErrorText] = useState('')
  const editedRef = useRef(false)

  useEffect(() => {
    if (editedRef.current) return // don't clobber a manual edit
    setMessage(buildAutoSummary(profileItems, latestAnswer))
  }, [profileItems, latestAnswer])

  const handleSend = async () => {
    if (!to || status === 'sending') return
    setStatus('sending')
    try {
      await onSendEmail?.({ to, subject, body: message })
      setStatus('sent')
      setTimeout(() => setStatus('idle'), 2200)
    } catch (err) {
      setErrorText(err?.message || 'Send failed')
      setStatus('error')
      setTimeout(() => setStatus('idle'), 3000)
    }
  }

  return (
    <div className="flex-none border-t border-slate-200/70 dark:border-white/10">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500 hover:text-indigo-500 dark:hover:text-gold transition-colors"
      >
        <Mail size={12} /> Send follow-up email
        <ChevronDown size={13} className={`ml-auto transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="flex flex-col gap-2.5 px-4 pb-4">
              <input
                type="email"
                value={to}
                onChange={(e) => setTo(e.target.value)}
                placeholder="customer@email.com"
                className="w-full rounded-lg border border-slate-200 dark:border-white/10 bg-white dark:bg-black/20 px-3 py-2 text-[12.5px] text-slate-700 dark:text-slate-100 placeholder:text-slate-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400"
              />
              <input
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full rounded-lg border border-slate-200 dark:border-white/10 bg-white dark:bg-black/20 px-3 py-2 text-[12.5px] text-slate-700 dark:text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400"
              />
              <textarea
                rows={6}
                value={message}
                onChange={(e) => { editedRef.current = true; setMessage(e.target.value) }}
                className="w-full resize-none rounded-lg border border-slate-200 dark:border-white/10 bg-white dark:bg-black/20 px-3 py-2 text-[12px] leading-relaxed text-slate-700 dark:text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400"
              />
              <motion.button
                whileTap={{ scale: 0.97 }}
                onClick={handleSend}
                disabled={!to || status === 'sending'}
                className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 dark:from-gold dark:to-gold px-4 py-2.5 text-[13px] font-semibold text-white dark:text-ink shadow-md shadow-indigo-500/25 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
              >
                <AnimatePresence mode="wait" initial={false}>
                  {status === 'sending' ? (
                    <motion.span key="sending" className="flex items-center gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <Loader2 size={14} className="animate-spin" /> Sending…
                    </motion.span>
                  ) : status === 'sent' ? (
                    <motion.span key="sent" className="flex items-center gap-2" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                      <CheckCircle2 size={14} /> Sent!
                    </motion.span>
                  ) : (
                    <motion.span key="idle" className="flex items-center gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                      <Send size={14} /> Send email
                    </motion.span>
                  )}
                </AnimatePresence>
              </motion.button>
              {status === 'error' && (
                <p className="text-[11px] font-medium text-rose-500 dark:text-rose-400">{errorText}</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ChatTab({
  messages, input, setInput, onSend, onRegenerate, loading, inputRef,
  profileItems, latestAnswer, onSendEmail,
}) {
  const scrollRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, loading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  const lastAiIndex = [...messages].map((m) => m.role).lastIndexOf('ai')

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-4 scrollbar-thin">
        {messages.length === 0 && !loading ? (
          <div className="flex h-full flex-col">
            <EmptyState
              icon={Sparkles}
              title="Ask the AI copilot anything"
              subtitle="Paste something from the call, or try a suggestion below."
            />
            <div className="mt-2 flex flex-col gap-1.5 px-2">
              {SUGGESTED_PROMPTS.map((p) => (
                <button
                  key={p}
                  onClick={() => setInput(p)}
                  className="rounded-lg border border-slate-200 dark:border-white/10 bg-white/60 dark:bg-panel-2 px-3 py-2 text-left text-[12px] text-slate-600 dark:text-slate-300 hover:border-indigo-300 dark:hover:border-gold/40 transition-colors"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((m, idx) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div
                  className={`mt-0.5 flex h-6 w-6 flex-none items-center justify-center rounded-lg shadow-sm
                  ${m.role === 'user' ? 'bg-gradient-to-br from-indigo-400 to-purple-400 text-white dark:from-blue-500 dark:to-blue-700 dark:text-blue-100'
                                      : 'bg-gradient-to-br from-indigo-400/20 to-purple-400/20 dark:from-gold/15 dark:to-gold/5 text-indigo-500 dark:text-gold'}`}
                >
                  {m.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                </div>
                <div className="flex max-w-[86%] flex-col">
                  <div className={`rounded-2xl px-3.5 py-2.5 text-slate-700 dark:text-slate-200 shadow-md
                    ${m.role === 'user'
                      ? 'rounded-tr-sm bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-blue-500/15 dark:to-blue-500/5 border border-indigo-100 dark:border-blue-400/20 shadow-indigo-500/10'
                      : 'rounded-tl-sm bg-gradient-to-br from-white to-slate-50 dark:from-panel-2 dark:to-panel border border-slate-200 dark:border-white/10 shadow-slate-500/5'}`}
                  >
                    {m.role === 'ai' ? <Markdown content={m.text} /> : <p className="text-[13px] leading-relaxed">{m.text}</p>}
                    {m.role === 'ai' && (
                      <div className="mt-2 flex items-center gap-2">
                        <CopyButton text={m.text} />
                        {idx === lastAiIndex && (
                          <button
                            onClick={onRegenerate}
                            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 dark:border-white/10 px-2.5 py-1 text-[11px] text-slate-500 dark:text-slate-300 hover:text-slate-800 dark:hover:text-white"
                          >
                            <RefreshCw size={11} /> Regenerate
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                  <span className={`mt-1 text-[10px] text-slate-400 dark:text-slate-500 ${m.role === 'user' ? 'text-right' : ''}`}>
                    {formatTime(m.time)}
                  </span>
                </div>
              </motion.div>
            ))}
            {loading && (
              <div className="flex gap-2">
                <div className="mt-0.5 flex h-6 w-6 flex-none items-center justify-center rounded-lg bg-gradient-to-br from-indigo-400/20 to-purple-400/20 dark:from-gold/15 dark:to-gold/5 text-indigo-500 dark:text-gold">
                  <Bot size={12} />
                </div>
                <TypingDots />
              </div>
            )}
          </AnimatePresence>
        )}
      </div>

      <div className="flex-none border-t border-slate-200/70 dark:border-white/10 p-3">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question…"
            aria-label="Ask AI input"
            className="flex-1 resize-none rounded-xl border border-slate-200 dark:border-white/10 bg-white dark:bg-black/20
              px-3 py-2 text-[12.5px] text-slate-700 dark:text-slate-100 placeholder:text-slate-400
              focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400"
          />
          <motion.button
            whileTap={{ scale: 0.94 }}
            onClick={onSend}
            aria-label="Send message"
            className="flex h-9 w-9 flex-none items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 dark:from-gold dark:to-gold text-white dark:text-ink shadow-md shadow-indigo-500/25"
          >
            <Send size={15} />
          </motion.button>
        </div>
        <p className="mt-1.5 text-[10px] text-slate-400 dark:text-slate-500">
          Tip: use the copy button on any card, then paste it here.
        </p>
      </div>

      <EmailComposer profileItems={profileItems} latestAnswer={latestAnswer} onSendEmail={onSendEmail} />
    </div>
  )
}
