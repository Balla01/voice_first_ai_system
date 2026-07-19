import { useCallback, useEffect, useRef, useState } from 'react'
import TopBar from './components/TopBar'
import ControlBar from './components/ControlBar'
import TranscriptPanel from './components/TranscriptPanel'
import CopilotPanel from './components/CopilotPanel'
import Sidebar from './components/Sidebar'
import { useLiveCall } from './hooks/useLiveCall'
import { useTheme } from './context/ThemeContext'

let chatUid = 0
const nextChatId = () => `chat-${Date.now()}-${chatUid++}`

function buildAiReply(userText) {
  const lower = userText.toLowerCase()
  if (lower.includes('summar')) {
    return "Here's a quick summary:\n\n- Family of 5, budget **₹15,000/month**\n- Wife has diabetes — 36-month waiting period applies\n- Father-in-law (68) added with a senior citizen rider\n- Recommended: **Family Floater — Plan A**, ~₹14,800/month\n\nWant this formatted for your manager or the customer?"
  }
  if (lower.includes('whatsapp') || lower.includes('follow-up') || lower.includes('follow up')) {
    return "Here's a short WhatsApp-style follow-up:\n\n```\nHi! Thanks for the call today. Sharing the plan we discussed — Family Floater A, ~₹14,800/month, covers your wife's diabetes and adds your father-in-law with a senior rider. Let me know if you'd like the proposal PDF.\n```"
  }
  if (lower.includes('objection')) {
    return "A likely next objection is **premium sensitivity** once the father-in-law's rider is added. Consider leading with the tax benefit under 80D and offering a comparison against paying his medical costs out-of-pocket."
  }
  if (lower.includes('compare') || lower.includes('cheaper')) {
    return "A lower-cost alternative would drop the senior rider to a separate standalone senior citizen policy — cheaper per month, but loses the single-claim convenience of one combined floater. Want the numbers side by side?"
  }
  return "Based on the current policy details and what's been shared on this call, here's my take:\n\n" + userText + '\n\nI can shorten this, compare it with another plan, or turn it into a customer-facing message — just say which.'
}

export default function App() {
  useTheme() // ensures theme class applied at root before first paint of children relying on dark: variants

  const [micOn, setMicOn] = useState(false)
  const [speakerOn, setSpeakerOn] = useState(false)
  const [activeTab, setActiveTab] = useState('policy')

  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [pendingFocus, setPendingFocus] = useState(false)
  const chatInputRef = useRef(null)

  const { state, start, stop, replay, overrideIntent, askAI, sendEmail } = useLiveCall()

  // Start/stop live capture from the toggle handlers (NOT a useEffect):
  // getDisplayMedia() requires transient user activation, so the async
  // startCapture() must run in the same task as the user's click.
  const syncCapture = useCallback((nextMic, nextSpeaker) => {
    if (nextMic && nextSpeaker) {
      start().catch(() => {
        // Permission denied or no tab-audio shared — roll the toggles back.
        setMicOn(false)
        setSpeakerOn(false)
      })
    } else {
      stop()
    }
  }, [start, stop])

  const handleMic = useCallback((on) => {
    setMicOn(on)
    syncCapture(on, speakerOn)
  }, [syncCapture, speakerOn])

  const handleSpeaker = useCallback((on) => {
    setSpeakerOn(on)
    syncCapture(micOn, on)
  }, [syncCapture, micOn])

  useEffect(() => {
    if (activeTab === 'chat' && pendingFocus) {
      chatInputRef.current?.focus()
      setPendingFocus(false)
    }
  }, [activeTab, pendingFocus])

  const handleBoth = useCallback((on) => {
    setMicOn(on)
    setSpeakerOn(on)
    syncCapture(on, on)
  }, [syncCapture])

  const handleEndCall = useCallback(() => {
    setMicOn(false)
    setSpeakerOn(false)
    stop()
  }, [stop])

  const handleAskAI = useCallback((answer) => {
    const prefill = `Original question: ${answer.customerQuestion}\n\nAI suggestion: ${answer.text}\n\nFollow-up: `
    setChatInput(prefill)
    setActiveTab('chat')
    setPendingFocus(true)
  }, [])

  // Tries the real copilot (Layer 5 RAG, via useLiveCall's askAI) first, so
  // Ask AI answers with real call context when a session is live; falls back
  // to the canned reply so the tab still works before a call has started or
  // if the RAG service is unreachable.
  const runAiReply = useCallback((userText) => {
    setChatLoading(true)
    askAI(userText)
      .catch(() => buildAiReply(userText))
      .then((text) => {
        setChatMessages((prev) => [...prev, { id: nextChatId(), role: 'ai', text, time: Date.now() }])
        setChatLoading(false)
      })
  }, [askAI])

  const handleSendChat = useCallback(() => {
    const text = chatInput.trim()
    if (!text) return
    setChatMessages((prev) => [...prev, { id: nextChatId(), role: 'user', text, time: Date.now() }])
    setChatInput('')
    runAiReply(text)
  }, [chatInput, runAiReply])

  const handleRegenerateChat = useCallback(() => {
    const lastUser = [...chatMessages].reverse().find((m) => m.role === 'user')
    setChatMessages((prev) => {
      const lastAiIdx = [...prev].map((m) => m.role).lastIndexOf('ai')
      if (lastAiIdx === -1) return prev
      return prev.slice(0, lastAiIdx)
    })
    if (lastUser) runAiReply(lastUser.text + ' (regenerate)')
  }, [chatMessages, runAiReply])

  const handleManualIntent = useCallback((intentKey) => {
    overrideIntent(intentKey)
  }, [overrideIntent])

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-gradient-to-br from-white via-sky-50 to-indigo-50 dark:bg-ink dark:bg-none">
      <div className="pointer-events-none absolute inset-0 overflow-hidden dark:hidden" aria-hidden="true">
        <div className="absolute -left-24 -top-24 h-72 w-72 rounded-full bg-purple-200/50 blur-3xl" />
        <div className="absolute right-0 top-1/3 h-80 w-80 rounded-full bg-sky-200/50 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-pink-200/40 blur-3xl" />
        <div className="absolute right-1/4 bottom-10 h-64 w-64 rounded-full bg-cyan-200/40 blur-3xl" />
      </div>
      <div className="pointer-events-none absolute inset-0 hidden dark:block" aria-hidden="true">
        <div className="absolute -left-40 -top-40 h-[500px] w-[500px] rounded-full bg-indigo-500/5 blur-3xl" />
        <div className="absolute -right-20 bottom-0 h-[400px] w-[400px] rounded-full bg-teal-500/5 blur-3xl" />
      </div>

      <div className="relative z-10 flex h-full flex-col">
        <TopBar running={micOn && speakerOn} onEndCall={handleEndCall} />
        <ControlBar
          micOn={micOn}
          speakerOn={speakerOn}
          onMic={handleMic}
          onSpeaker={handleSpeaker}
          onBoth={handleBoth}
          running={micOn && speakerOn}
        />

        <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden p-4 lg:flex-row">
          <div className="flex min-h-0 w-full flex-col lg:w-[320px]">
            <TranscriptPanel turns={state.turns} milestones={state.milestones} />
          </div>

          <div className="flex min-h-0 w-full flex-1 flex-col">
            <CopilotPanel
              state={state}
              onAskAI={handleAskAI}
              onManualIntent={handleManualIntent}
              onReplay={replay}
              running={micOn && speakerOn}
            />
          </div>

          <Sidebar
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            simState={state}
            chatMessages={chatMessages}
            chatInput={chatInput}
            setChatInput={setChatInput}
            onSendChat={handleSendChat}
            onRegenerateChat={handleRegenerateChat}
            chatLoading={chatLoading}
            chatInputRef={chatInputRef}
            onSendEmail={sendEmail}
          />
        </main>
      </div>
    </div>
  )
}
