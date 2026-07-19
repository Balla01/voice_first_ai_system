import { useCallback, useRef, useState } from 'react'
import { startCapture } from '../lib/captureClient'

// Live counterpart to useCallSimulation: instead of replaying CALL_SCRIPT, it
// drives the exact same state shape from the backend's WebSocket frames
//   {"type":"transcript","speaker":"agent"|"customer","text":string}
//   {"type":"suggestion","query":string,"answer":string,"tool"?:string}
//   {"type":"intent","intents":string[],"source":string}
//   {"type":"profile","profile":{name,age,profession,location,family}}
// pushed over the mic socket (auido_capture/main.py :: _send_to_session), plus
// REST calls to the same backend for intent override / ask AI / email send /
// session reset.
//
// The UI components (TranscriptPanel, CopilotPanel, AnswerCard, Sidebar) are
// unchanged, so every field they read must exist here with a safe default —
// the backend doesn't send reasons / sources / nextQuestion, so we fill those
// with empty values rather than let AnswerCard.map() crash.

const API_BASE = 'http://localhost:8000'
const STREAM_TICK = 18
const STREAM_CHUNK = 3

let uid = 0
const nextId = () => `id-${Date.now()}-${uid++}`

// Maps the flat {name,age,profession,location,family} profile the backend
// sends into ProfileTab's existing {id,icon,text} row shape. Fixed ids per
// field (not nextId()) so a later update REPLACES the row in place instead of
// appending a duplicate — this is what makes the profile card "build live".
function profileToItems(profile) {
  const items = []
  if (profile.name) items.push({ id: 'profile-name', icon: 'user-plus', text: `Name: ${profile.name}` })
  if (profile.age) items.push({ id: 'profile-age', icon: 'check', text: `Age: ${profile.age}` })
  if (profile.profession) items.push({ id: 'profile-profession', icon: 'wallet', text: `Profession: ${profile.profession}` })
  if (profile.location) items.push({ id: 'profile-location', icon: 'check', text: `Location: ${profile.location}` })
  if (profile.family && profile.family.length)
    items.push({ id: 'profile-family', icon: 'users', text: `Family: ${profile.family.join(', ')}` })
  return items
}

const initialState = {
  running: false,
  progress: 'Ready',
  turns: [],
  milestones: [],
  profileItems: [],
  answers: [],
  activeAnswerId: null,
  thinking: null,
  banner: null,
  policyVisible: false,
  complianceVisible: false,
  complianceItems: [],
  autoIntents: [],
  manualIntent: null,
  sessionId: null,
}

// Shape a backend "suggestion" frame into the answer object AnswerCard expects.
function answerFromSuggestion(msg) {
  return {
    id: nextId(),
    text: typeof msg.answer === 'string' ? msg.answer : String(msg.answer ?? ''),
    customerQuestion: msg.query || '',
    intents: [],
    sources: msg.tool ? [msg.tool] : [],
    reasons: [],
    nextQuestion: null,
    revealed: 0,
    status: 'streaming',
  }
}

export function useLiveCall() {
  const [state, setState] = useState(initialState)
  const captureRef = useRef(null) // { sessionId, micPipeline, systemPipeline }
  const streamIntervals = useRef({})

  const clearStreams = useCallback(() => {
    Object.values(streamIntervals.current).forEach(clearInterval)
    streamIntervals.current = {}
  }, [])

  // Cosmetic char-by-char reveal, matching the simulation's feel.
  const streamAnswer = useCallback((id, fullText) => {
    let revealed = 0
    const interval = setInterval(() => {
      revealed += STREAM_CHUNK
      setState((prev) => ({
        ...prev,
        answers: prev.answers.map((a) =>
          a.id === id ? { ...a, revealed: Math.min(revealed, fullText.length) } : a
        ),
      }))
      if (revealed >= fullText.length) {
        clearInterval(interval)
        delete streamIntervals.current[id]
        setState((prev) => ({
          ...prev,
          answers: prev.answers.map((a) => (a.id === id ? { ...a, status: 'done' } : a)),
        }))
      }
    }, STREAM_TICK)
    streamIntervals.current[id] = interval
  }, [])

  const handleTranscript = useCallback((msg) => {
    if (!msg.text) return
    setState((prev) => ({
      ...prev,
      turns: [
        ...prev.turns,
        {
          id: nextId(),
          speaker: msg.speaker === 'agent' ? 'agent' : 'customer',
          text: msg.text,
          important: false,
          time: Date.now(),
        },
      ],
    }))
  }, [])

  const handleSuggestion = useCallback((msg) => {
    const answer = answerFromSuggestion(msg)
    setState((prev) => ({
      ...prev,
      answers: [
        answer,
        // Fade the previously-active card, matching the simulation's supersede.
        ...prev.answers.map((a) =>
          a.id === prev.activeAnswerId ? { ...a, status: 'superseded' } : a
        ),
      ],
      activeAnswerId: answer.id,
      manualIntent: null,
    }))
    streamAnswer(answer.id, answer.text)
  }, [streamAnswer])

  const handleIntent = useCallback((msg) => {
    setState((prev) => ({ ...prev, autoIntents: Array.isArray(msg.intents) ? msg.intents : [] }))
  }, [])

  const handleProfile = useCallback((msg) => {
    if (!msg.profile) return
    setState((prev) => ({ ...prev, profileItems: profileToItems(msg.profile) }))
  }, [])

  const onServerMessage = useCallback((msg) => {
    if (!msg || typeof msg !== 'object') return
    if (msg.type === 'transcript') handleTranscript(msg)
    else if (msg.type === 'suggestion') handleSuggestion(msg)
    else if (msg.type === 'intent') handleIntent(msg)
    else if (msg.type === 'profile') handleProfile(msg)
    // Unknown types are ignored so a future backend frame can't break the UI.
  }, [handleTranscript, handleSuggestion, handleIntent, handleProfile])

  const stop = useCallback(() => {
    clearStreams()
    const sessionId = captureRef.current?.sessionId
    if (captureRef.current) {
      captureRef.current.micPipeline?.stop()
      captureRef.current.systemPipeline?.stop()
      captureRef.current = null
    }
    if (sessionId) {
      fetch(`${API_BASE}/session/${sessionId}/reset`, { method: 'POST' }).catch(() => {})
    }
    // Clear everything immediately on End Call rather than waiting for the
    // next Start — transcript, answers, profile, intents all wipe right away.
    setState(initialState)
  }, [clearStreams])

  // MUST be awaited from a user-gesture handler (getDisplayMedia needs it).
  const start = useCallback(async () => {
    if (captureRef.current) return // already capturing
    setState({ ...initialState, running: true, progress: 'Connecting…' })
    try {
      const capture = await startCapture(onServerMessage)
      captureRef.current = capture
      setState((prev) => ({ ...prev, progress: 'Listening…', sessionId: capture.sessionId }))
    } catch (err) {
      console.error('startCapture failed', err)
      clearStreams()
      captureRef.current = null
      setState({
        ...initialState,
        running: false,
        progress: 'Could not start capture',
        banner: { id: nextId(), text: err?.message || 'Failed to start audio capture.' },
      })
      throw err // let App reset the toggles
    }
  }, [onServerMessage, clearStreams])

  const reset = useCallback(() => {
    stop()
    setState(initialState)
  }, [stop])

  // Agent clicked an intent card to correct the auto-detected intent — POSTs
  // the override to the backend and reflects it locally right away (optimistic;
  // the backend call is best-effort/fire-and-forget from the UI's perspective).
  const overrideIntent = useCallback((intentKey) => {
    const nextIntent = state.manualIntent === intentKey ? null : intentKey
    setState((prev) => ({ ...prev, manualIntent: nextIntent }))
    fetch(`${API_BASE}/intent/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId || '', intent: nextIntent }),
    }).catch(() => {})
  }, [state.manualIntent, state.sessionId])

  // Ask AI box: forwards a free-text question to Layer 5 (RAG) with call
  // context via the same session_id. Returns the answer text (or throws).
  const askAI = useCallback(async (question) => {
    const sessionId = captureRef.current?.sessionId || 'ask'
    const r = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, question }),
    })
    const j = await r.json()
    if (!r.ok) throw new Error(j.detail || 'Ask failed')
    return j.answer
  }, [])

  // Follow-up email, sent via the backend's Gmail integration.
  const sendEmail = useCallback(async ({ to, subject, body }) => {
    const r = await fetch(`${API_BASE}/email/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to, subject, body }),
    })
    const j = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(j.detail || 'Send failed')
    return true
  }, [])

  // `replay` is meaningless for a live call; expose it as a no-op alias so
  // CopilotPanel's "Replay" button (disabled unless running) stays harmless.
  return { state, start, stop, replay: reset, reset, overrideIntent, askAI, sendEmail }
}
