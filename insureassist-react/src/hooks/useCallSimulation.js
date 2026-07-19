import { useCallback, useRef, useState } from 'react'
import {
  CALL_SCRIPT,
  THINKING_STEPS_FULL,
  THINKING_STEPS_UPDATE,
  COMPLIANCE_ITEMS,
} from '../data/script'

const STEP_DURATION = 420
const STEP_SETTLE = 350
const STREAM_TICK = 18
const STREAM_CHUNK = 3

let uid = 0
const nextId = () => `id-${Date.now()}-${uid++}`

const initialState = {
  running: false,
  progress: 'Ready',
  turns: [],
  milestones: [],
  profileItems: [],
  answers: [],
  activeAnswerId: null,
  thinking: null, // { mode: 'full'|'update', steps: [{label, status}] }
  banner: null,
  policyVisible: false,
  complianceVisible: false,
  complianceItems: [],
  manualIntent: null,
}

export function useCallSimulation() {
  const [state, setState] = useState(initialState)
  const timers = useRef([])
  const streamIntervals = useRef({})

  const schedule = useCallback((fn, delay) => {
    const t = setTimeout(fn, delay)
    timers.current.push(t)
    return t
  }, [])

  const clearAll = useCallback(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    Object.values(streamIntervals.current).forEach(clearInterval)
    streamIntervals.current = {}
  }, [])

  const streamAnswer = useCallback((answer) => {
    const id = answer.id
    let revealed = 0
    const full = answer.text
    const interval = setInterval(() => {
      revealed += STREAM_CHUNK
      setState((prev) => ({
        ...prev,
        answers: prev.answers.map((a) =>
          a.id === id ? { ...a, revealed: Math.min(revealed, full.length) } : a
        ),
      }))
      if (revealed >= full.length) {
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

  const pushAnswer = useCallback((result) => {
    const answer = { ...result, id: result.id || nextId(), revealed: 0, status: 'streaming' }
    setState((prev) => ({
      ...prev,
      answers: [
        answer,
        ...prev.answers.map((a) => (a.id === prev.activeAnswerId ? { ...a, status: 'superseded' } : a)),
      ],
      activeAnswerId: answer.id,
      manualIntent: null,
    }))
    streamAnswer(answer)
  }, [streamAnswer])

  const runThinking = useCallback((mode, onDone) => {
    const steps = (mode === 'update' ? THINKING_STEPS_UPDATE : THINKING_STEPS_FULL).map((label) => ({
      label,
      status: 'pending',
    }))
    setState((prev) => ({ ...prev, thinking: { mode, steps } }))

    let i = 0
    const advance = () => {
      setState((prev) => {
        if (!prev.thinking) return prev
        const steps = prev.thinking.steps.map((s, idx) => {
          if (idx < i) return { ...s, status: 'done' }
          if (idx === i) return { ...s, status: 'active' }
          return s
        })
        return { ...prev, thinking: { ...prev.thinking, steps } }
      })
      i += 1
      if (i <= steps.length) {
        schedule(advance, STEP_DURATION)
      } else {
        schedule(() => {
          setState((prev) => ({ ...prev, thinking: null }))
          onDone()
        }, STEP_SETTLE)
      }
    }
    advance()
  }, [schedule])

  const reset = useCallback(() => {
    clearAll()
    setState(initialState)
  }, [clearAll])

  const start = useCallback(() => {
    clearAll()
    setState({ ...initialState, running: true, progress: 'In progress' })

    CALL_SCRIPT.forEach((event) => {
      schedule(() => {
        switch (event.type) {
          case 'turn':
            setState((prev) => ({
              ...prev,
              turns: [
                ...prev.turns,
                { id: nextId(), speaker: event.speaker, text: event.text, important: !!event.important, time: Date.now() },
              ],
            }))
            break
          case 'milestone':
            setState((prev) => ({
              ...prev,
              milestones: [...prev.milestones, { id: nextId(), text: event.text, time: Date.now() }],
            }))
            break
          case 'profile':
            setState((prev) => ({
              ...prev,
              profileItems: [...prev.profileItems, { id: nextId(), icon: event.icon, text: event.text }],
            }))
            break
          case 'banner':
            setState((prev) => ({ ...prev, banner: { id: nextId(), text: event.text } }))
            schedule(() => setState((prev) => ({ ...prev, banner: null })), 3200)
            break
          case 'thinking':
            runThinking(event.mode, () => pushAnswer(event.result))
            break
          case 'policy':
            setState((prev) => ({ ...prev, policyVisible: true }))
            break
          case 'compliance':
            setState((prev) => ({ ...prev, complianceVisible: true, complianceItems: [] }))
            COMPLIANCE_ITEMS.forEach((item, idx) => {
              schedule(() => {
                setState((prev) => ({ ...prev, complianceItems: [...prev.complianceItems, item] }))
              }, idx * 180)
            })
            break
          case 'done':
            setState((prev) => ({ ...prev, progress: 'Call complete' }))
            break
          default:
            break
        }
      }, event.t)
    })
  }, [clearAll, schedule, runThinking, pushAnswer])

  const stop = useCallback(() => {
    clearAll()
    setState((prev) => ({ ...prev, running: false, progress: 'Ready' }))
  }, [clearAll])

  const setManualIntent = useCallback((intentKey) => {
    setState((prev) => ({ ...prev, manualIntent: intentKey }))
  }, [])

  return { state, start, stop, replay: start, reset, setManualIntent }
}
