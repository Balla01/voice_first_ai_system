import { Mic, Headphones, Link2 } from 'lucide-react'
import Toggle from './Toggle'

export default function ControlBar({ micOn, speakerOn, onMic, onSpeaker, onBoth, running }) {
  const bothOn = micOn && speakerOn

  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-slate-200/70 dark:border-white/10 bg-white/40 dark:bg-panel/60 px-5 py-2.5">
      <Toggle id="mic-toggle" label="My mic" icon={Mic} checked={micOn} onChange={onMic} />
      <Toggle id="speaker-toggle" label="Customer audio" icon={Headphones} checked={speakerOn} onChange={onSpeaker} />
      <span className="hidden sm:block h-1 w-1 rounded-full bg-slate-300 dark:bg-slate-700" />
      <Toggle id="both-toggle" label="Listen to both" icon={Link2} checked={bothOn} onChange={onBoth} />

      <div className="ml-auto flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400" role="status">
        <span
          className={`relative h-2 w-2 rounded-full ${running ? 'bg-emerald-400' : 'bg-slate-400 dark:bg-slate-600'}`}
        >
          {running && (
            <span className="absolute inset-[-4px] rounded-full border border-emerald-400 animate-pulse-ring" />
          )}
        </span>
        {running ? 'Listening…' : 'Not listening'}
      </div>
    </div>
  )
}
