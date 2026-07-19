import { motion } from 'framer-motion'
import { ALL_INTENT_OPTIONS } from '../data/intents'

// Cards double as the manual-override control: clicking one calls onSelect,
// which toggles it as the agent's manual pick (backend gets a POST
// /intent/override). Auto-detected intents (from the live classifier) glow
// with a "detected" badge; a manual pick glows with "set" and wins over auto.
// Includes the two override-only intents (Renewal, Cancellation) the
// classifier never fires automatically — they can only ever show "set".
export default function IntentGrid({ activeKeys, manualIntent, onSelect }) {
  return (
    <div>
      <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
        Detected intent
      </h3>
      <div className="grid grid-cols-3 gap-2.5" role="list">
        {ALL_INTENT_OPTIONS.map((intent) => {
          const isActive = activeKeys.includes(intent.key)
          const isManual = manualIntent === intent.key
          const Icon = intent.Icon
          return (
            <motion.button
              key={intent.key}
              type="button"
              role="listitem"
              onClick={() => onSelect?.(intent.key)}
              whileTap={{ scale: 0.96 }}
              animate={{ y: isActive || isManual ? -2 : 0, scale: isActive || isManual ? 1.02 : 1 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className={`relative flex flex-col items-center gap-2 rounded-xl border px-3 py-3.5 text-center transition-all cursor-pointer
                focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400
                ${isActive || isManual
                  ? 'border-transparent shadow-lg'
                  : 'border-slate-200 dark:border-white/10 bg-white/50 dark:bg-panel-2 hover:border-indigo-200 dark:hover:border-white/20 hover:-translate-y-0.5'}`}
              style={
                isActive || isManual
                  ? {
                      backgroundImage: `linear-gradient(135deg, ${intent.color}22, ${intent.color}08)`,
                      boxShadow: `0 0 0 1px ${intent.color}55, 0 10px 24px -14px ${intent.color}`,
                    }
                  : undefined
              }
            >
              <Icon
                size={18}
                style={{ color: isActive || isManual ? intent.color : undefined }}
                className={isActive || isManual ? '' : 'text-slate-400 dark:text-slate-500'}
              />
              <span className={`text-[11.5px] font-medium ${isActive || isManual ? 'text-slate-800 dark:text-white' : 'text-slate-500 dark:text-slate-400'}`}>
                {intent.label}
              </span>
              {(isActive || isManual) && (
                <span
                  className="absolute -top-1.5 -right-1.5 rounded-full px-1.5 py-0.5 text-[9px] font-semibold text-white shadow"
                  style={{ backgroundColor: intent.color }}
                >
                  {isManual ? 'set' : 'detected'}
                </span>
              )}
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}
