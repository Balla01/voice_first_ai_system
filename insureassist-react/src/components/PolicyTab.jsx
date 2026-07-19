import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ShieldCheck, AlertTriangle, Check, FileStack } from 'lucide-react'

// The plans an agent can flag as relevant to this customer. Multi-select —
// unlike intent (one active classification), several plans can apply at once
// (e.g. a family floater plus a senior citizen rider).
const POLICIES = [
  { key: 'family_floater', label: 'Family Floater', desc: 'One cover for the whole family' },
  { key: 'senior_citizen', label: 'Senior Citizen', desc: 'For parents / in-laws 60+' },
  { key: 'critical_illness', label: 'Critical Illness', desc: 'Lump sum on major diagnoses' },
  { key: 'term_life', label: 'Term Life', desc: 'Pure life cover, high sum assured' },
  { key: 'health_topup', label: 'Health Top-up', desc: 'Extends an existing base cover' },
  { key: 'personal_accident', label: 'Personal Accident', desc: 'Accidental death & disability' },
]

export default function PolicyTab({ complianceVisible, complianceItems }) {
  const [selected, setSelected] = useState(() => new Set())

  const toggle = (key) => {
    setSelected((prev) => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-slate-200 dark:border-white/10 bg-white/70 dark:bg-panel-2 p-4 shadow-sm">
        <p className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
          <FileStack size={12} /> Plans to discuss / recommend
        </p>
        <p className="mb-3 text-[11px] text-slate-400 dark:text-slate-500">Select one or more plans relevant to this customer.</p>

        <div className="flex flex-col gap-2" role="list">
          {POLICIES.map((p) => {
            const on = selected.has(p.key)
            return (
              <motion.button
                key={p.key}
                type="button"
                role="listitem"
                whileTap={{ scale: 0.98 }}
                onClick={() => toggle(p.key)}
                className={`flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-all
                  focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400
                  ${on
                    ? 'border-transparent bg-gradient-to-br from-indigo-50 to-teal-50 dark:from-blue-500/10 dark:to-teal-500/10 shadow-md shadow-indigo-500/10'
                    : 'border-slate-200 dark:border-white/10 bg-white/50 dark:bg-black/20 hover:border-indigo-200 dark:hover:border-white/20'}`}
              >
                <span
                  className={`mt-0.5 flex h-[18px] w-[18px] flex-none items-center justify-center rounded-md border text-[11px] font-bold transition-colors
                    ${on ? 'border-indigo-500 bg-indigo-500 text-white dark:border-gold dark:bg-gold dark:text-ink' : 'border-slate-300 dark:border-slate-600 text-transparent'}`}
                >
                  ✓
                </span>
                <span className="flex flex-col">
                  <span className="text-[12.5px] font-semibold text-slate-800 dark:text-white">{p.label}</span>
                  <span className="text-[10.5px] text-slate-500 dark:text-slate-400">{p.desc}</span>
                </span>
              </motion.button>
            )
          })}
        </div>

        {selected.size > 0 && (
          <p className="mt-3 text-[11px] font-medium text-indigo-500 dark:text-gold">
            Selected: {POLICIES.filter((p) => selected.has(p.key)).map((p) => p.label).join(', ')}
          </p>
        )}
      </div>

      <AnimatePresence>
        {complianceVisible && complianceItems?.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-amber-200/60 dark:border-amber-400/20 bg-amber-50/50 dark:bg-amber-500/5 p-4"
          >
            <p className="mb-2.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-amber-600 dark:text-amber-400">
              <ShieldCheck size={12} /> Compliance checklist
            </p>
            <ul className="space-y-1.5">
              <AnimatePresence>
                {complianceItems.map((item, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="flex items-start gap-2 text-[12.5px] text-slate-600 dark:text-slate-300"
                  >
                    <span className={`mt-0.5 flex h-4 w-4 flex-none items-center justify-center rounded-full
                      ${item.ok ? 'bg-emerald-100 dark:bg-emerald-500/10 text-emerald-500' : 'bg-amber-100 dark:bg-amber-500/10 text-amber-500'}`}>
                      {item.ok ? <Check size={10} /> : <AlertTriangle size={10} />}
                    </span>
                    {item.text}
                  </motion.li>
                ))}
              </AnimatePresence>
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
