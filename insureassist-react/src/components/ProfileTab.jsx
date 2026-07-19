import { AnimatePresence, motion } from 'framer-motion'
import { Users, Wallet, UserPlus, CheckCircle2, UserCircle } from 'lucide-react'
import EmptyState from './EmptyState'

const ICONS = {
  users: Users,
  wallet: Wallet,
  'user-plus': UserPlus,
  check: CheckCircle2,
}

export default function ProfileTab({ profileItems }) {
  return (
    <div className="rounded-2xl border border-slate-200 dark:border-white/10 bg-white/70 dark:bg-panel-2 p-4 shadow-sm">
      <p className="mb-3 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
        <UserCircle size={12} /> Customer profile
        <span className="font-normal normal-case text-slate-400 dark:text-slate-500">— builds as the call goes on</span>
      </p>

      {profileItems.length === 0 ? (
        <EmptyState icon={UserCircle} title="Nothing captured yet" subtitle="Facts the customer shares will show up here automatically." />
      ) : (
        <div className="flex flex-col gap-2">
          <AnimatePresence initial={false}>
            {profileItems.map((item) => {
              const Icon = ICONS[item.icon] || Users
              return (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-2.5 rounded-xl border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-black/20 px-3 py-2.5"
                >
                  <Icon size={15} className="flex-none text-indigo-400 dark:text-gold" />
                  <span className="text-[12.5px] text-slate-700 dark:text-slate-200">{item.text}</span>
                </motion.div>
              )
            })}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
