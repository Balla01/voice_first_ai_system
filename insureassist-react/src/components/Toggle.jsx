export default function Toggle({ checked, onChange, label, icon: Icon, id }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-slate-200/70 dark:border-white/10 bg-white/70 dark:bg-black/20 px-3 py-2">
      {Icon && <Icon size={16} className="text-slate-500 dark:text-slate-300" aria-hidden="true" />}
      <label htmlFor={id} className="text-xs font-medium text-slate-700 dark:text-slate-100 cursor-pointer select-none">
        {label}
      </label>
      <button
        id={id}
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-400
          ${checked ? 'bg-emerald-400' : 'bg-slate-300 dark:bg-slate-700'}`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform
            ${checked ? 'translate-x-[19px]' : 'translate-x-[3px]'}`}
        />
      </button>
    </div>
  )
}
