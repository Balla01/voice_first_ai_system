import { FileText, UserCircle, MessageSquare } from 'lucide-react'
import PolicyTab from './PolicyTab'
import ProfileTab from './ProfileTab'
import ChatTab from './ChatTab'

const TABS = [
  { key: 'policy', label: 'Policy', Icon: FileText },
  { key: 'profile', label: 'Profile', Icon: UserCircle },
  { key: 'chat', label: 'Ask AI', Icon: MessageSquare },
]

export default function Sidebar({
  activeTab, setActiveTab, simState,
  chatMessages, chatInput, setChatInput, onSendChat, onRegenerateChat, chatLoading, chatInputRef,
  onSendEmail,
}) {
  const latestAnswer = simState.answers.find((a) => a.id === simState.activeAnswerId)

  return (
    <aside className="flex min-h-0 w-full flex-col overflow-hidden rounded-2xl border border-slate-200/70 dark:border-white/10
      bg-white/60 dark:glass-dark shadow-lg dark:shadow-glow lg:w-[340px]">
      <div className="flex-none p-2">
        <div role="tablist" aria-label="Sidebar sections" className="grid grid-cols-3 gap-1 rounded-xl bg-slate-100/70 dark:bg-panel-2 p-1">
          {TABS.map(({ key, label, Icon }) => (
            <button
              key={key}
              role="tab"
              aria-selected={activeTab === key}
              onClick={() => setActiveTab(key)}
              className={`flex flex-col items-center gap-1 rounded-lg px-1.5 py-1.5 text-[10.5px] font-medium transition-colors
                focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400
                ${activeTab === key
                  ? 'bg-white dark:bg-panel text-slate-800 dark:text-white shadow-sm'
                  : 'text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300'}`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden" role="tabpanel">
        {activeTab === 'policy' && (
          <div className="h-full overflow-y-auto p-3.5 scrollbar-thin">
            <PolicyTab
              complianceVisible={simState.complianceVisible}
              complianceItems={simState.complianceItems}
            />
          </div>
        )}
        {activeTab === 'profile' && (
          <div className="h-full overflow-y-auto p-3.5 scrollbar-thin">
            <ProfileTab profileItems={simState.profileItems} />
          </div>
        )}
        {activeTab === 'chat' && (
          <ChatTab
            messages={chatMessages}
            input={chatInput}
            setInput={setChatInput}
            onSend={onSendChat}
            onRegenerate={onRegenerateChat}
            loading={chatLoading}
            inputRef={chatInputRef}
            profileItems={simState.profileItems}
            latestAnswer={latestAnswer}
            onSendEmail={onSendEmail}
          />
        )}
      </div>
    </aside>
  )
}
