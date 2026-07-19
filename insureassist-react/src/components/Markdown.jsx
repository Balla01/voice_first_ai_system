import CopyButton from './CopyButton'

// A deliberately small markdown renderer covering what an agent-facing chat
// actually needs (bold, inline code, fenced code blocks, bullet lists) —
// avoids pulling in a full remark/rehype toolchain for this prototype.

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${keyPrefix}-${i}`} className="font-semibold text-slate-800 dark:text-white">{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={`${keyPrefix}-${i}`} className="rounded bg-slate-100 dark:bg-black/40 px-1.5 py-0.5 font-mono text-[12px] text-rose-500 dark:text-teal">
          {part.slice(1, -1)}
        </code>
      )
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>
  })
}

export default function Markdown({ content }) {
  const blocks = content.split(/```([\s\S]*?)```/g)

  return (
    <div className="space-y-2">
      {blocks.map((block, idx) => {
        if (idx % 2 === 1) {
          // fenced code block
          return (
            <div key={idx} className="relative rounded-lg bg-slate-900 dark:bg-black/50 p-3 pr-16">
              <pre className="overflow-x-auto font-mono text-[12px] leading-relaxed text-emerald-300">
                <code>{block.trim()}</code>
              </pre>
              <CopyButton text={block.trim()} className="absolute right-2 top-2 !border-white/10 !text-slate-300" />
            </div>
          )
        }

        const lines = block.split('\n').filter((l) => l.trim() !== '')
        return (
          <div key={idx} className="space-y-1">
            {lines.map((line, i) => {
              const bulletMatch = line.match(/^\s*[-*]\s+(.*)/)
              if (bulletMatch) {
                return (
                  <div key={i} className="flex gap-2 pl-1 text-[13px] leading-relaxed">
                    <span className="text-slate-400 dark:text-slate-500">•</span>
                    <span>{renderInline(bulletMatch[1], `${idx}-${i}`)}</span>
                  </div>
                )
              }
              return (
                <p key={i} className="text-[13px] leading-relaxed">
                  {renderInline(line, `${idx}-${i}`)}
                </p>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
