import { FileText, DollarSign, Ban, Scale, FolderOpen, CheckCircle2, RotateCcw, XCircle } from 'lucide-react'

// The six intents the pipeline actually detects automatically.
export const INTENTS = [
  { key: 'policy_inquiry', label: 'Policy Inquiry', Icon: FileText, color: '#57c6bd' },
  { key: 'premium_concern', label: 'Premium Concern', Icon: DollarSign, color: '#c9a24b' },
  { key: 'exclusion_concern', label: 'Exclusion Concern', Icon: Ban, color: '#e08a63' },
  { key: 'objection', label: 'Objection', Icon: Scale, color: '#b586e0' },
  { key: 'claim_question', label: 'Claim Question', Icon: FolderOpen, color: '#7fa3d6' },
  { key: 'buying_signal', label: 'Buying Signal', Icon: CheckCircle2, color: '#5fd88a' },
]

// A couple of extra options only available in the manual-override dropdown,
// for cases the automatic classifier isn't scoped to handle yet.
export const OVERRIDE_ONLY_INTENTS = [
  { key: 'renewal', label: 'Renewal', Icon: RotateCcw, color: '#8fb3d9' },
  { key: 'cancellation', label: 'Cancellation', Icon: XCircle, color: '#e08a8a' },
]

export const ALL_INTENT_OPTIONS = [...INTENTS, ...OVERRIDE_ONLY_INTENTS]

export function getIntentMeta(key) {
  return ALL_INTENT_OPTIONS.find((i) => i.key === key)
}
