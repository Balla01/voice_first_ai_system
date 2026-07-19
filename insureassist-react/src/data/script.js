// This scripts a single realistic call so the prototype demonstrates real
// timing and behavior (thinking -> answer -> mid-call update) rather than
// static mockups. Swap this out for the live pipeline's events later.

export const THINKING_STEPS_FULL = [
  'Listening to the customer',
  'Understanding what they need',
  'Checking policy details',
  'Preparing your answer',
]

export const THINKING_STEPS_UPDATE = [
  'Noticed new detail from the customer',
  'Updating your answer',
]

export const CALL_SCRIPT = [
  { t: 400, type: 'turn', speaker: 'customer',
    text: "Hi, I'm looking for a family health plan. My wife has diabetes, and we have two kids." },

  { t: 2400, type: 'turn', speaker: 'agent',
    text: 'Sure, let me understand your requirements a bit better.' },

  { t: 4600, type: 'turn', speaker: 'customer', important: true,
    text: "Our budget is around 15,000 rupees a month. Does this cover diabetes, and what's the waiting period?" },

  { t: 4900, type: 'milestone', text: 'Budget & diabetes detected' },
  { t: 4900, type: 'profile', icon: 'users', text: 'Wife: diabetic · 2 children' },
  { t: 4950, type: 'profile', icon: 'wallet', text: 'Budget: ₹15,000 / month' },

  { t: 5100, type: 'thinking', mode: 'full',
    result: {
      id: 'a1',
      intents: ['premium_concern', 'exclusion_concern'],
      sources: ['Premium Rates', 'Exclusions List', 'Policy Wording'],
      customerQuestion: "Does this cover diabetes, and what's the waiting period? Budget is ₹15,000/month.",
      text: "For a ₹15,000/month budget, Plan A fits comfortably. Diabetes is covered after a 36-month waiting period (24 months with portability). Premium for a family of four works out to roughly ₹11,200/month, with 80D tax benefit applicable.",
      reasons: [
        'Fits within the ₹15,000 budget',
        'Covers pre-existing diabetes after the waiting period',
        'Covers both children under the same floater',
      ],
      nextQuestion: 'Has your wife been diagnosed with diabetes for more than 4 years? It affects the waiting period.',
    } },

  { t: 9400, type: 'turn', speaker: 'customer', important: true,
    text: "Also — can we add my father-in-law? He's 68." },

  { t: 9700, type: 'milestone', text: 'Buying signal — father-in-law added' },
  { t: 9700, type: 'profile', icon: 'user-plus', text: 'Father-in-law: 68, wants cover added' },
  { t: 9800, type: 'banner', text: 'Updating the answer — new detail from the customer' },

  { t: 10000, type: 'thinking', mode: 'update',
    result: {
      id: 'a2',
      intents: ['policy_inquiry', 'premium_concern'],
      sources: ['Policy Documents', 'Premium Rates', 'Discount Rules'],
      customerQuestion: "Can we add my father-in-law? He's 68.",
      text: "For your father-in-law at 68, he'd need our senior citizen rider. Combined with the family floater covering your wife's diabetes (36-month waiting period) and the two kids, the total premium for a family of five including him comes to approximately ₹14,800/month — just within your budget.",
      reasons: [
        'Senior rider available for age 68',
        "Keeps the rest of the family's floater intact",
        'Still lands under the ₹15,000 budget',
      ],
      nextQuestion: 'Would he need portability from an existing policy, or is this a fresh cover?',
    } },

  { t: 12000, type: 'policy' },
  { t: 12200, type: 'profile', icon: 'check', text: 'Fits budget — ₹14,800 / month total' },
  { t: 15000, type: 'compliance' },

  { t: 17800, type: 'turn', speaker: 'agent',
    text: 'Great, that works well within budget — let me note that down for you.' },
  { t: 18100, type: 'milestone', text: 'Plan recommended — Family Floater A + senior rider' },
  { t: 18400, type: 'done' },
]

export const COMPLIANCE_ITEMS = [
  { ok: true, text: 'Waiting period mentioned clearly' },
  { ok: true, text: 'Exclusions mentioned before quoting premium' },
  { ok: true, text: 'No claim approval was guaranteed' },
  { ok: false, text: 'Reminder: mention the portability benefit' },
]

export const POLICY_SNAPSHOT = {
  plan: 'Family Floater — Plan A',
  waitingPeriod: '36 mo (24 mo if ported)',
  seniorCover: 'Available with rider',
  premium: '₹14,800 (family of 5)',
}

export const SUGGESTED_PROMPTS = [
  'Summarize this call in 3 bullet points for my manager',
  'Draft a WhatsApp follow-up message to the customer',
  'What objection should I prepare for next?',
  'Compare Plan A against a cheaper alternative',
]
