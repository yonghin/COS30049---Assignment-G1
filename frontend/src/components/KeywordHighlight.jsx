import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import Chip from '@mui/material/Chip'

// Curated spam-signal vocabulary. This mirrors the kind of signals the
// Random-Forest model keys on (has_* keyword flags, urls, digits) but is an
// explanatory heuristic, NOT the model's exact internal weights.
const SPAM_WORDS = new Set([
  'free', 'win', 'winner', 'won', 'cash', 'prize', 'prizes', 'award', 'awarded',
  'urgent', 'congratulations', 'congrats', 'claim', 'offer', 'offers', 'guaranteed',
  'click', 'subscribe', 'unsubscribe', 'credit', 'loan', 'casino', 'bonus', 'deal',
  'limited', 'voucher', 'reward', 'selected', 'ringtone', 'mobile', 'txt', 'text',
  'reply', 'stop', 'call', 'now', 'today', 'risk', 'discount', 'cheap', 'buy',
  'order', 'guarantee', 'trial', 'investment', 'bitcoin', 'crypto', 'viagra', 'pills',
])

// Classify a single token. Returns a reason string when it looks spammy, else null.
function spamReason(token) {
  const bare = token.toLowerCase().replace(/[^a-z0-9$£€]/gi, '')
  if (!bare) return null
  if (/^(https?:\/\/|www\.)/i.test(token) || /\.(com|net|org|info|biz|ru)\b/i.test(token)) return 'link'
  if (/[$£€]/.test(token)) return 'money'
  if (/^\d{4,}$/.test(bare)) return 'number / code'
  const letters = token.replace(/[^a-zA-Z]/g, '')
  if (letters.length >= 3 && letters === letters.toUpperCase()) return 'ALL CAPS'
  if (SPAM_WORDS.has(bare)) return 'spam keyword'
  return null
}

// Re-renders a message with spam-signal tokens highlighted, so the user can see
// why it may have scored high. Splits on whitespace, preserving spacing.
function KeywordHighlight({ text }) {
  if (!text || !text.trim()) return null

  const parts = text.split(/(\s+)/) // keep separators
  let hits = 0
  const nodes = parts.map((part, i) => {
    if (/^\s+$/.test(part) || part === '') return part
    const reason = spamReason(part)
    if (!reason) return part
    hits += 1
    return (
      <Box
        key={i}
        component="mark"
        title={`Signal: ${reason}`}
        sx={{
          bgcolor: 'rgba(255,77,77,0.22)',
          color: 'error.main',
          borderRadius: 1,
          px: 0.5,
        }}
      >
        {part}
      </Box>
    )
  })

  return (
    <Box sx={{ mt: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Typography sx={{ fontSize: 13, fontWeight: 600, color: 'text.primary' }}>
          Why it scored this way
        </Typography>
        <Chip label="heuristic" size="small" variant="outlined" color="warning" sx={{ height: 20, fontSize: 10 }} />
      </Box>
      <Typography
        sx={{
          fontSize: 14,
          lineHeight: 1.7,
          color: 'text.primary',
          bgcolor: 'action.hover',
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
          p: 1.5,
          m: 0,
          wordBreak: 'break-word',
        }}
      >
        {nodes}
      </Typography>
      <Typography sx={{ fontSize: 12, color: 'text.secondary', mt: 1, lineHeight: 1.5 }}>
        {hits > 0
          ? `${hits} spam-signal word${hits === 1 ? '' : 's'} highlighted.`
          : 'No common spam signals found in this message.'}
        {' '}This is an explanatory heuristic, not the model's exact weights.
      </Typography>
    </Box>
  )
}

export default KeywordHighlight