import './SentimentGauge.css';

const SENTIMENT_CONFIG = {
  Positive: {
    color: '#267F4B',
    gradientStart: '#267F4B',
    gradientEnd: '#34D399',
    bg: '#EAF5EE',
    label: 'Positive',
  },
  Neutral: {
    color: '#267F4B',
    gradientStart: '#1E6B3E',
    gradientEnd: '#52B788',
    bg: '#EAF5EE',
    label: 'Neutral',
  },
  Negative: {
    color: '#DC2626',
    gradientStart: '#DC2626',
    gradientEnd: '#F87171',
    bg: '#FEF2F2',
    label: 'Negative',
  },
};

export default function SentimentGauge({ label, score }) {
  const config = SENTIMENT_CONFIG[label] || SENTIMENT_CONFIG.Neutral;
  const percentage = Math.round(score * 100);

  // SVG arc calculation (semi-circle radius 82, center 110, 102)
  const radius = 82;
  const circumference = Math.PI * radius; // half circle ~ 257.6
  const offset = circumference - (Math.min(Math.max(score, 0), 1) * circumference);

  // Endpoint calculation for glowing indicator dot
  const angle = Math.PI - (Math.min(Math.max(score, 0), 1) * Math.PI);
  const dotX = 110 + radius * Math.cos(angle);
  const dotY = 102 - radius * Math.sin(angle);

  const gradId = `gaugeGrad-${label.toLowerCase()}`;

  return (
    <div className="sentiment-gauge" id="sentiment-gauge">
      <div className="gauge-card-header">
        <span className="gauge-header-title">Sentiment Score</span>
      </div>

      <div className="gauge-visual">
        <svg viewBox="0 0 220 125" className="gauge-svg">
          <defs>
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={config.gradientStart} />
              <stop offset="100%" stopColor={config.gradientEnd} />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor={config.color} floodOpacity="0.3" />
            </filter>
          </defs>

          {/* Background arc */}
          <path
            d="M 28 102 A 82 82 0 0 1 192 102"
            fill="none"
            stroke="rgba(38, 127, 75, 0.08)"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Subtle tick markers */}
          <path
            d="M 28 102 A 82 82 0 0 1 192 102"
            fill="none"
            stroke="#CBD5E1"
            strokeWidth="1.5"
            strokeDasharray="2 12"
            opacity="0.5"
          />

          {/* Filled active arc */}
          <path
            d="M 28 102 A 82 82 0 0 1 192 102"
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="gauge-fill"
            filter="url(#glow)"
          />

          {/* Glowing endpoint indicator */}
          {score > 0.02 && (
            <circle
              cx={dotX}
              cy={dotY}
              r="6.5"
              fill="#FFFFFF"
              stroke={config.color}
              strokeWidth="3.5"
              className="gauge-endpoint-dot"
            />
          )}
        </svg>

        <div className="gauge-center">
          <span className="gauge-percentage" style={{ color: '#172033' }}>
            {percentage}%
          </span>
          <span className="gauge-confidence-label">Confidence</span>
        </div>
      </div>
    </div>
  );
}

