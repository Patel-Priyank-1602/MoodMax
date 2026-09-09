import './SentimentGauge.css';

const SENTIMENT_CONFIG = {
  Positive: { emoji: '😊', color: '#00b894', rotation: 135, label: 'Positive' },
  Negative: { emoji: '😔', color: '#ff7675', rotation: -135, label: 'Negative' },
  Neutral: { emoji: '😐', color: '#636e72', rotation: 0, label: 'Neutral' },
};

export default function SentimentGauge({ label, score }) {
  const config = SENTIMENT_CONFIG[label] || SENTIMENT_CONFIG.Neutral;
  const percentage = Math.round(score * 100);

  // SVG arc calculation
  const radius = 80;
  const circumference = Math.PI * radius; // half circle
  const offset = circumference - (score * circumference);

  return (
    <div className="sentiment-gauge" id="sentiment-gauge">
      <div className="gauge-visual">
        <svg viewBox="0 0 200 120" className="gauge-svg">
          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(108, 92, 231, 0.1)"
            strokeWidth="12"
            strokeLinecap="round"
          />
          {/* Filled arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke={config.color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="gauge-fill"
            style={{ filter: `drop-shadow(0 0 8px ${config.color}40)` }}
          />
        </svg>
        <div className="gauge-center">
          <span className="gauge-emoji">{config.emoji}</span>
          <span className="gauge-percentage" style={{ color: config.color }}>
            {percentage}%
          </span>
        </div>
      </div>
      <div className="gauge-label" style={{ color: config.color }}>
        {config.label}
      </div>
    </div>
  );
}
