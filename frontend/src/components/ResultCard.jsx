import './ResultCard.css';

const SENTIMENT_CONFIG = {
  Positive: { color: '#267F4B', bg: '#EAF5EE', border: 'rgba(38, 127, 75, 0.25)', label: 'Positive' },
  Negative: { color: '#DC2626', bg: '#FEF2F2', border: 'rgba(220, 38, 38, 0.25)', label: 'Negative' },
  Neutral: { color: '#267F4B', bg: '#EAF5EE', border: 'rgba(38, 127, 75, 0.25)', label: 'Neutral' },
};

const EMOTION_GRADIENTS = {
  joy: 'linear-gradient(90deg, #F59E0B, #FBBF24)',
  sadness: 'linear-gradient(90deg, #3B82F6, #60A5FA)',
  anger: 'linear-gradient(90deg, #EF4444, #F87171)',
  fear: 'linear-gradient(90deg, #8B5CF6, #A78BFA)',
  surprise: 'linear-gradient(90deg, #EC4899, #F472B6)',
  disgust: 'linear-gradient(90deg, #267F4B, #34D399)',
  neutral: 'linear-gradient(90deg, #64748B, #94A3B8)',
};

const EMOTION_COLORS = {
  joy: '#F59E0B',
  sadness: '#3B82F6',
  anger: '#EF4444',
  fear: '#8B5CF6',
  surprise: '#EC4899',
  disgust: '#267F4B',
  neutral: '#64748B',
};

const LANG_NAMES = {
  en: 'English', hi: 'Hindi', es: 'Spanish', fr: 'French',
  de: 'German', pt: 'Portuguese', it: 'Italian', zh: 'Chinese',
  ja: 'Japanese', ko: 'Korean', ar: 'Arabic', ru: 'Russian',
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const { sentiment, emotions, dominant_emotion, detected_lang } = result;
  const sentimentCfg = SENTIMENT_CONFIG[sentiment.label] || SENTIMENT_CONFIG.Neutral;

  // Sort emotions by score descending
  const sortedEmotions = Object.entries(emotions)
    .sort(([, a], [, b]) => b - a);

  const langName = LANG_NAMES[detected_lang?.toLowerCase()] || detected_lang || 'English';
  const dominantColor = EMOTION_COLORS[dominant_emotion?.toLowerCase()] || '#267F4B';

  return (
    <div className="result-card card animate-fade-in-up" id="analysis-result">
      {/* Top 3-Metric Summary Banner */}
      <div className="result-summary-grid">
        {/* Metric 1: Overall Sentiment */}
        <div className="summary-item summary-sentiment" style={{ borderColor: sentimentCfg.border }}>
          <div className="summary-item-header">
            <span className="summary-item-label">Sentiment</span>
            <span className="summary-status-badge" style={{ background: sentimentCfg.bg, color: sentimentCfg.color }}>
              {(sentiment.score * 100).toFixed(0)}% Match
            </span>
          </div>
          <div className="summary-item-main">
            <span className="summary-sentiment-title" style={{ color: sentimentCfg.color }}>
              {sentiment.label}
            </span>
            <span className="summary-item-sub">
              {(sentiment.score * 100).toFixed(1)}% confidence
            </span>
          </div>
        </div>

        {/* Metric 2: Dominant Emotion */}
        <div className="summary-item summary-dominant">
          <div className="summary-item-header">
            <span className="summary-item-label">Dominant Emotion</span>
            <span className="dominant-chip-badge" style={{ background: `${dominantColor}15`, color: dominantColor }}>
              Peak
            </span>
          </div>
          <div className="summary-item-main">
            <div className="dominant-val-row">
              <span className="dominant-color-dot" style={{ background: dominantColor }}></span>
              <span className="summary-dominant-title">{dominant_emotion}</span>
            </div>
            <span className="summary-item-sub">
              Primary emotional tone
            </span>
          </div>
        </div>

        {/* Metric 3: Language */}
        <div className="summary-item summary-lang">
          <div className="summary-item-header">
            <span className="summary-item-label">Language</span>
            <span className="lang-code-badge">{detected_lang?.toUpperCase() || 'EN'}</span>
          </div>
          <div className="summary-item-main">
            <span className="summary-lang-title">{langName}</span>
            <span className="summary-item-sub">Auto-detected dialect</span>
          </div>
        </div>
      </div>

      {/* Emotion Distribution Section */}
      <div className="result-emotions">
        <div className="result-section-header">
          <div className="result-title-group">
            <svg className="result-title-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#267F4B" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            <h3 className="result-section-title">Emotion Distribution</h3>
          </div>
          <span className="result-section-sub">Sorted by intensity</span>
        </div>

        <div className="emotion-bars stagger-children">
          {sortedEmotions.map(([emotion, score]) => {
            const emotionKey = emotion.toLowerCase();
            const color = EMOTION_COLORS[emotionKey] || '#267F4B';
            const gradient = EMOTION_GRADIENTS[emotionKey] || 'linear-gradient(90deg, #267F4B, #34D399)';
            const isDominant = emotionKey === dominant_emotion?.toLowerCase();
            const isNonZero = score > 0.001;

            return (
              <div
                key={emotion}
                className={`emotion-bar-row animate-fade-in-up ${isDominant ? 'row-dominant' : ''}`}
              >
                <div className="emotion-bar-label">
                  <span className="emotion-dot" style={{ background: color }}></span>
                  <span className="emotion-bar-name">{emotion}</span>
                  {isDominant && (
                    <span className="top-indicator-badge">Top</span>
                  )}
                </div>

                <div className="emotion-bar-track">
                  <div
                    className="emotion-bar-fill"
                    style={{
                      width: `${Math.max(score * 100, isNonZero ? 2 : 0)}%`,
                      background: isNonZero ? gradient : 'transparent',
                      boxShadow: isNonZero ? `0 2px 8px ${color}40` : 'none',
                    }}
                  />
                </div>

                <span className={`emotion-bar-value ${isNonZero ? 'value-active' : 'value-muted'}`}>
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

