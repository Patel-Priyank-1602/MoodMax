import './ResultCard.css';

const SENTIMENT_CONFIG = {
  Positive: { emoji: '😊', color: 'var(--color-positive)', bg: 'rgba(0, 184, 148, 0.1)' },
  Negative: { emoji: '😔', color: 'var(--color-negative)', bg: 'rgba(255, 118, 117, 0.1)' },
  Neutral: { emoji: '😐', color: 'var(--text-secondary)', bg: 'rgba(99, 110, 114, 0.1)' },
};

const EMOTION_EMOJIS = {
  joy: '😄',
  sadness: '😢',
  anger: '😠',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  neutral: '😐',
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

  return (
    <div className="result-card glass-card animate-fade-in-up" id="analysis-result">
      {/* Header: Sentiment + Language */}
      <div className="result-header">
        <div className="result-sentiment" style={{ background: sentimentCfg.bg }}>
          <span className="result-sentiment-emoji">{sentimentCfg.emoji}</span>
          <div className="result-sentiment-info">
            <span className="result-sentiment-label" style={{ color: sentimentCfg.color }}>
              {sentiment.label}
            </span>
            <span className="result-sentiment-score">
              {(sentiment.score * 100).toFixed(1)}% confident
            </span>
          </div>
        </div>

        <div className="result-meta">
          <div className="result-meta-item">
            <span className="result-meta-label">Language</span>
            <span className="result-meta-value">
              {LANG_NAMES[detected_lang] || detected_lang}
            </span>
          </div>
          <div className="result-meta-item">
            <span className="result-meta-label">Dominant</span>
            <span className="result-meta-value">
              {EMOTION_EMOJIS[dominant_emotion] || '❓'} {dominant_emotion}
            </span>
          </div>
        </div>
      </div>

      {/* Emotion Bars */}
      <div className="result-emotions">
        <h3 className="result-section-title">Emotion Distribution</h3>
        <div className="emotion-bars stagger-children">
          {sortedEmotions.map(([emotion, score]) => (
            <div key={emotion} className="emotion-bar-row animate-fade-in-up">
              <div className="emotion-bar-label">
                <span className="emotion-bar-emoji">{EMOTION_EMOJIS[emotion] || '❓'}</span>
                <span className="emotion-bar-name">{emotion}</span>
              </div>
              <div className="emotion-bar-track">
                <div
                  className="emotion-bar-fill"
                  style={{
                    width: `${Math.max(score * 100, 1)}%`,
                    background: `var(--color-${emotion}, var(--accent-primary))`,
                  }}
                />
              </div>
              <span className="emotion-bar-value">
                {(score * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
