import { useState } from 'react';
import './TextInput.css';

export default function TextInput({ onAnalyze, isLoading }) {
  const [text, setText] = useState('');
  const maxLength = 5000;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onAnalyze(text.trim());
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e);
    }
  };

  return (
    <form className="text-input-container glass-card" onSubmit={handleSubmit} id="text-input-form">
      <div className="text-input-header">
        <label htmlFor="analyze-textarea" className="text-input-label">
          Enter text to analyze
        </label>
        <span className="text-input-counter">
          {text.length} / {maxLength}
        </span>
      </div>

      <textarea
        id="analyze-textarea"
        className="text-input-area"
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, maxLength))}
        onKeyDown={handleKeyDown}
        placeholder="Paste any social media text, tweet, comment, or review here... Supports multiple languages"
        rows={5}
        maxLength={maxLength}
        disabled={isLoading}
      />

      <div className="text-input-footer">
        <span className="text-input-hint">
          Ctrl + Enter to analyze
        </span>
        <div className="text-input-actions">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => setText('')}
            disabled={!text || isLoading}
          >
            Clear
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={!text.trim() || isLoading}
            id="analyze-button"
          >
            {isLoading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                Analyzing...
              </>
            ) : (
              <>Analyze</>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
