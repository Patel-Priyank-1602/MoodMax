import { useState } from 'react';
import './POSTagViewer.css';

export default function POSTagViewer({ posData }) {
  const tokens = posData?.tokens || [];
  const summary = posData?.summary || {};

  // Find the first descriptor or target to select as default highlight
  const initialSelected =
    tokens.find((t) => t.sentiment_role === 'descriptor') ||
    tokens.find((t) => t.sentiment_role === 'target') ||
    tokens[0] ||
    null;

  const [selectedToken, setSelectedToken] = useState(initialSelected);
  const [activeFilter, setActiveFilter] = useState('ALL');
  const [viewMode, setViewMode] = useState('flow'); // 'flow' | 'table'
  const [copiedState, setCopiedState] = useState(null);

  // If posData or tokens are empty, return null after hooks
  if (!posData || !posData.tokens || posData.tokens.length === 0) {
    return null;
  }

  // Filter categories
  const categories = summary?.categories || [];

  const handleCopyTags = (format = 'text') => {
    let textToCopy = '';
    if (format === 'text') {
      textToCopy = tokens.map((t) => `${t.text}/${t.tag}`).join(' ');
    } else if (format === 'json') {
      textToCopy = JSON.stringify(posData, null, 2);
    }

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopiedState(format);
      setTimeout(() => setCopiedState(null), 2200);
    });
  };

  const getFilteredTokens = () => {
    if (activeFilter === 'ALL') return tokens;
    if (activeFilter === 'DESCRIPTORS') {
      return tokens.filter((t) => t.sentiment_role === 'descriptor');
    }
    if (activeFilter === 'TARGETS') {
      return tokens.filter((t) => t.sentiment_role === 'target');
    }
    return tokens.filter((t) => t.pos === activeFilter);
  };

  const filteredTokens = getFilteredTokens();

  const isTokenHighlighted = (token) => {
    if (activeFilter === 'ALL') return true;
    if (activeFilter === 'DESCRIPTORS') return token.sentiment_role === 'descriptor';
    if (activeFilter === 'TARGETS') return token.sentiment_role === 'target';
    return token.pos === activeFilter;
  };

  const getRoleBadge = (role) => {
    switch (role) {
      case 'descriptor':
        return { label: 'Sentiment Driver', color: '#8B5CF6', bg: '#F5F3FF', icon: '✨' };
      case 'target':
        return { label: 'Aspect Target', color: '#267F4B', bg: '#EAF5EE', icon: '🎯' };
      case 'negator':
        return { label: 'Negation Inverter', color: '#EF4444', bg: '#FEF2F2', icon: '⚡' };
      case 'action':
        return { label: 'Action / State', color: '#3B82F6', bg: '#EFF6FF', icon: '⚡' };
      case 'connector':
        return { label: 'Clausal Connector', color: '#EC4899', bg: '#FDF2F8', icon: '🔗' };
      default:
        return { label: 'Grammatical', color: '#64748B', bg: '#F8FAFC', icon: '•' };
    }
  };

  return (
    <div className="pos-viewer-section card animate-fade-in-up" id="pos-tagging-analysis">
      {/* ─── Header ─── */}
      <div className="pos-header">
        <div className="pos-title-group">
          <div className="pos-icon-badge">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#267F4B" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 7V4h16v3" />
              <path d="M9 20h6" />
              <path d="M12 4v16" />
            </svg>
          </div>
          <div>
            <div className="pos-title-row">
              <h3 className="pos-title">Part-of-Speech & Syntactic Tagging</h3>
              <span className="pos-badge-live">Universal POS</span>
            </div>
            <p className="pos-subtitle">
              Token-level grammatical dissection, syntactic dependencies, and sentiment role mapping
            </p>
          </div>
        </div>

        {/* View Mode & Actions */}
        <div className="pos-header-actions">
          <div className="pos-view-switch">
            <button
              type="button"
              className={`view-switch-btn ${viewMode === 'flow' ? 'active' : ''}`}
              onClick={() => setViewMode('flow')}
              title="Visual Word Capsule Stream"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" rx="1.5" />
                <rect x="14" y="3" width="7" height="7" rx="1.5" />
                <rect x="3" y="14" width="7" height="7" rx="1.5" />
                <rect x="14" y="14" width="7" height="7" rx="1.5" />
              </svg>
              <span>Visual Flow</span>
            </button>
            <button
              type="button"
              className={`view-switch-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              title="Linguistic Syntax Table"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3h18v18H3z" />
                <path d="M3 9h18" />
                <path d="M3 15h18" />
                <path d="M9 3v18" />
              </svg>
              <span>Syntax Table</span>
            </button>
          </div>

          <div className="pos-action-buttons">
            <button
              type="button"
              className="pos-btn-utility"
              onClick={() => handleCopyTags('text')}
              title="Copy tagged sentence (e.g. Word/TAG)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <span>{copiedState === 'text' ? 'Copied Tags!' : 'Copy Tags'}</span>
            </button>

            <button
              type="button"
              className="pos-btn-utility"
              onClick={() => handleCopyTags('json')}
              title="Export structured JSON tokens"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="16 18 22 12 16 6" />
                <polyline points="8 6 2 12 8 18" />
              </svg>
              <span>{copiedState === 'json' ? 'Copied JSON!' : 'JSON'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* ─── Linguistic Metrics Bar ─── */}
      <div className="pos-metrics-grid">
        <div className="pos-metric-card">
          <span className="pos-metric-label">Lexical Density</span>
          <div className="pos-metric-val-row">
            <span className="pos-metric-number highlight-emerald">{summary?.lexical_density ?? 0}%</span>
            <span className="pos-metric-tag">Content Ratio</span>
          </div>
          <span className="pos-metric-hint">{summary?.total_words || 0} words • {summary?.total_tokens || 0} tokens</span>
        </div>

        <div className="pos-metric-card">
          <span className="pos-metric-label">Sentiment Drivers</span>
          <div className="pos-metric-val-row">
            <span className="pos-metric-number highlight-purple">{summary?.key_descriptors?.length || 0}</span>
            <span className="pos-metric-tag">Adjectives/Adverbs</span>
          </div>
          <div className="pos-chip-list">
            {(summary?.key_descriptors || []).slice(0, 3).map((desc, i) => (
              <span key={i} className="mini-chip chip-descriptor">"{desc}"</span>
            ))}
            {(summary?.key_descriptors || []).length === 0 && (
              <span className="pos-metric-hint">None detected</span>
            )}
          </div>
        </div>

        <div className="pos-metric-card">
          <span className="pos-metric-label">Aspect Targets</span>
          <div className="pos-metric-val-row">
            <span className="pos-metric-number highlight-blue">{summary?.key_targets?.length || 0}</span>
            <span className="pos-metric-tag">Noun Entities</span>
          </div>
          <div className="pos-chip-list">
            {(summary?.key_targets || []).slice(0, 3).map((target, i) => (
              <span key={i} className="mini-chip chip-target">"{target}"</span>
            ))}
            {(summary?.key_targets || []).length === 0 && (
              <span className="pos-metric-hint">None detected</span>
            )}
          </div>
        </div>

        <div className="pos-metric-card">
          <span className="pos-metric-label">Syntactic Balance</span>
          <div className="pos-metric-val-row">
            <span className="pos-metric-number highlight-slate">{categories.length}</span>
            <span className="pos-metric-tag">Grammar Classes</span>
          </div>
          <span className="pos-metric-hint">Penn Treebank & UPOS mapped</span>
        </div>
      </div>

      {/* ─── Category Filter Tabs ─── */}
      <div className="pos-filters-container">
        <span className="pos-filter-legend">Filter by Category:</span>
        <div className="pos-filter-pills">
          <button
            type="button"
            className={`filter-pill ${activeFilter === 'ALL' ? 'active' : ''}`}
            onClick={() => setActiveFilter('ALL')}
          >
            <span>All Tokens</span>
            <span className="pill-count">{tokens.length}</span>
          </button>

          <button
            type="button"
            className={`filter-pill pill-descriptor ${activeFilter === 'DESCRIPTORS' ? 'active' : ''}`}
            onClick={() => setActiveFilter('DESCRIPTORS')}
          >
            <span className="filter-dot" style={{ background: '#8B5CF6' }} />
            <span>Sentiment Drivers</span>
            <span className="pill-count">
              {tokens.filter((t) => t.sentiment_role === 'descriptor').length}
            </span>
          </button>

          <button
            type="button"
            className={`filter-pill pill-target ${activeFilter === 'TARGETS' ? 'active' : ''}`}
            onClick={() => setActiveFilter('TARGETS')}
          >
            <span className="filter-dot" style={{ background: '#267F4B' }} />
            <span>Aspect Targets</span>
            <span className="pill-count">
              {tokens.filter((t) => t.sentiment_role === 'target').length}
            </span>
          </button>

          {categories.map((cat) => (
            <button
              key={cat.pos_code}
              type="button"
              className={`filter-pill ${activeFilter === cat.pos_code ? 'active' : ''}`}
              onClick={() => setActiveFilter(cat.pos_code)}
            >
              <span className="filter-dot" style={{ background: cat.color }} />
              <span>{cat.category}</span>
              <span className="pill-count">{cat.count}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ─── Mode 1: Interactive Visual Flow Stream ─── */}
      {viewMode === 'flow' && (
        <div className="pos-flow-wrapper">
          <div className="pos-flow-canvas">
            {tokens.map((token) => {
              const highlighted = isTokenHighlighted(token);
              const isSelected = selectedToken?.id === token.id;

              return (
                <div
                  key={token.id}
                  className={`pos-token-capsule ${highlighted ? 'highlighted' : 'dimmed'} ${isSelected ? 'selected' : ''
                    }`}
                  onClick={() => setSelectedToken(token)}
                  style={{
                    '--token-color': token.color,
                    '--token-bg': token.bg,
                    '--token-border': token.border,
                  }}
                  title={`Click to inspect '${token.text}' (${token.pos_label})`}
                >
                  <div className="capsule-tag-row">
                    <span
                      className="capsule-tag"
                      style={{
                        background: token.bg,
                        color: token.color,
                        borderColor: token.border,
                      }}
                    >
                      {token.pos}
                    </span>
                    {token.sentiment_role === 'descriptor' && (
                      <span className="capsule-role-dot" title="Sentiment Driver">✨</span>
                    )}
                    {token.sentiment_role === 'target' && (
                      <span className="capsule-role-dot" title="Aspect Target">🎯</span>
                    )}
                    {token.sentiment_role === 'negator' && (
                      <span className="capsule-role-dot" title="Negation Inverter">⚡</span>
                    )}
                  </div>

                  <span className="capsule-word">{token.text}</span>

                  <span className="capsule-dep" title={`Dependency: ${token.dep_label}`}>
                    {token.dep}
                  </span>
                </div>
              );
            })}
          </div>

          {/* ─── Selected Token Deep Inspector Panel ─── */}
          {selectedToken && (
            <div className="pos-inspector-card animate-fade-in">
              <div className="inspector-header">
                <div className="inspector-token-title-group">
                  <span
                    className="inspector-tag-badge"
                    style={{
                      background: selectedToken.bg,
                      color: selectedToken.color,
                      borderColor: selectedToken.border,
                    }}
                  >
                    {selectedToken.pos} • {selectedToken.pos_label}
                  </span>
                  <h4 className="inspector-word">"{selectedToken.text}"</h4>
                  <span className="inspector-lemma">
                    Lemma: <code>{selectedToken.lemma}</code>
                  </span>
                </div>

                <div className="inspector-role-badge">
                  {(() => {
                    const badge = getRoleBadge(selectedToken.sentiment_role);
                    return (
                      <span
                        className="role-pill"
                        style={{
                          background: badge.bg,
                          color: badge.color,
                        }}
                      >
                        {badge.icon} {badge.label}
                      </span>
                    );
                  })()}
                </div>
              </div>

              <div className="inspector-details-grid">
                <div className="inspector-field">
                  <span className="field-label">Penn Treebank Tag</span>
                  <div className="field-value-row">
                    <code className="tag-code">{selectedToken.tag}</code>
                    <span className="field-desc">{selectedToken.tag_description}</span>
                  </div>
                </div>

                <div className="inspector-field">
                  <span className="field-label">Syntactic Dependency</span>
                  <div className="field-value-row">
                    <code className="dep-code">{selectedToken.dep}</code>
                    <span className="field-desc">{selectedToken.dep_label}</span>
                  </div>
                </div>

                <div className="inspector-field">
                  <span className="field-label">Governing Head</span>
                  <div className="field-value-row">
                    <span className="head-word">"{selectedToken.head_text}"</span>
                    <span className="field-desc">Parent node in syntax tree</span>
                  </div>
                </div>

                <div className="inspector-field">
                  <span className="field-label">Span Coordinates</span>
                  <div className="field-value-row">
                    <span className="char-span">Chars [{selectedToken.start_char}..{selectedToken.end_char}]</span>
                    <span className="field-desc">{selectedToken.is_stop ? 'Stopword' : 'Lexical token'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─── Mode 2: Detailed Linguistic Syntax Table ─── */}
      {viewMode === 'table' && (
        <div className="pos-table-container animate-fade-in">
          <div className="pos-table-scroll">
            <table className="pos-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Word Token</th>
                  <th>Lemma</th>
                  <th>Universal POS</th>
                  <th>Penn Tag</th>
                  <th>Dependency</th>
                  <th>Head Node</th>
                  <th>Sentiment Syntactic Role</th>
                </tr>
              </thead>
              <tbody>
                {filteredTokens.map((token, index) => {
                  const roleBadge = getRoleBadge(token.sentiment_role);
                  const isSelected = selectedToken?.id === token.id;

                  return (
                    <tr
                      key={token.id}
                      className={`pos-table-row ${isSelected ? 'row-selected' : ''}`}
                      onClick={() => setSelectedToken(token)}
                    >
                      <td className="col-idx">{index + 1}</td>
                      <td className="col-word">
                        <strong>{token.text}</strong>
                      </td>
                      <td className="col-lemma">
                        <code>{token.lemma}</code>
                      </td>
                      <td className="col-pos">
                        <span
                          className="pos-table-pill"
                          style={{
                            background: token.bg,
                            color: token.color,
                            borderColor: token.border,
                          }}
                        >
                          {token.pos}
                        </span>
                        <span className="pos-table-label">{token.pos_label}</span>
                      </td>
                      <td className="col-tag">
                        <span className="tag-fine">{token.tag}</span>
                        <span className="tag-fine-desc">{token.tag_description}</span>
                      </td>
                      <td className="col-dep">
                        <span className="dep-fine">{token.dep}</span>
                        <span className="dep-fine-desc">{token.dep_label}</span>
                      </td>
                      <td className="col-head">
                        <span className="head-badge">"{token.head_text}"</span>
                      </td>
                      <td className="col-role">
                        <span
                          className="role-pill-sm"
                          style={{
                            background: roleBadge.bg,
                            color: roleBadge.color,
                          }}
                        >
                          {roleBadge.icon} {roleBadge.label}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ─── Category Distribution Meter ─── */}
      <div className="pos-distribution-footer">
        <div className="distribution-header">
          <span className="dist-title">Grammatical Composition</span>
          <span className="dist-sub">Distribution of parts of speech</span>
        </div>

        <div className="pos-distribution-bar">
          {categories.map((cat) => (
            <div
              key={cat.pos_code}
              className="dist-segment"
              style={{
                width: `${cat.percentage}%`,
                background: cat.color,
              }}
              title={`${cat.category} (${cat.pos_code}): ${cat.count} (${cat.percentage}%)`}
            />
          ))}
        </div>

        <div className="pos-distribution-legend">
          {categories.slice(0, 7).map((cat) => (
            <div key={cat.pos_code} className="legend-item">
              <span className="legend-color-dot" style={{ background: cat.color }} />
              <span className="legend-label">{cat.category}</span>
              <span className="legend-pct">{cat.percentage}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
