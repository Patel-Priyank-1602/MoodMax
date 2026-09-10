import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getHistory } from '../api/client';
import './History.css';



export default function History() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const limit = 15;

  useEffect(() => {
    fetchHistory();
  }, [offset, search]);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getHistory({ limit, offset, search });
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load history:', err);
      if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to backend. Make sure the server is running.');
      } else {
        setError('Failed to load history.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setOffset(0);
    setSearch(searchInput);
  };

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  const truncateText = (text, maxLen = 80) => {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + '...';
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  };

  return (
    <div className="history-page" id="history-page">
      <div className="page-header">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">
          Browse and search through your past analyses.
          {total > 0 && ` ${total} total results.`}
        </p>
      </div>

      {/* Search */}
      <form className="history-search glass-card" onSubmit={handleSearch}>
        <input
          type="text"
          className="input history-search-input"
          placeholder="Search by text content..."
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          id="history-search-input"
        />
        <button type="submit" className="btn btn-primary" id="history-search-btn">
          Search
        </button>
        {search && (
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => { setSearchInput(''); setSearch(''); setOffset(0); }}
          >
            Clear
          </button>
        )}
      </form>

      {error && (
        <div className="history-error glass-card animate-fade-in-up">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {isLoading ? (
        <div className="history-loading">
          <div className="spinner" style={{ width: 40, height: 40 }}></div>
          <p>Loading history...</p>
        </div>
      ) : items.length === 0 ? (
        <div className="empty-state glass-card">
          <div className="empty-state-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="1.5">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <h3 className="empty-state-title">No analyses yet</h3>
          <p className="empty-state-text">
            Go to the <Link to="/">Analyzer</Link> to analyze some text first.
          </p>
        </div>
      ) : (
        <>
          <div className="history-table-container glass-card animate-fade-in-up">
            <table className="data-table" id="history-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Text</th>
                  <th>Sentiment</th>
                  <th>Emotion</th>
                  <th>Lang</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <Link to={`/history/${item.id}`} className="history-id-link">
                        #{item.id}
                      </Link>
                    </td>
                    <td className="history-text-cell">
                      <Link to={`/history/${item.id}`}>
                        {truncateText(item.input_text)}
                      </Link>
                    </td>
                    <td>
                      <span className={`badge badge-${item.sentiment_label?.toLowerCase()}`}>
                        {item.sentiment_label}
                      </span>
                    </td>
                    <td>
                      <span className="history-emotion">
                        {item.dominant_emotion}
                      </span>
                    </td>
                    <td className="history-lang">{item.detected_lang?.toUpperCase()}</td>
                    <td className="history-date">{formatDate(item.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="history-pagination">
              <button
                className="btn btn-secondary"
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
              >
                ← Previous
              </button>
              <span className="pagination-info">
                Page {currentPage} of {totalPages}
              </span>
              <button
                className="btn btn-secondary"
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
