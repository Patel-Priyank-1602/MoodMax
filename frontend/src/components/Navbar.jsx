import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner">
        {/* Brand Logo & Name */}
        <NavLink to="/" className="navbar-brand" id="nav-brand">
          <div className="navbar-logo-wrapper">
            <img src="/logo.png" alt="MoodMax logo" className="navbar-logo-img" />
          </div>
          <span className="navbar-title">MoodMax</span>
        </NavLink>

        {/* Center Pill Capsule (Analyzer & Batch) */}
        <div className="navbar-pill-group" role="tablist">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `navbar-pill-link ${isActive ? 'navbar-pill-link-active' : ''}`
            }
            id="nav-analyzer"
          >
            <svg
              className="nav-pill-icon"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <span>Analyzer</span>
          </NavLink>

          <NavLink
            to="/batch"
            className={({ isActive }) =>
              `navbar-pill-link ${isActive ? 'navbar-pill-link-active' : ''}`
            }
            id="nav-batch"
          >
            <svg
              className="nav-pill-icon"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/>
              <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>
              <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/>
            </svg>
            <span>Batch</span>
          </NavLink>
        </div>

        {/* Right Section: Clean Email & GitHub Icons (No AI Ready) */}
        <div className="navbar-right-actions">
          <a
            href="mailto:patelpriyank2526@gmail.com"
            className="navbar-text-icon-btn"
            title="Email (patelpriyank2526@gmail.com)"
            aria-label="Send email"
            id="nav-email-link"
          >
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.9"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect width="20" height="16" x="2" y="4" rx="2" />
              <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
            </svg>
          </a>

          <a
            href="https://github.com/Patel-Priyank-1602/MoodMax"
            target="_blank"
            rel="noopener noreferrer"
            className="navbar-text-icon-btn"
            title="GitHub Repository"
            aria-label="GitHub repository"
            id="nav-github-link"
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
              />
            </svg>
          </a>
        </div>
      </div>
    </nav>
  );
}
