import { NavLink } from 'react-router-dom';
import './Navbar.css';

const navItems = [
  { path: '/', label: 'Analyzer', icon: '🔍' },
  { path: '/batch', label: 'Batch', icon: '📊' },
  { path: '/history', label: 'History', icon: '📋' },
];

export default function Navbar() {
  return (
    <nav className="navbar" id="main-navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand">
          <span className="navbar-logo">🧠</span>
          <span className="navbar-title">MoodMax</span>
        </NavLink>

        <div className="navbar-links">
          {navItems.map(({ path, label, icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `navbar-link ${isActive ? 'navbar-link-active' : ''}`
              }
              id={`nav-${label.toLowerCase()}`}
            >
              <span className="navbar-link-icon">{icon}</span>
              <span className="navbar-link-label">{label}</span>
            </NavLink>
          ))}
        </div>

        <div className="navbar-status">
          <span className="status-dot"></span>
          <span className="status-text">AI Ready</span>
        </div>
      </div>
    </nav>
  );
}
