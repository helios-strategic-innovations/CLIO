import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  BookOpenText,
  Bot,
  BriefcaseBusiness,
  LayoutDashboard,
  ListOrdered,
  LockKeyhole,
  LogOut,
  Menu,
  Newspaper,
  ScrollText,
  ShieldCheck,
  SlidersHorizontal,
  X,
  type LucideIcon,
} from 'lucide-react';
import { setToken } from '../lib/api';

const NAV: { to: string; label: string; desc: string; icon: LucideIcon }[] = [
  { to: '/', label: 'Overview', desc: 'Operating picture', icon: LayoutDashboard },
  { to: '/portfolio', label: 'Portfolio', desc: 'Capital and tax lots', icon: BriefcaseBusiness },
  { to: '/orders', label: 'Orders', desc: 'Execution state', icon: ListOrdered },
  { to: '/hydrinia', label: 'Hydrinia', desc: 'Autonomy engine', icon: Bot },
  { to: '/knowledge', label: 'Knowledge', desc: 'Memory corpus', icon: BookOpenText },
  { to: '/analytics', label: 'Analytics', desc: 'Performance lens', icon: BarChart3 },
  { to: '/news', label: 'Intelligence', desc: 'Market context', icon: Newspaper },
  { to: '/logs', label: 'Logs', desc: 'Event stream', icon: ScrollText },
  { to: '/control', label: 'Control', desc: 'Operator tools', icon: SlidersHorizontal },
];

export default function Layout() {
  const [menuOpen, setMenuOpen] = useState(false);

  const signOut = () => {
    setToken(null);
    window.location.href = '/login';
  };

  return (
    <div className="app-shell">
      <nav className="sidebar-rail" aria-label="Primary navigation">
        <div className="brand-panel">
          <div className="brand-lockup">
            <img
              className="brand-lockup-image"
              src="/atlas-lockup-v2.png"
              alt="Atlas Strategies - Carrying the Wealth of Your World"
            />
            <button
              type="button"
              className="mobile-menu-button"
              aria-expanded={menuOpen}
              aria-controls="atlas-nav"
              onClick={() => setMenuOpen(value => !value)}
            >
              {menuOpen ? <X size={18} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
              <span>Menu</span>
            </button>
          </div>
          <div className="brand-status" aria-label="Mission Control status">
            <div className="brand-status-row">
              <span><Activity size={14} aria-hidden="true" /> Mission Control</span>
              <strong className="status-positive">Live</strong>
            </div>
            <div className="brand-status-row">
              <span><LockKeyhole size={14} aria-hidden="true" /> Execution Gate</span>
              <strong className="status-warning">Off</strong>
            </div>
          </div>
        </div>

        <div id="atlas-nav" className={`nav-scroll ${menuOpen ? 'nav-scroll-open' : ''}`}>
          {NAV.map(item => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) => `nav-item ${isActive ? 'nav-item-active' : ''}`}
              >
                <Icon className="nav-icon" size={17} strokeWidth={1.5} aria-hidden="true" />
                <span className="nav-copy">
                  <span className="nav-label">{item.label}</span>
                  <span className="nav-desc">{item.desc}</span>
                </span>
              </NavLink>
            );
          })}
        </div>

        <div className="rail-footer">
          <p className="rail-footer-title">Atlas Strategies Trading LLC</p>
          <p className="rail-footer-note">Monitoring only. Execution gate remains off.</p>
          <button type="button" onClick={signOut} className="signout-link">
            <LogOut size={15} aria-hidden="true" />
            <span>Sign Out</span>
          </button>
        </div>
      </nav>

      <main className="main-pane">
        <div className="workspace-topline">
          <div className="workspace-title">
            <ShieldCheck size={16} aria-hidden="true" />
            <span>Mission Control</span>
          </div>
          <div className="workspace-health">
            <span className="source-status">Sources guarded</span>
            <span className="account-chip">Paper / Soak</span>
          </div>
        </div>
        <div className="content-scroll">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
