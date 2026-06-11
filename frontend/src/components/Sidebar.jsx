import React, { useState, useEffect } from 'react';
import { NavLink, Link, useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { toggleSidebar, closeMobileSidebar } from './sidebarHelper';

const Sidebar = ({ project, datasetStatus, hasModels }) => {
  const { project_id } = useParams();
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleLogout = () => {
    closeMobileSidebar();
    logout();
    navigate('/login');
  };

  const hasData = datasetStatus?.data_loaded;

  const backTarget = project_id ? "/dashboard" : "/";

  return (
    <div className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1.5rem 1.25rem', borderBottom: '1px solid var(--border-color)', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button 
            onClick={toggleSidebar}
            style={{ 
              background: 'none', 
              border: 'none', 
              color: 'var(--text-muted)', 
              cursor: 'pointer', 
              fontSize: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '4px',
              transition: 'color 0.15s ease',
              marginRight: '0.25rem'
            }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-main)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
            title="Collapse Sidebar"
          >
            <i className="fa-solid fa-bars"></i>
          </button>
          <Link to="/" onClick={closeMobileSidebar} className="sidebar-logo" style={{ textDecoration: 'none', cursor: 'pointer', padding: 0, margin: 0, display: 'flex', gap: '0.55rem', alignItems: 'center' }}>
            <i className="fa-solid fa-cube"></i>
            <span className="text-gradient-display">instaml</span>
          </Link>
        </div>
        <button 
          onClick={toggleTheme} 
          style={{ 
            background: 'none', 
            border: '1px solid var(--border-color)', 
            color: 'var(--text-main)', 
            cursor: 'pointer', 
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '28px',
            height: '28px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--bg-primary)',
            transition: 'all 0.15s ease'
          }}
          onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--border-hover)'}
          onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-color)'}
          title={theme === 'dark' ? 'Switch to Solar Mode' : 'Switch to Dark Mode'}
        >
          <i className={theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun'}></i>
        </button>
      </div>

      <ul className="sidebar-menu" onClick={closeMobileSidebar}>
        <li className="menu-item">
          <NavLink to="/dashboard" end className={({ isActive }) => isActive ? 'active' : ''}>
            <i className="fa-solid fa-grip-vertical"></i>
            <span>Dashboard</span>
          </NavLink>
        </li>

        <li className="menu-item">
          <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>
            <i className="fa-solid fa-gear"></i>
            <span>Settings</span>
          </NavLink>
        </li>

        <li className="menu-item">
          <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">
            <i className="fa-solid fa-book"></i>
            <span>API Docs</span>
          </a>
        </li>

        <li className="menu-item" style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem', marginBottom: '0.5rem' }}>
          <NavLink to="/guidelines" className={({ isActive }) => isActive ? 'active' : ''}>
            <i className="fa-solid fa-circle-question"></i>
            <span>Guidelines</span>
          </NavLink>
        </li>

        {project_id && (
          <>
            <li className="menu-item-section-title">
              ML Pipeline
            </li>
            
            <li className="menu-item">
              <NavLink to={`/projects/${project_id}/upload`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-cloud-arrow-up"></i>
                <span>Data Upload</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasData ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/preprocess`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-sliders"></i>
                <span>Preprocessing</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasData ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/eda`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-chart-bar"></i>
                <span>EDA Analysis</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasData ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/feature-eng`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-wand-magic-sparkles"></i>
                <span>Feature Engineering</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasData ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/train`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-gears"></i>
                <span>Train Model</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasModels ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/test`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-flask"></i>
                <span>Test Model</span>
              </NavLink>
            </li>

            <li className={`menu-item ${!hasModels ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/deploy`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-rocket"></i>
                <span>Model Deploy</span>
              </NavLink>
            </li>

            <li className="menu-item-section-title">
              Management
            </li>

            <li className={`menu-item ${!hasData ? 'disabled' : ''}`}>
              <NavLink to={`/projects/${project_id}/versions`} className={({ isActive }) => isActive ? 'active' : ''}>
                <i className="fa-solid fa-code-commit"></i>
                <span>Data History</span>
              </NavLink>
            </li>

            {project?.data_type === 'text' && (
              <li className="menu-item">
                <NavLink to={`/projects/${project_id}/playground`} className={({ isActive }) => isActive ? 'active' : ''}>
                  <i className="fa-solid fa-brain"></i>
                  <span>NLP Playground</span>
                </NavLink>
              </li>
            )}
          </>
        )}
        <li className="menu-item" style={{ marginTop: 'auto', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem' }}>
          <NavLink to="/" end>
            <i className="fa-solid fa-house"></i>
            <span>Home</span>
          </NavLink>
        </li>
      </ul>

      <div className="sidebar-footer">
        {project && (
          <div style={{ paddingBottom: '0.75rem', borderBottom: '1px solid var(--border-color)', fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.15rem', marginBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Modality</span>
            <span style={{ fontWeight: '700', color: 'var(--text-main)', textTransform: 'capitalize', fontSize: '0.85rem' }}>{project.data_type}</span>
          </div>
        )}

        <div className="user-profile" style={{ margin: '0.25rem 0' }}>
          <div className="user-avatar">
            {user?.username?.substring(0, 2).toUpperCase() || 'US'}
          </div>
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
            <span style={{ fontWeight: '600', color: 'var(--text-main)', fontSize: '0.875rem', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{user?.username}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{user?.email}</span>
          </div>
        </div>
        <button onClick={handleLogout} className="btn btn-secondary" style={{ width: '100%' }}>
          <i className="fa-solid fa-arrow-right-from-bracket"></i>
          <span>Log Out</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;
