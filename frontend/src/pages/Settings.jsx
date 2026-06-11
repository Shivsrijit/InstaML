import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import { toggleSidebar, closeMobileSidebar } from '../components/sidebarHelper';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import { toast } from 'react-hot-toast';

const Settings = () => {
  const { user, login } = useAuth();
  const navigate = useNavigate();

  // Profile Form state
  const [email, setEmail] = useState(user?.email || '');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    // Initial sidebar load check
    const isHidden = localStorage.getItem('sidebar-hidden') === 'true';
    if (isHidden) {
      document.documentElement.classList.add('sidebar-hidden');
    } else {
      document.documentElement.classList.remove('sidebar-hidden');
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password && password !== confirmPassword) {
      toast.error("New passwords do not match.");
      return;
    }

    setSaving(true);
    try {
      const payload = {};
      if (email.trim() && email !== user.email) {
        payload.email = email;
      }
      if (password.trim()) {
        payload.password = password;
      }

      if (Object.keys(payload).length === 0) {
        toast.error("No changes made to update.");
        setSaving(false);
        return;
      }

      const res = await api.put('/auth/me', payload);
      
      if (user) {
        user.email = res.data.email;
      }

      toast.success("Profile settings updated successfully!");
      setPassword('');
      setConfirmPassword('');
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to update settings.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar />
      
      {/* Floating expand/toggle button */}
      <button 
        className="sidebar-toggle-floating" 
        onClick={toggleSidebar} 
        title="Toggle Sidebar"
      >
        <i className="fa-solid fa-bars"></i>
      </button>

      {/* Backdrop overlay for mobile drawer */}
      <div 
        className="main-content-overlay" 
        onClick={closeMobileSidebar}
      />

      <div 
        className="main-content" 
        style={{ 
          padding: '3rem 4rem', 
          minHeight: '100vh', 
          flex: 1
        }}
      >
        <div className="hero-grid-background"></div>
        {/* Header bar with Title */}
        <div className="header-bar" style={{ marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="page-title-section">
              <h1 className="page-title" style={{ fontSize: '2.25rem', fontWeight: 500, marginBottom: '0.15rem', lineHeight: '1.2' }}>Account Settings</h1>
              <p className="page-subtitle" style={{ fontSize: '0.95rem', color: 'var(--text-muted)', fontFamily: 'var(--font-body)', margin: 0 }}>Manage your developer profile and compute subscription plans.</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem', alignItems: 'flex-start' }}>
          
          {/* Card 1: User Profile Form */}
          <div className="card" style={{ padding: '2rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, fontFamily: 'var(--font-heading)', color: 'var(--text-main)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <i className="fa-solid fa-user-gear" style={{ color: 'var(--accent-primary)', fontSize: '0.95rem' }}></i>
              <span>Profile Credentials</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Update your user email address or change your account password.</p>



            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.15rem' }}>
              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Username</label>
                <input 
                  type="text" 
                  className="form-control" 
                  value={user?.username || ''} 
                  disabled 
                  style={{ opacity: 0.6, cursor: 'not-allowed', backgroundColor: 'var(--bg-tertiary)' }}
                />
                <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '0.25rem', display: 'block' }}>Usernames cannot be changed once registered.</span>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Email Address</label>
                <input 
                  type="email" 
                  className="form-control" 
                  placeholder="you@domain.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              <div style={{ borderTop: '1px solid var(--border-color)', margin: '0.5rem 0' }}></div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">New Password</label>
                <input 
                  type="password" 
                  className="form-control" 
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '0.25rem', display: 'block' }}>Leave blank to keep your current password.</span>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Confirm Password</label>
                <input 
                  type="password" 
                  className="form-control" 
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>

              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{ width: 'fit-content', padding: '0.6rem 1.5rem', alignSelf: 'flex-start', marginTop: '0.5rem' }}
                disabled={saving}
              >
                {saving ? <span className="spinner"></span> : <span>Save Profile Changes</span>}
              </button>
            </form>
          </div>

          {/* Right Column details */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            
            {/* Card 2: Subscription plan */}
            <div className="card" style={{ padding: '2rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, fontFamily: 'var(--font-heading)', color: 'var(--text-main)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <i className="fa-solid fa-credit-card" style={{ color: 'var(--accent-purple)', fontSize: '0.95rem' }}></i>
                <span>Compute Subscription</span>
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>View details of your active compute resource subscription.</p>

              <div style={{ border: '1px solid var(--border-color)', borderRadius: '12px', padding: '1rem 1.25rem', backgroundColor: 'var(--bg-tertiary)', marginBottom: '1.25rem' }}>
                <span style={{ fontSize: '0.6rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: 700, display: 'block', letterSpacing: '0.05em' }}>Current Active Plan</span>
                <span style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)', display: 'block', margin: '0.2rem 0' }}>Free Public Beta</span>
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-green)' }}>$0.00 / month</span>
              </div>

              <ul style={{ fontSize: '0.8rem', color: 'var(--text-muted)', paddingLeft: '1.15rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', lineHeight: '1.4' }}>
                <li><strong>Unlimited workspaces</strong>: Create as many tabular, text, image, or audio pipelines as needed.</li>
                <li><strong>Distributed Tuning</strong>: Execute background Optuna hyperparameter searches with up to 50 trials per run.</li>
                <li><strong>Stateless API Serving</strong>: Immediate REST deployment endpoints on Port 8000 & 8002 with 0 latency limits.</li>
              </ul>
            </div>

            {/* Card 3: Developer resources */}
            <div className="card" style={{ padding: '1.75rem 2rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 600, fontFamily: 'var(--font-heading)', color: 'var(--text-main)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <i className="fa-solid fa-network-wired" style={{ color: 'var(--accent-green)', fontSize: '0.95rem' }}></i>
                <span>API & Developer Resources</span>
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Quickly access backend OpenAPI references and documentation gates.</p>

              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <li>
                  <a 
                    href="http://127.0.0.1:8000/docs" 
                    target="_blank" 
                    rel="noreferrer" 
                    className="btn btn-secondary"
                    style={{ display: 'flex', width: '100%', justifyContent: 'flex-start', fontSize: '0.8rem' }}
                  >
                    <i className="fa-solid fa-book"></i>
                    <span>Swagger UI API Interactive Docs</span>
                  </a>
                </li>
                <li>
                  <a 
                    href="http://127.0.0.1:8000/redoc" 
                    target="_blank" 
                    rel="noreferrer" 
                    className="btn btn-secondary"
                    style={{ display: 'flex', width: '100%', justifyContent: 'flex-start', fontSize: '0.8rem' }}
                  >
                    <i className="fa-solid fa-scroll"></i>
                    <span>ReDoc Static API Reference</span>
                  </a>
                </li>
              </ul>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
};

export default Settings;
