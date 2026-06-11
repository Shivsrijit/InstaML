import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AppMockup from '../components/AppMockup';
import { toast } from 'react-hot-toast';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
 
    try {
      await login(username, password);
      toast.success("Welcome back to InstaML!");
      navigate('/');
    } catch (err) {
      toast.error(err.response?.data?.detail || String(err) || "Failed to log in.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper" style={{ display: 'flex', minHeight: '100vh', width: '100vw' }}>
      <div className="hero-grid-background"></div>
      {/* Left Marketing Panel (Hidden on mobile) */}
      <div 
        className="auth-left-panel"
        style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'center', 
          padding: '4rem 4.5rem', 
          flex: 1.2, 
          borderRight: '1px solid var(--border-color)', 
          backgroundImage: 'radial-gradient(var(--dot-color) 1px, transparent 1px)', 
          backgroundSize: '24px 24px',
          backgroundColor: 'transparent'
        }}
      >
        <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem', textDecoration: 'none', alignSelf: 'flex-start' }}>
          <i className="fa-solid fa-cube" style={{ fontSize: '2.25rem', color: 'var(--accent-primary)' }}></i>
          <span style={{ fontSize: '2.25rem', fontWeight: 700, fontFamily: 'var(--font-heading)', letterSpacing: '-0.02em', color: 'var(--text-main)' }}>instaml</span>
        </Link>
        <h2 style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: '1rem', letterSpacing: '-0.01em', fontFamily: 'var(--font-display)', lineHeight: '1.3' }}>
          Developer-first no-code machine learning.
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '460px', lineHeight: '1.6', marginBottom: '2.5rem' }}>
          Connect tabular datasets, preprocess features, run distributed optuna tuning parameter jobs, and deploy serverless REST prediction endpoints in one click.
        </p>

        {/* Unified Application Mockup Dashboard */}
        <AppMockup />
      </div>

      {/* Right Form Panel */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem', backgroundColor: 'transparent' }}>
        <div className="card auth-card" style={{ width: '100%', maxWidth: '380px', border: '1px solid var(--border-color)', boxShadow: 'none' }}>
          <Link to="/" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.8rem', fontWeight: 600, marginBottom: '1.5rem', transition: 'color 0.15s ease' }} onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-main)'} onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}>
            <i className="fa-solid fa-arrow-left" style={{ fontSize: '0.7rem' }}></i>
            <span>Back to Home</span>
          </Link>

          <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.35rem', letterSpacing: '-0.03em' }}>Welcome back</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '2rem' }}>
            Sign in to manage your workspaces
          </p>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Username</label>
              <div className="input-group-custom">
                <i className="fa-solid fa-user"></i>
                <input
                  type="text"
                  className="form-control"
                  placeholder="name"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: '2rem' }}>
              <label className="form-label">Password</label>
              <div className="input-group-custom">
                <i className="fa-solid fa-lock"></i>
                <input
                  type="password"
                  className="form-control"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', marginBottom: '1.5rem' }}
              disabled={loading}
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                <>
                  <span>Sign In</span>
                  <i className="fa-solid fa-arrow-right" style={{ fontSize: '0.75rem' }}></i>
                </>
              )}
            </button>
          </form>

          <p style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            Don't have an account?{' '}
            <Link to="/register" style={{ color: 'var(--text-main)', textDecoration: 'underline', fontWeight: 600 }}>
              Sign Up
            </Link>
          </p>
        </div>
      </div>

      {/* Inline styles for hiding left panel on small screens */}
      <style>{`
        @media (max-width: 820px) {
          .auth-left-panel {
            display: none !important;
          }
        }
      `}</style>
    </div>
  );
};

export default Login;
