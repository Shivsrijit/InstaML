import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import AppMockup from '../components/AppMockup';

const Landing = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [emailInput, setEmailInput] = useState('');
  const [subscribed, setSubscribed] = useState(false);

  // FAQ Accordion state
  const [openFaq, setOpenFaq] = useState(null);

  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const faqs = [
    {
      q: "What data formats are supported?",
      a: "InstaML currently supports structured tabular tables (CSV, Excel, Parquet). Support for copy-pasted text logs, notepad files (.txt), and ZIP folder archives containing categories for image and sound classification are coming soon."
    },
    {
      q: "How does the serverless deployment work?",
      a: "When you click 'Deploy Active API' on your validated checkpoint, InstaML immediately provisions a secure, stateless prediction gateway endpoint on Port 8000/8002 that accepts JSON features and returns serverless inferences in under 50ms."
    },
    {
      q: "What libraries power the model training?",
      a: "Our unified trainer utilizes high-performance algorithms including Scikit-Learn pipelines, PyTorch networks, and Optuna distributed trials. It automatically handles features scaling, encoders, and target type classification or regression detection."
    },
    {
      q: "Can I export or rollback datasets?",
      a: "Yes! Every preprocessing step creates a versioned SQLite transaction node. You can compare changes side-by-side or rollback the workspace state to older configurations instantly on the Data History timeline."
    },
    {
      q: "Is there a limit on dataset upload sizes?",
      a: "Yes. The cloud deployment enforces a dataset upload limit of 30MB to preserve shared server stability. When running InstaML locally, there is no strict software limit—the capacity is entirely dependent on your PC's hardware capacity and available RAM."
    },
    {
      q: "What machine learning models are available for training?",
      a: "For tabular tasks, we support XGBoost, Random Forest, LightGBM, and Linear classifiers or regressors. Support for training text/NLP models (using linear algorithms or zero-shot classifiers) and image/sound classification (using custom CNNs and ResNet architectures) is coming soon."
    },
    {
      q: "How do I integrate the deployed API endpoints into my application?",
      a: "Every active API deployment provides copy-pasteable Python, JavaScript, and shell cURL request snippets. You can query the predictions gateway using secure authorization headers and JSON payloads from any HTTP client."
    }
  ];

  const handleSubscribe = (e) => {
    e.preventDefault();
    if (emailInput.trim()) {
      setSubscribed(true);
      setEmailInput('');
    }
  };

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', color: 'var(--text-main)', position: 'relative', zIndex: 1 }}>
      
      {/* Top Navbar */}
      <header className="landing-header">
        <div className="landing-header-container">
          <Link to="/" className="landing-logo-container">
            <i className="fa-solid fa-cube" style={{ fontSize: '1.5rem', color: 'var(--accent-purple)' }}></i>
            <span className="text-gradient-display" style={{ fontSize: '1.6rem', fontWeight: 800, fontFamily: 'var(--font-heading)', letterSpacing: '-0.03em' }}>instaml</span>
          </Link>

          <nav className="landing-nav">
            <a href="#features" className="landing-nav-link">Features</a>
            <a href="#beta" className="landing-nav-link">Beta Program</a>
            <a href="#faqs" className="landing-nav-link">FAQs</a>
            <a href="#contact" className="landing-nav-link">Contact</a>
          </nav>

          <div className="landing-header-actions">
            {/* GitHub Link Icon */}
            <a 
              href="https://github.com/shivsrijit/instaml" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-glass-pill landing-header-btn-round"
              title="View on GitHub"
            >
              <i className="fa-brands fa-github"></i>
            </a>

            {/* Theme Toggle Button */}
            <button 
              onClick={toggleTheme} 
              className="btn-glass-pill landing-header-btn-round"
              title={theme === 'dark' ? 'Switch to Solar Mode' : 'Switch to Dark Mode'}
            >
              <i className={theme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun'}></i>
            </button>

            {user ? (
              <Link to="/dashboard" className="btn-glass-pill landing-header-btn-action">
                <span>Dashboard</span>
                <i className="fa-solid fa-arrow-right" style={{ fontSize: '0.75rem' }}></i>
              </Link>
            ) : (
              <>
                <Link to="/login" className="landing-nav-link landing-header-signin-link" style={{ fontWeight: 600 }}>Sign In</Link>
                <Link to="/register" className="btn-glass-pill landing-header-btn-action">Join Beta</Link>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section 
        style={{ 
          width: '100%',
          borderBottom: '1px solid var(--border-color)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <div className="hero-grid-background"></div>
        <div 
          className="landing-section"
          style={{ 
            padding: '6rem 2rem 4rem', 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            textAlign: 'center',
            position: 'relative',
            zIndex: 1
          }}
        >
          <span style={{ 
            fontSize: '0.7rem', 
            fontWeight: 600, 
            color: 'var(--accent-purple)', 
            textTransform: 'uppercase', 
            letterSpacing: '0.15em',
            backgroundColor: 'rgba(201, 90, 73, 0.08)',
            padding: '0.4rem 1rem',
            borderRadius: '9999px',
            border: '1px solid rgba(201, 90, 73, 0.15)',
            fontFamily: 'var(--font-mono)',
            marginBottom: '1.5rem'
          }}>
            No-Code ML Orchestration for Devs
          </span>
          <h1 style={{ 
            fontSize: '3.75rem', 
            fontWeight: 700, 
            fontFamily: 'var(--font-display)', 
            maxWidth: '820px', 
            lineHeight: '1.15', 
            letterSpacing: '-0.03em',
            color: 'var(--text-main)'
          }}>
            <span className="text-gradient-display">Automated machine learning.</span><br />Styled for developer minds.
          </h1>
          <p style={{ 
            fontSize: '1.05rem', 
            color: 'var(--text-muted)', 
            maxWidth: '540px', 
            lineHeight: '1.6', 
            marginTop: '1.5rem',
            marginBottom: '3rem',
            fontFamily: 'var(--font-body)'
          }}>
            Connect datasets, clean features, compute correlation grids, optimize hyperparameters, and serve secure REST endpoints in a clean graphic workspace.
          </p>

          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
            {user ? (
              <button onClick={() => navigate('/dashboard')} className="btn btn-primary" style={{ padding: '0.75rem 2rem', borderRadius: 'var(--radius-sm)', fontSize: '0.9rem' }}>
                Open Console Dashboard
              </button>
            ) : (
              <>
                <button onClick={() => navigate('/register')} className="btn btn-primary" style={{ padding: '0.75rem 2rem', borderRadius: 'var(--radius-sm)', fontSize: '0.9rem' }}>
                  Join the Free Beta
                </button>
                <button onClick={() => navigate('/login')} className="btn-glass-pill" style={{ padding: '0.75rem 2.25rem', fontSize: '0.9rem' }}>
                  Sign in to Console
                </button>
              </>
            )}
          </div>

          {/* Highlighted Upload Size Limits */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)',
            backgroundColor: 'var(--bg-glass)',
            border: '1px solid var(--border-color)',
            padding: '0.45rem 1.15rem',
            borderRadius: '9999px',
            marginBottom: '3.5rem',
            boxShadow: 'var(--shadow-sm)',
            backdropFilter: 'blur(8px)'
          }}>
            <i className="fa-solid fa-cloud-arrow-up" style={{ color: 'var(--accent-purple)' }}></i>
            <span>Cloud Upload Limit: <strong style={{ color: 'var(--text-main)' }}>30MB</strong></span>
            <span style={{ color: 'var(--border-hover)' }}>|</span>
            <i className="fa-solid fa-laptop" style={{ color: 'var(--accent-green)' }}></i>
            <span>Local Setup Limit: <strong style={{ color: 'var(--text-main)' }}>Depends on PC RAM</strong></span>
          </div>

          {/* Unified Application Mockup Dashboard */}
          <AppMockup />
        </div>
      </section>

      {/* Features Bento Grid Section */}
      <section id="features" className="landing-section" style={{ padding: '6rem 2rem', borderBottom: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: '4rem' }}>
          <h2 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-display)', fontWeight: 500, color: 'var(--text-main)', letterSpacing: '-0.02em', margin: '0 auto' }}>Automatic ML Pipeline Stages</h2>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontFamily: 'var(--font-body)' }}>Everything you need to go from raw file formats to production serving APIs.</p>
        </div>

        <div className="landing-grid">
          
          {/* Feature 1: Tabular / Audio / Image Upload */}
          <div className="card bento-card">
            <div className="bento-icon-wrapper accent-purple">
              <i className="fa-solid fa-cloud-arrow-up"></i>
            </div>
            <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-heading)', fontWeight: 600, color: 'var(--text-main)', marginTop: '0.5rem' }}>1. Flexible Data Formats</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)' }}>
              Easily upload CSV sheets, zip folder paths of images or audio categories, or simply paste raw text logs. InstaML detects formatting automatically.
            </p>
          </div>

          {/* Feature 2: Smart Preprocessing */}
          <div className="card bento-card">
            <div className="bento-icon-wrapper accent-purple">
              <i className="fa-solid fa-sliders"></i>
            </div>
            <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-heading)', fontWeight: 600, color: 'var(--text-main)', marginTop: '0.5rem' }}>2. Smart Preprocessing</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)' }}>
              Impute missing cells, scale numeric fields (Standard/MinMax), and encode strings with One-Hot or Ordinal values in a few checklist clicks.
            </p>
          </div>

          {/* Feature 3: EDA & Heatmaps */}
          <div className="card bento-card">
            <div className="bento-icon-wrapper accent-green">
              <i className="fa-solid fa-chart-line"></i>
            </div>
            <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-heading)', fontWeight: 600, color: 'var(--text-main)', marginTop: '0.5rem' }}>3. Exploratory Analysis</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)' }}>
              Inspect distribution histograms, bivariate scatter plots, Explained Variance PCA graphs, and a complete correlation heatmap matrix.
            </p>
          </div>

          {/* Feature 4: Hyperparameter Optuna Tuning */}
          <div className="card bento-card bento-card-wide">
            <div className="bento-icon-wrapper accent-purple">
              <i className="fa-solid fa-gears"></i>
            </div>
            <div>
              <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-heading)', fontWeight: 600, color: 'var(--text-main)' }}>4. Real-time Training & Tuning</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)', marginTop: '0.5rem' }}>
                Choose algorithm models (Random Forest, Logistic Regression, ResNet18, CNNs) and trigger background training threads. Enable Optuna search options to run trials, optimizing weights and hyperparameters automatically with real-time CLI terminal log feedback.
              </p>
            </div>
          </div>

          {/* Feature 5: One-Click REST Deploys */}
          <div className="card bento-card">
            <div className="bento-icon-wrapper accent-green">
              <i className="fa-solid fa-rocket"></i>
            </div>
            <h3 style={{ fontSize: '1.2rem', fontFamily: 'var(--font-heading)', fontWeight: 600, color: 'var(--text-main)', marginTop: '0.5rem' }}>5. Serverless serving APIs</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)' }}>
              Deploy models with one click, run live single/batch tests on forms, and instantly export auto-generated Python integration request templates.
            </p>
          </div>

        </div>
      </section>

      {/* Beta Status Section */}
      <section id="beta" className="landing-section" style={{ padding: '6rem 2rem', borderBottom: '1px solid var(--border-color)' }}>
        <div className="card" style={{ maxWidth: '720px', margin: '0 auto', padding: '4rem 3rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.25rem', textAlign: 'center' }}>
          <div className="bento-icon-wrapper accent-green" style={{ width: '48px', height: '48px', borderRadius: '50%', fontSize: '1.25rem' }}>
            <i className="fa-solid fa-lock-open"></i>
          </div>
          <h2 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-display)', fontWeight: 500, color: 'var(--text-main)', letterSpacing: '-0.02em', margin: 0 }}>
            InstaML Public Beta
          </h2>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', lineHeight: '1.6', fontFamily: 'var(--font-body)', maxWidth: '540px' }}>
            InstaML is currently in open public beta. During this period, all pipelines, datasets, Optuna parameters, and REST serving endpoints are <strong>completely free</strong>. We support dataset uploads up to 30MB on our cloud platform, while local setups have no strict software limits and depend entirely on your PC's available memory and hardware capacity.
          </p>
          <div style={{ 
            border: '1px solid var(--border-color)', 
            padding: '0.75rem 2rem', 
            borderRadius: '9999px', 
            backgroundColor: 'var(--bg-primary)', 
            fontSize: '0.85rem', 
            color: 'var(--text-muted)', 
            marginTop: '1.25rem',
            fontFamily: 'var(--font-body)',
            fontWeight: 600
          }}>
            $0 / month — Free Public Access
          </div>
        </div>
      </section>

      {/* FAQ Accordion Section */}
      <section id="faqs" className="landing-section" style={{ padding: '6rem 2rem', borderBottom: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: '4.5rem' }}>
          <h2 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-display)', fontWeight: 500, color: 'var(--text-main)', letterSpacing: '-0.02em', margin: '0 auto' }}>Frequently Asked Questions</h2>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', marginTop: '0.5rem', fontFamily: 'var(--font-body)' }}>Quick answers to structural queries about the InstaML platform.</p>
        </div>

        <div style={{ maxWidth: '680px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {faqs.map((faq, idx) => (
            <div key={idx} className="card faq-card">
              <button onClick={() => toggleFaq(idx)} className="faq-btn">
                <span>{faq.q}</span>
                <i className={`fa-solid ${openFaq === idx ? 'fa-minus' : 'fa-plus'}`} style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}></i>
              </button>
              {openFaq === idx && (
                <div className="faq-answer">
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Newsletter / Join the Beta Section */}
      <section id="contact" className="landing-section" style={{ padding: '6rem 2rem', textAlign: 'center' }}>
        <div className="card" style={{ 
          maxWidth: '720px', 
          margin: '0 auto', 
          padding: '4rem 3rem', 
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem'
        }}>
          <h2 style={{ fontSize: '2.5rem', fontFamily: 'var(--font-display)', fontWeight: 500, color: 'var(--text-main)', margin: 0, letterSpacing: '-0.02em' }}>
            Join the InstaML Beta Program
          </h2>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', maxWidth: '480px', lineHeight: '1.6', fontFamily: 'var(--font-body)', margin: '0 0 1rem' }}>
            Ready to clean and deploy machine learning models with zero setup configuration? Enter your email address to receive beta announcements and access updates.
          </p>

          {subscribed ? (
            <div className="alert alert-success" style={{ margin: 0, padding: '0.75rem 1.5rem', width: '100%', maxWidth: '400px', justifyContent: 'center' }}>
              <i className="fa-solid fa-circle-check"></i>
              <span>Thank you! Your email has been registered for the beta program.</span>
            </div>
          ) : (
            <form onSubmit={handleSubscribe} style={{ display: 'flex', width: '100%', maxWidth: '440px', gap: '0.5rem' }}>
              <input 
                type="email" 
                placeholder="developer@domain.com"
                value={emailInput}
                onChange={(e) => setEmailInput(e.target.value)}
                className="form-control"
                style={{ flex: 1, padding: '0.75rem 1rem', border: '1px solid var(--border-color)' }}
                required
              />
              <button type="submit" className="btn btn-primary" style={{ padding: '0 1.5rem', borderRadius: 'var(--radius-sm)' }}>
                <span>Join Beta</span>
              </button>
            </form>
          )}
        </div>
      </section>

      {/* Simple Footer */}
      <footer className="landing-footer">
        <div className="landing-footer-container">
          <span className="landing-footer-text">&copy; {new Date().getFullYear()} InstaML. All rights reserved.</span>
          <div className="landing-footer-links">
            <a href="https://github.com/shivsrijit/instaml" target="_blank" rel="noopener noreferrer" className="landing-footer-link" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
              <i className="fa-brands fa-github"></i>
              <span>GitHub</span>
            </a>
            <a href="#features" className="landing-footer-link">Privacy Policy</a>
            <a href="#features" className="landing-footer-link">Terms of Service</a>
            <a href="mailto:shivsrijit@gmail.com" className="landing-footer-link">shivsrijit@gmail.com</a>
          </div>
        </div>
      </footer>

    </div>
  );
};

export default Landing;
