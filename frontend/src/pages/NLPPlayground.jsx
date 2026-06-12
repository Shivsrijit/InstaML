import React from 'react';

const NLPPlayground = () => {
  return (
    <div>
      <div className="header-bar" style={{ marginBottom: '2rem' }}>
        <div className="page-title-section">
          <h1 className="page-title">NLP Analysis Playground</h1>
          <p className="page-subtitle" style={{ color: 'var(--accent-purple)', fontSize: '0.8rem', letterSpacing: '0.05em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            <i className="fa-solid fa-lock" style={{ marginRight: '0.4rem' }}></i> Feature Locked
          </p>
        </div>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px',
        padding: '3rem 2rem',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-glass)',
        backdropFilter: 'blur(20px)',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-md)'
      }}>
        {/* Subtle decorative gradient glow behind */}
        <div style={{
          position: 'absolute',
          width: '280px',
          height: '280px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(201, 90, 73, 0.1) 0%, rgba(255, 255, 255, 0) 70%)',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          pointerEvents: 'none',
          zIndex: 0
        }}></div>

        <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{
            width: '70px',
            height: '70px',
            borderRadius: '50%',
            backgroundColor: 'rgba(201, 90, 73, 0.06)',
            border: '1px solid rgba(201, 90, 73, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.75rem',
            color: 'var(--accent-purple)',
            marginBottom: '1.5rem',
            boxShadow: '0 0 20px rgba(201, 90, 73, 0.05)'
          }}>
            <i className="fa-solid fa-brain"></i>
          </div>

          <span style={{
            fontSize: '0.7rem',
            fontWeight: 700,
            color: 'var(--accent-purple)',
            textTransform: 'uppercase',
            letterSpacing: '0.15em',
            backgroundColor: 'rgba(201, 90, 73, 0.08)',
            padding: '0.35rem 0.85rem',
            borderRadius: '9999px',
            border: '1px solid rgba(201, 90, 73, 0.15)',
            fontFamily: 'var(--font-mono)',
            marginBottom: '1rem'
          }}>
            Coming Soon
          </span>

          <h2 style={{ fontSize: '1.75rem', fontWeight: 600, fontFamily: 'var(--font-heading)', color: 'var(--text-main)', marginBottom: '0.75rem', letterSpacing: '-0.02em' }}>
            Advanced Text & Natural Language Pipelines
          </h2>
          
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', maxWidth: '460px', lineHeight: '1.6', marginBottom: '2rem', fontFamily: 'var(--font-body)' }}>
            We are building zero-shot sentiment classifiers, topic keyword extractors, and custom transformer embedding pipelines. This page will unlock soon as part of the public beta.
          </p>

          <div style={{ display: 'flex', gap: '1rem' }}>
            <a href="/#contact" className="btn btn-primary" style={{ padding: '0.6rem 1.5rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', textDecoration: 'none', display: 'flex', alignItems: 'center' }}>
              <span>Notify Me</span>
            </a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="btn btn-secondary" style={{ padding: '0.6rem 1.5rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none' }}>
              <i className="fa-brands fa-github"></i>
              <span>Star on GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NLPPlayground;
