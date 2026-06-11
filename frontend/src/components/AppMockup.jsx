import React from 'react';

const AppMockup = () => {
  return (
    <div className="app-mockup-window">
      {/* Window Chrome Header */}
      <div 
        className="mockup-chrome"
        style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.75rem', 
          padding: '0.75rem 1.25rem', 
          borderBottom: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-tertiary)'
        }}
      >
        {/* Mac Traffic Light dots */}
        <div style={{ display: 'flex', gap: '0.35rem' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#ff5f56', display: 'inline-block' }}></span>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#ffbd2e', display: 'inline-block' }}></span>
          <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: '#27c93f', display: 'inline-block' }}></span>
        </div>
        {/* Address Bar */}
        <div 
          style={{ 
            flex: 1, 
            backgroundColor: 'var(--bg-secondary)', 
            borderRadius: '6px', 
            fontSize: '0.75rem', 
            color: 'var(--text-dim)', 
            textAlign: 'center', 
            padding: '0.25rem 0', 
            maxWidth: '320px', 
            margin: '0 auto',
            border: '1px solid var(--border-color)',
            fontFamily: 'var(--font-body)'
          }}
        >
          instaml.ai/workspace/flight-delay-classifier
        </div>
      </div>
      
      {/* Dashboard Mockup Content Area */}
      <div className="mockup-content-area" style={{ minHeight: '280px' }}>
        {/* Mockup Sidebar */}
        <div className="mockup-sidebar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', fontWeight: 700, fontFamily: 'var(--font-heading)' }}>
            <i className="fa-solid fa-cube" style={{ color: 'var(--accent-primary)', fontSize: '0.85rem' }}></i>
            <span>instaml</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
            <span style={{ fontSize: '0.55rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', paddingLeft: '0.35rem', marginBottom: '0.15rem' }}>ML Steps</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.68rem', padding: '0.25rem 0.35rem', borderRadius: '4px', color: 'var(--text-muted)' }}>
              <i className="fa-solid fa-cloud-arrow-up" style={{ width: '10px' }}></i>
              <span>1. Upload</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.68rem', padding: '0.25rem 0.35rem', borderRadius: '4px', color: 'var(--text-muted)' }}>
              <i className="fa-solid fa-sliders" style={{ width: '10px' }}></i>
              <span>2. Preprocess</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.68rem', padding: '0.25rem 0.35rem', borderRadius: '4px', color: 'var(--text-muted)' }}>
              <i className="fa-solid fa-chart-bar" style={{ width: '10px' }}></i>
              <span>3. EDA Plots</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.68rem', padding: '0.25rem 0.35rem', borderRadius: '4px', color: 'var(--text-main)', backgroundColor: 'var(--bg-active)', borderLeft: '2px solid var(--accent-primary)', borderTopLeftRadius: 0, borderBottomLeftRadius: 0, fontWeight: 600 }}>
              <i className="fa-solid fa-gears" style={{ width: '10px', color: 'var(--accent-primary)' }}></i>
              <span>4. Training</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.68rem', padding: '0.25rem 0.35rem', borderRadius: '4px', color: 'var(--text-muted)' }}>
              <i className="fa-solid fa-rocket" style={{ width: '10px' }}></i>
              <span>5. Deployment</span>
            </div>
          </div>
        </div>

        {/* Mockup Main Canvas */}
        <div className="mockup-main-canvas">
          {/* Main Top Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1.25rem', borderBottom: '1px solid var(--border-color)', backgroundColor: 'rgba(0,0,0,0.03)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-main)' }}>Flight Delay Classifier</span>
              <span style={{ fontSize: '0.6rem', padding: '0.1rem 0.4rem', borderRadius: '10px', border: '1px solid var(--border-color)', color: 'var(--text-muted)', backgroundColor: 'var(--bg-secondary)' }}>SQLite Transaction v1.4</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--accent-green)', display: 'inline-block' }}></span>
              <span style={{ fontSize: '0.65rem', color: 'var(--accent-green)', fontWeight: 600 }}>REST API Active (Port 8000)</span>
            </div>
          </div>

          {/* Main Panel Content Area */}
          <div className="mockup-grid">
            {/* Column 1: Preprocess & Tuning logs */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Preprocessed Checklist Card */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>1. Clean Features List</span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', padding: '0.3rem 0.55rem', border: '1px solid var(--border-color)', borderRadius: '6px', backgroundColor: 'var(--bg-secondary)', fontSize: '0.68rem' }}>
                    <i className="fa-solid fa-square-check" style={{ color: 'var(--accent-green)' }}></i>
                    <span style={{ fontWeight: 600 }}>passenger_age</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', padding: '0.3rem 0.55rem', border: '1px solid var(--border-color)', borderRadius: '6px', backgroundColor: 'var(--bg-secondary)', fontSize: '0.68rem' }}>
                    <i className="fa-solid fa-square-check" style={{ color: 'var(--accent-green)' }}></i>
                    <span style={{ fontWeight: 600 }}>flight_distance</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', padding: '0.3rem 0.55rem', border: '1px solid var(--border-color)', borderRadius: '6px', backgroundColor: 'var(--bg-secondary)', fontSize: '0.68rem' }}>
                    <i className="fa-solid fa-square-check" style={{ color: 'var(--accent-green)' }}></i>
                    <span style={{ fontWeight: 600 }}>departure_delay</span>
                  </div>
                </div>
              </div>

              {/* Hyperparameter Tuning Widget */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>2. Optuna Tuner</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    <i className="fa-solid fa-gear fa-spin" style={{ color: 'var(--accent-primary)', fontSize: '0.7rem' }}></i>
                    <span>Trial 24/50 running</span>
                  </div>
                </div>

                <div 
                  className="mockup-progress-wrapper"
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    height: '6px',
                    width: '100%',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  <div 
                    style={{
                      height: '100%',
                      backgroundColor: 'var(--accent-primary)',
                      width: '48%',
                      borderRadius: '4px'
                    }}
                  ></div>
                </div>

                {/* Console Logs */}
                <div 
                  style={{ 
                    backgroundColor: 'var(--bg-tertiary)', 
                    border: '1px solid var(--border-color)', 
                    borderRadius: '8px', 
                    padding: '0.6rem 0.75rem', 
                    fontFamily: 'monospace', 
                    fontSize: '0.65rem', 
                    color: 'var(--text-muted)', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    gap: '0.2rem', 
                    height: '84px', 
                    overflow: 'hidden' 
                  }}
                >
                  <div style={{ color: 'var(--accent-green)' }}>[Trial 12] Validation Accuracy: 96.42% (Best)</div>
                  <div>[Trial 13] Params: max_depth=12, learning_rate=0.01</div>
                  <div>[Trial 18] Algorithm chosen: XGBoostClassifier</div>
                  <div style={{ color: 'var(--text-dim)' }}>[Active] Training checkpoints updated.</div>
                </div>
              </div>
            </div>

            {/* Column 2: REST Payload & Inference Output */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '1rem', borderRadius: '10px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--accent-purple)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>3. Serverless Inference API</span>
                <div style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '0.5rem 0.65rem', fontFamily: 'monospace', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                  <div style={{ color: 'var(--accent-primary)', fontWeight: 700, marginBottom: '0.15rem' }}>POST /api/predict</div>
                  <div style={{ color: 'var(--text-dim)' }}>{"{"}</div>
                  <div style={{ paddingLeft: '0.5rem' }}>"passenger_age": 34,</div>
                  <div style={{ paddingLeft: '0.5rem' }}>"flight_distance": 840,</div>
                  <div style={{ paddingLeft: '0.5rem' }}>"departure_delay": 12</div>
                  <div style={{ color: 'var(--text-dim)' }}>{"}"}</div>
                </div>
              </div>

              {/* Output Confidence Box */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Response output</span>
                <div 
                  style={{ 
                    backgroundColor: 'var(--bg-primary)', 
                    border: '1px solid var(--border-color)', 
                    borderRadius: '8px', 
                    padding: '0.65rem 0.85rem', 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center' 
                  }}
                >
                  <div>
                    <span style={{ fontSize: '0.55rem', textTransform: 'uppercase', color: 'var(--text-dim)', display: 'block' }}>Prediction</span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-main)', marginTop: '0.1rem', display: 'block' }}>Delayed (Class 1)</span>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '0.55rem', textTransform: 'uppercase', color: 'var(--text-dim)', display: 'block' }}>Latency / Conf</span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-green)', marginTop: '0.1rem', display: 'block' }}>34ms / 98.42%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AppMockup;
