import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import { toggleSidebar, closeMobileSidebar } from '../components/sidebarHelper';

const Guidelines = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Initial sidebar load check
    const isHidden = localStorage.getItem('sidebar-hidden') === 'true';
    if (isHidden) {
      document.documentElement.classList.add('sidebar-hidden');
    } else {
      document.documentElement.classList.remove('sidebar-hidden');
    }
  }, []);

  const steps = [
    {
      step: "01",
      title: "Choose the Correct Modality",
      description: "InstaML supports four core data formats: Tabular (spreadsheets, metrics), Text (NLP classification, sentiment), Image (computer vision, YOLO detection), and Audio (classification, voice signals). Create a workspace matching your target raw data.",
      icon: "fa-solid fa-shapes",
      color: "var(--accent-primary)"
    },
    {
      step: "02",
      title: "Data Preparation & Upload",
      description: "Upload clean CSV or Excel files for tabular/text workloads. For Computer Vision (Image) or Audio modality classifiers, zip your dataset folders structured by class labels (e.g. cat/ dog/ folders) and upload the ZIP archive.",
      icon: "fa-solid fa-cloud-arrow-up",
      color: "var(--accent-purple)"
    },
    {
      step: "03",
      title: "Interactive Preprocessing",
      description: "Drop columns that are uniquely identifying (IDs, index, names) to avoid overfitting. Configure imputation for missing values (mean, median, or most-frequent) and select numeric feature scaling algorithms.",
      icon: "fa-solid fa-sliders",
      color: "var(--accent-green)"
    },
    {
      step: "04",
      title: "EDA Visualization Checks",
      description: "Before kicking off training pipelines, check the EDA tab. Analyze feature distribution plots, correlation heatmaps, class balances, and outlier counts to catch dataset issues early.",
      icon: "fa-solid fa-chart-bar",
      color: "var(--accent-yellow)"
    },
    {
      step: "05",
      title: "Configure Training & Optuna Tuning",
      description: "Select validation ratios (e.g. 80/20 train-test splits) and configure machine learning models (XGBoost, Random Forest, ResNet, Text CNNs). Scale up background hyperparameter tuning with Optuna search trials.",
      icon: "fa-solid fa-gears",
      color: "var(--accent-primary)"
    },
    {
      step: "06",
      title: "Testing & REST API Deployments",
      description: "Evaluate your models on the Test page (accuracy, confusion matrix, loss logs). Deploy a model dynamically as a serverless REST endpoint, then copy the auto-generated integration snippets (curl, javascript, python) to embed inside your application code.",
      icon: "fa-solid fa-rocket",
      color: "var(--accent-purple)"
    }
  ];

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
        {/* Page Header */}
        <div className="header-bar" style={{ marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', width: '100%' }}>
            <div className="page-title-section">
              <h1 className="page-title" style={{ fontSize: '2.25rem', fontWeight: 500, marginBottom: '0.15rem', lineHeight: '1.2' }}>Developer Guidelines</h1>
              <p className="page-subtitle" style={{ fontSize: '0.95rem', color: 'var(--text-muted)', fontFamily: 'var(--font-body)', margin: 0 }}>Follow this developer manual to configure pipelines, run hyperparameter searches, and deploy models.</p>
            </div>
          </div>
        </div>

        {/* Timeline Grid layout */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem', marginBottom: '3rem' }}>
          {steps.map((item, idx) => (
            <div 
              key={idx} 
              className="card" 
              style={{ 
                padding: '2rem', 
                border: '1px solid var(--border-color)', 
                backgroundColor: 'var(--bg-secondary)', 
                borderRadius: 'var(--radius-md)',
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'space-between',
                transition: 'all 0.2s ease',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div>
                {/* Step badge */}
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  marginBottom: '1.5rem' 
                }}>
                  <div style={{ 
                    padding: '0.55rem', 
                    borderRadius: '12px', 
                    backgroundColor: 'var(--bg-tertiary)', 
                    color: item.color, 
                    display: 'flex', 
                    border: '1px solid var(--border-color)' 
                  }}>
                    <i className={item.icon} style={{ fontSize: '1.1rem' }}></i>
                  </div>
                  <span style={{ 
                    fontSize: '2.5rem', 
                    fontWeight: 800, 
                    color: 'var(--border-color)', 
                    fontFamily: 'var(--font-heading)',
                    lineHeight: '1'
                  }}>
                    {item.step}
                  </span>
                </div>

                <h3 style={{ 
                  fontSize: '1.2rem', 
                  fontWeight: 600, 
                  fontFamily: 'var(--font-heading)', 
                  color: 'var(--text-main)', 
                  marginBottom: '0.75rem' 
                }}>
                  {item.title}
                </h3>
                
                <p style={{ 
                  fontSize: '0.85rem', 
                  color: 'var(--text-muted)', 
                  lineHeight: '1.6', 
                  fontFamily: 'var(--font-body)',
                  margin: 0 
                }}>
                  {item.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Quick Tips footer card */}
        <div className="card" style={{ padding: '2rem', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div style={{ 
            width: '54px', 
            height: '54px', 
            borderRadius: '50%', 
            backgroundColor: 'var(--bg-tertiary)', 
            color: 'var(--accent-purple)', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            fontSize: '1.5rem', 
            border: '1px solid var(--border-color)',
            flexShrink: 0
          }}>
            <i className="fa-solid fa-lightbulb"></i>
          </div>
          <div>
            <h4 style={{ fontSize: '1rem', fontWeight: 600, fontFamily: 'var(--font-heading)', color: 'var(--text-main)', marginBottom: '0.25rem' }}>Developer Tip: Keep it Clean</h4>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', fontFamily: 'var(--font-body)', margin: 0, lineHeight: '1.5' }}>
              Ensure your columns and datasets are normalized. High variance, unscaled dimensions, and class imbalances are the most common source of suboptimal validation metrics. Use Optuna searches with 20+ trials for tabular datasets to explore learning rate schedules.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Guidelines;
