import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useParams, Outlet, useOutletContext, useLocation, useNavigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AuthProvider, useAuth } from './context/AuthContext';
import api from './services/api';

// Components & Layout
import Sidebar from './components/Sidebar';
import InteractiveBackground from './components/InteractiveBackground';
import { toggleSidebar, closeMobileSidebar } from './components/sidebarHelper';

// Pages
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Settings from './pages/Settings';
import Guidelines from './pages/Guidelines';
import DataUpload from './pages/DataUpload';
import Preprocessing from './pages/Preprocessing';
import EDA from './pages/EDA';
import TrainModel from './pages/TrainModel';
import TestModel from './pages/TestModel';
import Deployment from './pages/Deployment';
import Versions from './pages/Versions';
import NLPPlayground from './pages/NLPPlayground';

// Protected Route Guard
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--bg-primary)' }}>
        <div className="spinner"></div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

// Step-by-step Pipeline Navigation Wizard (Back / Next)
const PipelineNavigation = ({ datasetStatus, models }) => {
  const { project_id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const steps = ['upload', 'preprocess', 'eda', 'train', 'test', 'deploy'];
  const stepTitles = [
    'Data Upload',
    'Preprocessing',
    'EDA Analysis',
    'Train Model',
    'Test Model',
    'Model Deploy'
  ];

  const pathParts = location.pathname.split('/');
  const currentStep = pathParts[pathParts.length - 1];
  const currentIndex = steps.indexOf(currentStep);

  if (currentIndex === -1) return null;

  const handleBack = () => {
    if (currentIndex > 0) {
      navigate(`/projects/${project_id}/${steps[currentIndex - 1]}`);
    }
  };

  const handleNext = () => {
    if (currentIndex < steps.length - 1) {
      navigate(`/projects/${project_id}/${steps[currentIndex + 1]}`);
    }
  };

  let isNextDisabled = false;
  let nextTooltip = '';

  if (currentStep === 'upload' && !datasetStatus?.data_loaded) {
    isNextDisabled = true;
    nextTooltip = 'Please upload a dataset first to proceed.';
  } else if (currentStep === 'train' && (!models || models.length === 0)) {
    isNextDisabled = true;
    nextTooltip = 'Please train at least one model first to proceed.';
  }

  const showBack = currentIndex > 0;
  const showNext = currentIndex < steps.length - 1;

  return (
    <div style={{ marginTop: 'auto', paddingTop: '3rem' }}>
      <div style={{ borderTop: '1px solid var(--border-color)', margin: '1.5rem 0' }}></div>
      <div className="pipeline-nav-container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
        {showBack ? (
          <button 
            onClick={handleBack} 
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem' }}
          >
            <i className="fa-solid fa-arrow-left"></i>
            <span>Back: {stepTitles[currentIndex - 1]}</span>
          </button>
        ) : (
          <div />
        )}

        {showNext && (
          <div title={nextTooltip}>
            <button 
              onClick={handleNext} 
              className="btn btn-primary"
              disabled={isNextDisabled}
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem', 
                padding: '0.65rem 1.25rem',
                opacity: isNextDisabled ? 0.5 : 1,
                cursor: isNextDisabled ? 'not-allowed' : 'pointer'
              }}
            >
              <span>Next: {stepTitles[currentIndex + 1]}</span>
              <i className="fa-solid fa-arrow-right"></i>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

// Layout wrapping the 6 Pipeline Pages
const PipelineLayout = () => {
  const { project_id } = useParams();
  const [project, setProject] = useState(null);
  const [datasetStatus, setDatasetStatus] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const fetchProjectMetadata = async () => {
    try {
      setError('');
      // 1. Fetch project info
      const projRes = await api.get(`/projects/${project_id}`);
      setProject(projRes.data);

      // 2. Fetch current dataset status
      const dataRes = await api.get(`/projects/${project_id}/data/current`);
      setDatasetStatus({ ...dataRes.data, project_id });

      // 3. Fetch trained models list
      const modelsRes = await api.get(`/projects/${project_id}/models`);
      setModels(modelsRes.data);
    } catch (err) {
      console.error("Failed to load project pipeline metadata", err);
      setError(err.response?.data?.detail || "Project not found or access denied.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjectMetadata();
  }, [project_id]);

  useEffect(() => {
    // Initial sidebar load check
    const isHidden = localStorage.getItem('sidebar-hidden') === 'true';
    if (isHidden) {
      document.documentElement.classList.add('sidebar-hidden');
    } else {
      document.documentElement.classList.remove('sidebar-hidden');
    }
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--bg-primary)' }}>
        <div className="spinner"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: 'var(--bg-primary)', color: 'var(--text-main)', gap: '1rem', padding: '2rem', textAlign: 'center' }}>
        <i className="fa-solid fa-circle-exclamation" style={{ fontSize: '3rem', color: 'var(--accent-red)' }}></i>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Access Denied / Not Found</h2>
        <p style={{ color: 'var(--text-muted)', maxWidth: '400px', marginBottom: '1.5rem' }}>{error}</p>
        <button onClick={() => window.location.href = '/'} className="btn btn-primary" style={{ padding: '0.6rem 1.5rem', borderRadius: 'var(--radius-sm)' }}>
          Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar
        project={project}
        datasetStatus={datasetStatus}
        hasModels={models.length > 0}
      />
      
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

      <div className="main-content">
        <div className="hero-grid-background"></div>
        {/* Workspace global breadcrumbs/header */}
        <div className="workspace-global-header" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2.5rem', paddingBottom: '1rem', borderBottom: '1px solid var(--border-color)' }}>
          <span style={{ fontSize: '0.825rem', color: 'var(--text-muted)', fontWeight: 500, fontFamily: 'var(--font-body)' }}>
            Workspace / <strong style={{ color: 'var(--text-main)', fontWeight: 600 }}>{project?.name}</strong>
          </span>
        </div>

        {/* Render active sub-route page passing layout context */}
        <Outlet context={{ project, datasetStatus, models, refreshStatus: fetchProjectMetadata }} />
        
        {/* Step-by-step pipeline page-level navigation */}
        <PipelineNavigation datasetStatus={datasetStatus} models={models} />
      </div>
    </div>
  );
};

// Route adapters to extract context hooks inside layout
const DataUploadAdapter = () => {
  const { project, datasetStatus, refreshStatus } = useOutletContext();
  return <DataUpload project={project} datasetStatus={datasetStatus} refreshStatus={refreshStatus} />;
};

const PreprocessingAdapter = () => {
  const { datasetStatus, refreshStatus } = useOutletContext();
  return <Preprocessing datasetStatus={datasetStatus} refreshStatus={refreshStatus} />;
};

const EDAAdapter = () => {
  const { datasetStatus } = useOutletContext();
  return <EDA datasetStatus={datasetStatus} />;
};

const TrainModelAdapter = () => {
  const { project, datasetStatus, refreshStatus } = useOutletContext();
  return <TrainModel project={project} datasetStatus={datasetStatus} refreshModels={refreshStatus} />;
};

const TestModelAdapter = () => {
  const { project, models, refreshStatus, datasetStatus } = useOutletContext();
  return <TestModel project={project} models={models} refreshModels={refreshStatus} datasetStatus={datasetStatus} />;
};

const DeploymentAdapter = () => {
  const { project, datasetStatus, models } = useOutletContext();
  return <Deployment project={project} datasetStatus={datasetStatus} models={models} />;
};

const VersionsAdapter = () => {
  const { datasetStatus, refreshStatus } = useOutletContext();
  return <Versions datasetStatus={datasetStatus} refreshStatus={refreshStatus} />;
};

const NLPPlaygroundAdapter = () => {
  const { project } = useOutletContext();
  return <NLPPlayground project={project} />;
};

const App = () => {
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  return (
    <AuthProvider>
      <Router>
        <InteractiveBackground />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'var(--bg-secondary)',
              color: 'var(--text-main)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-sm)',
              fontFamily: 'var(--font-body)',
              fontSize: '0.85rem'
            }
          }}
        />
        <Routes>
          {/* Public Landing & Auth Routes */}
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Protected Main Dashboard */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />

          {/* Protected Settings Page */}
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <Settings />
              </ProtectedRoute>
            }
          />

          {/* Protected Guidelines Page */}
          <Route
            path="/guidelines"
            element={
              <ProtectedRoute>
                <Guidelines />
              </ProtectedRoute>
            }
          />

          {/* Protected Pipeline Stages */}
          <Route
            path="/projects/:project_id"
            element={
              <ProtectedRoute>
                <PipelineLayout />
              </ProtectedRoute>
            }
          >
            <Route path="upload" element={<DataUploadAdapter />} />
            <Route path="preprocess" element={<PreprocessingAdapter />} />
            <Route path="eda" element={<EDAAdapter />} />
            <Route path="train" element={<TrainModelAdapter />} />
            <Route path="test" element={<TestModelAdapter />} />
            <Route path="deploy" element={<DeploymentAdapter />} />
            <Route path="versions" element={<VersionsAdapter />} />
            <Route path="playground" element={<NLPPlaygroundAdapter />} />
            
            {/* Catch-all project route redirects to upload */}
            <Route index element={<Navigate to="upload" replace />} />
          </Route>

          {/* Fallback Catch-All Redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
};

export default App;
