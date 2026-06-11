import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { useAuth } from '../context/AuthContext';
import Sidebar from '../components/Sidebar';
import { toggleSidebar, closeMobileSidebar } from '../components/sidebarHelper';

const getTasksForDataType = (type) => {
  switch(type) {
    case 'text':
      return [
        'Text Classification',
        'Sentiment Analysis',
        'Intent Classification',
        'Spam Detection',
        'Toxicity Detection',
        'Named Entity Recognition (NER)',
        'Text Summarization'
      ];
    case 'image':
      return [
        'Image Classification',
        'Object Detection',
        'Face Detection',
        'OCR (Optical Character Recognition)',
        'Image Denoising',
        'Super Resolution'
      ];
    case 'audio':
      return [
        'Speech Recognition (ASR)',
        'Audio Classification',
        'Sound Event Detection',
        'Wake Word Detection',
        'Noise Reduction'
      ];
    case 'tabular':
      return ['Tabular Classification', 'Tabular Regression'];
    default:
      return ['Classification', 'Regression'];
  }
};

const Dashboard = () => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');

  // Theme state for Dashboard
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // New Project Form States
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [dataType, setDataType] = useState('tabular');
  const [task, setTask] = useState('Tabular Classification');
  const [formError, setFormError] = useState('');
  const [creating, setCreating] = useState(false);

  const handleDataTypeChange = (type) => {
    setDataType(type);
    const tasks = getTasksForDataType(type);
    setTask(tasks[0] || 'Classification');
  };

  const { logout, user } = useAuth();
  const navigate = useNavigate();

  const fetchProjects = async () => {
    try {
      const res = await api.get('/projects');
      setProjects(res.data);
    } catch (err) {
      console.error("Failed to load projects", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();

    // Initial sidebar load check
    const isHidden = localStorage.getItem('sidebar-hidden') === 'true';
    if (isHidden) {
      document.documentElement.classList.add('sidebar-hidden');
    } else {
      document.documentElement.classList.remove('sidebar-hidden');
    }
  }, []);

  const handleCreateProject = async (e) => {
    e.preventDefault();
    setFormError('');
    setCreating(true);

    try {
      const res = await api.post('/projects', {
        name,
        description,
        data_type: dataType,
        task: task
      });
      setModalOpen(false);
      setName('');
      setDescription('');
      setDataType('tabular');
      setTask('Tabular Classification');
      // Redirect directly to the project's upload step
      navigate(`/projects/${res.data.id}/upload`);
    } catch (err) {
      setFormError(err.response?.data?.detail || "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteProject = async (id, e) => {
    e.stopPropagation(); // Avoid navigating to the project
    if (!window.confirm("Are you sure you want to delete this project and all its uploaded datasets and saved models? This action cannot be undone.")) {
      return;
    }
    try {
      await api.delete(`/projects/${id}`);
      setProjects(projects.filter(p => p.id !== id));
    } catch (err) {
      alert("Failed to delete project");
    }
  };

  const getDataTypeIcon = (type) => {
    switch(type) {
      case 'tabular':
        return 'fa-solid fa-table';
      case 'text':
        return 'fa-solid fa-file-lines';
      case 'image':
        return 'fa-solid fa-image';
      case 'audio':
        return 'fa-solid fa-music';
      default:
        return 'fa-solid fa-chart-line';
    }
  };

  const filteredProjects = projects.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (p.description && p.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

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
        {/* Header bar */}
        <div className="header-bar" style={{ marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div className="page-title-section">
              <h1 className="page-title" style={{ fontSize: '2.25rem', fontWeight: 500, marginBottom: '0.15rem', lineHeight: '1.2' }}>Hey {user?.username || 'developer'},</h1>
              <p className="page-subtitle" style={{ fontSize: '0.95rem', color: 'var(--text-muted)', fontFamily: 'var(--font-body)', margin: 0 }}>Let's build and deploy some amazing machine learning models today.</p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center' }}>
            <button onClick={() => setModalOpen(true)} className="btn btn-primary">
              <i className="fa-solid fa-plus"></i>
              <span>New Workspace</span>
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '40vh' }}>
            <div className="spinner"></div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h2 style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
                Your Workspaces ({filteredProjects.length})
              </h2>
              {/* Search filter bar */}
              <div className="input-group-custom" style={{ maxWidth: '240px' }}>
                <i className="fa-solid fa-magnifying-glass"></i>
                <input
                  type="text"
                  className="form-control"
                  style={{ padding: '0.45rem 0.75rem 0.45rem 2.25rem', fontSize: '0.8rem' }}
                  placeholder="Filter workspaces..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {filteredProjects.length === 0 ? (
              <div className="card" style={{ padding: '4rem 2.5rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.25rem' }}>
                <div style={{ width: '42px', height: '42px', borderRadius: 'var(--radius-sm)', backgroundColor: 'var(--bg-tertiary)', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-color)' }}>
                  <i className="fa-solid fa-folder-open" style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}></i>
                </div>
                <h2 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
                  {searchQuery ? "No matching workspaces" : "No workspaces yet"}
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.825rem', lineHeight: '1.6', maxWidth: '380px', margin: '0 auto' }}>
                  {searchQuery ? "Try refining your search filter query." : "Get started by creating a new workspace project. Choose from tabular datasets, image vision models, text sentiment, or audio processing."}
                </p>
                {!searchQuery && (
                  <button onClick={() => setModalOpen(true)} className="btn btn-primary" style={{ marginTop: '0.5rem' }}>
                    <i className="fa-solid fa-plus"></i>
                    <span>Create Workspace</span>
                  </button>
                )}
              </div>
            ) : (
              <div className="workspaces-grid">
                {filteredProjects.map((project) => (
                  <div
                    key={project.id}
                    className="card"
                    style={{ 
                      cursor: 'pointer', 
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'space-between',
                      padding: '1.5rem', 
                      minHeight: '160px',
                      border: '1px solid var(--border-color)',
                      backgroundColor: 'var(--bg-secondary)',
                      borderRadius: 'var(--radius-md)',
                      transition: 'all 0.15s ease'
                    }}
                    onClick={() => navigate(`/projects/${project.id}/upload`)}
                  >
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem', minWidth: 0 }}>
                          <div style={{ 
                            padding: '0.55rem', 
                            borderRadius: '12px', 
                            backgroundColor: 'var(--bg-tertiary)', 
                            color: 'var(--accent-primary)', 
                            display: 'flex', 
                            border: '1px solid var(--border-color)', 
                            flexShrink: 0 
                          }}>
                            <i className={getDataTypeIcon(project.data_type)} style={{ fontSize: '1rem' }}></i>
                          </div>
                          <h3 style={{ 
                            fontSize: '1.05rem', 
                            fontWeight: 600, 
                            fontFamily: 'var(--font-heading)',
                            margin: 0, 
                            color: 'var(--text-main)',
                            textOverflow: 'ellipsis', 
                            overflow: 'hidden', 
                            whiteSpace: 'nowrap' 
                          }}>
                            {project.name}
                          </h3>
                        </div>
                        <button
                          onClick={(e) => handleDeleteProject(project.id, e)}
                          style={{ 
                            background: 'none', 
                            border: 'none', 
                            color: 'var(--text-dim)', 
                            cursor: 'pointer', 
                            transition: 'color 0.15s ease', 
                            padding: '0.35rem',
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: 'var(--bg-tertiary)'
                          }}
                          onMouseEnter={(e) => {
                            e.stopPropagation();
                            e.currentTarget.style.color = 'var(--accent-red)';
                          }}
                          onMouseLeave={(e) => {
                            e.stopPropagation();
                            e.currentTarget.style.color = 'var(--text-dim)';
                          }}
                        >
                          <i className="fa-solid fa-trash-can" style={{ fontSize: '0.8rem' }}></i>
                        </button>
                      </div>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.825rem', lineHeight: '1.5', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', fontFamily: 'var(--font-body)' }}>
                        {project.description || "No description provided."}
                      </p>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem', marginTop: '0.75rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <span className="badge badge-info" style={{ textTransform: 'capitalize', fontSize: '0.65rem', padding: '0.2rem 0.6rem', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>
                          {project.data_type}
                        </span>
                        {project.task && (
                          <span className="badge" style={{ fontSize: '0.65rem', padding: '0.2rem 0.6rem', border: '1px solid rgba(139, 92, 246, 0.2)', backgroundColor: 'rgba(139, 92, 246, 0.1)', color: 'var(--accent-purple)' }}>
                            {project.task}
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: '0.725rem', color: 'var(--text-dim)', fontFamily: 'var(--font-body)' }}>
                        {new Date(project.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      {modalOpen && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '-0.02em' }}>New Workspace</h3>
              <button onClick={() => setModalOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '1.1rem', cursor: 'pointer' }}>
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>

            {formError && (
              <div className="alert alert-danger" style={{ padding: '0.65rem 0.85rem', fontSize: '0.8rem', marginBottom: '1.25rem' }}>
                <i className="fa-solid fa-circle-exclamation"></i>
                <span>{formError}</span>
              </div>
            )}

            <form onSubmit={handleCreateProject}>
              <div className="form-group">
                <label className="form-label">Project Name</label>
                <input
                  type="text"
                  className="form-control"
                  placeholder="e.g. Flight Delay Predictor"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">Description</label>
                <textarea
                  className="form-control"
                  style={{ minHeight: '80px', resize: 'vertical' }}
                  placeholder="Describe the goals of this workspace..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Data Format</label>
                <select
                  className="form-control"
                  value={dataType}
                  onChange={(e) => handleDataTypeChange(e.target.value)}
                  style={{ cursor: 'pointer' }}
                >
                  <option value="tabular">Tabular (CSV, Excel, Parquet)</option>
                  <option value="text">Raw Text (NLP classification)</option>
                  <option value="image">Image Dataset (Image classification, Detection)</option>
                  <option value="audio">Audio Signals (Sound classification)</option>
                  <option value="multi_dimensional">Multi-Dimensional Arrays</option>
                </select>
              </div>

              {dataType !== 'multi_dimensional' && (
                <div className="form-group" style={{ marginBottom: '2rem' }}>
                  <label className="form-label">Specific Task</label>
                  <select
                    className="form-control"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    {getTasksForDataType(dataType).map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
                <button type="button" onClick={() => setModalOpen(false)} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={creating}>
                  {creating ? <span className="spinner"></span> : <span>Create Workspace</span>}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;

