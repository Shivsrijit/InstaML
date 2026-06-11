import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';
import GuideDrawer from '../components/GuideDrawer';

const Deployment = ({ project, datasetStatus, models }) => {
  const [activeTab, setActiveTab] = useState('realtime');
  const [deployedModel, setDeployedModel] = useState(null);
  const [guideOpen, setGuideOpen] = useState(false);
  
  // Realtime Pred States
  const [features, setFeatures] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);

  // Batch Pred States
  const [batchFile, setBatchFile] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Specialized Task States
  const [pastedText, setPastedText] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioPreview, setAudioPreview] = useState(null);

  const dataType = project?.data_type || 'tabular';

  useEffect(() => {
    // Find deployed model from list
    const deployed = models.find(m => m.is_deployed);
    setDeployedModel(deployed || null);
    
    // Initialize default inputs
    const formColumns = datasetStatus?.initial_columns || datasetStatus?.columns;
    const formDtypes = datasetStatus?.initial_dtypes || datasetStatus?.dtypes;
    
    if (formColumns && deployed) {
      const initVals = {};
      const target = deployed.target_col;
      
      formColumns.forEach(col => {
        if (col !== target) {
          // Default numeric columns to 0, categoricals to empty string
          const dtype = formDtypes?.[col];
          if (dtype?.includes('int') || dtype?.includes('float')) {
            initVals[col] = 0.0;
          } else {
            initVals[col] = '';
          }
        }
      });
      setFeatures(initVals);
    }
  }, [models, datasetStatus]);

  const handleFeatureChange = (col, value, isNum) => {
    setFeatures({
      ...features,
      [col]: isNum ? parseFloat(value) || 0.0 : value
    });
  };

  const handlePredictRealtime = async (e) => {
    e.preventDefault();
    setPredLoading(true);
    setPredictionResult(null);

    try {
      const res = await api.post(`/projects/${datasetStatus.project_id}/deploy/predict`, {
        features
      });
      setPredictionResult(res.data);
      toast.success("Prediction completed!");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Inference call failed. Validate input types.");
    } finally {
      setPredLoading(false);
    }
  };

  const handlePredictText = async (e) => {
    e.preventDefault();
    if (!pastedText.trim()) return;
    setPredLoading(true);
    setPredictionResult(null);

    try {
      const res = await api.post(`/projects/${datasetStatus.project_id}/deploy/predict`, {
        features: { text: pastedText }
      });
      setPredictionResult(res.data);
      toast.success("NLP analysis complete!");
    } catch (err) {
      toast.error(err.response?.data?.detail || "Text inference failed.");
    } finally {
      setPredLoading(false);
    }
  };

  const handlePredictFile = async (e) => {
    e.preventDefault();
    const file = dataType === 'image' ? imageFile : audioFile;
    if (!file) return;
    setPredLoading(true);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await api.post(`/projects/${datasetStatus.project_id}/deploy/predict-file`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setPredictionResult(res.data);
      toast.success("File inference completed!");
    } catch (err) {
      toast.error(err.response?.data?.detail || "File processing failed.");
    } finally {
      setPredLoading(false);
    }
  };

  const handleClearPlayground = () => {
    setPastedText('');
    setImageFile(null);
    setImagePreview(null);
    setAudioFile(null);
    setAudioPreview(null);
    setPredictionResult(null);
  };

  const renderHighlightedEntities = (text, entities) => {
    if (!entities || entities.length === 0) return <div style={{ color: 'var(--text-muted)' }}>No entities detected.</div>;
    
    const elements = [];
    let lastIdx = 0;
    
    entities.forEach((ent, i) => {
      if (ent.start > lastIdx) {
        elements.push(<span key={`t-${i}`}>{text.substring(lastIdx, ent.start)}</span>);
      }
      
      let badgeColor = 'var(--accent-primary)';
      if (ent.label === 'PERSON') badgeColor = 'var(--accent-green)';
      if (ent.label === 'ORGANIZATION') badgeColor = 'var(--accent-purple)';
      if (ent.label === 'LOCATION') badgeColor = 'var(--accent-yellow)';
      if (ent.label === 'EMAIL') badgeColor = 'var(--accent-blue)';
      if (ent.label === 'PHONE') badgeColor = 'var(--accent-red)';
      
      elements.push(
        <span 
          key={`e-${i}`} 
          style={{ 
            backgroundColor: badgeColor + '22', 
            border: `1px solid ${badgeColor}`, 
            color: 'var(--text-main)',
            padding: '0.15rem 0.4rem', 
            borderRadius: '4px',
            margin: '0 0.15rem',
            fontWeight: 600,
            fontSize: '0.85em',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.25rem',
            whiteSpace: 'nowrap'
          }}
        >
          {ent.text}
          <span style={{ fontSize: '0.65em', textTransform: 'uppercase', opacity: 0.7, marginLeft: '0.2rem', padding: '0.05rem 0.2rem', backgroundColor: badgeColor + '33', borderRadius: '2px' }}>
            {ent.label}
          </span>
        </span>
      );
      lastIdx = ent.end;
    });
    
    if (lastIdx < text.length) {
      elements.push(<span key="t-end">{text.substring(lastIdx)}</span>);
    }
    
    return (
      <div style={{ lineHeight: '2', fontSize: '0.95rem', color: 'var(--text-main)', textAlign: 'left', wordBreak: 'break-word' }}>
        {elements}
      </div>
    );
  };

  const renderPlaygroundResults = () => {
    if (!predictionResult) {
      return (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <i className="fa-solid fa-terminal" style={{ fontSize: '3rem', marginBottom: '1.5rem', opacity: 0.35, color: 'var(--accent-primary)' }}></i>
          <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.25rem' }}>Awaiting Inference Trigger</p>
          <p style={{ fontSize: '0.85rem' }}>Provide inputs on the left and click predict to evaluate outputs.</p>
        </div>
      );
    }

    const task = project?.task || 'Classification';

    if (task === 'Named Entity Recognition (NER)') {
      return (
        <div style={{ width: '100%' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <i className="fa-solid fa-tags" style={{ color: 'var(--accent-purple)' }}></i>
            Highlighted Named Entities
          </h4>
          {renderHighlightedEntities(pastedText, predictionResult.entities)}
        </div>
      );
    }

    if (task === 'Text Summarization') {
      return (
        <div style={{ width: '100%' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <i className="fa-solid fa-quote-left" style={{ color: 'var(--accent-green)' }}></i>
            Extracted Summary Output
          </h4>
          <div style={{ backgroundColor: 'var(--bg-tertiary)', borderLeft: '4px solid var(--accent-green)', padding: '1.25rem', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0', fontSize: '0.95rem', lineHeight: '1.7', color: 'var(--text-main)', fontStyle: 'italic', textAlign: 'left' }}>
            "{predictionResult.summary}"
          </div>
        </div>
      );
    }

    if (task === 'OCR (Optical Character Recognition)') {
      return (
        <div style={{ width: '100%' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <i className="fa-solid fa-file-invoice" style={{ color: 'var(--accent-primary)' }}></i>
            Extracted OCR Document Text
          </h4>
          <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontSize: '0.85rem', fontFamily: 'Consolas, Monaco, monospace', color: 'var(--text-main)', lineHeight: '1.6', textAlign: 'left', whiteSpace: 'pre-wrap', margin: 0 }}>
            {predictionResult.text}
          </pre>
        </div>
      );
    }

    if (task === 'Speech Recognition (ASR)') {
      return (
        <div style={{ width: '100%' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <i className="fa-solid fa-microphone" style={{ color: 'var(--accent-purple)' }}></i>
            Transcribed ASR Text Output
          </h4>
          <div style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontSize: '0.95rem', lineHeight: '1.6', color: 'var(--text-main)', textAlign: 'left' }}>
            {predictionResult.text}
          </div>
        </div>
      );
    }

    if (task === 'Face Detection') {
      return (
        <div style={{ width: '100%', textAlign: 'center' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem' }}>
            Face Detection Output ({predictionResult.faces_count} face(s) found)
          </h4>
          <div style={{ position: 'relative', display: 'inline-block', maxWidth: '100%', border: '2px solid var(--accent-primary)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
            <img src={`data:image/jpeg;base64,${predictionResult.image}`} style={{ maxWidth: '100%', maxHeight: '280px', display: 'block' }} alt="Faces Detected" />
          </div>
        </div>
      );
    }

    if (task === 'Image Denoising' || task === 'Super Resolution') {
      return (
        <div style={{ width: '100%', textAlign: 'center' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1.25rem' }}>
            Processed Visual Output vs Original
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', width: '100%' }}>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>Original Input</span>
              <img src={imagePreview} style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', borderRadius: '4px', border: '1px solid var(--border-color)' }} alt="Original" />
            </div>
            <div>
              <span style={{ fontSize: '0.7rem', color: 'var(--accent-green)', display: 'block', marginBottom: '0.25rem' }}>Processed Model Output</span>
              <img src={`data:image/jpeg;base64,${predictionResult.image}`} style={{ width: '100%', maxHeight: '180px', objectFit: 'contain', borderRadius: '4px', border: '1px solid var(--accent-green)' }} alt="Processed" />
            </div>
          </div>
        </div>
      );
    }

    if (task === 'Noise Reduction') {
      return (
        <div style={{ width: '100%', textAlign: 'center' }}>
          <h4 style={{ color: 'var(--text-muted)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1.25rem' }}>
            Original vs Denoised Sound Player
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%', padding: '0.5rem 0' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Original Audio Input:</span>
              <audio controls src={audioPreview} style={{ width: '100%', height: '36px' }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.7rem', color: 'var(--accent-green)', fontWeight: 600 }}>Denoised Model Output:</span>
              <audio controls src={`data:audio/wav;base64,${predictionResult.audio}`} autoPlay style={{ width: '100%', height: '36px' }} />
            </div>
          </div>
        </div>
      );
    }

    // Default classification / regression result widget
    return (
      <div style={{ textAlign: 'center', width: '100%' }}>
        <i className="fa-solid fa-circle-nodes" style={{ fontSize: '3rem', color: 'var(--accent-primary)', marginBottom: '1.5rem' }}></i>
        <h4 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Prediction Result</h4>
        <div style={{ fontSize: '3.5rem', fontWeight: 800, margin: '0.5rem 0 1rem', color: 'var(--text-main)', wordBreak: 'break-all' }}>
          {predictionResult.prediction}
        </div>

        {predictionResult.confidence !== undefined && (
          <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em' }}>Confidence Score</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent-green)', marginTop: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <i className="fa-solid fa-circle-nodes"></i>
              {(predictionResult.confidence * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    );
  };

  // const handlePredictRealtime = async (e) => {
  //   e.preventDefault();
  //   setPredLoading(true);
  //   setPredictionResult(null);

  //   try {
  //     const res = await api.post(`/projects/${datasetStatus.project_id}/deploy/predict`, {
  //       features
  //     });
  //     setPredictionResult(res.data);
  //     toast.success("Prediction completed!");
  //   } catch (err) {
  //     toast.error(err.response?.data?.detail || "Inference call failed. Validate input types.");
  //   } finally {
  //     setPredLoading(false);
  //   }
  // };

  const handleBatchPredict = async (e) => {
    e.preventDefault();
    if (!batchFile) return;
    setBatchLoading(true);

    const formData = new FormData();
    formData.append("file", batchFile);

    try {
      const res = await api.post(`/projects/${datasetStatus.project_id}/deploy/predict-batch`, formData, {
        responseType: 'blob' // Important to handle file stream download
      });

      // Trigger browser file download
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `batch_predictions_${batchFile.name}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);

      toast.success("Batch predictions computed and downloaded successfully!");
      setBatchFile(null);
    } catch (err) {
      toast.error("Batch processing failed. Make sure columns match the training features.");
    } finally {
      setBatchLoading(false);
    }
  };

  if (models.length === 0) return null;

  if (!deployedModel) {
    return (
      <div className="card" style={{ padding: '3rem', textAlign: 'center', marginTop: '2rem' }}>
        <i className="fa-solid fa-circle-pause" style={{ fontSize: '3rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}></i>
        <h2 style={{ marginBottom: '0.75rem' }}>Service Offline</h2>
        <p style={{ color: 'var(--text-muted)' }}>
          No models are currently deployed. Go to the <strong>Test Model</strong> page, choose a model, and click "Deploy Active API" to enable serving.
        </p>
      </div>
    );
  }

  const formColumns = datasetStatus?.initial_columns || datasetStatus?.columns || [];
  const formDtypes = datasetStatus?.initial_dtypes || datasetStatus?.dtypes || {};
  const featureCols = formColumns.filter(c => c !== deployedModel.target_col) || [];

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Deploy & Serve Model</h1>
          <p className="page-subtitle">Expose REST APIs, perform batch predictions, and fetch integration code snippets</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button 
            onClick={() => setGuideOpen(true)} 
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
          >
            <i className="fa-solid fa-graduation-cap"></i>
            <span>Get to know more</span>
          </button>
          <span className="badge badge-success" style={{ padding: '0.4rem 0.8rem', borderRadius: '20px', height: 'auto', display: 'inline-flex', alignItems: 'center' }}>
            <i className="fa-solid fa-circle-check" style={{ marginRight: '0.4rem' }}></i>
            Deployed: {deployedModel.model_type}
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="tab-container">
        <button className={`tab-btn ${activeTab === 'realtime' ? 'active' : ''}`} onClick={() => setActiveTab('realtime')}>
          <i className="fa-solid fa-bolt" style={{ marginRight: '0.5rem' }}></i>
          Real-time Prediction Form
        </button>
        <button className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`} onClick={() => setActiveTab('batch')}>
          <i className="fa-solid fa-file-invoice" style={{ marginRight: '0.5rem' }}></i>
          Batch Predictions (CSV)
        </button>
        <button className={`tab-btn ${activeTab === 'api' ? 'active' : ''}`} onClick={() => setActiveTab('api')}>
          <i className="fa-solid fa-code" style={{ marginRight: '0.5rem' }}></i>
          REST API Code Integration
        </button>
      </div>

      {/* Content cards */}
      <div style={{ position: 'relative' }}>
        {/* Tab 1: Real-time Form */}
        {activeTab === 'realtime' && (
          <div className="grid-2" style={{ alignItems: 'start' }}>
            <div className="card">
              <h3 className="card-title">Inference Playground</h3>
              <p className="card-subtitle">Provide sample input values to evaluate your deployed model instantly.</p>

              {dataType === 'text' ? (
                <form onSubmit={handlePredictText}>
                  <div className="form-group">
                    <label className="form-label" style={{ marginBottom: '0.5rem' }}>Raw Text Content</label>
                    <textarea
                      className="form-control"
                      style={{ height: '220px', resize: 'none', fontFamily: 'var(--font-body)', fontSize: '0.9rem', lineHeight: '1.5' }}
                      placeholder="Type or paste your raw text sentence or document paragraph here..."
                      value={pastedText}
                      onChange={(e) => setPastedText(e.target.value)}
                      disabled={predLoading}
                      required
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                    <button type="submit" className="btn btn-primary" style={{ flex: 1 }} disabled={predLoading || !pastedText.trim()}>
                      {predLoading ? <span className="spinner"></span> : <span>Run NLP Inference</span>}
                    </button>
                    {pastedText && (
                      <button type="button" onClick={handleClearPlayground} className="btn btn-secondary" disabled={predLoading}>
                        Clear
                      </button>
                    )}
                  </div>
                </form>
              ) : (dataType === 'image' ? (
                <form onSubmit={handlePredictFile}>
                  <div 
                    className="uploader-box" 
                    style={{ padding: '2.5rem 1rem', borderStyle: imagePreview ? 'solid' : 'dashed' }} 
                    onClick={() => document.getElementById('image-playground-input').click()}
                  >
                    {imagePreview ? (
                      <img src={imagePreview} style={{ maxHeight: '180px', borderRadius: 'var(--radius-sm)', maxWidth: '100%', objectFit: 'contain' }} alt="Playground Preview" />
                    ) : (
                      <>
                        <i className="fa-solid fa-image uploader-icon" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}></i>
                        <h4 style={{ fontSize: '0.85rem', fontWeight: 600 }}>Select or Drop Image File</h4>
                        <p style={{ color: 'var(--text-dim)', fontSize: '0.75rem', marginTop: '0.25rem' }}>Supports PNG, JPG, JPEG, WEBP, BMP</p>
                      </>
                    )}
                    <input
                      id="image-playground-input"
                      type="file"
                      style={{ display: 'none' }}
                      accept="image/*"
                      onChange={(e) => {
                        if (e.target.files?.[0]) {
                          const f = e.target.files[0];
                          setImageFile(f);
                          setImagePreview(URL.createObjectURL(f));
                          setPredictionResult(null);
                        }
                      }}
                      disabled={predLoading}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                    <button type="submit" className="btn btn-primary" style={{ flex: 1 }} disabled={predLoading || !imageFile}>
                      {predLoading ? <span className="spinner"></span> : <span>Execute Vision Inference</span>}
                    </button>
                    {imageFile && (
                      <button type="button" onClick={handleClearPlayground} className="btn btn-secondary" disabled={predLoading}>
                        Clear
                      </button>
                    )}
                  </div>
                </form>
              ) : (dataType === 'audio' ? (
                <form onSubmit={handlePredictFile}>
                  <div 
                    className="uploader-box" 
                    style={{ padding: '2.5rem 1rem' }} 
                    onClick={() => document.getElementById('audio-playground-input').click()}
                  >
                    {audioFile ? (
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }} onClick={(e) => e.stopPropagation()}>
                        <i className="fa-solid fa-file-audio" style={{ fontSize: '2.5rem', color: 'var(--accent-primary)' }}></i>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)' }}>{audioFile.name}</span>
                        <audio controls src={audioPreview} style={{ width: '220px', height: '36px' }} />
                      </div>
                    ) : (
                      <>
                        <i className="fa-solid fa-file-audio uploader-icon" style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}></i>
                        <h4 style={{ fontSize: '0.85rem', fontWeight: 600 }}>Select or Drop Audio File</h4>
                        <p style={{ color: 'var(--text-dim)', fontSize: '0.75rem', marginTop: '0.25rem' }}>Supports WAV, MP3, OGG, FLAC</p>
                      </>
                    )}
                    <input
                      id="audio-playground-input"
                      type="file"
                      style={{ display: 'none' }}
                      accept="audio/*"
                      onChange={(e) => {
                        if (e.target.files?.[0]) {
                          const f = e.target.files[0];
                          setAudioFile(f);
                          setAudioPreview(URL.createObjectURL(f));
                          setPredictionResult(null);
                        }
                      }}
                      disabled={predLoading}
                    />
                  </div>
                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                    <button type="submit" className="btn btn-primary" style={{ flex: 1 }} disabled={predLoading || !audioFile}>
                      {predLoading ? <span className="spinner"></span> : <span>Execute Audio Inference</span>}
                    </button>
                    {audioFile && (
                      <button type="button" onClick={handleClearPlayground} className="btn btn-secondary" disabled={predLoading}>
                        Clear
                      </button>
                    )}
                  </div>
                </form>
              ) : (
                <form onSubmit={handlePredictRealtime}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', maxHeight: '400px', overflowY: 'auto', paddingRight: '0.5rem', marginBottom: '1.5rem' }}>
                    {featureCols.map(col => {
                      const dtype = formDtypes[col];
                      const isNum = dtype?.includes('int') || dtype?.includes('float');
                      return (
                        <div className="form-group" key={col} style={{ margin: 0 }}>
                          <label className="form-label" style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {col} ({dtype})
                          </label>
                          <input
                            type={isNum ? "number" : "text"}
                            step="any"
                            className="form-control"
                            value={features[col] ?? ''}
                            onChange={(e) => handleFeatureChange(col, e.target.value, isNum)}
                            required
                            placeholder={isNum ? "0.0" : "text value"}
                          />
                        </div>
                      );
                    })}
                  </div>

                  <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={predLoading}>
                    {predLoading ? (
                      <>
                        <span className="spinner" style={{ marginRight: '0.5rem' }}></span>
                        Computing...
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-play" style={{ marginRight: '0.25rem' }}></i>
                        Run Real-time Inference
                      </>
                    )}
                  </button>
                </form>
              )))}
            </div>

            {/* Results Display */}
            <div className="card" style={{ minHeight: '320px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '2rem', background: 'var(--bg-glass)', backdropFilter: 'blur(12px)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)' }}>
              {renderPlaygroundResults()}
            </div>
          </div>
        )}

        {/* Tab 2: Batch Predictions */}
        {activeTab === 'batch' && (
          <div className="grid-2" style={{ alignItems: 'start' }}>
            <div className="card">
              <h3 className="card-title">Upload Batch File</h3>
              <p className="card-subtitle">Upload a CSV file containing columns with matching feature headers to perform bulk inference.</p>

              <form onSubmit={handleBatchPredict}>
                <div
                  className="uploader-box"
                  style={{ padding: '2.5rem 1rem' }}
                  onClick={() => document.getElementById('batch-upload-input').click()}
                >
                  <i className="fa-solid fa-file-excel uploader-icon" style={{ color: batchFile ? 'var(--accent-green)' : 'var(--text-dim)' }}></i>
                  <h4 style={{ fontWeight: 600, color: batchFile ? 'var(--text-main)' : 'var(--text-muted)' }}>{batchFile ? batchFile.name : "Choose CSV File"}</h4>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>Only .csv files supported</p>
                  <input
                    id="batch-upload-input"
                    type="file"
                    style={{ display: 'none' }}
                    accept=".csv"
                    onChange={(e) => {
                      if (e.target.files?.[0]) {
                        setBatchFile(e.target.files[0]);
                      }
                    }}
                  />
                </div>

                <button
                  type="submit"
                  className="btn btn-primary"
                  style={{ width: '100%', marginTop: '1.5rem' }}
                  disabled={batchLoading || !batchFile}
                >
                  {batchLoading ? (
                    <>
                      <span className="spinner" style={{ marginRight: '0.5rem' }}></span>
                      Processing Batch...
                    </>
                  ) : (
                    <>
                      <i className="fa-solid fa-cloud-arrow-down" style={{ marginRight: '0.25rem' }}></i>
                      Run Batch Inference & Download
                    </>
                  )}
                </button>
              </form>
            </div>

            <div className="card" style={{ padding: '2rem' }}>
              <h3 className="card-title">Batch Formatting Guidelines</h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.6', marginBottom: '1rem' }}>
                For batch processing, your CSV file should contain all dataset headers EXCEPT the target column. 
              </p>
              <div style={{ backgroundColor: 'var(--bg-primary)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontFamily: 'Consolas, Monaco, monospace', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                {featureCols.slice(0, 4).join(',')}...
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.6' }}>
                The output file will automatically append a <strong>"prediction"</strong> column (and a <strong>"prediction_confidence"</strong> column if classification) and trigger a download of the modified CSV file.
              </p>
            </div>
          </div>
        )}

        {/* Tab 3: API Integration Code */}
        {activeTab === 'api' && (
          <div className="card" style={{ padding: '2.5rem' }}>
            <h3 className="card-title">REST Endpoint Documentation</h3>
            <p className="card-subtitle">Integrate your active model serving deployment directly into external apps or workflows.</p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div>
                <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: '0.5rem' }}>Endpoint url</h4>
                <div style={{ backgroundColor: 'var(--bg-primary)', padding: '0.85rem 1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', color: 'var(--accent-primary)', fontWeight: 700, fontSize: '0.9rem', fontFamily: 'Consolas, Monaco, monospace' }}>
                  {dataType === 'image' || dataType === 'audio' 
                    ? `POST http://127.0.0.1:8000/api/projects/${datasetStatus.project_id}/deploy/predict-file`
                    : `POST http://127.0.0.1:8000/api/projects/${datasetStatus.project_id}/deploy/predict`
                  }
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700, marginBottom: '0.5rem' }}>Python Integration Script</h4>
                <div style={{ display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  {/* Top bar mockup */}
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '0.6rem 1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', gap: '0.4rem' }}>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-red)', display: 'inline-block' }}></span>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-yellow)', display: 'inline-block' }}></span>
                      <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-green)', display: 'inline-block' }}></span>
                    </div>
                    <span style={{ fontSize: '0.75rem', fontFamily: 'Consolas, Monaco, monospace', color: 'var(--text-muted)' }}>python — prediction.py</span>
                    <div style={{ width: '38px' }}></div>
                  </div>
                  <pre
                    style={{
                      margin: 0,
                      padding: '1.25rem',
                      fontFamily: 'Consolas, Monaco, monospace',
                      fontSize: '0.8rem',
                      color: 'var(--text-main)',
                      overflowX: 'auto',
                      lineHeight: '1.6'
                    }}
                  >
{dataType === 'image' || dataType === 'audio' ? `import requests

url = "http://127.0.0.1:8000/api/projects/${datasetStatus.project_id}/deploy/predict-file"
headers = {
    "Authorization": "Bearer YOUR_JWT_ACCESS_TOKEN"
}
# Upload raw media file for pipeline processing
files = {
    "file": ("sample.${dataType === 'image' ? 'jpg' : 'wav'}", open("sample.${dataType === 'image' ? 'jpg' : 'wav'}", "rb"), "${dataType === 'image' ? 'image/jpeg' : 'audio/wav'}")
}

response = requests.post(url, headers=headers, files=files)
result = response.json()
print("Prediction Result:", result["prediction"])
if "image" in result:
    print("Base64 encoded visual result present.")` : `import requests

url = "http://127.0.0.1:8000/api/projects/${datasetStatus.project_id}/deploy/predict"
headers = {
    "Authorization": "Bearer YOUR_JWT_ACCESS_TOKEN",
    "Content-Type": "application/json"
}
data = {
    "features": {
${dataType === 'text' ? '        "text": "Your sample text statement goes here"' : featureCols.slice(0, 3).map(c => `        "${c}": ${datasetStatus.dtypes?.[c]?.includes('int') || datasetStatus.dtypes?.[c]?.includes('float') ? '0.0' : '"value"'}`).join(',\n')}
    }
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print("Prediction:", result["prediction"])
if "confidence" in result:
    print("Confidence:", result["confidence"])`}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      <GuideDrawer isOpen={guideOpen} onClose={() => setGuideOpen(false)} initialTopic="deployment" />
    </div>
  );
};

export default Deployment;
