import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import api from '../services/api';

const TrainModel = ({ project, datasetStatus, refreshModels }) => {
  const [targetCol, setTargetCol] = useState('');
  const [textCol, setTextCol] = useState('');
  const [modelName, setModelName] = useState('Random Forest');
  const [useTuning, setUseTuning] = useState(false);
  const [trials, setTrials] = useState(10);
  const [valSplit, setValSplit] = useState(0.2);

  const [availableOptions, setAvailableOptions] = useState({ classification: [], regression: [] });
  const [training, setTraining] = useState(false);
  const [logs, setLogs] = useState([]);
  const [hasTrained, setHasTrained] = useState(false);
  
  const consoleEndRef = useRef(null);
  const navigate = useNavigate();
  const pollIntervalRef = useRef(null);

  const fetchTrainingOptions = async () => {
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/training/options`);
      setAvailableOptions(res.data);
    } catch (err) {
      console.error("Failed to load options", err);
    }
  };

  const isPretrainedTask = (task) => {
    const pretrainedTasks = [
      'Named Entity Recognition (NER)',
      'Text Summarization',
      'Face Detection',
      'OCR (Optical Character Recognition)',
      'Image Denoising',
      'Super Resolution',
      'Speech Recognition (ASR)',
      'Noise Reduction'
    ];
    return pretrainedTasks.includes(task);
  };

  useEffect(() => {
    fetchTrainingOptions();
    
    // Auto-detect target and text columns
    if (project?.task && isPretrainedTask(project.task)) {
      setModelName(`${project.task} Pipeline`);
    } else if (datasetStatus?.columns?.length > 0) {
      const cols = datasetStatus.columns;
      const defaultTarget = (project?.target_col && cols.includes(project.target_col))
        ? project.target_col
        : cols[cols.length - 1];
      
      if (project?.data_type === 'text') {
        setTextCol(cols[0]);
        setTargetCol(defaultTarget);
        setModelName('Logistic Regression');
      } else if (project?.data_type === 'image') {
        setModelName('resnet18');
      } else if (project?.data_type === 'audio') {
        setModelName('cnn');
      } else {
        setTargetCol(defaultTarget);
        setModelName('Random Forest');
      }
    }

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [datasetStatus, project]);

  // Scroll console to bottom on logs update
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const getTargetType = () => {
    if (project?.data_type === 'image') return 'Computer Vision';
    if (project?.data_type === 'audio') return 'Audio DSP Signals';
    if (project?.data_type === 'text') return 'Natural Language Classification';
    if (!targetCol || !datasetStatus?.dtypes) return 'Unknown';
    const dtype = datasetStatus.dtypes[targetCol];
    if (dtype?.includes('int') || dtype?.includes('float')) {
      const isBinary = datasetStatus.unique_counts?.[targetCol] === 2;
      return isBinary ? 'Binary Classification' : 'Regression';
    }
    return 'Classification';
  };

  const getAlgorithmList = () => {
    const task = project?.task || 'Classification';
    if (isPretrainedTask(task)) {
      return [`${task} Pipeline`];
    }
    if (task === 'Object Detection') {
      return ['yolo'];
    }
    if (project?.data_type === 'image') {
      return ['resnet18', 'resnet50', 'vgg16', 'mobilenet'];
    }
    if (project?.data_type === 'audio') {
      return ['cnn', 'lstm'];
    }
    if (project?.data_type === 'text') {
      return ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'XGBoost'];
    }
    const type = getTargetType();
    if (type.includes('Classification')) {
      return availableOptions.classification || [];
    }
    return availableOptions.regression || [];
  };

  // Start checking training status
  const startPollingStatus = () => {
    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await api.get(`/projects/${datasetStatus.project_id}/training/status`);
        setLogs(res.data.logs || []);
        
        if (res.data.status === 'completed') {
          clearInterval(pollIntervalRef.current);
          setTraining(false);
          setHasTrained(true);
          toast.success("Model training finished successfully!");
          refreshModels();
        } else if (res.data.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          setTraining(false);
          toast.error("Model training failed. Review console logs.");
        }
      } catch (err) {
        console.error("Error polling training status", err);
      }
    }, 2000);
  };

  const handleTrain = async (e) => {
    e.preventDefault();
    setTraining(true);
    setHasTrained(false);
    setLogs(["Submitting job..."]);

    const isFolderBased = project?.data_type === 'image' || project?.data_type === 'audio';

    try {
      await api.post(`/projects/${datasetStatus.project_id}/training/train`, {
        model_name: modelName,
        target_col: isFolderBased ? '' : targetCol,
        text_col: project?.data_type === 'text' ? textCol : null,
        use_hyperparameter_tuning: useTuning,
        trials: trials,
        validation_split: valSplit
      });
      startPollingStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to trigger training.");
      setTraining(false);
    }
  };

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Model Training</h1>
          <p className="page-subtitle">Configure hyperparameters and train machine learning models on your dataset</p>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Settings Card */}
        <div className="card">
          <h3 className="card-title">Training Settings</h3>
          <p className="card-subtitle">Select features and tuning options for your model pipeline.</p>

          <form onSubmit={handleTrain}>
            {isPretrainedTask(project?.task) ? (
              <div style={{ marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: 'var(--radius-sm)', padding: '1.25rem', display: 'flex', gap: '0.75rem' }}>
                  <i className="fa-solid fa-circle-check" style={{ color: 'var(--accent-green)', fontSize: '1.2rem', marginTop: '0.1rem' }}></i>
                  <div>
                    <h4 style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.9rem', marginBottom: '0.25rem' }}>Pre-trained Pipeline Engine Active</h4>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.5', margin: 0 }}>
                      This task utilizes a specialized pre-trained model. No custom training is necessary! Click below to compile and register the pipeline for immediate inference.
                    </p>
                  </div>
                </div>
                
                <div className="form-group">
                  <label className="form-label">Active Model Pipeline</label>
                  <input className="form-control" type="text" value={`${project?.task || 'Specialized'} Base Engine`} disabled />
                </div>
              </div>
            ) : (
              <>
                {project?.data_type === 'text' && (
                  <div className="form-group">
                    <label className="form-label">Input Text Column</label>
                    <select className="form-control" value={textCol} onChange={(e) => setTextCol(e.target.value)} disabled={training} style={{ cursor: 'pointer' }}>
                      {datasetStatus?.columns?.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                )}

                {project?.data_type !== 'image' && project?.data_type !== 'audio' && (
                  <div className="form-group">
                    <label className="form-label">Target Column (Label)</label>
                    <select className="form-control" value={targetCol} onChange={(e) => setTargetCol(e.target.value)} disabled={training} style={{ cursor: 'pointer' }}>
                      {datasetStatus?.columns?.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                    <small style={{ color: 'var(--text-muted)', display: 'block', marginTop: '0.5rem', fontSize: '0.8rem' }}>
                      Detected task: <strong style={{ color: 'var(--accent-primary)' }}>{getTargetType()}</strong>
                    </small>
                  </div>
                )}

                <div className="form-group">
                  <label className="form-label">ML Model Algorithm</label>
                  <select className="form-control" value={modelName} onChange={(e) => setModelName(e.target.value)} disabled={training} style={{ cursor: 'pointer' }}>
                    {getAlgorithmList()?.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group" style={{ display: 'flex', gap: '0.75rem', padding: '0.75rem 0', alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    id="tuning-checkbox"
                    checked={useTuning}
                    onChange={(e) => setUseTuning(e.target.checked)}
                    disabled={training}
                    style={{ cursor: 'pointer', width: '18px', height: '18px', accentColor: 'var(--accent-primary)' }}
                  />
                  <label htmlFor="tuning-checkbox" style={{ fontWeight: 500, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text-main)' }}>
                    Optimize Hyperparameters (Optuna Search)
                  </label>
                </div>

                {useTuning && (
                  <div className="form-group" style={{ paddingLeft: '1.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <label className="form-label" style={{ margin: 0 }}>Optuna Trials Limit</label>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-purple)' }}>{trials} trials</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={trials}
                      onChange={(e) => setTrials(parseInt(e.target.value))}
                      disabled={training}
                      style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-primary)' }}
                    />
                  </div>
                )}

                <div className="form-group" style={{ marginBottom: '2rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <label className="form-label" style={{ margin: 0 }}>Validation Split Rate</label>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)' }}>{Math.round(valSplit * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.4"
                    step="0.05"
                    value={valSplit}
                    onChange={(e) => setValSplit(parseFloat(e.target.value))}
                    disabled={training}
                    style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent-primary)' }}
                  />
                </div>
              </>
            )}

            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                type="submit"
                className="btn btn-primary"
                style={{ flex: 1 }}
                disabled={training || (!isPretrainedTask(project?.task) && project?.data_type !== 'image' && project?.data_type !== 'audio' && !targetCol)}
              >
                {training ? (
                  <>
                    <span className="spinner" style={{ marginRight: '0.5rem' }}></span>
                    Training Model...
                  </>
                ) : (
                  <>
                    <i className="fa-solid fa-bolt" style={{ marginRight: '0.25rem' }}></i>
                    {isPretrainedTask(project?.task) ? 'Compile & Register Pipeline' : 'Run Pipeline Training'}
                  </>
                )}
              </button>
              {hasTrained && (
                <button
                  type="button"
                  onClick={() => navigate(`/projects/${datasetStatus.project_id}/test`)}
                  className="btn btn-secondary"
                  style={{ color: 'var(--accent-green)', borderColor: 'rgba(16, 185, 129, 0.3)' }}
                >
                  <i className="fa-solid fa-chevron-right" style={{ marginRight: '0.25rem' }}></i>
                  Test Results
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Live Logs Terminal Mockup */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '480px' }}>
          <h3 className="card-title">Training Output Console</h3>
          <p className="card-subtitle">Real-time status and logs from background training threads.</p>

          <div style={{ display: 'flex', flexDirection: 'column', flex: 1, backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
            {/* Terminal Top bar */}
            <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '0.6rem 1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-red)', display: 'inline-block' }}></span>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-yellow)', display: 'inline-block' }}></span>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--accent-green)', display: 'inline-block' }}></span>
              </div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'Consolas, Monaco, monospace', color: 'var(--text-muted)' }}>bash — instaml-trainer</span>
              <div style={{ width: '38px' }}></div>
            </div>

            {/* Terminal logs content */}
            <div
              style={{
                flex: 1,
                padding: '1.25rem',
                overflowY: 'auto',
                fontFamily: 'Consolas, Monaco, monospace',
                fontSize: '0.825rem',
                color: 'var(--text-main)',
                lineHeight: '1.6'
              }}
            >
              {logs.length === 0 ? (
                <span style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>Console idle. Awaiting training trigger...</span>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} style={{ marginBottom: '0.35rem', borderLeft: '2px solid var(--border-color)', paddingLeft: '0.5rem' }}>
                    <span style={{ color: 'var(--text-dim)' }}>[{idx + 1}]</span> {log}
                  </div>
                ))
              )}
              <div ref={consoleEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainModel;

