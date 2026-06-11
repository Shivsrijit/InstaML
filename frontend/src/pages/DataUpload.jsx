import React, { useState } from 'react';
import api from '../services/api';
import { toast } from 'react-hot-toast';
import GuideDrawer from '../components/GuideDrawer';

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

const DataUpload = ({ project, datasetStatus, refreshStatus }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pastedText, setPastedText] = useState('');
  const [activeTab, setActiveTab] = useState('file'); // 'file' or 'paste'
  const [taskUpdating, setTaskUpdating] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);

  const dataType = project?.data_type || 'tabular';
  const isImageOrAudio = dataType === 'image' || dataType === 'audio';
  const allowedExtensions = dataType === 'text'
    ? ['.csv', '.xlsx', '.xls', '.parquet', '.zip', '.txt']
    : (dataType === 'image'
      ? ['.zip', '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
      : (dataType === 'audio'
        ? ['.zip', '.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
        : ['.csv', '.xlsx', '.xls', '.parquet']));
        
  const acceptString = dataType === 'text'
    ? '.csv, .xlsx, .xls, .parquet, .zip, .txt'
    : (dataType === 'image'
      ? '.zip, .jpg, .jpeg, .png, .bmp, .webp, .tiff'
      : (dataType === 'audio'
        ? '.zip, .wav, .mp3, .ogg, .flac, .m4a, .aac'
        : '.csv, .xlsx, .xls, .parquet'));

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      const selectedExt = '.' + selectedFile.name.split('.').pop().toLowerCase();
      if (!allowedExtensions.includes(selectedExt)) {
        toast.error(`Unsupported file format. Please upload a file with one of these extensions: ${allowedExtensions.join(', ')}`);
        setFile(null);
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      toast.error("Please select a file to upload first.");
      return;
    }

    const selectedExt = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(selectedExt)) {
      toast.error(`Unsupported file format. Please upload a file with one of these extensions: ${allowedExtensions.join(', ')}`);
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    const projectId = project?.id || datasetStatus?.project_id;

    try {
      await api.post(`/projects/${projectId}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      toast.success(`Dataset "${file.name}" uploaded successfully!`);
      setFile(null);
      refreshStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || `Upload failed. Verify file format (${isImageOrAudio ? '.zip' : 'CSV, Excel, Parquet'}).`);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadPastedText = async (e) => {
    e.preventDefault();
    if (!pastedText.trim()) return;

    setLoading(true);

    const blob = new Blob([pastedText], { type: 'text/plain' });
    const fileOfBlob = new File([blob], 'pasted_text.txt', { type: 'text/plain' });

    const formData = new FormData();
    formData.append("file", fileOfBlob);

    const projectId = project?.id || datasetStatus?.project_id;

    try {
      await api.post(`/projects/${projectId}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      toast.success("Pasted text loaded and registered successfully!");
      setPastedText('');
      refreshStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to process pasted text.");
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchToText = async () => {
    setLoading(true);
    const projectId = project?.id || datasetStatus?.project_id;
    try {
      await api.patch(`/projects/${projectId}`, { data_type: 'text' });
      toast.success("Successfully updated project format to Raw Text!");
      refreshStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to switch project format to text.");
    } finally {
      setLoading(false);
    }
  };

  const renderCurrentDataInfo = () => {
    if (!datasetStatus?.data_loaded) return null;

    return (
      <div style={{ marginTop: '3rem' }}>
        <h3 style={{ fontSize: '0.75rem', marginBottom: '1.25rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>Active Dataset Summary</h3>
        
        {/* Metric widgets */}
        <div className="grid-4" style={{ marginBottom: '2.5rem' }}>
          <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700', letterSpacing: '0.05em' }}>Rows (Samples)</span>
            <span style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--text-main)' }}>
              {datasetStatus.shape?.[0]?.toLocaleString()}
            </span>
          </div>

          <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700', letterSpacing: '0.05em' }}>Columns (Features)</span>
            <span style={{ fontSize: '1.5rem', fontWeight: '700', color: 'var(--text-main)' }}>
              {datasetStatus.shape?.[1]?.toLocaleString()}
            </span>
          </div>

          <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700', letterSpacing: '0.05em' }}>Missing Cells</span>
            <span style={{ fontSize: '1.5rem', fontWeight: '700', color: datasetStatus.missing_total > 0 ? 'var(--accent-yellow)' : 'var(--text-main)' }}>
              {datasetStatus.missing_total?.toLocaleString()}
            </span>
          </div>

          <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '700', letterSpacing: '0.05em' }}>Duplicate Rows</span>
            <span style={{ fontSize: '1.5rem', fontWeight: '700', color: datasetStatus.duplicate_count > 0 ? 'var(--accent-red)' : 'var(--text-main)' }}>
              {datasetStatus.duplicate_count?.toLocaleString()}
            </span>
          </div>
        </div>

        {/* Data Preview */}
        {datasetStatus.preview && datasetStatus.preview.length > 0 && (
          <div className="card" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
            <h4 style={{ fontSize: '0.9rem', marginBottom: '1.25rem', fontWeight: 600, color: 'var(--text-main)', letterSpacing: '-0.01em' }}>Dataset Sample Preview (First 10 records)</h4>
            <div className="table-container" style={{ maxHeight: '350px' }}>
              <table className="table">
                <thead>
                  <tr>
                    {datasetStatus.columns?.map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {datasetStatus.preview.slice(0, 10).map((row, idx) => (
                    <tr key={idx}>
                      {datasetStatus.columns?.map((col) => (
                        <td key={col} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '200px' }}>
                          {row[col] === null ? <span style={{ color: 'var(--accent-red)', fontSize: '0.8rem', fontStyle: 'italic' }}>null</span> : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const getPageSubtitle = () => {
    switch (dataType) {
      case 'image':
        return 'Upload image datasets inside class directories compressed in a .zip file';
      case 'audio':
        return 'Upload audio datasets inside class directories compressed in a .zip file';
      case 'text':
        return 'Upload your raw text dataset (CSV, Excel, Parquet, or ZIP folder structures)';
      default:
        return 'Upload your structured tabular dataset (CSV, Excel, or Parquet)';
    }
  };

  const getSubtitle = () => {
    if (dataType === 'text') {
      return `Upload a spreadsheet with text and label columns, a single text file (.txt), a folder-based .zip, or paste raw text.`;
    }
    if (dataType === 'image') {
      return `Upload a .zip file containing category subfolders, or select a single or couple of raw images (.jpg, .png, .webp).`;
    }
    if (dataType === 'audio') {
      return `Upload a .zip file containing category subfolders, or select a single or couple of raw sound files (.wav, .mp3).`;
    }
    return `Select a file containing your dataset from your device. Supported formats: .csv, .xlsx, .xls, .parquet`;
  };

  const getUploaderIcon = () => {
    if (dataType === 'image') return 'fa-solid fa-file-image';
    if (dataType === 'audio') return 'fa-solid fa-file-audio';
    if (dataType === 'text') return 'fa-solid fa-file-lines';
    return 'fa-solid fa-file-csv';
  };

  const getUploaderText = () => {
    if (dataType === 'image') {
      return 'Choose ZIP Archive or Raw Image(s)';
    }
    if (dataType === 'audio') {
      return 'Choose ZIP Archive or Raw Sound(s)';
    }
    if (dataType === 'text') {
      return 'Choose CSV, Parquet, Excel, or ZIP Archive';
    }
    return 'Choose Tabular Dataset';
  };

  const renderGuidance = () => {
    if (dataType === 'text') {
      return (
        <div className="card">
          <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Preparing Your Text Data</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.6', marginBottom: '1.25rem' }}>
            InstaML supports two convenient structures for text classification datasets:
          </p>
          <div style={{ marginBottom: '1.25rem' }}>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-main)', marginBottom: '0.25rem', fontWeight: 600 }}>Option A: Labeled Tabular File</h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.5' }}>
              Upload a single <code>.csv</code>, <code>.xlsx</code>, or <code>.parquet</code> table. Ensure it has at least one column for the raw text and one column for the target labels.
            </p>
          </div>
          <div>
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-main)', marginBottom: '0.25rem', fontWeight: 600 }}>Option B: Folder-Based ZIP Archive</h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.5' }}>
              Upload a <code>.zip</code> file containing subdirectories named after your classification classes (e.g. <code>positive/</code> and <code>negative/</code>). Place individual <code>.txt</code> documents inside.
            </p>
          </div>
        </div>
      );
    }
    if (isImageOrAudio) {
      return (
        <div className="card">
          <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Preparing Your ZIP Archive</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.6', marginBottom: '1.25rem' }}>
            To train {dataType} models, InstaML expects a standard dataset format where subfolder names correspond to classification classes.
          </p>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'start', fontSize: '0.8rem' }}>
              <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)', marginTop: '0.15rem' }}></i>
              <span>Create separate subfolders for each class category (e.g., <code>class_A/</code>, <code>class_B/</code>).</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'start', fontSize: '0.8rem' }}>
              <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)', marginTop: '0.15rem' }}></i>
              <span>Put raw {dataType} files inside their respective folders.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'start', fontSize: '0.8rem' }}>
              <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)', marginTop: '0.15rem' }}></i>
              <span>Compress the parent folder as a single <code>.zip</code> file.</span>
            </li>
          </ul>
        </div>
      );
    }
    return (
      <div className="card">
        <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>Preparing Your Dataset</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', lineHeight: '1.6', marginBottom: '1.25rem' }}>
          Machine learning models require clean, well-structured datasets. InstaML parses, indexes, and caches your tables automatically.
        </p>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.8rem' }}>
            <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)' }}></i>
            <span>Ensure columns contain clear, unique headers.</span>
          </li>
          <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.8rem' }}>
            <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)' }}></i>
            <span>Identify your target label variable to predict.</span>
          </li>
          <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.8rem' }}>
            <i className="fa-solid fa-circle-check" style={{ color: 'var(--text-muted)' }}></i>
            <span>Keep datasets under 100MB for optimal browser speed.</span>
          </li>
        </ul>
      </div>
    );
  };

  return (
    <div>
      <div className="header-bar" style={{ marginBottom: '2.5rem' }}>
        <div className="page-title-section">
          <h1 className="page-title">Data Upload</h1>
          <p className="page-subtitle">{getPageSubtitle()}</p>
        </div>
        <div>
          <button 
            onClick={() => setGuideOpen(true)} 
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
          >
            <i className="fa-solid fa-graduation-cap"></i>
            <span>Get to know more</span>
          </button>
        </div>
      </div>

      {project && (
        <div className="card" style={{ marginBottom: '2rem', padding: '1.25rem 1.5rem', background: 'var(--bg-glass)', backdropFilter: 'blur(8px)', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
              <div style={{ padding: '0.45rem', borderRadius: '10px', backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--accent-primary)', display: 'flex' }}>
                <i className={dataType === 'image' ? 'fa-solid fa-image' : (dataType === 'audio' ? 'fa-solid fa-music' : (dataType === 'text' ? 'fa-solid fa-file-lines' : 'fa-solid fa-table'))} style={{ fontSize: '0.9rem' }}></i>
              </div>
              <div>
                <h4 style={{ margin: 0, fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  {dataType.toUpperCase()} Workspace modality
                </h4>
                <p style={{ margin: '0.15rem 0 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Active Model Target Task: <strong style={{ color: 'var(--accent-purple)' }}>{project.task || 'Classification'}</strong>
                </p>
              </div>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Change Task:</span>
              {taskUpdating && <span className="spinner" style={{ width: '14px', height: '14px', border: '2px solid var(--accent-purple)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></span>}
              <select
                className="form-control"
                style={{ width: '240px', padding: '0.35rem 0.65rem', fontSize: '0.8rem', cursor: 'pointer', height: 'auto' }}
                value={project.task || 'Classification'}
                disabled={taskUpdating}
                onChange={async (e) => {
                  const newTask = e.target.value;
                  setTaskUpdating(true);
                  try {
                    await api.patch(`/projects/${project.id}`, { task: newTask });
                    toast.success(`Project task updated to: ${newTask}`);
                    await refreshStatus();
                  } catch (err) {
                    toast.error("Failed to update project task.");
                  } finally {
                    setTaskUpdating(false);
                  }
                }}
              >
                {getTasksForDataType(dataType).map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Upload Box */}
        <div className="card">
          <h3 className="card-title" style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.25rem' }}>Select Data Source</h3>
          <p className="card-subtitle" style={{ marginBottom: '1.5rem' }}>{getSubtitle()}</p>




          {dataType === 'text' && (
            <div className="tab-container" style={{ marginBottom: '1.5rem' }}>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'file' ? 'active' : ''}`}
                onClick={() => { setActiveTab('file'); setError(''); setSuccess(''); }}
                onClick={() => { setActiveTab('file'); }}
              >
                Upload File / ZIP
              </button>
              <button
                type="button"
                className={`tab-btn ${activeTab === 'paste' ? 'active' : ''}`}
                onClick={() => { setActiveTab('paste'); }}
              >
                Paste Text Lines
              </button>
            </div>
          )}

          {activeTab === 'file' ? (
            <form onSubmit={handleUpload}>
              <div className="uploader-box" onClick={() => document.getElementById('file-upload-input').click()}>
                <i className={`${getUploaderIcon()} uploader-icon`} style={{ color: file ? 'var(--text-main)' : 'var(--text-dim)', fontSize: '2rem', marginBottom: '0.75rem', display: 'block' }}></i>
                <h4 style={{ marginBottom: '0.25rem', fontSize: '0.9rem', fontWeight: 600 }}>{file ? file.name : getUploaderText()}</h4>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                  {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` : "Drag and drop or browse files"}
                </p>
                <input
                  id="file-upload-input"
                  type="file"
                  style={{ display: 'none' }}
                  accept={acceptString}
                  onChange={handleFileChange}
                />
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                <button
                  type="submit"
                  className="btn btn-primary"
                  style={{ flex: 1 }}
                  disabled={loading || !file}
                >
                  {loading ? <span className="spinner"></span> : <span>Upload Dataset</span>}
                </button>
                {file && (
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    className="btn btn-secondary"
                  >
                    Clear
                  </button>
                )}
              </div>
            </form>
          ) : (
            <form onSubmit={handleUploadPastedText}>
              <div className="form-group">
                <label className="form-label" style={{ marginBottom: '0.5rem' }}>Pasted Text Content</label>
                <textarea
                  className="form-control"
                  style={{ height: '180px', resize: 'none', fontFamily: 'var(--font-body)', fontSize: '0.85rem', lineHeight: '1.5' }}
                  placeholder="Paste raw text sentences here, with one document or sample per line..."
                  value={pastedText}
                  onChange={(e) => setPastedText(e.target.value)}
                  disabled={loading}
                />
              </div>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                <button
                  type="submit"
                  className="btn btn-primary"
                  style={{ flex: 1 }}
                  disabled={loading || !pastedText.trim()}
                >
                  {loading ? <span className="spinner"></span> : <span>Upload Pasted Text</span>}
                </button>
                {pastedText && (
                  <button
                    type="button"
                    onClick={() => setPastedText('')}
                    className="btn btn-secondary"
                    disabled={loading}
                  >
                    Clear
                  </button>
                )}
              </div>
            </form>
          )}
        </div>

        {/* Informative Guidance */}
        {renderGuidance()}
      </div>

      {renderCurrentDataInfo()}

      <GuideDrawer isOpen={guideOpen} onClose={() => setGuideOpen(false)} initialTopic="data_upload" />
    </div>
  );
};

export default DataUpload;
