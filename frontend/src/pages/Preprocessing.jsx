import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';
import GuideDrawer from '../components/GuideDrawer';

const BoxPlot = ({ stats }) => {
  if (!stats) return null;
  const { min, max, q1, median, q3, lower_whisker, upper_whisker, outliers, total_outliers } = stats;
  
  // Map values to a 0-100% scale for SVG positions
  const range = max - min || 1;
  const scale = (val) => 100 - ((val - min) / range) * 100; // inverted: higher val -> top
  
  const yMin = scale(min);
  const yMax = scale(max);
  const yQ1 = scale(q1);
  const yMedian = scale(median);
  const yQ3 = scale(q3);
  const yLower = scale(lower_whisker);
  const yUpper = scale(upper_whisker);
  
  // Color palette
  const colors = {
    line: 'var(--text-muted)',
    boxStroke: 'var(--accent-primary)',
    boxFill: 'rgba(59, 130, 246, 0.15)',
    median: 'var(--accent-purple)',
    outliers: 'var(--accent-red)'
  };
  
  return (
    <div style={{ display: 'flex', gap: '2rem', alignItems: 'center', margin: '1.5rem 0', padding: '1.25rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', flexWrap: 'wrap' }}>
      {/* Boxplot SVG */}
      <div style={{ position: 'relative', width: '100px', height: '240px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'center', padding: '10px', overflow: 'visible' }}>
        <svg width="80" height="220" viewBox="0 0 80 100" style={{ overflow: 'visible' }}>
          {/* Whiskers line */}
          <line x1="40" y1={yLower} x2="40" y2={yUpper} stroke={colors.line} strokeWidth="1.5" strokeDasharray="3 3" />
          
          {/* Lower whisker cap */}
          <line x1="30" y1={yLower} x2="50" y2={yLower} stroke={colors.line} strokeWidth="1.5" />
          
          {/* Upper whisker cap */}
          <line x1="30" y1={yUpper} x2="50" y2={yUpper} stroke={colors.line} strokeWidth="1.5" />
          
          {/* IQR Box */}
          <rect x="20" y={Math.min(yQ1, yQ3)} width="40" height={Math.abs(yQ1 - yQ3)} fill={colors.boxFill} stroke={colors.boxStroke} strokeWidth="2" rx="2" />
          
          {/* Median line */}
          <line x1="20" y1={yMedian} x2="60" y2={yMedian} stroke={colors.median} strokeWidth="3" />
          
          {/* Outliers */}
          {outliers.map((val, idx) => (
            <circle
              key={idx}
              cx={40 + (idx % 2 === 0 ? 1 : -1) * (4 + (idx % 3) * 2)} // horizontal jitter
              cy={scale(val)}
              r="2.5"
              fill={colors.outliers}
              opacity="0.65"
            />
          ))}
        </svg>
      </div>
      
      {/* Boxplot statistics text */}
      <div style={{ flex: 1, minWidth: '220px', display: 'flex', flexDirection: 'column', gap: '0.45rem', fontSize: '0.85rem' }}>
        <h4 style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-main)' }}>
          Outlier Distribution Statistics ({stats.column})
        </h4>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.2rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Upper Whisker limit</span>
          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{upper_whisker.toFixed(4)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.2rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Q3 (75th percentile)</span>
          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{q3.toFixed(4)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.2rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Median (50th percentile)</span>
          <span style={{ fontWeight: 600, color: 'var(--accent-purple)' }}>{median.toFixed(4)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.2rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Q1 (25th percentile)</span>
          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{q1.toFixed(4)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.2rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Lower Whisker limit</span>
          <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{lower_whisker.toFixed(4)}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px dashed var(--border-color)' }}>
          <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Detected Outliers:</span>
          <span style={{ fontWeight: 700, color: 'var(--accent-red)' }}>{total_outliers} rows ({((total_outliers / (stats.total_rows || 1)) * 100).toFixed(1)}%)</span>
        </div>
      </div>
    </div>
  );
};

const Preprocessing = ({ datasetStatus, refreshStatus }) => {
  const [activeTab, setActiveTab] = useState('columns');
  const [loading, setLoading] = useState(false);

  // Drop Columns state
  const [colsToKeep, setColsToKeep] = useState(datasetStatus?.columns || []);
  
  // Impute state
  const [imputeCol, setImputeCol] = useState(datasetStatus?.columns?.[0] || '');
  const [imputeStrategy, setImputeStrategy] = useState('mean');

  // Scale state
  const [scaleCols, setScaleCols] = useState([]);
  const [scaleMethod, setScaleMethod] = useState('standard');

  // Encode state
  const [encodeCols, setEncodeCols] = useState([]);
  const [encodeMethod, setEncodeMethod] = useState('onehot');

  // Outlier state
  const [outlierMethod, setOutlierMethod] = useState('iqr');
  const [outlierCol, setOutlierCol] = useState(
    datasetStatus?.columns?.find(c => datasetStatus.dtypes?.[c]?.includes('int') || datasetStatus.dtypes?.[c]?.includes('float')) || ''
  );
  const [zscoreThreshold, setZscoreThreshold] = useState(3.0);
  const [contamination, setContamination] = useState(0.05);
  const [outlierCols, setOutlierCols] = useState([]);

  // PCA state
  const [pcaCols, setPcaCols] = useState([]);
  const [pcaNumComponents, setPcaNumComponents] = useState(2);

  // Boxplot state
  const [boxplotStats, setBoxplotStats] = useState(null);
  const [boxplotLoading, setBoxplotLoading] = useState(false);

  // Preview state
  const [previewRows, setPreviewRows] = useState(10);
  const [previewData, setPreviewData] = useState([]);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Guide state
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideTopic, setGuideTopic] = useState('duplicates');

  const openGuide = (topic) => {
    setGuideTopic(topic);
    setGuideOpen(true);
  };

  // Sync columns if dataset status changes
  useEffect(() => {
    if (datasetStatus?.columns) {
      setColsToKeep(datasetStatus.columns);
      if (!imputeCol) setImputeCol(datasetStatus.columns[0]);
      if (!outlierCol) {
        const firstNumeric = datasetStatus.columns.find(c => {
          const dtype = datasetStatus.dtypes?.[c]?.toLowerCase() || '';
          return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('number');
        });
        setOutlierCol(firstNumeric || '');
      }
    }
  }, [datasetStatus]);

  // Fetch preview when dataset status or row count changes
  const fetchPreview = async () => {
    if (!datasetStatus?.project_id) return;
    setPreviewLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/data/current`, {
        params: { preview_rows: previewRows }
      });
      setPreviewData(res.data.preview || []);
    } catch (err) {
      console.error("Failed to load preview data", err);
    } finally {
      setPreviewLoading(false);
    }
  };

  useEffect(() => {
    fetchPreview();
  }, [datasetStatus?.version_id, previewRows]);

  // Fetch boxplot statistics for the selected outlier column
  const fetchBoxplotStats = async () => {
    if (!outlierCol || !datasetStatus?.project_id) {
      setBoxplotStats(null);
      return;
    }
    setBoxplotLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/eda/boxplot/${outlierCol}`);
      setBoxplotStats(res.data);
    } catch (err) {
      console.error("Failed to load boxplot stats", err);
      setBoxplotStats(null);
    } finally {
      setBoxplotLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'outliers' && (outlierMethod === 'iqr' || outlierMethod === 'zscore')) {
      fetchBoxplotStats();
    } else {
      setBoxplotStats(null);
    }
  }, [outlierCol, outlierMethod, activeTab, datasetStatus?.version_id]);

  const getNumericColumns = () => {
    if (!datasetStatus?.columns) return [];
    return datasetStatus.columns.filter(c => {
      const dtype = datasetStatus.dtypes?.[c]?.toLowerCase() || '';
      return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('number');
    });
  };

  const getCategoricalColumns = () => {
    if (!datasetStatus?.columns) return [];
    const numCols = getNumericColumns();
    return datasetStatus.columns.filter(col => !numCols.includes(col));
  };

  const applyPreprocessing = async (operations) => {
    setLoading(true);
    try {
      await api.post(`/projects/${datasetStatus.project_id}/preprocess`, { operations });
      toast.success("Preprocessing operations applied successfully!");
      refreshStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Preprocessing failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleDropColumns = () => {
    const dropped = datasetStatus.columns.filter(c => !colsToKeep.includes(c));
    if (dropped.length === 0) {
      toast.error("No columns selected for deletion.");
      return;
    }
    applyPreprocessing([{ op: "drop_cols", columns: dropped }]);
  };

  const handleDropDuplicates = () => {
    applyPreprocessing([{ op: "drop_duplicates" }]);
  };

  const handleImpute = () => {
    if (!imputeCol) {
      toast.error("Please select a column to impute.");
      return;
    }
    applyPreprocessing([{ op: "fill_missing", strategy: imputeStrategy, columns: [imputeCol] }]);
  };

  const handleScale = () => {
    if (scaleCols.length === 0) {
      toast.error("Please select at least one numeric column to scale.");
      return;
    }
    applyPreprocessing([{ op: "scale", method: scaleMethod, columns: scaleCols }]);
  };

  const handleEncode = () => {
    if (encodeCols.length === 0) {
      toast.error("Please select at least one column to encode.");
      return;
    }
    applyPreprocessing([{ op: "encode", method: encodeMethod, columns: encodeCols }]);
  };

  const handleRemoveOutliers = () => {
    if (outlierMethod === 'iqr') {
      if (!outlierCol) {
        toast.error("Please select a numeric column for IQR outlier cleaning.");
        return;
      }
      applyPreprocessing([{ op: "remove_outliers", method: "iqr", column: outlierCol }]);
    } else if (outlierMethod === 'zscore') {
      if (!outlierCol) {
        toast.error("Please select a numeric column for Z-Score outlier cleaning.");
        return;
      }
      applyPreprocessing([{ op: "remove_outliers", method: "zscore", column: outlierCol, threshold: parseFloat(zscoreThreshold) || 3.0 }]);
    } else if (outlierMethod === 'isolation_forest' || outlierMethod === 'lof') {
      applyPreprocessing([{
        op: "remove_outliers",
        method: outlierMethod,
        columns: outlierCols,
        contamination: parseFloat(contamination) || 0.05
      }]);
    }
  };

  const toggleColumnToKeep = (col) => {
    if (colsToKeep.includes(col)) {
      if (colsToKeep.length === 1) {
        toast.error("You must keep at least one column.");
        return;
      }
      setColsToKeep(colsToKeep.filter(c => c !== col));
    } else {
      setColsToKeep([...colsToKeep, col]);
    }
  };

  const toggleScaleColumn = (col) => {
    if (scaleCols.includes(col)) {
      setScaleCols(scaleCols.filter(c => c !== col));
    } else {
      setScaleCols([...scaleCols, col]);
    }
  };

  const toggleEncodeColumn = (col) => {
    if (encodeCols.includes(col)) {
      setEncodeCols(encodeCols.filter(c => c !== col));
    } else {
      setEncodeCols([...encodeCols, col]);
    }
  };

  const toggleOutlierColumn = (col) => {
    if (outlierCols.includes(col)) {
      setOutlierCols(outlierCols.filter(c => c !== col));
    } else {
      setOutlierCols([...outlierCols, col]);
    }
  };

  const togglePcaColumn = (col) => {
    if (pcaCols.includes(col)) {
      setPcaCols(pcaCols.filter(c => c !== col));
    } else {
      setPcaCols([...pcaCols, col]);
    }
  };

  const handleApplyPca = () => {
    if (pcaCols.length < pcaNumComponents) {
      toast.error(`Please select at least ${pcaNumComponents} columns for PCA.`);
      return;
    }
    applyPreprocessing([{ op: "pca", columns: pcaCols, n_components: pcaNumComponents }]);
  };

  if (!datasetStatus?.data_loaded) return null;

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Data Preprocessing</h1>
          <p className="page-subtitle">Clean, scale, encode, and format your dataset for training</p>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Active: <strong style={{ color: 'var(--text-main)' }}>{datasetStatus.version_id}</strong> ({datasetStatus.shape?.[0]} x {datasetStatus.shape?.[1]})
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="tab-container" style={{ marginBottom: '2rem' }}>
        <button className={`tab-btn ${activeTab === 'columns' ? 'active' : ''}`} onClick={() => setActiveTab('columns')}>
          <i className="fa-solid fa-table-columns" style={{ marginRight: '0.5rem' }}></i>
          Columns & Duplicates
        </button>
        <button className={`tab-btn ${activeTab === 'missing' ? 'active' : ''}`} onClick={() => setActiveTab('missing')}>
          <i className="fa-solid fa-circle-question" style={{ marginRight: '0.5rem' }}></i>
          Missing Values
        </button>
        <button className={`tab-btn ${activeTab === 'scaling' ? 'active' : ''}`} onClick={() => setActiveTab('scaling')}>
          <i className="fa-solid fa-arrows-up-down-left-right" style={{ marginRight: '0.5rem' }}></i>
          Scaling & Encoding
        </button>
        <button className={`tab-btn ${activeTab === 'outliers' ? 'active' : ''}`} onClick={() => setActiveTab('outliers')}>
          <i className="fa-solid fa-circle-nodes" style={{ marginRight: '0.5rem' }}></i>
          Outliers Clean
        </button>
        <button className={`tab-btn ${activeTab === 'pca' ? 'active' : ''}`} onClick={() => setActiveTab('pca')}>
          <i className="fa-solid fa-calculator" style={{ marginRight: '0.5rem' }}></i>
          PCA / Dim Reduction
        </button>
      </div>

      <div className="card" style={{ minHeight: '300px', position: 'relative' }}>
        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}

        {/* Tab 1: Columns & Duplicates */}
        {activeTab === 'columns' && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Select Columns to Retain</h3>
              <button onClick={() => openGuide('duplicates')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                <i className="fa-solid fa-graduation-cap"></i>
                <span>Get to know more</span>
              </button>
            </div>
            <p className="card-subtitle">Uncheck columns you do not want to use in training, then click apply to remove them.</p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.75rem', margin: '1.5rem 0 2.5rem' }}>
              {datasetStatus.columns?.map((col) => (
                <label
                  key={col}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '0.75rem 1rem',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid ' + (colsToKeep.includes(col) ? 'var(--border-focus)' : 'var(--border-color)'),
                    backgroundColor: colsToKeep.includes(col) ? 'var(--bg-active)' : 'transparent',
                    cursor: 'pointer',
                    userSelect: 'none',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <input
                    type="checkbox"
                    checked={colsToKeep.includes(col)}
                    onChange={() => toggleColumnToKeep(col)}
                    style={{ cursor: 'pointer', accentColor: 'var(--text-main)' }}
                  />
                  <span style={{ fontSize: '0.85rem', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', fontWeight: colsToKeep.includes(col) ? '600' : '500', color: colsToKeep.includes(col) ? 'var(--text-main)' : 'var(--text-muted)' }}>
                    {col}
                  </span>
                </label>
              ))}
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1.5rem', marginBottom: '1.5rem' }}>
              <button onClick={handleDropColumns} className="btn btn-primary">
                Apply Column Filter
              </button>
              <button
                onClick={() => setColsToKeep(datasetStatus.columns || [])}
                className="btn btn-secondary"
              >
                Reset Selection
              </button>
            </div>

            {/* Handle Duplicates */}
            <div>
              <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem', fontWeight: 600 }}>Handle Duplicate Rows</h3>
              <p className="card-subtitle" style={{ margin: 0 }}>
                Your dataset currently has <strong style={{ color: datasetStatus.duplicate_count > 0 ? 'var(--accent-red)' : 'var(--text-main)' }}>{datasetStatus.duplicate_count}</strong> duplicate rows.
              </p>
              <button
                onClick={handleDropDuplicates}
                className="btn btn-primary"
                disabled={datasetStatus.duplicate_count === 0}
                style={{ marginTop: '1rem' }}
              >
                Drop Duplicate Rows
              </button>
            </div>
          </div>
        )}

        {/* Tab 2: Missing Values */}
        {activeTab === 'missing' && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Impute Missing Values</h3>
              <button onClick={() => openGuide('imputation')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                <i className="fa-solid fa-graduation-cap"></i>
                <span>Get to know more</span>
              </button>
            </div>
            <p className="card-subtitle">Select a column containing missing values and choose an imputation strategy.</p>

            {/* Missing Values Overview */}
            <div style={{ marginBottom: '1.5rem', backgroundColor: 'var(--bg-secondary)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-main)' }}>Missing Values Overview</h4>
              {Object.keys(datasetStatus.missing_counts || {}).filter(col => datasetStatus.missing_counts[col] > 0).length === 0 ? (
                <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--accent-green)', fontWeight: 500 }}>
                  <i className="fa-solid fa-circle-check" style={{ marginRight: '0.5rem' }}></i> No missing values detected in the active dataset!
                </p>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.5rem' }}>
                  {Object.keys(datasetStatus.missing_counts || {}).map(col => {
                    const count = datasetStatus.missing_counts[col];
                    if (count > 0) {
                      return (
                        <div key={col} style={{ padding: '0.35rem 0.65rem', borderRadius: '6px', backgroundColor: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)', fontSize: '0.75rem', color: 'var(--accent-yellow)', fontWeight: 600 }}>
                          {col}: {count} nulls ({((count / datasetStatus.shape[0]) * 100).toFixed(1)}%)
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              )}
            </div>

            <div className="grid-2" style={{ margin: '1.5rem 0 2.5rem' }}>
              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Target Column</label>
                <select className="form-control" value={imputeCol} onChange={(e) => setImputeCol(e.target.value)} style={{ cursor: 'pointer' }}>
                  <option value="">-- Select Column --</option>
                  {datasetStatus.columns?.map((col) => {
                    const missingCount = datasetStatus.missing_counts?.[col] || 0;
                    return (
                      <option key={col} value={col}>
                        {col} {missingCount > 0 ? `(${missingCount} missing)` : '(No missing)'}
                      </option>
                    );
                  })}
                </select>
              </div>

              <div className="form-group" style={{ margin: 0 }}>
                <label className="form-label">Imputation Method</label>
                <select className="form-control" value={imputeStrategy} onChange={(e) => setImputeStrategy(e.target.value)} style={{ cursor: 'pointer' }}>
                  <option value="mean">Mean (Average) — Numeric only</option>
                  <option value="median">Median (Middle value) — Numeric only</option>
                  <option value="most_frequent">Most Frequent (Mode) — All types</option>
                  <option value="constant_zero">Constant Value (0 or "Missing")</option>
                  <option value="drop">Drop Rows with Missing Data</option>
                </select>
              </div>
            </div>

            <button onClick={handleImpute} className="btn btn-primary">
              Impute Column values
            </button>
          </div>
        )}

        {/* Tab 3: Scaling & Encoding */}
        {activeTab === 'scaling' && (
          <div className="grid-2" style={{ alignItems: 'start' }}>
            {/* Numeric Feature Scaling */}
            <div style={{ borderRight: '1px solid var(--border-color)', paddingRight: '2.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Feature Scaling</h3>
                <button onClick={() => openGuide('scaling')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                  <i className="fa-solid fa-graduation-cap"></i>
                  <span>Get to know more</span>
                </button>
              </div>
              <p className="card-subtitle">Normalize or standardize numerical columns to the same scale range.</p>

              <div className="form-group">
                <label className="form-label">Scaling Method</label>
                <select className="form-control" value={scaleMethod} onChange={(e) => setScaleMethod(e.target.value)} style={{ cursor: 'pointer' }}>
                  <option value="standard">StandardScaler (Mean=0, Var=1)</option>
                  <option value="minmax">MinMaxScaler (Range [0, 1])</option>
                  <option value="robust">RobustScaler (Handles outliers)</option>
                  <option value="maxabs">MaxAbsScaler (Range [-1, 1])</option>
                  <option value="normalizer">Normalizer (Norms samples to unit norm)</option>
                  <option value="quantile">QuantileTransformer (Uniform/Gaussian mapping)</option>
                  <option value="power">PowerTransformer (Yeo-Johnson power mapping)</option>
                </select>
              </div>

              <label className="form-label" style={{ marginTop: '1.5rem' }}>Select Columns to Scale</label>
              <div style={{ maxHeight: '180px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {getNumericColumns().map(col => (
                  <label key={col} style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', padding: '0.25rem 0.5rem', cursor: 'pointer', fontSize: '0.85rem', userSelect: 'none', borderRadius: '4px', border: '1px solid ' + (scaleCols.includes(col) ? 'var(--border-focus)' : 'transparent'), backgroundColor: scaleCols.includes(col) ? 'var(--bg-active)' : 'transparent' }}>
                    <input type="checkbox" checked={scaleCols.includes(col)} onChange={() => toggleScaleColumn(col)} style={{ cursor: 'pointer', accentColor: 'var(--text-main)' }} />
                    <span style={{ fontWeight: scaleCols.includes(col) ? '600' : '500', color: scaleCols.includes(col) ? 'var(--text-main)' : 'var(--text-muted)' }}>{col}</span>
                  </label>
                ))}
              </div>

              <button onClick={handleScale} className="btn btn-primary" style={{ width: '100%' }}>
                Apply Scaling
              </button>
            </div>

            {/* Categorical Encoding */}
            <div style={{ paddingLeft: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Categorical Feature Encoding</h3>
                <button onClick={() => openGuide('encoding')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                  <i className="fa-solid fa-graduation-cap"></i>
                  <span>Get to know more</span>
                </button>
              </div>
              <p className="card-subtitle">Convert categorical variables into numerical values.</p>

              <div className="form-group">
                <label className="form-label">Encoding Method</label>
                <select className="form-control" value={encodeMethod} onChange={(e) => setEncodeMethod(e.target.value)} style={{ cursor: 'pointer' }}>
                  <option value="onehot">One-Hot Encoding (Dummies)</option>
                  <option value="label">Label Encoding (Ordered integers)</option>
                </select>
              </div>

              <label className="form-label" style={{ marginTop: '1.5rem' }}>Select Columns to Encode</label>
              <div style={{ maxHeight: '180px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {getCategoricalColumns().map(col => (
                  <label key={col} style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', padding: '0.25rem 0.5rem', cursor: 'pointer', fontSize: '0.85rem', userSelect: 'none', borderRadius: '4px', border: '1px solid ' + (encodeCols.includes(col) ? 'var(--border-focus)' : 'transparent'), backgroundColor: encodeCols.includes(col) ? 'var(--bg-active)' : 'transparent' }}>
                    <input type="checkbox" checked={encodeCols.includes(col)} onChange={() => toggleEncodeColumn(col)} style={{ cursor: 'pointer', accentColor: 'var(--text-main)' }} />
                    <span style={{ fontWeight: encodeCols.includes(col) ? '600' : '500', color: encodeCols.includes(col) ? 'var(--text-main)' : 'var(--text-muted)' }}>{col}</span>
                  </label>
                ))}
              </div>

              <button onClick={handleEncode} className="btn btn-primary" style={{ width: '100%' }}>
                Apply Encoding
              </button>
            </div>
          </div>
        )}

        {/* Tab 4: Outliers */}
        {activeTab === 'outliers' && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Filter Outliers</h3>
              <button onClick={() => openGuide('outliers')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                <i className="fa-solid fa-graduation-cap"></i>
                <span>Get to know more</span>
              </button>
            </div>
            <p className="card-subtitle">Identify and drop extreme outliers using IQR, Z-Score, Isolation Forest, or LOF.</p>

            <div className="grid-2" style={{ margin: '1.5rem 0 2.5rem', alignItems: 'start' }}>
              <div>
                <div className="form-group">
                  <label className="form-label">Outlier Detection Method</label>
                  <select
                    className="form-control"
                    value={outlierMethod}
                    onChange={(e) => setOutlierMethod(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    <option value="iqr">IQR Rule (1.5 * IQR, column-specific)</option>
                    <option value="zscore">Z-Score Method (column-specific)</option>
                    <option value="isolation_forest">Isolation Forest (Multivariate, all/selected columns)</option>
                    <option value="lof">Local Outlier Factor (LOF, density-based)</option>
                  </select>
                </div>

                {/* Conditional Inputs */}
                {(outlierMethod === 'iqr' || outlierMethod === 'zscore') && (
                  <div className="form-group" style={{ marginTop: '1rem' }}>
                    <label className="form-label">Target Column</label>
                    <select
                      className="form-control"
                      value={outlierCol}
                      onChange={(e) => setOutlierCol(e.target.value)}
                      style={{ cursor: 'pointer' }}
                    >
                      <option value="">-- Choose Numeric Column --</option>
                      {getNumericColumns().map((col) => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                )}

                {outlierMethod === 'zscore' && (
                  <div className="form-group" style={{ marginTop: '1rem' }}>
                    <label className="form-label">Z-Score Threshold (e.g. 3.0)</label>
                    <input
                      type="number"
                      step="0.1"
                      className="form-control"
                      value={zscoreThreshold}
                      onChange={(e) => setZscoreThreshold(parseFloat(e.target.value) || '')}
                    />
                  </div>
                )}

                {(outlierMethod === 'isolation_forest' || outlierMethod === 'lof') && (
                  <div className="form-group" style={{ marginTop: '1rem' }}>
                    <label className="form-label">Contamination Ratio (0.01 to 0.5, default 0.05)</label>
                    <input
                      type="number"
                      step="0.01"
                      className="form-control"
                      value={contamination}
                      onChange={(e) => setContamination(parseFloat(e.target.value) || '')}
                    />
                  </div>
                )}
              </div>

              {(outlierMethod === 'isolation_forest' || outlierMethod === 'lof') && (
                <div>
                  <label className="form-label">Select Columns for Outlier Detection (Optional)</label>
                  <p className="card-subtitle" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
                    If no columns are selected, all numeric columns will be used.
                  </p>
                  <div style={{ maxHeight: '180px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {getNumericColumns().map(col => (
                      <label key={col} style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', padding: '0.25rem 0.5rem', cursor: 'pointer', fontSize: '0.85rem', userSelect: 'none', borderRadius: '4px', border: '1px solid ' + (outlierCols.includes(col) ? 'var(--border-focus)' : 'transparent'), backgroundColor: outlierCols.includes(col) ? 'var(--bg-active)' : 'transparent' }}>
                        <input type="checkbox" checked={outlierCols.includes(col)} onChange={() => toggleOutlierColumn(col)} style={{ cursor: 'pointer', accentColor: 'var(--text-main)' }} />
                        <span style={{ fontWeight: outlierCols.includes(col) ? '600' : '500', color: outlierCols.includes(col) ? 'var(--text-main)' : 'var(--text-muted)' }}>{col}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Dynamic Boxplot Visualization */}
            {boxplotLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
                <div className="spinner"></div>
              </div>
            ) : boxplotStats ? (
              <BoxPlot stats={boxplotStats} />
            ) : null}

            <button onClick={handleRemoveOutliers} className="btn btn-primary" style={{ marginTop: '1rem' }}>
              Filter Outlier Rows
            </button>
          </div>
        )}

        {/* Tab 5: PCA */}
        {activeTab === 'pca' && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Principal Component Analysis (PCA)</h3>
              <button onClick={() => openGuide('pca')} className="btn btn-secondary" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.25rem 0.5rem', fontSize: '0.7rem', height: 'auto', borderRadius: '4px' }}>
                <i className="fa-solid fa-graduation-cap"></i>
                <span>Get to know more</span>
              </button>
            </div>
            <p className="card-subtitle">Apply dimensionality reduction to map selected numerical features onto lower-dimensional principal components.</p>

            <div className="grid-2" style={{ margin: '1.5rem 0 2.5rem', alignItems: 'start' }}>
              <div>
                <div className="form-group">
                  <label className="form-label">Number of Principal Components</label>
                  <input
                    type="number"
                    min="1"
                    max={pcaCols.length || 10}
                    className="form-control"
                    value={pcaNumComponents}
                    onChange={(e) => setPcaNumComponents(Math.max(1, Number(e.target.value)))}
                  />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem', display: 'block' }}>
                    Must be less than or equal to the number of selected columns ({pcaCols.length}).
                  </span>
                </div>
              </div>

              <div>
                <label className="form-label">Select Numerical Columns to Reduce</label>
                <div style={{ maxHeight: '200px', overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {getNumericColumns().map(col => (
                    <label key={col} style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', padding: '0.25rem 0.5rem', cursor: 'pointer', fontSize: '0.85rem', userSelect: 'none', borderRadius: '4px', border: '1px solid ' + (pcaCols.includes(col) ? 'var(--border-focus)' : 'transparent'), backgroundColor: pcaCols.includes(col) ? 'var(--bg-active)' : 'transparent' }}>
                      <input type="checkbox" checked={pcaCols.includes(col)} onChange={() => togglePcaColumn(col)} style={{ cursor: 'pointer', accentColor: 'var(--text-main)' }} />
                      <span style={{ fontWeight: pcaCols.includes(col) ? '600' : '500', color: pcaCols.includes(col) ? 'var(--text-main)' : 'var(--text-muted)' }}>{col}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <button onClick={handleApplyPca} className="btn btn-primary" style={{ marginTop: '1rem' }}>
              Apply PCA Dimensionality Reduction
            </button>
          </div>
        )}
      </div>

      {/* Preview Section */}
      <div className="card" style={{ marginTop: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>Dataset Preview</h3>
            <p className="card-subtitle" style={{ margin: 0 }}>Inspect the dataset values after preprocessing steps</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Show:</span>
            <select
              className="form-control"
              value={previewRows}
              onChange={(e) => setPreviewRows(Number(e.target.value))}
              style={{ width: '80px', height: '35px', padding: '0 0.5rem', cursor: 'pointer', margin: 0 }}
            >
              <option value="5">5</option>
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
            </select>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>rows</span>
          </div>
        </div>

        {previewLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
            <div className="spinner"></div>
          </div>
        ) : previewData.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
            No preview data available.
          </div>
        ) : (
          <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}>
            <table className="table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)', backgroundColor: 'var(--bg-active)' }}>
                  {datasetStatus.columns?.map(col => (
                    <th key={col} style={{ padding: '0.75rem 1rem', fontSize: '0.8rem', fontWeight: '700', color: 'var(--text-main)', borderBottom: '1px solid var(--border-color)' }}>
                      <div>{col}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'normal' }}>
                        {datasetStatus.dtypes?.[col] || 'unknown'}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewData.map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: idx < previewData.length - 1 ? '1px solid var(--border-color)' : 'none' }}>
                    {datasetStatus.columns?.map(col => (
                      <td key={col} style={{ padding: '0.75rem 1rem', fontSize: '0.85rem', color: 'var(--text-muted)', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden', maxWidth: '200px' }}>
                        {row[col] !== undefined && row[col] !== null ? String(row[col]) : <em style={{ color: 'var(--text-dim)' }}>NaN</em>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <GuideDrawer isOpen={guideOpen} onClose={() => setGuideOpen(false)} initialTopic={guideTopic} />
    </div>
  );
};

export default Preprocessing;
