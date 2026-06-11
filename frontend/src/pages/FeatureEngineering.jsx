import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';

const FeatureEngineering = ({ datasetStatus, refreshStatus }) => {
  const [activeTab, setActiveTab] = useState('engineering'); // 'engineering' or 'selection'
  const [loading, setLoading] = useState(false);

  // Feature Engineering state
  const [feType, setFeType] = useState('math_op'); // 'math_op', 'math_transform', 'binning'
  const [feCol1, setFeCol1] = useState('');
  const [feCol2, setFeCol2] = useState('');
  const [feOperator, setFeOperator] = useState('+');
  const [feTransformCol, setFeTransformCol] = useState('');
  const [feTransform, setFeTransform] = useState('log');
  const [feBinCol, setFeBinCol] = useState('');
  const [feBinNum, setFeBinNum] = useState(4);
  const [feNewCol, setFeNewCol] = useState('');

  // Feature Selection state
  const [selMethod, setSelMethod] = useState('variance_threshold'); // 'variance_threshold', 'select_k_best', 'correlation_threshold', 'rfe'
  const [vtThreshold, setVtThreshold] = useState(0.01);
  const [skbK, setSkbK] = useState(5);
  const [skbTask, setSkbTask] = useState('classification');
  const [corrThreshold, setCorrThreshold] = useState(0.85);
  const [rfeType, setRfeType] = useState('rfe'); // 'rfe', 'rfecv'
  const [rfeNFeatures, setRfeNFeatures] = useState(5);
  const [rfeTask, setRfeTask] = useState('classification');
  const [selTargetCol, setSelTargetCol] = useState('');

  // Preview state
  const [previewRows, setPreviewRows] = useState(10);
  const [previewData, setPreviewData] = useState([]);
  const [previewLoading, setPreviewLoading] = useState(false);

  // Initialize targets and defaults
  useEffect(() => {
    if (datasetStatus?.columns) {
      const numericCols = getNumericColumns();
      if (numericCols.length > 0) {
        if (!feCol1) setFeCol1(numericCols[0]);
        if (!feCol2) setFeCol2(numericCols[Math.min(1, numericCols.length - 1)]);
        if (!feTransformCol) setFeTransformCol(numericCols[0]);
        if (!feBinCol) setFeBinCol(numericCols[0]);
      }
      
      // Auto-detect target column
      const possibleTargets = ['target', 'label', 'class', 'output', 'y', 'prediction', 'survived', 'status', 'price'];
      let target = datasetStatus.columns.find(c => possibleTargets.includes(c.toLowerCase()));
      if (!target && datasetStatus.columns.length > 0) {
        target = datasetStatus.columns[datasetStatus.columns.length - 1];
      }
      setSelTargetCol(target || '');
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

  const getNumericColumns = () => {
    if (!datasetStatus?.columns) return [];
    return datasetStatus.columns.filter(c => {
      const dtype = datasetStatus.dtypes?.[c]?.toLowerCase() || '';
      return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('number');
    });
  };

  const applyPreprocessing = async (operations) => {
    setLoading(true);
    try {
      await api.post(`/projects/${datasetStatus.project_id}/preprocess`, { operations });
      toast.success("Operations applied successfully!");
      refreshStatus();
    } catch (err) {
      toast.error(err.response?.data?.detail || "Operation failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureEngineering = () => {
    if (!feNewCol.trim()) {
      toast.error("Please provide a name for the new column.");
      return;
    }
    
    if (datasetStatus.columns?.includes(feNewCol.trim())) {
      toast.error("A column with this name already exists.");
      return;
    }
    
    let operation = {
      op: "feature_eng",
      fe_type: feType,
      new_col: feNewCol.trim()
    };
    
    if (feType === 'math_op') {
      if (!feCol1 || !feCol2) {
        toast.error("Please select both columns for the mathematical operation.");
        return;
      }
      operation.col1 = feCol1;
      operation.col2 = feCol2;
      operation.operator = feOperator;
    } else if (feType === 'math_transform') {
      if (!feTransformCol) {
        toast.error("Please select a column to transform.");
        return;
      }
      operation.column = feTransformCol;
      operation.transform = feTransform;
    } else if (feType === 'binning') {
      if (!feBinCol) {
        toast.error("Please select a column to bin.");
        return;
      }
      operation.column = feBinCol;
      operation.bins = parseInt(feBinNum) || 4;
    }
    
    applyPreprocessing([operation]);
    setFeNewCol('');
  };

  const handleFeatureSelection = () => {
    let operation = {
      op: "feature_select",
      method: selMethod,
      target_col: selTargetCol
    };

    if (selMethod === 'variance_threshold') {
      operation.threshold = parseFloat(vtThreshold) || 0.0;
    } else if (selMethod === 'select_k_best') {
      operation.k = parseInt(skbK) || 5;
      operation.task = skbTask;
    } else if (selMethod === 'correlation_threshold') {
      operation.threshold = parseFloat(corrThreshold) || 0.85;
    } else if (selMethod === 'rfe') {
      operation.method = rfeType; // 'rfe' or 'rfecv'
      operation.n_features = parseInt(rfeNFeatures) || 5;
      operation.task = rfeTask;
    }

    applyPreprocessing([operation]);
  };

  if (!datasetStatus?.data_loaded) return null;

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Feature Engineering & Selection</h1>
          <p className="page-subtitle">Generate strong training signals and optimize feature subsets</p>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Active: <strong style={{ color: 'var(--text-main)' }}>{datasetStatus.version_id}</strong> ({datasetStatus.shape?.[0]} x {datasetStatus.shape?.[1]})
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div className="tab-container" style={{ marginBottom: '2rem' }}>
        <button className={`tab-btn ${activeTab === 'engineering' ? 'active' : ''}`} onClick={() => setActiveTab('engineering')}>
          <i className="fa-solid fa-wand-magic-sparkles" style={{ marginRight: '0.5rem' }}></i>
          Feature Engineering
        </button>
        <button className={`tab-btn ${activeTab === 'selection' ? 'active' : ''}`} onClick={() => setActiveTab('selection')}>
          <i className="fa-solid fa-filter" style={{ marginRight: '0.5rem' }}></i>
          Feature Selection
        </button>
      </div>

      <div className="card" style={{ minHeight: '300px', position: 'relative' }}>
        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}

        {/* Tab 1: Feature Engineering */}
        {activeTab === 'engineering' && (
          <div>
            <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600, marginBottom: '0.5rem' }}>Transform Features & Create Columns</h3>
            <p className="card-subtitle">Combine variables, apply mathematical distributions, or split values into ranges.</p>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem' }}>
              <button 
                type="button" 
                className={`btn ${feType === 'math_op' ? 'btn-primary' : 'btn-secondary'}`} 
                onClick={() => setFeType('math_op')}
                style={{ fontSize: '0.8rem', padding: '0.5rem 1rem' }}
              >
                Math Operation (+, -, *, /)
              </button>
              <button 
                type="button" 
                className={`btn ${feType === 'math_transform' ? 'btn-primary' : 'btn-secondary'}`} 
                onClick={() => setFeType('math_transform')}
                style={{ fontSize: '0.8rem', padding: '0.5rem 1rem' }}
              >
                Math Transform (log, sqrt, square)
              </button>
              <button 
                type="button" 
                className={`btn ${feType === 'binning' ? 'btn-primary' : 'btn-secondary'}`} 
                onClick={() => setFeType('binning')}
                style={{ fontSize: '0.8rem', padding: '0.5rem 1rem' }}
              >
                Continuous Binning
              </button>
            </div>

            <div className="grid-2" style={{ alignItems: 'start', margin: '1.5rem 0' }}>
              <div>
                {/* Mode 1: Math Operation */}
                {feType === 'math_op' && (
                  <div>
                    <div className="form-group">
                      <label className="form-label">First Column (Operand 1)</label>
                      <select 
                        className="form-control" 
                        value={feCol1} 
                        onChange={(e) => setFeCol1(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        {getNumericColumns().map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Operator</label>
                      <select 
                        className="form-control" 
                        value={feOperator} 
                        onChange={(e) => setFeOperator(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        <option value="+">Addition (+)</option>
                        <option value="-">Subtraction (-)</option>
                        <option value="*">Multiplication (*)</option>
                        <option value="/">Division (/)</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Second Column (Operand 2)</label>
                      <select 
                        className="form-control" 
                        value={feCol2} 
                        onChange={(e) => setFeCol2(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        {getNumericColumns().map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}

                {/* Mode 2: Math Transform */}
                {feType === 'math_transform' && (
                  <div>
                    <div className="form-group">
                      <label className="form-label">Select Column to Transform</label>
                      <select 
                        className="form-control" 
                        value={feTransformCol} 
                        onChange={(e) => setFeTransformCol(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        {getNumericColumns().map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Mathematical Transformation</label>
                      <select 
                        className="form-control" 
                        value={feTransform} 
                        onChange={(e) => setFeTransform(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        <option value="log">Logarithm (ln(x))</option>
                        <option value="sqrt">Square Root (sqrt(x))</option>
                        <option value="square">Square (x^2)</option>
                        <option value="abs">Absolute Value (|x|)</option>
                      </select>
                    </div>
                  </div>
                )}

                {/* Mode 3: Binning */}
                {feType === 'binning' && (
                  <div>
                    <div className="form-group">
                      <label className="form-label">Select Numerical Column</label>
                      <select 
                        className="form-control" 
                        value={feBinCol} 
                        onChange={(e) => setFeBinCol(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        {getNumericColumns().map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label">Number of Bins (Quantiles)</label>
                      <input 
                        type="number" 
                        min="2" 
                        max="20"
                        className="form-control" 
                        value={feBinNum} 
                        onChange={(e) => setFeBinNum(Math.max(2, parseInt(e.target.value) || 4))}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div>
                <div className="form-group">
                  <label className="form-label">New Column Output Name</label>
                  <input 
                    type="text" 
                    placeholder="e.g. log_income or age_binned"
                    className="form-control"
                    value={feNewCol}
                    onChange={(e) => setFeNewCol(e.target.value.replace(/\s+/g, '_'))}
                    required
                  />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem', display: 'block' }}>
                    Standard spaces will be automatically replaced with underscores.
                  </span>
                </div>
              </div>
            </div>

            <button onClick={handleFeatureEngineering} className="btn btn-primary" style={{ marginTop: '1rem' }} disabled={!feNewCol.trim()}>
              Create Engineered Feature
            </button>
          </div>
        )}

        {/* Tab 2: Feature Selection */}
        {activeTab === 'selection' && (
          <div>
            <h3 style={{ fontSize: '1rem', margin: 0, fontWeight: 600, marginBottom: '0.5rem' }}>Prune Dimensions with Feature Selection</h3>
            <p className="card-subtitle">Keep only columns containing relevant training signals to simplify models and prevent overfitting.</p>

            <div className="grid-2" style={{ alignItems: 'start', margin: '1.5rem 0' }}>
              <div>
                <div className="form-group">
                  <label className="form-label">Target Column (y)</label>
                  <select 
                    className="form-control" 
                    value={selTargetCol} 
                    onChange={(e) => setSelTargetCol(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    {datasetStatus.columns?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)', marginTop: '0.25rem', display: 'block' }}>
                    Required for supervised selection methods (Select K Best, RFE/RFECV).
                  </span>
                </div>

                <div className="form-group">
                  <label className="form-label">Selection Method</label>
                  <select 
                    className="form-control" 
                    value={selMethod} 
                    onChange={(e) => setSelMethod(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    <option value="variance_threshold">Variance Threshold (Drops low variance features)</option>
                    <option value="select_k_best">Select K Best ( ANOVA F-Value score )</option>
                    <option value="correlation_threshold">Correlation Filter (Drops collinear columns)</option>
                    <option value="rfe">Recursive Feature Elimination (RFE / RFECV)</option>
                  </select>
                </div>
              </div>

              <div>
                {/* Variance Threshold Options */}
                {selMethod === 'variance_threshold' && (
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <h4 style={{ fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 600 }}>Variance Threshold Configuration</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.4' }}>
                      Removes all numerical columns whose variance does not exceed the threshold. Useful to drop constants or near-constants.
                    </p>
                    <div className="form-group" style={{ margin: 0 }}>
                      <label className="form-label">Minimum Variance Threshold</label>
                      <input 
                        type="number" 
                        step="0.001" 
                        min="0"
                        className="form-control" 
                        value={vtThreshold} 
                        onChange={(e) => setVtThreshold(parseFloat(e.target.value) || 0.0)}
                      />
                    </div>
                  </div>
                )}

                {/* Select K Best Options */}
                {selMethod === 'select_k_best' && (
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <h4 style={{ fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 600 }}>Select K Best (ANOVA)</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.4' }}>
                      Computes ANOVA F-scores between each numeric feature and the target vector to select the top $K$ features.
                    </p>
                    <div className="form-group">
                      <label className="form-label">Number of Features to Keep (K)</label>
                      <input 
                        type="number" 
                        min="1"
                        max={getNumericColumns().length || 10}
                        className="form-control" 
                        value={skbK} 
                        onChange={(e) => setSkbK(Math.max(1, parseInt(e.target.value) || 5))}
                      />
                    </div>
                    <div className="form-group" style={{ margin: 0 }}>
                      <label className="form-label">Statistical Task Type</label>
                      <select 
                        className="form-control" 
                        value={skbTask} 
                        onChange={(e) => setSkbTask(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        <option value="classification">Classification (ANOVA F-Classif)</option>
                        <option value="regression">Regression (ANOVA F-Regression)</option>
                      </select>
                    </div>
                  </div>
                )}

                {/* Correlation Filter Options */}
                {selMethod === 'correlation_threshold' && (
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <h4 style={{ fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 600 }}>Correlation-based Collinearity Filter</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.4' }}>
                      Computes the absolute Pearson correlation coefficient matrix and removes one feature from each pair exceeding the limit.
                    </p>
                    <div className="form-group" style={{ margin: 0 }}>
                      <label className="form-label">Max Correlation Threshold (0.0 to 1.0)</label>
                      <input 
                        type="number" 
                        step="0.05" 
                        min="0"
                        max="1"
                        className="form-control" 
                        value={corrThreshold} 
                        onChange={(e) => setCorrThreshold(parseFloat(e.target.value) || 0.85)}
                      />
                    </div>
                  </div>
                )}

                {/* RFE / RFECV Options */}
                {selMethod === 'rfe' && (
                  <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <h4 style={{ fontSize: '0.85rem', marginBottom: '0.5rem', fontWeight: 600 }}>Recursive Feature Elimination</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem', lineHeight: '1.4' }}>
                      Fits a Random Forest model recursively, ranks feature coefficients/importances, and discards the least contributing features.
                    </p>
                    <div className="form-group">
                      <label className="form-label">Selection Mode</label>
                      <select 
                        className="form-control" 
                        value={rfeType} 
                        onChange={(e) => setRfeType(e.target.value)}
                        style={{ cursor: 'pointer', marginBottom: '1rem' }}
                      >
                        <option value="rfe">RFE (Fixed subset target)</option>
                        <option value="rfecv">RFECV (Cross-validated search for optimal size)</option>
                      </select>
                    </div>
                    {rfeType === 'rfe' && (
                      <div className="form-group">
                        <label className="form-label">Number of Features to Select</label>
                        <input 
                          type="number" 
                          min="1"
                          max={getNumericColumns().length || 10}
                          className="form-control" 
                          value={rfeNFeatures} 
                          onChange={(e) => setRfeNFeatures(Math.max(1, parseInt(e.target.value) || 5))}
                        />
                      </div>
                    )}
                    <div className="form-group" style={{ margin: 0 }}>
                      <label className="form-label">Supervised Task Type</label>
                      <select 
                        className="form-control" 
                        value={rfeTask} 
                        onChange={(e) => setRfeTask(e.target.value)}
                        style={{ cursor: 'pointer' }}
                      >
                        <option value="classification">Classification (RandomForestClassifier)</option>
                        <option value="regression">Regression (RandomForestRegressor)</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <button onClick={handleFeatureSelection} className="btn btn-primary" style={{ marginTop: '1rem' }}>
              Run Feature Selection
            </button>
          </div>
        )}
      </div>

      {/* Dataset Preview */}
      <div className="card" style={{ marginTop: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0 }}>Dataset Preview</h3>
            <p className="card-subtitle" style={{ margin: 0 }}>Inspect the dataset features and active schema columns</p>
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
    </div>
  );
};

export default FeatureEngineering;
