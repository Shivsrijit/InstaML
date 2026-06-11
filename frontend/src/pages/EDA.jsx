import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  Legend,
  Label
} from 'recharts';

const EDA = ({ datasetStatus }) => {
  const [activeTab, setActiveTab] = useState('univariate');
  const [loading, setLoading] = useState(false);

  // Univariate States
  const [selectedCol, setSelectedCol] = useState('');
  const [uniData, setUniData] = useState([]);
  const [uniStats, setUniStats] = useState(null);
  const [isNumeric, setIsNumeric] = useState(true);

  // Bivariate States
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterData, setScatterData] = useState([]);

  // Correlation States
  const [corrCols, setCorrCols] = useState([]);
  const [corrMatrix, setCorrMatrix] = useState([]);

  // Projection States
  const [projMethod, setProjMethod] = useState('PCA');
  const [projColorBy, setProjColorBy] = useState('');
  const [projCandidates, setProjCandidates] = useState([]);
  const [projPoints, setProjPoints] = useState([]);

  const getNumericColumns = () => {
    if (!datasetStatus?.columns) return [];
    return datasetStatus.columns.filter(c => {
      const dtype = datasetStatus.dtypes?.[c]?.toLowerCase() || '';
      return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('number');
    });
  };

  // Run Univariate Analysis
  const runUnivariate = async (col) => {
    if (!col) return;
    setLoading(true);
    
    // Check if column is numeric
    const numeric = getNumericColumns().includes(col);
    setIsNumeric(numeric);

    try {
      if (numeric) {
        const res = await api.get(`/projects/${datasetStatus.project_id}/eda/histogram/${col}?bins=25`);
        const formatted = res.data.labels.map((lbl, idx) => ({
          range: lbl,
          count: res.data.counts[idx]
        }));
        setUniData(formatted);
        setUniStats({
          mean: res.data.mean,
          median: res.data.median,
          std: res.data.std
        });
      } else {
        const res = await api.get(`/projects/${datasetStatus.project_id}/eda/categories/${col}?top_n=10`);
        const formatted = res.data.labels.map((lbl, idx) => ({
          name: lbl,
          count: res.data.values[idx]
        }));
        setUniData(formatted);
        setUniStats({
          unique: res.data.total_unique
        });
      }
    } catch (err) {
      toast.error("Failed to run univariate analysis.");
    } finally {
      setLoading(false);
    }
  };

  // Run Bivariate Analysis
  const runBivariate = async (x, y) => {
    if (!x || !y) return;
    setLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/eda/scatter?x=${x}&y=${y}&max_points=500`);
      setScatterData(res.data);
    } catch (err) {
      toast.error("Failed to generate scatter plot.");
    } finally {
      setLoading(false);
    }
  };

  // Run Correlation Matrix
  const runCorrelation = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/eda/correlation`);
      setCorrCols(res.data.columns);
      setCorrMatrix(res.data.matrix);
    } catch (err) {
      toast.error("Need at least 2 numerical columns to calculate correlation.");
    } finally {
      setLoading(false);
    }
  };

  // Run Dimensional Projections
  const runProjection = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/eda/dim-reduction`, {
        params: {
          method: projMethod,
          color_by: projColorBy !== 'None' && projColorBy !== '' ? projColorBy : undefined
        }
      });
      setProjPoints(res.data.points || []);
      setProjCandidates(res.data.color_candidates || []);
      if (res.data.color_by && !projColorBy) {
        setProjColorBy(res.data.color_by);
      }
    } catch (err) {
      toast.error("Need at least 2 numeric columns for projections.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'univariate') {
      const firstCol = datasetStatus?.columns?.[0];
      if (firstCol && !selectedCol) {
        setSelectedCol(firstCol);
        runUnivariate(firstCol);
      }
    } else if (activeTab === 'bivariate') {
      const numCols = getNumericColumns();
      if (numCols.length >= 2 && !scatterX) {
        setScatterX(numCols[0]);
        setScatterY(numCols[1]);
        runBivariate(numCols[0], numCols[1]);
      }
    } else if (activeTab === 'correlation') {
      runCorrelation();
    } else if (activeTab === 'projection') {
      runProjection();
    }
  }, [activeTab, projMethod, projColorBy]);

  if (!datasetStatus?.data_loaded) return null;

  // Group points by color value for multi-color plotting
  const groupedPoints = {};
  projPoints.forEach(pt => {
    const group = pt.color || 'All';
    if (!groupedPoints[group]) {
      groupedPoints[group] = [];
    }
    groupedPoints[group].push(pt);
  });

  const colors = [
    'var(--accent-primary)',
    'var(--accent-purple)',
    'var(--accent-green)',
    'var(--accent-yellow)',
    'var(--accent-red)',
    '#3b82f6',
    '#ec4899',
    '#f59e0b',
    '#10b981',
    '#8b5cf6',
  ];

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Exploratory Data Analysis</h1>
          <p className="page-subtitle">Visualize distributions, correlations, and projections of your dataset features</p>
        </div>
      </div>

      {/* Navigation tabs */}
      <div className="tab-container">
        <button className={`tab-btn ${activeTab === 'univariate' ? 'active' : ''}`} onClick={() => setActiveTab('univariate')}>
          <i className="fa-solid fa-chart-bar" style={{ marginRight: '0.5rem' }}></i>
          Univariate (Distributions)
        </button>
        <button className={`tab-btn ${activeTab === 'bivariate' ? 'active' : ''}`} onClick={() => setActiveTab('bivariate')}>
          <i className="fa-solid fa-circle-nodes" style={{ marginRight: '0.5rem' }}></i>
          Bivariate (Scatter)
        </button>
        <button className={`tab-btn ${activeTab === 'correlation' ? 'active' : ''}`} onClick={() => setActiveTab('correlation')}>
          <i className="fa-solid fa-border-all" style={{ marginRight: '0.5rem' }}></i>
          Correlation Heatmap
        </button>
        <button className={`tab-btn ${activeTab === 'projection' ? 'active' : ''}`} onClick={() => setActiveTab('projection')}>
          <i className="fa-solid fa-diagram-project" style={{ marginRight: '0.5rem' }}></i>
          Dimensional Projections
        </button>
      </div>

      <div style={{ position: 'relative', minHeight: '300px' }}>
        {loading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}

        {/* 1. Univariate Analysis */}
        {activeTab === 'univariate' && (
          <div>
            <div className="card" style={{ marginBottom: '2rem', background: 'var(--bg-glass)' }}>
              <div className="form-group" style={{ maxWidth: '400px', margin: 0 }}>
                <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-magnifying-glass-chart" style={{ color: 'var(--text-muted)' }}></i>
                  Choose Feature to Inspect
                </label>
                <select
                  className="form-control"
                  value={selectedCol}
                  onChange={(e) => { setSelectedCol(e.target.value); runUnivariate(e.target.value); }}
                  style={{ cursor: 'pointer' }}
                >
                  {datasetStatus.columns?.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
            </div>

            {uniData.length > 0 && (
              <div className="grid-2" style={{ alignItems: 'stretch' }}>
                <div className="card" style={{ padding: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-chart-simple" style={{ color: 'var(--text-muted)' }}></i>
                    Feature Distribution
                  </h4>
                  <div style={{ height: '350px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={uniData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                        <XAxis dataKey={isNumeric ? "range" : "name"} stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value={selectedCol} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value="Count" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Bar dataKey="count" fill="var(--accent-primary)" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card" style={{ padding: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-square-poll-vertical" style={{ color: 'var(--text-muted)' }}></i>
                    Statistical Metrics
                  </h4>
                  {isNumeric && uniStats ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Mean (Average)</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{uniStats.mean?.toFixed(4)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Median (Middle value)</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{uniStats.median?.toFixed(4)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Standard Deviation</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{uniStats.std?.toFixed(4)}</span>
                      </div>
                    </div>
                  ) : uniStats ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Unique Categories</span>
                        <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{uniStats.unique}</span>
                      </div>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                        This is a categorical column. Category frequency counts are displayed in the bar chart on the left (showing top 10 categories).
                      </p>
                    </div>
                  ) : null}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 2. Bivariate Analysis */}
        {activeTab === 'bivariate' && (
          <div>
            <div className="card" style={{ marginBottom: '2rem', background: 'var(--bg-glass)' }}>
              <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-arrow-right-arrow-left" style={{ color: 'var(--text-muted)' }}></i>
                    X-Axis Variable (Numeric)
                  </label>
                  <select className="form-control" value={scatterX} onChange={(e) => { setScatterX(e.target.value); runBivariate(e.target.value, scatterY); }} style={{ cursor: 'pointer' }}>
                    {getNumericColumns().map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-arrow-up-down" style={{ color: 'var(--text-muted)' }}></i>
                    Y-Axis Variable (Numeric)
                  </label>
                  <select className="form-control" value={scatterY} onChange={(e) => { setScatterY(e.target.value); runBivariate(scatterX, e.target.value); }} style={{ cursor: 'pointer' }}>
                    {getNumericColumns().map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {scatterData.length > 0 && (
              <div className="card" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-chart-line" style={{ color: 'var(--text-muted)' }}></i>
                  {scatterY} vs {scatterX} Scatter Plot
                </h4>
                <div style={{ height: '400px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                      <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name={scatterX} stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value={scatterX} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                      </XAxis>
                      <YAxis type="number" dataKey="y" name={scatterY} stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value={scatterY} angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                      </YAxis>
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                      <Scatter name="Points" data={scatterData} fill="var(--accent-purple)" opacity={0.7} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 3. Correlation Matrix */}
        {activeTab === 'correlation' && corrMatrix.length > 0 && (
          <div className="card" style={{ padding: '2rem' }}>
            <h4 style={{ fontSize: '1.1rem', marginBottom: '0.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <i className="fa-solid fa-table-cells" style={{ color: 'var(--text-muted)' }}></i>
              Feature Correlation Matrix
            </h4>
            <p className="card-subtitle" style={{ marginBottom: '2rem' }}>
              Values range from -1.00 (strong negative correlation) to +1.00 (strong positive correlation).
            </p>
            <div className="table-container">
              <table className="table" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ backgroundColor: 'rgba(255, 255, 255, 0.005)', borderRight: '1px solid var(--border-color)' }}>Feature</th>
                    {corrCols.map(col => (
                      <th key={col} style={{ textAlign: 'center', minWidth: '100px', fontSize: '0.8rem' }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {corrCols.map((rowCol, rowIdx) => (
                    <tr key={rowCol}>
                      <td style={{ fontWeight: 600, fontSize: '0.8rem', backgroundColor: 'rgba(255, 255, 255, 0.002)', borderRight: '1px solid var(--border-color)' }}>{rowCol}</td>
                      {corrCols.map((col, colIdx) => {
                        const val = corrMatrix[rowIdx][colIdx];
                        let cellBg = 'transparent';
                        if (val > 0) {
                          cellBg = `rgba(189, 43, 43, ${val * 0.35})`;
                        } else if (val < 0) {
                          cellBg = `rgba(201, 90, 73, ${Math.abs(val) * 0.35})`;
                        }
                        return (
                          <td
                             key={col}
                             style={{
                               textAlign: 'center',
                               backgroundColor: cellBg,
                               border: '1px solid var(--border-color)',
                               fontWeight: Math.abs(val) > 0.5 ? '700' : 'normal',
                               fontSize: '0.8rem',
                               color: 'var(--text-main)'
                             }}
                          >
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 4. Dimensional Projections */}
        {activeTab === 'projection' && (
          <div>
            <div className="card" style={{ marginBottom: '2rem', background: 'var(--bg-glass)' }}>
              <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-sliders" style={{ color: 'var(--text-muted)' }}></i>
                    Projection Method
                  </label>
                  <select
                    className="form-control"
                    value={projMethod}
                    onChange={(e) => setProjMethod(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    <option value="PCA">PCA (Linear Unsupervised)</option>
                    <option value="TSNE">t-SNE (Non-Linear Unsupervised)</option>
                    <option value="LDA">LDA (Supervised Target Projection)</option>
                    <option value="UMAP">UMAP / Manifold Fallback</option>
                  </select>
                </div>

                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-palette" style={{ color: 'var(--text-muted)' }}></i>
                    Color Grouping Column
                  </label>
                  <select
                    className="form-control"
                    value={projColorBy}
                    onChange={(e) => setProjColorBy(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    <option value="None">None (Single Color)</option>
                    {projCandidates.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {projPoints.length > 0 ? (
              <div className="card" style={{ padding: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                  <div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 600, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <i className="fa-solid fa-circle-nodes" style={{ color: 'var(--text-muted)' }}></i>
                      {projMethod} 2D Projection Space
                    </h4>
                    <p className="card-subtitle" style={{ margin: 0 }}>
                      Colored by: <strong style={{ color: 'var(--accent-primary)' }}>{projColorBy || 'None'}</strong>
                    </p>
                  </div>
                </div>

                <div style={{ height: '450px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 25 }}>
                      <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name="Dimension 1" stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value="Dimension 1" offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                      </XAxis>
                      <YAxis type="number" dataKey="y" name="Dimension 2" stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value="Dimension 2" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                      </YAxis>
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                      <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '12px' }} />
                      {Object.keys(groupedPoints).map((groupName, index) => (
                        <Scatter
                          key={groupName}
                          name={groupName}
                          data={groupedPoints[groupName]}
                          fill={colors[index % colors.length]}
                          opacity={0.8}
                        />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="card" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                No projection points calculated.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default EDA;
