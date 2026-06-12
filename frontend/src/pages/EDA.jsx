import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';
import GuideDrawer from '../components/GuideDrawer';
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
  Label,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area,
  ComposedChart
} from 'recharts';

const EDA = ({ datasetStatus }) => {
  const [activeTab, setActiveTab] = useState('univariate');
  const [loading, setLoading] = useState(false);

  // Guide state
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideTopic, setGuideTopic] = useState('eda_univariate');

  const openGuide = (topic) => {
    setGuideTopic(topic);
    setGuideOpen(true);
  };

  // Univariate States
  const [selectedCol, setSelectedCol] = useState('');
  const [uniPlotType, setUniPlotType] = useState('histogram'); // histogram, boxplot, kde, cdf for numeric; bar, pie for categorical
  const [uniData, setUniData] = useState([]);
  const [uniStats, setUniStats] = useState(null);
  const [isNumeric, setIsNumeric] = useState(true);
  const [boxData, setBoxData] = useState(null);
  const [kdeData, setKdeData] = useState([]);
  const [cdfData, setCdfData] = useState([]);

  // Bivariate States
  const [bivX, setBivX] = useState('');
  const [bivY, setBivY] = useState('');
  const [bivPlotType, setBivPlotType] = useState('scatter'); // scatter, line, grouped_bar
  const [bivData, setBivData] = useState([]);
  const [bivGroupedData, setBivGroupedData] = useState(null);

  // Correlation States
  const [corrCols, setCorrCols] = useState([]);
  const [corrMatrix, setCorrMatrix] = useState([]);
  const [corrThreshold, setCorrThreshold] = useState(0.0);
  const [hideUncorrelated, setHideUncorrelated] = useState(false);

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
  const runUnivariate = async (col, plotType = uniPlotType) => {
    if (!col) return;
    setLoading(true);
    
    // Check if column is numeric
    const numeric = getNumericColumns().includes(col);
    setIsNumeric(numeric);

    // Enforce valid plot types depending on type
    let targetPlotType = plotType;
    if (numeric) {
      if (!['histogram', 'boxplot', 'kde', 'cdf'].includes(targetPlotType)) {
        targetPlotType = 'histogram';
      }
    } else {
      if (!['bar', 'pie'].includes(targetPlotType)) {
        targetPlotType = 'bar';
      }
    }

    if (targetPlotType !== uniPlotType) {
      setUniPlotType(targetPlotType);
    }

    try {
      if (numeric) {
        if (targetPlotType === 'histogram') {
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
        } else if (targetPlotType === 'boxplot') {
          const res = await api.get(`/projects/${datasetStatus.project_id}/eda/boxplot/${col}`);
          setBoxData(res.data);
          setUniStats(null);
        } else if (targetPlotType === 'kde') {
          const res = await api.get(`/projects/${datasetStatus.project_id}/eda/kde/${col}`);
          const formatted = res.data.x.map((xVal, idx) => ({
            x: xVal,
            density: res.data.y[idx]
          }));
          setKdeData(formatted);
          setUniStats(null);
        } else if (targetPlotType === 'cdf') {
          const res = await api.get(`/projects/${datasetStatus.project_id}/eda/cdf/${col}`);
          const formatted = res.data.x.map((xVal, idx) => ({
            x: xVal,
            probability: res.data.y[idx]
          }));
          setCdfData(formatted);
          setUniStats(null);
        }
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
      console.error(err);
      toast.error("Failed to run univariate analysis.");
    } finally {
      setLoading(false);
    }
  };

  // Run Bivariate Analysis
  const runBivariate = async (x, y, plotType = bivPlotType) => {
    if (!x || !y) return;
    setLoading(true);

    const isXNumeric = getNumericColumns().includes(x);
    const isYNumeric = getNumericColumns().includes(y);

    let resolvedPlotType = plotType;
    if (isXNumeric && isYNumeric) {
      if (resolvedPlotType !== 'scatter' && resolvedPlotType !== 'line') {
        resolvedPlotType = 'scatter';
      }
    } else if ((isXNumeric && !isYNumeric) || (!isXNumeric && isYNumeric)) {
      resolvedPlotType = 'grouped_bar';
    } else {
      toast.error("At least one numerical variable is required for bivariate visualization.");
      setLoading(false);
      return;
    }

    if (resolvedPlotType !== bivPlotType) {
      setBivPlotType(resolvedPlotType);
    }

    try {
      if (resolvedPlotType === 'grouped_bar') {
        const numeric_col = isXNumeric ? x : y;
        const categorical_col = isXNumeric ? y : x;
        const res = await api.get(`/projects/${datasetStatus.project_id}/eda/grouped`, {
          params: { numeric_col, categorical_col }
        });
        const formatted = res.data.labels.map((lbl, idx) => ({
          category: lbl,
          mean: res.data.means[idx],
          count: res.data.counts[idx],
          std: res.data.stds[idx]
        }));
        setBivGroupedData({
          numeric_col,
          categorical_col,
          points: formatted
        });
        setBivData([]);
      } else {
        const res = await api.get(`/projects/${datasetStatus.project_id}/eda/scatter`, {
          params: { x, y, max_points: 500 }
        });
        let dataPoints = res.data;
        if (resolvedPlotType === 'line') {
          dataPoints = [...dataPoints].sort((a, b) => a.x - b.x);
        }
        setBivData(dataPoints);
        setBivGroupedData(null);
      }
    } catch (err) {
      console.error(err);
      toast.error("Failed to generate bivariate plot.");
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

  // Set up initial column selections once datasetStatus loads
  useEffect(() => {
    if (datasetStatus?.columns?.length > 0) {
      const firstCol = datasetStatus.columns[0];
      if (!selectedCol) {
        setSelectedCol(firstCol);
        const numeric = getNumericColumns().includes(firstCol);
        setUniPlotType(numeric ? 'histogram' : 'bar');
      }

      if (datasetStatus.columns.length >= 2 && (!bivX || !bivY)) {
        setBivX(datasetStatus.columns[0]);
        setBivY(datasetStatus.columns[1]);
        const isXNum = getNumericColumns().includes(datasetStatus.columns[0]);
        const isYNum = getNumericColumns().includes(datasetStatus.columns[1]);
        setBivPlotType(isXNum && isYNum ? 'scatter' : 'grouped_bar');
      }
    }
  }, [datasetStatus]);

  // Handle updates when selections change
  useEffect(() => {
    if (activeTab === 'univariate' && selectedCol) {
      runUnivariate(selectedCol, uniPlotType);
    }
  }, [selectedCol, uniPlotType, activeTab]);

  useEffect(() => {
    if (activeTab === 'bivariate' && bivX && bivY) {
      runBivariate(bivX, bivY, bivPlotType);
    }
  }, [bivX, bivY, bivPlotType, activeTab]);

  useEffect(() => {
    if (activeTab === 'correlation') {
      runCorrelation();
    }
  }, [activeTab]);

  useEffect(() => {
    if (activeTab === 'projection') {
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
        <div>
          <button 
            onClick={() => openGuide(
              activeTab === 'univariate' ? 'eda_univariate' :
              activeTab === 'bivariate' ? 'eda_bivariate' :
              activeTab === 'correlation' ? 'eda_correlation' :
              'eda_projections'
            )} 
            className="btn btn-secondary"
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', padding: '0.5rem 0.85rem' }}
          >
            <i className="fa-solid fa-graduation-cap"></i>
            <span>Get to know more</span>
          </button>
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
              <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <div className="form-group" style={{ flex: 1, minWidth: '240px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-magnifying-glass-chart" style={{ color: 'var(--text-muted)' }}></i>
                    Choose Feature to Inspect
                  </label>
                  <select
                    className="form-control"
                    value={selectedCol}
                    onChange={(e) => { setSelectedCol(e.target.value); }}
                    style={{ cursor: 'pointer' }}
                  >
                    {datasetStatus.columns?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group" style={{ flex: 1, minWidth: '240px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-sliders" style={{ color: 'var(--text-muted)' }}></i>
                    Visualization Type
                  </label>
                  <select
                    className="form-control"
                    value={uniPlotType}
                    onChange={(e) => setUniPlotType(e.target.value)}
                    style={{ cursor: 'pointer' }}
                  >
                    {isNumeric ? (
                      <>
                        <option value="histogram">Histogram / Bar Chart</option>
                        <option value="boxplot">Box Plot & Outliers</option>
                        <option value="kde">KDE Density Curve</option>
                        <option value="cdf">CDF Cumulative Curve</option>
                      </>
                    ) : (
                      <>
                        <option value="bar">Bar Chart</option>
                        <option value="pie">Pie / Donut Chart</option>
                      </>
                    )}
                  </select>
                </div>
              </div>
            </div>

            <div className="grid-2" style={{ alignItems: 'stretch' }}>
              <div className="card" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-chart-simple" style={{ color: 'var(--text-muted)' }}></i>
                  {selectedCol} Distribution ({uniPlotType.toUpperCase()})
                </h4>
                <div style={{ height: '350px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {uniPlotType === 'histogram' && uniData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={uniData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                        <XAxis dataKey="range" stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value={selectedCol} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value="Count" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Bar dataKey="count" fill="var(--accent-primary)" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}

                  {uniPlotType === 'boxplot' && boxData && (
                    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      {(() => {
                        const { min, q1, median, q3, max, outliers, total_outliers, total_rows } = boxData;
                        const range = max - min;
                        const scale = (val) => range === 0 ? 50 : ((val - min) / range) * 100;
                        return (
                          <div style={{ padding: '1rem', width: '100%' }}>
                            <div style={{ position: 'relative', height: '60px', width: '100%', borderBottom: '1px dashed var(--border-color)', marginBottom: '1.5rem' }}>
                              <div style={{ position: 'absolute', top: '30px', left: `${scale(min)}%`, right: `${100 - scale(max)}%`, height: '2px', backgroundColor: 'var(--text-muted)' }}></div>
                              <div style={{ position: 'absolute', top: '20px', bottom: '20px', left: `${scale(min)}%`, width: '2px', backgroundColor: 'var(--text-muted)' }}></div>
                              <div style={{ position: 'absolute', top: '20px', bottom: '20px', left: `${scale(max)}%`, width: '2px', backgroundColor: 'var(--text-muted)' }}></div>
                              <div style={{ position: 'absolute', top: '10px', bottom: '10px', left: `${scale(q1)}%`, right: `${100 - scale(q3)}%`, backgroundColor: 'rgba(139, 92, 246, 0.2)', border: '2px solid var(--accent-purple)', borderRadius: '4px' }}></div>
                              <div style={{ position: 'absolute', top: '10px', bottom: '10px', left: `${scale(median)}%`, width: '3px', backgroundColor: 'var(--accent-primary)' }}></div>
                            </div>
                            
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.5rem', textAlign: 'center', fontSize: '0.8rem', marginTop: '1rem' }}>
                              <div>
                                <div style={{ color: 'var(--text-muted)' }}>Min</div>
                                <div style={{ fontWeight: 600 }}>{min.toFixed(4)}</div>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)' }}>Q1 (25%)</div>
                                <div style={{ fontWeight: 600 }}>{q1.toFixed(4)}</div>
                              </div>
                              <div>
                                <div style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>Median</div>
                                <div style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>{median.toFixed(4)}</div>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)' }}>Q3 (75%)</div>
                                <div style={{ fontWeight: 600 }}>{q3.toFixed(4)}</div>
                              </div>
                              <div>
                                <div style={{ color: 'var(--text-muted)' }}>Max</div>
                                <div style={{ fontWeight: 600 }}>{max.toFixed(4)}</div>
                              </div>
                            </div>
                            
                            <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                              <h5 style={{ fontSize: '0.85rem', fontWeight: 600, display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <span>Outliers Count: <strong style={{ color: total_outliers > 0 ? 'var(--accent-red)' : 'var(--text-muted)' }}>{total_outliers}</strong></span>
                                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Percentage: {((total_outliers / total_rows) * 100).toFixed(2)}%</span>
                              </h5>
                              {outliers.length > 0 ? (
                                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', maxHeight: '80px', overflowY: 'auto', padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', border: '1px solid var(--border-color)' }}>
                                  {outliers.slice(0, 50).map((out, idx) => (
                                    <span key={idx} className="badge badge-warning" style={{ fontSize: '0.65rem', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--accent-red)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                                      {out.toFixed(4)}
                                    </span>
                                  ))}
                                  {outliers.length > 50 && <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>+{outliers.length - 50} more</span>}
                                </div>
                              ) : (
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>No outliers detected.</div>
                              )}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {uniPlotType === 'kde' && kdeData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={kdeData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                        <XAxis dataKey="x" stroke="var(--text-muted)" fontSize={11} tickLine={false} type="number" domain={['dataMin', 'dataMax']}>
                          <Label value={selectedCol} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value="Density" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Area type="monotone" dataKey="density" stroke="var(--accent-purple)" fill="rgba(139, 92, 246, 0.15)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  )}

                  {uniPlotType === 'cdf' && cdfData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cdfData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                        <XAxis dataKey="x" stroke="var(--text-muted)" fontSize={11} tickLine={false} type="number" domain={['dataMin', 'dataMax']}>
                          <Label value={selectedCol} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={[0, 1]}>
                          <Label value="Probability" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Line type="monotone" dataKey="probability" stroke="var(--accent-primary)" dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  )}

                  {uniPlotType === 'bar' && uniData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={uniData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                        <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value={selectedCol} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                          <Label value="Count" angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Bar dataKey="count" fill="var(--accent-primary)" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}

                  {uniPlotType === 'pie' && uniData.length > 0 && (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={uniData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          fill="#8884d8"
                          paddingAngle={3}
                          dataKey="count"
                        >
                          {uniData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Legend verticalAlign="bottom" height={36} wrapperStyle={{ fontSize: '11px' }} />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>

              <div className="card" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-square-poll-vertical" style={{ color: 'var(--text-muted)' }}></i>
                  Statistical Metrics
                </h4>
                {isNumeric && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {uniStats ? (
                      <>
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
                      </>
                    ) : boxData ? (
                      <>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Interquartile Range (IQR)</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{(boxData.q3 - boxData.q1).toFixed(4)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Lower Outlier Bound</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{boxData.lower_whisker.toFixed(4)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Upper Outlier Bound</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{boxData.upper_whisker.toFixed(4)}</span>
                        </div>
                      </>
                    ) : (
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        Loading feature parameters...
                      </div>
                    )}
                  </div>
                )}
                {!isNumeric && uniStats && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Unique Categories</span>
                      <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.85rem' }}>{uniStats.unique}</span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                      This is a categorical column. Category frequency counts are displayed in the {uniPlotType} chart on the left (showing top 10 categories).
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 2. Bivariate Analysis */}
        {activeTab === 'bivariate' && (
          <div>
            <div className="card" style={{ marginBottom: '2rem', background: 'var(--bg-glass)' }}>
              <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-arrow-right-arrow-left" style={{ color: 'var(--text-muted)' }}></i>
                    X-Axis Variable
                  </label>
                  <select className="form-control" value={bivX} onChange={(e) => { setBivX(e.target.value); }} style={{ cursor: 'pointer' }}>
                    {datasetStatus.columns?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                  <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <i className="fa-solid fa-arrow-up-down" style={{ color: 'var(--text-muted)' }}></i>
                    Y-Axis Variable
                  </label>
                  <select className="form-control" value={bivY} onChange={(e) => { setBivY(e.target.value); }} style={{ cursor: 'pointer' }}>
                    {datasetStatus.columns?.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                {getNumericColumns().includes(bivX) && getNumericColumns().includes(bivY) && (
                  <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
                    <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <i className="fa-solid fa-sliders" style={{ color: 'var(--text-muted)' }}></i>
                      Bivariate Chart Type
                    </label>
                    <select className="form-control" value={bivPlotType} onChange={(e) => setBivPlotType(e.target.value)} style={{ cursor: 'pointer' }}>
                      <option value="scatter">Scatter Plot</option>
                      <option value="line">Line Trend (Sorted X)</option>
                    </select>
                  </div>
                )}
              </div>
            </div>

            {bivData.length > 0 && bivPlotType !== 'grouped_bar' && (
              <div className="card" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-chart-line" style={{ color: 'var(--text-muted)' }}></i>
                  {bivY} vs {bivX} {bivPlotType === 'scatter' ? 'Scatter' : 'Line'} Plot
                </h4>
                <div style={{ height: '400px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    {bivPlotType === 'scatter' ? (
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="x" name={bivX} stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['dataMin', 'dataMax']}>
                          <Label value={bivX} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis type="number" dataKey="y" name={bivY} stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['dataMin', 'dataMax']}>
                          <Label value={bivY} angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Scatter name="Points" data={bivData} fill="var(--accent-purple)" opacity={0.7} />
                      </ScatterChart>
                    ) : (
                      <LineChart data={bivData} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                        <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="x" name={bivX} stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['dataMin', 'dataMax']}>
                          <Label value={bivX} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                        </XAxis>
                        <YAxis type="number" dataKey="y" name={bivY} stroke="var(--text-muted)" fontSize={11} tickLine={false} domain={['dataMin', 'dataMax']}>
                          <Label value={bivY} angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                        </YAxis>
                        <Tooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }} />
                        <Line type="monotone" dataKey="y" stroke="var(--accent-purple)" dot={false} strokeWidth={2} />
                      </LineChart>
                    )}
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {bivGroupedData && bivPlotType === 'grouped_bar' && (
              <div className="card" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '0.9rem', marginBottom: '1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-chart-column" style={{ color: 'var(--text-muted)' }}></i>
                  Average {bivGroupedData.numeric_col} Grouped by {bivGroupedData.categorical_col}
                </h4>
                <div style={{ height: '400px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={bivGroupedData.points} margin={{ top: 10, right: 10, bottom: 25, left: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                      <XAxis dataKey="category" stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value={bivGroupedData.categorical_col} offset={-15} position="insideBottom" fill="var(--text-muted)" fontSize={12} fontWeight={600} />
                      </XAxis>
                      <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false}>
                        <Label value={`Average ${bivGroupedData.numeric_col}`} angle={-90} position="insideLeft" offset={-10} style={{ textAnchor: 'middle', fill: 'var(--text-muted)', fontSize: 12, fontWeight: 600 }} />
                      </YAxis>
                      <Tooltip
                        contentStyle={{ backgroundColor: 'var(--bg-tertiary)', borderColor: 'var(--border-color)', color: 'var(--text-main)', borderRadius: '6px' }}
                        formatter={(value, name, props) => {
                          if (name === 'mean') return [`${value?.toFixed(4)}`, `Average`];
                          return [value, name];
                        }}
                      />
                      <Bar dataKey="mean" fill="var(--accent-primary)" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 3. Correlation Matrix */}
        {activeTab === 'correlation' && corrMatrix.length > 0 && (
          <div className="card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div>
                <h4 style={{ fontSize: '1.1rem', margin: 0, fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <i className="fa-solid fa-table-cells" style={{ color: 'var(--text-muted)' }}></i>
                  Feature Correlation Matrix
                </h4>
                <p className="card-subtitle" style={{ margin: '0.25rem 0 0 0' }}>
                  Values range from -1.00 (strong negative correlation) to +1.00 (strong positive correlation).
                </p>
              </div>

              {/* Filtering Controls */}
              <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Correlation Threshold:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)', width: '40px' }}>&ge; {corrThreshold.toFixed(2)}</span>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={corrThreshold}
                    onChange={(e) => setCorrThreshold(parseFloat(e.target.value))}
                    style={{ width: '120px', cursor: 'pointer', accentColor: 'var(--accent-primary)' }}
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="checkbox"
                    id="hide-uncorrelated-check"
                    checked={hideUncorrelated}
                    onChange={(e) => setHideUncorrelated(e.target.checked)}
                    style={{ cursor: 'pointer', width: '16px', height: '16px', accentColor: 'var(--accent-primary)' }}
                  />
                  <label htmlFor="hide-uncorrelated-check" style={{ fontSize: '0.85rem', cursor: 'pointer', userSelect: 'none', color: 'var(--text-muted)' }}>
                    Hide uncorrelated columns
                  </label>
                </div>
              </div>
            </div>

            {(() => {
              // Filter columns based on threshold
              let filteredCols = [...corrCols];
              if (hideUncorrelated && corrCols.length > 0 && corrMatrix.length > 0) {
                filteredCols = corrCols.filter((col, colIdx) => {
                  return corrCols.some((otherCol, otherIdx) => {
                    if (colIdx === otherIdx) return false;
                    const val = corrMatrix[colIdx][otherIdx];
                    return Math.abs(val) >= corrThreshold;
                  });
                });
              }

              if (filteredCols.length === 0) {
                return (
                  <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    No features meet the current correlation threshold of &ge; {corrThreshold.toFixed(2)}. Try lowering the threshold.
                  </div>
                );
              }

              return (
                <div className="table-container">
                  <table className="table" style={{ borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ backgroundColor: 'rgba(255, 255, 255, 0.005)', borderRight: '1px solid var(--border-color)', fontSize: '0.8rem' }}>Feature</th>
                        {filteredCols.map(col => (
                          <th key={col} style={{ textAlign: 'center', minWidth: '90px', fontSize: '0.75rem' }}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredCols.map((rowCol) => {
                        const rowOriginalIdx = corrCols.indexOf(rowCol);
                        return (
                          <tr key={rowCol}>
                            <td style={{ fontWeight: 600, fontSize: '0.8rem', backgroundColor: 'rgba(255, 255, 255, 0.002)', borderRight: '1px solid var(--border-color)', whiteSpace: 'nowrap' }}>{rowCol}</td>
                            {filteredCols.map((col) => {
                              const colOriginalIdx = corrCols.indexOf(col);
                              const val = corrMatrix[rowOriginalIdx][colOriginalIdx];
                              const isAboveThreshold = Math.abs(val) >= corrThreshold;
                              
                              let cellBg = 'transparent';
                              if (isAboveThreshold) {
                                if (val > 0) {
                                  cellBg = `rgba(189, 43, 43, ${val * 0.35})`;
                                } else if (val < 0) {
                                  cellBg = `rgba(201, 90, 73, ${Math.abs(val) * 0.35})`;
                                }
                              }

                              return (
                                <td
                                  key={col}
                                  style={{
                                    textAlign: 'center',
                                    backgroundColor: cellBg,
                                    border: '1px solid var(--border-color)',
                                    fontWeight: isAboveThreshold && Math.abs(val) > 0.5 ? '700' : 'normal',
                                    fontSize: '0.8rem',
                                    color: isAboveThreshold ? 'var(--text-main)' : 'var(--text-dim)',
                                    opacity: isAboveThreshold ? 1 : 0.25
                                  }}
                                >
                                  {isAboveThreshold ? val.toFixed(2) : '-'}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              );
            })()}
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
      <GuideDrawer isOpen={guideOpen} onClose={() => setGuideOpen(false)} initialTopic={guideTopic} />
    </div>
  );
};

export default EDA;
