import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';

const Versions = ({ datasetStatus, refreshStatus }) => {
  const [versions, setVersions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [restoreConfirmVersion, setRestoreConfirmVersion] = useState(null);

  // Compare States
  const [v1, setV1] = useState('');
  const [v2, setV2] = useState('');
  const [comparison, setComparison] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const fetchVersions = async () => {
    setLoading(true);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/data/versions`);
      setVersions(res.data);
      if (res.data.length >= 2) {
        setV1(res.data[1].version_id);
        setV2(res.data[0].version_id);
      }
    } catch (err) {
      console.error("Failed to load versions", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVersions();
  }, [datasetStatus]);

  const handleRestore = (ver) => {
    setRestoreConfirmVersion(ver);
  };

  const confirmRestore = async () => {
    if (!restoreConfirmVersion) return;
    const versionId = restoreConfirmVersion.version_id;
    setRestoreConfirmVersion(null);
    setLoading(true);
    try {
      await api.post(`/projects/${datasetStatus.project_id}/data/versions/${versionId}/restore`);
      toast.success(`Dataset successfully restored to checkpoint "${versionId}"`);
      refreshStatus();
    } catch (err) {
      toast.error("Restore failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!v1 || !v2) return;
    setCompareLoading(true);
    setComparison(null);
    try {
      const res = await api.get(`/projects/${datasetStatus.project_id}/data/versions/compare?v1=${v1}&v2=${v2}`);
      setComparison(res.data);
    } catch (err) {
      toast.error("Comparison request failed.");
    } finally {
      setCompareLoading(false);
    }
  };

  return (
    <div>
      <div className="header-bar">
        <div className="page-title-section">
          <h1 className="page-title">Dataset Versioning</h1>
          <p className="page-subtitle">Track, compare, and restore dataset modifications across pipeline steps</p>
        </div>
      </div>

      {/* Compare Tool */}
      {versions.length >= 2 && (
        <div className="card" style={{ marginBottom: '2.5rem', background: 'var(--bg-glass)', backdropFilter: 'blur(12px)' }}>
          <h3 className="card-title">Compare Datasets</h3>
          <p className="card-subtitle">Select two checkpoints to inspect differences in structure and columns.</p>

          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
            <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <i className="fa-solid fa-code-compare" style={{ color: 'var(--accent-primary)' }}></i>
                Base Version (Older)
              </label>
              <select className="form-control" value={v1} onChange={(e) => setV1(e.target.value)} style={{ cursor: 'pointer' }}>
                {versions.map(v => (
                  <option key={v.id} value={v.version_id}>
                    {v.step_name} ({v.version_id.substring(0, 8)})
                  </option>
                ))}
              </select>
            </div>
            
            <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <i className="fa-solid fa-square-check" style={{ color: 'var(--accent-purple)' }}></i>
                Compare Version (Newer)
              </label>
              <select className="form-control" value={v2} onChange={(e) => setV2(e.target.value)} style={{ cursor: 'pointer' }}>
                {versions.map(v => (
                  <option key={v.id} value={v.version_id}>
                    {v.step_name} ({v.version_id.substring(0, 8)})
                  </option>
                ))}
              </select>
            </div>

            <button onClick={handleCompare} className="btn btn-primary" disabled={compareLoading}>
              {compareLoading ? (
                <>
                  <span className="spinner" style={{ marginRight: '0.5rem' }}></span>
                  Comparing...
                </>
              ) : (
                <>
                  <i className="fa-solid fa-magnifying-glass" style={{ marginRight: '0.25rem' }}></i>
                  Run Side-by-Side Compare
                </>
              )}
            </button>
          </div>

          {comparison && (
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem', animation: 'fadeIn 0.2s ease-out' }}>
              <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                {/* Metric Summary Comparison */}
                <div className="table-container">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Base ({v1.substring(0, 8)})</th>
                        <th>Compare ({v2.substring(0, 8)})</th>
                        <th>Difference</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Rows (Samples)</td>
                        <td>{comparison.v1.shape[0]}</td>
                        <td>{comparison.v2.shape[0]}</td>
                        <td style={{ color: comparison.changes.row_diff < 0 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 700 }}>
                          {comparison.changes.row_diff > 0 ? `+${comparison.changes.row_diff}` : comparison.changes.row_diff}
                        </td>
                      </tr>
                      <tr>
                        <td>Columns (Features)</td>
                        <td>{comparison.v1.shape[1]}</td>
                        <td>{comparison.v2.shape[1]}</td>
                        <td style={{ color: comparison.changes.col_diff < 0 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 700 }}>
                          {comparison.changes.col_diff > 0 ? `+${comparison.changes.col_diff}` : comparison.changes.col_diff}
                        </td>
                      </tr>
                      <tr>
                        <td>Missing Cells</td>
                        <td>{comparison.v1.missing_total}</td>
                        <td>{comparison.v2.missing_total}</td>
                        <td style={{ color: (comparison.v2.missing_total - comparison.v1.missing_total) > 0 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 700 }}>
                          {(comparison.v2.missing_total - comparison.v1.missing_total) > 0 ? `+${comparison.v2.missing_total - comparison.v1.missing_total}` : (comparison.v2.missing_total - comparison.v1.missing_total)}
                        </td>
                      </tr>
                      <tr>
                        <td>Duplicate Rows</td>
                        <td>{comparison.v1.duplicate_count}</td>
                        <td>{comparison.v2.duplicate_count}</td>
                        <td style={{ color: (comparison.v2.duplicate_count - comparison.v1.duplicate_count) > 0 ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 700 }}>
                          {(comparison.v2.duplicate_count - comparison.v1.duplicate_count) > 0 ? `+${comparison.v2.duplicate_count - comparison.v1.duplicate_count}` : (comparison.v2.duplicate_count - comparison.v1.duplicate_count)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Column additions/deletions info */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  <div className="card" style={{ padding: '1.25rem', background: 'rgba(255, 255, 255, 0.005)' }}>
                    <h4 style={{ fontSize: '0.9rem', color: 'var(--accent-green)', marginBottom: '0.75rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <i className="fa-solid fa-circle-plus"></i>
                      Columns Added ({comparison.changes.columns_added.length})
                    </h4>
                    {comparison.changes.columns_added.length > 0 ? (
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {comparison.changes.columns_added.map(c => (
                          <span key={c} className="badge badge-success" style={{ fontSize: '0.7rem' }}>{c}</span>
                        ))}
                      </div>
                    ) : (
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>None</span>
                    )}
                  </div>

                  <div className="card" style={{ padding: '1.25rem', background: 'rgba(255, 255, 255, 0.005)' }}>
                    <h4 style={{ fontSize: '0.9rem', color: 'var(--accent-red)', marginBottom: '0.75rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <i className="fa-solid fa-circle-minus"></i>
                      Columns Removed ({comparison.changes.columns_removed.length})
                    </h4>
                    {comparison.changes.columns_removed.length > 0 ? (
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {comparison.changes.columns_removed.map(c => (
                          <span key={c} className="badge badge-warning" style={{ fontSize: '0.7rem' }}>{c}</span>
                        ))}
                      </div>
                    ) : (
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>None</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* History Timeline */}
      <div className="card">
        <h3 className="card-title">Version History Checkpoints</h3>
        <p className="card-subtitle">Roll back the pipeline workspace to older configurations at any time.</p>

        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '3rem' }}>
            <div className="spinner"></div>
          </div>
        ) : (
          <div className="timeline" style={{ marginTop: '2rem' }}>
            {versions.map((ver, idx) => (
              <div key={ver.id} className={`timeline-item ${idx === 0 ? 'active' : ''}`}>
                <div className="timeline-dot" style={{ borderColor: idx === 0 ? 'var(--accent-primary)' : 'var(--border-hover)' }}></div>
                <div
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    border: idx === 0 ? '1px solid var(--border-focus)' : '1px solid var(--border-color)',
                    boxShadow: idx === 0 ? 'var(--shadow-glow)' : 'none',
                    borderRadius: 'var(--radius-md)',
                    padding: '1.5rem',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '1.5rem',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ flex: 1, minWidth: '240px' }}>
                    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                      <h4 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-main)' }}>{ver.step_name}</h4>
                      <span className="badge badge-info" style={{ fontSize: '0.7rem', textTransform: 'none', fontFamily: 'Consolas, Monaco, monospace' }}>
                        {ver.version_id.substring(0, 12)}
                      </span>
                      {idx === 0 && (
                        <span className="badge badge-success" style={{ fontSize: '0.65rem' }}>
                          Active Target
                        </span>
                      )}
                    </div>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.75rem', lineHeight: '1.5' }}>
                      {ver.description}
                    </p>
                    <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.8rem', color: 'var(--text-dim)', flexWrap: 'wrap' }}>
                      <span>
                        <i className="fa-solid fa-table-list" style={{ marginRight: '0.4rem', color: 'var(--accent-primary)' }}></i>
                        {ver.shape_rows} rows × {ver.shape_cols} cols
                      </span>
                      <span>
                        <i className="fa-solid fa-clock" style={{ marginRight: '0.4rem', color: 'var(--accent-purple)' }}></i>
                        {new Date(ver.created_at).toLocaleString()}
                      </span>
                    </div>
                  </div>

                  <div>
                    <button
                      onClick={() => handleRestore(ver)}
                      className="btn btn-secondary"
                      style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
                      disabled={idx === 0} // Can't restore the active latest version
                    >
                      <i className="fa-solid fa-arrow-rotate-left"></i>
                      <span>Restore</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Restore Confirmation Modal */}
      {restoreConfirmVersion && (
        <div className="modal-backdrop">
          <div className="modal-content" style={{ maxWidth: '420px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--accent-purple)' }}>Restore Dataset?</h3>
              <button onClick={() => setRestoreConfirmVersion(null)} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '1.1rem', cursor: 'pointer' }}>
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.6', marginBottom: '2rem' }}>
              Are you sure you want to restore the project dataset to checkpoint <strong>"{restoreConfirmVersion.step_name}"</strong> ({restoreConfirmVersion.version_id.substring(0, 12)})? A new checkpoint will be created.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
              <button type="button" onClick={() => setRestoreConfirmVersion(null)} className="btn btn-secondary">
                Cancel
              </button>
              <button type="button" onClick={confirmRestore} className="btn btn-primary" style={{ backgroundColor: 'var(--accent-purple)', borderColor: 'var(--accent-purple)' }}>
                Restore Checkpoint
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Versions;
