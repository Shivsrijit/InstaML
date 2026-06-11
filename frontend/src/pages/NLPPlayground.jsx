import React, { useState } from 'react';
import toast from 'react-hot-toast';
import api from '../services/api';

const NLPPlayground = ({ project }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type !== "text/plain" && !selectedFile.name.endsWith('.txt')) {
        toast.error("Please upload a valid plain text (.txt) file.");
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setText(''); // clear text when file is loaded
      setResults(null);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setResults(null);

    // Validate input
    if (!text.trim() && !file) {
      toast.error("Please paste some text or upload a .txt file first.");
      return;
    }

    setLoading(true);

    try {
      let res;
      if (file) {
        // Upload file endpoint
        const formData = new FormData();
        formData.append("file", file);
        res = await api.post("/nlp/analyze-file", formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
      } else {
        // Direct text analysis
        res = await api.post("/nlp/analyze", { text });
      }
      setResults(res.data);
      toast.success("Text analyzed successfully!");
    } catch (err) {
      toast.error(err.response?.data?.detail || "NLP analysis failed. Please check your text or file format.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setFile(null);
    setResults(null);
    // Reset file input element
    const fileInput = document.getElementById('nlp-file-input');
    if (fileInput) fileInput.value = '';
  };

  const renderStats = () => {
    if (!results?.statistics) return null;
    const { word_count, char_count, sentence_count, avg_word_length, reading_time_seconds } = results.statistics;

    return (
      <div className="card" style={{ height: '100%' }}>
        <h3 className="card-title">
          <i className="fa-solid fa-calculator" style={{ color: 'var(--accent-primary)' }}></i>
          <span>Document Stats</span>
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Word Count</span>
            <span style={{ fontWeight: '600' }}>{word_count.toLocaleString()} words</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Character Count</span>
            <span style={{ fontWeight: '600' }}>{char_count.toLocaleString()} chars</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Sentence Count</span>
            <span style={{ fontWeight: '600' }}>{sentence_count.toLocaleString()}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Avg. Word Length</span>
            <span style={{ fontWeight: '600' }}>{avg_word_length} chars</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Est. Reading Time</span>
            <span style={{ fontWeight: '600', color: 'var(--accent-green)' }}>
              {reading_time_seconds < 60 ? `${reading_time_seconds}s` : `${Math.round(reading_time_seconds/60)} min`}
            </span>
          </div>
        </div>
      </div>
    );
  };

  const renderSentiment = () => {
    if (!results?.sentiment) return null;
    const { label, score, positive_count, negative_count } = results.sentiment;

    let sentimentColor = 'var(--text-muted)';
    let sentimentIcon = 'fa-meh';
    if (label === 'positive') {
      sentimentColor = 'var(--accent-green)';
      sentimentIcon = 'fa-smile';
    } else if (label === 'negative') {
      sentimentColor = 'var(--accent-red)';
      sentimentIcon = 'fa-frown';
    }

    return (
      <div className="card" style={{ height: '100%' }}>
        <h3 className="card-title">
          <i className="fa-solid fa-heart" style={{ color: 'var(--accent-purple)' }}></i>
          <span>Sentiment Analysis</span>
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '1.5rem 0', gap: '1rem' }}>
          <i className={`fa-regular ${sentimentIcon}`} style={{ fontSize: '4rem', color: sentimentColor }}></i>
          <div style={{ textAlign: 'center' }}>
            <h2 style={{ textTransform: 'capitalize', color: sentimentColor, fontWeight: 800, fontSize: '1.75rem' }}>{label}</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>Polarity Score: {score}</p>
          </div>
        </div>
        
        {/* Progress Bar Gauge */}
        <div style={{ width: '100%', backgroundColor: 'var(--bg-tertiary)', height: '6px', borderRadius: '3px', position: 'relative', margin: '0.5rem 0' }}>
          <div style={{
            position: 'absolute',
            left: `${(score + 1) * 50}%`,
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: sentimentColor,
            top: '-3px',
            transform: 'translateX(-50%)',
            boxShadow: `0 0 8px ${sentimentColor}`
          }}></div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          <span>Negative (-1.0)</span>
          <span>Neutral (0.0)</span>
          <span>Positive (1.0)</span>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
          <div className="badge badge-success" style={{ flex: 1, justifyContent: 'center', padding: '0.5rem' }}>
            <span>Pos Words: {positive_count}</span>
          </div>
          <div className="badge badge-warning" style={{ flex: 1, justifyContent: 'center', color: 'var(--accent-red)', borderColor: 'rgba(189, 43, 43, 0.2)', backgroundColor: 'rgba(189, 43, 43, 0.1)' }}>
            <span>Neg Words: {negative_count}</span>
          </div>
        </div>
      </div>
    );
  };

  const renderKeywords = () => {
    if (!results?.keywords || results.keywords.length === 0) return null;

    return (
      <div className="card" style={{ height: '100%' }}>
        <h3 className="card-title">
          <i className="fa-solid fa-tags" style={{ color: 'var(--accent-yellow)' }}></i>
          <span>Topic Keywords</span>
        </h3>
        <p className="card-subtitle" style={{ marginBottom: '1rem' }}>Extracted key thematic keywords based on relevance frequency.</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1rem' }}>
          {results.keywords.map((kw, i) => (
            <div key={i} className="badge badge-info" style={{ padding: '0.5rem 0.85rem', fontSize: '0.85rem', gap: '0.5rem', display: 'inline-flex', alignItems: 'center' }}>
              <span style={{ fontWeight: 600 }}>{kw.word}</span>
              <span style={{ opacity: 0.6, fontSize: '0.75rem', backgroundColor: 'rgba(255,255,255,0.1)', padding: '0.1rem 0.35rem', borderRadius: '4px' }}>
                {kw.count}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderSummary = () => {
    if (!results?.summary) return null;

    return (
      <div className="card" style={{ gridColumn: 'span 3' }}>
        <h3 className="card-title">
          <i className="fa-solid fa-quote-left" style={{ color: 'var(--accent-green)' }}></i>
          <span>Extractive Text Summary</span>
        </h3>
        <p className="card-subtitle">AI-extracted summary highlighting the top sentences of the document.</p>
        <div style={{
          backgroundColor: 'var(--bg-tertiary)',
          borderLeft: '4px solid var(--accent-green)',
          padding: '1.25rem',
          borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
          fontSize: '0.95rem',
          lineHeight: '1.7',
          color: 'var(--text-main)',
          fontStyle: 'italic',
          marginTop: '1.25rem'
        }}>
          "{results.summary}"
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className="header-bar" style={{ marginBottom: '2rem' }}>
        <div className="page-title-section">
          <h1 className="page-title">NLP Analysis Playground</h1>
          <p className="page-subtitle">Paste raw text or upload a plain notepad file (.txt) for instant NLP calculations.</p>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: 'start', marginBottom: '2rem' }}>
        {/* Input Area */}
        <div className="card">
          <h3 className="card-title">Input Document</h3>
          <p className="card-subtitle">Paste a block of text, copy an article, or upload a notepad .txt file.</p>

          <form onSubmit={handleAnalyze}>
            <div className="form-group" style={{ position: 'relative' }}>
              <textarea
                className="form-control"
                style={{ height: '220px', resize: 'none', fontFamily: 'var(--font-body)', fontSize: '0.9rem', lineHeight: '1.5' }}
                placeholder="Paste your raw text document here..."
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  setFile(null); // clear file if they start typing
                }}
                disabled={loading || !!file}
              />
              {file && (
                <div style={{
                  position: 'absolute',
                  top: 0, left: 0, right: 0, bottom: 0,
                  backgroundColor: 'var(--bg-glass)',
                  backdropFilter: 'blur(8px)',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-color)'
                }}>
                  <i className="fa-solid fa-file-lines" style={{ fontSize: '3rem', color: 'var(--accent-primary)', marginBottom: '0.75rem' }}></i>
                  <h4 style={{ fontWeight: 600 }}>{file.name}</h4>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{(file.size / 1024).toFixed(1)} KB loaded</p>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
              {/* File Upload Button */}
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <label htmlFor="nlp-file-input" className="btn btn-secondary" style={{ padding: '0.5rem 1rem', borderRadius: 'var(--radius-sm)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                  <i className="fa-solid fa-file-import"></i>
                  <span>Upload Notepad .txt</span>
                </label>
                <input
                  id="nlp-file-input"
                  type="file"
                  style={{ display: 'none' }}
                  accept=".txt"
                  onChange={handleFileChange}
                  disabled={loading}
                />
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                {(text || file) && (
                  <button type="button" className="btn btn-secondary" onClick={handleClear} disabled={loading} style={{ padding: '0.5rem 1rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
                    Clear
                  </button>
                )}
                <button
                  type="submit"
                  className="btn btn-primary"
                  style={{ padding: '0.5rem 1.5rem', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}
                  disabled={loading || (!text.trim() && !file)}
                >
                  {loading ? <span className="spinner"></span> : <span>Analyze Text</span>}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Informative Card */}
        <div className="card" style={{ height: '335px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h3 className="card-title">NLP Utility Toolkit</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.6', marginBottom: '1rem' }}>
            Don't want to compile full spreadsheets or train neural networks? Paste raw documents or upload text logs here for instantaneous calculations:
          </p>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.75rem' }}>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.85rem' }}>
              <i className="fa-solid fa-check" style={{ color: 'var(--accent-green)' }}></i>
              <span><strong>Readability Stats</strong>: Calculates WPM speeds and length counts.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.85rem' }}>
              <i className="fa-solid fa-check" style={{ color: 'var(--accent-green)' }}></i>
              <span><strong>Extractive Summary</strong>: Trims essays into key sentences automatically.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.85rem' }}>
              <i className="fa-solid fa-check" style={{ color: 'var(--accent-green)' }}></i>
              <span><strong>Sentiment Index</strong>: Matches emotional polarity against a core lexicon.</span>
            </li>
            <li style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', fontSize: '0.85rem' }}>
              <i className="fa-solid fa-check" style={{ color: 'var(--accent-green)' }}></i>
              <span><strong>Keywords/Tags</strong>: Pinpoints the most prominent word tags.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Results Dashboard Grid */}
      {results && (
        <div className="grid-3" style={{ marginTop: '2rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
          {renderStats()}
          {renderSentiment()}
          {renderKeywords()}
          {renderSummary()}
        </div>
      )}
    </div>
  );
};

export default NLPPlayground;
