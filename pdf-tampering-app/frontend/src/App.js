import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile && uploadedFile.type === 'application/pdf') {
      setFile(uploadedFile);
      setResult(null);
      setError(null);
    } else {
      setError('Please upload a valid PDF file');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1
  });

  const analyzePDF = async () => {
    if (!file) {
      setError('Please select a PDF file first');
      return;
    }

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🔍 PDF Tampering Detector</h1>
        <p>Hybrid Font Metadata & Glyph Matching Algorithm</p>
      </header>

      <main className="main">
        {!result ? (
          <div className="upload-section">
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <div className="dropzone-content">
                <div className="upload-icon">📄</div>
                {isDragActive ? (
                  <p>Drop your PDF here...</p>
                ) : (
                  <>
                    <p>Drag & drop a PDF file here</p>
                    <p className="small">or click to browse</p>
                  </>
                )}
              </div>
            </div>

            {file && (
              <div className="file-info">
                <span className="file-name">📄 {file.name}</span>
                <span className="file-size">({(file.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}

            <div className="button-group">
              <button 
                className="btn btn-primary" 
                onClick={analyzePDF} 
                disabled={!file || analyzing}
              >
                {analyzing ? 'Analyzing...' : '🔍 Detect Tampering'}
              </button>
              {file && (
                <button className="btn btn-secondary" onClick={resetAnalysis}>
                  Clear
                </button>
              )}
            </div>

            {error && <div className="error-message">{error}</div>}
          </div>
        ) : (
          <div className="results-section">
            <div className={`result-card ${result.is_tampered ? 'tampered' : 'genuine'}`}>
              <div className="result-icon">
                {result.is_tampered ? '⚠️' : '✅'}
              </div>
              <div className="result-status">
                <h2>{result.is_tampered ? 'TAMPERED DETECTED' : 'GENUINE DOCUMENT'}</h2>
                <div className="confidence-bar">
                  <div 
                    className={`confidence-fill ${result.is_tampered ? 'tampered' : 'genuine'}`}
                    style={{ width: `${result.confidence}%` }}
                  />
                  <span className="confidence-text">{result.confidence}% Confidence</span>
                </div>
              </div>
            </div>

            <div className="details-card">
              <h3>📊 Analysis Details</h3>
              <div className="details-grid">
                <div className="detail-item">
                  <span className="detail-label">Font Consistency:</span>
                  <span className="detail-value">{result.details.font_consistency}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Unique Fonts:</span>
                  <span className="detail-value">{result.details.unique_fonts}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Font Variance:</span>
                  <span className="detail-value">{result.details.font_variance}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Avg Line Length:</span>
                  <span className="detail-value">{result.details.avg_line_length}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Avg Word Length:</span>
                  <span className="detail-value">{result.details.avg_word_length}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Space Ratio:</span>
                  <span className="detail-value">{result.details.space_ratio}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Tampering Score:</span>
                  <span className="detail-value">{result.details.tampering_score}</span>
                </div>
              </div>
            </div>

            {result.issues && result.issues.length > 0 && (
              <div className="issues-card">
                <h3>⚠️ Detected Issues</h3>
                <ul className="issues-list">
                  {result.issues.map((issue, idx) => (
                    <li key={idx}>{issue}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="recommendation-card">
              <h3>💡 Recommendation</h3>
              <p>{result.recommendation}</p>
            </div>

            <div className="file-info-footer">
              <span>📄 {result.filename}</span>
              <button className="btn-new" onClick={resetAnalysis}>
                + Analyze New PDF
              </button>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Hybrid Font Metadata & Glyph Matching Algorithm | Research Project</p>
      </footer>
    </div>
  );
}

export default App;