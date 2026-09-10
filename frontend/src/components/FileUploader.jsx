import { useState, useRef } from 'react';
import './FileUploader.css';

export default function FileUploader({ onUpload, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.name.endsWith('.csv')) {
      alert('Please upload a CSV file.');
      return;
    }
    setSelectedFile(file);
  };

  const handleSubmit = () => {
    if (selectedFile && onUpload) {
      onUpload(selectedFile);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <div className="file-uploader" id="file-uploader">
      <div
        className={`upload-zone glass-card ${dragActive ? 'upload-zone-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleChange}
          className="upload-input"
          id="csv-file-input"
        />

        <div className="upload-icon">
          <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        </div>
        <h3 className="upload-title">
          {dragActive ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
        </h3>
        <p className="upload-text">
          or click to browse · CSV must have a "text" column
        </p>
      </div>

      {selectedFile && (
        <div className="upload-preview card animate-fade-in-up">
          <div className="upload-file-info">
            <svg className="upload-file-icon" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
            <div className="upload-file-details">
              <span className="upload-file-name">{selectedFile.name}</span>
              <span className="upload-file-size">{formatSize(selectedFile.size)}</span>
            </div>
            <button
              type="button"
              className="upload-remove-btn"
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
              disabled={isLoading}
              title="Remove file"
              aria-label="Remove uploaded file"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
          <button
            className="btn btn-primary upload-submit-btn"
            onClick={handleSubmit}
            disabled={isLoading}
            id="upload-submit-button"
          >
            {isLoading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                Processing...
              </>
            ) : (
              <>Start Batch Analysis</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
