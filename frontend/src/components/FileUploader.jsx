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

        <div className="upload-icon">📁</div>
        <h3 className="upload-title">
          {dragActive ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
        </h3>
        <p className="upload-text">
          or click to browse · CSV must have a "text" column
        </p>
      </div>

      {selectedFile && (
        <div className="upload-preview glass-card animate-fade-in-up">
          <div className="upload-file-info">
            <span className="upload-file-icon">📄</span>
            <div className="upload-file-details">
              <span className="upload-file-name">{selectedFile.name}</span>
              <span className="upload-file-size">{formatSize(selectedFile.size)}</span>
            </div>
            <button
              className="btn btn-ghost"
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
              disabled={isLoading}
            >
              ✕
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
              <>🚀 Start Batch Analysis</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
