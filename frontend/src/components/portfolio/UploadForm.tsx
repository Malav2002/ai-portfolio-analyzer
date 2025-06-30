'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadedFile extends File {
  preview?: string;
}

export default function UploadForm() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [portfolioName, setPortfolioName] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map(file => 
      Object.assign(file, {
        preview: URL.createObjectURL(file)
      })
    );
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    },
    maxFiles: 5,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          📸 Upload Portfolio Screenshot
        </h1>
        <p className="text-gray-600">
          Upload a screenshot from any broker app and let AI extract your portfolio data
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
        
        {/* Portfolio Name Input */}
        <div>
          <label htmlFor="portfolioName" className="block text-sm font-medium text-gray-700 mb-2">
            Portfolio Name (Optional)
          </label>
          <input
            type="text"
            id="portfolioName"
            value={portfolioName}
            onChange={(e) => setPortfolioName(e.target.value)}
            placeholder="e.g., My Robinhood Portfolio"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Dropzone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400'
          }`}
        >
          <input {...getInputProps()} />
          
          <div className="text-6xl mb-4">📸</div>
          
          {isDragActive ? (
            <p className="text-lg text-blue-600">Drop the files here...</p>
          ) : (
            <div>
              <p className="text-lg text-gray-600 mb-2">
                Drag & drop portfolio screenshots here, or click to select files
              </p>
              <p className="text-sm text-gray-500">
                PNG, JPG, GIF up to 10MB (max 5 files)
              </p>
            </div>
          )}
        </div>

        {/* File Preview */}
        {files.length > 0 && (
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900">Selected Files:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {files.map((file, index) => (
                <div key={index} className="relative bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">📄</div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {file.name}
                      </p>
                      <p className="text-sm text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(1)} MB
                      </p>
                    </div>
                    <button
                      onClick={() => setFiles(files => files.filter(f => f !== file))}
                      className="text-red-400 hover:text-red-600 text-lg"
                    >
                      ❌
                    </button>
                  </div>
                  
                  {file.preview && (
                    <div className="mt-3">
                      <img
                        src={file.preview}
                        alt="Preview"
                        className="w-full h-32 object-cover rounded-md"
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>

            <button
              onClick={() => alert('Upload feature will be connected to backend next!')}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              🚀 Analyze Portfolio
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
