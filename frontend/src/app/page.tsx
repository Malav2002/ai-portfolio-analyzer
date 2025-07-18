'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadedFile extends File {
  preview: string;
}

// COMPLETELY SAFE formatting utility functions
const safeFormatNumber = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0';
  }
  return num.toLocaleString();
};

const safeFormatCurrency = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '$0.00';
  }
  return `$${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

const safeFormatPercentage = (num: any): string => {
  // SAFE CHECK BEFORE calling toFixed()
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0.00%';
  }
  return `${num.toFixed(2)}%`;
};

const safeFormatDecimal = (num: any, decimals: number = 2): string => {
  // SAFE CHECK BEFORE calling toFixed()
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    // Return proper number of zeros based on decimals
    return '0.' + '0'.repeat(decimals);
  }
  return num.toFixed(decimals);
};

// Additional safe formatter for file sizes
const safeFormatFileSize = (bytes: any): string => {
  if (typeof bytes !== 'number' || isNaN(bytes) || bytes === null || bytes === undefined) {
    return '0.0';
  }
  const megabytes = bytes / (1024 * 1024);
  return megabytes.toFixed(1);
};

function UploadForm() {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [portfolioName, setPortfolioName] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map(file => 
      Object.assign(file, {
        preview: URL.createObjectURL(file)
      })
    );
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
    setResult(null); // Reset previous results
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    },
    maxFiles: 5,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const uploadAndAnalyze = async () => {
    if (files.length === 0) {
      alert('Please select at least one image');
      return;
    }

    setIsUploading(true);
    
    try {
      const file = files[0]; // Process first file
      const formData = new FormData();
      formData.append('image', file);
      if (portfolioName) {
        formData.append('snapshot_name', portfolioName);
      }

      const response = await fetch('http://localhost:3001/api/portfolio/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
      
    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

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

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-400 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p className="text-blue-600">Drop the files here...</p>
          ) : (
            <div className="space-y-2">
              <p className="text-gray-600">
                Drag and drop portfolio screenshots here, or click to select files
              </p>
              <p className="text-sm text-gray-500">
                Supports PNG, JPG, GIF (max 10MB per file)
              </p>
            </div>
          )}
        </div>

        {files.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Selected Files:</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {files.map((file, index) => (
                <div key={index} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <img 
                      src={file.preview} 
                      alt={`Preview ${index}`} 
                      className="w-16 h-16 object-cover rounded"
                    />
                    <div>
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">
                        {safeFormatFileSize(file.size)} MB
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          onClick={uploadAndAnalyze}
          disabled={isUploading || files.length === 0}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isUploading ? 'Analyzing...' : 'Upload & Analyze Portfolio'}
        </button>
      </div>

      {result && (
        <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
          <div className="flex items-center space-x-2">
            <span className="text-2xl">✅</span>
            <h2 className="text-xl font-bold text-gray-900">Analysis Complete!</h2>
          </div>

          {result.success && result.data && (
            <div className="space-y-6">
              {/* Portfolio Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-blue-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-blue-600 font-medium">Total Holdings</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {result.data.portfolio?.holdings?.length || 0}
                  </p>
                </div>
                <div className="bg-green-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-green-600 font-medium">Total Value</p>
                  <p className="text-2xl font-bold text-green-900">
                    {safeFormatCurrency(result.data.portfolio?.totalValue)}
                  </p>
                </div>
                <div className="bg-purple-50 rounded-lg p-4 text-center">
                  <p className="text-sm text-purple-600 font-medium">Analysis Score</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {safeFormatDecimal((result.data.analysis?.confidence || 0) * 100, 0)}%
                  </p>
                </div>
              </div>

              {/* Holdings Table */}
              {result.data.portfolio?.holdings && result.data.portfolio.holdings.length > 0 && (
                <div className="bg-white rounded-lg border overflow-hidden">
                  <div className="px-6 py-4 bg-gray-50 border-b">
                    <h3 className="text-lg font-medium text-gray-900">Portfolio Holdings</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Quantity</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gain/Loss</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {result.data.portfolio.holdings.map((holding: any, index: number) => (
                          <tr key={index}>
                            <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                              {holding.symbol || 'N/A'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                              {safeFormatNumber(holding.quantity)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                              {safeFormatCurrency(holding.price)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                              {safeFormatCurrency(holding.value)}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              {holding.gainLoss !== undefined && holding.gainLoss !== null ? (
                                <span className={holding.gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}>
                                  {holding.gainLoss >= 0 ? '+' : ''}{safeFormatCurrency(holding.gainLoss)}
                                </span>
                              ) : (
                                <span className="text-gray-500">-</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error or No Data Message */}
          {!result.success && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <span className="text-red-400 text-xl">❌</span>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Analysis Failed</h3>
                  <div className="mt-2 text-sm text-red-700">
                    <p>{result.error || 'Unknown error occurred during analysis'}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <UploadForm />
    </div>
  );
}