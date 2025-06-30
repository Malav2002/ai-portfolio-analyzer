'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface UploadedFile extends File {
  preview?: string;
}

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
      alert(`Upload failed: ${error.message}`);
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

        {files.length > 0 && !result && (
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
                      disabled={isUploading}
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
              onClick={uploadAndAnalyze}
              disabled={isUploading}
              className={`w-full font-semibold py-3 px-6 rounded-lg transition-colors ${
                isUploading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isUploading ? '🔄 Processing...' : '🚀 Analyze Portfolio'}
            </button>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6 border-t pt-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">✅ Analysis Complete!</h3>
              <button
                onClick={() => {
                  setResult(null);
                  setFiles([]);
                  setPortfolioName('');
                }}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Upload Another
              </button>
            </div>

            {/* OCR Results */}
            {result.data.ocr_result && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">
                  📄 Extracted Text (Confidence: {(result.data.ocr_result.confidence * 100).toFixed(1)}%)
                </h4>
                <p className="text-sm text-gray-700 font-mono bg-white p-3 rounded border">
                  {result.data.ocr_result.text}
                </p>
              </div>
            )}

            {/* Portfolio Summary */}
            {result.data.portfolio && (
              <div>
                <h4 className="font-medium text-gray-900 mb-3">📊 Portfolio Summary</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-blue-50 rounded-lg p-4 text-center">
                    <p className="text-sm text-blue-600 font-medium">Total Value</p>
                    <p className="text-2xl font-bold text-blue-900">
                      ${result.data.portfolio.total_value?.toLocaleString() || 'N/A'}
                    </p>
                  </div>
                  <div className="bg-green-50 rounded-lg p-4 text-center">
                    <p className="text-sm text-green-600 font-medium">Holdings</p>
                    <p className="text-2xl font-bold text-green-900">
                      {result.data.portfolio.holdings_count}
                    </p>
                  </div>
                  <div className="bg-purple-50 rounded-lg p-4 text-center">
                    <p className="text-sm text-purple-600 font-medium">Broker</p>
                    <p className="text-2xl font-bold text-purple-900 capitalize">
                      {result.data.portfolio.broker}
                    </p>
                  </div>
                </div>
{/* Live Market Data Section */}
{result.data.portfolio.live_total_value && (
  <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 mb-6">
    <div className="flex items-center justify-between mb-4">
      <h4 className="text-lg font-semibold text-gray-900 flex items-center">
        📈 Live Market Data
        <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
          REAL-TIME
        </span>
      </h4>
      <div className="text-xs text-gray-500">
        Updated: {result.data.portfolio.market_data_timestamp ? 
          new Date(result.data.portfolio.market_data_timestamp).toLocaleTimeString() : 
          'Just now'}
      </div>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
      <div className="bg-white rounded-lg p-4 text-center shadow-sm">
        <p className="text-sm text-blue-600 font-medium">Live Portfolio Value</p>
        <p className="text-2xl font-bold text-blue-900">
          ${result.data.portfolio.live_total_value?.toLocaleString()}
        </p>
        <p className="text-xs text-gray-500">vs ${result.data.portfolio.total_value?.toLocaleString()} parsed</p>
      </div>
      
      <div className="bg-white rounded-lg p-4 text-center shadow-sm">
        <p className="text-sm text-green-600 font-medium">Live Gain/Loss</p>
        <p className={`text-2xl font-bold ${
          (result.data.portfolio.live_total_gain_loss || 0) >= 0 ? 'text-green-900' : 'text-red-900'
        }`}>
          {(result.data.portfolio.live_total_gain_loss || 0) >= 0 ? '+' : ''}
          ${result.data.portfolio.live_total_gain_loss?.toFixed(2) || '0.00'}
        </p>
        <p className={`text-sm ${
          (result.data.portfolio.live_total_gain_loss_percent || 0) >= 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {(result.data.portfolio.live_total_gain_loss_percent || 0) >= 0 ? '+' : ''}
          {result.data.portfolio.live_total_gain_loss_percent?.toFixed(2) || '0.00'}%
        </p>
      </div>
      
      <div className="bg-white rounded-lg p-4 text-center shadow-sm">
        <p className="text-sm text-purple-600 font-medium">Real Data</p>
        <p className="text-2xl font-bold text-purple-900">
          {result.data.portfolio.real_data_count || 0}
        </p>
        <p className="text-xs text-gray-500">
          of {result.data.portfolio.holdings_with_live_data || 0} holdings
        </p>
      </div>
      
      <div className="bg-white rounded-lg p-4 text-center shadow-sm">
        <p className="text-sm text-orange-600 font-medium">Data Source</p>
        <p className="text-lg font-bold text-orange-900">
          {result.data.portfolio.real_data_count === result.data.portfolio.holdings_with_live_data ? 
            '✅ All Real' : 
            `${result.data.portfolio.mock_data_count || 0} Mock`}
        </p>
      </div>
    </div>
  </div>
)}

{/* Enhanced Holdings Table */}
{result.data.portfolio.holdings && result.data.portfolio.holdings.length > 0 && (
  <div>
    <h4 className="font-medium text-gray-900 mb-3">📊 Live Holdings Data</h4>
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Quantity</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Live Price</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Live Value</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Price Change</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Data Source</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {result.data.portfolio.holdings.map((holding: any, index: number) => (
            <tr key={index} className={holding.is_real_data ? 'bg-green-50' : 'bg-yellow-50'}>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center">
                  <span className="font-medium text-gray-900">{holding.symbol}</span>
                  {holding.is_real_data && <span className="ml-2 text-green-600">✅</span>}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                {holding.quantity || '-'}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-gray-900 font-medium">
                  ${holding.live_price?.toFixed(2) || holding.current_price?.toFixed(2) || '-'}
                </div>
                {holding.live_price && (
                  <div className={`text-xs ${holding.price_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {holding.price_change >= 0 ? '+' : ''}${holding.price_change?.toFixed(2)} 
                    ({holding.price_change_percent?.toFixed(2)}%)
                  </div>
                )}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                <div className="font-medium">
                  ${holding.live_market_value?.toLocaleString() || holding.market_value?.toLocaleString() || '-'}
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                {holding.price_change !== undefined ? (
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    holding.price_change >= 0 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {holding.price_change >= 0 ? '↗️' : '↘️'} 
                    {holding.price_change >= 0 ? '+' : ''}
                    {holding.price_change_percent?.toFixed(2)}%
                  </span>
                ) : '-'}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  holding.is_real_data 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-yellow-100 text-yellow-800'
                }`}>
                  {holding.data_source || 'unknown'}
                </span>
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
          </div>
        )}
      </div>
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
