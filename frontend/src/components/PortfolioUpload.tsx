'use client';

import { useState, useCallback } from 'react';
import { Upload, Image, AlertCircle, Clock, CheckCircle, Loader } from 'lucide-react';
import { portfolioAPI, PortfolioAnalysis } from '@/services/api';

interface PortfolioUploadProps {
  onAnalysisComplete: (analysis: PortfolioAnalysis) => void;
  onStartAnalysis: () => void;
}

interface ProgressStep {
  message: string;
  step: number;
  totalSteps: number;
  timestamp: string;
}

// Enhanced analysis result type that matches backend response
interface EnhancedAnalysisResult {
  success: boolean;
  analysis: {
    portfolio_data: {
      holdings: any[];
      holdings_count: number;
      total_value: number;
      total_live_value?: number;
      broker: string;
      confidence: number;
    };
    ai_insights: any;
    recommendations: any[];
    risk_analysis: any;
    extraction_details: any;
  };
  debug_report?: any;
  error?: string;
}

// Safe formatting utility functions
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
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0.00%';
  }
  return `${num.toFixed(2)}%`;
};

const safeFormatDecimal = (num: any, decimals: number = 2): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0.00';
  }
  return num.toFixed(decimals);
};

// Debug component for analyzing data structure
export function DebugAnalysisData({ analysis }: { analysis: any }) {
  console.log('🔍 Full Analysis Object:', analysis);
  
  const formatValue = (value: any, path: string = ''): string => {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    if (typeof value === 'number') return `${value} (number)`;
    if (typeof value === 'string') return `"${value}" (string)`;
    if (typeof value === 'boolean') return `${value} (boolean)`;
    if (Array.isArray(value)) return `Array[${value.length}]`;
    if (typeof value === 'object') return `Object{${Object.keys(value).length} keys}`;
    return `${value} (${typeof value})`;
  };

  const renderObject = (obj: any, depth: number = 0, path: string = ''): JSX.Element => {
    if (depth > 3) return <span className="text-gray-500">...</span>;
    
    if (obj === null || obj === undefined) {
      return <span className="text-red-600">{formatValue(obj, path)}</span>;
    }
    
    if (typeof obj !== 'object') {
      return <span className="text-blue-600">{formatValue(obj, path)}</span>;
    }
    
    if (Array.isArray(obj)) {
      return (
        <div className="ml-4">
          <span className="text-purple-600">Array[{obj.length}]</span>
          {obj.slice(0, 3).map((item, index) => (
            <div key={index} className="ml-2">
              [{index}]: {renderObject(item, depth + 1, `${path}[${index}]`)}
            </div>
          ))}
          {obj.length > 3 && <div className="ml-2 text-gray-500">... {obj.length - 3} more items</div>}
        </div>
      );
    }
    
    return (
      <div className="ml-4">
        <span className="text-green-600">Object{`{${Object.keys(obj).length} keys}`}</span>
        {Object.entries(obj).slice(0, 10).map(([key, value]) => (
          <div key={key} className="ml-2">
            <span className="text-gray-700 font-medium">{key}:</span>{' '}
            {renderObject(value, depth + 1, `${path}.${key}`)}
          </div>
        ))}
        {Object.keys(obj).length > 10 && (
          <div className="ml-2 text-gray-500">... {Object.keys(obj).length - 10} more keys</div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-50 border rounded-lg p-4 mt-4">
      <h4 className="font-medium text-gray-900 mb-4">🔍 Analysis Data Structure Debug</h4>
      <div className="text-sm font-mono bg-white p-4 rounded border overflow-auto max-h-96">
        {renderObject(analysis)}
      </div>
      
      {/* Quick checks for common issues */}
      <div className="mt-4 p-3 bg-yellow-50 rounded border">
        <h5 className="font-medium text-yellow-800 mb-2">Quick Checks:</h5>
        <div className="space-y-1 text-sm">
          <div>✓ Analysis exists: {analysis ? 'Yes' : 'No'}</div>
          <div>✓ Has analysis.analysis: {analysis?.analysis ? 'Yes' : 'No'}</div>
          <div>✓ Has portfolio_data: {analysis?.analysis?.portfolio_data ? 'Yes' : 'No'}</div>
          <div>✓ Has holdings: {analysis?.analysis?.portfolio_data?.holdings ? 'Yes' : 'No'}</div>
          <div>✓ Holdings count: {analysis?.analysis?.portfolio_data?.holdings?.length || 0}</div>
          <div>✓ Total value type: {typeof analysis?.analysis?.portfolio_data?.total_value}</div>
          <div>✓ Total value: {analysis?.analysis?.portfolio_data?.total_value}</div>
          <div>✓ Has ai_insights: {analysis?.analysis?.ai_insights ? 'Yes' : 'No'}</div>
        </div>
      </div>
    </div>
  );
}

export default function PortfolioUpload({ onAnalysisComplete, onStartAnalysis }: PortfolioUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
  const [currentProgress, setCurrentProgress] = useState<ProgressStep | null>(null);
  const [analysisResult, setAnalysisResult] = useState<EnhancedAnalysisResult | null>(null);

  const addProgressStep = useCallback((message: string, step: number, totalSteps: number) => {
    const timestamp = new Date().toISOString();
    const progressStep = { message, step, totalSteps, timestamp };
    
    console.log(`📊 Progress [${step}/${totalSteps}]: ${message}`);
    
    setCurrentProgress(progressStep);
    setProgressSteps(prev => [...prev, progressStep]);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      setSelectedFile(imageFile);
      setError(null);
      // SAFE file size calculation
      const fileSizeMB = imageFile.size / 1024 / 1024;
      console.log(`📁 File selected: ${imageFile.name} (${safeFormatDecimal(fileSizeMB)} MB)`);
    } else {
      setError('Please select an image file');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      // SAFE file size calculation
      const fileSizeMB = file.size / 1024 / 1024;
      console.log(`📁 File selected: ${file.name} (${safeFormatDecimal(fileSizeMB)} MB)`);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);
    setProgressSteps([]);
    setCurrentProgress(null);
    setAnalysisResult(null);
    onStartAnalysis();

    const startTime = Date.now();
    console.log(`🚀 Starting portfolio analysis at ${new Date().toISOString()}`);

    try {
      addProgressStep('Starting analysis...', 1, 8);
      
      // Create form data
      const formData = new FormData();
      formData.append('image', selectedFile);

      addProgressStep('Uploading image...', 2, 8);

      // Call the backend API
      const response = await fetch('http://localhost:3001/api/ai/analyze', {
        method: 'POST',
        body: formData,
      });

      addProgressStep('Processing with AI...', 3, 8);

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Upload failed: ${response.status} ${errorData}`);
      }

      const result: EnhancedAnalysisResult = await response.json();
      
      addProgressStep('Analyzing portfolio data...', 4, 8);
      
      console.log('📊 Raw backend response:', result);
      
      // Check if analysis was successful
      if (!result.success) {
        throw new Error(result.error || 'Analysis failed');
      }

      addProgressStep('Extracting holdings...', 5, 8);
      
      // SAFE data extraction with null checks
      const portfolioData = result.analysis?.portfolio_data;
      const holdingsCount = portfolioData?.holdings_count || 0;
      const totalValue = portfolioData?.total_value || portfolioData?.total_live_value || 0;
      
      addProgressStep('Generating insights...', 6, 8);
      
      // SAFE logging with proper formatting
      console.log(`📊 Analysis results: ${holdingsCount} holdings, ${safeFormatCurrency(totalValue)} total value`);
      
      addProgressStep('Finalizing results...', 7, 8);
      
      // Store the complete result
      setAnalysisResult(result);
      
      const duration = Date.now() - startTime;
      console.log(`✅ Analysis completed successfully in ${duration}ms`);
      
      addProgressStep('Analysis completed successfully!', 8, 8);
      
      // Pass the properly formatted analysis to parent component
      if (onAnalysisComplete && result.analysis) {
        onAnalysisComplete(result.analysis as PortfolioAnalysis);
      }
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(`Analysis failed: ${errorMessage}`);
      
      addProgressStep(`Analysis failed: ${errorMessage}`, 8, 8);
    } finally {
      setIsUploading(false);
    }
  };

  // Reset function
  const resetAnalysis = () => {
    setSelectedFile(null);
    setAnalysisResult(null);
    setError(null);
    setProgressSteps([]);
    setCurrentProgress(null);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          📊 AI Portfolio Analyzer
        </h2>

        {!analysisResult && (
          <>
            {/* File Upload Area */}
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                isDragOver
                  ? 'border-blue-400 bg-blue-50'
                  : selectedFile
                  ? 'border-green-400 bg-green-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isUploading}
              />

              {selectedFile ? (
                <div className="space-y-4">
                  <CheckCircle className="w-16 h-16 text-green-500 mx-auto" />
                  <div>
                    <h3 className="text-lg font-medium text-green-900">File Selected</h3>
                    <p className="text-green-700">{selectedFile.name}</p>
                    <p className="text-sm text-green-600">
                      {safeFormatDecimal(selectedFile.size / 1024 / 1024)} MB
                    </p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="w-16 h-16 text-gray-400 mx-auto" />
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Upload Portfolio Screenshot</h3>
                    <p className="text-gray-600">
                      Drag and drop your portfolio screenshot here, or click to browse
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      Supports: PNG, JPG, JPEG (max 10MB)
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Upload Button */}
            {selectedFile && (
              <div className="mt-6 flex space-x-4">
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {isUploading ? 'Analyzing...' : 'Analyze Portfolio'}
                </button>
                <button
                  onClick={resetAnalysis}
                  className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  Reset
                </button>
              </div>
            )}

            {/* Progress Steps */}
            {progressSteps.length > 0 && (
              <div className="mt-6 space-y-2">
                <h4 className="font-medium text-gray-900">Analysis Progress</h4>
                {progressSteps.map((step, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span className="text-gray-700">{step.message}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Current Progress */}
            {currentProgress && isUploading && (
              <div className="mt-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-2">
                    <Loader className="w-5 h-5 text-blue-600 animate-spin" />
                    <span className="text-blue-900 font-medium">
                      {currentProgress.message}
                    </span>
                  </div>
                  <div className="w-full bg-blue-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentProgress.step / currentProgress.totalSteps) * 100}%` }}
                    />
                  </div>
                  <p className="text-sm text-blue-700 mt-1">
                    Step {currentProgress.step} of {currentProgress.totalSteps}
                  </p>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="w-5 h-5 text-red-600" />
                  <span className="text-red-800 font-medium">Error</span>
                </div>
                <p className="text-red-700 mt-1">{error}</p>
              </div>
            )}
          </>
        )}

        {/* Analysis Results */}
        {analysisResult && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-gray-900">✅ Analysis Complete!</h3>
              <button
                onClick={resetAnalysis}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                Upload Another
              </button>
            </div>

            {/* Analysis Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <p className="text-sm text-blue-600 font-medium">Holdings Found</p>
                <p className="text-2xl font-bold text-blue-900">
                  {analysisResult.analysis?.portfolio_data?.holdings_count || 0}
                </p>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <p className="text-sm text-green-600 font-medium">Total Value</p>
                <p className="text-2xl font-bold text-green-900">
                  {safeFormatCurrency(
                    analysisResult.analysis?.portfolio_data?.total_value || 
                    analysisResult.analysis?.portfolio_data?.total_live_value || 0
                  )}
                </p>
              </div>
              <div className="bg-purple-50 rounded-lg p-4 text-center">
                <p className="text-sm text-purple-600 font-medium">Recommendations</p>
                <p className="text-2xl font-bold text-purple-900">
                  {analysisResult.analysis?.recommendations?.length || 0}
                </p>
              </div>
            </div>

            {/* Holdings Table */}
            {analysisResult.analysis?.portfolio_data?.holdings && 
             analysisResult.analysis.portfolio_data.holdings.length > 0 && (
              <div className="bg-white rounded-lg border">
                <h4 className="text-lg font-medium text-gray-900 p-4 border-b">📊 Holdings</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Value</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Shares</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Price</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {analysisResult.analysis.portfolio_data.holdings.map((holding: any, index: number) => (
                        <tr key={index}>
                          <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                            {holding.symbol || 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                            {safeFormatCurrency(holding.market_value || holding.total_value || holding.value)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                            {safeFormatNumber(holding.shares || holding.quantity)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-gray-700">
                            {safeFormatCurrency(holding.price || holding.current_price)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* No Holdings Found */}
            {(!analysisResult.analysis?.portfolio_data?.holdings || 
              analysisResult.analysis.portfolio_data.holdings.length === 0) && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                <div className="flex items-center space-x-2 mb-3">
                  <AlertCircle className="w-6 h-6 text-yellow-600" />
                  <h4 className="text-lg font-medium text-yellow-800">No Holdings Detected</h4>
                </div>
                <p className="text-yellow-700 mb-4">
                  The AI couldn't detect any stock holdings in your portfolio screenshot. This could be due to:
                </p>
                <ul className="list-disc list-inside text-yellow-700 space-y-1 mb-4">
                  <li>Image quality is too low for text recognition</li>
                  <li>Portfolio format not supported</li>
                  <li>Stock symbols not clearly visible</li>
                  <li>Screenshot doesn't show the holdings section</li>
                </ul>
                <div className="text-sm text-yellow-600">
                  💡 Try uploading a clearer, higher resolution image showing your portfolio holdings.
                </div>
              </div>
            )}

            {/* Debug Component - Enable this to see data structure */}
            {/* <DebugAnalysisData analysis={analysisResult} /> */}
          </div>
        )}
      </div>
    </div>
  );
}