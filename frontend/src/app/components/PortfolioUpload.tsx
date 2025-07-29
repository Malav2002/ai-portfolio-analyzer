'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, AlertCircle, Camera, TrendingUp, CheckCircle, Brain, Star } from 'lucide-react'
import { analyzePortfolio } from '../utils/api'

interface PortfolioUploadProps {
  onAnalysisComplete: (data: any) => void
  onAnalysisStart: () => void
  isAnalyzing: boolean
}

export default function PortfolioUpload({ 
  onAnalysisComplete, 
  onAnalysisStart, 
  isAnalyzing 
}: PortfolioUploadProps) {
  const [error, setError] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)

  const getProgressStep = (progress: number) => {
    if (progress < 25) return { text: 'Uploading image...', icon: Upload, color: 'from-blue-500 to-indigo-600' }
    if (progress < 50) return { text: 'AI reading portfolio...', icon: Camera, color: 'from-purple-500 to-pink-600' }
    if (progress < 75) return { text: 'Getting market data...', icon: TrendingUp, color: 'from-green-500 to-emerald-600' }
    if (progress < 95) return { text: 'Generating insights...', icon: Brain, color: 'from-indigo-500 to-purple-600' }
    return { text: 'Analysis complete!', icon: CheckCircle, color: 'from-green-500 to-emerald-600' }
  }

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    const file = acceptedFiles[0]
    setError(null)
    onAnalysisStart()
    
    try {
      const result = await analyzePortfolio(file, (progress) => {
        setUploadProgress(progress)
      })
      
      setTimeout(() => {
        onAnalysisComplete(result)
        setUploadProgress(0)
      }, 1000)
      
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please try again.')
      setUploadProgress(0)
    }
  }, [onAnalysisComplete, onAnalysisStart])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
    disabled: isAnalyzing
  })

  const currentStepInfo = getProgressStep(uploadProgress)

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'dragover' : ''} ${
          isAnalyzing ? 'opacity-75 cursor-not-allowed' : 'cursor-pointer group'
        }`}
      >
        <input {...getInputProps()} />
        
        {isAnalyzing ? (
          <div className="space-y-8">
            {/* Animated processing */}
            <div className="relative flex justify-center">
              <div className={`w-32 h-32 bg-gradient-to-br ${currentStepInfo.color} rounded-3xl flex items-center justify-center shadow-2xl animate-pulse-slow relative overflow-hidden`}>
                <currentStepInfo.icon className="w-16 h-16 text-white z-10" />
              </div>
              <div className="absolute inset-0 w-32 h-32 mx-auto bg-gradient-to-br from-blue-400/30 to-indigo-500/30 rounded-3xl animate-ping"></div>
            </div>
            
            <div className="space-y-6">
              <h3 className="text-3xl font-bold text-gray-800">
                AI is Analyzing Your Portfolio
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-center space-x-3">
                  <Star className="w-6 h-6 text-blue-500 animate-spin" />
                  <p className="text-xl font-bold text-gray-700">
                    {currentStepInfo.text}
                  </p>
                </div>
                
                {/* Progress bar */}
                <div className="relative w-full bg-gray-200/60 rounded-full h-6 shadow-inner overflow-hidden">
                  <div 
                    className="progress-bar h-6 flex items-center justify-center transition-all duration-700 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  >
                    <span className="text-sm text-white font-bold z-10">
                      {Math.round(uploadProgress)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Upload icon */}
            <div className="relative flex justify-center">
              <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl flex items-center justify-center shadow-2xl animate-float group-hover:scale-110 transition-transform duration-500">
                <Upload className="w-16 h-16 text-white" />
              </div>
              {isDragActive && (
                <div className="absolute inset-0 w-32 h-32 mx-auto bg-gradient-to-br from-blue-400/70 to-indigo-500/70 rounded-3xl animate-ping"></div>
              )}
            </div>
            
            <div className="space-y-6">
              <h3 className="text-4xl font-black text-gray-800">
                {isDragActive ? (
                  <span className="text-gradient animate-pulse">Drop Your Portfolio! 🎯</span>
                ) : (
                  'Upload Portfolio Screenshot'
                )}
              </h3>
              
              <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed font-medium">
                {isDragActive ? (
                  'Release to unleash AI-powered analysis'
                ) : (
                  'Drag & drop your portfolio screenshot or click to browse files'
                )}
              </p>
              
              {/* File formats */}
              <div className="flex items-center justify-center space-x-4 flex-wrap">
                {['PNG', 'JPG', 'JPEG'].map((format, index) => (
                  <span key={format} 
                        className="px-4 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 rounded-full text-sm font-bold shadow-md hover:scale-110 transition-transform duration-300">
                    {format}
                  </span>
                ))}
                <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                <span className="text-sm text-gray-500 font-medium">Maximum 10MB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-8 p-8 bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-3xl shadow-xl animate-slide-up">
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-gradient-to-br from-red-500 to-rose-600 rounded-2xl shadow-lg">
              <AlertCircle className="h-8 w-8 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-red-800 font-bold text-lg mb-2">Upload Error</h4>
              <p className="text-red-700 leading-relaxed">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="mt-4 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg font-medium transition-colors duration-200"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
