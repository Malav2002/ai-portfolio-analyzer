'use client'

import { useState, useEffect } from 'react'
import PortfolioUpload from './components/PortfolioUpload'
import AnalysisResults from './components/AnalysisResults'
import { TrendingUp, Shield, Brain, Star, CheckCircle, Target, Zap } from 'lucide-react'

export default function HomePage() {
  const [analysisData, setAnalysisData] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const handleAnalysisComplete = (data: any) => {
    setAnalysisData(data)
    setIsAnalyzing(false)
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
    setAnalysisData(null)
  }

  if (!mounted) return null

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <div className="text-center space-y-8 animate-fade-in-up">
        <div className="inline-flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full border border-blue-200 shadow-lg">
          <Star className="w-5 h-5 text-blue-600 animate-pulse" />
          <span className="text-sm font-bold text-blue-700">Next-Generation AI Portfolio Analysis</span>
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
        </div>
        
        <h2 className="text-5xl md:text-7xl font-black text-gradient leading-tight">
          Transform Screenshots
          <br />
          <span className="relative inline-block">
            into Smart Insights
            <div className="absolute -bottom-4 left-0 right-0 h-2 bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 rounded-full shadow-lg"></div>
          </span>
        </h2>
        
        <p className="text-xl md:text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed font-medium">
          Upload your portfolio screenshot and unlock{' '}
          <span className="text-green-600 font-bold">real-time market analysis</span>,{' '}
          <span className="text-blue-600 font-bold">AI-powered insights</span>, and{' '}
          <span className="text-purple-600 font-bold">personalized investment strategies</span>
        </p>

        {/* Feature highlights */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-5xl mx-auto mt-16">
          {[
            { icon: Brain, title: "AI Analysis", desc: "Advanced neural networks", color: "from-blue-500 to-indigo-600" },
            { icon: Zap, title: "Live Data", desc: "Real-time market prices", color: "from-green-500 to-emerald-600" },
            { icon: Shield, title: "Risk Scoring", desc: "Comprehensive assessment", color: "from-yellow-500 to-orange-600" },
            { icon: Target, title: "Smart Tips", desc: "Personalized recommendations", color: "from-purple-500 to-pink-600" }
          ].map((feature, index) => (
            <div key={index} 
                 className="glass-effect rounded-3xl p-8 hover:scale-110 transition-all duration-500 animate-slide-up group" 
                 style={{animationDelay: `${index * 0.1}s`}}>
              <div className={`p-4 bg-gradient-to-br ${feature.color} rounded-2xl w-fit mx-auto mb-4 shadow-xl group-hover:rotate-12 transition-transform duration-300`}>
                <feature.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg text-gray-900 mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-600 leading-relaxed">{feature.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Upload Section */}
      <div className="animate-fade-in-up" style={{animationDelay: '0.5s'}}>
        <PortfolioUpload 
          onAnalysisComplete={handleAnalysisComplete}
          onAnalysisStart={handleAnalysisStart}
          isAnalyzing={isAnalyzing}
        />
      </div>

      {/* Results Section */}
      {analysisData && (
        <div className="animate-fade-in-up" style={{animationDelay: '0.2s'}}>
          <AnalysisResults data={analysisData} />
        </div>
      )}

      {/* Demo Preview */}
      {!analysisData && !isAnalyzing && (
        <div className="mt-20 space-y-12 animate-fade-in-up" style={{animationDelay: '0.8s'}}>
          <div className="text-center space-y-4">
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full">
              <Star className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-bold text-purple-700">Preview Results</span>
            </div>
            <h3 className="text-3xl font-bold text-gray-800">See What You'll Get</h3>
            <p className="text-gray-600 text-lg">Real portfolio analysis with actionable insights</p>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="metric-card text-center group hover:shadow-glow-green">
              <div className="p-4 bg-gradient-to-br from-green-500 to-emerald-600 rounded-3xl w-fit mx-auto mb-6 shadow-xl group-hover:scale-125 transition-transform duration-500">
                <TrendingUp className="h-10 w-10 text-white" />
              </div>
              <p className="text-4xl font-black text-green-600 mb-3">$247,891</p>
              <p className="text-gray-600 font-bold">Portfolio Value</p>
              <div className="mt-3 h-1 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full"></div>
            </div>
            
            <div className="metric-card text-center group hover:shadow-glow">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl w-fit mx-auto mb-6 shadow-xl group-hover:scale-125 transition-transform duration-500">
                <TrendingUp className="h-10 w-10 text-white" />
              </div>
              <p className="text-4xl font-black text-blue-600 mb-3">+12.4%</p>
              <p className="text-gray-600 font-bold">Performance</p>
              <div className="mt-3 h-1 bg-gradient-to-r from-blue-400 to-indigo-500 rounded-full"></div>
            </div>
            
            <div className="metric-card text-center group hover:shadow-glow-purple">
              <div className="p-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-3xl w-fit mx-auto mb-6 shadow-xl group-hover:scale-125 transition-transform duration-500">
                <Shield className="h-10 w-10 text-white" />
              </div>
              <p className="text-4xl font-black text-purple-600 mb-3">Medium</p>
              <p className="text-gray-600 font-bold">Risk Level</p>
              <div className="mt-3 h-1 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full"></div>
            </div>
          </div>

          {/* Success stories */}
          <div className="glass-effect rounded-3xl p-8 shadow-glow">
            <div className="text-center space-y-6">
              <h4 className="text-2xl font-bold text-gray-800">Trusted by Smart Investors</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  { metric: "99.2%", label: "Accuracy Rate", icon: CheckCircle },
                  { metric: "50ms", label: "Analysis Speed", icon: Zap },
                  { metric: "10K+", label: "Portfolios Analyzed", icon: Target }
                ].map((stat, index) => (
                  <div key={index} className="text-center">
                    <stat.icon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                    <p className="text-2xl font-bold text-gray-900">{stat.metric}</p>
                    <p className="text-gray-600 font-medium">{stat.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
