'use client';

import { useState } from 'react';
import { PortfolioAnalysis } from '@/services/api';
import PortfolioOverview from './PortfolioOverview';
import HoldingsTable from './HoldingsTable';
import RecommendationsPanel from './RecommendationsPanel';
import RiskAnalysisPanel from './RiskAnalysisPanel';
import { BarChart3, TrendingUp, Shield, Lightbulb } from 'lucide-react';

interface PortfolioDashboardProps {
  analysis: PortfolioAnalysis;
}

export default function PortfolioDashboard({ analysis }: PortfolioDashboardProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'holdings' | 'recommendations' | 'risk'>('overview');

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'holdings', label: 'Holdings', icon: TrendingUp },
    { id: 'recommendations', label: 'AI Insights', icon: Lightbulb },
    { id: 'risk', label: 'Risk Analysis', icon: Shield },
  ] as const;

  // Safe data extraction with null checks
  const portfolioData = analysis?.analysis?.portfolio_data || {};
  const aiInsights = analysis?.analysis?.ai_insights || {};
  const portfolioMetrics = aiInsights.portfolio_metrics || {};
  const riskAnalysis = analysis?.analysis?.risk_analysis || {};
  const aiRiskPrediction = riskAnalysis.ai_risk_prediction || {};

  // Safe formatting functions
  const formatNumber = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    return num.toLocaleString();
  };

  const formatCurrency = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '$0';
    return `$${num.toLocaleString()}`;
  };

  const formatPercentage = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0.00%';
    return `${num.toFixed(2)}%`;
  };

  const formatDecimal = (num: number | undefined | null, decimals: number = 2): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0.00';
    return num.toFixed(decimals);
  };

  // Safe values with fallbacks
  const totalValue = portfolioData.total_live_value || 
                    portfolioData.total_value || 
                    portfolioMetrics.total_value || 0;
  
  const totalReturnPercent = portfolioMetrics.total_return_percent || 0;
  const qualityScore = aiInsights.quality_score || 0;
  const riskLevel = aiRiskPrediction.predicted_risk_level || 'Unknown';

  return (
    <div className="space-y-6">
      {/* Header with key metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-1">Portfolio Value</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(totalValue)}
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-1">Total Return</p>
          <p className={`text-2xl font-bold ${
            totalReturnPercent >= 0 
              ? 'text-green-600' 
              : 'text-red-600'
          }`}>
            {formatPercentage(totalReturnPercent)}
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-1">Quality Score</p>
          <p className="text-2xl font-bold text-blue-600">
            {formatDecimal(qualityScore, 0)}/100
          </p>
        </div>
        <div className="card text-center">
          <p className="text-sm text-gray-600 mb-1">Risk Level</p>
          <p className={`text-lg font-bold capitalize ${
            riskLevel === 'high' 
              ? 'text-red-600'
              : riskLevel === 'moderate'
              ? 'text-yellow-600'
              : riskLevel === 'low'
              ? 'text-green-600'
              : 'text-gray-600'
          }`}>
            {riskLevel}
          </p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'overview' && <PortfolioOverview analysis={analysis} />}
        {activeTab === 'holdings' && <HoldingsTable analysis={analysis} />}
        {activeTab === 'recommendations' && <RecommendationsPanel analysis={analysis} />}
        {activeTab === 'risk' && <RiskAnalysisPanel analysis={analysis} />}
      </div>
    </div>
  );
}