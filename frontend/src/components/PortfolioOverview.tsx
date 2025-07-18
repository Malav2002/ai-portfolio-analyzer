'use client';

import { PortfolioAnalysis } from '@/services/api';
import { AlertCircle, TrendingUp, TrendingDown, DollarSign, BarChart3 } from 'lucide-react';

interface PortfolioOverviewProps {
  analysis: PortfolioAnalysis;
}

export default function PortfolioOverview({ analysis }: PortfolioOverviewProps) {
  // CRASH-SAFE VERSION: Check if we have the required data structure
  console.log('🔍 PortfolioOverview received analysis:', analysis);
  
  if (!analysis || !analysis.analysis || !analysis.analysis.portfolio_data) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-center space-x-2 mb-3">
          <AlertCircle className="w-6 h-6 text-yellow-600" />
          <h4 className="text-lg font-medium text-yellow-800">Portfolio Overview Unavailable</h4>
        </div>
        <p className="text-yellow-700 mb-4">
          Portfolio analysis data is not available or incomplete. This usually happens when:
        </p>
        <ul className="list-disc list-inside text-yellow-700 space-y-1">
          <li>No holdings were detected in the portfolio screenshot</li>
          <li>The image quality was too poor for analysis</li>
          <li>The portfolio format is not supported</li>
        </ul>
        <div className="mt-4 p-3 bg-yellow-100 rounded border">
          <p className="text-sm text-yellow-800">
            <strong>Debug Info:</strong> Analysis success: {analysis?.success ? 'Yes' : 'No'}, 
            Holdings count: {analysis?.analysis?.portfolio_data?.holdings_count || 0}
          </p>
        </div>
      </div>
    );
  }

  // If we have data, extract it safely with proper null checks
  const portfolioData = analysis.analysis.portfolio_data;
  const aiInsights = analysis.analysis.ai_insights || {};
  const portfolioMetrics = aiInsights.portfolio_metrics || {};
  const diversification = aiInsights.diversification || {};

  // Safe values with fallbacks and null checks
  const totalValue = portfolioData.total_live_value || 
                    portfolioData.total_value || 
                    portfolioMetrics.total_value || 0;
  
  const totalReturnPercent = portfolioMetrics.total_return_percent || 0;
  const numHoldings = portfolioData.holdings_count || 0;
  const qualityScore = aiInsights.quality_score || 0;
  const sectorWeights = diversification.sector_weights || {};
  const concentrationRisk = diversification.concentration_risk || 'Unknown';
  const diversificationScore = diversification.diversification_score || 0;

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

  return (
    <div className="space-y-6">
      {/* Portfolio Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-medium text-blue-900">Portfolio Value</h4>
            <DollarSign className="w-6 h-6 text-blue-600" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-blue-700">Total Value:</span>
              <span className="font-medium text-blue-900">
                {formatCurrency(totalValue)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-700">Holdings:</span>
              <span className="font-medium text-blue-900">
                {formatNumber(numHoldings)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-blue-700">Avg Position:</span>
              <span className="font-medium text-blue-900">
                {formatCurrency(numHoldings > 0 ? totalValue / numHoldings : 0)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-medium text-yellow-900">Diversification</h4>
            <BarChart3 className="w-6 h-6 text-yellow-600" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-yellow-700">Score:</span>
              <span className="font-medium text-yellow-900">
                {formatDecimal(diversificationScore)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-yellow-700">Risk Level:</span>
              <span className={`font-medium ${
                concentrationRisk === 'High' 
                  ? 'text-red-600'
                  : concentrationRisk === 'Moderate'
                  ? 'text-yellow-600'
                  : 'text-green-600'
              }`}>
                {concentrationRisk}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-yellow-700">Sectors:</span>
              <span className="font-medium text-yellow-900">
                {Object.keys(sectorWeights).length}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-medium text-green-900">Performance</h4>
            {totalReturnPercent >= 0 ? (
              <TrendingUp className="w-6 h-6 text-green-600" />
            ) : (
              <TrendingDown className="w-6 h-6 text-red-600" />
            )}
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-green-700">Total Return:</span>
              <span className={`font-medium ${
                totalReturnPercent >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {formatPercentage(totalReturnPercent)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-green-700">Quality Score:</span>
              <span className="font-medium text-green-900">
                {formatDecimal(qualityScore, 0)}/100
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-green-700">Broker:</span>
              <span className="font-medium text-green-900">
                {portfolioData.broker || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-purple-50 rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-lg font-medium text-purple-900">Analysis</h4>
            <BarChart3 className="w-6 h-6 text-purple-600" />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-purple-700">Confidence:</span>
              <span className="font-medium text-purple-900">
                {formatPercentage(portfolioData.confidence)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-700">Recommendations:</span>
              <span className="font-medium text-purple-900">
                {analysis.analysis.recommendations?.length || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-purple-700">Risk Analysis:</span>
              <span className="font-medium text-purple-900">
                {analysis.analysis.risk_analysis ? 'Available' : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Sector Breakdown */}
      {Object.keys(sectorWeights).length > 0 && (
        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-medium text-gray-900 mb-4">Sector Allocation</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(sectorWeights).map(([sector, weight]) => (
              <div key={sector} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-gray-700 capitalize">{sector}</span>
                <span className="font-medium text-gray-900">
                  {formatPercentage(weight as number)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional Metrics */}
      <div className="bg-white rounded-lg border p-6">
        <h4 className="text-lg font-medium text-gray-900 mb-4">Additional Metrics</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-medium text-gray-900 mb-3">Portfolio Metrics</h5>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Holdings:</span>
                <span className="font-medium">{formatNumber(numHoldings)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total Value:</span>
                <span className="font-medium">{formatCurrency(totalValue)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Position Size:</span>
                <span className="font-medium">
                  {formatCurrency(numHoldings > 0 ? totalValue / numHoldings : 0)}
                </span>
              </div>
            </div>
          </div>
          <div>
            <h5 className="font-medium text-gray-900 mb-3">Analysis Quality</h5>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Detection Confidence:</span>
                <span className="font-medium">{formatPercentage(portfolioData.confidence)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Quality Score:</span>
                <span className="font-medium">{formatDecimal(qualityScore, 0)}/100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Diversification Score:</span>
                <span className="font-medium">{formatDecimal(diversificationScore)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}