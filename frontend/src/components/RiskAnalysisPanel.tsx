'use client';

import { PortfolioAnalysis } from '@/services/api';
import { AlertTriangle, TrendingDown, Shield, AlertCircle } from 'lucide-react';

interface RiskAnalysisPanelProps {
  analysis: PortfolioAnalysis;
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

export default function RiskAnalysisPanel({ analysis }: RiskAnalysisPanelProps) {
  console.log('🔍 RiskAnalysisPanel received analysis:', analysis);

  // Safe data extraction with null checks
  const riskAnalysis = analysis?.analysis?.risk_analysis || {};
  const aiRiskPrediction = riskAnalysis.ai_risk_prediction || {};
  const traditionalMetrics = riskAnalysis.traditional_metrics || {};
  const riskFactors = riskAnalysis.risk_factors || [];

  // Check if we have risk analysis data
  if (!riskAnalysis || Object.keys(riskAnalysis).length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-center space-x-2 mb-3">
          <AlertCircle className="w-6 h-6 text-yellow-600" />
          <h4 className="text-lg font-medium text-yellow-800">Risk Analysis Unavailable</h4>
        </div>
        <p className="text-yellow-700 mb-4">
          Risk analysis data is not available. This could be because:
        </p>
        <ul className="list-disc list-inside text-yellow-700 space-y-1">
          <li>The portfolio analysis is still in progress</li>
          <li>Not enough data was available for risk calculations</li>
          <li>The analysis failed to complete</li>
        </ul>
      </div>
    );
  }

  // Safe values with fallbacks
  const overallRiskScore = riskAnalysis.overall_risk_score || 0;
  const riskLevel = aiRiskPrediction.predicted_risk_level || 'Unknown';
  const confidence = aiRiskPrediction.confidence || 0;
  const riskDistribution = aiRiskPrediction.risk_distribution || {};

  // Traditional metrics with safe defaults
  const portfolioVolatility = traditionalMetrics.portfolio_volatility || 0;
  const maxDrawdown = traditionalMetrics.max_drawdown || 0;
  const sharpeRatio = traditionalMetrics.sharpe_ratio || 0;
  const beta = traditionalMetrics.beta || 0;

  // Risk level styling and icons
  const getRiskIcon = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'high':
        return <AlertTriangle className="w-8 h-8 text-red-500" />;
      case 'moderate':
        return <AlertCircle className="w-8 h-8 text-yellow-500" />;
      case 'low':
        return <Shield className="w-8 h-8 text-green-500" />;
      default:
        return <AlertCircle className="w-8 h-8 text-gray-500" />;
    }
  };

  const getRiskColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'high':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'moderate':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'low':
        return 'bg-green-50 border-green-200 text-green-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* AI Risk Assessment */}
      <div className={`rounded-lg border p-6 ${getRiskColor(riskLevel)}`}>
        <div className="flex items-center space-x-3 mb-4">
          {getRiskIcon(riskLevel)}
          <div>
            <h4 className="text-xl font-bold capitalize">
              {riskLevel} Risk Level
            </h4>
            <p className="text-sm opacity-80">
              AI Confidence: {safeFormatPercentage(confidence * 100)}
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm font-medium mb-1">Risk Score</p>
            <p className="text-2xl font-bold">{safeFormatDecimal(overallRiskScore * 100, 0)}/100</p>
          </div>
          <div>
            <p className="text-sm font-medium mb-1">Risk Distribution</p>
            <div className="space-y-1">
              {Object.entries(riskDistribution).map(([level, prob]) => (
                <div key={level} className="flex justify-between text-sm">
                  <span className="capitalize">{level}:</span>
                  <span>{safeFormatPercentage((prob as number) * 100)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Factors */}
      {riskFactors.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h4 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
            Identified Risk Factors
          </h4>
          <ul className="space-y-3">
            {riskFactors.map((factor, index) => (
              <li key={index} className="flex items-start space-x-3">
                <span className="flex-shrink-0 w-2 h-2 bg-red-500 rounded-full mt-2"></span>
                <span className="text-gray-700">{factor}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Traditional Risk Metrics */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h4 className="text-lg font-medium text-gray-900 mb-4">Risk Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Volatility</p>
            <p className="text-xl font-bold text-gray-900">
              {safeFormatDecimal(portfolioVolatility, 1)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Max Drawdown</p>
            <p className="text-xl font-bold text-red-600">
              -{safeFormatDecimal(maxDrawdown, 1)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Sharpe Ratio</p>
            <p className="text-xl font-bold text-blue-600">
              {safeFormatDecimal(sharpeRatio, 2)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-1">Beta</p>
            <p className="text-xl font-bold text-purple-600">
              {safeFormatDecimal(beta, 2)}
            </p>
          </div>
        </div>
      </div>

      {/* Risk Analysis Summary */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h4 className="text-lg font-medium text-gray-900 mb-4">Risk Analysis Summary</h4>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Overall Risk Score:</span>
            <span className="font-medium text-gray-900">
              {safeFormatDecimal(overallRiskScore * 100, 0)}/100
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Risk Level:</span>
            <span className={`font-medium capitalize ${
              riskLevel === 'high' ? 'text-red-600' :
              riskLevel === 'moderate' ? 'text-yellow-600' :
              riskLevel === 'low' ? 'text-green-600' : 'text-gray-600'
            }`}>
              {riskLevel}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">AI Confidence:</span>
            <span className="font-medium text-gray-900">
              {safeFormatPercentage(confidence * 100)}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Risk Factors Identified:</span>
            <span className="font-medium text-gray-900">
              {riskFactors.length}
            </span>
          </div>
        </div>
      </div>

      {/* Risk Recommendations */}
      <div className="bg-blue-50 rounded-lg border border-blue-200 p-6">
        <h4 className="text-lg font-medium text-blue-900 mb-4">Risk Management Recommendations</h4>
        <div className="space-y-3">
          {riskLevel === 'high' && (
            <>
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                <p className="text-gray-700">
                  Consider reducing position sizes in high-risk holdings
                </p>
              </div>
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                <p className="text-gray-700">
                  Diversify across different sectors and asset classes
                </p>
              </div>
            </>
          )}
          {riskLevel === 'moderate' && (
            <>
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                <p className="text-gray-700">
                  Monitor portfolio performance and volatility regularly
                </p>
              </div>
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5" />
                <p className="text-gray-700">
                  Consider adding defensive positions if risk increases
                </p>
              </div>
            </>
          )}
          {riskLevel === 'low' && (
            <>
              <div className="flex items-start space-x-3">
                <Shield className="w-5 h-5 text-green-600 mt-0.5" />
                <p className="text-gray-700">
                  Portfolio shows good risk management characteristics
                </p>
              </div>
              <div className="flex items-start space-x-3">
                <Shield className="w-5 h-5 text-green-600 mt-0.5" />
                <p className="text-gray-700">
                  Continue monitoring and maintain current allocation strategy
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}