'use client';

import { PortfolioAnalysis } from '@/services/api';
import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';

interface HoldingsTableProps {
  analysis: PortfolioAnalysis;
}

export default function HoldingsTable({ analysis }: HoldingsTableProps) {
  console.log('🔍 HoldingsTable received analysis:', analysis);

  // Safe data extraction with null checks
  const portfolioData = analysis?.analysis?.portfolio_data;
  const holdings = portfolioData?.holdings || [];

  // Safe formatting functions
  const formatNumber = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    return num.toLocaleString();
  };

  const formatCurrency = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '$0.00';
    return `$${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercentage = (num: number | undefined | null): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0.00%';
    return `${num.toFixed(2)}%`;
  };

  const formatDecimal = (num: number | undefined | null, decimals: number = 2): string => {
    if (typeof num !== 'number' || isNaN(num)) return '0.00';
    return num.toFixed(decimals);
  };

  // Check if we have holdings data
  if (!portfolioData || !holdings || holdings.length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <div className="flex items-center space-x-2 mb-3">
          <AlertCircle className="w-6 h-6 text-yellow-600" />
          <h4 className="text-lg font-medium text-yellow-800">No Holdings Found</h4>
        </div>
        <p className="text-yellow-700 mb-4">
          No stock holdings were detected in your portfolio screenshot. This could be because:
        </p>
        <ul className="list-disc list-inside text-yellow-700 space-y-1">
          <li>The image quality was too poor for text recognition</li>
          <li>The portfolio format is not supported</li>
          <li>Stock symbols are not clearly visible</li>
          <li>The screenshot doesn't show the holdings section</li>
        </ul>
        <div className="mt-4 p-3 bg-yellow-100 rounded border">
          <p className="text-sm text-yellow-800">
            <strong>Debug Info:</strong> 
            Portfolio data exists: {portfolioData ? 'Yes' : 'No'}, 
            Holdings count: {holdings.length}
          </p>
        </div>
      </div>
    );
  }

  // Calculate total portfolio value for percentage calculations
  const totalPortfolioValue = holdings.reduce((sum, holding) => {
    const value = holding.market_value || holding.total_value || holding.value || 0;
    return sum + value;
  }, 0);

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-6">
          <h4 className="text-lg font-medium text-blue-900 mb-2">Total Holdings</h4>
          <p className="text-3xl font-bold text-blue-600">{holdings.length}</p>
        </div>
        <div className="bg-green-50 rounded-lg p-6">
          <h4 className="text-lg font-medium text-green-900 mb-2">Total Value</h4>
          <p className="text-3xl font-bold text-green-600">{formatCurrency(totalPortfolioValue)}</p>
        </div>
        <div className="bg-purple-50 rounded-lg p-6">
          <h4 className="text-lg font-medium text-purple-900 mb-2">Average Position</h4>
          <p className="text-3xl font-bold text-purple-600">
            {formatCurrency(holdings.length > 0 ? totalPortfolioValue / holdings.length : 0)}
          </p>
        </div>
        <div className="bg-orange-50 rounded-lg p-6">
          <h4 className="text-lg font-medium text-orange-900 mb-2">Largest Position</h4>
          <p className="text-3xl font-bold text-orange-600">
            {formatCurrency(Math.max(...holdings.map(h => h.market_value || h.total_value || h.value || 0)))}
          </p>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-white rounded-lg border overflow-hidden">
        <div className="px-6 py-4 border-b bg-gray-50">
          <h3 className="text-lg font-medium text-gray-900">Portfolio Holdings</h3>
          <p className="text-sm text-gray-600 mt-1">
            {holdings.length} holdings with total value of {formatCurrency(totalPortfolioValue)}
          </p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Company
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Shares
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Market Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  % of Portfolio
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Day Change
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {holdings.map((holding, index) => {
                // Safe data extraction for each holding
                const symbol = holding.symbol || holding.ticker || 'N/A';
                const company = holding.company_name || holding.name || holding.company || symbol;
                const shares = holding.shares || holding.quantity || holding.position_size || 0;
                const price = holding.price || holding.current_price || holding.market_price || 0;
                const marketValue = holding.market_value || holding.total_value || holding.value || 0;
                const dayChange = holding.day_change || holding.daily_change || 0;
                const dayChangePercent = holding.day_change_percent || holding.daily_change_percent || 0;
                
                // Calculate percentage of portfolio
                const portfolioPercent = totalPortfolioValue > 0 ? (marketValue / totalPortfolioValue) * 100 : 0;

                return (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm font-medium text-gray-900">
                          {symbol}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{company}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatNumber(shares)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatCurrency(price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatCurrency(marketValue)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatPercentage(portfolioPercent)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-1">
                        {dayChange !== 0 && (
                          <>
                            {dayChange > 0 ? (
                              <TrendingUp className="w-4 h-4 text-green-500" />
                            ) : (
                              <TrendingDown className="w-4 h-4 text-red-500" />
                            )}
                          </>
                        )}
                        <span className={`text-sm ${
                          dayChange > 0 ? 'text-green-600' :
                          dayChange < 0 ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {dayChange !== 0 ? (
                            <>
                              {dayChange > 0 ? '+' : ''}{formatCurrency(dayChange)}
                              {dayChangePercent !== 0 && (
                                <span className="text-xs ml-1">
                                  ({dayChangePercent > 0 ? '+' : ''}{formatPercentage(dayChangePercent)})
                                </span>
                              )}
                            </>
                          ) : (
                            'N/A'
                          )}
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Additional Holdings Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-medium text-gray-900 mb-4">Top Holdings</h4>
          <div className="space-y-3">
            {holdings
              .sort((a, b) => (b.market_value || b.total_value || b.value || 0) - (a.market_value || a.total_value || a.value || 0))
              .slice(0, 5)
              .map((holding, index) => {
                const symbol = holding.symbol || holding.ticker || 'N/A';
                const value = holding.market_value || holding.total_value || holding.value || 0;
                const percent = totalPortfolioValue > 0 ? (value / totalPortfolioValue) * 100 : 0;
                
                return (
                  <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <div>
                      <span className="font-medium text-gray-900">{symbol}</span>
                      <span className="text-sm text-gray-600 ml-2">{formatPercentage(percent)}</span>
                    </div>
                    <span className="font-medium text-gray-900">{formatCurrency(value)}</span>
                  </div>
                );
              })}
          </div>
        </div>

        <div className="bg-white rounded-lg border p-6">
          <h4 className="text-lg font-medium text-gray-900 mb-4">Portfolio Statistics</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Holdings:</span>
              <span className="font-medium">{holdings.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Value:</span>
              <span className="font-medium">{formatCurrency(totalPortfolioValue)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Average Position:</span>
              <span className="font-medium">{formatCurrency(totalPortfolioValue / holdings.length)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Largest Position:</span>
              <span className="font-medium">
                {formatCurrency(Math.max(...holdings.map(h => h.market_value || h.total_value || h.value || 0)))}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Smallest Position:</span>
              <span className="font-medium">
                {formatCurrency(Math.min(...holdings.map(h => h.market_value || h.total_value || h.value || 0)))}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}