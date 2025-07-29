#!/bin/bash
# fix-dynamic-portfolio-data.sh
# Fix portfolio overview to show real dynamic data from analysis

echo "🔧 Fixing Dynamic Portfolio Data Display"
echo "======================================="

cd frontend

echo ""
echo "1️⃣ Updating AnalysisResults to extract real portfolio data..."

cat > src/app/components/AnalysisResults.tsx << 'EOF'
'use client'

import { TrendingUp, TrendingDown, Shield, AlertTriangle, DollarSign, Target, Star, CheckCircle, Brain, Upload } from 'lucide-react'

interface AnalysisResultsProps {
  data: any
}

export default function AnalysisResults({ data }: AnalysisResultsProps) {
  console.log('🔍 Full analysis data:', JSON.stringify(data, null, 2));

  if (!data?.success) {
    return (
      <div className="glass-effect rounded-3xl p-8 border border-red-200">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-gradient-to-br from-red-500 to-rose-600 rounded-2xl">
            <AlertTriangle className="w-8 h-8 text-white" />
          </div>
          <div>
            <h4 className="font-bold text-red-800 text-lg">Analysis Failed</h4>
            <p className="text-red-700">{data?.error || data?.message || 'Unknown error occurred'}</p>
          </div>
        </div>
      </div>
    )
  }

  // Extract portfolio data from different possible structures
  const analysis = data.analysis || data.result || data
  const portfolio = analysis.portfolio || analysis
  const market_data = analysis.market_data || {}
  const ocr_result = analysis.ocr_result || {}
  
  // Get positions/holdings from multiple possible locations
  const positions = portfolio.positions || 
                   portfolio.holdings || 
                   analysis.positions || 
                   analysis.holdings || 
                   market_data.positions || 
                   market_data.holdings ||
                   ocr_result.positions ||
                   ocr_result.holdings ||
                   []

  const recommendations = analysis.recommendations || 
                         analysis.ai_recommendations || 
                         []
  
  // Calculate real portfolio metrics from actual data
  let totalValue = 0
  let totalChange = 0
  let totalGainLoss = 0
  
  if (positions && positions.length > 0) {
    // Calculate from actual positions
    totalValue = positions.reduce((sum: number, pos: any) => {
      const value = pos.market_value || pos.value || pos.total_value || 
                   (pos.shares || pos.quantity || 0) * (pos.current_price || pos.price || 0)
      return sum + (value || 0)
    }, 0)
    
    const totalCost = positions.reduce((sum: number, pos: any) => {
      const cost = pos.cost_basis || pos.purchase_value || 
                  (pos.shares || pos.quantity || 0) * (pos.purchase_price || pos.avg_cost || pos.current_price || 0)
      return sum + (cost || 0)
    }, 0)
    
    if (totalCost > 0) {
      totalChange = ((totalValue - totalCost) / totalCost) * 100
    }
    
    totalGainLoss = totalValue - totalCost
  } else {
    // Fallback to provided totals
    totalValue = portfolio.total_value || 
                portfolio.portfolio_value || 
                market_data.total_value ||
                analysis.total_value ||
                0
                
    totalChange = portfolio.total_change || 
                 portfolio.performance?.total_change ||
                 market_data.total_change ||
                 analysis.total_change ||
                 0
                 
    totalGainLoss = portfolio.total_gain_loss ||
                   analysis.total_gain_loss ||
                   0
  }

  console.log('📊 Calculated metrics:', {
    positions: positions.length,
    totalValue,
    totalChange,
    totalGainLoss
  });
  
  return (
    <div className="space-y-10 animate-fade-in-up">
      {/* Success header */}
      <div className="text-center space-y-6">
        <div className="inline-flex items-center space-x-3 px-8 py-4 bg-gradient-to-r from-green-100 via-emerald-100 to-green-100 rounded-full border border-green-200 shadow-xl">
          <CheckCircle className="w-6 h-6 text-green-600" />
          <span className="font-black text-green-700 text-lg">✨ Analysis Complete ✨</span>
          <Star className="w-5 h-5 text-green-500 animate-spin" />
        </div>
        <h3 className="text-4xl md:text-5xl font-black text-gradient">Portfolio Analysis Results</h3>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">Your AI-powered investment insights are ready</p>
      </div>
      
      {/* Portfolio Overview - DYNAMIC DATA */}
      <div className="card-gradient rounded-[2rem] p-10 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-400/20 to-indigo-600/20 rounded-full blur-2xl"></div>
        
        <div className="relative z-10">
          <div className="flex items-center justify-between mb-8">
            <h4 className="text-3xl font-black text-gray-800 flex items-center">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl mr-4 shadow-2xl">
                <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              Portfolio Overview
            </h4>
            <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full shadow-lg">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-bold text-blue-700">Live Data</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Total Value - DYNAMIC */}
            <div className="metric-card text-center group hover:shadow-glow-green relative overflow-hidden">
              <div className="p-5 bg-gradient-to-br from-green-500 to-emerald-600 rounded-3xl w-fit mx-auto mb-6 shadow-2xl group-hover:scale-125 transition-transform duration-500">
                <DollarSign className="h-12 w-12 text-white" />
              </div>
              <p className="text-5xl font-black text-green-600 mb-3">
                ${totalValue > 0 ? totalValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : 'Calculating...'}
              </p>
              <p className="text-gray-600 font-bold text-lg">Total Portfolio Value</p>
              {totalGainLoss !== 0 && (
                <p className={`text-sm font-medium mt-2 ${totalGainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {totalGainLoss >= 0 ? '+' : ''}${Math.abs(totalGainLoss).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} total
                </p>
              )}
              <div className="mt-4 h-2 bg-gradient-to-r from-green-400 via-emerald-500 to-green-600 rounded-full shadow-lg"></div>
            </div>
            
            {/* Holdings Count - DYNAMIC */}
            <div className="metric-card text-center group hover:shadow-glow relative overflow-hidden">
              <div className="p-5 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl w-fit mx-auto mb-6 shadow-2xl group-hover:scale-125 transition-transform duration-500">
                <Target className="h-12 w-12 text-white" />
              </div>
              <p className="text-5xl font-black text-blue-600 mb-3">
                {positions.length > 0 ? positions.length : 'Multiple'}
              </p>
              <p className="text-gray-600 font-bold text-lg">Active Holdings</p>
              {positions.length > 0 && (
                <p className="text-sm text-gray-500 mt-2">
                  {positions.map((p: any) => p.symbol || p.ticker).filter(Boolean).slice(0, 3).join(', ')}
                  {positions.length > 3 && ` +${positions.length - 3} more`}
                </p>
              )}
              <div className="mt-4 h-2 bg-gradient-to-r from-blue-400 via-indigo-500 to-blue-600 rounded-full shadow-lg"></div>
            </div>
            
            {/* Performance - DYNAMIC */}
            <div className="metric-card text-center group hover:shadow-glow relative overflow-hidden">
              <div className={`p-5 rounded-3xl w-fit mx-auto mb-6 shadow-2xl group-hover:scale-125 transition-transform duration-500 ${
                totalChange >= 0 
                  ? 'bg-gradient-to-br from-green-500 to-emerald-600' 
                  : 'bg-gradient-to-br from-red-500 to-rose-600'
              }`}>
                {totalChange >= 0 ? (
                  <TrendingUp className="h-12 w-12 text-white" />
                ) : (
                  <TrendingDown className="h-12 w-12 text-white" />
                )}
              </div>
              <p className={`text-5xl font-black mb-3 ${
                totalChange >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {totalChange >= 0 ? '+' : ''}{totalChange.toFixed(2)}%
              </p>
              <p className="text-gray-600 font-bold text-lg">Performance</p>
              {totalGainLoss !== 0 && (
                <p className={`text-sm font-medium mt-2 ${totalGainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {totalGainLoss >= 0 ? '+' : ''}${Math.abs(totalGainLoss).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                </p>
              )}
              <div className={`mt-4 h-2 rounded-full shadow-lg ${
                totalChange >= 0 
                  ? 'bg-gradient-to-r from-green-400 via-emerald-500 to-green-600' 
                  : 'bg-gradient-to-r from-red-400 via-rose-500 to-red-600'
              }`}></div>
            </div>
          </div>
        </div>
      </div>

      {/* Holdings Table - DYNAMIC */}
      {positions.length > 0 && (
        <div className="holdings-table animate-slide-up shadow-2xl">
          <div className="p-8 border-b border-blue-100/50 bg-gradient-to-r from-blue-50/80 to-indigo-50/60">
            <div className="flex items-center justify-between">
              <h4 className="text-2xl font-black text-gray-800 flex items-center">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl mr-4 shadow-xl">
                  <Target className="h-6 w-6 text-white" />
                </div>
                Portfolio Holdings
              </h4>
              <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full shadow-lg">
                <Star className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-bold text-purple-700">{positions.length} positions</span>
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gradient-to-r from-blue-50 to-indigo-50">
                <tr>
                  <th className="px-8 py-6 text-left text-sm font-black text-gray-700 uppercase tracking-wider">Symbol</th>
                  <th className="px-8 py-6 text-left text-sm font-black text-gray-700 uppercase tracking-wider">Shares</th>
                  <th className="px-8 py-6 text-left text-sm font-black text-gray-700 uppercase tracking-wider">Current Price</th>
                  <th className="px-8 py-6 text-left text-sm font-black text-gray-700 uppercase tracking-wider">Market Value</th>
                  <th className="px-8 py-6 text-left text-sm font-black text-gray-700 uppercase tracking-wider">Change</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-blue-100/50">
                {positions.map((position: any, index: number) => {
                  const symbol = position.symbol || position.ticker || `STOCK${index + 1}`
                  const shares = position.shares || position.quantity || position.units || 0
                  const currentPrice = position.current_price || position.price || position.last_price || 0
                  const marketValue = position.market_value || position.value || position.total_value || (shares * currentPrice)
                  const changePercent = position.change_percent || position.change || position.daily_change || position.percent_change || 0
                  
                  console.log(`📈 Position ${index}:`, { symbol, shares, currentPrice, marketValue, changePercent });
                  
                  return (
                    <tr key={index} className="hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-indigo-50/30 transition-all duration-300 group">
                      <td className="px-8 py-6">
                        <div className="flex items-center space-x-4">
                          <div className="w-12 h-12 bg-gradient-to-br from-gray-600 to-gray-800 rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                            <span className="text-white font-black text-sm">{symbol.charAt(0)}</span>
                          </div>
                          <div>
                            <span className="font-black text-gray-900 text-lg block">{symbol}</span>
                            <span className="text-xs text-gray-500">
                              {position.company_name || position.name || 'Stock Position'}
                            </span>
                          </div>
                        </div>
                      </td>
                      <td className="px-8 py-6 font-bold text-gray-700 text-lg">
                        {shares.toLocaleString()}
                      </td>
                      <td className="px-8 py-6 font-bold text-gray-900 text-lg">
                        ${currentPrice.toFixed(2)}
                      </td>
                      <td className="px-8 py-6 font-black text-gray-900 text-lg">
                        ${marketValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                      </td>
                      <td className="px-8 py-6">
                        <div className={`inline-flex items-center space-x-3 px-4 py-2 rounded-2xl font-bold text-lg shadow-lg transition-all duration-300 hover:scale-110 ${
                          changePercent >= 0 
                            ? 'bg-gradient-to-r from-green-100 to-emerald-100 text-green-700' 
                            : 'bg-gradient-to-r from-red-100 to-rose-100 text-red-700'
                        }`}>
                          {changePercent >= 0 ? (
                            <TrendingUp className="h-5 w-5" />
                          ) : (
                            <TrendingDown className="h-5 w-5" />
                          )}
                          <span>{changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%</span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* No Holdings Message */}
      {positions.length === 0 && (
        <div className="glass-effect rounded-3xl p-8 border border-yellow-200">
          <div className="text-center space-y-4">
            <div className="p-4 bg-gradient-to-br from-yellow-500 to-orange-600 rounded-3xl w-fit mx-auto shadow-xl">
              <AlertTriangle className="h-10 w-10 text-white" />
            </div>
            <h4 className="text-2xl font-bold text-yellow-800">No Holdings Detected</h4>
            <p className="text-yellow-700 max-w-md mx-auto">
              The AI couldn't extract specific stock positions from your portfolio image. 
              Try uploading a clearer screenshot with visible stock symbols and quantities.
            </p>
          </div>
        </div>
      )}

      {/* AI Recommendations - DYNAMIC */}
      {recommendations.length > 0 && (
        <div className="card-gradient rounded-[2rem] p-10 shadow-2xl animate-slide-up relative overflow-hidden">
          <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-purple-400/20 to-pink-600/20 rounded-full blur-3xl"></div>
          
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-8">
              <h4 className="text-3xl font-black text-gray-800 flex items-center">
                <div className="p-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-3xl mr-4 shadow-2xl">
                  <Brain className="h-8 w-8 text-white" />
                </div>
                AI Recommendations
              </h4>
              <div className="flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-purple-100 to-pink-100 rounded-full shadow-xl">
                <Star className="w-5 h-5 text-purple-600" />
                <span className="text-sm font-black text-purple-700">{recommendations.length} Expert Insights</span>
              </div>
            </div>
            
            <div className="grid gap-8">
              {recommendations.map((rec: any, index: number) => {
                if (typeof rec === 'string') {
                  return (
                    <div key={index} className="recommendation-card group relative overflow-hidden">
                      <div className="relative z-10 flex items-start space-x-6">
                        <div className="p-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl shadow-xl group-hover:scale-110 transition-transform duration-500">
                          <Target className="w-6 h-6 text-white" />
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-800 font-bold text-lg leading-relaxed">{rec}</p>
                        </div>
                      </div>
                    </div>
                  );
                } else if (typeof rec === 'object' && rec !== null) {
                  const iconMap = {
                    'diversification': { icon: Target, color: 'from-purple-500 to-pink-600' },
                    'performance': { icon: TrendingUp, color: 'from-green-500 to-emerald-600' },
                    'sector_diversification': { icon: Target, color: 'from-blue-500 to-indigo-600' },
                    'risk_reduction': { icon: Shield, color: 'from-red-500 to-rose-600' }
                  };
                  
                  const iconConfig = iconMap[rec.type as keyof typeof iconMap] || { 
                    icon: Brain, 
                    color: 'from-gray-500 to-gray-600'
                  };
                  const IconComponent = iconConfig.icon;
                  
                  return (
                    <div key={index} className="recommendation-card group relative overflow-hidden">
                      <div className="relative z-10 space-y-6">
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-6">
                            <div className={`p-4 bg-gradient-to-br ${iconConfig.color} rounded-3xl shadow-2xl group-hover:scale-110 transition-transform duration-500`}>
                              <IconComponent className="w-8 h-8 text-white" />
                            </div>
                            <div className="flex-1">
                              {rec.title && (
                                <h5 className="text-2xl font-black text-gray-900 mb-3">{rec.title}</h5>
                              )}
                              {rec.description && (
                                <p className="text-gray-700 leading-relaxed text-lg font-medium">{rec.description}</p>
                              )}
                            </div>
                          </div>
                          {rec.confidence && (
                            <div className="text-center">
                              <div className="px-4 py-2 bg-white/90 rounded-2xl border border-blue-200 shadow-lg">
                                <p className="text-2xl font-black text-blue-700">
                                  {Math.round(rec.confidence * 100)}%
                                </p>
                                <p className="text-xs text-gray-500 font-bold">CONFIDENCE</p>
                              </div>
                            </div>
                          )}
                        </div>
                        
                        <div className="flex flex-wrap gap-3">
                          {rec.type && (
                            <span className={`inline-flex items-center px-6 py-3 rounded-2xl text-sm font-black shadow-xl ${
                              rec.type.includes('diversification') ? 'bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800' :
                              rec.type.includes('performance') ? 'bg-gradient-to-r from-green-100 to-emerald-100 text-green-800' :
                              rec.type.includes('risk') ? 'bg-gradient-to-r from-red-100 to-rose-100 text-red-800' :
                              'bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-800'
                            }`}>
                              {rec.type.replace('_', ' ').toUpperCase()}
                            </span>
                          )}
                          
                          {rec.priority && (
                            <span className={`inline-flex items-center px-4 py-2 rounded-full text-xs font-black shadow-lg ${
                              rec.priority === 'high' ? 'bg-gradient-to-r from-red-100 to-rose-100 text-red-700 ring-2 ring-red-300' :
                              rec.priority === 'medium' ? 'bg-gradient-to-r from-yellow-100 to-orange-100 text-yellow-700 ring-2 ring-yellow-300' :
                              'bg-gradient-to-r from-green-100 to-emerald-100 text-green-700 ring-2 ring-green-300'
                            }`}>
                              {rec.priority.toUpperCase()} PRIORITY
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          </div>
        </div>
      )}

      {/* Call to Action */}
      <div className="glass-effect rounded-[2rem] p-10 text-center shadow-2xl">
        <div className="space-y-6">
          <div className="inline-flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full shadow-lg">
            <Star className="w-5 h-5 text-blue-600 animate-pulse" />
            <span className="text-sm font-bold text-blue-700">Ready for More Analysis?</span>
          </div>
          <h4 className="text-3xl font-black text-gray-800">
            Analyze Another Portfolio
          </h4>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto font-medium">
            Upload different screenshots to compare timeframes or track progress
          </p>
          <button 
            onClick={() => window.location.reload()} 
            className="glow-button inline-flex items-center space-x-3 hover:shadow-2xl"
          >
            <Upload className="w-6 h-6" />
            <span className="text-lg">Start New Analysis</span>
            <Star className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Debug section for development */}
      {process.env.NODE_ENV === 'development' && (
        <div className="glass-effect rounded-2xl p-6 border border-gray-200">
          <details className="text-sm">
            <summary className="cursor-pointer font-bold text-gray-700 mb-4">
              🔍 Debug: Raw Analysis Data Structure
            </summary>
            <div className="space-y-4">
              <div>
                <strong>Positions Found:</strong> {positions.length}
                <pre className="mt-2 overflow-auto max-h-40 bg-white p-3 rounded border text-xs">
                  {JSON.stringify(positions, null, 2)}
                </pre>
              </div>
              <div>
                <strong>Calculated Metrics:</strong>
                <pre className="mt-2 bg-white p-3 rounded border text-xs">
{`Total Value: $${totalValue}
Total Change: ${totalChange}%
Holdings Count: ${positions.length}
Recommendations: ${recommendations.length}`}
                </pre>
              </div>
            </div>
          </details>
        </div>
      )}
    </div>
  )
}
EOF

echo ""
echo "2️⃣ Rebuilding with dynamic data extraction..."
cd ..
docker-compose build --no-cache frontend

echo ""
echo "3️⃣ Restarting frontend..."
docker-compose restart frontend

echo ""
echo "✅ Portfolio Overview now shows REAL dynamic data!"
echo ""
echo "🔧 What was fixed:"
echo "   ✅ Portfolio value calculated from actual positions"
echo "   ✅ Holdings count shows real number of detected stocks"
echo "   ✅ Performance calculated from real market data"
echo "   ✅ Holdings table populated with actual positions"
echo "   ✅ Debug section shows raw data structure"
echo ""
echo "📊 The overview will now show:"
echo "   • Real portfolio value from your analysis"
echo "   • Actual number of holdings detected"
echo "   • Calculated performance based on real data"
echo "   • Dynamic color coding based on gains/losses"
echo ""
echo "🧪 Test it: Upload a new portfolio screenshot and see real data!"
echo "🌐 Visit: http://localhost:3000"