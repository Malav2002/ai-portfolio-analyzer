import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // Increased timeout to 60 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Progress callback type
export type ProgressCallback = (message: string, step: number, totalSteps: number) => void;

// Request interceptor with detailed logging
api.interceptors.request.use(
  (config) => {
    const timestamp = new Date().toISOString();
    console.log(`🌐 [${timestamp}] API Request: ${config.method?.toUpperCase()} ${config.url}`);
    console.log(`📊 Request details:`, {
      url: config.url,
      method: config.method,
      timeout: config.timeout,
      headers: config.headers
    });
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor with detailed logging
api.interceptors.response.use(
  (response) => {
    const timestamp = new Date().toISOString();
    const duration = Date.now() - (response.config as any)._startTime;
    console.log(`✅ [${timestamp}] API Response: ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status} (${duration}ms)`);
    console.log(`📈 Response size:`, new Blob([JSON.stringify(response.data)]).size, 'bytes');
    return response;
  },
  (error) => {
    const timestamp = new Date().toISOString();
    console.error(`❌ [${timestamp}] API Response Error:`, {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message
    });
    return Promise.reject(error);
  }
);

// Add request timing
api.interceptors.request.use((config) => {
  (config as any)._startTime = Date.now();
  return config;
});

export interface StockQuote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume?: number;
  source: string;
  timestamp: string;
}

export interface Holding {
  symbol: string;
  shares: number;
  market_value: number;
  live_market_value?: number;
  gain_loss: number;
  live_gain_loss?: number;
  return_pct: number;
  live_gain_loss_percent?: number;
  sector: string;
  live_price?: number;
  data_source?: string;
}

export interface Recommendation {
  type: string;
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  actions: string[];
  confidence: number;
  source?: string;
}

export interface RiskAnalysis {
  ai_risk_prediction: {
    predicted_risk_level: string;
    risk_distribution: Record<string, number>;
    confidence: number;
  };
  overall_risk_score: number;
  risk_factors: string[];
  traditional_metrics: {
    portfolio_volatility: number;
    max_drawdown: number;
    sharpe_ratio: number;
    beta: number;
  };
}

export interface PortfolioAnalysis {
  portfolio_data: {
    holdings: Holding[];
    total_live_value?: number;
    total_live_gain_loss?: number;
    enriched_at: string;
    data_quality?: {
      quality_score: number;
      quality_rating: string;
    };
  };
  ai_insights: {
    portfolio_metrics: {
      total_value: number;
      total_gain_loss: number;
      total_return_percent: number;
      num_holdings: number;
      win_rate: number;
    };
    diversification: {
      score: number;
      sector_weights: Record<string, number>;
      concentration_risk: string;
    };
    quality_score: number;
  };
  recommendations: Recommendation[];
  risk_analysis: RiskAnalysis;
  timestamp: string;
}

// Enhanced API Functions with progress tracking
export const portfolioAPI = {
  // Upload and analyze portfolio image with progress tracking
  analyzePortfolio: async (file: File, onProgress?: ProgressCallback): Promise<PortfolioAnalysis> => {
    const startTime = Date.now();
    console.log(`🚀 [${new Date().toISOString()}] Starting portfolio analysis for file: ${file.name}`);
    console.log(`📁 File details:`, {
      name: file.name,
      size: `${file.size ? (file.size / 1024 / 1024).toFixed(2) : '0.00'} MB`,
      type: file.type
    });

    onProgress?.('Preparing image upload...', 1, 8);

    const formData = new FormData();
    formData.append('image', file);
    
    try {
      onProgress?.('Uploading image to server...', 2, 8);
      
      const response = await api.post('/api/ai/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            console.log(`📤 Upload progress: ${percentCompleted}%`);
            onProgress?.(`Uploading image... ${percentCompleted}%`, 2, 8);
          }
        }
      });

      const duration = Date.now() - startTime;
      console.log(`✅ [${new Date().toISOString()}] Portfolio analysis completed in ${duration}ms`);
      console.log(`📊 Analysis result:`, {
        success: response.data.success,
        holdingsCount: response.data.analysis?.portfolio_data?.holdings?.length || 0,
        recommendationsCount: response.data.analysis?.recommendations?.length || 0,
        riskLevel: response.data.analysis?.risk_analysis?.ai_risk_prediction?.predicted_risk_level
      });

      onProgress?.('Analysis complete!', 8, 8);
      return response.data.analysis;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Portfolio analysis failed after ${duration}ms:`, {
        error: error.message,
        status: error.response?.status,
        data: error.response?.data
      });
      throw error;
    }
  },

  // Get AI recommendations with progress tracking
  getRecommendations: async (holdings: Holding[], onProgress?: ProgressCallback): Promise<{ recommendations: Recommendation[] }> => {
    const startTime = Date.now();
    console.log(`💡 [${new Date().toISOString()}] Generating recommendations for ${holdings.length} holdings`);
    
    onProgress?.('Analyzing portfolio composition...', 1, 4);
    
    try {
      onProgress?.('Fetching market data...', 2, 4);
      onProgress?.('Generating AI recommendations...', 3, 4);
      
      const response = await api.post('/api/ai/recommendations', { holdings });
      
      const duration = Date.now() - startTime;
      console.log(`✅ [${new Date().toISOString()}] Recommendations generated in ${duration}ms`);
      console.log(`📝 Generated ${response.data.data.recommendations.length} recommendations`);
      
      onProgress?.('Recommendations ready!', 4, 4);
      return response.data.data;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Recommendations failed after ${duration}ms:`, error);
      throw error;
    }
  },

  // Get risk analysis with progress tracking
  getRiskAnalysis: async (holdings: Holding[], onProgress?: ProgressCallback): Promise<RiskAnalysis> => {
    const startTime = Date.now();
    console.log(`⚠️ [${new Date().toISOString()}] Performing risk analysis for ${holdings.length} holdings`);
    
    onProgress?.('Calculating risk metrics...', 1, 3);
    
    try {
      onProgress?.('Running AI risk models...', 2, 3);
      
      const response = await api.post('/api/ai/risk-analysis', { holdings });
      
      const duration = Date.now() - startTime;
      console.log(`✅ [${new Date().toISOString()}] Risk analysis completed in ${duration}ms`);
      console.log(`📊 Risk level: ${response.data.risk_analysis.ai_risk_prediction.predicted_risk_level}`);
      
      onProgress?.('Risk analysis complete!', 3, 3);
      return response.data.risk_analysis;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Risk analysis failed after ${duration}ms:`, error);
      throw error;
    }
  },

  // Get stock quote with timing
  getStockQuote: async (symbol: string): Promise<StockQuote> => {
    const startTime = Date.now();
    console.log(`📊 [${new Date().toISOString()}] Fetching quote for ${symbol}`);
    
    try {
      const response = await api.get(`/api/ai/quote/${symbol}`);
      const duration = Date.now() - startTime;
      console.log(`✅ [${new Date().toISOString()}] Quote for ${symbol} fetched in ${duration}ms from ${response.data.quote.source}`);
      return response.data.quote;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Quote failed for ${symbol} after ${duration}ms:`, error);
      throw error;
    }
  },

  // Get batch stock quotes with detailed logging
  getBatchQuotes: async (symbols: string[]): Promise<Record<string, StockQuote>> => {
    const startTime = Date.now();
    console.log(`📊 [${new Date().toISOString()}] Fetching batch quotes for ${symbols.length} symbols:`, symbols);
    
    try {
      const response = await api.get(`/api/ai/quotes?symbols=${symbols.join(',')}`);
      const duration = Date.now() - startTime;
      const successCount = Object.keys(response.data.quotes).length;
      console.log(`✅ [${new Date().toISOString()}] Batch quotes completed in ${duration}ms: ${successCount}/${symbols.length} successful`);
      
      // Log individual quote sources
      Object.entries(response.data.quotes).forEach(([symbol, quote]: [string, any]) => {
        if (quote) {
          console.log(`   ${symbol}: $${quote.price} (${quote.source})`);
        } else {
          console.log(`   ${symbol}: Failed to fetch`);
        }
      });
      
      return response.data.quotes;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Batch quotes failed after ${duration}ms:`, error);
      throw error;
    }
  },

  // Health check with detailed info
  healthCheck: async () => {
    const startTime = Date.now();
    console.log(`🏥 [${new Date().toISOString()}] Performing health check...`);
    
    try {
      const response = await api.get('/api/ai/health');
      const duration = Date.now() - startTime;
      console.log(`✅ [${new Date().toISOString()}] Health check completed in ${duration}ms:`, {
        status: response.data.success,
        mlService: response.data.ml_service_available,
        dataSource: response.data.data_source
      });
      return response.data;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      console.error(`❌ [${new Date().toISOString()}] Health check failed after ${duration}ms:`, error);
      throw error;
    }
  },
};

export default api;
