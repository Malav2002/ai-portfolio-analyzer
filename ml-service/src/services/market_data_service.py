import asyncio
import logging
from typing import Dict, List, Optional
import aiohttp
import yfinance as yf
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import time
import random

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Enhanced market data service with rate limiting protection
    """
    
    def __init__(self):
        # API Keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        # Rate limiting protection
        self.last_request_time = 0
        self.min_delay_between_requests = 1.0  # 1 second delay minimum
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Session for HTTP requests
        self.session = None
        
        # Data sources priority
        self.data_sources = ['yfinance', 'alpha_vantage', 'fmp', 'polygon']
        
    async def initialize(self):
        """Initialize the market data service"""
        logger.info("📊 Initializing Market Data Service with rate limiting...")
        
        self.session = aiohttp.ClientSession()
        
        logger.info("✅ Market Data Service initialized")
    
    async def close(self):
        """Close the market data service"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_delay(self):
        """Add delay to prevent rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay_between_requests:
            delay = self.min_delay_between_requests - time_since_last
            # Add random jitter to avoid thundering herd
            delay += random.uniform(0.1, 0.5)
            logger.info(f"⏱️ Rate limiting delay: {delay:.2f}s")
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
    
    async def get_stock_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get stock quote with rate limiting and retry logic
        """
        try:
            logger.info(f"📊 Getting quote for {symbol}")
            
            # Add rate limiting delay
            await self._rate_limit_delay()
            
            # Try yfinance with retry logic
            for attempt in range(self.max_retries):
                try:
                    quote = await self._get_yfinance_quote_with_retry(symbol)
                    if quote:
                        return quote
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
            
            # If yfinance fails, return basic quote from portfolio data
            logger.info(f"🔄 Creating basic quote for {symbol}")
            return self._create_basic_quote(symbol)
            
        except Exception as e:
            logger.error(f"❌ Failed to get quote for {symbol}: {e}")
            return self._create_basic_quote(symbol)
    
    async def _get_yfinance_quote_with_retry(self, symbol: str) -> Optional[Dict]:
        """Get yfinance quote with better error handling"""
        try:
            # Use yfinance with timeout
            ticker = yf.Ticker(symbol)
            
            # Get basic info first
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                logger.warning(f"⚠️ No market data for {symbol}")
                return None
            
            # Extract key data
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', current_price)
            
            quote = {
                'symbol': symbol,
                'price': float(current_price),
                'previous_close': float(previous_close),
                'change': float(current_price - previous_close),
                'change_percent': ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
                'volume': int(info.get('volume', 0)),
                'market_cap': int(info.get('marketCap', 0)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'source': 'yfinance',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Got quote for {symbol}: ${current_price}")
            return quote
            
        except Exception as e:
            if '429' in str(e) or 'Too Many Requests' in str(e):
                logger.warning(f"⚠️ Rate limited for {symbol}, will retry...")
                raise  # Re-raise to trigger retry
            else:
                logger.error(f"❌ YFinance error for {symbol}: {e}")
                return None
    
    def _create_basic_quote(self, symbol: str) -> Dict:
        """Create a basic quote when market data is unavailable"""
        return {
            'symbol': symbol,
            'price': 0,
            'previous_close': 0,
            'change': 0,
            'change_percent': 0,
            'volume': 0,
            'market_cap': 0,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'source': 'basic_fallback',
            'timestamp': datetime.utcnow().isoformat(),
            'note': 'Market data unavailable due to rate limiting'
        }
    
    async def enrich_portfolio_data(self, portfolio_data: Dict) -> Dict:
        """
        Enrich portfolio data with market information (rate-limited)
        """
        try:
            logger.info("📈 Starting portfolio enrichment with rate limiting...")
            
            holdings = portfolio_data.get('holdings', [])
            if not holdings:
                logger.warning("⚠️ No holdings to enrich")
                return portfolio_data
            
            logger.info(f"📊 Enriching {len(holdings)} holdings with delays...")
            
            enriched_holdings = []
            
            for i, holding in enumerate(holdings):
                symbol = holding.get('symbol', '')
                if not symbol:
                    enriched_holdings.append(holding)
                    continue
                
                logger.info(f"📊 Processing {symbol} ({i+1}/{len(holdings)})")
                
                # Get market data with rate limiting
                quote = await self.get_stock_quote(symbol)
                
                if quote and quote.get('price', 0) > 0:
                    # Enrich with live market data
                    enriched_holding = {
                        **holding,
                        'live_price': quote['price'],
                        'live_change': quote['change'],
                        'live_change_percent': quote['change_percent'],
                        'sector': quote.get('sector', 'Unknown'),
                        'industry': quote.get('industry', 'Unknown'),
                        'market_cap': quote.get('market_cap', 0),
                        'live_market_value': holding.get('shares', 0) * quote['price'],
                        'data_source': quote['source'],
                        'last_updated': quote['timestamp']
                    }
                    logger.info(f"✅ Enriched {symbol} with live data")
                else:
                    # Use portfolio data as fallback
                    enriched_holding = {
                        **holding,
                        'live_price': holding.get('price', 0),
                        'live_change': 0,
                        'live_change_percent': 0,
                        'sector': 'Unknown',
                        'industry': 'Unknown',
                        'market_cap': 0,
                        'live_market_value': holding.get('market_value', 0),
                        'data_source': 'portfolio_data',
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    logger.info(f"⚠️ Using portfolio data for {symbol}")
                
                enriched_holdings.append(enriched_holding)
            
            # Update portfolio data
            enriched_portfolio = {
                **portfolio_data,
                'holdings': enriched_holdings,
                'total_live_value': sum(h.get('live_market_value', 0) for h in enriched_holdings),
                'enrichment_status': 'completed_with_rate_limiting',
                'enriched_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Portfolio enrichment completed: {len(enriched_holdings)} holdings")
            return enriched_portfolio
            
        except Exception as e:
            logger.error(f"❌ Portfolio enrichment failed: {e}")
            # Return original data if enrichment fails
            return {
                **portfolio_data,
                'enrichment_status': 'failed',
                'enrichment_error': str(e)
            }