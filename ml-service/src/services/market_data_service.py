import httpx
import asyncio
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import time
import os

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_request_time = {}
        self.min_request_interval = 2  # 2 seconds between requests for free APIs
        
        # API Keys from environment
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fmp_key = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        logger.info(f"Market Data Service initialized with keys: AV={bool(self.alpha_vantage_key)}, FMP={bool(self.fmp_key)}, Polygon={bool(self.polygon_key)}")

    async def get_stock_quote_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time stock quote from Alpha Vantage with actual API key
        """
        if not self.alpha_vantage_key:
            return None
            
        try:
            # Rate limiting for free tier
            await self.rate_limit_check('alpha_vantage')
            
            async with httpx.AsyncClient() as client:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_key
                }
                
                response = await client.get(url, params=params, timeout=15.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for API limit error
                    if 'Error Message' in data:
                        logger.error(f"Alpha Vantage error: {data['Error Message']}")
                        return None
                    
                    if 'Note' in data:
                        logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        return None
                    
                    if 'Global Quote' in data and data['Global Quote']:
                        quote = data['Global Quote']
                        price = quote.get('05. price')
                        
                        if price:
                            quote_data = {
                                'symbol': symbol,
                                'current_price': float(price),
                                'previous_close': float(quote.get('08. previous close', 0)),
                                'change': float(quote.get('09. change', 0)),
                                'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                                'volume': int(quote.get('06. volume', 0)),
                                'high': float(quote.get('03. high', 0)),
                                'low': float(quote.get('04. low', 0)),
                                'open': float(quote.get('02. open', 0)),
                                'source': 'alpha_vantage_real',
                                'timestamp': datetime.utcnow().isoformat(),
                                'last_trading_day': quote.get('07. latest trading day')
                            }
                            
                            logger.info(f"Alpha Vantage (real) got quote for {symbol}: ${quote_data['current_price']}")
                            return quote_data
                        
        except Exception as e:
            logger.error(f"Alpha Vantage real API error for {symbol}: {e}")
            return None

    async def get_stock_quote_fmp(self, symbol: str) -> Optional[Dict]:
        """
        Get stock quote from Financial Modeling Prep
        """
        if not self.fmp_key:
            return None
            
        try:
            await self.rate_limit_check('fmp')
            
            async with httpx.AsyncClient() as client:
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
                params = {
                    "apikey": self.fmp_key
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and len(data) > 0:
                        quote = data[0]
                        
                        quote_data = {
                            'symbol': symbol,
                            'current_price': quote.get('price'),
                            'previous_close': quote.get('previousClose'),
                            'change': quote.get('change'),
                            'change_percent': quote.get('changesPercentage'),
                            'volume': quote.get('volume'),
                            'market_cap': quote.get('marketCap'),
                            'high': quote.get('dayHigh'),
                            'low': quote.get('dayLow'),
                            'open': quote.get('open'),
                            'source': 'financial_modeling_prep',
                            'timestamp': datetime.utcnow().isoformat(),
                        }
                        
                        logger.info(f"FMP got quote for {symbol}: ${quote_data['current_price']}")
                        return quote_data
                        
        except Exception as e:
            logger.error(f"FMP API error for {symbol}: {e}")
            return None

    async def get_stock_quote_polygon(self, symbol: str) -> Optional[Dict]:
        """
        Get stock quote from Polygon.io
        """
        if not self.polygon_key:
            return None
            
        try:
            await self.rate_limit_check('polygon')
            
            async with httpx.AsyncClient() as client:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
                params = {
                    "adjusted": "true",
                    "apikey": self.polygon_key
                }
                
                response = await client.get(url, params=params, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'results' in data and data['results']:
                        result = data['results'][0]
                        
                        # Also get current price from different endpoint
                        current_url = f"https://api.polygon.io/v1/last/stocks/{symbol}"
                        current_response = await client.get(current_url, params={"apikey": self.polygon_key}, timeout=10.0)
                        
                        current_price = result['c']  # Close price as fallback
                        if current_response.status_code == 200:
                            current_data = current_response.json()
                            if 'last' in current_data:
                                current_price = current_data['last']['price']
                        
                        previous_close = result['c']
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                        
                        quote_data = {
                            'symbol': symbol,
                            'current_price': current_price,
                            'previous_close': previous_close,
                            'change': change,
                            'change_percent': change_percent,
                            'volume': result.get('v'),
                            'high': result.get('h'),
                            'low': result.get('l'),
                            'open': result.get('o'),
                            'source': 'polygon',
                            'timestamp': datetime.utcnow().isoformat(),
                        }
                        
                        logger.info(f"Polygon got quote for {symbol}: ${quote_data['current_price']}")
                        return quote_data
                        
        except Exception as e:
            logger.error(f"Polygon API error for {symbol}: {e}")
            return None

    async def get_stock_quote_yahoo_scraper(self, symbol: str) -> Optional[Dict]:
        """
        Backup Yahoo Finance scraper method
        """
        try:
            await self.rate_limit_check('yahoo_scraper')
            
            async with httpx.AsyncClient() as client:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        meta = result['meta']
                        
                        if 'regularMarketPrice' in meta:
                            quote_data = {
                                'symbol': symbol,
                                'current_price': meta.get('regularMarketPrice'),
                                'previous_close': meta.get('previousClose'),
                                'change': None,
                                'change_percent': None,
                                'volume': meta.get('regularMarketVolume'),
                                'high': meta.get('regularMarketDayHigh'),
                                'low': meta.get('regularMarketDayLow'),
                                'open': meta.get('regularMarketDayOpen'),
                                'source': 'yahoo_scraper',
                                'timestamp': datetime.utcnow().isoformat(),
                            }
                            
                            # Calculate change
                            if quote_data['current_price'] and quote_data['previous_close']:
                                quote_data['change'] = quote_data['current_price'] - quote_data['previous_close']
                                quote_data['change_percent'] = (quote_data['change'] / quote_data['previous_close']) * 100
                            
                            logger.info(f"Yahoo scraper got quote for {symbol}: ${quote_data['current_price']}")
                            return quote_data
                        
        except Exception as e:
            logger.error(f"Yahoo scraper error for {symbol}: {e}")
            return None

    async def rate_limit_check(self, source: str):
        """
        Implement rate limiting for API calls
        """
        current_time = time.time()
        last_time_key = f"{source}_last_request"
        
        if last_time_key in self.last_request_time:
            time_since_last = current_time - self.last_request_time[last_time_key]
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.info(f"Rate limiting {source}: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        self.last_request_time[last_time_key] = current_time

    async def get_mock_quote(self, symbol: str) -> Dict:
        """
        Generate mock data when all real sources fail
        """
        mock_prices = {
            'AAPL': 178.50, 'TSLA': 248.90, 'GOOGL': 142.80, 'MSFT': 338.25,
            'NVDA': 465.20, 'AVGO': 271.52, 'EXEL': 43.67, 'FNV': 161.32,
            'GILD': 110.90, 'HOOD': 84.18, 'LINK': 13.77, 'NVO': 68.60,
            'PLTR': 135.98, 'SFM': 166.00, 'SPOT': 775.53, 'XRP': 2.21
        }
        
        base_price = mock_prices.get(symbol, 100.0)
        import random
        price_variation = random.uniform(-0.02, 0.02)  # ±2%
        current_price = base_price * (1 + price_variation)
        change = base_price * price_variation
        change_percent = price_variation * 100
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'previous_close': base_price,
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'source': 'mock_fallback',
            'timestamp': datetime.utcnow().isoformat(),
            'note': '⚠️  Mock data - Add real API keys for live prices'
        }

    async def get_stock_quote(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get stock quote with prioritized real data sources
        """
        # Check cache first
        if use_cache and self.is_cache_valid(symbol):
            logger.info(f"Returning cached data for {symbol}")
            return self.cache[symbol]['data']
        
        # Try real data sources in order of preference
        real_sources = []
        
        if self.alpha_vantage_key:
            real_sources.append(("Alpha Vantage", self.get_stock_quote_alpha_vantage))
        if self.fmp_key:
            real_sources.append(("Financial Modeling Prep", self.get_stock_quote_fmp))
        if self.polygon_key:
            real_sources.append(("Polygon", self.get_stock_quote_polygon))
        
        # Always add Yahoo scraper as fallback
        real_sources.append(("Yahoo Scraper", self.get_stock_quote_yahoo_scraper))
        
        quote = None
        
        for source_name, source_func in real_sources:
            try:
                logger.info(f"Trying {source_name} for {symbol}")
                quote = await source_func(symbol)
                if quote and quote.get('current_price'):
                    logger.info(f"✅ Got real data from {source_name} for {symbol}: ${quote['current_price']}")
                    break
            except Exception as e:
                logger.error(f"{source_name} failed for {symbol}: {e}")
                continue
        
        # Fallback to mock data only if all real sources fail
        if not quote:
            logger.warning(f"⚠️  All real sources failed for {symbol}, using mock data")
            quote = await self.get_mock_quote(symbol)
        
        # Cache the result
        if quote:
            self.cache[symbol] = {
                'data': quote,
                'cached_at': datetime.utcnow().isoformat()
            }
        
        return quote

    def is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[symbol]['cached_at'])
        return (datetime.utcnow() - cached_time).seconds < self.cache_duration

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols with proper rate limiting"""
        if not symbols:
            return {}
        
        logger.info(f"🔄 Fetching quotes for {len(symbols)} symbols")
        
        # Limit symbols to avoid hitting API limits
        limited_symbols = symbols[:10]  # Limit to 10 for free APIs
        quotes = {}
        
        for i, symbol in enumerate(limited_symbols):
            quote = await self.get_stock_quote(symbol)
            if quote:
                quotes[symbol] = quote
                
            # Progress logging
            logger.info(f"📊 Processed {i+1}/{len(limited_symbols)}: {symbol}")
            
            # Small delay between requests for API health
            if i < len(limited_symbols) - 1:
                await asyncio.sleep(0.5)
        
        success_count = len([q for q in quotes.values() if q.get('current_price')])
        logger.info(f"✅ Successfully fetched {success_count}/{len(limited_symbols)} real quotes")
        
        return quotes

    def enhance_portfolio_with_market_data(self, portfolio_data: Dict, market_quotes: Dict) -> Dict:
        """Enhance portfolio with live market data"""
        enhanced_portfolio = portfolio_data.copy()
        
        if 'holdings' not in enhanced_portfolio:
            return enhanced_portfolio
        
        total_current_value = 0
        total_gain_loss = 0
        holdings_with_data = 0
        real_data_count = 0
        
        for holding in enhanced_portfolio['holdings']:
            symbol = holding.get('symbol')
            if symbol and symbol in market_quotes:
                quote = market_quotes[symbol]
                current_price = quote.get('current_price')
                
                if current_price and holding.get('quantity'):
                    # Mark if this is real vs mock data
                    is_real_data = quote.get('source') != 'mock_fallback'
                    if is_real_data:
                        real_data_count += 1
                    
                    # Update with market data
                    holding['live_price'] = current_price
                    holding['live_market_value'] = current_price * holding['quantity']
                    holding['price_change'] = quote.get('change', 0)
                    holding['price_change_percent'] = quote.get('change_percent', 0)
                    holding['data_source'] = quote.get('source')
                    holding['last_updated'] = quote.get('timestamp')
                    holding['is_real_data'] = is_real_data
                    
                    # Calculate gains/losses
                    if holding.get('average_cost'):
                        cost_basis = holding['average_cost'] * holding['quantity']
                        holding['live_gain_loss'] = holding['live_market_value'] - cost_basis
                        holding['live_gain_loss_percent'] = ((holding['live_market_value'] - cost_basis) / cost_basis) * 100
                    
                    total_current_value += holding['live_market_value']
                    if holding.get('live_gain_loss'):
                        total_gain_loss += holding['live_gain_loss']
                    
                    holdings_with_data += 1
        
        # Update portfolio totals
        if holdings_with_data > 0:
            enhanced_portfolio['live_total_value'] = total_current_value
            enhanced_portfolio['live_total_gain_loss'] = total_gain_loss
            enhanced_portfolio['live_total_gain_loss_percent'] = (total_gain_loss / (total_current_value - total_gain_loss)) * 100 if (total_current_value - total_gain_loss) > 0 else 0
            enhanced_portfolio['holdings_with_live_data'] = holdings_with_data
            enhanced_portfolio['real_data_count'] = real_data_count
            enhanced_portfolio['mock_data_count'] = holdings_with_data - real_data_count
            enhanced_portfolio['market_data_timestamp'] = datetime.utcnow().isoformat()
            
            logger.info(f"📊 Portfolio enhanced: live_value=${total_current_value:.2f}, real_data={real_data_count}, mock_data={holdings_with_data - real_data_count}")
        
        return enhanced_portfolio