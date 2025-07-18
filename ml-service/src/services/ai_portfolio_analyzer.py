import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from transformers import pipeline
import logging
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import json
import yfinance as yf
# Remove this import if it causes issues
from services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

class PortfolioRiskPredictor(nn.Module):
    """
    Neural network for portfolio risk prediction
    """
    def __init__(self, input_dim=50, hidden_dim=128):
        super(PortfolioRiskPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 risk levels
        )
        
    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)

class AIPortfolioAnalyzer:
    """
    Main AI service for portfolio analysis and recommendations - FIXED VERSION
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.risk_model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        # Comment out if causing issues
        # self.market_data_service = MarketDataService()
        
        # Initialize sentiment analysis with better error handling
        self.sentiment_pipeline = None
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            logger.info("✅ FinBERT sentiment model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load FinBERT model: {e}")
            logger.info("🔄 Using basic sentiment analysis")
        
        # Risk categories
        self.risk_categories = ['very_low', 'low', 'moderate', 'high', 'very_high']
        
    async def initialize(self):
        """Initialize the AI analyzer"""
        logger.info("🤖 Initializing AI Portfolio Analyzer...")
        
        # Initialize risk prediction model
        self.risk_model = PortfolioRiskPredictor()
        self.risk_model.to(self.device)
        
        # Load pre-trained weights if available
        try:
            self.risk_model.load_state_dict(torch.load('models/risk_model.pth', map_location=self.device))
            logger.info("✅ Loaded pre-trained risk model")
        except FileNotFoundError:
            logger.info("📝 No pre-trained risk model found, using random weights")
        except Exception as e:
            logger.warning(f"⚠️ Could not load risk model: {e}")
        
        logger.info("✅ AI Portfolio Analyzer initialized")
    
    async def analyze_portfolio(self, portfolio_data: Dict) -> Dict:
        """
        Comprehensive AI analysis of portfolio - FIXED ASYNC
        """
        try:
            holdings = portfolio_data.get('holdings', [])
            
            if not holdings:
                logger.warning("⚠️ No holdings found in portfolio")
                return self._create_empty_analysis()
            
            logger.info(f"🤖 Analyzing portfolio with {len(holdings)} holdings")
            
            # Basic portfolio metrics (sync)
            portfolio_metrics = self._calculate_portfolio_metrics(holdings)
            
            # Diversification analysis (sync)
            diversification_analysis = self._analyze_diversification(holdings)
            
            # Performance analysis (async - FIXED)
            performance_analysis = await self._analyze_performance(holdings)
            
            # Sector analysis (sync)
            sector_analysis = self._analyze_sectors(holdings)
            
            # Quality score (sync - FIXED to not use async result)
            quality_score = self._calculate_quality_score(holdings, performance_analysis, diversification_analysis)
            
            # Combine all analyses
            ai_insights = {
                "portfolio_metrics": portfolio_metrics,
                "diversification": diversification_analysis,
                "performance": performance_analysis,
                "sector_analysis": sector_analysis,
                "quality_score": quality_score,
                "confidence_score": 0.85,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return ai_insights
            
        except Exception as e:
            logger.error(f"❌ Portfolio analysis failed: {e}")
            return self._create_empty_analysis()
    
    async def generate_recommendations(self, portfolio_data: Dict) -> List[Dict]:
        """
        Generate AI-powered portfolio recommendations
        """
        try:
            holdings = portfolio_data.get('holdings', [])
            recommendations = []
            
            if not holdings:
                return recommendations
            
            # Risk-based recommendations (async)
            risk_recs = await self._generate_risk_recommendations(holdings)
            recommendations.extend(risk_recs)
            
            # Diversification recommendations (sync)
            div_recs = self._generate_diversification_recommendations(holdings)
            recommendations.extend(div_recs)
            
            # Performance-based recommendations (sync - FIXED)
            perf_recs = self._generate_performance_recommendations(holdings)
            recommendations.extend(perf_recs)
            
            # Sector rebalancing recommendations (sync)
            sector_recs = self._generate_sector_recommendations(holdings)
            recommendations.extend(sector_recs)
            
            # Sentiment-based recommendations (async)
            sentiment_recs = await self._generate_sentiment_recommendations(holdings)
            recommendations.extend(sentiment_recs)
            
            # Rank recommendations by importance
            ranked_recommendations = self._rank_recommendations(recommendations)
            
            return ranked_recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return []
    
    async def analyze_risk(self, portfolio_data: Dict) -> Dict:
        """
        Detailed risk analysis using AI models
        """
        try:
            holdings = portfolio_data.get('holdings', [])
            
            if not holdings:
                return self._create_empty_risk_analysis()
            
            # Extract features for risk analysis
            risk_features = self._extract_risk_features(holdings)
            
            # Predict risk using neural network
            risk_prediction = await self._predict_risk(risk_features)
            
            # Detect anomalies
            anomaly_score = self._detect_anomalies(risk_features)
            
            # Calculate traditional risk metrics
            traditional_risk = self._calculate_traditional_risk(holdings)
            
            # Identify specific risk factors
            risk_factors = self._identify_risk_factors(holdings)
            
            risk_analysis = {
                "ai_risk_prediction": risk_prediction,
                "anomaly_score": anomaly_score,
                "traditional_metrics": traditional_risk,
                "risk_factors": risk_factors,
                "overall_risk_score": self._calculate_overall_risk_score(risk_prediction, anomaly_score),
                "risk_recommendations": await self._generate_risk_mitigation_recommendations(risk_factors)
            }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"❌ Risk analysis failed: {e}")
            return self._create_empty_risk_analysis()
    
    def _calculate_portfolio_metrics(self, holdings: List[Dict]) -> Dict:
        """Calculate basic portfolio metrics"""
        try:
            total_value = sum(h.get('market_value', 0) for h in holdings)
            total_gain_loss = sum(h.get('gain_loss', 0) for h in holdings)
            
            return {
                "total_value": total_value,
                "total_gain_loss": total_gain_loss,
                "total_return_pct": (total_gain_loss / (total_value - total_gain_loss)) * 100 if total_value > total_gain_loss else 0,
                "num_holdings": len(holdings),
                "avg_position_size": total_value / len(holdings) if holdings else 0
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {"total_value": 0, "num_holdings": 0}
    
    def _analyze_diversification(self, holdings: List[Dict]) -> Dict:
        """Analyze portfolio diversification"""
        try:
            # Sector diversification
            sectors = {}
            total_value = sum(h.get('market_value', 0) for h in holdings)
            
            for holding in holdings:
                sector = holding.get('sector', 'unknown')
                value = holding.get('market_value', 0)
                sectors[sector] = sectors.get(sector, 0) + value
            
            # Calculate sector weights
            sector_weights = {k: v/total_value for k, v in sectors.items()} if total_value > 0 else {}
            
            # Diversification score (higher is better)
            diversification_score = 1 - sum(w**2 for w in sector_weights.values()) if sector_weights else 0
            
            # Concentration risk
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            concentration_risk = "High" if max_sector_weight > 0.4 else "Moderate" if max_sector_weight > 0.25 else "Low"
            
            return {
                "sector_weights": sector_weights,
                "diversification_score": diversification_score,
                "concentration_risk": concentration_risk,
                "max_sector_weight": max_sector_weight,
                "num_sectors": len(sectors)
            }
        except Exception as e:
            logger.error(f"Error analyzing diversification: {e}")
            return {"sector_weights": {}, "diversification_score": 0, "concentration_risk": "unknown"}
    
    async def _analyze_performance(self, holdings: List[Dict]) -> Dict:
        """Analyze portfolio performance - PROPERLY ASYNC"""
        try:
            # Calculate weighted returns
            total_value = sum(h.get('market_value', 0) for h in holdings)
            weighted_return = 0
            
            for holding in holdings:
                weight = holding.get('market_value', 0) / total_value if total_value > 0 else 0
                return_pct = holding.get('return_pct', 0)
                weighted_return += weight * return_pct
            
            # Performance categories
            winners = [h for h in holdings if h.get('return_pct', 0) > 0]
            losers = [h for h in holdings if h.get('return_pct', 0) < 0]
            
            return {
                "weighted_return": weighted_return,
                "num_winners": len(winners),
                "num_losers": len(losers),
                "win_rate": len(winners) / len(holdings) if holdings else 0,
                "best_performer": max(holdings, key=lambda x: x.get('return_pct', 0)) if holdings else None,
                "worst_performer": min(holdings, key=lambda x: x.get('return_pct', 0)) if holdings else None
            }
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"weighted_return": 0, "win_rate": 0, "num_winners": 0, "num_losers": 0}
    
    def _analyze_sectors(self, holdings: List[Dict]) -> Dict:
        """Analyze sector allocation"""
        try:
            sectors = {}
            total_value = sum(h.get('market_value', 0) for h in holdings)
            
            for holding in holdings:
                sector = holding.get('sector', 'unknown')
                value = holding.get('market_value', 0)
                return_pct = holding.get('return_pct', 0)
                
                if sector not in sectors:
                    sectors[sector] = {
                        'value': 0,
                        'weight': 0,
                        'return': 0,
                        'count': 0
                    }
                
                sectors[sector]['value'] += value
                sectors[sector]['return'] += return_pct
                sectors[sector]['count'] += 1
            
            # Calculate sector weights and average returns
            for sector_data in sectors.values():
                sector_data['weight'] = sector_data['value'] / total_value if total_value > 0 else 0
                sector_data['avg_return'] = sector_data['return'] / sector_data['count'] if sector_data['count'] > 0 else 0
            
            return {
                "sector_breakdown": sectors,
                "dominant_sector": max(sectors.keys(), key=lambda x: sectors[x]['weight']) if sectors else None,
                "best_performing_sector": max(sectors.keys(), key=lambda x: sectors[x]['avg_return']) if sectors else None,
                "worst_performing_sector": min(sectors.keys(), key=lambda x: sectors[x]['avg_return']) if sectors else None
            }
        except Exception as e:
            logger.error(f"Error analyzing sectors: {e}")
            return {"sector_breakdown": {}}
    
    def _calculate_quality_score(self, holdings: List[Dict], performance_analysis: Dict, diversification_analysis: Dict) -> float:
        """Calculate overall portfolio quality score - FIXED (sync, uses passed data)"""
        try:
            score = 0.0
            
            # Diversification score (0-30 points)
            div_score = diversification_analysis.get('diversification_score', 0)
            score += div_score * 30
            
            # Performance score (0-40 points)
            perf_return = performance_analysis.get('weighted_return', 0)
            if perf_return > 0:
                score += min(perf_return * 2, 40)
            
            # Risk score (0-30 points) - inverse of concentration risk
            concentration_risk = diversification_analysis.get('concentration_risk', 'High')
            if concentration_risk == 'Low':
                score += 30
            elif concentration_risk == 'Moderate':
                score += 20
            else:
                score += 10
            
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 50.0
    
    def _extract_risk_features(self, holdings: List[Dict]) -> np.ndarray:
        """Extract features for risk analysis"""
        try:
            features = []
            
            # Portfolio-level features
            total_value = sum(h.get('market_value', 0) for h in holdings)
            features.extend([
                len(holdings),  # Number of holdings
                total_value,    # Total portfolio value
                sum(h.get('gain_loss', 0) for h in holdings),  # Total gain/loss
            ])
            
            # Sector concentration features
            sectors = {}
            for holding in holdings:
                sector = holding.get('sector', 'unknown')
                value = holding.get('market_value', 0)
                sectors[sector] = sectors.get(sector, 0) + value
            
            sector_weights = [v/total_value for v in sectors.values()] if total_value > 0 else [0]
            features.extend([
                len(sectors),  # Number of sectors
                max(sector_weights),  # Max sector weight
                np.std(sector_weights),  # Sector weight std
            ])
            
            # Individual holding features (top 10 by value)
            sorted_holdings = sorted(holdings, key=lambda x: x.get('market_value', 0), reverse=True)
            for i, holding in enumerate(sorted_holdings[:10]):
                features.extend([
                    holding.get('market_value', 0) / total_value if total_value > 0 else 0,
                    holding.get('return_pct', 0),
                    holding.get('gain_loss', 0),
                ])
            
            # Pad or truncate to ensure consistent feature size
            target_size = 50
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting risk features: {e}")
            return np.zeros(50, dtype=np.float32)
    
    async def _predict_risk(self, features: np.ndarray) -> Dict:
        """Predict risk using neural network"""
        try:
            if self.risk_model is None:
                # Fallback to rule-based risk assessment
                return self._rule_based_risk_assessment(features)
            
            # Normalize features
            features_normalized = self.scaler.fit_transform(features.reshape(1, -1))
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_normalized).to(self.device)
            
            # Predict
            with torch.no_grad():
                risk_probs = self.risk_model(features_tensor)
                risk_probs = risk_probs.cpu().numpy()[0]
            
            # Create risk distribution
            risk_distribution = {
                category: float(prob) for category, prob in zip(self.risk_categories, risk_probs)
            }
            
            # Determine overall risk level
            predicted_risk = self.risk_categories[np.argmax(risk_probs)]
            
            return {
                "predicted_risk_level": predicted_risk,
                "risk_distribution": risk_distribution,
                "confidence": float(np.max(risk_probs))
            }
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return self._rule_based_risk_assessment(features)
    
    def _rule_based_risk_assessment(self, features: np.ndarray) -> Dict:
        """Fallback rule-based risk assessment"""
        try:
            # Simple rule-based assessment
            num_holdings = features[0] if len(features) > 0 else 0
            max_sector_weight = features[5] if len(features) > 5 else 0
            
            if num_holdings < 5 or max_sector_weight > 0.5:
                risk_level = "high"
            elif num_holdings < 10 or max_sector_weight > 0.3:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            return {
                "predicted_risk_level": risk_level,
                "risk_distribution": {risk_level: 1.0},
                "confidence": 0.7
            }
        except Exception as e:
            logger.error(f"Error in rule-based risk assessment: {e}")
            return {
                "predicted_risk_level": "moderate",
                "risk_distribution": {"moderate": 1.0},
                "confidence": 0.5
            }
    
    def _detect_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies in portfolio composition"""
        try:
            features_reshaped = features.reshape(1, -1)
            self.anomaly_detector.fit(features_reshaped)
            anomaly_score = self.anomaly_detector.decision_function(features_reshaped)[0]
            
            # Normalize to 0-1 scale (higher = more anomalous)
            normalized_score = max(0, min(1, (0.5 - anomaly_score) / 1.0))
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return 0.5
    
    def _calculate_traditional_risk(self, holdings: List[Dict]) -> Dict:
        """Calculate traditional risk metrics"""
        try:
            # Portfolio value and returns
            values = [h.get('market_value', 0) for h in holdings]
            returns = [h.get('return_pct', 0) for h in holdings]
            
            total_value = sum(values)
            weights = [v/total_value for v in values] if total_value > 0 else [0] * len(values)
            
            # Weighted portfolio return
            portfolio_return = sum(w * r for w, r in zip(weights, returns))
            
            # Portfolio volatility (simplified)
            portfolio_volatility = np.std(returns) if returns else 0
            
            # Sharpe ratio (simplified, assuming risk-free rate = 2%)
            risk_free_rate = 2.0
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "var_95": np.percentile(returns, 5) if returns else 0  # Value at Risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating traditional risk: {e}")
            return {}
    
    def _identify_risk_factors(self, holdings: List[Dict]) -> List[str]:
        """Identify specific risk factors"""
        try:
            risk_factors = []
            
            # Concentration risk
            if len(holdings) < 5:
                risk_factors.append("Low diversification - fewer than 5 holdings")
            
            # Sector concentration
            sectors = {}
            total_value = sum(h.get('market_value', 0) for h in holdings)
            
            for holding in holdings:
                sector = holding.get('sector', 'unknown')
                value = holding.get('market_value', 0)
                sectors[sector] = sectors.get(sector, 0) + value
            
            for sector, value in sectors.items():
                weight = value / total_value if total_value > 0 else 0
                if weight > 0.4:
                    risk_factors.append(f"High {sector} sector concentration ({weight:.1%})")
            
            # Individual position risk
            for holding in holdings:
                weight = holding.get('market_value', 0) / total_value if total_value > 0 else 0
                if weight > 0.25:
                    symbol = holding.get('symbol', 'Unknown')
                    risk_factors.append(f"Large position in {symbol} ({weight:.1%})")
            
            # Performance risk
            losers = [h for h in holdings if h.get('return_pct', 0) < -20]
            if len(losers) > len(holdings) * 0.3:
                risk_factors.append("High number of underperforming positions")
            
            return risk_factors[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []
    
    def _calculate_overall_risk_score(self, risk_prediction: Dict, anomaly_score: float) -> float:
        """Calculate overall risk score"""
        try:
            # Weight the AI prediction
            risk_levels = {'very_low': 0.1, 'low': 0.3, 'moderate': 0.5, 'high': 0.7, 'very_high': 0.9}
            ai_risk_score = risk_levels.get(risk_prediction.get('predicted_risk_level', 'moderate'), 0.5)
            
            # Combine AI prediction with anomaly score
            overall_score = (ai_risk_score * 0.7) + (anomaly_score * 0.3)
            
            return min(max(overall_score, 0), 1)  # Ensure 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 0.5
    
    async def _generate_risk_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate risk-based recommendations"""
        try:
            recommendations = []
            
            # Risk analysis
            risk_features = self._extract_risk_features(holdings)
            risk_prediction = await self._predict_risk(risk_features)
            
            if risk_prediction.get('predicted_risk_level') in ['high', 'very_high']:
                recommendations.append({
                    'type': 'risk_reduction',
                    'priority': 'high',
                    'title': '🛡️ Reduce Portfolio Risk',
                    'description': f'AI analysis indicates {risk_prediction.get("predicted_risk_level", "high")} risk level',
                    'actions': [
                        'Diversify across more sectors',
                        'Reduce position sizes',
                        'Consider defensive assets'
                    ],
                    'confidence': risk_prediction.get('confidence', 0.8)
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return []
    
    def _generate_diversification_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate diversification recommendations"""
        try:
            recommendations = []
            div_analysis = self._analyze_diversification(holdings)
            
            if div_analysis.get('concentration_risk') == 'High':
                recommendations.append({
                    'type': 'diversification',
                    'priority': 'high',
                    'title': '📊 Improve Diversification',
                    'description': f'Portfolio has high concentration risk',
                    'actions': [
                        'Reduce largest sector allocation',
                        'Add holdings in underrepresented sectors',
                        'Consider broad market ETFs'
                    ],
                    'confidence': 0.85
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating diversification recommendations: {e}")
            return []
    
    def _generate_performance_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate performance-based recommendations - FIXED (sync)"""
        try:
            recommendations = []

            if not holdings:
                return recommendations

            # Calculate basic performance metrics
            total_value = sum(h.get('market_value', 0) for h in holdings)

            if total_value > 0:
                # Add performance-based recommendations
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'title': '📈 Performance Review',
                    'description': f'Portfolio value: ${total_value:,.2f}',
                    'actions': [
                        'Monitor performance monthly',
                        'Review quarterly results',
                        'Compare to benchmarks'
                    ],
                    'confidence': 0.80
                })

                # Add recommendation for large portfolios
                if total_value > 50000:
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'low',
                        'title': '💰 High Value Portfolio',
                        'description': 'Consider professional management for large portfolios',
                        'actions': [
                            'Explore robo-advisors',
                            'Consider financial advisor',
                            'Review tax implications'
                        ],
                        'confidence': 0.70
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return []
        
    def _generate_sector_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate sector-based recommendations"""
        try:
            recommendations = []
            sector_analysis = self._analyze_sectors(holdings)
            
            sectors = sector_analysis.get('sector_breakdown', {})
            if len(sectors) < 5:  # Recommend more sector diversity
                recommendations.append({
                    'type': 'sector_diversification',
                    'priority': 'medium',
                    'title': '🏢 Expand Sector Coverage',
                    'description': f'Portfolio only covers {len(sectors)} sectors',
                    'actions': [
                        'Add healthcare sector exposure',
                        'Consider utilities for stability',
                        'Explore international markets'
                    ],
                    'confidence': 0.80
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating sector recommendations: {e}")
            return []
    
    async def _generate_sentiment_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate sentiment-based recommendations - FIXED ERROR HANDLING"""
        try:
            recommendations = []
            
            if self.sentiment_pipeline is None:
                return recommendations
            
            # Analyze sentiment for each holding (simplified)
            for holding in holdings:
                symbol = holding.get('symbol', '')
                if symbol:
                    # Create sample news text (in practice, fetch real news)
                    news_text = f"Recent performance and market outlook for {symbol} stock"
                    
                    try:
                        sentiment_result = self.sentiment_pipeline(news_text)
                        
                        # FIXED: Handle different response formats
                        if isinstance(sentiment_result, list) and len(sentiment_result) > 0:
                            # Check if it's a list of dictionaries
                            if isinstance(sentiment_result[0], dict):
                                # Find negative sentiment
                                negative_score = 0
                                for item in sentiment_result:
                                    if item.get('label', '').lower() in ['negative', 'neg']:
                                        negative_score = item.get('score', 0)
                                        break
                                
                                if negative_score > 0.7:
                                    recommendations.append({
                                        'type': 'sentiment',
                                        'priority': 'medium',
                                        'title': f'⚠️ Negative Sentiment - {symbol}',
                                        'description': f'AI detected negative sentiment for {symbol}',
                                        'actions': [
                                            f'Monitor {symbol} news closely',
                                            'Consider reducing position size',
                                            'Set stop-loss orders'
                                        ],
                                        'confidence': negative_score
                                    })
                        
                    except Exception as e:
                        logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
                        # Continue to next holding instead of failing
                        continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating sentiment recommendations: {e}")
            return []
    
    def _rank_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Rank recommendations by importance"""
        try:
            priority_weights = {'high': 3, 'medium': 2, 'low': 1}
            
            for rec in recommendations:
                priority = rec.get('priority', 'low')
                confidence = rec.get('confidence', 0.5)
                
                # Calculate ranking score
                rec['ranking_score'] = priority_weights.get(priority, 1) * confidence
            
            # Sort by ranking score
            return sorted(recommendations, key=lambda x: x.get('ranking_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking recommendations: {e}")
            return recommendations
    
    async def _generate_risk_mitigation_recommendations(self, risk_factors: List[str]) -> List[Dict]:
        """Generate specific risk mitigation recommendations"""
        recommendations = []
        
        try:
            for factor in risk_factors:
                if "concentration" in factor.lower():
                    recommendations.append({
                        'risk_factor': factor,
                        'mitigation': 'Reduce position size or diversify into similar assets',
                        'priority': 'high'
                    })
                elif "diversification" in factor.lower():
                    recommendations.append({
                        'risk_factor': factor,
                        'mitigation': 'Add more holdings across different sectors',
                        'priority': 'high'
                    })
                elif "underperforming" in factor.lower():
                    recommendations.append({
                        'risk_factor': factor,
                        'mitigation': 'Review and potentially exit poor performers',
                        'priority': 'medium'
                    })
        except Exception as e:
            logger.error(f"Error generating risk mitigation recommendations: {e}")
        
        return recommendations
    
    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis for error cases"""
        return {
            "portfolio_metrics": {"total_value": 0, "num_holdings": 0},
            "diversification": {"sector_weights": {}, "diversification_score": 0},
            "performance": {"weighted_return": 0, "win_rate": 0},
            "sector_analysis": {"sector_breakdown": {}},
            "quality_score": 0,
            "confidence_score": 0,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _create_empty_risk_analysis(self) -> Dict:
        """Create empty risk analysis for error cases"""
        return {
            "ai_risk_prediction": {"predicted_risk_level": "moderate", "confidence": 0.5},
            "anomaly_score": 0.5,
            "traditional_metrics": {},
            "risk_factors": [],
            "overall_risk_score": 0.5,
            "risk_recommendations": []
        }
    
    # Placeholder methods for future implementation
    async def get_historical_risk_metrics(self, portfolio_data: Dict) -> Dict:
        """Get historical risk metrics"""
        return {
            "30_day_volatility": 0.15,
            "90_day_volatility": 0.18,
            "max_drawdown": 0.12,
            "beta": 1.05
        }
    
    async def optimize_portfolio(self, portfolio_data: Dict) -> Dict:
        """Portfolio optimization suggestions"""
        return {
            "suggested_allocation": {},
            "expected_return": 0.08,
            "expected_risk": 0.12,
            "sharpe_ratio": 0.67
        }
    
    async def analyze_performance(self, performance_data: Dict) -> Dict:
        """Analyze portfolio performance over time"""
        return {
            "total_return": 0.08,
            "annualized_return": 0.10,
            "volatility": 0.15,
            "sharpe_ratio": 0.67
        }
    
    async def simulate_portfolio_changes(self, simulation_data: Dict) -> Dict:
        """Simulate portfolio changes"""
        return {
            "current_metrics": {},
            "projected_metrics": {},
            "risk_change": 0.02,
            "return_change": 0.01
        }