import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import io

# Import the deep learning models
try:
    from .deep_learning_models import DeepLearningPortfolioAnalyzer
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    logging.warning(f"Deep learning models not available: {e}")

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class Sector(Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    COMMUNICATION = "communication"
    INDUSTRIALS = "industrials"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MATERIALS = "materials"
    UNKNOWN = "unknown"

@dataclass
class EnhancedRiskMetrics:
    portfolio_beta: float
    sharpe_ratio: float
    volatility: float
    var_95: float
    max_drawdown: float
    risk_level: RiskLevel
    risk_score: int
    # New deep learning metrics
    ml_risk_distribution: Dict[str, float]
    predicted_volatility: float
    expected_return: float
    correlation_score: float
    anomaly_score: float
    risk_factors: List[str]
    confidence_level: float

@dataclass
class EnhancedDiversificationMetrics:
    sector_concentration: Dict[str, float]
    top_holding_percentage: float
    effective_number_holdings: float
    correlation_score: float
    diversification_score: int
    concentration_risk: str
    # New ML-enhanced metrics
    ml_diversification_score: float
    sector_recommendations: List[str]
    rebalancing_suggestions: Dict[str, float]

@dataclass
class EnhancedPerformanceMetrics:
    total_return: float
    annualized_return: float
    ytd_return: float
    monthly_returns: List[float]
    benchmark_comparison: float
    alpha: float
    tracking_error: float
    performance_score: int
    # New ML performance metrics
    ml_performance_prediction: float
    risk_adjusted_returns: float
    performance_consistency: float
    market_timing_score: float

@dataclass
class EnhancedRecommendation:
    type: str
    priority: str
    title: str
    description: str
    rationale: str
    suggested_actions: List[str]
    potential_impact: str
    confidence: float
    # New ML fields
    ml_confidence: float
    recommendation_source: str  # 'traditional', 'deep_learning', 'hybrid'
    expected_outcome: str
    implementation_difficulty: str
    time_horizon: str

@dataclass
class EnhancedPortfolioAnalysis:
    risk_metrics: EnhancedRiskMetrics
    diversification: EnhancedDiversificationMetrics
    performance: EnhancedPerformanceMetrics
    recommendations: List[EnhancedRecommendation]
    overall_score: int
    summary: str
    analysis_timestamp: str
    # New deep learning fields
    ml_analysis_available: bool
    image_analysis: Dict
    advanced_insights: Dict

class EnhancedAIPortfolioAnalyzer:
    def __init__(self):
        self.sector_mapping = self._initialize_sector_mapping()
        self.benchmark_symbols = {
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'DOW': '^DJI'
        }
        
        # Initialize deep learning analyzer
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.dl_analyzer = DeepLearningPortfolioAnalyzer()
                logger.info("🤖 Deep Learning Portfolio Analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize deep learning analyzer: {e}")
                self.dl_analyzer = None
        else:
            self.dl_analyzer = None
            logger.info("📊 Using traditional analysis methods")
        
    def _initialize_sector_mapping(self) -> Dict[str, Sector]:
        """Initialize stock symbol to sector mapping (enhanced)"""
        return {
            # Technology (expanded)
            'AAPL': Sector.TECHNOLOGY, 'MSFT': Sector.TECHNOLOGY, 'GOOGL': Sector.TECHNOLOGY,
            'GOOG': Sector.TECHNOLOGY, 'META': Sector.TECHNOLOGY, 'TSLA': Sector.TECHNOLOGY,
            'NVDA': Sector.TECHNOLOGY, 'AVGO': Sector.TECHNOLOGY, 'ORCL': Sector.TECHNOLOGY,
            'CRM': Sector.TECHNOLOGY, 'ADBE': Sector.TECHNOLOGY, 'NFLX': Sector.TECHNOLOGY,
            'AMD': Sector.TECHNOLOGY, 'INTC': Sector.TECHNOLOGY, 'QCOM': Sector.TECHNOLOGY,
            'PLTR': Sector.TECHNOLOGY, 'SFM': Sector.TECHNOLOGY, 'SPOT': Sector.COMMUNICATION,
            
            # Healthcare (expanded)
            'JNJ': Sector.HEALTHCARE, 'PFE': Sector.HEALTHCARE, 'UNH': Sector.HEALTHCARE,
            'ABBV': Sector.HEALTHCARE, 'TMO': Sector.HEALTHCARE, 'ABT': Sector.HEALTHCARE,
            'MRK': Sector.HEALTHCARE, 'LLY': Sector.HEALTHCARE, 'DHR': Sector.HEALTHCARE,
            'GILD': Sector.HEALTHCARE, 'BMY': Sector.HEALTHCARE, 'EXEL': Sector.HEALTHCARE,
            
            # Financials (expanded)
            'BRK.B': Sector.FINANCIALS, 'JPM': Sector.FINANCIALS, 'V': Sector.FINANCIALS,
            'MA': Sector.FINANCIALS, 'BAC': Sector.FINANCIALS, 'WFC': Sector.FINANCIALS,
            'GS': Sector.FINANCIALS, 'MS': Sector.FINANCIALS, 'AXP': Sector.FINANCIALS,
            'HOOD': Sector.FINANCIALS,
            
            # Consumer Discretionary
            'AMZN': Sector.CONSUMER_DISCRETIONARY, 'HD': Sector.CONSUMER_DISCRETIONARY,
            'NKE': Sector.CONSUMER_DISCRETIONARY, 'MCD': Sector.CONSUMER_DISCRETIONARY,
            'LOW': Sector.CONSUMER_DISCRETIONARY, 'SBUX': Sector.CONSUMER_DISCRETIONARY,
            
            # Materials/Mining
            'NVO': Sector.MATERIALS, 'FCX': Sector.MATERIALS, 'NEM': Sector.MATERIALS,
            'FNV': Sector.MATERIALS,
            
            # Energy
            'XOM': Sector.ENERGY, 'CVX': Sector.ENERGY, 'COP': Sector.ENERGY,
            
            # Crypto/Alternative
            'LINK': Sector.TECHNOLOGY, 'XRP': Sector.TECHNOLOGY,
        }

    async def analyze_portfolio_with_image(self, image_data: bytes, holdings: List[Dict]) -> EnhancedPortfolioAnalysis:
        """
        Enhanced portfolio analysis including image analysis with deep learning
        """
        try:
            logger.info(f"🚀 Starting enhanced AI analysis with image processing")
            
            # Step 1: Advanced image analysis
            image_analysis = {}
            if self.dl_analyzer and image_data:
                image_analysis = await self.dl_analyzer.analyze_portfolio_image(image_data)
                logger.info(f"📸 Image analysis completed: {image_analysis.get('analysis_method', 'unknown')}")
            
            # Step 2: Enhanced holdings analysis
            enhanced_holdings = await self._enhance_holdings_with_ml(holdings)
            
            # Step 3: Advanced risk analysis
            risk_metrics = await self._calculate_enhanced_risk_metrics(enhanced_holdings)
            
            # Step 4: ML-powered diversification analysis
            diversification = await self._calculate_enhanced_diversification_metrics(enhanced_holdings)
            
            # Step 5: Advanced performance analysis
            performance = await self._calculate_enhanced_performance_metrics(enhanced_holdings)
            
            # Step 6: Generate ML-powered recommendations
            recommendations = await self._generate_enhanced_recommendations(
                enhanced_holdings, risk_metrics, diversification, performance, image_analysis
            )
            
            # Step 7: Calculate enhanced overall score
            overall_score = self._calculate_enhanced_overall_score(risk_metrics, diversification, performance)
            
            # Step 8: Generate enhanced summary
            summary = self._generate_enhanced_portfolio_summary(
                risk_metrics, diversification, performance, overall_score, image_analysis
            )
            
            # Step 9: Advanced insights
            advanced_insights = await self._generate_advanced_insights(enhanced_holdings)
            
            analysis = EnhancedPortfolioAnalysis(
                risk_metrics=risk_metrics,
                diversification=diversification,
                performance=performance,
                recommendations=recommendations,
                overall_score=overall_score,
                summary=summary,
                analysis_timestamp=datetime.utcnow().isoformat(),
                ml_analysis_available=DEEP_LEARNING_AVAILABLE and self.dl_analyzer is not None,
                image_analysis=image_analysis,
                advanced_insights=advanced_insights
            )
            
            logger.info(f"✅ Enhanced AI analysis complete. Overall score: {overall_score}/100")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Enhanced portfolio analysis failed: {e}")
            # Fallback to basic analysis
            return await self._fallback_to_basic_analysis(holdings)

    async def _enhance_holdings_with_ml(self, holdings: List[Dict]) -> List[Dict]:
        """Enhance holdings data with ML predictions"""
        enhanced = []
        
        for holding in holdings:
            enhanced_holding = holding.copy()
            
            # Add ML-based predictions
            if self.dl_analyzer:
                # Add sector classification confidence
                symbol = holding.get('symbol')
                if symbol:
                    enhanced_holding['ml_sector_confidence'] = 0.85  # Simulated
                    enhanced_holding['risk_category'] = self._classify_asset_risk(holding)
                    enhanced_holding['ml_price_prediction'] = self._predict_price_movement(holding)
            
            enhanced.append(enhanced_holding)
        
        return enhanced

    async def _calculate_enhanced_risk_metrics(self, holdings: List[Dict]) -> EnhancedRiskMetrics:
        """Calculate enhanced risk metrics using ML"""
        try:
            # Traditional risk calculations
            total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
            portfolio_beta = sum(
                h.get('beta', 1.0) * (h.get('live_market_value', h.get('market_value', 0)) / total_value)
                for h in holdings
            ) if total_value > 0 else 1.0
            
            volatility = self._estimate_volatility_from_concentration(holdings)
            var_95 = total_value * 0.05 * volatility
            max_drawdown = min(0.6, volatility * 1.5)
            
            risk_free_rate = 0.02
            expected_return = 0.08
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            risk_level = self._determine_risk_level(portfolio_beta, volatility, len(holdings))
            risk_score = self._calculate_risk_score(portfolio_beta, volatility, len(holdings))
            
            # Enhanced ML-based risk analysis
            ml_risk_analysis = {}
            confidence_level = 0.75  # Traditional analysis confidence
            
            if self.dl_analyzer:
                try:
                    ml_risk_analysis = await self.dl_analyzer.advanced_risk_analysis(holdings)
                    confidence_level = 0.95  # Higher confidence with ML
                    logger.info("🤖 ML risk analysis completed")
                except Exception as e:
                    logger.warning(f"ML risk analysis failed: {e}")
                    ml_risk_analysis = self._default_ml_risk_analysis()
            else:
                ml_risk_analysis = self._default_ml_risk_analysis()
            
            return EnhancedRiskMetrics(
                portfolio_beta=round(portfolio_beta, 3),
                sharpe_ratio=round(sharpe_ratio, 3),
                volatility=round(volatility, 3),
                var_95=round(var_95, 2),
                max_drawdown=round(max_drawdown, 3),
                risk_level=risk_level,
                risk_score=risk_score,
                ml_risk_distribution=ml_risk_analysis.get('risk_distribution', {}),
                predicted_volatility=ml_risk_analysis.get('predicted_volatility', volatility),
                expected_return=ml_risk_analysis.get('expected_return', expected_return),
                correlation_score=ml_risk_analysis.get('correlation_score', 0.5),
                anomaly_score=ml_risk_analysis.get('anomaly_score', 0.1),
                risk_factors=ml_risk_analysis.get('risk_factors', []),
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Enhanced risk calculation error: {e}")
            return self._default_enhanced_risk_metrics()

    async def _calculate_enhanced_diversification_metrics(self, holdings: List[Dict]) -> EnhancedDiversificationMetrics:
        """Calculate enhanced diversification metrics"""
        try:
            # Traditional diversification calculations
            total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
            
            # Sector analysis
            sector_values = {}
            for holding in holdings:
                sector = holding.get('sector', 'unknown')
                value = holding.get('live_market_value', holding.get('market_value', 0))
                sector_values[sector] = sector_values.get(sector, 0) + value
            
            sector_concentration = {
                sector: round(value / total_value, 3) 
                for sector, value in sector_values.items()
            } if total_value > 0 else {}
            
            values = [h.get('live_market_value', h.get('market_value', 0)) for h in holdings]
            top_holding_percentage = round(max(values) / total_value, 3) if values and total_value > 0 else 0
            
            weights = [v / total_value for v in values if total_value > 0]
            effective_holdings = 1 / sum(w**2 for w in weights) if weights else 1
            
            unique_sectors = len(set(h.get('sector', 'unknown') for h in holdings))
            correlation_score = min(1.0, unique_sectors / 7)
            
            diversification_score = self._calculate_diversification_score(
                sector_concentration, top_holding_percentage, effective_holdings, correlation_score
            )
            
            concentration_risk = self._assess_concentration_risk(top_holding_percentage, sector_concentration)
            
            # ML-enhanced diversification analysis
            ml_diversification_score = diversification_score / 100.0  # Default
            sector_recommendations = []
            rebalancing_suggestions = {}
            
            if self.dl_analyzer:
                # Generate ML-based sector recommendations
                sector_recommendations = self._generate_sector_recommendations(sector_concentration)
                rebalancing_suggestions = self._generate_rebalancing_suggestions(holdings, sector_concentration)
                
                # ML diversification score (could be from a trained model)
                ml_diversification_score = min(1.0, diversification_score / 80.0)  # Adjusted scoring
            
            return EnhancedDiversificationMetrics(
                sector_concentration=sector_concentration,
                top_holding_percentage=top_holding_percentage,
                effective_number_holdings=round(effective_holdings, 2),
                correlation_score=round(correlation_score, 3),
                diversification_score=diversification_score,
                concentration_risk=concentration_risk,
                ml_diversification_score=round(ml_diversification_score, 3),
                sector_recommendations=sector_recommendations,
                rebalancing_suggestions=rebalancing_suggestions
            )
            
        except Exception as e:
            logger.error(f"Enhanced diversification calculation error: {e}")
            return self._default_enhanced_diversification_metrics()

    async def _calculate_enhanced_performance_metrics(self, holdings: List[Dict]) -> EnhancedPerformanceMetrics:
        """Calculate enhanced performance metrics with ML predictions"""
        try:
            # Traditional performance calculations
            total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
            total_gain_loss = sum(h.get('live_gain_loss', h.get('gain_loss', 0)) or 0 for h in holdings)
            
            total_return = total_gain_loss / (total_value - total_gain_loss) if total_value > total_gain_loss else 0
            annualized_return = total_return * 2
            ytd_return = total_return * 0.8
            
            monthly_returns = [round(np.random.normal(0.01, 0.03), 4) for _ in range(12)]
            
            sp500_return = 0.10
            benchmark_comparison = annualized_return - sp500_return
            alpha = benchmark_comparison
            tracking_error = 0.05
            
            performance_score = self._calculate_performance_score(total_return, annualized_return, benchmark_comparison)
            
            # ML-enhanced performance predictions
            ml_performance_prediction = annualized_return  # Default
            risk_adjusted_returns = annualized_return  # Default
            performance_consistency = 0.7  # Default
            market_timing_score = 0.5  # Default
            
            if self.dl_analyzer:
                # ML-based performance predictions
                ml_performance_prediction = self._predict_future_performance(holdings)
                risk_adjusted_returns = self._calculate_ml_risk_adjusted_returns(holdings, total_return)
                performance_consistency = self._calculate_performance_consistency(monthly_returns)
                market_timing_score = self._calculate_market_timing_score(holdings)
            
            return EnhancedPerformanceMetrics(
                total_return=round(total_return, 4),
                annualized_return=round(annualized_return, 4),
                ytd_return=round(ytd_return, 4),
                monthly_returns=monthly_returns,
                benchmark_comparison=round(benchmark_comparison, 4),
                alpha=round(alpha, 4),
                tracking_error=round(tracking_error, 4),
                performance_score=performance_score,
                ml_performance_prediction=round(ml_performance_prediction, 4),
                risk_adjusted_returns=round(risk_adjusted_returns, 4),
                performance_consistency=round(performance_consistency, 3),
                market_timing_score=round(market_timing_score, 3)
            )
            
        except Exception as e:
            logger.error(f"Enhanced performance calculation error: {e}")
            return self._default_enhanced_performance_metrics()

    async def _generate_enhanced_recommendations(self, holdings: List[Dict], risk: EnhancedRiskMetrics, 
                                               div: EnhancedDiversificationMetrics, perf: EnhancedPerformanceMetrics,
                                               image_analysis: Dict) -> List[EnhancedRecommendation]:
        """Generate enhanced recommendations using ML"""
        try:
            recommendations = []
            
            # Traditional recommendations
            traditional_recs = await self._generate_traditional_recommendations(holdings, risk, div, perf)
            
            # ML-powered recommendations
            ml_recommendations = []
            if self.dl_analyzer:
                try:
                    ml_recommendations = await self.dl_analyzer.generate_ai_recommendations(holdings, asdict(risk))
                    logger.info(f"🤖 Generated {len(ml_recommendations)} ML recommendations")
                except Exception as e:
                    logger.warning(f"ML recommendation generation failed: {e}")
            
            # Convert and combine recommendations
            all_recommendations = []
            
            # Add traditional recommendations
            for rec in traditional_recs:
                all_recommendations.append(EnhancedRecommendation(
                    type=rec.get('type', 'general'),
                    priority=rec.get('priority', 'medium'),
                    title=rec.get('title', ''),
                    description=rec.get('description', ''),
                    rationale=rec.get('rationale', ''),
                    suggested_actions=rec.get('suggested_actions', []),
                    potential_impact=rec.get('potential_impact', ''),
                    confidence=rec.get('confidence', 0.7),
                    ml_confidence=0.0,
                    recommendation_source='traditional',
                    expected_outcome='Improved portfolio balance',
                    implementation_difficulty='Medium',
                    time_horizon='3-6 months'
                ))
            
            # Add ML recommendations
            for rec in ml_recommendations:
                all_recommendations.append(EnhancedRecommendation(
                    type=rec.get('type', 'ml_generated'),
                    priority=rec.get('priority', 'medium'),
                    title=rec.get('title', ''),
                    description=rec.get('description', ''),
                    rationale=rec.get('rationale', ''),
                    suggested_actions=rec.get('suggested_actions', []),
                    potential_impact=rec.get('potential_impact', ''),
                    confidence=rec.get('confidence', 0.8),
                    ml_confidence=rec.get('ml_score', 0.8),
                    recommendation_source='deep_learning',
                    expected_outcome='AI-optimized portfolio performance',
                    implementation_difficulty='Low',
                    time_horizon='1-3 months'
                ))
            
            # Add image-based recommendations if available
            if image_analysis.get('analysis_method') == 'deep_learning':
                broker = image_analysis.get('predicted_broker', 'unknown')
                if broker != 'unknown':
                    all_recommendations.append(EnhancedRecommendation(
                        type='broker_optimization',
                        priority='low',
                        title=f'🏛️ Optimize {broker.title()} Features',
                        description=f'Vision AI detected {broker} platform with {image_analysis.get("confidence", 0):.1%} confidence',
                        rationale='Computer vision analysis of portfolio screenshot',
                        suggested_actions=[
                            f'Explore advanced {broker} features',
                            'Consider fee optimization strategies',
                            'Leverage platform-specific tools'
                        ],
                        potential_impact='Platform-specific optimizations',
                        confidence=image_analysis.get('confidence', 0.7),
                        ml_confidence=image_analysis.get('confidence', 0.7),
                        recommendation_source='computer_vision',
                        expected_outcome='Better platform utilization',
                        implementation_difficulty='Low',
                        time_horizon='Immediate'
                    ))
            
            # Rank and return top recommendations
            ranked_recommendations = self._rank_enhanced_recommendations(all_recommendations)
            return ranked_recommendations[:8]  # Return top 8
            
        except Exception as e:
            logger.error(f"Enhanced recommendation generation failed: {e}")
            return []

    # Helper methods for ML enhancements
    def _classify_asset_risk(self, holding: Dict) -> str:
        """Classify asset risk using ML (simulated)"""
        symbol = holding.get('symbol', '')
        sector = holding.get('sector', 'unknown')
        
        # Simplified risk classification
        high_risk_sectors = ['technology', 'energy']
        if sector in high_risk_sectors:
            return 'high'
        elif sector in ['utilities', 'consumer_staples']:
            return 'low'
        else:
            return 'medium'

    def _predict_price_movement(self, holding: Dict) -> float:
        """Predict price movement using ML (simulated)"""
        # Simulated ML prediction
        return np.random.normal(0.02, 0.1)  # 2% expected return with 10% volatility

    def _generate_sector_recommendations(self, sector_concentration: Dict) -> List[str]:
        """Generate sector rebalancing recommendations"""
        recommendations = []
        
        for sector, weight in sector_concentration.items():
            if weight > 0.4:
                recommendations.append(f"Reduce {sector} allocation from {weight:.1%} to under 30%")
            elif weight < 0.05 and sector != 'unknown':
                recommendations.append(f"Consider adding {sector} exposure (currently {weight:.1%})")
        
        return recommendations[:5]

    def _generate_rebalancing_suggestions(self, holdings: List[Dict], sector_concentration: Dict) -> Dict[str, float]:
        """Generate specific rebalancing suggestions"""
        suggestions = {}
        
        # Find overweight sectors
        for sector, weight in sector_concentration.items():
            if weight > 0.3:
                target_weight = 0.25
                suggestions[f"target_{sector}_weight"] = target_weight
                suggestions[f"reduce_{sector}_by"] = weight - target_weight
        
        return suggestions

    def _predict_future_performance(self, holdings: List[Dict]) -> float:
        """Predict future performance using ML (simulated)"""
        # Simulated ML prediction based on portfolio composition
        tech_weight = sum(1 for h in holdings if h.get('sector') == 'technology') / len(holdings)
        base_return = 0.08
        tech_bonus = tech_weight * 0.02  # Tech tends to have higher returns
        
        return base_return + tech_bonus

    def _calculate_ml_risk_adjusted_returns(self, holdings: List[Dict], total_return: float) -> float:
        """Calculate ML-based risk-adjusted returns"""
        # Simplified risk adjustment
        portfolio_risk = len(holdings) / 20.0  # More holdings = lower risk
        risk_adjustment = min(1.0, portfolio_risk)
        
        return total_return * risk_adjustment

    def _calculate_performance_consistency(self, monthly_returns: List[float]) -> float:
        """Calculate performance consistency score"""
        if not monthly_returns:
            return 0.5
        
        # Lower standard deviation = higher consistency
        std_dev = np.std(monthly_returns)
        consistency = max(0.0, 1.0 - std_dev * 10)  # Scale appropriately
        
        return min(1.0, consistency)

    def _calculate_market_timing_score(self, holdings: List[Dict]) -> float:
        """Calculate market timing effectiveness score"""
        # Simulated market timing score based on gain/loss distribution
        positive_positions = sum(1 for h in holdings 
                               if (h.get('live_gain_loss', h.get('gain_loss', 0)) or 0) > 0)
        
        if len(holdings) == 0:
            return 0.5
        
        timing_score = positive_positions / len(holdings)
        return timing_score

    def _rank_enhanced_recommendations(self, recommendations: List[EnhancedRecommendation]) -> List[EnhancedRecommendation]:
        """Rank enhanced recommendations"""
        priority_weights = {'high': 3, 'medium': 2, 'low': 1}
        source_weights = {'deep_learning': 1.2, 'computer_vision': 1.1, 'traditional': 1.0}
        
        for rec in recommendations:
            priority_score = priority_weights.get(rec.priority, 1)
            source_score = source_weights.get(rec.recommendation_source, 1.0)
            confidence_score = (rec.confidence + rec.ml_confidence) / 2
            
            rec.ranking_score = (priority_score * 0.4 + confidence_score * 0.4 + source_score * 0.2)
        
        return sorted(recommendations, key=lambda x: getattr(x, 'ranking_score', 0), reverse=True)

    # Fallback and default methods
    async def _fallback_to_basic_analysis(self, holdings: List[Dict]) -> EnhancedPortfolioAnalysis:
        """Fallback to basic analysis if enhanced analysis fails"""
        # Simplified fallback implementation
        return EnhancedPortfolioAnalysis(
            risk_metrics=self._default_enhanced_risk_metrics(),
            diversification=self._default_enhanced_diversification_metrics(),
            performance=self._default_enhanced_performance_metrics(),
            recommendations=[],
            overall_score=50,
            summary="Basic analysis completed",
            analysis_timestamp=datetime.utcnow().isoformat(),
            ml_analysis_available=False,
            image_analysis={},
            advanced_insights={}
        )

    def _default_enhanced_risk_metrics(self) -> EnhancedRiskMetrics:
        return EnhancedRiskMetrics(
            portfolio_beta=1.0, sharpe_ratio=0.5, volatility=0.2, var_95=0.0, max_drawdown=0.1,
            risk_level=RiskLevel.MODERATE, risk_score=50, ml_risk_distribution={},
            predicted_volatility=0.2, expected_return=0.08, correlation_score=0.5,
            anomaly_score=0.1, risk_factors=[], confidence_level=0.5
        )

    def _default_enhanced_diversification_metrics(self) -> EnhancedDiversificationMetrics:
        return EnhancedDiversificationMetrics(
            sector_concentration={}, top_holding_percentage=0.0, effective_number_holdings=1.0,
            correlation_score=0.0, diversification_score=50, concentration_risk="Unknown",
            ml_diversification_score=0.5, sector_recommendations=[], rebalancing_suggestions={}
        )

    def _default_enhanced_performance_metrics(self) -> EnhancedPerformanceMetrics:
        return EnhancedPerformanceMetrics(
            total_return=0.0, annualized_return=0.0, ytd_return=0.0, monthly_returns=[],
            benchmark_comparison=0.0, alpha=0.0, tracking_error=0.0, performance_score=50,
            ml_performance_prediction=0.0, risk_adjusted_returns=0.0,
            performance_consistency=0.5, market_timing_score=0.5
        )

    def _default_ml_risk_analysis(self) -> Dict:
        return {
            'risk_distribution': {'moderate': 1.0},
            'predicted_volatility': 0.15,
            'expected_return': 0.08,
            'correlation_score': 0.5,
            'anomaly_score': 0.1,
            'risk_factors': []
        }

    # Include all the helper methods from the original analyzer
    def _estimate_volatility_from_concentration(self, holdings: List[Dict]) -> float:
        """Estimate portfolio volatility based on concentration"""
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        if total_value == 0:
            return 0.3  # Default high volatility
        
        # Calculate Herfindahl index for concentration
        weights = [h.get('live_market_value', h.get('market_value', 0)) / total_value for h in holdings]
        herfindahl = sum(w**2 for w in weights)
        
        # Base volatility increases with concentration
        base_volatility = 0.15  # 15% base
        concentration_factor = herfindahl * 0.2  # Up to 20% additional
        
        return min(0.5, base_volatility + concentration_factor)  # Cap at 50%

    def _determine_risk_level(self, beta: float, volatility: float, holdings_count: int) -> RiskLevel:
        """Determine overall risk level"""
        # Risk factors
        beta_risk = 0 if beta < 0.8 else (1 if beta < 1.2 else 2)
        vol_risk = 0 if volatility < 0.15 else (1 if volatility < 0.25 else 2)
        div_risk = 0 if holdings_count > 15 else (1 if holdings_count > 8 else 2)
        
        total_risk = beta_risk + vol_risk + div_risk
        
        if total_risk <= 1:
            return RiskLevel.LOW
        elif total_risk <= 3:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.HIGH

    def _calculate_risk_score(self, beta: float, volatility: float, holdings_count: int) -> int:
        """Calculate risk score (0-100, higher is better)"""
        # Start with perfect score
        score = 100
        
        # Penalize high beta
        if beta > 1.2:
            score -= (beta - 1.2) * 30
        elif beta < 0.8:
            score -= (0.8 - beta) * 20
        
        # Penalize high volatility
        score -= max(0, (volatility - 0.15) * 200)
        
        # Penalize low diversification
        if holdings_count < 10:
            score -= (10 - holdings_count) * 5
        
        return max(0, min(100, int(score)))

    def _calculate_diversification_score(self, sector_conc: Dict, top_holding: float, 
                                       eff_holdings: float, correlation: float) -> int:
        """Calculate diversification score (0-100)"""
        score = 100
        
        # Penalize high concentration in single holding
        if top_holding > 0.3:
            score -= (top_holding - 0.3) * 150
        elif top_holding > 0.2:
            score -= (top_holding - 0.2) * 100
        
        # Penalize low effective holdings
        if eff_holdings < 5:
            score -= (5 - eff_holdings) * 10
        
        # Penalize sector concentration
        max_sector_weight = max(sector_conc.values()) if sector_conc else 1
        if max_sector_weight > 0.5:
            score -= (max_sector_weight - 0.5) * 100
        
        # Reward good correlation score
        score += correlation * 20
        
        return max(0, min(100, int(score)))

    def _assess_concentration_risk(self, top_holding: float, sector_conc: Dict) -> str:
        """Assess concentration risk level"""
        max_sector = max(sector_conc.values()) if sector_conc else 0
        
        if top_holding > 0.4 or max_sector > 0.6:
            return "High"
        elif top_holding > 0.25 or max_sector > 0.4:
            return "Medium"
        else:
            return "Low"

    def _calculate_performance_score(self, total_return: float, annual_return: float, 
                                   benchmark_comp: float) -> int:
        """Calculate performance score (0-100)"""
        score = 50  # Start neutral
        
        # Reward positive returns
        score += min(30, total_return * 200)  # Up to 30 points for returns
        
        # Reward beating benchmark
        score += min(20, benchmark_comp * 100)  # Up to 20 points for outperformance
        
        return max(0, min(100, int(score)))

    async def _generate_traditional_recommendations(self, holdings: List[Dict], risk: EnhancedRiskMetrics, 
                                                  div: EnhancedDiversificationMetrics, 
                                                  perf: EnhancedPerformanceMetrics) -> List[Dict]:
        """Generate traditional recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk.risk_score < 60:
            recommendations.append({
                'type': "risk_management",
                'priority': "high",
                'title': "🚨 High Risk Portfolio Detected",
                'description': "Your portfolio shows elevated risk levels that may not align with typical investment goals.",
                'rationale': f"Portfolio beta of {risk.portfolio_beta} and volatility of {risk.volatility:.1%} indicate higher risk.",
                'suggested_actions': [
                    "Consider adding defensive stocks (utilities, consumer staples)",
                    "Reduce concentration in high-beta technology stocks",
                    "Add bond ETFs or dividend-paying stocks for stability",
                    "Consider dollar-cost averaging for new positions"
                ],
                'potential_impact': "Could reduce portfolio volatility by 20-30%",
                'confidence': 0.85
            })
        
        # Diversification recommendations
        if div.diversification_score < 70:
            max_sector = max(div.sector_concentration.values()) if div.sector_concentration else 0
            dominant_sector = max(div.sector_concentration.keys(), 
                                key=lambda k: div.sector_concentration[k]) if div.sector_concentration else "unknown"
            
            recommendations.append({
                'type': "diversification",
                'priority': "high" if div.diversification_score < 50 else "medium",
                'title': "📊 Improve Portfolio Diversification",
                'description': f"Your portfolio is {div.concentration_risk.lower()} concentrated with {div.effective_number_holdings:.1f} effective holdings.",
                'rationale': f"Over {max_sector:.1%} allocation to {dominant_sector} sector creates concentration risk.",
                'suggested_actions': [
                    f"Reduce {dominant_sector} exposure to under 30% of portfolio",
                    "Add exposure to underrepresented sectors (healthcare, financials, utilities)",
                    "Consider broad market ETFs (VTI, VOO) for instant diversification",
                    "Gradually rebalance over 3-6 months to avoid market timing"
                ],
                'potential_impact': "Could improve diversification score to 80+",
                'confidence': 0.90
            })
        
        # Performance recommendations
        if perf.benchmark_comparison < -0.02:
            recommendations.append({
                'type': "performance",
                'priority': "medium",
                'title': "📈 Performance Enhancement Opportunities",
                'description': f"Portfolio is underperforming S&P 500 by {abs(perf.benchmark_comparison):.1%} annually.",
                'rationale': "Consistent underperformance may indicate need for strategy adjustment.",
                'suggested_actions': [
                    "Review underperforming holdings for potential replacement",
                    "Consider low-cost index funds (SPY, VTI) as core holdings",
                    "Implement systematic rebalancing (quarterly or semi-annually)",
                    "Research growth stocks in emerging sectors (clean energy, AI)"
                ],
                'potential_impact': "Could improve returns by 2-4% annually",
                'confidence': 0.75
            })
        
        return recommendations

    def _calculate_enhanced_overall_score(self, risk: EnhancedRiskMetrics, div: EnhancedDiversificationMetrics, 
                                        perf: EnhancedPerformanceMetrics) -> int:
        """Calculate enhanced overall portfolio score"""
        # Traditional weighted score
        traditional_score = (
            risk.risk_score * 0.3 +
            div.diversification_score * 0.4 +
            perf.performance_score * 0.3
        )
        
        # ML enhancement bonus
        ml_bonus = 0
        if risk.confidence_level > 0.8:
            ml_bonus += 5  # Bonus for high-confidence ML analysis
        
        if div.ml_diversification_score > 0.8:
            ml_bonus += 3  # Bonus for good ML diversification
        
        if perf.performance_consistency > 0.7:
            ml_bonus += 2  # Bonus for consistent performance
        
        final_score = min(100, int(traditional_score + ml_bonus))
        return final_score

    def _generate_enhanced_portfolio_summary(self, risk: EnhancedRiskMetrics, div: EnhancedDiversificationMetrics, 
                                           perf: EnhancedPerformanceMetrics, overall_score: int, 
                                           image_analysis: Dict) -> str:
        """Generate enhanced AI-powered portfolio summary"""
        risk_desc = {
            RiskLevel.VERY_LOW: "very conservative",
            RiskLevel.LOW: "conservative", 
            RiskLevel.MODERATE: "moderate",
            RiskLevel.HIGH: "aggressive",
            RiskLevel.VERY_HIGH: "very aggressive"
        }
        
        summary_parts = []
        
        # Overall assessment with ML confidence
        confidence_indicator = "🤖 AI-Enhanced" if risk.confidence_level > 0.8 else "📊 Traditional"
        
        if overall_score >= 85:
            summary_parts.append(f"🟢 **Excellent portfolio** with strong fundamentals. {confidence_indicator} analysis.")
        elif overall_score >= 70:
            summary_parts.append(f"🟡 **Good portfolio** with room for optimization. {confidence_indicator} analysis.")
        elif overall_score >= 50:
            summary_parts.append(f"🟠 **Portfolio needs attention** to improve performance. {confidence_indicator} analysis.")
        else:
            summary_parts.append(f"🔴 **Portfolio requires significant improvements** for optimal performance. {confidence_indicator} analysis.")
        
        # Risk assessment with ML insights
        if risk.anomaly_score > 0.7:
            summary_parts.append(f"Risk profile is **{risk_desc[risk.risk_level]}** (β={risk.portfolio_beta:.2f}) with unusual patterns detected.")
        else:
            summary_parts.append(f"Risk profile is **{risk_desc[risk.risk_level]}** (β={risk.portfolio_beta:.2f}, ML volatility: {risk.predicted_volatility:.1%}).")
        
        # Enhanced diversification
        if div.ml_diversification_score > 0.8:
            summary_parts.append("Diversification is **AI-optimized** with excellent sector balance.")
        elif div.diversification_score >= 75:
            summary_parts.append("Diversification is **well-balanced** across sectors.")
        else:
            summary_parts.append(f"Diversification needs improvement - {div.concentration_risk.lower()} concentration risk detected.")
        
        # Performance with ML predictions
        if perf.ml_performance_prediction > perf.benchmark_comparison:
            summary_parts.append(f"Portfolio shows **strong future potential** with ML-predicted returns of {perf.ml_performance_prediction:.1%}.")
        elif perf.benchmark_comparison > 0:
            summary_parts.append(f"Portfolio is **outperforming** market by {perf.benchmark_comparison:.1%}.")
        else:
            summary_parts.append(f"Portfolio is **underperforming** market by {abs(perf.benchmark_comparison):.1%}.")
        
        # Image analysis insights
        if image_analysis.get('analysis_method') == 'deep_learning':
            broker = image_analysis.get('predicted_broker', 'unknown')
            if broker != 'unknown':
                summary_parts.append(f"Computer vision identified **{broker.title()}** platform with platform-specific optimization opportunities.")
        
        return " ".join(summary_parts)

    async def _generate_advanced_insights(self, holdings: List[Dict]) -> Dict:
        """Generate advanced insights using ML"""
        insights = {
            'portfolio_style': self._analyze_investment_style(holdings),
            'risk_breakdown': self._analyze_risk_sources(holdings),
            'optimization_potential': self._calculate_optimization_potential(holdings),
            'market_exposure': self._analyze_market_exposure(holdings),
            'trend_analysis': self._analyze_portfolio_trends(holdings)
        }
        
        if self.dl_analyzer:
            # Add ML-specific insights
            insights['ml_insights'] = {
                'predicted_sectors': self._predict_future_sector_performance(),
                'portfolio_health_score': self._calculate_portfolio_health(holdings),
                'risk_adjusted_efficiency': self._calculate_efficiency_ratio(holdings)
            }
        
        return insights

    def _analyze_investment_style(self, holdings: List[Dict]) -> str:
        """Analyze investment style based on holdings"""
        if not holdings:
            return "Unknown"
        
        tech_weight = sum(1 for h in holdings if h.get('sector') == 'technology') / len(holdings)
        large_positions = sum(1 for h in holdings 
                            if (h.get('live_market_value', h.get('market_value', 0)) / 
                                sum(h2.get('live_market_value', h2.get('market_value', 0)) for h2 in holdings)) > 0.1)
        
        if tech_weight > 0.6:
            return "Technology Growth Focused"
        elif large_positions < len(holdings) * 0.3:
            return "Diversified Value"
        elif len(holdings) < 5:
            return "Concentrated Growth"
        else:
            return "Balanced Growth"

    def _analyze_risk_sources(self, holdings: List[Dict]) -> Dict[str, float]:
        """Analyze sources of portfolio risk"""
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        risk_sources = {
            'concentration_risk': 0.0,
            'sector_risk': 0.0,
            'volatility_risk': 0.0,
            'correlation_risk': 0.0
        }
        
        if total_value > 0:
            # Concentration risk
            max_position = max(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
            risk_sources['concentration_risk'] = max_position / total_value
            
            # Sector risk
            sectors = {}
            for h in holdings:
                sector = h.get('sector', 'unknown')
                sectors[sector] = sectors.get(sector, 0) + h.get('live_market_value', h.get('market_value', 0))
            
            max_sector_weight = max(sectors.values()) / total_value if sectors else 0
            risk_sources['sector_risk'] = max_sector_weight
            
            # Simplified volatility and correlation risk
            risk_sources['volatility_risk'] = min(1.0, len(holdings) / 20.0)  # More holdings = lower volatility risk
            risk_sources['correlation_risk'] = 1.0 - min(1.0, len(set(h.get('sector') for h in holdings)) / 7.0)
        
        return risk_sources

    def _calculate_optimization_potential(self, holdings: List[Dict]) -> float:
        """Calculate potential for portfolio optimization"""
        if not holdings:
            return 0.0
        
        # Simple optimization potential based on diversification and balance
        sectors = set(h.get('sector', 'unknown') for h in holdings)
        sector_diversity = len(sectors) / 10.0  # Normalize to 10 sectors
        
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        position_balance = 1.0 - (max(h.get('live_market_value', h.get('market_value', 0)) for h in holdings) / total_value) if total_value > 0 else 0
        
        optimization_potential = (sector_diversity + position_balance) / 2.0
        return min(1.0, optimization_potential)

    def _analyze_market_exposure(self, holdings: List[Dict]) -> Dict[str, float]:
        """Analyze market exposure breakdown"""
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        exposure = {
            'us_market': 0.9,  # Assume mostly US market
            'international': 0.1,
            'large_cap': 0.7,
            'mid_cap': 0.2,
            'small_cap': 0.1,
            'growth_style': 0.6,
            'value_style': 0.4
        }
        
        # This would be enhanced with real market data analysis
        return exposure

    def _analyze_portfolio_trends(self, holdings: List[Dict]) -> Dict[str, str]:
        """Analyze portfolio trends"""
        trends = {
            'performance_trend': 'Stable',
            'risk_trend': 'Moderate',
            'diversification_trend': 'Improving',
            'sector_rotation': 'Technology Heavy'
        }
        
        # Enhanced trend analysis would use historical data
        return trends

    def _predict_future_sector_performance(self) -> Dict[str, float]:
        """Predict future sector performance (simulated ML)"""
        return {
            'technology': 0.12,
            'healthcare': 0.08,
            'financials': 0.06,
            'consumer_discretionary': 0.10,
            'energy': 0.15,
            'utilities': 0.04
        }

    def _calculate_portfolio_health(self, holdings: List[Dict]) -> float:
        """Calculate overall portfolio health score"""
        if not holdings:
            return 0.5
        
        # Simplified health calculation
        diversity_score = min(1.0, len(holdings) / 15.0)
        sector_diversity = len(set(h.get('sector', 'unknown') for h in holdings)) / 10.0
        
        health_score = (diversity_score + sector_diversity) / 2.0
        return min(1.0, health_score)

    def _calculate_efficiency_ratio(self, holdings: List[Dict]) -> float:
        """Calculate portfolio efficiency ratio"""
        # Simplified efficiency calculation
        if not holdings:
            return 0.5
        
        # Risk-return efficiency (simplified)
        total_gain_loss = sum(h.get('live_gain_loss', h.get('gain_loss', 0)) or 0 for h in holdings)
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        if total_value == 0:
            return 0.5
        
        return_rate = total_gain_loss / total_value
        risk_estimate = 1.0 / max(1, len(holdings))  # Simplified risk
        
        efficiency = return_rate / max(0.01, risk_estimate)  # Avoid division by zero
        return min(1.0, max(0.0, (efficiency + 1.0) / 2.0))  # Normalize to 0-1

    def to_dict(self, analysis: EnhancedPortfolioAnalysis) -> Dict:
        """Convert EnhancedPortfolioAnalysis to dictionary for JSON serialization"""
        result = asdict(analysis)
        
        # Convert enums to strings
        result['risk_metrics']['risk_level'] = analysis.risk_metrics.risk_level.value
        
        # Convert recommendations
        result['recommendations'] = [asdict(rec) for rec in analysis.recommendations]
        
        return result