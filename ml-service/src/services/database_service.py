import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Database service for storing and retrieving portfolio analysis data
    """
    
    def __init__(self):
        # Database configuration
        self.db_url = os.getenv('DATABASE_URL', 'postgresql://portfolio_user:portfolio_pass@postgres:5432/portfolio_analyzer')
        self.redis_url = os.getenv('REDIS_URL', 'redis://:redis_pass@redis:6379')
        
        # Convert to async URL
        self.async_db_url = self.db_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        self.engine = None
        self.redis_client = None
        self.session_factory = None
        
        # Define tables
        self.metadata = MetaData()
        self._define_tables()
    
    def _define_tables(self):
        """Define database tables"""
        
        # Portfolio analyses table
        self.portfolio_analyses = Table(
            'portfolio_analyses',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(100)),
            Column('analysis_data', JSON),
            Column('confidence_score', Float),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # Portfolio holdings table
        self.portfolio_holdings = Table(
            'portfolio_holdings',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(100)),
            Column('symbol', String(20)),
            Column('shares', Float),
            Column('market_value', Float),
            Column('gain_loss', Float),
            Column('return_pct', Float),
            Column('sector', String(100)),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # Recommendations table
        self.recommendations = Table(
            'recommendations',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(100)),
            Column('recommendation_type', String(50)),
            Column('priority', String(20)),
            Column('title', String(200)),
            Column('description', Text),
            Column('actions', JSON),
            Column('confidence', Float),
            Column('status', String(20), default='active'),
            Column('created_at', DateTime, default=datetime.utcnow),
            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        )
        
        # Risk analyses table
        self.risk_analyses = Table(
            'risk_analyses',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(100)),
            Column('risk_level', String(20)),
            Column('risk_score', Float),
            Column('risk_factors', JSON),
            Column('risk_distribution', JSON),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Performance tracking table
        self.performance_tracking = Table(
            'performance_tracking',
            self.metadata,
            Column('id', Integer, primary_key=True),
            Column('user_id', String(100)),
            Column('portfolio_value', Float),
            Column('total_return', Float),
            Column('return_pct', Float),
            Column('tracked_at', DateTime, default=datetime.utcnow)
        )
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            logger.info("🗄️ Initializing database service...")
            
            # Create async engine
            self.engine = create_async_engine(
                self.async_db_url,
                echo=False,
                pool_pre_ping=True
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(self.metadata.create_all)
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            logger.info("✅ Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
        if self.redis_client:
            await self.redis_client.close()
    
    async def save_analysis(self, analysis_data: Dict):
        """Save portfolio analysis to database"""
        try:
            async with self.session_factory() as session:
                # Extract user_id (for now, use a default)
                user_id = analysis_data.get('user_id', 'default_user')
                
                # Save main analysis
                insert_query = self.portfolio_analyses.insert().values(
                    user_id=user_id,
                    analysis_data=analysis_data,
                    confidence_score=analysis_data.get('confidence_score', 0.85)
                )
                await session.execute(insert_query)
                
                # Save holdings
                holdings = analysis_data.get('portfolio_data', {}).get('holdings', [])
                for holding in holdings:
                    holding_query = self.portfolio_holdings.insert().values(
                        user_id=user_id,
                        symbol=holding.get('symbol', ''),
                        shares=holding.get('shares', 0),
                        market_value=holding.get('market_value', 0),
                        gain_loss=holding.get('gain_loss', 0),
                        return_pct=holding.get('return_pct', 0),
                        sector=holding.get('sector', 'unknown')
                    )
                    await session.execute(holding_query)
                
                # Save recommendations
                recommendations = analysis_data.get('recommendations', [])
                for rec in recommendations:
                    rec_query = self.recommendations.insert().values(
                        user_id=user_id,
                        recommendation_type=rec.get('type', 'general'),
                        priority=rec.get('priority', 'medium'),
                        title=rec.get('title', ''),
                        description=rec.get('description', ''),
                        actions=rec.get('actions', []),
                        confidence=rec.get('confidence', 0.8)
                    )
                    await session.execute(rec_query)
                
                # Save risk analysis
                risk_analysis = analysis_data.get('risk_analysis', {})
                if risk_analysis:
                    risk_query = self.risk_analyses.insert().values(
                        user_id=user_id,
                        risk_level=risk_analysis.get('ai_risk_prediction', {}).get('predicted_risk_level', 'moderate'),
                        risk_score=risk_analysis.get('overall_risk_score', 0.5),
                        risk_factors=risk_analysis.get('risk_factors', []),
                        risk_distribution=risk_analysis.get('ai_risk_prediction', {}).get('risk_distribution', {})
                    )
                    await session.execute(risk_query)
                
                await session.commit()
                logger.info("✅ Analysis saved to database")
                
        except Exception as e:
            logger.error(f"❌ Failed to save analysis: {e}")
            await session.rollback()
            raise
    
    async def get_user_portfolio(self, user_id: str) -> Optional[Dict]:
        """Get user's latest portfolio data"""
        try:
            async with self.session_factory() as session:
                # Get latest analysis
                analysis_query = f"""
                    SELECT analysis_data, created_at 
                    FROM portfolio_analyses 
                    WHERE user_id = '{user_id}' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """
                result = await session.execute(analysis_query)
                row = result.fetchone()
                
                if row:
                    return {
                        'analysis_data': row[0],
                        'created_at': row[1].isoformat()
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get user portfolio: {e}")
            return None
    
    async def get_portfolio_performance(self, user_id: str, days: int = 30) -> Dict:
        """Get portfolio performance data"""
        try:
            async with self.session_factory() as session:
                # Get performance data for specified period
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                performance_query = f"""
                    SELECT portfolio_value, total_return, return_pct, tracked_at
                    FROM performance_tracking 
                    WHERE user_id = '{user_id}' 
                    AND tracked_at >= '{cutoff_date}'
                    ORDER BY tracked_at DESC
                """
                result = await session.execute(performance_query)
                rows = result.fetchall()
                
                performance_data = []
                for row in rows:
                    performance_data.append({
                        'portfolio_value': row[0],
                        'total_return': row[1],
                        'return_pct': row[2],
                        'tracked_at': row[3].isoformat()
                    })
                
                return {
                    'performance_data': performance_data,
                    'period_days': days,
                    'data_points': len(performance_data)
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio performance: {e}")
            return {'error': str(e)}
    
    async def cache_market_data(self, symbol: str, data: Dict, ttl: int = 300):
        """Cache market data in Redis"""
        try:
            key = f"market_data:{symbol}"
            await self.redis_client.setex(key, ttl, json.dumps(data))
            logger.debug(f"✅ Cached market data for {symbol}")
        except Exception as e:
            logger.error(f"❌ Failed to cache market data: {e}")
    
    async def get_cached_market_data(self, symbol: str) -> Optional[Dict]:
        """Get cached market data from Redis"""
        try:
            key = f"market_data:{symbol}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached market data: {e}")
            return None
    
    async def cache_analysis_result(self, user_id: str, analysis: Dict, ttl: int = 1800):
        """Cache analysis result"""
        try:
            key = f"analysis:{user_id}"
            await self.redis_client.setex(key, ttl, json.dumps(analysis))
            logger.debug(f"✅ Cached analysis for {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to cache analysis: {e}")
    
    async def get_cached_analysis(self, user_id: str) -> Optional[Dict]:
        """Get cached analysis result"""
        try:
            key = f"analysis:{user_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get cached analysis: {e}")
            return None
    
    async def save_recommendation_feedback(self, user_id: str, recommendation_id: int, feedback: str):
        """Save user feedback on recommendations"""
        try:
            async with self.session_factory() as session:
                # Update recommendation with feedback
                update_query = f"""
                    UPDATE recommendations 
                    SET status = '{feedback}', updated_at = '{datetime.utcnow()}'
                    WHERE id = {recommendation_id} AND user_id = '{user_id}'
                """
                await session.execute(update_query)
                await session.commit()
                logger.info(f"✅ Saved feedback for recommendation {recommendation_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save recommendation feedback: {e}")
            raise
    
    async def get_user_recommendations(self, user_id: str, status: str = 'active') -> List[Dict]:
        """Get user's active recommendations"""
        try:
            async with self.session_factory() as session:
                rec_query = f"""
                    SELECT id, recommendation_type, priority, title, description, actions, confidence, created_at
                    FROM recommendations 
                    WHERE user_id = '{user_id}' AND status = '{status}'
                    ORDER BY created_at DESC
                """
                result = await session.execute(rec_query)
                rows = result.fetchall()
                
                recommendations = []
                for row in rows:
                    recommendations.append({
                        'id': row[0],
                        'type': row[1],
                        'priority': row[2],
                        'title': row[3],
                        'description': row[4],
                        'actions': row[5],
                        'confidence': row[6],
                        'created_at': row[7].isoformat()
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Failed to get user recommendations: {e}")
            return []
    
    async def track_portfolio_performance(self, user_id: str, portfolio_value: float, total_return: float, return_pct: float):
        """Track portfolio performance over time"""
        try:
            async with self.session_factory() as session:
                performance_query = self.performance_tracking.insert().values(
                    user_id=user_id,
                    portfolio_value=portfolio_value,
                    total_return=total_return,
                    return_pct=return_pct
                )
                await session.execute(performance_query)
                await session.commit()
                logger.debug(f"✅ Tracked performance for {user_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to track portfolio performance: {e}")
            raise
    
    async def get_risk_history(self, user_id: str, days: int = 30) -> List[Dict]:
        """Get risk analysis history"""
        try:
            async with self.session_factory() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                risk_query = f"""
                    SELECT risk_level, risk_score, risk_factors, created_at
                    FROM risk_analyses 
                    WHERE user_id = '{user_id}' 
                    AND created_at >= '{cutoff_date}'
                    ORDER BY created_at DESC
                """
                result = await session.execute(risk_query)
                rows = result.fetchall()
                
                risk_history = []
                for row in rows:
                    risk_history.append({
                        'risk_level': row[0],
                        'risk_score': row[1],
                        'risk_factors': row[2],
                        'created_at': row[3].isoformat()
                    })
                
                return risk_history
                
        except Exception as e:
            logger.error(f"❌ Failed to get risk history: {e}")
            return []