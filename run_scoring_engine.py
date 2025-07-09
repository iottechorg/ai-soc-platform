#!/usr/bin/env python3
# run_enhanced_scoring_engine.py - FIXED VERSION with enhanced alert generation

import sys
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import deque

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import the enhanced scoring engine
    from services.scoring_engine import FixedAdaptiveScoringEngine as ScoringEngine
    from config.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
    from shared.database import ClickHouseClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class EnhancedScoringEngineRunner:
    """Enhanced Scoring Engine runner with improved debugging and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # Initialize enhanced scoring engine with improved config
        scoring_config = self.config.get_section('scoring')
        
        # Enhanced configuration with better defaults
        enhanced_config = {
            **scoring_config,
            'target_alert_rate': 0.05,  # 5% target
            'aggregation_method': 'adaptive_weighted',
            'model_weights': {
                'isolation_forest': 0.4,
                'clustering': 0.3,
                'forbidden_ratio': 0.3
            }
        }
        
        self.scoring_engine = ScoringEngine(enhanced_config)
        
        # Initialize database with proper error handling
        try:
            self.db = ClickHouseClient(
                host=self.config.get('clickhouse.host'),
                port=9000,  # Use native port
                user=self.config.get('clickhouse.user'),
                password=self.config.get('clickhouse.password'),
                database=self.config.get('clickhouse.database')
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.db = None
        
        # Initialize Kafka
        self.kafka_client = KafkaClient(self.config.get('kafka.broker'))
        
        # Processing state with enhanced tracking
        self.running = False
        self.anomaly_buffer = deque()
        self.entity_buffers = {}
        self.scores_processed = 0
        self.alerts_generated = 0
        self.processing_stats = {
            'last_reset': datetime.now(),
            'scores_per_minute': deque(maxlen=60),
            'alerts_per_minute': deque(maxlen=60),
            'processing_errors': 0
        }
    
    def start(self):
        """Start enhanced scoring engine microservice"""
        self.logger.info("🚀 Starting ENHANCED Scoring Engine Microservice...")
        
        # Connect to database
        if self.db:
            try:
                self.db.connect()
                if self.db.is_connected():
                    self.logger.info("✅ Connected to ClickHouse")
                else:
                    self.logger.warning("⚠️ ClickHouse connection failed - continuing without database")
                    self.db = None
            except Exception as e:
                self.logger.warning(f"⚠️ Database connection error: {e} - continuing without database")
                self.db = None
        
        # Start Kafka consumer for anomaly scores
        try:
            anomaly_scores_topic = self.config.get('kafka.topics.anomaly_scores', 'anomaly-scores')
            self.kafka_client.create_consumer(
                topics=[anomaly_scores_topic],
                group_id="enhanced-scoring-engine-consumer",
                message_handler=self._process_anomaly_score
            )
            self.logger.info(f"✅ Started Kafka consumer for {anomaly_scores_topic}")
        except Exception as e:
            self.logger.error(f"❌ Kafka consumer failed: {e}")
            return False
        
        self.running = True
        
        # Start enhanced processing threads
        self._start_processing_threads()
        
        self.logger.info("🎯 ENHANCED Scoring Engine Microservice is running!")
        self.logger.info("🔧 Features: Improved thresholds, better alert logic, enhanced debugging")
        return True
    
    def _start_processing_threads(self):
        """Start all processing threads"""
        try:
            # Main scoring loop
            scoring_thread = threading.Thread(target=self._enhanced_scoring_loop, daemon=True)
            scoring_thread.start()
            
            # Enhanced status reporter
            status_thread = threading.Thread(target=self._enhanced_status_reporter, daemon=True)
            status_thread.start()
            
            # Performance monitor
            perf_thread = threading.Thread(target=self._performance_monitor, daemon=True)
            perf_thread.start()
            
            self.logger.info("✅ Started all processing threads")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start processing threads: {e}")
    
    def _process_anomaly_score(self, score_data: Dict[str, Any]):
        """Process incoming anomaly score with enhanced error handling"""
        try:
            # Validate score data
            if not score_data or not isinstance(score_data, dict):
                self.logger.warning("Received invalid score data")
                return
            
            entity_id = score_data.get('entity_id')
            score = score_data.get('score', 0.0)
            model_name = score_data.get('model_name')
            
            if not entity_id or not model_name:
                self.logger.warning(f"Incomplete score data: {score_data}")
                return
            
            self.anomaly_buffer.append(score_data)
            self.scores_processed += 1
            
            # Track scores per minute
            self.processing_stats['scores_per_minute'].append(datetime.now())
            
            self.logger.debug(
                f"📥 Received score: {model_name} → {entity_id} = {score:.3f}"
            )
            
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.error(f"❌ Error processing anomaly score: {e}")
    
    def _enhanced_scoring_loop(self):
        """Enhanced main scoring loop with better aggregation logic"""
        while self.running:
            try:
                # Process buffered anomaly scores
                processed_count = 0
                
                while self.anomaly_buffer and processed_count < 50:  # Batch processing
                    score_data = self.anomaly_buffer.popleft()
                    entity_id = score_data.get('entity_id', 'unknown')
                    
                    # Enhanced entity grouping
                    if entity_id not in self.entity_buffers:
                        self.entity_buffers[entity_id] = {
                            'scores': {},
                            'last_update': datetime.now(),
                            'score_count': 0
                        }
                    
                    # Add score to entity buffer
                    model_name = score_data.get('model_name')
                    self.entity_buffers[entity_id]['scores'][model_name] = {
                        'score': score_data.get('score', 0.0),
                        'is_anomaly': score_data.get('is_anomaly', False),
                        'timestamp': score_data.get('timestamp', datetime.now()),
                        'features': score_data.get('features', '{}')
                    }
                    self.entity_buffers[entity_id]['last_update'] = datetime.now()
                    self.entity_buffers[entity_id]['score_count'] += 1
                    
                    processed_count += 1
                
                # Process entities ready for aggregation
                self._process_entity_buffers()
                
                # Clean up old entity buffers
                self._cleanup_entity_buffers()
                
                time.sleep(0.5)  # Process twice per second
                
            except Exception as e:
                self.processing_stats['processing_errors'] += 1
                self.logger.error(f"❌ Error in enhanced scoring loop: {e}")
                time.sleep(5)
    
    def _process_entity_buffers(self):
        """Process entity buffers with enhanced logic"""
        entities_to_remove = []
        
        for entity_id, buffer_data in self.entity_buffers.items():
            scores = buffer_data['scores']
            last_update = buffer_data['last_update']
            score_count = buffer_data['score_count']
            
            # Enhanced readiness criteria
            time_since_update = (datetime.now() - last_update).seconds
            should_process = False
            
            # Process if we have multiple model scores
            if len(scores) >= 2:
                should_process = True
                reason = f"multiple_models({len(scores)})"
                
            # Process if we have scores and haven't updated recently
            elif len(scores) >= 1 and time_since_update > 10:
                should_process = True
                reason = f"timeout({time_since_update}s)"
                
            # Process if buffer is getting full
            elif score_count >= 5:
                should_process = True
                reason = f"buffer_full({score_count})"
            
            if should_process:
                try:
                    # Enhanced aggregation and alert generation
                    scoring_result = self.scoring_engine.aggregate_scores(entity_id, scores)
                    
                    # Enhanced alert generation with debugging
                    if self.scoring_engine.should_generate_alert(scoring_result):
                        self._publish_enhanced_alert_to_kafka(scoring_result, reason)
                    
                    # Store scoring result
                    self._store_scoring_result(scoring_result)
                    
                    entities_to_remove.append(entity_id)
                    
                    self.logger.debug(
                        f"✅ Processed {entity_id}: score={scoring_result.get('aggregated_score', 0):.3f}, "
                        f"risk={scoring_result.get('risk_level', 'unknown')}, reason={reason}"
                    )
                    
                except Exception as e:
                    self.processing_stats['processing_errors'] += 1
                    self.logger.error(f"❌ Error processing entity {entity_id}: {e}")
                    entities_to_remove.append(entity_id)  # Remove problematic entities
        
        # Clean up processed entities
        for entity_id in entities_to_remove:
            if entity_id in self.entity_buffers:
                del self.entity_buffers[entity_id]
    
    def _cleanup_entity_buffers(self):
        """Clean up old entity buffers"""
        cutoff_time = datetime.now().timestamp() - 600  # 10 minutes
        old_entities = [
            entity_id for entity_id, data in self.entity_buffers.items()
            if data['last_update'].timestamp() < cutoff_time
        ]
        
        for entity_id in old_entities:
            self.logger.debug(f"🧹 Cleaning up old entity buffer: {entity_id}")
            del self.entity_buffers[entity_id]
    
    def _publish_enhanced_alert_to_kafka(self, scoring_result: Dict[str, Any], processing_reason: str):
        """Publish enhanced alert to Kafka with additional metadata"""
        try:
            # Create enhanced alert data
            alert_data = {
                'alert_id': f"alert_{int(datetime.now().timestamp())}_{scoring_result['entity_id']}",
                'timestamp': scoring_result['timestamp'],
                'severity': scoring_result['severity'],
                'entity_id': scoring_result['entity_id'],
                'message': f"Enhanced threat detection: {scoring_result['entity_id']} - Score: {scoring_result['aggregated_score']:.3f}",
                'aggregated_score': scoring_result['aggregated_score'],
                'confidence': scoring_result.get('confidence', 0.0),
                'risk_level': scoring_result.get('risk_level', 'unknown'),
                'contributing_models': scoring_result['contributing_models'],
                'model_scores': scoring_result.get('model_scores', {}),
                'tags': scoring_result.get('tags', []),
                'actions': scoring_result.get('actions', []),
                'score_trend': scoring_result.get('score_trend', 'unknown'),
                'status': 'open',
                
                # Enhanced metadata
                'processing_reason': processing_reason,
                'adaptive_thresholds': scoring_result.get('adaptive_thresholds', {}),
                'entity_metadata': scoring_result.get('entity_metadata', {}),
                'generation_timestamp': datetime.now(),
                'scoring_engine_version': 'enhanced_v1.0'
            }
            
            # Publish to Kafka
            alerts_topic = self.config.get('kafka.topics.alerts', 'alerts')
            self.kafka_client.send_message(
                topic=alerts_topic,
                message=alert_data,
                key=alert_data['entity_id']
            )
            
            # Store alert in database if available
            if self.db and self.db.is_connected():
                try:
                    self.db.insert_alerts([alert_data])
                except Exception as e:
                    self.logger.warning(f"Failed to store alert in database: {e}")
            
            self.alerts_generated += 1
            
            # Track alerts per minute
            self.processing_stats['alerts_per_minute'].append(datetime.now())
            
            self.logger.info(
                f"🚨 ENHANCED ALERT: {alert_data['severity'].upper()} for {alert_data['entity_id']} "
                f"(score: {alert_data['aggregated_score']:.3f}, "
                f"confidence: {alert_data['confidence']:.3f}, "
                f"risk: {alert_data['risk_level']}, "
                f"reason: {processing_reason}) → Kafka"
            )
            
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.error(f"❌ Error publishing enhanced alert: {e}")
    
    def _store_scoring_result(self, scoring_result: Dict[str, Any]):
        """Store scoring result for analytics"""
        try:
            # Could implement detailed scoring result storage here
            self.logger.debug(
                f"📊 Scoring result: {scoring_result['entity_id']} = "
                f"{scoring_result['aggregated_score']:.3f} ({scoring_result.get('risk_level', 'unknown')})"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error storing scoring result: {e}")
    
    def _enhanced_status_reporter(self):
        """Enhanced status reporter with detailed metrics"""
        while self.running:
            try:
                time.sleep(60)  # Report every minute
                
                # Calculate rates
                now = datetime.now()
                minute_ago = now - timedelta(minutes=1)
                
                recent_scores = len([t for t in self.processing_stats['scores_per_minute'] 
                                   if t >= minute_ago])
                recent_alerts = len([t for t in self.processing_stats['alerts_per_minute'] 
                                   if t >= minute_ago])
                
                buffer_size = len(self.anomaly_buffer)
                entity_count = len(self.entity_buffers)
                
                # Get scoring engine metrics
                perf_metrics = self.scoring_engine.get_performance_metrics()
                
                self.logger.info(
                    f"📈 ENHANCED STATUS: "
                    f"Processed: {self.scores_processed} scores, "
                    f"Generated: {self.alerts_generated} alerts, "
                    f"Rate: {recent_scores}/min scores, {recent_alerts}/min alerts"
                )
                
                self.logger.info(
                    f"📊 Buffers: {buffer_size} pending scores, "
                    f"{entity_count} active entities, "
                    f"Errors: {self.processing_stats['processing_errors']}"
                )
                
                # Report scoring engine performance
                if perf_metrics:
                    alert_rate = perf_metrics.get('alert_rate', 0)
                    thresholds = perf_metrics.get('current_thresholds', {})
                    
                    self.logger.info(
                        f"🎯 Engine Metrics: Alert rate: {alert_rate:.3f}, "
                        f"Entities tracked: {perf_metrics.get('entities_tracked', 0)}"
                    )
                    
                    if 'score_stats' in perf_metrics:
                        stats = perf_metrics['score_stats']
                        self.logger.info(
                            f"📊 Score Stats: "
                            f"mean={stats['mean']:.3f}, "
                            f"median={stats['median']:.3f}, "
                            f"range=[{stats['min']:.3f}, {stats['max']:.3f}]"
                        )
                    
                    # Report current thresholds
                    self.logger.info(f"🎯 Thresholds: {thresholds}")
                
            except Exception as e:
                self.logger.error(f"❌ Status reporter error: {e}")
    
    def _performance_monitor(self):
        """Monitor performance and log detailed statistics"""
        while self.running:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Get detailed performance metrics
                perf_metrics = self.scoring_engine.get_performance_metrics()
                
                if perf_metrics:
                    self.logger.info("🔍 DETAILED PERFORMANCE METRICS:")
                    self.logger.info(f"   Total scored: {perf_metrics.get('total_scored', 0)}")
                    self.logger.info(f"   Alerts generated: {perf_metrics.get('alerts_generated', 0)}")
                    self.logger.info(f"   Alert rate: {perf_metrics.get('alert_rate', 0):.3f}")
                    
                    # Alert severity breakdown
                    severity_breakdown = perf_metrics.get('alerts_by_severity', {})
                    if severity_breakdown:
                        self.logger.info(f"   Severity breakdown: {severity_breakdown}")
                    
                    # Alert generation reasons
                    reasons = perf_metrics.get('alert_generation_reasons', {})
                    if reasons:
                        self.logger.info(f"   Alert reasons: {reasons}")
                    
                    # Model weights
                    weights = perf_metrics.get('model_weights', {})
                    if weights:
                        self.logger.info(f"   Model weights: {weights}")
                
            except Exception as e:
                self.logger.error(f"❌ Performance monitor error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        try:
            engine_metrics = self.scoring_engine.get_performance_metrics()
            
            return {
                'running': self.running,
                'scores_processed': self.scores_processed,
                'alerts_generated': self.alerts_generated,
                'buffer_size': len(self.anomaly_buffer),
                'active_entities': len(self.entity_buffers),
                'processing_errors': self.processing_stats['processing_errors'],
                'engine_metrics': engine_metrics,
                'database_connected': self.db.is_connected() if self.db else False
            }
        except Exception as e:
            return {'error': str(e), 'running': self.running}
    
    def stop(self):
        """Stop the enhanced scoring engine"""
        self.running = False
        if self.scoring_engine:
            self.scoring_engine.stop()
        self.logger.info("Enhanced scoring engine stopped")

def main():
    """Main entry point for enhanced scoring engine microservice"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting ENHANCED Scoring Engine Microservice...")
        logger.info("🔧 Features: Improved alert generation, better thresholds, enhanced debugging")
        
        # Initialize enhanced scoring engine
        scoring_engine = EnhancedScoringEngineRunner()
        
        if scoring_engine.start():
            logger.info("✅ ENHANCED Scoring Engine Microservice started successfully")
            logger.info("📨 Publishing enhanced alerts to Kafka with better logic")
            
            # Keep running with enhanced monitoring
            while True:
                time.sleep(60)
                
                # Periodic status report
                try:
                    status = scoring_engine.get_status()
                    scores_processed = status.get('scores_processed', 0)
                    alerts_generated = status.get('alerts_generated', 0)
                    
                    if scores_processed > 0:
                        alert_rate = alerts_generated / scores_processed
                        logger.info(
                            f"💫 Status: {scores_processed} scores → {alerts_generated} alerts "
                            f"(rate: {alert_rate:.3f})"
                        )
                    
                    # Log any processing errors
                    errors = status.get('processing_errors', 0)
                    if errors > 0:
                        logger.warning(f"⚠️ Processing errors: {errors}")
                        
                except Exception as e:
                    logger.error(f"❌ Error getting status: {e}")
        else:
            logger.error("❌ Failed to start enhanced scoring engine microservice")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down enhanced scoring engine...")
        if 'scoring_engine' in locals():
            scoring_engine.stop()
    except Exception as e:
        logger.error(f"💥 Enhanced scoring engine error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()