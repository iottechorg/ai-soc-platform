#!/usr/bin/env python3
# data_storage_service.py - Dedicated service for storing data in ClickHouse

import sys
import time
import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import psutil
from flask import Flask, jsonify
import signal

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from shared.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
    from shared.database import ClickHouseClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class DataStorageService:
    """Dedicated service for storing raw logs and anomaly scores in ClickHouse"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # Initialize ClickHouse client
        self.db = ClickHouseClient(
            host=self.config.get('clickhouse.host'),
            port=9000,
            user=self.config.get('clickhouse.user'),
            password=self.config.get('clickhouse.password'),
            database=self.config.get('clickhouse.database')
        )
        
        # Initialize Kafka client
        self.kafka_client = KafkaClient(self.config.get('kafka.broker'))
        
        # Service state
        self.running = False
        self.start_time = None
        
        # Processing statistics
        self.stats = {
            'raw_logs_processed': 0,
            'raw_logs_stored': 0,
            'anomaly_scores_processed': 0,
            'anomaly_scores_stored': 0,
            'alerts_processed': 0,
            'alerts_stored': 0,
            'batch_processing_errors': 0,
            'database_errors': 0,
            'last_storage_time': None
        }
        
        # Processing buffers for batch insertion
        self.raw_logs_buffer = deque()
        self.anomaly_scores_buffer = deque()
        self.alerts_buffer = deque()
        
        # Configuration
        self.batch_size = self.config.get('data_storage.batch_size', 50)
        self.batch_timeout = self.config.get('data_storage.batch_timeout_seconds', 5)
        self.max_retries = self.config.get('data_storage.max_retries', 3)
        
        # Kafka topics
        self.topics = {
            'raw_logs': self.config.get('kafka.topics.raw_logs', 'raw-logs'),
            'anomaly_scores': self.config.get('kafka.topics.anomaly_scores', 'anomaly-scores'),
            'alerts': self.config.get('kafka.topics.alerts', 'alerts'),
            'system_metrics': self.config.get('kafka.topics.system_metrics', 'system-metrics')
        }
        
        # Health check Flask app
        self.app = Flask(__name__)
        self._setup_health_endpoints()
    
    def _setup_health_endpoints(self):
        """Setup health check endpoints"""
        @self.app.route('/health', methods=['GET'])
        def health_check():
            status = self.get_service_status()
            return jsonify(status), 200 if status.get('healthy') else 503
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            return jsonify(self.stats)
        
        @self.app.route('/tables', methods=['GET'])
        def get_tables():
            try:
                if self.db and self.db.is_connected():
                    tables = self.db.client.execute("SHOW TABLES")
                    table_info = {}
                    
                    for table in tables:
                        table_name = table[0]
                        try:
                            count = self.db.client.execute(f"SELECT count() FROM {table_name}")[0][0]
                            table_info[table_name] = {'row_count': count}
                        except:
                            table_info[table_name] = {'row_count': 'error'}
                    
                    return jsonify(table_info)
                else:
                    return jsonify({'error': 'Database not connected'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self):
        """Start the data storage service"""
        self.logger.info("🗄️ Starting Data Storage Service...")
        self.start_time = datetime.now()
        
        # Connect to ClickHouse
        try:
            self.db.connect()
            if self.db.is_connected():
                self.logger.info("✅ Connected to ClickHouse")
                self._ensure_database_schema()
            else:
                self.logger.error("❌ Failed to connect to ClickHouse")
                return False
        except Exception as e:
            self.logger.error(f"❌ Database connection error: {e}")
            return False
        
        # Setup Kafka consumers
        try:
            # Consumer for raw logs
            self.kafka_client.create_consumer(
                topics=[self.topics['raw_logs']],
                group_id="data-storage-raw-logs-consumer",
                message_handler=self._handle_raw_log
            )
            
            # Consumer for anomaly scores
            self.kafka_client.create_consumer(
                topics=[self.topics['anomaly_scores']],
                group_id="data-storage-anomaly-scores-consumer",
                message_handler=self._handle_anomaly_score
            )
            
            # Consumer for alerts
            self.kafka_client.create_consumer(
                topics=[self.topics['alerts']],
                group_id="data-storage-alerts-consumer",
                message_handler=self._handle_alert
            )
            
            self.logger.info("✅ Kafka consumers setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Kafka setup error: {e}")
            return False
        
        self.running = True
        
        # Start background threads
        self._start_background_threads()
        
        # Start Flask health server
        self._start_health_server()
        
        self.logger.info("🚀 Data Storage Service is running!")
        return True
    
    def _ensure_database_schema(self):
        """Ensure all required database tables exist with proper schema"""
        try:
            # Check existing tables
            tables = self.db.client.execute("SHOW TABLES")
            existing_tables = [table[0] for table in tables]
            
            # Create raw_logs table if it doesn't exist
            if 'raw_logs' not in existing_tables:
                self.logger.info("📋 Creating raw_logs table...")
                self.db.client.execute("""
                    CREATE TABLE raw_logs (
                        timestamp DateTime,
                        event_type String,
                        source_ip String,
                        destination_ip String,
                        port UInt16,
                        protocol String,
                        message String,
                        severity String,
                        threat_indicators Array(String),
                        event_id String,
                        session_id String,
                        user_agent String,
                        bytes_transferred UInt64,
                        duration_seconds UInt32
                    ) ENGINE = MergeTree() 
                    ORDER BY timestamp
                    SETTINGS index_granularity = 8192
                """)
                
                # Create indexes for better query performance
                self.db.client.execute("""
                    ALTER TABLE raw_logs ADD INDEX idx_source_ip source_ip TYPE bloom_filter GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE raw_logs ADD INDEX idx_severity severity TYPE set(0) GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE raw_logs ADD INDEX idx_threat_indicators length(threat_indicators) TYPE minmax GRANULARITY 4
                """)
                
                self.logger.info("✅ Created raw_logs table with indexes")
            
            # Create anomaly_scores table if it doesn't exist
            if 'anomaly_scores' not in existing_tables:
                self.logger.info("📋 Creating anomaly_scores table...")
                self.db.client.execute("""
                    CREATE TABLE anomaly_scores (
                        timestamp DateTime,
                        model_name String,
                        entity_id String,
                        score Float64,
                        features String,
                        is_anomaly UInt8,
                        confidence Float64,
                        model_version String,
                        processing_time_ms Float64,
                        original_severity String,
                        threat_indicators_count UInt32
                    ) ENGINE = MergeTree() 
                    ORDER BY (timestamp, model_name, entity_id)
                    SETTINGS index_granularity = 8192
                """)
                
                # Create indexes for anomaly_scores
                self.db.client.execute("""
                    ALTER TABLE anomaly_scores ADD INDEX idx_entity_id entity_id TYPE bloom_filter GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE anomaly_scores ADD INDEX idx_model_name model_name TYPE set(0) GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE anomaly_scores ADD INDEX idx_score score TYPE minmax GRANULARITY 4
                """)
                
                self.logger.info("✅ Created anomaly_scores table with indexes")
            
            # Create alerts table if it doesn't exist
            if 'alerts' not in existing_tables:
                self.logger.info("📋 Creating alerts table...")
                self.db.client.execute("""
                    CREATE TABLE alerts (
                        timestamp DateTime,
                        alert_id String,
                        severity String,
                        entity_id String,
                        message String,
                        aggregated_score Float64,
                        contributing_models Array(String),
                        status String DEFAULT 'open',
                        created_at DateTime DEFAULT now(),
                        updated_at DateTime DEFAULT now()
                    ) ENGINE = MergeTree() 
                    ORDER BY timestamp
                    SETTINGS index_granularity = 8192
                """)
                
                # Create indexes for alerts
                self.db.client.execute("""
                    ALTER TABLE alerts ADD INDEX idx_alert_severity severity TYPE set(0) GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE alerts ADD INDEX idx_alert_entity entity_id TYPE bloom_filter GRANULARITY 4
                """)
                self.db.client.execute("""
                    ALTER TABLE alerts ADD INDEX idx_alert_status status TYPE set(0) GRANULARITY 4
                """)
                
                self.logger.info("✅ Created alerts table with indexes")
            
            # Create materialized views for analytics
            self._create_materialized_views()
            
            self.logger.info("✅ Database schema ensured successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error ensuring database schema: {e}")
            raise
    
    def _create_materialized_views(self):
        """Create materialized views for better dashboard performance"""
        try:
            # Hourly aggregation view for raw logs
            self.db.client.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS raw_logs_hourly
                ENGINE = AggregatingMergeTree()
                ORDER BY (hour, severity, event_type)
                AS SELECT
                    toStartOfHour(timestamp) as hour,
                    severity,
                    event_type,
                    countState() as count,
                    uniqState(source_ip) as unique_sources,
                    avgState(bytes_transferred) as avg_bytes,
                    sumState(arrayLength(threat_indicators)) as total_threats
                FROM raw_logs
                GROUP BY hour, severity, event_type
            """)
            
            # Hourly aggregation view for anomaly scores
            self.db.client.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS anomaly_scores_hourly
                ENGINE = AggregatingMergeTree()
                ORDER BY (hour, model_name)
                AS SELECT
                    toStartOfHour(timestamp) as hour,
                    model_name,
                    countState() as count,
                    avgState(score) as avg_score,
                    sumState(is_anomaly) as anomaly_count,
                    avgState(confidence) as avg_confidence,
                    avgState(processing_time_ms) as avg_processing_time
                FROM anomaly_scores
                GROUP BY hour, model_name
            """)
            
            self.logger.info("✅ Created materialized views for analytics")
            
        except Exception as e:
            self.logger.debug(f"Materialized views creation info: {e}")
    
    def _start_health_server(self):
        """Start Flask health check server"""
        def run_health_server():
            try:
                self.app.run(host='0.0.0.0', port=8003, debug=False, use_reloader=False)
            except Exception as e:
                self.logger.error(f"Health server error: {e}")
        
        health_thread = threading.Thread(target=run_health_server, daemon=True, name="HealthServer")
        health_thread.start()
        self.logger.info("✅ Health check server started on port 8003")
    
    def _start_background_threads(self):
        """Start background processing threads"""
        threads = [
            ("Raw Logs Batch Processor", self._raw_logs_batch_processor),
            ("Anomaly Scores Batch Processor", self._anomaly_scores_batch_processor),
            ("Alerts Batch Processor", self._alerts_batch_processor),
            ("System Metrics Reporter", self._system_metrics_loop),
            ("Database Health Monitor", self._database_health_monitor)
        ]
        
        for thread_name, target_function in threads:
            try:
                thread = threading.Thread(target=target_function, daemon=True, name=thread_name)
                thread.start()
                self.logger.info(f"✅ Started {thread_name} thread")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {thread_name} thread: {e}")
    
    def _handle_raw_log(self, log_data: Dict[str, Any]):
        """Handle raw log from Kafka"""
        try:
            self.stats['raw_logs_processed'] += 1
            self.raw_logs_buffer.append(log_data)
            
            # Process batch if size limit reached
            if len(self.raw_logs_buffer) >= self.batch_size:
                self._process_raw_logs_batch()
                
        except Exception as e:
            self.logger.error(f"❌ Error handling raw log: {e}")
    
    def _handle_anomaly_score(self, score_data: Dict[str, Any]):
        """Handle anomaly score from Kafka"""
        try:
            self.stats['anomaly_scores_processed'] += 1
            self.anomaly_scores_buffer.append(score_data)
            
            # Process batch if size limit reached
            if len(self.anomaly_scores_buffer) >= self.batch_size:
                self._process_anomaly_scores_batch()
                
        except Exception as e:
            self.logger.error(f"❌ Error handling anomaly score: {e}")
    
    def _handle_alert(self, alert_data: Dict[str, Any]):
        """Handle alert from Kafka"""
        try:
            self.stats['alerts_processed'] += 1
            self.alerts_buffer.append(alert_data)
            
            # Process batch if size limit reached
            if len(self.alerts_buffer) >= self.batch_size:
                self._process_alerts_batch()
                
        except Exception as e:
            self.logger.error(f"❌ Error handling alert: {e}")
    
    def _raw_logs_batch_processor(self):
        """Process raw logs batches periodically"""
        while self.running:
            try:
                time.sleep(self.batch_timeout)
                if self.raw_logs_buffer:
                    self._process_raw_logs_batch()
            except Exception as e:
                self.logger.error(f"❌ Raw logs batch processor error: {e}")
                time.sleep(self.batch_timeout)
    
    def _anomaly_scores_batch_processor(self):
        """Process anomaly scores batches periodically"""
        while self.running:
            try:
                time.sleep(self.batch_timeout)
                if self.anomaly_scores_buffer:
                    self._process_anomaly_scores_batch()
            except Exception as e:
                self.logger.error(f"❌ Anomaly scores batch processor error: {e}")
                time.sleep(self.batch_timeout)
    
    def _alerts_batch_processor(self):
        """Process alerts batches periodically"""
        while self.running:
            try:
                time.sleep(self.batch_timeout)
                if self.alerts_buffer:
                    self._process_alerts_batch()
            except Exception as e:
                self.logger.error(f"❌ Alerts batch processor error: {e}")
                time.sleep(self.batch_timeout)
    
    def _process_raw_logs_batch(self):
        """Process and store raw logs batch"""
        if not self.raw_logs_buffer:
            return
        
        current_batch = list(self.raw_logs_buffer)
        self.raw_logs_buffer.clear()
        
        try:
            # Prepare logs for ClickHouse
            processed_logs = []
            for log_data in current_batch:
                processed_log = self._prepare_raw_log_for_db(log_data)
                if processed_log:
                    processed_logs.append(processed_log)
            
            if processed_logs:
                # Insert batch into ClickHouse
                for retry in range(self.max_retries):
                    try:
                        self.db.client.execute(
                            "INSERT INTO raw_logs VALUES",
                            processed_logs
                        )
                        self.stats['raw_logs_stored'] += len(processed_logs)
                        self.stats['last_storage_time'] = datetime.now()
                        break
                    except Exception as e:
                        if retry == self.max_retries - 1:
                            self.logger.error(f"❌ Failed to store raw logs batch after {self.max_retries} retries: {e}")
                            self.stats['database_errors'] += 1
                        else:
                            self.logger.warning(f"⚠️ Retry {retry + 1} for raw logs batch: {e}")
                            time.sleep(1)
                
                self.logger.info(f"📝 Stored {len(processed_logs)} raw logs in ClickHouse")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing raw logs batch: {e}")
            self.stats['batch_processing_errors'] += 1
    
    def _process_anomaly_scores_batch(self):
        """Process and store anomaly scores batch"""
        if not self.anomaly_scores_buffer:
            return
        
        current_batch = list(self.anomaly_scores_buffer)
        self.anomaly_scores_buffer.clear()
        
        try:
            # Prepare scores for ClickHouse
            processed_scores = []
            for score_data in current_batch:
                processed_score = self._prepare_anomaly_score_for_db(score_data)
                if processed_score:
                    processed_scores.append(processed_score)
            
            if processed_scores:
                # Insert batch into ClickHouse
                for retry in range(self.max_retries):
                    try:
                        self.db.client.execute(
                            "INSERT INTO anomaly_scores VALUES",
                            processed_scores
                        )
                        self.stats['anomaly_scores_stored'] += len(processed_scores)
                        self.stats['last_storage_time'] = datetime.now()
                        break
                    except Exception as e:
                        if retry == self.max_retries - 1:
                            self.logger.error(f"❌ Failed to store anomaly scores batch after {self.max_retries} retries: {e}")
                            self.stats['database_errors'] += 1
                        else:
                            self.logger.warning(f"⚠️ Retry {retry + 1} for anomaly scores batch: {e}")
                            time.sleep(1)
                
                self.logger.info(f"📊 Stored {len(processed_scores)} anomaly scores in ClickHouse")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing anomaly scores batch: {e}")
            self.stats['batch_processing_errors'] += 1
    
    def _process_alerts_batch(self):
        """Process and store alerts batch"""
        if not self.alerts_buffer:
            return
        
        current_batch = list(self.alerts_buffer)
        self.alerts_buffer.clear()
        
        try:
            # Prepare alerts for ClickHouse
            processed_alerts = []
            for alert_data in current_batch:
                processed_alert = self._prepare_alert_for_db(alert_data)
                if processed_alert:
                    processed_alerts.append(processed_alert)
            
            if processed_alerts:
                # Insert batch into ClickHouse
                for retry in range(self.max_retries):
                    try:
                        self.db.client.execute(
                            "INSERT INTO alerts VALUES",
                            processed_alerts
                        )
                        self.stats['alerts_stored'] += len(processed_alerts)
                        self.stats['last_storage_time'] = datetime.now()
                        break
                    except Exception as e:
                        if retry == self.max_retries - 1:
                            self.logger.error(f"❌ Failed to store alerts batch after {self.max_retries} retries: {e}")
                            self.stats['database_errors'] += 1
                        else:
                            self.logger.warning(f"⚠️ Retry {retry + 1} for alerts batch: {e}")
                            time.sleep(1)
                
                self.logger.info(f"🚨 Stored {len(processed_alerts)} alerts in ClickHouse")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing alerts batch: {e}")
            self.stats['batch_processing_errors'] += 1
    
    def _prepare_raw_log_for_db(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare raw log data for ClickHouse insertion"""
        try:
            return {
                'timestamp': self._parse_timestamp_for_db(log_data.get('timestamp', datetime.now())),
                'event_type': str(log_data.get('event_type', 'unknown'))[:255],
                'source_ip': str(log_data.get('source_ip', '0.0.0.0'))[:45],
                'destination_ip': str(log_data.get('destination_ip', '0.0.0.0'))[:45],
                'port': max(0, min(65535, int(log_data.get('port', 0)))),
                'protocol': str(log_data.get('protocol', 'unknown'))[:50],
                'message': str(log_data.get('message', ''))[:1000],
                'severity': str(log_data.get('severity', 'info'))[:20],
                'threat_indicators': list(log_data.get('threat_indicators', []))[:20],  # Limit array size
                'event_id': str(log_data.get('event_id', f'evt_{int(time.time())}'))[:255],
                'session_id': str(log_data.get('session_id', f'sess_{int(time.time())}'))[:255],
                'user_agent': str(log_data.get('user_agent', 'unknown'))[:500],
                'bytes_transferred': max(0, int(log_data.get('bytes_transferred', 0))),
                'duration_seconds': max(0, int(log_data.get('duration_seconds', 0)))
            }
        except Exception as e:
            self.logger.debug(f"Error preparing raw log for DB: {e}")
            return None
    
    def _prepare_anomaly_score_for_db(self, score_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare anomaly score data for ClickHouse insertion"""
        try:
            return {
                'timestamp': self._parse_timestamp_for_db(score_data.get('timestamp', datetime.now())),
                'model_name': str(score_data.get('model_name', 'unknown'))[:100],
                'entity_id': str(score_data.get('entity_id', 'unknown'))[:100],
                'score': max(0.0, min(1.0, float(score_data.get('score', 0.0)))),
                'features': str(score_data.get('features', '{}'))[:2000],
                'is_anomaly': 1 if score_data.get('is_anomaly', False) else 0,
                'confidence': max(0.0, min(1.0, float(score_data.get('confidence', 0.0)))),
                'model_version': str(score_data.get('model_version', '1.0'))[:50],
                'processing_time_ms': max(0.0, float(score_data.get('processing_time_ms', 0.0))),
                'original_severity': str(score_data.get('original_severity', 'info'))[:20],
                'threat_indicators_count': max(0, int(score_data.get('threat_indicators_count', 0)))
            }
        except Exception as e:
            self.logger.debug(f"Error preparing anomaly score for DB: {e}")
            return None
    
    def _prepare_alert_for_db(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare alert data for ClickHouse insertion"""
        try:
            return {
                'timestamp': self._parse_timestamp_for_db(alert_data.get('timestamp', datetime.now())),
                'alert_id': str(alert_data.get('alert_id', f'alert_{int(time.time())}'))[:255],
                'severity': str(alert_data.get('severity', 'medium'))[:20],
                'entity_id': str(alert_data.get('entity_id', 'unknown'))[:100],
                'message': str(alert_data.get('message', ''))[:1000],
                'aggregated_score': max(0.0, min(1.0, float(alert_data.get('aggregated_score', 0.0)))),
                'contributing_models': list(alert_data.get('contributing_models', []))[:10],
                'status': str(alert_data.get('status', 'open'))[:20],
                'created_at': self._parse_timestamp_for_db(alert_data.get('created_at', datetime.now())),
                'updated_at': self._parse_timestamp_for_db(alert_data.get('updated_at', datetime.now()))
            }
        except Exception as e:
            self.logger.debug(f"Error preparing alert for DB: {e}")
            return None
    
    def _parse_timestamp_for_db(self, timestamp) -> datetime:
        """Parse timestamp for ClickHouse insertion"""
        try:
            if isinstance(timestamp, str):
                # Handle ISO format
                timestamp = timestamp.replace('Z', '+00:00')
                return datetime.fromisoformat(timestamp)
            elif isinstance(timestamp, datetime):
                return timestamp
            else:
                return datetime.now()
        except Exception as e:
            self.logger.debug(f"Error parsing timestamp: {e}")
            return datetime.now()
    
    def _system_metrics_loop(self):
        """Send system metrics periodically"""
        while self.running:
            try:
                time.sleep(30)  # Every 30 seconds
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'service': 'data-storage',
                    'cpu_usage': cpu_percent / 100.0,
                    'memory_usage': memory.percent / 100.0,
                    'processing_latency': 0.05,  # Data storage is typically fast
                    'raw_logs_processed': self.stats['raw_logs_processed'],
                    'raw_logs_stored': self.stats['raw_logs_stored'],
                    'anomaly_scores_processed': self.stats['anomaly_scores_processed'],
                    'anomaly_scores_stored': self.stats['anomaly_scores_stored'],
                    'alerts_processed': self.stats['alerts_processed'],
                    'alerts_stored': self.stats['alerts_stored'],
                    'database_errors': self.stats['database_errors'],
                    'batch_processing_errors': self.stats['batch_processing_errors'],
                    'buffer_sizes': {
                        'raw_logs': len(self.raw_logs_buffer),
                        'anomaly_scores': len(self.anomaly_scores_buffer),
                        'alerts': len(self.alerts_buffer)
                    }
                }
                
                self.kafka_client.send_message(
                    topic=self.topics['system_metrics'],
                    message=metrics
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error sending system metrics: {e}")
                time.sleep(30)
    
    def _database_health_monitor(self):
        """Monitor database health and connection"""
        while self.running:
            try:
                time.sleep(60)  # Every minute
                
                if self.db and self.db.is_connected():
                    # Test database connection
                    try:
                        self.db.client.execute("SELECT 1")
                        
                        # Log storage statistics
                        raw_logs_count = self.db.client.execute("SELECT count() FROM raw_logs")[0][0]
                        scores_count = self.db.client.execute("SELECT count() FROM anomaly_scores")[0][0]
                        alerts_count = self.db.client.execute("SELECT count() FROM alerts")[0][0]
                        
                        self.logger.info(f"💾 Database status: {raw_logs_count} raw logs, "
                                       f"{scores_count} anomaly scores, {alerts_count} alerts")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Database health check failed: {e}")
                        # Try to reconnect
                        try:
                            self.db.connect()
                        except:
                            pass
                
            except Exception as e:
                self.logger.error(f"❌ Database health monitor error: {e}")
                time.sleep(60)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            db_connected = self.db.is_connected() if self.db else False
            
            # Calculate processing rates
            uptime_seconds = (datetime.now() - self.start_time).total_seconds() if self.start_time else 1
            
            return {
                'healthy': db_connected and self.running,
                'running': self.running,
                'database_connected': db_connected,
                'uptime_seconds': uptime_seconds,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'processing_stats': {
                    'raw_logs_processed': self.stats['raw_logs_processed'],
                    'raw_logs_stored': self.stats['raw_logs_stored'],
                    'raw_logs_rate_per_min': (self.stats['raw_logs_processed'] / uptime_seconds) * 60,
                    'anomaly_scores_processed': self.stats['anomaly_scores_processed'],
                    'anomaly_scores_stored': self.stats['anomaly_scores_stored'],
                    'anomaly_scores_rate_per_min': (self.stats['anomaly_scores_processed'] / uptime_seconds) * 60,
                    'alerts_processed': self.stats['alerts_processed'],
                    'alerts_stored': self.stats['alerts_stored'],
                    'alerts_rate_per_min': (self.stats['alerts_processed'] / uptime_seconds) * 60
                },
                'error_stats': {
                    'database_errors': self.stats['database_errors'],
                    'batch_processing_errors': self.stats['batch_processing_errors'],
                    'total_errors': self.stats['database_errors'] + self.stats['batch_processing_errors']
                },
                'buffer_status': {
                    'raw_logs_buffer_size': len(self.raw_logs_buffer),
                    'anomaly_scores_buffer_size': len(self.anomaly_scores_buffer),
                    'alerts_buffer_size': len(self.alerts_buffer),
                    'batch_size_limit': self.batch_size
                },
                'last_storage_time': self.stats['last_storage_time'].isoformat() if self.stats['last_storage_time'] else None,
                'configuration': {
                    'batch_size': self.batch_size,
                    'batch_timeout': self.batch_timeout,
                    'max_retries': self.max_retries
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                'healthy': False,
                'running': self.running,
                'error': str(e)
            }
    
    def stop(self):
        """Stop the data storage service gracefully"""
        self.logger.info("🛑 Stopping Data Storage Service...")
        self.running = False
        
        try:
            # Process remaining buffers
            if self.raw_logs_buffer:
                self.logger.info(f"Processing remaining {len(self.raw_logs_buffer)} raw logs...")
                self._process_raw_logs_batch()
            
            if self.anomaly_scores_buffer:
                self.logger.info(f"Processing remaining {len(self.anomaly_scores_buffer)} anomaly scores...")
                self._process_anomaly_scores_batch()
            
            if self.alerts_buffer:
                self.logger.info(f"Processing remaining {len(self.alerts_buffer)} alerts...")
                self._process_alerts_batch()
            
            # Close database connection
            if self.db and hasattr(self.db, 'close'):
                self.db.close()
                self.logger.info("✅ Database connection closed")
            
            # Final statistics
            self.logger.info(f"📊 Final stats: {self.stats['raw_logs_stored']} raw logs stored, "
                           f"{self.stats['anomaly_scores_stored']} anomaly scores stored, "
                           f"{self.stats['alerts_stored']} alerts stored")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def setup_signal_handlers(service):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for data storage service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🗄️ Starting Data Storage Service...")
        
        service = DataStorageService()
        
        # Setup graceful shutdown
        setup_signal_handlers(service)
        
        if service.start():
            logger.info("✅ Data Storage Service started successfully")
            logger.info("📊 Processing raw logs, anomaly scores, and alerts")
            logger.info("🌐 Health check available at http://localhost:8003/health")
            
            # Status monitoring loop
            while True:
                time.sleep(60)  # Status update every minute
                
                try:
                    status = service.get_service_status()
                    
                    # Log processing statistics
                    proc_stats = status.get('processing_stats', {})
                    logger.info(f"📊 Processed: {proc_stats.get('raw_logs_processed', 0)} raw logs, "
                               f"{proc_stats.get('anomaly_scores_processed', 0)} scores, "
                               f"{proc_stats.get('alerts_processed', 0)} alerts")
                    
                    # Log storage statistics
                    logger.info(f"💾 Stored: {proc_stats.get('raw_logs_stored', 0)} raw logs, "
                               f"{proc_stats.get('anomaly_scores_stored', 0)} scores, "
                               f"{proc_stats.get('alerts_stored', 0)} alerts")
                    
                    # Log buffer status if significant
                    buffer_status = status.get('buffer_status', {})
                    total_buffered = (buffer_status.get('raw_logs_buffer_size', 0) + 
                                    buffer_status.get('anomaly_scores_buffer_size', 0) + 
                                    buffer_status.get('alerts_buffer_size', 0))
                    
                    if total_buffered > 0:
                        logger.info(f"📦 Buffered: {buffer_status.get('raw_logs_buffer_size', 0)} raw logs, "
                                   f"{buffer_status.get('anomaly_scores_buffer_size', 0)} scores, "
                                   f"{buffer_status.get('alerts_buffer_size', 0)} alerts")
                    
                    # Log errors if any
                    error_stats = status.get('error_stats', {})
                    if error_stats.get('total_errors', 0) > 0:
                        logger.warning(f"⚠️ Errors: {error_stats.get('database_errors', 0)} database, "
                                     f"{error_stats.get('batch_processing_errors', 0)} batch processing")
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
        else:
            logger.error("❌ Failed to start Data Storage Service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💥 Service error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if 'service' in locals():
            service.stop()

if __name__ == "__main__":
    main()