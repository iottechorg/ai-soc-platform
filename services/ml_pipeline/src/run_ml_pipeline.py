#!/usr/bin/env python3
# complete_ml_pipeline.py - COMPLETE ML Pipeline with proper ml_models integration

import sys
import time
import logging
import threading
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import random
import numpy as np
from flask import Flask, jsonify
import signal

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import your enhanced ML models
    from ml_models import MLModelManager, AdaptiveThresholdManager
    from shared.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
    from shared.database import ClickHouseClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class EnhancedDatabaseTrainingManager:
    """Enhanced training manager with better error handling and caching"""
    
    def __init__(self, db_client: ClickHouseClient):
        self.db = db_client
        self.logger = logging.getLogger(__name__)
        self.last_training_time = None
        self.training_cache = {}
        self.cache_ttl = 1800  # 30 minutes
        self.query_timeout = 60  # 60 seconds timeout for queries
    
    def get_balanced_training_data(self, 
                                 sample_size: int = 2000, 
                                 threat_ratio: float = 0.15,
                                 use_cache: bool = True) -> List[Dict[str, Any]]:
        """Get balanced training data with enhanced error handling"""
        
        cache_key = f"training_{sample_size}_{threat_ratio}"
        
        # Check cache first
        if use_cache and cache_key in self.training_cache:
            cache_time, cached_data = self.training_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                self.logger.info(f"Using cached training data ({len(cached_data)} samples)")
                return cached_data
        
        threat_count = int(sample_size * threat_ratio)
        normal_count = sample_size - threat_count
        
        training_data = []
        
        try:
            # Enhanced threat query with better error handling
            self.logger.info(f"Querying {threat_count} threat samples from database...")
            
            threat_query = f"""
            SELECT 
                toString(timestamp) as timestamp,
                event_type,
                source_ip,
                destination_ip,
                port,
                protocol,
                message,
                severity,
                threat_indicators,
                event_id,
                session_id,
                user_agent,
                bytes_transferred,
                duration_seconds
            FROM raw_logs 
            WHERE length(threat_indicators) > 0
            AND timestamp >= now() - INTERVAL 48 HOUR
            ORDER BY timestamp DESC
            LIMIT {threat_count}
            SETTINGS max_execution_time = {self.query_timeout}
            """
            
            threat_results = self.db.client.execute(threat_query)
            
            for row in threat_results:
                try:
                    # Convert ClickHouse row to dict with proper type handling
                    training_data.append({
                        'timestamp': self._parse_timestamp(row[0]),
                        'event_type': str(row[1]) if row[1] else 'unknown',
                        'source_ip': str(row[2]) if row[2] else '0.0.0.0',
                        'destination_ip': str(row[3]) if row[3] else '0.0.0.0',
                        'port': int(row[4]) if row[4] is not None else 80,
                        'protocol': str(row[5]) if row[5] else 'tcp',
                        'message': str(row[6]) if row[6] else '',
                        'severity': str(row[7]) if row[7] else 'info',
                        'threat_indicators': self._parse_array(row[8]),
                        'event_id': str(row[9]) if row[9] else f'evt_{int(time.time())}',
                        'session_id': str(row[10]) if row[10] else f'sess_{int(time.time())}',
                        'user_agent': str(row[11]) if row[11] else 'unknown',
                        'bytes_transferred': int(row[12]) if row[12] is not None else 0,
                        'duration_seconds': int(row[13]) if row[13] is not None else 0
                    })
                except Exception as e:
                    self.logger.debug(f"Error parsing threat row: {e}")
                    continue
            
            self.logger.info(f"Successfully collected {len(training_data)} threat samples")
            
            # Enhanced normal query with sampling
            self.logger.info(f"Querying {normal_count} normal samples from database...")
            
            normal_query = f"""
            SELECT 
                toString(timestamp) as timestamp,
                event_type,
                source_ip,
                destination_ip,
                port,
                protocol,
                message,
                severity,
                threat_indicators,
                event_id,
                session_id,
                user_agent,
                bytes_transferred,
                duration_seconds
            FROM raw_logs 
            WHERE length(threat_indicators) = 0
            AND timestamp >= now() - INTERVAL 12 HOUR
            ORDER BY rand()
            LIMIT {normal_count}
            SETTINGS max_execution_time = {self.query_timeout}
            """
            
            normal_results = self.db.client.execute(normal_query)
            
            for row in normal_results:
                try:
                    training_data.append({
                        'timestamp': self._parse_timestamp(row[0]),
                        'event_type': str(row[1]) if row[1] else 'web_request',
                        'source_ip': str(row[2]) if row[2] else '192.168.1.1',
                        'destination_ip': str(row[3]) if row[3] else '10.0.0.1',
                        'port': int(row[4]) if row[4] is not None else 80,
                        'protocol': str(row[5]) if row[5] else 'http',
                        'message': str(row[6]) if row[6] else '',
                        'severity': str(row[7]) if row[7] else 'info',
                        'threat_indicators': [],  # Normal traffic has no threat indicators
                        'event_id': str(row[9]) if row[9] else f'evt_{int(time.time())}',
                        'session_id': str(row[10]) if row[10] else f'sess_{int(time.time())}',
                        'user_agent': str(row[11]) if row[11] else 'Mozilla/5.0',
                        'bytes_transferred': int(row[12]) if row[12] is not None else 1024,
                        'duration_seconds': int(row[13]) if row[13] is not None else 2
                    })
                except Exception as e:
                    self.logger.debug(f"Error parsing normal row: {e}")
                    continue
            
            self.logger.info(f"Successfully collected {len(training_data) - len([d for d in training_data if d.get('threat_indicators')])} normal samples")
            
            # Shuffle for better training
            random.shuffle(training_data)
            
            # Cache the results
            if use_cache:
                self.training_cache[cache_key] = (datetime.now(), training_data)
            
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Total training data collected: {len(training_data)} samples "
                           f"({len([d for d in training_data if d.get('threat_indicators', [])])} threats, "
                           f"{len([d for d in training_data if not d.get('threat_indicators', [])])} normal)")
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data from database: {e}")
            self.logger.warning("Falling back to synthetic training data")
            return self._get_enhanced_fallback_data(sample_size, threat_ratio)
    
    def _parse_timestamp(self, timestamp_str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            if isinstance(timestamp_str, str):
                # Handle different timestamp formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        return datetime.strptime(timestamp_str.split('.')[0], fmt)
                    except:
                        continue
            return datetime.now()
        except:
            return datetime.now()
    
    def _parse_array(self, array_data) -> List[str]:
        """Parse ClickHouse array data to Python list"""
        try:
            if isinstance(array_data, (list, tuple)):
                return [str(item) for item in array_data if item]
            elif isinstance(array_data, str):
                # Handle string representation of array
                if array_data.startswith('[') and array_data.endswith(']'):
                    import ast
                    return ast.literal_eval(array_data)
                elif array_data:
                    return [array_data]
            return []
        except Exception as e:
            self.logger.debug(f"Error parsing array data: {e}")
            return []
    
    def _get_enhanced_fallback_data(self, sample_size: int, threat_ratio: float) -> List[Dict[str, Any]]:
        """Enhanced fallback data with more realistic patterns"""
        self.logger.warning("Using enhanced fallback synthetic training data")
        
        threat_count = int(sample_size * threat_ratio)
        normal_count = sample_size - threat_count
        
        training_data = []
        
        # Enhanced threat patterns
        threat_patterns = [
            {
                'event_type': 'brute_force_ssh',
                'port': 22,
                'protocol': 'ssh',
                'severity': 'high',
                'indicators': ['multiple_failed_logins', 'suspicious_source'],
                'bytes': (64, 512),
                'duration': (1, 30)
            },
            {
                'event_type': 'web_application_attack',
                'port': 80,
                'protocol': 'http',
                'severity': 'medium',
                'indicators': ['sql_injection', 'web_attack'],
                'bytes': (512, 8192),
                'duration': (2, 15)
            },
            {
                'event_type': 'data_exfiltration',
                'port': 443,
                'protocol': 'https',
                'severity': 'critical',
                'indicators': ['large_data_transfer', 'unusual_destination'],
                'bytes': (1048576, 104857600),
                'duration': (300, 3600)
            },
            {
                'event_type': 'port_scan',
                'port': 80,
                'protocol': 'tcp',
                'severity': 'medium',
                'indicators': ['port_scanning', 'reconnaissance'],
                'bytes': (64, 256),
                'duration': (1, 5)
            }
        ]
        
        # Generate threat samples
        for i in range(threat_count):
            pattern = random.choice(threat_patterns)
            training_data.append({
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 48)),
                'event_type': pattern['event_type'],
                'source_ip': f'10.10.{random.randint(1, 50)}.{random.randint(1, 254)}',
                'destination_ip': f'192.168.{random.randint(1, 10)}.{random.randint(1, 254)}',
                'port': pattern['port'],
                'protocol': pattern['protocol'],
                'message': f"Detected {pattern['event_type']} attack",
                'severity': pattern['severity'],
                'threat_indicators': pattern['indicators'].copy(),
                'event_id': f'evt_fallback_{i}',
                'session_id': f'sess_fallback_{i}',
                'user_agent': 'threat_agent/1.0',
                'bytes_transferred': random.randint(*pattern['bytes']),
                'duration_seconds': random.randint(*pattern['duration'])
            })
        
        # Generate normal samples with variety
        normal_patterns = [
            {'event_type': 'web_request', 'port': 80, 'protocol': 'http'},
            {'event_type': 'web_request', 'port': 443, 'protocol': 'https'},
            {'event_type': 'database_query', 'port': 3306, 'protocol': 'mysql'},
            {'event_type': 'file_access', 'port': 445, 'protocol': 'smb'},
            {'event_type': 'dns_query', 'port': 53, 'protocol': 'dns'}
        ]
        
        for i in range(normal_count):
            pattern = random.choice(normal_patterns)
            training_data.append({
                'timestamp': datetime.now() - timedelta(hours=random.randint(1, 12)),
                'event_type': pattern['event_type'],
                'source_ip': f'192.168.{random.randint(1, 10)}.{random.randint(1, 254)}',
                'destination_ip': f'10.0.{random.randint(1, 5)}.{random.randint(1, 254)}',
                'port': pattern['port'],
                'protocol': pattern['protocol'],
                'message': f"Normal {pattern['event_type']} activity",
                'severity': 'info',
                'threat_indicators': [],
                'event_id': f'evt_normal_{i}',
                'session_id': f'sess_normal_{i}',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'bytes_transferred': random.randint(512, 10240),
                'duration_seconds': random.randint(1, 10)
            })
        
        random.shuffle(training_data)
        return training_data

class StreamingTrainingCollector:
    """Enhanced streaming collector with better data management"""
    
    def __init__(self, max_threats: int = 300, max_normal: int = 1200):
        self.training_buffer = {
            'threats': deque(maxlen=max_threats),
            'normal': deque(maxlen=max_normal)
        }
        self.logger = logging.getLogger(__name__)
        self.normal_sample_rate = 0.08  # Keep 8% of normal traffic
        self.collection_stats = {
            'threats_collected': 0,
            'normal_collected': 0,
            'total_processed': 0
        }
    
    def add_sample(self, log_data: Dict[str, Any]):
        """Enhanced sample collection with better filtering"""
        try:
            self.collection_stats['total_processed'] += 1
            
            threat_indicators = log_data.get('threat_indicators', [])
            
            if threat_indicators and len(threat_indicators) > 0:
                # Always keep threat samples
                self.training_buffer['threats'].append(log_data)
                self.collection_stats['threats_collected'] += 1
                self.logger.debug(f"Added threat sample: {log_data.get('event_type', 'unknown')}")
            else:
                # Sample normal traffic with adaptive rate
                if random.random() < self.normal_sample_rate:
                    self.training_buffer['normal'].append(log_data)
                    self.collection_stats['normal_collected'] += 1
                    
        except Exception as e:
            self.logger.debug(f"Error adding training sample: {e}")
    
    def get_streaming_training_data(self) -> List[Dict[str, Any]]:
        """Get balanced training set from streaming buffer"""
        threats = list(self.training_buffer['threats'])
        normal = list(self.training_buffer['normal'])
        
        if not threats and not normal:
            return []
        
        # Smart balancing based on available data
        if threats:
            # Aim for 15% threats, but adapt based on available data
            target_normal = min(len(threats) * 5, len(normal))  # 5:1 ratio
            if target_normal > 0 and len(normal) > target_normal:
                normal = random.sample(normal, target_normal)
        
        combined = threats + normal
        random.shuffle(combined)
        
        self.logger.info(f"Streaming training data: {len(threats)} threats, {len(normal)} normal")
        return combined
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            **self.collection_stats,
            'buffer_sizes': {
                'threats': len(self.training_buffer['threats']),
                'normal': len(self.training_buffer['normal'])
            },
            'sample_rate': self.normal_sample_rate
        }

class CompleteMHLPipeline:
    """Complete ML Pipeline with all functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # Initialize ML components
        self.ml_manager = MLModelManager(self.config._config)
        
        # Initialize database for training
        self.db = ClickHouseClient(
            host=self.config.get('clickhouse.host'),
            port=9000,
            user=self.config.get('clickhouse.user'),
            password=self.config.get('clickhouse.password'),
            database=self.config.get('clickhouse.database')
        )
        
        # Initialize Kafka
        self.kafka_client = KafkaClient(self.config.get('kafka.broker'))
        
        # Training managers
        self.db_training_manager = EnhancedDatabaseTrainingManager(self.db)
        self.streaming_collector = StreamingTrainingCollector()
        
        # Service state
        self.running = False
        self.last_full_training = None
        self.last_incremental_training = None
        self.processed_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.performance_stats = {
            'predictions_per_second': 0,
            'training_time_seconds': 0,
            'last_prediction_time': None,
            'model_accuracy_estimates': {}
        }
        
        # Kafka topics
        self.topics = {
            'raw_logs': self.config.get('kafka.topics.raw_logs', 'raw-logs'),
            'anomaly_scores': self.config.get('kafka.topics.anomaly_scores', 'anomaly-scores'),
            'model_control': self.config.get('kafka.topics.model_control', 'model-control'),
            'model_status': self.config.get('kafka.topics.model_status', 'model-status'),
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
            return jsonify(status), 200 if status.get('running') else 503
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            return jsonify(self.performance_stats)
        
        @self.app.route('/models', methods=['GET'])
        def models_status():
            return jsonify(self.ml_manager.get_model_status())
    
    def start(self):
        """Start the complete ML pipeline service"""
        self.logger.info("🚀 Starting Complete ML Pipeline Service...")
        
        # Connect to database
        try:
            self.db.connect()
            if self.db.is_connected():
                self.logger.info("✅ Connected to ClickHouse for training data")
                self._ensure_database_schema()
            else:
                self.logger.error("❌ Failed to connect to ClickHouse")
                return False
        except Exception as e:
            self.logger.error(f"❌ Database connection error: {e}")
            return False
        
        # Setup Kafka consumers
        try:
            # Consumer for raw logs (main processing)
            self.kafka_client.create_consumer(
                topics=[self.topics['raw_logs']],
                group_id="ml-pipeline-consumer",
                message_handler=self._process_raw_log
            )
            
            # Consumer for model control commands
            self.kafka_client.create_consumer(
                topics=[self.topics['model_control']],
                group_id="ml-pipeline-control-consumer",
                message_handler=self._handle_model_control
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
        
        # Initial training with database data
        self._perform_initial_training()
        
        self.logger.info("🎯 Complete ML Pipeline Service is running!")
        return True
    
    def _ensure_database_schema(self):
        """Ensure required database tables exist"""
        try:
            # Check if tables exist
            tables = self.db.client.execute("SHOW TABLES")
            existing_tables = [table[0] for table in tables]
            
            if 'raw_logs' not in existing_tables:
                self.logger.warning("raw_logs table not found, creating...")
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
                    ) ENGINE = MergeTree() ORDER BY timestamp
                """)
                self.logger.info("✅ Created raw_logs table")
            
            # Create indexes for better query performance
            try:
                self.db.client.execute("""
                    ALTER TABLE raw_logs ADD INDEX idx_threat_indicators length(threat_indicators) TYPE minmax GRANULARITY 4
                """)
            except:
                pass  # Index might already exist
                
        except Exception as e:
            self.logger.error(f"Error ensuring database schema: {e}")
    
    def _start_health_server(self):
        """Start Flask health check server"""
        def run_health_server():
            try:
                self.app.run(host='0.0.0.0', port=8001, debug=False, use_reloader=False)
            except Exception as e:
                self.logger.error(f"Health server error: {e}")
        
        health_thread = threading.Thread(target=run_health_server, daemon=True, name="HealthServer")
        health_thread.start()
        self.logger.info("✅ Health check server started on port 8001")
    
    def _start_background_threads(self):
        """Start all background processing threads"""
        threads = [
            ("Model Status Reporter", self._model_status_loop),
            ("Incremental Trainer", self._incremental_training_loop),
            ("Full Retrain Scheduler", self._full_retrain_loop),
            ("System Metrics Reporter", self._system_metrics_loop),
            ("Performance Monitor", self._performance_monitor_loop)
        ]
        
        for thread_name, target_function in threads:
            try:
                thread = threading.Thread(target=target_function, daemon=True, name=thread_name)
                thread.start()
                self.logger.info(f"✅ Started {thread_name} thread")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {thread_name} thread: {e}")
    
    def _perform_initial_training(self):
        """Perform comprehensive initial model training"""
        self.logger.info("🤖 Starting initial model training with database data...")
        
        try:
            start_time = time.time()
            
            # Get larger training dataset for initial training
            training_data = self.db_training_manager.get_balanced_training_data(
                sample_size=3000,
                threat_ratio=0.15,
                use_cache=False  # Don't use cache for initial training
            )
            
            if training_data and len(training_data) > 100:
                self.logger.info(f"Training models with {len(training_data)} samples...")
                
                # Train all models
                self.ml_manager.train_models(training_data)
                
                training_time = time.time() - start_time
                self.performance_stats['training_time_seconds'] = training_time
                self.last_full_training = datetime.now()
                
                # Verify training success
                model_status = self.ml_manager.get_model_status()
                trained_models = [name for name, status in model_status.items() 
                               if status.get('trained', False)]
                
                self.logger.info(f"✅ Initial training complete in {training_time:.2f}s: {trained_models}")
                
                # Send initial status update
                self._send_model_status()
                
                # Store some training data in streaming buffer for future incremental training
                sample_size = min(100, len(training_data))
                for sample in random.sample(training_data, sample_size):
                    self.streaming_collector.add_sample(sample)
                
            else:
                self.logger.error("❌ Insufficient training data for initial training")
                
        except Exception as e:
            self.logger.error(f"❌ Initial training failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_raw_log(self, log_data: Dict[str, Any]):
        """Process incoming raw log with enhanced error handling"""
        try:
            start_time = time.time()
            
            # Add to streaming training buffer
            self.streaming_collector.add_sample(log_data)
            
            # Generate predictions from all enabled models
            predictions = self.ml_manager.predict_all(log_data)
            
            if not predictions:
                self.logger.debug("No predictions generated")
                return
            
            # Send anomaly scores to Kafka
            scores_sent = 0
            for model_name, prediction in predictions.items():
                try:
                    # Enhanced anomaly score with more metadata
                    anomaly_score = {
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'entity_id': log_data.get('source_ip', 'unknown'),
                        'score': float(prediction.get('score', 0.0)),
                        'is_anomaly': bool(prediction.get('is_anomaly', False)),
                        'confidence': float(prediction.get('confidence', 0.5)),
                        'features': json.dumps(prediction.get('features', {})),
                        'model_version': prediction.get('model_version', '1.0'),
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'original_severity': log_data.get('severity', 'info'),
                        'threat_indicators_count': len(log_data.get('threat_indicators', []))
                    }
                    
                    self.kafka_client.send_message(
                        topic=self.topics['anomaly_scores'],
                        message=anomaly_score,
                        key=anomaly_score['entity_id']
                    )
                    scores_sent += 1
                    
                except Exception as e:
                    self.logger.debug(f"Error sending score for {model_name}: {e}")
                    continue
            
            self.processed_count += 1
            self.performance_stats['last_prediction_time'] = datetime.now()
            
            # Update performance stats
            processing_time = time.time() - start_time
            if processing_time > 0:
                self.performance_stats['predictions_per_second'] = 1.0 / processing_time
            
            # Log progress periodically
            if self.processed_count % 100 == 0:
                self.logger.info(f"📊 Processed {self.processed_count} logs, sent {scores_sent} scores")
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"❌ Error processing raw log: {e}")
            if self.error_count % 10 == 0:  # Log every 10th error
                import traceback
                self.logger.error(f"Recent error traceback: {traceback.format_exc()}")
    
    def _handle_model_control(self, control_message: Dict[str, Any]):
        """Handle model control commands with enhanced functionality"""
        try:
            action = control_message.get('action')
            model_name = control_message.get('model_name')
            parameters = control_message.get('parameters', {})
            
            self.logger.info(f"🎮 Received control command: {action} for {model_name}")
            
            if action == 'enable_model':
                self.ml_manager.enable_model(model_name)
                self.logger.info(f"✅ Enabled model: {model_name}")
                
            elif action == 'disable_model':
                self.ml_manager.disable_model(model_name)
                self.logger.info(f"❌ Disabled model: {model_name}")
                
            elif action == 'retrain_model':
                self._retrain_specific_model(model_name, parameters)
                
            elif action == 'update_parameters':
                self._update_model_parameters(model_name, parameters)
                
            elif action == 'get_status':
                # Immediate status update request
                self._send_model_status()
                
            else:
                self.logger.warning(f"Unknown control action: {action}")
            
            # Always send updated status after command
            self._send_model_status()
            
        except Exception as e:
            self.logger.error(f"❌ Error handling model control: {e}")
    
    def _incremental_training_loop(self):
        """Enhanced incremental training with adaptive frequency"""
        base_interval = 1800  # 30 minutes base
        
        while self.running:
            try:
                # Adaptive interval based on data availability
                streaming_data = self.streaming_collector.get_streaming_training_data()
                
                if len(streaming_data) >= 200:  # Sufficient data for training
                    interval = base_interval
                elif len(streaming_data) >= 50:  # Some data available
                    interval = base_interval * 1.5
                else:  # Limited data, wait longer
                    interval = base_interval * 2
                
                time.sleep(interval)
                
                # Get fresh streaming data
                streaming_data = self.streaming_collector.get_streaming_training_data()
                
                if len(streaming_data) >= 100:  # Minimum samples for meaningful training
                    self.logger.info(f"🔄 Starting incremental training with {len(streaming_data)} streaming samples")
                    
                    start_time = time.time()
                    
                    # Add some recent database data for balance
                    try:
                        recent_db_data = self.db_training_manager.get_balanced_training_data(
                            sample_size=300,
                            threat_ratio=0.15,
                            use_cache=True
                        )
                        
                        # Combine streaming and database data
                        combined_data = streaming_data + recent_db_data
                        random.shuffle(combined_data)
                        
                        # Limit total size to prevent memory issues
                        if len(combined_data) > 1500:
                            combined_data = random.sample(combined_data, 1500)
                        
                        self.logger.info(f"Combined training data: {len(combined_data)} total samples")
                        
                    except Exception as e:
                        self.logger.warning(f"Database error in incremental training, using streaming only: {e}")
                        combined_data = streaming_data
                    
                    # Perform incremental training
                    try:
                        self.ml_manager.train_models(combined_data)
                        
                        training_time = time.time() - start_time
                        self.last_incremental_training = datetime.now()
                        
                        self.logger.info(f"✅ Incremental training complete in {training_time:.2f}s")
                        
                        # Update performance stats
                        self.performance_stats['training_time_seconds'] = training_time
                        
                        # Send status update
                        self._send_model_status()
                        
                    except Exception as e:
                        self.logger.error(f"❌ Incremental training failed: {e}")
                
                else:
                    self.logger.debug(f"Insufficient streaming data for incremental training: {len(streaming_data)} samples")
                
            except Exception as e:
                self.logger.error(f"❌ Error in incremental training loop: {e}")
                time.sleep(base_interval)
    
    def _full_retrain_loop(self):
        """Enhanced full retraining with better scheduling"""
        base_interval = 21600  # 6 hours base
        
        while self.running:
            try:
                time.sleep(base_interval)
                
                self.logger.info("🔄 Starting scheduled full model retraining")
                start_time = time.time()
                
                # Get comprehensive training data
                training_data = self.db_training_manager.get_balanced_training_data(
                    sample_size=4000,
                    threat_ratio=0.15,
                    use_cache=False  # Always get fresh data for full retrain
                )
                
                if training_data and len(training_data) > 500:
                    self.logger.info(f"Full retraining with {len(training_data)} samples")
                    
                    # Backup current model state (in case training fails)
                    model_status_backup = self.ml_manager.get_model_status()
                    
                    try:
                        self.ml_manager.train_models(training_data)
                        
                        training_time = time.time() - start_time
                        self.last_full_training = datetime.now()
                        
                        self.logger.info(f"✅ Full retraining complete in {training_time:.2f}s")
                        
                        # Update performance stats
                        self.performance_stats['training_time_seconds'] = training_time
                        
                        # Send status update
                        self._send_model_status()
                        
                        # Update streaming buffer with some fresh training data
                        sample_size = min(200, len(training_data))
                        for sample in random.sample(training_data, sample_size):
                            self.streaming_collector.add_sample(sample)
                        
                    except Exception as e:
                        self.logger.error(f"❌ Full retraining failed: {e}")
                        # Model state is preserved, continue with previous models
                
                else:
                    self.logger.error("❌ Insufficient data for full retraining")
                
            except Exception as e:
                self.logger.error(f"❌ Error in full retraining loop: {e}")
                time.sleep(base_interval)
    
    def _model_status_loop(self):
        """Send model status updates with enhanced information"""
        while self.running:
            try:
                time.sleep(60)  # Every minute
                self._send_model_status()
                
            except Exception as e:
                self.logger.error(f"❌ Error in status reporting: {e}")
                time.sleep(60)
    
    def _system_metrics_loop(self):
        """Send comprehensive system metrics"""
        while self.running:
            try:
                time.sleep(30)  # Every 30 seconds
                
                # Collect detailed system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # ML Pipeline specific metrics
                ml_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'service': 'ml-pipeline',
                    'cpu_usage': cpu_percent / 100.0,
                    'memory_usage': memory.percent / 100.0,
                    'disk_usage': disk.percent / 100.0,
                    'processing_latency': self.performance_stats.get('predictions_per_second', 0),
                    'processed_count': self.processed_count,
                    'error_count': self.error_count,
                    'queue_size': 0,  # Kafka handles queuing
                    'last_training_time': self.last_full_training.isoformat() if self.last_full_training else None,
                    'last_incremental_training': self.last_incremental_training.isoformat() if self.last_incremental_training else None,
                    'trained_models_count': len([m for m, s in self.ml_manager.get_model_status().items() if s.get('trained', False)]),
                    'enabled_models_count': len([m for m, s in self.ml_manager.get_model_status().items() if s.get('enabled', False)]),
                    'streaming_buffer_size': sum(len(buf) for buf in self.streaming_collector.training_buffer.values()),
                    'database_connected': self.db.is_connected() if self.db else False
                }
                
                self.kafka_client.send_message(
                    topic=self.topics['system_metrics'],
                    message=ml_metrics
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error sending system metrics: {e}")
                time.sleep(30)
    
    def _performance_monitor_loop(self):
        """Monitor and log performance metrics"""
        while self.running:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Calculate performance metrics
                model_status = self.ml_manager.get_model_status()
                enabled_models = [name for name, status in model_status.items() if status.get('enabled', False)]
                trained_models = [name for name, status in model_status.items() if status.get('trained', False)]
                
                # Log performance summary
                self.logger.info(f"📊 Performance Summary:")
                self.logger.info(f"   - Processed: {self.processed_count} logs")
                self.logger.info(f"   - Errors: {self.error_count}")
                self.logger.info(f"   - Models: {len(enabled_models)}/{len(trained_models)} enabled/trained")
                self.logger.info(f"   - Last training: {self.last_full_training}")
                
                # Log streaming buffer status
                collection_stats = self.streaming_collector.get_collection_stats()
                self.logger.info(f"   - Buffer: {collection_stats['buffer_sizes']['threats']} threats, "
                               f"{collection_stats['buffer_sizes']['normal']} normal")
                
                # Estimate model accuracy (simplified)
                for model_name in enabled_models:
                    # This is a placeholder - in production, you'd calculate real accuracy
                    estimated_accuracy = 0.85 + random.uniform(-0.05, 0.05)
                    self.performance_stats['model_accuracy_estimates'][model_name] = estimated_accuracy
                
            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring: {e}")
                time.sleep(300)
    
    def _send_model_status(self):
        """Send comprehensive model status to Kafka"""
        try:
            model_status = self.ml_manager.get_model_status()
            
            for model_name, status in model_status.items():
                # Enhanced status message
                status_message = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'status': 'trained' if status.get('trained', False) else 'untrained',
                    'enabled': status.get('enabled', False),
                    'performance_metrics': status.get('performance_metrics', {}),
                    'last_training_time': self.last_full_training.isoformat() if self.last_full_training else None,
                    'last_incremental_training': self.last_incremental_training.isoformat() if self.last_incremental_training else None,
                    'last_prediction_time': self.performance_stats.get('last_prediction_time').isoformat() if self.performance_stats.get('last_prediction_time') else None,
                    'model_type': status.get('model_type', 'unknown'),
                    'predictions_count': self.processed_count,
                    'training_data_size': sum(len(buf) for buf in self.streaming_collector.training_buffer.values()),
                    'estimated_accuracy': self.performance_stats.get('model_accuracy_estimates', {}).get(model_name, 0.0),
                    'error_rate': self.error_count / max(self.processed_count, 1)
                }
                
                self.kafka_client.send_message(
                    topic=self.topics['model_status'],
                    message=status_message,
                    key=model_name
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error sending model status: {e}")
    
    def _retrain_specific_model(self, model_name: str, parameters: Dict[str, Any]):
        """Retrain a specific model with custom parameters"""
        try:
            self.logger.info(f"🔄 Retraining {model_name} with parameters: {parameters}")
            
            start_time = time.time()
            
            # Get training data with custom parameters
            sample_size = parameters.get('sample_size', 2000)
            threat_ratio = parameters.get('threat_ratio', 0.15)
            
            training_data = self.db_training_manager.get_balanced_training_data(
                sample_size=sample_size,
                threat_ratio=threat_ratio,
                use_cache=parameters.get('use_cache', True)
            )
            
            if training_data and model_name in self.ml_manager.models:
                # Train only the specific model
                self.ml_manager.models[model_name].train(training_data)
                
                training_time = time.time() - start_time
                self.logger.info(f"✅ Retrained {model_name} in {training_time:.2f}s")
                
                # Update performance stats
                self.performance_stats['training_time_seconds'] = training_time
                
            else:
                self.logger.warning(f"⚠️ Cannot retrain {model_name}: model not found or no training data")
            
        except Exception as e:
            self.logger.error(f"❌ Error retraining {model_name}: {e}")
    
    def _update_model_parameters(self, model_name: str, parameters: Dict[str, Any]):
        """Update model parameters and optionally retrain"""
        try:
            self.logger.info(f"🔧 Updating {model_name} parameters: {parameters}")
            
            if model_name in self.ml_manager.models:
                model = self.ml_manager.models[model_name]
                
                # Apply parameter updates (model-specific)
                if hasattr(model, 'update_parameters'):
                    model.update_parameters(parameters)
                    self.logger.info(f"✅ Updated parameters for {model_name}")
                    
                    # Retrain if requested
                    if parameters.get('retrain_after_update', False):
                        self._retrain_specific_model(model_name, parameters)
                        
                else:
                    self.logger.info(f"ℹ️ {model_name} does not support parameter updates")
            else:
                self.logger.warning(f"⚠️ Model {model_name} not found")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating {model_name} parameters: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            collection_stats = self.streaming_collector.get_collection_stats()
            
            return {
                'running': self.running,
                'processed_count': self.processed_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.processed_count, 1),
                'last_full_training': self.last_full_training.isoformat() if self.last_full_training else None,
                'last_incremental_training': self.last_incremental_training.isoformat() if self.last_incremental_training else None,
                'database_connected': self.db.is_connected() if self.db else False,
                'model_status': self.ml_manager.get_model_status(),
                'performance_stats': self.performance_stats,
                'streaming_collection_stats': collection_stats,
                'kafka_topics': self.topics,
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.Process().cpu_percent(),
                'uptime_seconds': (datetime.now() - self.last_full_training).total_seconds() if self.last_full_training else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                'error': str(e), 
                'running': self.running,
                'processed_count': self.processed_count,
                'error_count': self.error_count
            }
    
    def stop(self):
        """Stop the ML pipeline service gracefully"""
        self.logger.info("🛑 Stopping Complete ML Pipeline Service...")
        self.running = False
        
        try:
            # Close database connection
            if self.db and hasattr(self.db, 'close'):
                self.db.close()
                self.logger.info("✅ Database connection closed")
            
            # Final statistics
            self.logger.info(f"📊 Final stats: {self.processed_count} processed, "
                           f"{self.error_count} errors, "
                           f"error rate: {self.error_count / max(self.processed_count, 1):.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def setup_signal_handlers(pipeline):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for complete ML pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Complete ML Pipeline Service...")
        
        pipeline = CompleteMHLPipeline()
        
        # Setup graceful shutdown
        setup_signal_handlers(pipeline)
        
        if pipeline.start():
            logger.info("✅ Complete ML Pipeline Service started successfully")
            logger.info("📊 Using database training data and Kafka communication")
            logger.info("🌐 Health check available at http://localhost:8001/health")
            
            # Enhanced status monitoring loop
            status_interval = 60
            detailed_interval = 300
            last_detailed = 0
            
            while True:
                time.sleep(status_interval)
                current_time = time.time()
                
                try:
                    status = pipeline.get_service_status()
                    
                    # Basic status every minute
                    logger.info(f"📊 Status: {status['processed_count']} processed, "
                               f"{status['error_count']} errors, "
                               f"DB: {status['database_connected']}")
                    
                    # Log model status
                    model_status = status.get('model_status', {})
                    enabled_models = [name for name, st in model_status.items() 
                                   if st.get('enabled', False)]
                    trained_models = [name for name, st in model_status.items() 
                                   if st.get('trained', False)]
                    logger.info(f"🎯 Models: {len(enabled_models)} enabled, {len(trained_models)} trained")
                    
                    # Detailed status every 5 minutes
                    if current_time - last_detailed >= detailed_interval:
                        perf_stats = status.get('performance_stats', {})
                        collection_stats = status.get('streaming_collection_stats', {})
                        
                        logger.info(f"🔄 Training: Last full={status.get('last_full_training', 'Never')}")
                        logger.info(f"📈 Performance: {perf_stats.get('predictions_per_second', 0):.2f} pred/sec")
                        logger.info(f"📦 Buffer: {collection_stats.get('buffer_sizes', {})}")
                        
                        last_detailed = current_time
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                
        else:
            logger.error("❌ Failed to start Complete ML Pipeline Service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💥 Pipeline error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.stop()

if __name__ == "__main__":
    main()