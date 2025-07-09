#!/usr/bin/env python3
# run_orchestrator.py - IMPROVED VERSION with better error handling and features

import sys
import time
import logging
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque
import signal

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# IMPROVED: Better import handling with specific error messages
try:
    from config.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
    from shared.database import ClickHouseClient
    from shared.monitoring import MetricsCollector
    
    # FIXED: Direct import without circular dependency
    from services.orchestrator import RLOrchestrator
    
except ImportError as e:
    print(f"❌ Critical import error: {e}")
    print("📝 Please ensure all required modules are available")
    sys.exit(1)

# IMPROVED: Enhanced ML model detection
def detect_ml_models():
    """Detect which ML models are available"""
    ml_info = {
        'enhanced_available': False,
        'basic_available': False,
        'ml_manager_class': None,
        'source': None
    }
    
    # Try enhanced models first
    try:
        from ml_models import MLModelManager as EnhancedMLModelManager
        ml_info.update({
            'enhanced_available': True,
            'ml_manager_class': EnhancedMLModelManager,
            'source': 'enhanced'
        })
        print("✅ Using Enhanced ML Models (with adaptive thresholds)")
        return ml_info
    except ImportError:
        print("⚠️ Enhanced ML Models not found, trying basic models...")
    
    # Fallback to basic models
    try:
        from services.ml_models import MLModelManager
        ml_info.update({
            'basic_available': True,
            'ml_manager_class': MLModelManager,
            'source': 'basic'
        })
        print("✅ Using Basic ML Models")
        return ml_info
    except ImportError:
        print("❌ No ML Models available")
        return ml_info

class ImprovedOrchestratorIntegrator:
    """Improved orchestrator integrator with enhanced features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # IMPROVED: Better ML model initialization
        ml_info = detect_ml_models()
        if not ml_info['ml_manager_class']:
            raise ImportError("No ML Model Manager available")
        
        self.using_enhanced_models = ml_info['enhanced_available']
        self.ml_manager = self._initialize_ml_manager(ml_info['ml_manager_class'])
        
        # IMPROVED: Better orchestrator initialization with retry logic
        self.orchestrator = self._initialize_orchestrator_with_retry()
        
        # IMPROVED: Enhanced component initialization
        self.kafka_client = self._initialize_kafka()
        self.db = self._initialize_database()
        self.metrics = self._initialize_metrics()
        
        # IMPROVED: Enhanced state tracking
        self.running = False
        self.start_time = None
        self.system_metrics = {}
        self.performance_data = deque(maxlen=200)  # Increased buffer
        self.health_checks = {
            'orchestrator': True,
            'ml_manager': True,
            'database': self.db is not None,
            'kafka': self.kafka_client is not None
        }
        
        # IMPROVED: Performance tracking
        self.stats = {
            'decisions_made': 0,
            'model_switches': 0,
            'last_decision_time': None,
            'uptime_seconds': 0
        }
    
    def _initialize_ml_manager(self, ml_manager_class):
        """Initialize ML manager with enhanced configuration"""
        try:
            if self.using_enhanced_models:
                # Enhanced configuration for better performance
                enhanced_config = {
                    'ml_models': {
                        'isolation_forest': {
                            'enabled': True,
                            'contamination': 0.06,  # Slightly reduced for better balance
                            'n_estimators': 150,    # Reduced for faster training
                            'max_samples': 256
                        },
                        'clustering': {
                            'enabled': True,
                            'n_clusters': 6,        # Reduced for faster processing
                            'max_iter': 300         # Reduced for faster convergence
                        },
                        'forbidden_ratio': {
                            'enabled': True,
                            'window_size': 150      # Reduced window for faster response
                        }
                    }
                }
                ml_manager = ml_manager_class(enhanced_config)
                self.logger.info("✅ Initialized Enhanced ML Model Manager")
            else:
                # Use configuration from file
                ml_manager = ml_manager_class(self.config._config)
                self.logger.info("✅ Initialized Basic ML Model Manager")
            
            return ml_manager
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ML Manager: {e}")
            raise
    
    def _initialize_orchestrator_with_retry(self, max_retries=3):
        """Initialize orchestrator with retry logic"""
        for attempt in range(max_retries):
            try:
                orchestrator_config = self.config.get_section('orchestrator')
                orchestrator = RLOrchestrator(self.ml_manager, orchestrator_config)
                self.logger.info(f"✅ RL Orchestrator initialized successfully (attempt {attempt + 1})")
                return orchestrator
                
            except Exception as e:
                self.logger.warning(f"⚠️ Orchestrator init attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("❌ All orchestrator initialization attempts failed")
                    return self._create_enhanced_fallback_orchestrator()
                time.sleep(2)  # Wait before retry
    
    def _create_enhanced_fallback_orchestrator(self):
        """Create an enhanced fallback orchestrator"""
        class EnhancedFallbackOrchestrator:
            def __init__(self, ml_manager):
                self.logger = logging.getLogger("EnhancedFallbackOrchestrator")
                self.ml_manager = ml_manager
                self.running = False
                self.decision_count = 0
                self.last_policy_change = datetime.now()
                
                # Smart default policy based on model capabilities
                self.policy = self._create_smart_default_policy()
                
            def _create_smart_default_policy(self):
                """Create intelligent default policy"""
                models = list(self.ml_manager.models.keys())
                # Enable all models initially, but could be made smarter
                return {model: True for model in models}
                
            def start(self):
                self.running = True
                self.logger.info("✅ Enhanced fallback orchestrator started")
                
            def stop(self):
                self.running = False
                self.logger.info("🛑 Enhanced fallback orchestrator stopped")
                
            def get_current_policy(self):
                return self.policy.copy()
                
            def get_performance_summary(self):
                return {
                    'orchestrator_type': 'Enhanced_Fallback',
                    'total_decisions': self.decision_count,
                    'recent_avg_reward': 0.5,  # Neutral reward
                    'exploration_rate': 0.0,
                    'learning_enabled': False,
                    'last_policy_change': self.last_policy_change
                }
                
            def update_system_metrics(self, metrics):
                # Could implement adaptive policy based on system load
                cpu_usage = metrics.get('cpu_usage', 0)
                if cpu_usage > 0.8:  # High CPU usage
                    # Disable some models to reduce load
                    models = list(self.policy.keys())
                    if len(models) > 1 and all(self.policy.values()):
                        self.policy[models[-1]] = False  # Disable last model
                        self.last_policy_change = datetime.now()
                        self.logger.info("🔧 Reduced model load due to high CPU usage")
                
            def add_performance_feedback(self, performance):
                self.decision_count += 1
        
        self.logger.warning("🔄 Using enhanced fallback orchestrator")
        return EnhancedFallbackOrchestrator(self.ml_manager)
    
    def _initialize_kafka(self) -> Optional[KafkaClient]:
        """Initialize Kafka with better error handling"""
        try:
            kafka_client = KafkaClient(self.config.get('kafka.broker'))
            self.logger.info("✅ Kafka client initialized")
            return kafka_client
        except Exception as e:
            self.logger.warning(f"⚠️ Kafka not available: {e}")
            return None
    
    def _initialize_database(self) -> Optional[ClickHouseClient]:
        """Initialize database with better error handling"""
        try:
            db = ClickHouseClient(
                host=self.config.get('clickhouse.host'),
                port=9000,  # Native port
                user=self.config.get('clickhouse.user'),
                password=self.config.get('clickhouse.password'),
                database=self.config.get('clickhouse.database')
            )
            self.logger.info("✅ Database client initialized")
            return db
        except Exception as e:
            self.logger.warning(f"⚠️ Database not available: {e}")
            return None
    
    def _initialize_metrics(self) -> Optional[MetricsCollector]:
        """Initialize metrics collector"""
        try:
            metrics = MetricsCollector()
            self.logger.info("✅ Metrics collector initialized")
            return metrics
        except Exception as e:
            self.logger.warning(f"⚠️ Metrics collector not available: {e}")
            return None
    
    def start(self):
        """Start the improved orchestrator with better initialization"""
        self.logger.info("🧠 Starting Improved RL Orchestrator Integration...")
        self.start_time = datetime.now()
        
        # IMPROVED: Better database connection handling
        if self.db:
            try:
                self.db.connect()
                if self.db.is_connected():
                    self.logger.info("✅ Connected to ClickHouse")
                    self.health_checks['database'] = True
                else:
                    self.logger.warning("⚠️ ClickHouse connection failed")
                    self.health_checks['database'] = False
                    self.db = None
            except Exception as e:
                self.logger.warning(f"⚠️ Database connection error: {e}")
                self.health_checks['database'] = False
                self.db = None
        
        # IMPROVED: Enhanced model training
        if self.using_enhanced_models:
            self._train_enhanced_models_improved()
        
        # Start orchestrator
        try:
            self.orchestrator.start()
            self.logger.info("✅ Orchestrator core started")
            self.health_checks['orchestrator'] = True
        except Exception as e:
            self.logger.error(f"❌ Failed to start orchestrator: {e}")
            self.health_checks['orchestrator'] = False
            return False
        
        self.running = True
        
        # IMPROVED: Start enhanced threads with better error handling
        self._start_enhanced_threads()
        
        orchestrator_type = "Enhanced RL" if self.using_enhanced_models else "Basic RL"
        self.logger.info(f"🚀 {orchestrator_type} Orchestrator Integration started successfully")
        return True
    
    def _train_enhanced_models_improved(self):
        """Improved model training with better data and error handling"""
        self.logger.info("🤖 Training enhanced models with improved sample data...")
        
        try:
            sample_data = []
            
            # IMPROVED: More realistic training data
            # Normal traffic (90%)
            for i in range(180):
                sample_data.append({
                    'timestamp': datetime.now(),
                    'event_type': 'web_request',
                    'source_ip': f'192.168.{i % 10 + 1}.{i % 50 + 1}',
                    'destination_ip': f'10.0.{i % 5}.1',
                    'port': [80, 443, 8080, 8443][i % 4],
                    'protocol': ['http', 'https'][i % 2],
                    'severity': 'info',
                    'threat_indicators': [],
                    'bytes_transferred': 1024 + (i * 10) % 10000,
                    'duration_seconds': 1 + i % 10
                })
            
            # Threat samples with variety (10%)
            threat_types = [
                ('brute_force_ssh', 22, 'ssh', 'high', ['multiple_failed_logins', 'suspicious_source']),
                ('port_scan', 80, 'tcp', 'medium', ['port_scanning', 'reconnaissance']),
                ('data_exfiltration', 443, 'https', 'critical', ['large_data_transfer', 'unusual_destination']),
                ('malware_c2', 8080, 'http', 'high', ['c2_communication', 'malware_signature'])
            ]
            
            for i in range(20):
                threat_type, port, protocol, severity, indicators = threat_types[i % len(threat_types)]
                sample_data.append({
                    'timestamp': datetime.now(),
                    'event_type': threat_type,
                    'source_ip': f'10.10.{i % 10 + 1}.{i % 20 + 1}',
                    'destination_ip': '192.168.1.100',
                    'port': port,
                    'protocol': protocol,
                    'severity': severity,
                    'threat_indicators': indicators,
                    'bytes_transferred': 512 + i * 100,
                    'duration_seconds': 5 + i % 60
                })
            
            # Train models
            self.ml_manager.train_models(sample_data)
            self.logger.info(f"✅ Enhanced models trained with {len(sample_data)} samples")
            
            # Verify model training
            model_status = self.ml_manager.get_model_status()
            trained_models = [name for name, status in model_status.items() 
                            if status.get('trained', False)]
            self.logger.info(f"📊 Successfully trained models: {trained_models}")
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced model training failed: {e}")
    
    def _start_enhanced_threads(self):
        """Start enhanced monitoring threads with better error handling"""
        threads = [
            ("System Metrics", self._enhanced_system_metrics_loop),
            ("Performance Data", self._enhanced_performance_loop),
            ("Model Control", self._enhanced_model_control_loop),
            ("Health Monitor", self._health_monitor_loop)
        ]
        
        for thread_name, target_function in threads:
            try:
                thread = threading.Thread(target=target_function, daemon=True, name=thread_name)
                thread.start()
                self.logger.info(f"✅ Started {thread_name} thread")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {thread_name} thread: {e}")
    
    def _enhanced_system_metrics_loop(self):
        """Enhanced system metrics collection with better data"""
        while self.running:
            try:
                # Collect comprehensive system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # IMPROVED: More detailed metrics
                self.system_metrics = {
                    'cpu_usage': cpu_percent / 100.0,
                    'memory_usage': memory.percent / 100.0,
                    'disk_usage': disk.percent / 100.0,
                    'processing_latency': 0.1,  # Placeholder
                    'timestamp': datetime.now(),
                    'cpu_count': psutil.cpu_count(),
                    'memory_available_gb': memory.available / (1024**3),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                }
                
                # Update orchestrator
                try:
                    self.orchestrator.update_system_metrics(self.system_metrics)
                except Exception as e:
                    self.logger.debug(f"Error updating system metrics: {e}")
                
                # IMPROVED: Adaptive logging based on resource usage
                if cpu_percent > 80 or memory.percent > 90:
                    self.logger.warning(f"⚠️ High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                else:
                    self.logger.debug(f"📊 System: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%")
                
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"❌ Error in system metrics collection: {e}")
                time.sleep(10)
    
    def _enhanced_performance_loop(self):
        """Enhanced performance data collection"""
        while self.running:
            try:
                model_status = self.ml_manager.get_model_status()
                
                for model_name, status in model_status.items():
                    if status.get('enabled', False):
                        # IMPROVED: More realistic performance simulation
                        base_performance = {
                            'isolation_forest': {'precision': 0.87, 'recall': 0.82, 'f1_score': 0.84},
                            'clustering': {'precision': 0.81, 'recall': 0.79, 'f1_score': 0.80},
                            'forbidden_ratio': {'precision': 0.75, 'recall': 0.88, 'f1_score': 0.81}
                        }
                        
                        base_metrics = base_performance.get(model_name, 
                                                          {'precision': 0.80, 'recall': 0.80, 'f1_score': 0.80})
                        
                        # Add some realistic variance
                        performance = {
                            'model_name': model_name,
                            'precision': max(0.5, min(1.0, base_metrics['precision'] + (hash(str(datetime.now())) % 10 - 5) / 100)),
                            'recall': max(0.5, min(1.0, base_metrics['recall'] + (hash(str(datetime.now())) % 10 - 5) / 100)),
                            'f1_score': max(0.5, min(1.0, base_metrics['f1_score'] + (hash(str(datetime.now())) % 10 - 5) / 100)),
                            'processing_time': 0.05 + (hash(model_name) % 10) / 200,
                            'timestamp': datetime.now()
                        }
                        
                        # Update F1 score based on precision and recall
                        if performance['precision'] + performance['recall'] > 0:
                            performance['f1_score'] = 2 * (performance['precision'] * performance['recall']) / \
                                                    (performance['precision'] + performance['recall'])
                        
                        try:
                            self.orchestrator.add_performance_feedback(performance)
                            self.performance_data.append(performance)
                        except Exception as e:
                            self.logger.debug(f"Error adding performance feedback: {e}")
                
                self.logger.debug(f"📈 Updated performance for {len(model_status)} models")
                time.sleep(45)  # Reduced frequency for better performance
                
            except Exception as e:
                self.logger.error(f"❌ Error in performance collection: {e}")
                time.sleep(45)
    
    def _enhanced_model_control_loop(self):
        """Enhanced model control with better policy tracking"""
        previous_policy = None
        
        while self.running:
            try:
                current_policy = self.orchestrator.get_current_policy()
                
                if current_policy is None:
                    self.logger.debug("⚠️ No policy available yet")
                    time.sleep(30)
                    continue
                
                # IMPROVED: Track policy changes
                if previous_policy != current_policy:
                    self.stats['model_switches'] += 1
                    self.logger.info(f"🔄 Policy changed: {previous_policy} → {current_policy}")
                
                # Apply policy
                if isinstance(current_policy, dict):
                    for model_name, should_enable in current_policy.items():
                        try:
                            current_status = self.ml_manager.get_model_status().get(model_name, {}).get('enabled', False)
                            
                            if should_enable and not current_status:
                                self.ml_manager.enable_model(model_name)
                                self.logger.debug(f"✅ Enabled {model_name}")
                            elif not should_enable and current_status:
                                self.ml_manager.disable_model(model_name)
                                self.logger.debug(f"❌ Disabled {model_name}")
                                
                        except Exception as e:
                            self.logger.debug(f"Error controlling model {model_name}: {e}")
                    
                    active_count = sum(1 for enabled in current_policy.values() if enabled)
                    total_count = len(current_policy)
                    self.logger.debug(f"🎮 Policy: {active_count}/{total_count} models active")
                
                previous_policy = current_policy.copy() if isinstance(current_policy, dict) else current_policy
                self.stats['decisions_made'] += 1
                self.stats['last_decision_time'] = datetime.now()
                
                time.sleep(25)  # Slightly faster decision cycle
                
            except Exception as e:
                self.logger.error(f"❌ Error in model control: {e}")
                time.sleep(25)
    
    def _health_monitor_loop(self):
        """Monitor system health and component status"""
        while self.running:
            try:
                # Check component health
                self.health_checks['orchestrator'] = hasattr(self.orchestrator, 'running') and \
                                                   getattr(self.orchestrator, 'running', False)
                
                if self.db:
                    self.health_checks['database'] = self.db.is_connected()
                
                # Log health status periodically
                healthy_components = sum(self.health_checks.values())
                total_components = len(self.health_checks)
                
                if healthy_components == total_components:
                    self.logger.debug(f"💚 All components healthy ({healthy_components}/{total_components})")
                else:
                    unhealthy = [name for name, status in self.health_checks.items() if not status]
                    self.logger.warning(f"⚠️ Unhealthy components: {unhealthy}")
                
                time.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in health monitoring: {e}")
                time.sleep(120)
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status with enhanced metrics"""
        try:
            base_status = self.orchestrator.get_performance_summary()
            
            enhanced_status = {
                **base_status,
                'system_metrics': self.system_metrics,
                'health_checks': self.health_checks,
                'component_status': {
                    'database_connected': self.db.is_connected() if self.db else False,
                    'kafka_available': self.kafka_client is not None,
                    'metrics_available': self.metrics is not None
                },
                'ml_models_status': self.ml_manager.get_model_status(),
                'current_policy': self.orchestrator.get_current_policy(),
                'runtime_stats': {
                    **self.stats,
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                    'uptime_formatted': str(datetime.now() - self.start_time).split('.')[0] if self.start_time else "0:00:00"
                },
                'running': self.running,
                'using_enhanced_models': self.using_enhanced_models,
                'start_time': self.start_time.isoformat() if self.start_time else None
            }
            
            return enhanced_status
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced status: {e}")
            return {
                'error': str(e), 
                'running': self.running, 
                'using_enhanced_models': self.using_enhanced_models,
                'health_checks': self.health_checks
            }
    
    def stop(self):
        """Improved shutdown with cleanup"""
        self.logger.info("🛑 Starting graceful shutdown...")
        self.running = False
        
        try:
            # Stop orchestrator
            if hasattr(self.orchestrator, 'stop'):
                self.orchestrator.stop()
                self.logger.info("✅ Orchestrator stopped")
            
            # Close database connection
            if self.db and hasattr(self.db, 'close'):
                try:
                    self.db.close()
                    self.logger.info("✅ Database connection closed")
                except:
                    pass
            
            # Final stats
            if self.start_time:
                uptime = datetime.now() - self.start_time
                self.logger.info(f"📊 Final stats: {self.stats['decisions_made']} decisions, "
                               f"{self.stats['model_switches']} policy changes, "
                               f"uptime: {str(uptime).split('.')[0]}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

def setup_signal_handlers(integrator):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        integrator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Improved main entry point with better error handling"""
    # IMPROVED: Enhanced logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Could add file handler here if needed
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧠 Starting IMPROVED RL Orchestrator...")
        logger.info(f"🐍 Python version: {sys.version}")
        logger.info(f"💻 Platform: {sys.platform}")
        
        # Initialize integrator
        integrator = ImprovedOrchestratorIntegrator()
        
        # Setup graceful shutdown
        setup_signal_handlers(integrator)
        
        if integrator.start():
            orchestrator_type = "Enhanced" if integrator.using_enhanced_models else "Basic"
            logger.info(f"✅ IMPROVED RL Orchestrator started successfully")
            logger.info(f"🚀 Using {orchestrator_type} ML Models")
            logger.info("🤖 Learning optimal model configurations...")
            
            # Enhanced monitoring loop
            status_interval = 60
            detailed_status_interval = 300  # 5 minutes
            last_detailed_status = 0
            
            while True:
                time.sleep(status_interval)
                current_time = time.time()
                
                try:
                    status = integrator.get_enhanced_status()
                    
                    # Basic status every minute
                    current_policy = status.get('current_policy', {})
                    if isinstance(current_policy, dict):
                        active_models = sum(1 for enabled in current_policy.values() if enabled)
                        total_models = len(current_policy)
                        policy_str = f"{active_models}/{total_models}"
                    else:
                        policy_str = "unknown"
                    
                    cpu_usage = status.get('system_metrics', {}).get('cpu_usage', 0) * 100
                    uptime = status.get('runtime_stats', {}).get('uptime_formatted', '0:00:00')
                    
                    logger.info(f"🧠 Status: {policy_str} models active, "
                               f"CPU: {cpu_usage:.1f}%, Uptime: {uptime}")
                    
                    # Detailed status every 5 minutes
                    if current_time - last_detailed_status >= detailed_status_interval:
                        decisions = status.get('runtime_stats', {}).get('decisions_made', 0)
                        switches = status.get('runtime_stats', {}).get('model_switches', 0)
                        
                        if 'recent_avg_reward' in status:
                            logger.info(f"🎯 Performance: {decisions} decisions, "
                                       f"{switches} policy changes, "
                                       f"avg reward: {status['recent_avg_reward']:.3f}")
                        
                        # Health check summary
                        health_checks = status.get('health_checks', {})
                        healthy = sum(health_checks.values())
                        total = len(health_checks)
                        logger.info(f"💚 Health: {healthy}/{total} components healthy")
                        
                        last_detailed_status = current_time
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
        else:
            logger.error("❌ Failed to start IMPROVED RL Orchestrator")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if 'integrator' in locals():
            integrator.stop()

if __name__ == "__main__":
    main()