#!/usr/bin/env python3
# orchestrator_separated.py - Orchestrator as separate service communicating via Kafka

import sys
import time
import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from shared.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
    from orchestrator import RLOrchestrator 
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class KafkaModelManager:
    """Virtual model manager that communicates with ML Pipeline via Kafka"""
    
    def __init__(self, kafka_client: KafkaClient, topics: Dict[str, str]):
        self.kafka_client = kafka_client
        self.topics = topics
        self.logger = logging.getLogger(__name__)
        
        # Track model status from Kafka messages
        self.model_status = {}
        self.performance_data = deque(maxlen=100)
        
        # Setup consumer for model status
        self.kafka_client.create_consumer(
            topics=[topics['model_status']],
            group_id="orchestrator-model-status-consumer",
            message_handler=self._handle_model_status
        )
    
    def _handle_model_status(self, status_message: Dict[str, Any]):
        """Handle model status updates from ML Pipeline"""
        try:
            model_name = status_message.get('model_name')
            if model_name:
                self.model_status[model_name] = {
                    'enabled': status_message.get('enabled', False),
                    'trained': status_message.get('status') == 'trained',
                    'performance_metrics': status_message.get('performance_metrics', {}),
                    'last_update': datetime.now(),
                    'model_type': status_message.get('model_type', 'unknown')
                }
                
                # Store performance data
                if status_message.get('performance_metrics'):
                    self.performance_data.append({
                        'model_name': model_name,
                        'timestamp': datetime.now(),
                        **status_message.get('performance_metrics', {})
                    })
                
                self.logger.debug(f"Updated status for {model_name}: {self.model_status[model_name]}")
                
        except Exception as e:
            self.logger.error(f"Error handling model status: {e}")
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current model status (simulates direct model manager interface)"""
        return self.model_status.copy()
    
    def enable_model(self, model_name: str):
        """Send command to enable model"""
        try:
            command = {
                'timestamp': datetime.now().isoformat(),
                'action': 'enable_model',
                'model_name': model_name,
                'parameters': {}
            }
            
            self.kafka_client.send_message(
                topic=self.topics['model_control'],
                message=command,
                key=model_name
            )
            
            self.logger.info(f"Sent enable command for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error enabling model {model_name}: {e}")
    
    def disable_model(self, model_name: str):
        """Send command to disable model"""
        try:
            command = {
                'timestamp': datetime.now().isoformat(),
                'action': 'disable_model',
                'model_name': model_name,
                'parameters': {}
            }
            
            self.kafka_client.send_message(
                topic=self.topics['model_control'],
                message=command,
                key=model_name
            )
            
            self.logger.info(f"Sent disable command for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error disabling model {model_name}: {e}")
    
    def retrain_model(self, model_name: str, parameters: Dict[str, Any] = None):
        """Send command to retrain specific model"""
        try:
            command = {
                'timestamp': datetime.now().isoformat(),
                'action': 'retrain_model',
                'model_name': model_name,
                'parameters': parameters or {}
            }
            
            self.kafka_client.send_message(
                topic=self.topics['model_control'],
                message=command,
                key=model_name
            )
            
            self.logger.info(f"Sent retrain command for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error retraining model {model_name}: {e}")
    
    def update_model_parameters(self, model_name: str, parameters: Dict[str, Any]):
        """Send command to update model parameters"""
        try:
            command = {
                'timestamp': datetime.now().isoformat(),
                'action': 'update_parameters',
                'model_name': model_name,
                'parameters': parameters
            }
            
            self.kafka_client.send_message(
                topic=self.topics['model_control'],
                message=command,
                key=model_name
            )
            
            self.logger.info(f"Sent parameter update for {model_name}: {parameters}")
            
        except Exception as e:
            self.logger.error(f"Error updating parameters for {model_name}: {e}")
    
    @property
    def models(self) -> Dict[str, Any]:
        """Simulate models property for compatibility"""
        return {name: {'enabled': status.get('enabled', False)} 
                for name, status in self.model_status.items()}

class SystemMetricsCollector:
    """Collects system metrics from various services via Kafka"""
    
    def __init__(self, kafka_client: KafkaClient, topics: Dict[str, str]):
        self.kafka_client = kafka_client
        self.topics = topics
        self.logger = logging.getLogger(__name__)
        
        # Track metrics from different services
        self.service_metrics = {}
        self.metrics_history = deque(maxlen=200)
        
        # Setup consumer for system metrics
        self.kafka_client.create_consumer(
            topics=[topics['system_metrics']],
            group_id="orchestrator-metrics-consumer",
            message_handler=self._handle_system_metrics
        )
    
    def _handle_system_metrics(self, metrics_message: Dict[str, Any]):
        """Handle system metrics from various services"""
        try:
            service_name = metrics_message.get('service', 'unknown')
            
            self.service_metrics[service_name] = {
                'cpu_usage': metrics_message.get('cpu_usage', 0.0),
                'memory_usage': metrics_message.get('memory_usage', 0.0),
                'processing_latency': metrics_message.get('processing_latency', 0.0),
                'last_update': datetime.now(),
                'additional_metrics': {
                    k: v for k, v in metrics_message.items() 
                    if k not in ['service', 'cpu_usage', 'memory_usage', 'processing_latency', 'timestamp']
                }
            }
            
            # Store in history
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'service': service_name,
                **metrics_message
            })
            
            self.logger.debug(f"Updated metrics for {service_name}")
            
        except Exception as e:
            self.logger.error(f"Error handling system metrics: {e}")
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated system metrics"""
        try:
            if not self.service_metrics:
                return {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'processing_latency': 0.0
                }
            
            # Calculate averages across services
            cpu_values = [metrics.get('cpu_usage', 0.0) for metrics in self.service_metrics.values()]
            memory_values = [metrics.get('memory_usage', 0.0) for metrics in self.service_metrics.values()]
            latency_values = [metrics.get('processing_latency', 0.0) for metrics in self.service_metrics.values()]
            
            return {
                'cpu_usage': np.mean(cpu_values) if cpu_values else 0.0,
                'memory_usage': np.mean(memory_values) if memory_values else 0.0,
                'processing_latency': np.mean(latency_values) if latency_values else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting aggregated metrics: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'processing_latency': 0.0}

class PerformanceDataCollector:
    """Collects performance feedback from scoring and alerting services"""
    
    def __init__(self, kafka_client: KafkaClient, topics: Dict[str, str]):
        self.kafka_client = kafka_client
        self.topics = topics
        self.logger = logging.getLogger(__name__)
        
        # Track performance data
        self.performance_data = deque(maxlen=500)
        self.alert_feedback = deque(maxlen=200)
        
        # Setup consumers for performance feedback
        if 'anomaly_scores' in topics:
            self.kafka_client.create_consumer(
                topics=[topics['anomaly_scores']],
                group_id="orchestrator-performance-consumer",
                message_handler=self._handle_performance_data
            )
        
        if 'alerts' in topics:
            self.kafka_client.create_consumer(
                topics=[topics['alerts']],
                group_id="orchestrator-alerts-consumer", 
                message_handler=self._handle_alert_feedback
            )
    
    def _handle_performance_data(self, score_message: Dict[str, Any]):
        """Handle performance data from scoring"""
        try:
            # Extract performance indicators from anomaly scores
            performance_data = {
                'timestamp': datetime.now(),
                'model_name': score_message.get('model_name'),
                'score': score_message.get('score', 0.0),
                'is_anomaly': score_message.get('is_anomaly', False),
                'entity_id': score_message.get('entity_id')
            }
            
            self.performance_data.append(performance_data)
            
        except Exception as e:
            self.logger.debug(f"Error handling performance data: {e}")
    
    def _handle_alert_feedback(self, alert_message: Dict[str, Any]):
        """Handle alert feedback"""
        try:
            alert_data = {
                'timestamp': datetime.now(),
                'severity': alert_message.get('severity'),
                'entity_id': alert_message.get('entity_id'),
                'verified': alert_message.get('verified', True),  # Assume verified unless marked otherwise
                'contributing_models': alert_message.get('contributing_models', [])
            }
            
            self.alert_feedback.append(alert_data)
            
        except Exception as e:
            self.logger.debug(f"Error handling alert feedback: {e}")
    
    def get_recent_performance(self, window_minutes: int = 30) -> List[Dict[str, Any]]:
        """Get recent performance data"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [data for data in self.performance_data if data['timestamp'] > cutoff_time]
    
    def get_recent_alerts(self, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alert feedback"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [data for data in self.alert_feedback if data['timestamp'] > cutoff_time]

class SeparatedOrchestratorService:
    """Orchestrator service that communicates with ML Pipeline via Kafka"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # Initialize Kafka
        self.kafka_client = KafkaClient(self.config.get('kafka.broker'))
        
        # Kafka topics
        self.topics = {
            'model_control': self.config.get('kafka.topics.model_control', 'model-control'),
            'model_status': self.config.get('kafka.topics.model_status', 'model-status'),
            'system_metrics': self.config.get('kafka.topics.system_metrics', 'system-metrics'),
            'anomaly_scores': self.config.get('kafka.topics.anomaly_scores', 'anomaly-scores'),
            'alerts': self.config.get('kafka.topics.alerts', 'alerts')
        }
        
        # Initialize virtual model manager (communicates via Kafka)
        self.model_manager = KafkaModelManager(self.kafka_client, self.topics)
        
        # Initialize metrics and performance collectors
        self.metrics_collector = SystemMetricsCollector(self.kafka_client, self.topics)
        self.performance_collector = PerformanceDataCollector(self.kafka_client, self.topics)
        
        # Initialize orchestrator with virtual model manager
        orchestrator_config = self.config.get_section('orchestrator')
        self.orchestrator = RLOrchestrator(self.model_manager, orchestrator_config)
        
        # Service state
        self.running = False
        self.start_time = None
        self.decision_count = 0
        
        # Performance tracking
        self.orchestrator_metrics = {
            'decisions_made': 0,
            'commands_sent': 0,
            'last_decision_time': None
        }
    
    def start(self):
        """Start the separated orchestrator service"""
        self.logger.info("🧠 Starting Separated Orchestrator Service...")
        self.start_time = datetime.now()
        
        try:
            # Start orchestrator core
            self.orchestrator.start()
            self.logger.info("✅ Orchestrator core started")
            
            self.running = True
            
            # Start background threads
            self._start_background_threads()
            
            # Wait for initial model status
            self._wait_for_model_status()
            
            self.logger.info("🚀 Separated Orchestrator Service is running!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start orchestrator service: {e}")
            return False
    
    def _start_background_threads(self):
        """Start background processing threads"""
        threads = [
            ("Decision Engine", self._decision_engine_loop),
            ("Performance Monitor", self._performance_monitor_loop),
            ("System Health Monitor", self._health_monitor_loop),
            ("Metrics Reporter", self._metrics_reporter_loop)
        ]
        
        for thread_name, target_function in threads:
            try:
                thread = threading.Thread(target=target_function, daemon=True, name=thread_name)
                thread.start()
                self.logger.info(f"✅ Started {thread_name} thread")
            except Exception as e:
                self.logger.error(f"❌ Failed to start {thread_name} thread: {e}")
    
    def _wait_for_model_status(self, timeout_seconds: int = 30):
        """Wait for initial model status from ML Pipeline"""
        self.logger.info("⏳ Waiting for model status from ML Pipeline...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.model_manager.model_status:
                models = list(self.model_manager.model_status.keys())
                self.logger.info(f"✅ Received status for models: {models}")
                return
            time.sleep(2)
        
        self.logger.warning("⚠️ Timeout waiting for model status, proceeding anyway")
    
    def _decision_engine_loop(self):
        """Main decision-making loop"""
        decision_interval = self.config.get('orchestrator.decision_interval', 45)
        
        while self.running:
            try:
                time.sleep(decision_interval)
                
                # Update orchestrator with current metrics
                aggregated_metrics = self.metrics_collector.get_aggregated_metrics()
                self.orchestrator.update_system_metrics(aggregated_metrics)
                
                # Add performance feedback
                recent_performance = self.performance_collector.get_recent_performance()
                for perf_data in recent_performance[-10:]:  # Last 10 samples
                    self.orchestrator.add_performance_feedback(perf_data)
                
                # Add alert feedback
                recent_alerts = self.performance_collector.get_recent_alerts()
                for alert_data in recent_alerts[-5:]:  # Last 5 alerts
                    self.orchestrator.add_alert_feedback(alert_data)
                
                # Get current policy
                current_policy = self.orchestrator.get_current_policy()
                
                if current_policy:
                    # Apply policy by sending commands to ML Pipeline
                    self._apply_policy(current_policy)
                    
                    self.decision_count += 1
                    self.orchestrator_metrics['decisions_made'] = self.decision_count
                    self.orchestrator_metrics['last_decision_time'] = datetime.now()
                    
                    active_models = sum(1 for enabled in current_policy.values() if enabled)
                    total_models = len(current_policy)
                    
                    self.logger.info(f"🎯 Decision #{self.decision_count}: {active_models}/{total_models} models active")
                
            except Exception as e:
                self.logger.error(f"❌ Error in decision engine: {e}")
                time.sleep(decision_interval)
    
    def _apply_policy(self, policy: Dict[str, bool]):
        """Apply orchestrator policy by sending commands to ML Pipeline"""
        try:
            current_status = self.model_manager.get_model_status()
            commands_sent = 0
            
            for model_name, should_enable in policy.items():
                current_enabled = current_status.get(model_name, {}).get('enabled', False)
                
                if should_enable and not current_enabled:
                    self.model_manager.enable_model(model_name)
                    commands_sent += 1
                elif not should_enable and current_enabled:
                    self.model_manager.disable_model(model_name)
                    commands_sent += 1
            
            if commands_sent > 0:
                self.orchestrator_metrics['commands_sent'] += commands_sent
                self.logger.info(f"📤 Sent {commands_sent} model control commands")
            
        except Exception as e:
            self.logger.error(f"❌ Error applying policy: {e}")
    
    def _performance_monitor_loop(self):
        """Monitor performance and trigger retraining if needed"""
        while self.running:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Analyze recent performance
                recent_performance = self.performance_collector.get_recent_performance()
                
                if len(recent_performance) >= 50:  # Enough data for analysis
                    # Calculate performance metrics by model
                    model_performance = {}
                    
                    for perf_data in recent_performance:
                        model_name = perf_data.get('model_name')
                        if model_name:
                            if model_name not in model_performance:
                                model_performance[model_name] = []
                            model_performance[model_name].append(perf_data)
                    
                    # Check for models that might need retraining
                    for model_name, perf_list in model_performance.items():
                        if len(perf_list) >= 20:  # Enough samples
                            # Calculate score variance (high variance might indicate drift)
                            scores = [p.get('score', 0.0) for p in perf_list]
                            score_variance = np.var(scores) if scores else 0.0
                            
                            # Trigger retraining if high variance detected
                            if score_variance > 0.1:  # Threshold for variance
                                self.logger.info(f"🔄 Triggering retrain for {model_name} (high variance: {score_variance:.3f})")
                                self.model_manager.retrain_model(model_name, {'reason': 'high_variance'})
                
            except Exception as e:
                self.logger.error(f"❌ Error in performance monitoring: {e}")
                time.sleep(300)
    
    def _health_monitor_loop(self):
        """Monitor system health and service status"""
        while self.running:
            try:
                time.sleep(120)  # Every 2 minutes
                
                # Check service health
                current_time = datetime.now()
                
                # Check if we're receiving metrics
                recent_metrics = any(
                    (current_time - metrics.get('last_update', current_time)).total_seconds() < 300
                    for metrics in self.metrics_collector.service_metrics.values()
                )
                
                # Check if we're receiving model status
                recent_model_updates = any(
                    (current_time - status.get('last_update', current_time)).total_seconds() < 300
                    for status in self.model_manager.model_status.values()
                )
                
                # Log health status
                health_status = {
                    'receiving_metrics': recent_metrics,
                    'receiving_model_status': recent_model_updates,
                    'active_services': list(self.metrics_collector.service_metrics.keys()),
                    'known_models': list(self.model_manager.model_status.keys())
                }
                
                self.logger.info(f"💚 Health: Metrics={recent_metrics}, Models={recent_model_updates}")
                self.logger.debug(f"Services: {health_status['active_services']}")
                
            except Exception as e:
                self.logger.error(f"❌ Error in health monitoring: {e}")
                time.sleep(120)
    
    def _metrics_reporter_loop(self):
        """Report orchestrator metrics to Kafka"""
        while self.running:
            try:
                time.sleep(60)  # Every minute
                
                # Send orchestrator metrics
                import psutil
                
                orchestrator_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'service': 'orchestrator',
                    'cpu_usage': psutil.cpu_percent() / 100.0,
                    'memory_usage': psutil.virtual_memory().percent / 100.0,
                    'processing_latency': 0.05,  # Orchestrator is typically fast
                    'decisions_made': self.orchestrator_metrics['decisions_made'],
                    'commands_sent': self.orchestrator_metrics['commands_sent'],
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                }
                
                self.kafka_client.send_message(
                    topic=self.topics['system_metrics'],
                    message=orchestrator_metrics
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error reporting metrics: {e}")
                time.sleep(60)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            orchestrator_summary = self.orchestrator.get_performance_summary()
            
            return {
                'running': self.running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'orchestrator_metrics': self.orchestrator_metrics,
                'orchestrator_summary': orchestrator_summary,
                'known_models': list(self.model_manager.model_status.keys()),
                'active_services': list(self.metrics_collector.service_metrics.keys()),
                'current_policy': self.orchestrator.get_current_policy(),
                'recent_performance_samples': len(self.performance_collector.get_recent_performance()),
                'recent_alerts': len(self.performance_collector.get_recent_alerts())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting service status: {e}")
            return {
                'error': str(e),
                'running': self.running,
                'orchestrator_metrics': self.orchestrator_metrics
            }
    
    def stop(self):
        """Stop the orchestrator service"""
        self.logger.info("🛑 Stopping Orchestrator Service...")
        self.running = False
        
        try:
            if hasattr(self.orchestrator, 'stop'):
                self.orchestrator.stop()
                self.logger.info("✅ Orchestrator core stopped")
        except:
            pass

def main():
    """Main entry point for separated orchestrator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧠 Starting Separated Orchestrator Service...")
        
        orchestrator = SeparatedOrchestratorService()
        
        if orchestrator.start():
            logger.info("✅ Separated Orchestrator Service started successfully")
            logger.info("📡 Communicating with ML Pipeline via Kafka")
            
            # Status monitoring loop
            while True:
                time.sleep(60)
                status = orchestrator.get_service_status()
                
                uptime = status.get('uptime_seconds', 0)
                uptime_formatted = str(timedelta(seconds=int(uptime)))
                
                logger.info(f"🧠 Status: {status['orchestrator_metrics']['decisions_made']} decisions, "
                           f"{status['orchestrator_metrics']['commands_sent']} commands sent, "
                           f"uptime: {uptime_formatted}")
                
                # Log current policy
                current_policy = status.get('current_policy', {})
                if isinstance(current_policy, dict):
                    active_models = sum(1 for enabled in current_policy.values() if enabled)
                    total_models = len(current_policy)
                    logger.info(f"🎯 Policy: {active_models}/{total_models} models active")
                
                # Log known services
                known_models = status.get('known_models', [])
                active_services = status.get('active_services', [])
                logger.info(f"📊 Connected: {len(active_services)} services, {len(known_models)} models")
                
        else:
            logger.error("❌ Failed to start Orchestrator Service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        if 'orchestrator' in locals():
            orchestrator.stop()
    except Exception as e:
        logger.error(f"💥 Orchestrator error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()