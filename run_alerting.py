#!/usr/bin/env python3

import sys
import time
import logging
import signal
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.alerting import AlertingService
    from config.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class AlertingServiceRunner:
    """Clean microservices alerting runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigLoader()
        
        # Initialize alerting service with enhanced configuration
        alerting_config = self.config.get_section('alerting')
        self.alerting_service = AlertingService(alerting_config)
        
        # Initialize Kafka client
        self.kafka_client = KafkaClient(self.config.get('kafka.broker'))
        
        # Runner state
        self.running = False
        self.alerts_processed = 0
        self.start_time = None
    
    def start(self):
        """Start the alerting service"""
        self.logger.info("🚀 Starting Enhanced Alerting Service...")
        self.start_time = datetime.now()
        
        # Start the alerting service
        self.alerting_service.start()
        
        # Start Kafka consumer for alerts
        try:
            alerts_topic = self.config.get('kafka.topics.alerts', 'alerts')
            self.kafka_client.create_consumer(
                topics=[alerts_topic],
                group_id="enhanced-alerting-service-consumer",
                message_handler=self._process_alert_from_kafka
            )
            self.logger.info(f"✅ Started Kafka consumer for {alerts_topic}")
        except Exception as e:
            self.logger.error(f"❌ Kafka consumer failed: {e}")
            return False
        
        self.running = True
        self.logger.info("📨 Enhanced Alerting Service is ready to process alerts")
        return True
    
    def _process_alert_from_kafka(self, alert_data: Dict[str, Any]):
        """Process incoming alert from Kafka"""
        try:
            # Send alert through the enhanced alerting service
            self.alerting_service.send_alert(alert_data)
            
            self.alerts_processed += 1
            
            # Log processing (reduced frequency)
            if self.alerts_processed % 10 == 0:
                severity = alert_data.get('severity', 'unknown')
                entity_id = alert_data.get('entity_id', 'unknown')
                score = alert_data.get('aggregated_score', 0.0)
                
                self.logger.info(
                    f"📨 Processed {self.alerts_processed} alerts "
                    f"(latest: {severity} for {entity_id}, score: {score:.3f})"
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error processing alert from Kafka: {e}")
    
    def test_alerting_channels(self):
        """Test all configured alerting channels"""
        self.logger.info("🧪 Testing alerting channels...")
        
        test_alert = {
            'alert_id': f'test_{int(datetime.now().timestamp())}',
            'timestamp': datetime.now(),
            'severity': 'medium',
            'entity_id': '192.168.1.100',
            'message': '🧪 Test Alert: Enhanced SOC Platform alerting system test',
            'aggregated_score': 0.75,
            'confidence': 0.80,
            'risk_level': 'medium',
            'contributing_models': ['test_model'],
            'tags': ['test', 'system_check'],
            'actions': ['verify_configuration'],
            'score_trend': 'stable'
        }
        
        for channel_name, channel in self.alerting_service.channels.items():
            try:
                if channel.is_enabled():
                    # Send test message through the service
                    self.alerting_service.send_alert(test_alert)
                    self.logger.info(f"✅ {channel_name} test queued successfully")
                else:
                    self.logger.warning(f"⚠️ {channel_name} is not enabled")
            except Exception as e:
                self.logger.error(f"❌ {channel_name} test failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get alerting service status"""
        try:
            alert_stats = self.alerting_service.get_enhanced_alert_stats()
            
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'running': self.running,
                'alerts_processed': self.alerts_processed,
                'uptime_seconds': uptime,
                'uptime_formatted': str(datetime.now() - self.start_time).split('.')[0] if self.start_time else "0:00:00",
                'alerting_stats': alert_stats,
                'enabled_channels': [
                    name for name, channel in self.alerting_service.channels.items() 
                    if channel.is_enabled()
                ],
                'start_time': self.start_time.isoformat() if self.start_time else None
            }
        except Exception as e:
            return {
                'error': str(e), 
                'running': self.running,
                'alerts_processed': self.alerts_processed
            }
    
    def stop(self):
        """Stop the alerting service"""
        self.logger.info("🛑 Stopping Enhanced Alerting Service...")
        self.running = False
        
        if self.alerting_service:
            self.alerting_service.stop()
        
        if self.kafka_client:
            self.kafka_client.close()
        
        # Final stats
        if self.start_time:
            uptime = datetime.now() - self.start_time
            self.logger.info(
                f"📊 Final stats: {self.alerts_processed} alerts processed, "
                f"uptime: {str(uptime).split('.')[0]}"
            )


def setup_signal_handlers(runner):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        runner.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for enhanced alerting service"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Enhanced Alerting Service...")
        
        # Initialize alerting runner
        alerting_runner = AlertingServiceRunner()
        
        # Setup graceful shutdown
        setup_signal_handlers(alerting_runner)
        
        # Test channels first
        alerting_runner.test_alerting_channels()
        
        if alerting_runner.start():
            logger.info("✅ Enhanced Alerting Service started successfully")
            logger.info("📨 Ready to process alerts from Kafka and send via configured channels")
            
            # Main monitoring loop
            last_status_time = 0
            status_interval = 60  # Report status every minute
            
            while True:
                time.sleep(10)  # Check every 10 seconds
                current_time = time.time()
                
                # Report status periodically
                if current_time - last_status_time >= status_interval:
                    try:
                        status = alerting_runner.get_status()
                        alerts_processed = status.get('alerts_processed', 0)
                        uptime = status.get('uptime_formatted', '0:00:00')
                        enabled_channels = status.get('enabled_channels', [])
                        
                        logger.info(f"📈 Status: {alerts_processed} alerts processed, uptime: {uptime}")
                        
                        if enabled_channels:
                            logger.info(f"📡 Active channels: {', '.join(enabled_channels)}")
                        else:
                            logger.warning("⚠️ No alerting channels enabled")
                        
                        # Show detailed stats every 5 minutes
                        if alerts_processed > 0 and current_time % 300 < 60:  # Every ~5 minutes
                            alert_stats = status.get('alerting_stats', {})
                            severity_dist = alert_stats.get('severity_distribution_1h', {})
                            if severity_dist:
                                logger.info(f"🚨 Alert distribution (1h): {severity_dist}")
                        
                        last_status_time = current_time
                        
                    except Exception as e:
                        logger.error(f"❌ Error getting status: {e}")
        else:
            logger.error("❌ Failed to start Enhanced Alerting Service")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if 'alerting_runner' in locals():
            alerting_runner.stop()


if __name__ == "__main__":
    main()