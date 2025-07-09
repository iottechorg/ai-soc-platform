# run_data_generator.py
#!/usr/bin/env python3

import sys
import time
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.data_generator import EnhancedSyntheticDataGenerator
    from config.config_loader import ConfigLoader
    from shared.kafka_client import KafkaClient
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    """Main entry point for data generator"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Data Generator...")
        
        # Load configuration
        config = ConfigLoader()
        
        # Initialize Kafka client
        kafka_client = KafkaClient(config.get('kafka.broker'))
        
        # Initialize data generator
        generator = EnhancedSyntheticDataGenerator(config.get_section('data_generator'))
        
        # Start continuous generation
        
        generator.start_enhanced_continuous_generation(
            kafka_client, 
            config.get('kafka.topics.raw_logs')
        )
        
        logger.info("Data Generator started successfully")
        
        # Keep running
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Shutting down Data Generator...")
    except Exception as e:
        logger.error(f"Data Generator error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
