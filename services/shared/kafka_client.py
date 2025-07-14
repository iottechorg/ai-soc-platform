import json
import logging
from typing import Optional, Dict, Any, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import threading
import time
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class KafkaClient:
    """Unified Kafka client for producer and consumer operations"""
    
    def __init__(self, broker: str):
        self.broker = broker
        self.producer = None
        self.consumers = {}
        self.logger = logging.getLogger(__name__)
    
    def get_producer(self) -> KafkaProducer:
        """Get or create Kafka producer with datetime serialization"""
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.broker],
                value_serializer=lambda v: json.dumps(v, cls=DateTimeEncoder).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                retry_backoff_ms=1000,
                max_request_size=10485760,  # 10MB
                buffer_memory=33554432      # 32MB
            )
        return self.producer
    
    def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """Send message to Kafka topic with proper datetime handling"""
        try:
            # Convert datetime objects to ISO format strings before sending
            serialized_message = self._serialize_datetime_objects(message)
            
            producer = self.get_producer()
            future = producer.send(topic, value=serialized_message, key=key)
            producer.flush()
            return future.get(timeout=10)
        except Exception as e:
            self.logger.error(f"Failed to send message to {topic}: {e}")
            self.logger.error(f"Message content: {message}")
            raise
    
    def _serialize_datetime_objects(self, obj):
        """Recursively convert datetime objects to ISO format strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._serialize_datetime_objects(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        else:
            return obj
    
    def create_consumer(self, topics: list, group_id: str, 
                       message_handler: Callable[[Dict[str, Any]], None]):
        """Create and start a Kafka consumer with datetime deserialization"""
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=[self.broker],
            group_id=group_id,
            value_deserializer=lambda m: self._deserialize_message(m),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            max_poll_records=100,
            fetch_max_wait_ms=1000
        )
        
        def consume_messages():
            self.logger.info(f"Starting consumer for topics: {topics}")
            try:
                for message in consumer:
                    try:
                        message_handler(message.value)
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}")
            except KeyboardInterrupt:
                self.logger.info("Consumer stopped by user")
            finally:
                consumer.close()
        
        thread = threading.Thread(target=consume_messages)
        thread.daemon = True
        thread.start()
        
        self.consumers[group_id] = consumer
        return consumer
    
    def _deserialize_message(self, message_bytes):
        """Deserialize message and convert ISO strings back to datetime objects"""
        try:
            data = json.loads(message_bytes.decode('utf-8'))
            return self._deserialize_datetime_objects(data)
        except Exception as e:
            self.logger.error(f"Failed to deserialize message: {e}")
            return {}
    
    def _deserialize_datetime_objects(self, obj):
        """Recursively convert ISO format strings back to datetime objects"""
        if isinstance(obj, str):
            # Try to parse as datetime
            try:
                # Check if it looks like an ISO datetime string
                if 'T' in obj and ('.' in obj or '+' in obj or 'Z' in obj or len(obj) >= 19):
                    return datetime.fromisoformat(obj.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
            return obj
        elif isinstance(obj, dict):
            return {k: self._deserialize_datetime_objects(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_datetime_objects(item) for item in obj]
        else:
            return obj
    
    def close(self):
        """Close all Kafka connections"""
        if self.producer:
            self.producer.close()
        for consumer in self.consumers.values():
            consumer.close()