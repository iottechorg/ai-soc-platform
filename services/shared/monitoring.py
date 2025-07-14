import logging
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any

class MetricsCollector:
    """Prometheus metrics collector for SOC platform"""
    
    def __init__(self):
        # Counters
        self.logs_processed = Counter('soc_logs_processed_total', 'Total logs processed')
        self.anomalies_detected = Counter('soc_anomalies_detected_total', 'Total anomalies detected', ['model'])
        self.alerts_generated = Counter('soc_alerts_generated_total', 'Total alerts generated', ['severity'])
        
        # Histograms
        self.processing_time = Histogram('soc_processing_time_seconds', 'Processing time', ['component'])
        self.model_inference_time = Histogram('soc_model_inference_time_seconds', 'Model inference time', ['model'])
        
        # Gauges
        self.active_models = Gauge('soc_active_models', 'Number of active models')
        self.system_load = Gauge('soc_system_load', 'System load percentage')
        
        # Start metrics server
        start_http_server(8000)
        logging.info("Prometheus metrics server started on port 8000")
    
    def record_log_processed(self):
        """Record a processed log"""
        self.logs_processed.inc()
    
    def record_anomaly_detected(self, model_name: str):
        """Record an anomaly detection"""
        self.anomalies_detected.labels(model=model_name).inc()
    
    def record_alert_generated(self, severity: str):
        """Record an alert generation"""
        self.alerts_generated.labels(severity=severity).inc()
    
    def record_processing_time(self, component: str, duration: float):
        """Record processing time"""
        self.processing_time.labels(component=component).observe(duration)
    
    def record_model_inference_time(self, model_name: str, duration: float):
        """Record model inference time"""
        self.model_inference_time.labels(model=model_name).observe(duration)
    
    def set_active_models(self, count: int):
        """Set number of active models"""
        self.active_models.set(count)
    
    def set_system_load(self, load: float):
        """Set system load percentage"""
        self.system_load.set(load)