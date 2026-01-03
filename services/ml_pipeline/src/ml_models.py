# enhanced_ml_models.py - UPDATED VERSION for separated architecture with comprehensive tracking

import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import json
import pickle

class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on historical data and performance"""
    
    def __init__(self, target_alert_rate: float = 0.05):
        self.target_alert_rate = target_alert_rate  # Target 5% alert rate
        self.score_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        # Adaptive thresholds
        self.thresholds = {
            'critical': 0.95,
            'high': 0.85,
            'medium': 0.75,
            'low': 0.65
        }
        
        self.logger = logging.getLogger(__name__)
    
    def update_score_history(self, scores: List[float]):
        """Update score history for threshold adaptation"""
        self.score_history.extend(scores)
    
    def calculate_adaptive_thresholds(self) -> Dict[str, float]:
        """Calculate adaptive thresholds based on score distribution"""
        if len(self.score_history) < 50:
            return self.thresholds
        
        scores = np.array(list(self.score_history))
        
        # Calculate percentiles for adaptive thresholds
        percentiles = np.percentile(scores, [60, 75, 85, 95])
        
        # Ensure minimum separation between thresholds
        new_thresholds = {
            'low': max(0.4, percentiles[0]),
            'medium': max(0.5, percentiles[1]),
            'high': max(0.6, percentiles[2]),
            'critical': max(0.7, percentiles[3])
        }
        
        # Ensure ascending order
        thresholds_list = [
            new_thresholds['low'],
            new_thresholds['medium'], 
            new_thresholds['high'],
            new_thresholds['critical']
        ]
        
        # Fix ordering and minimum gaps
        for i in range(1, len(thresholds_list)):
            thresholds_list[i] = max(thresholds_list[i], thresholds_list[i-1] + 0.1)
        
        self.thresholds = {
            'low': thresholds_list[0],
            'medium': thresholds_list[1],
            'high': thresholds_list[2],
            'critical': thresholds_list[3]
        }
        
        self.logger.info(f"Updated adaptive thresholds: {self.thresholds}")
        return self.thresholds
    
    def get_severity_for_score(self, score: float) -> str:
        """Get severity level for a given score"""
        if score >= self.thresholds['critical']:
            return 'critical'
        elif score >= self.thresholds['high']:
            return 'high'
        elif score >= self.thresholds['medium']:
            return 'medium'
        elif score >= self.thresholds['low']:
            return 'low'
        else:
            return 'info'

class BaseMLModel:
    """Enhanced base class with comprehensive tracking and parameter management"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.is_trained = False
        self.performance_metrics = {}
        self.threshold_manager = AdaptiveThresholdManager()
        
        # NEW: Enhanced tracking attributes
        self.model_version = "1.0"
        self.last_training_time = None
        self.training_history = deque(maxlen=10)  # Keep last 10 training sessions
        self.prediction_count = 0
        self.last_prediction_time = None
        self.confidence_scores = deque(maxlen=100)  # Track confidence over time
        self.error_count = 0
        self.processing_times = deque(maxlen=50)  # Track processing performance
        
        # Score normalization parameters
        self.score_min = 0.0
        self.score_max = 1.0
        self.score_history = deque(maxlen=500)
    
    def extract_features_enhanced(self, log_data: Dict[str, Any]) -> np.ndarray:
        """Enhanced feature extraction with better normalization"""
        features = []
        
        try:
            # Port features
            port = log_data.get('port', 0)
            features.append(min(port / 65535.0, 1.0))
            
            # Port category features (more granular)
            if port < 1024:
                features.extend([1.0, 0.0, 0.0])  # Well-known ports
            elif port < 49152:
                features.extend([0.0, 1.0, 0.0])  # Registered ports
            else:
                features.extend([0.0, 0.0, 1.0])  # Dynamic ports
            
            # Protocol encoding (expanded)
            protocol_mapping = {
                'tcp': [1, 0, 0, 0, 0],
                'udp': [0, 1, 0, 0, 0],
                'http': [0, 0, 1, 0, 0],
                'https': [0, 0, 0, 1, 0],
                'ssh': [0, 0, 0, 0, 1]
            }
            protocol = log_data.get('protocol', '')
            features.extend(protocol_mapping.get(protocol, [0, 0, 0, 0, 0]))
            
            # Severity features (one-hot encoded)
            severity_mapping = {
                'info': [1, 0, 0, 0, 0],
                'low': [0, 1, 0, 0, 0],
                'medium': [0, 0, 1, 0, 0],
                'high': [0, 0, 0, 1, 0],
                'critical': [0, 0, 0, 0, 1]
            }
            severity = log_data.get('severity', 'info')
            features.extend(severity_mapping.get(severity, [1, 0, 0, 0, 0]))
            
            # Threat indicators (enhanced)
            threat_indicators = log_data.get('threat_indicators', [])
            features.append(min(len(threat_indicators) / 5.0, 1.0))  # Normalized count
            
            # Specific threat indicator flags
            threat_flags = {
                'brute_force': any('brute' in str(ind).lower() for ind in threat_indicators),
                'suspicious_source': any('suspicious' in str(ind).lower() for ind in threat_indicators),
                'malware': any('malware' in str(ind).lower() for ind in threat_indicators),
                'data_exfiltration': any('exfiltration' in str(ind).lower() for ind in threat_indicators)
            }
            features.extend([float(flag) for flag in threat_flags.values()])
            
            # Time-based features (enhanced)
            timestamp = log_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            features.append(timestamp.hour / 24.0)
            features.append(timestamp.weekday() / 7.0)
            features.append(np.sin(2 * np.pi * timestamp.hour / 24))  # Cyclic hour
            features.append(np.cos(2 * np.pi * timestamp.hour / 24))
            
            # Network features
            source_ip = log_data.get('source_ip', '0.0.0.0')
            ip_parts = str(source_ip).split('.')
            
            if len(ip_parts) == 4:
                try:
                    # Private IP detection
                    first_octet = int(ip_parts[0])
                    is_private = first_octet in [10, 172, 192]
                    features.append(float(is_private))
                    
                    # IP entropy (simplified)
                    ip_entropy = np.mean([int(part) / 255.0 for part in ip_parts])
                    features.append(ip_entropy)
                except:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # Traffic volume features
            bytes_transferred = log_data.get('bytes_transferred', 0)
            features.append(min(np.log1p(bytes_transferred) / 20, 1.0))  # Log-normalized
            
            duration = log_data.get('duration_seconds', 0)
            features.append(min(duration / 300.0, 1.0))  # Normalized to 5 minutes
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.debug(f"Error in feature extraction: {e}")
            # Return default feature vector if extraction fails
            return np.zeros((1, 26))  # Adjust size based on expected features
    
    def normalize_score(self, raw_score: float) -> float:
        """Normalize score to 0-1 range using historical distribution"""
        self.score_history.append(raw_score)
        
        if len(self.score_history) < 20:
            # Not enough history, use simple normalization
            return max(0.0, min(1.0, raw_score))
        
        # Use percentile-based normalization
        scores = np.array(list(self.score_history))
        p5, p95 = np.percentile(scores, [5, 95])
        
        if p95 > p5:
            normalized = (raw_score - p5) / (p95 - p5)
            return max(0.0, min(1.0, normalized))
        else:
            return 0.5  # Default if no variance
    
    # NEW: Parameter update capability
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update model parameters (override in subclasses)"""
        self.logger.info(f"Base parameter update for {self.name}: {parameters}")
        return True
    
    # NEW: Comprehensive model information
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        return {
            'name': self.name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'enabled': self.enabled,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'performance_metrics': self.performance_metrics,
            'training_history_count': len(self.training_history),
            'avg_confidence': np.mean(list(self.confidence_scores)) if self.confidence_scores else 0.0,
            'avg_processing_time_ms': np.mean(list(self.processing_times)) if self.processing_times else 0.0
        }
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Enhanced training with comprehensive tracking"""
        if not training_data:
            return
        
        training_start = time.time()
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
        try:
            # Subclasses should override this method with actual training logic
            # This is the base implementation
            
            training_time = time.time() - training_start
            self.last_training_time = datetime.now()
            
            # Track training history
            training_session = {
                'timestamp': self.last_training_time,
                'sample_count': len(training_data),
                'training_time_seconds': training_time,
                'threat_ratio': len([d for d in training_data if d.get('threat_indicators', [])]) / len(training_data)
            }
            self.training_history.append(training_session)
            
            self.is_trained = True
            self.model_version = f"1.{int(time.time()) % 10000}"  # Simple versioning
            
            self.logger.info(f"Training {self.name} completed in {training_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.name}: {e}")
            self.error_count += 1
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with comprehensive tracking"""
        if not self.is_trained:
            return 0.0, False
        
        prediction_start = time.time()
        
        try:
            # Subclasses should override this method with actual prediction logic
            # This is the base implementation that returns default values
            score = 0.0
            is_anomaly = False
            
            # Track prediction performance
            processing_time = (time.time() - prediction_start) * 1000  # ms
            self.processing_times.append(processing_time)
            
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Calculate and track confidence (simple implementation)
            confidence = min(abs(score - 0.5) * 2, 1.0)  # Distance from 0.5, normalized
            self.confidence_scores.append(confidence)
            
            return score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {self.name}: {e}")
            self.error_count += 1
            return 0.0, False

class EnhancedIsolationForest(BaseMLModel):
    """Enhanced Isolation Forest with comprehensive tracking and parameter updates"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("isolation_forest", config)
        self.model = IsolationForest(
            contamination=config.get('contamination', 0.05),
            n_estimators=config.get('n_estimators', 200),
            max_samples=config.get('max_samples', 256),
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        self.baseline_scores = deque(maxlen=200)
    
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update Isolation Forest specific parameters"""
        try:
            updated = False
            
            if 'contamination' in parameters:
                new_contamination = float(parameters['contamination'])
                if 0.01 <= new_contamination <= 0.5:  # Reasonable bounds
                    self.model.contamination = new_contamination
                    updated = True
                    self.logger.info(f"Updated contamination to {new_contamination}")
            
            if 'n_estimators' in parameters:
                new_estimators = int(parameters['n_estimators'])
                if 10 <= new_estimators <= 1000:  # Reasonable bounds
                    # Note: sklearn IsolationForest doesn't support changing n_estimators after creation
                    # So we need to recreate the model
                    self.model = IsolationForest(
                        contamination=self.model.contamination,
                        n_estimators=new_estimators,
                        max_samples=self.model.max_samples,
                        random_state=42,
                        n_jobs=-1
                    )
                    self.is_trained = False  # Need to retrain with new model
                    updated = True
                    self.logger.info(f"Updated n_estimators to {new_estimators} - retraining required")
            
            if updated:
                self.model_version = f"1.{int(time.time()) % 10000}"  # Simple versioning
                
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating parameters for {self.name}: {e}")
            return False
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get Isolation Forest specific information"""
        info = self.get_model_info()
        
        if hasattr(self.model, 'contamination'):
            info['contamination'] = self.model.contamination
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_samples'):
            info['max_samples'] = self.model.max_samples
            
        info['baseline_scores_count'] = len(self.baseline_scores)
        
        return info
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Enhanced training with better score calibration"""
        if not training_data:
            return
        
        training_start = time.time()
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
        try:
            # Extract features
            features = []
            for log in training_data:
                feature_vector = self.extract_features_enhanced(log)
                features.append(feature_vector.flatten())
            
            X = np.array(features)
            
            # Fit scaler and transform
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            
            # Calibrate baseline scores
            baseline_scores = self.model.decision_function(X_scaled)
            self.baseline_scores.extend(baseline_scores)
            
            # Complete training tracking
            training_time = time.time() - training_start
            self.last_training_time = datetime.now()
            
            training_session = {
                'timestamp': self.last_training_time,
                'sample_count': len(training_data),
                'training_time_seconds': training_time,
                'threat_ratio': len([d for d in training_data if d.get('threat_indicators', [])]) / len(training_data)
            }
            self.training_history.append(training_session)
            
            self.is_trained = True
            self.model_version = f"1.{int(time.time()) % 10000}"
            
            self.logger.info(f"Training complete. Baseline score range: {np.min(baseline_scores):.3f} to {np.max(baseline_scores):.3f}")
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.name}: {e}")
            self.error_count += 1
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with better score normalization"""
        if not self.is_trained:
            return 0.0, False
        
        prediction_start = time.time()
        
        try:
            features = self.extract_features_enhanced(log_data)
            features_scaled = self.scaler.transform(features)
            
            # Get raw score
            raw_score = self.model.decision_function(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            
            # Enhanced score normalization
            if len(self.baseline_scores) > 0:
                baseline_scores = np.array(list(self.baseline_scores))
                
                # Use z-score with robust statistics
                median_score = np.median(baseline_scores)
                mad_score = np.median(np.abs(baseline_scores - median_score))
                
                if mad_score > 0:
                    z_score = (raw_score - median_score) / (1.4826 * mad_score)  # Robust z-score
                    # Convert z-score to probability-like score
                    normalized_score = 1 / (1 + np.exp(-z_score))  # Sigmoid
                else:
                    normalized_score = 0.5
            else:
                # Fallback normalization
                normalized_score = self.normalize_score(raw_score)
            
            # Apply additional calibration based on prediction
            if prediction == -1:  # Anomaly detected
                normalized_score = max(0.6, normalized_score)  # Boost anomaly scores
            else:
                normalized_score = min(0.4, normalized_score)  # Cap normal scores
            
            is_anomaly = prediction == -1
            
            # Update threshold manager
            self.threshold_manager.update_score_history([normalized_score])
            
            # Track prediction performance
            processing_time = (time.time() - prediction_start) * 1000  # ms
            self.processing_times.append(processing_time)
            
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Calculate and track confidence
            confidence = min(abs(normalized_score - 0.5) * 2, 1.0)
            self.confidence_scores.append(confidence)
            
            return normalized_score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            self.error_count += 1
            return 0.0, False

class EnhancedForbiddenRatioModel(BaseMLModel):
    """Enhanced forbidden ratio model with comprehensive tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("forbidden_ratio", config)
        self.adaptive_thresholds = {
            'high_port_ratio': 0.8,
            'threat_indicator_ratio': 0.3,
            'suspicious_protocol_ratio': 0.4
        }
        self.window_data = deque(maxlen=200)  # Larger window
        self.baseline_ratios = {}
    
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update ForbiddenRatio model specific parameters"""
        try:
            updated = False
            
            if 'window_size' in parameters:
                new_window_size = int(parameters['window_size'])
                if 50 <= new_window_size <= 1000:  # Reasonable bounds
                    self.window_data = deque(maxlen=new_window_size)
                    updated = True
                    self.logger.info(f"Updated window_size to {new_window_size}")
            
            if 'high_port_ratio' in parameters:
                new_ratio = float(parameters['high_port_ratio'])
                if 0.1 <= new_ratio <= 1.0:
                    self.adaptive_thresholds['high_port_ratio'] = new_ratio
                    updated = True
                    self.logger.info(f"Updated high_port_ratio to {new_ratio}")
            
            if 'threat_indicator_ratio' in parameters:
                new_ratio = float(parameters['threat_indicator_ratio'])
                if 0.1 <= new_ratio <= 1.0:
                    self.adaptive_thresholds['threat_indicator_ratio'] = new_ratio
                    updated = True
                    self.logger.info(f"Updated threat_indicator_ratio to {new_ratio}")
            
            if updated:
                self.model_version = f"1.{int(time.time()) % 10000}"
                
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating parameters for {self.name}: {e}")
            return False
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get ForbiddenRatio model specific information"""
        info = self.get_model_info()
        
        info['adaptive_thresholds'] = self.adaptive_thresholds.copy()
        info['baseline_ratios'] = self.baseline_ratios.copy()
        info['window_size'] = self.window_data.maxlen
        info['current_window_size'] = len(self.window_data)
        
        return info
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train with adaptive baseline calculation"""
        if not training_data:
            return
        
        training_start = time.time()
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
        try:
            # Calculate baseline ratios
            total_logs = len(training_data)
            
            high_port_count = sum(1 for log in training_data if log.get('port', 0) > 1024)
            threat_count = sum(1 for log in training_data if log.get('threat_indicators', []))
            suspicious_protocol_count = sum(1 for log in training_data 
                                          if log.get('protocol', '') in ['ssh', 'ftp', 'telnet'])
            
            self.baseline_ratios = {
                'high_port': high_port_count / total_logs,
                'threat': threat_count / total_logs,
                'suspicious_protocol': suspicious_protocol_count / total_logs
            }
            
            # Set adaptive thresholds based on baseline + margin
            self.adaptive_thresholds = {
                'high_port_ratio': min(0.95, self.baseline_ratios['high_port'] + 0.3),
                'threat_indicator_ratio': min(0.8, self.baseline_ratios['threat'] + 0.2),
                'suspicious_protocol_ratio': min(0.7, self.baseline_ratios['suspicious_protocol'] + 0.3)
            }
            
            # Complete training tracking
            training_time = time.time() - training_start
            self.last_training_time = datetime.now()
            
            training_session = {
                'timestamp': self.last_training_time,
                'sample_count': len(training_data),
                'training_time_seconds': training_time,
                'threat_ratio': len([d for d in training_data if d.get('threat_indicators', [])]) / len(training_data)
            }
            self.training_history.append(training_session)
            
            self.is_trained = True
            self.model_version = f"1.{int(time.time()) % 10000}"
            
            self.logger.info(f"Trained {self.name}: baselines={self.baseline_ratios}, thresholds={self.adaptive_thresholds}")
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.name}: {e}")
            self.error_count += 1
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with multiple ratio checks"""
        if not self.is_trained:
            return 0.0, False
        
        prediction_start = time.time()
        
        try:
            self.window_data.append(log_data)
            
            if len(self.window_data) < 20:
                return 0.1, False
            
            # Calculate current ratios
            window_logs = list(self.window_data)
            total_logs = len(window_logs)
            
            ratios = {
                'high_port': sum(1 for log in window_logs if log.get('port', 0) > 1024) / total_logs,
                'threat': sum(1 for log in window_logs if log.get('threat_indicators', [])) / total_logs,
                'suspicious_protocol': sum(1 for log in window_logs 
                                         if log.get('protocol', '') in ['ssh', 'ftp', 'telnet']) / total_logs
            }
            
            # Calculate violation scores
            violation_scores = []
            violations = []
            
            for ratio_name, current_ratio in ratios.items():
                threshold_key = f"{ratio_name}_ratio"
                if threshold_key in self.adaptive_thresholds:
                    threshold = self.adaptive_thresholds[threshold_key]
                    if current_ratio > threshold:
                        violation_severity = (current_ratio - threshold) / (1.0 - threshold)
                        violation_scores.append(min(violation_severity, 0.9))
                        violations.append(ratio_name)
            
            # Combine violation scores
            if violation_scores:
                # Use max score but boost if multiple violations
                base_score = max(violation_scores)
                multiplier = 1.0 + 0.2 * (len(violation_scores) - 1)  # Boost for multiple violations
                anomaly_score = min(base_score * multiplier, 1.0)
                is_anomaly = anomaly_score > 0.6
            else:
                anomaly_score = 0.1
                is_anomaly = False
            
            # Normalize score
            anomaly_score = self.normalize_score(anomaly_score)
            
            # Update threshold manager
            self.threshold_manager.update_score_history([anomaly_score])
            
            # Track prediction performance
            processing_time = (time.time() - prediction_start) * 1000  # ms
            self.processing_times.append(processing_time)
            
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Calculate and track confidence
            confidence = min(abs(anomaly_score - 0.5) * 2, 1.0)
            self.confidence_scores.append(confidence)
            
            return anomaly_score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            self.error_count += 1
            return 0.0, False

class TrainingDataManager:
    """Helper class for managing training data lifecycle"""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.training_cache = deque(maxlen=max_cache_size)
        self.threat_samples = deque(maxlen=200)
        self.normal_samples = deque(maxlen=800)
        self.logger = logging.getLogger(__name__)
    
    def add_training_sample(self, log_data: Dict[str, Any], is_verified: bool = False):
        """Add a training sample with verification flag"""
        enriched_sample = {
            **log_data,
            'added_timestamp': datetime.now(),
            'is_verified': is_verified,
            'sample_source': 'live_stream'
        }
        
        if log_data.get('threat_indicators', []):
            self.threat_samples.append(enriched_sample)
        else:
            # Sample normal data to prevent overwhelming
            if len(self.normal_samples) < self.normal_samples.maxlen * 0.8 or random.random() < 0.1:
                self.normal_samples.append(enriched_sample)
    
    def get_balanced_sample(self, size: int = 1000, threat_ratio: float = 0.15) -> List[Dict[str, Any]]:
        """Get balanced training sample"""
        threat_count = int(size * threat_ratio)
        normal_count = size - threat_count
        
        # Get available samples
        available_threats = list(self.threat_samples)
        available_normal = list(self.normal_samples)
        
        # Sample appropriately
        threats = random.sample(available_threats, min(threat_count, len(available_threats)))
        normal = random.sample(available_normal, min(normal_count, len(available_normal)))
        
        combined = threats + normal
        random.shuffle(combined)
        
        self.logger.info(f"Generated balanced sample: {len(threats)} threats, {len(normal)} normal")
        return combined
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get training cache statistics"""
        return {
            'total_cached': len(self.training_cache),
            'threat_samples': len(self.threat_samples),
            'normal_samples': len(self.normal_samples),
            'cache_utilization': len(self.training_cache) / self.max_cache_size,
            'threat_ratio': len(self.threat_samples) / (len(self.threat_samples) + len(self.normal_samples)) if (len(self.threat_samples) + len(self.normal_samples)) > 0 else 0
        }

class MLModelManager:
    """Enhanced ML Model Manager with comprehensive tracking and control"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.training_data = deque(maxlen=2000)  # Larger buffer
        self.global_threshold_manager = AdaptiveThresholdManager()
        
        # NEW: Enhanced tracking attributes
        self.manager_version = "2.0"
        self.total_predictions = 0
        self.training_sessions = deque(maxlen=50)
        self.performance_tracking = {
            'total_training_time': 0,
            'total_predictions': 0,
            'last_performance_update': None
        }
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        self.alert_rate_history = deque(maxlen=100)
        
        # Training data manager
        self.training_data_manager = TrainingDataManager()
    
    def _initialize_models(self):
        """Initialize enhanced ML models"""
        model_configs = self.config.get('ml_models', {})
        
        # Initialize enhanced models
        if model_configs.get('isolation_forest', {}).get('enabled', True):
            self.models['isolation_forest'] = EnhancedIsolationForest(
                model_configs.get('isolation_forest', {})
            )
        
        if model_configs.get('clustering', {}).get('enabled', True):
            self.models['clustering'] = EnhancedClusteringModel(
                model_configs.get('clustering', {})
            )
        
        if model_configs.get('forbidden_ratio', {}).get('enabled', True):
            self.models['forbidden_ratio'] = EnhancedForbiddenRatioModel(
                model_configs.get('forbidden_ratio', {})
            )
        
        self.logger.info(f"Initialized {len(self.models)} enhanced ML models")
    
    def train_models(self, training_data: List[Dict[str, Any]] = None):
        """Enhanced model training with comprehensive tracking"""
        if training_data is None:
            training_data = list(self.training_data)
        
        if not training_data:
            self.logger.warning("No training data available")
            return
        
        session_start = time.time()
        
        # Analyze training data distribution
        threat_logs = [log for log in training_data if log.get('threat_indicators', [])]
        normal_logs = [log for log in training_data if not log.get('threat_indicators', [])]
        
        self.logger.info(f"Training data: {len(normal_logs)} normal, {len(threat_logs)} threats "
                        f"({len(threat_logs)/len(training_data)*100:.1f}% threats)")
        
        # Train each model
        models_trained = 0
        for name, model in self.models.items():
            if model.enabled:
                try:
                    start_time = time.time()
                    model.train(training_data)
                    train_time = time.time() - start_time
                    models_trained += 1
                    self.logger.info(f"Trained {name} model in {train_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"Failed to train {name} model: {e}")
        
        # Track training session
        session_time = time.time() - session_start
        
        training_session = {
            'timestamp': datetime.now(),
            'total_samples': len(training_data),
            'threat_samples': len(threat_logs),
            'normal_samples': len(normal_logs),
            'models_trained': models_trained,
            'session_time_seconds': session_time,
            'threat_ratio': len(threat_logs) / len(training_data) if training_data else 0
        }
        
        self.training_sessions.append(training_session)
        self.performance_tracking['total_training_time'] += session_time
        self.performance_tracking['last_performance_update'] = datetime.now()
        
        self.logger.info(f"Training session complete: {models_trained} models in {session_time:.2f}s")
    
    def predict_all(self, log_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Enhanced prediction with comprehensive tracking"""
        predictions = {}
        scores = []
        
        prediction_start = time.time()
        
        for name, model in self.models.items():
            if model.enabled:
                try:
                    score, is_anomaly = model.predict(log_data)
                    
                    # Calculate confidence (distance from threshold)
                    confidence = abs(score - 0.5) * 2  # Simple confidence measure
                    
                    predictions[name] = {
                        'score': score,
                        'is_anomaly': is_anomaly,
                        'confidence': min(confidence, 1.0),
                        'timestamp': datetime.now(),
                        'model_version': getattr(model, 'model_version', '1.0'),
                        'features': self._extract_prediction_features(log_data, model),
                        'processing_time_ms': (time.time() - prediction_start) * 1000
                    }
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error in {name} prediction: {e}")
                    predictions[name] = {
                        'score': 0.0,
                        'is_anomaly': False,
                        'confidence': 0.0,
                        'timestamp': datetime.now(),
                        'error': str(e)
                    }
        
        # Update global tracking
        self.total_predictions += 1
        self.performance_tracking['total_predictions'] = self.total_predictions
        
        # Update threshold manager
        if scores:
            self.global_threshold_manager.update_score_history(scores)
        
        # Track prediction history for performance monitoring
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': predictions,
            'entity_id': log_data.get('source_ip', 'unknown'),
            'total_processing_time_ms': (time.time() - prediction_start) * 1000
        })
        
        return predictions
    
    def _extract_prediction_features(self, log_data: Dict[str, Any], model) -> Dict[str, Any]:
        """Extract relevant features for prediction tracking"""
        try:
            return {
                'port': log_data.get('port', 0),
                'protocol': log_data.get('protocol', ''),
                'severity': log_data.get('severity', ''),
                'threat_indicators_count': len(log_data.get('threat_indicators', [])),
                'bytes_transferred': log_data.get('bytes_transferred', 0),
                'model_type': type(model).__name__
            }
        except Exception as e:
            self.logger.debug(f"Error extracting features: {e}")
            return {}
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        return self.global_threshold_manager.calculate_adaptive_thresholds()
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced status of all models - REQUIRED BY ORCHESTRATOR"""
        status = {}
        for name, model in self.models.items():
            try:
                # Get model-specific info if available
                if hasattr(model, 'get_model_specific_info'):
                    model_info = model.get_model_specific_info()
                else:
                    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
                
                # Add manager-level information
                status[name] = {
                    **model_info,
                    'manager_version': self.manager_version,
                    'total_manager_predictions': self.total_predictions,
                    'last_performance_update': self.performance_tracking['last_performance_update'].isoformat() if self.performance_tracking['last_performance_update'] else None
                }
                
            except Exception as e:
                self.logger.error(f"Error getting status for {name}: {e}")
                status[name] = {
                    'enabled': getattr(model, 'enabled', False),
                    'trained': getattr(model, 'is_trained', False),
                    'error': str(e),
                    'model_type': type(model).__name__
                }
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        recent_predictions = list(self.prediction_history)[-100:]
        
        if not recent_predictions:
            return {
                'total_predictions': self.total_predictions,
                'recent_avg_processing_time': 0.0,
                'models_count': len(self.models),
                'enabled_models_count': sum(1 for m in self.models.values() if m.enabled),
                'trained_models_count': sum(1 for m in self.models.values() if m.is_trained)
            }
        
        # Calculate performance metrics
        processing_times = [p.get('total_processing_time_ms', 0) for p in recent_predictions]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Model-specific metrics
        model_stats = {}
        for name in self.models.keys():
            model_predictions = [p['predictions'].get(name, {}) for p in recent_predictions]
            model_predictions = [p for p in model_predictions if p and 'score' in p]
            
            if model_predictions:
                avg_score = sum(p.get('score', 0) for p in model_predictions) / len(model_predictions)
                avg_confidence = sum(p.get('confidence', 0) for p in model_predictions) / len(model_predictions)
                anomaly_rate = sum(1 for p in model_predictions if p.get('is_anomaly', False)) / len(model_predictions)
                
                model_stats[name] = {
                    'predictions_count': len(model_predictions),
                    'avg_score': avg_score,
                    'avg_confidence': avg_confidence,
                    'anomaly_rate': anomaly_rate
                }
        
        return {
            'total_predictions': self.total_predictions,
            'recent_predictions_count': len(recent_predictions),
            'recent_avg_processing_time_ms': avg_processing_time,
            'models_count': len(self.models),
            'enabled_models_count': sum(1 for m in self.models.values() if m.enabled),
            'trained_models_count': sum(1 for m in self.models.values() if m.is_trained),
            'model_stats': model_stats,
            'adaptive_thresholds': self.get_adaptive_thresholds(),
            'total_training_time': self.performance_tracking['total_training_time'],
            'training_sessions_count': len(self.training_sessions),
            'global_alert_rate': sum(
                1 for p in recent_predictions 
                if any(pred.get('is_anomaly', False) for pred in p['predictions'].values())
            ) / len(recent_predictions) if recent_predictions else 0
        }
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training session history"""
        return [
            {
                **session,
                'timestamp': session['timestamp'].isoformat()
            }
            for session in self.training_sessions
        ]
    
    def enable_model(self, model_name: str):
        """Enable a specific model"""
        if model_name in self.models:
            self.models[model_name].enabled = True
            self.logger.info(f"Enabled {model_name} model")
        else:
            self.logger.warning(f"Model {model_name} not found")
    
    def disable_model(self, model_name: str):
        """Disable a specific model"""
        if model_name in self.models:
            self.models[model_name].enabled = False
            self.logger.info(f"Disabled {model_name} model")
        else:
            self.logger.warning(f"Model {model_name} not found")
    
    def bulk_enable_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Enable multiple models and return success status"""
        results = {}
        for model_name in model_names:
            try:
                self.enable_model(model_name)
                results[model_name] = True
            except Exception as e:
                self.logger.error(f"Failed to enable {model_name}: {e}")
                results[model_name] = False
        return results
    
    def bulk_disable_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Disable multiple models and return success status"""
        results = {}
        for model_name in model_names:
            try:
                self.disable_model(model_name)
                results[model_name] = True
            except Exception as e:
                self.logger.error(f"Failed to disable {model_name}: {e}")
                results[model_name] = False
        return results
    
    def add_training_data(self, log_data: Dict[str, Any]):
        """Add data to training buffer"""
        self.training_data.append(log_data)
        self.training_data_manager.add_training_sample(log_data)
        self.logger.debug(f"Added log data to training buffer (buffer size: {len(self.training_data)})")
            
       

class EnhancedClusteringModel(BaseMLModel):
    """Enhanced clustering model with comprehensive tracking and parameter updates"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("clustering", config)
        self.model = KMeans(
            n_clusters=config.get('n_clusters', 8),
            random_state=42,
            n_init=20,
            max_iter=500
        )
        self.cluster_stats = {}
        self.cluster_centers = None
    
    def update_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update Clustering model specific parameters"""
        try:
            updated = False
            
            if 'n_clusters' in parameters:
                new_clusters = int(parameters['n_clusters'])
                if 2 <= new_clusters <= 20:  # Reasonable bounds
                    # Recreate model with new clusters
                    self.model = KMeans(
                        n_clusters=new_clusters,
                        random_state=42,
                        n_init=self.model.n_init,
                        max_iter=self.model.max_iter
                    )
                    self.is_trained = False  # Need to retrain
                    updated = True
                    self.logger.info(f"Updated n_clusters to {new_clusters} - retraining required")
            
            if 'max_iter' in parameters:
                new_max_iter = int(parameters['max_iter'])
                if 100 <= new_max_iter <= 2000:
                    self.model.max_iter = new_max_iter
                    updated = True
                    self.logger.info(f"Updated max_iter to {new_max_iter}")
            
            if updated:
                self.model_version = f"1.{int(time.time()) % 10000}"
                
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating parameters for {self.name}: {e}")
            return False
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get Clustering model specific information"""
        info = self.get_model_info()
        
        if hasattr(self.model, 'n_clusters'):
            info['n_clusters'] = self.model.n_clusters
        if hasattr(self.model, 'max_iter'):
            info['max_iter'] = self.model.max_iter
        if hasattr(self.model, 'n_init'):
            info['n_init'] = self.model.n_init
            
        info['cluster_stats_count'] = len(self.cluster_stats)
        
        return info
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Enhanced training with better cluster analysis"""
        if not training_data:
            return
        
        training_start = time.time()
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
        try:
            # Extract features
            features = []
            labels = []  # Track if log has threat indicators
            
            for log in training_data:
                feature_vector = self.extract_features_enhanced(log)
                features.append(feature_vector.flatten())
                # Label as threat if has threat indicators
                has_threats = len(log.get('threat_indicators', [])) > 0
                labels.append(1 if has_threats else 0)
            
            X = np.array(features)
            y = np.array(labels)
            
            # Fit scaler and transform
            X_scaled = self.scaler.fit_transform(X)
            
            # Train clustering model
            self.model.fit(X_scaled)
            cluster_labels = self.model.labels_
            distances = self.model.transform(X_scaled)
            
            # Analyze clusters for threat content
            self.cluster_stats = {}
            for cluster_id in range(self.model.n_clusters):
                mask = cluster_labels == cluster_id
                cluster_points = X_scaled[mask]
                cluster_threats = y[mask]
                cluster_distances = distances[mask, cluster_id]
                
                if len(cluster_points) > 0:
                    threat_ratio = np.mean(cluster_threats)
                    distance_stats = {
                        'mean': np.mean(cluster_distances),
                        'std': np.std(cluster_distances),
                        'q75': np.percentile(cluster_distances, 75),
                        'q95': np.percentile(cluster_distances, 95)
                    }
                    
                    # Adaptive threshold based on threat content
                    if threat_ratio > 0.3:  # High-threat cluster
                        threshold = distance_stats['q75']
                    elif threat_ratio > 0.1:  # Medium-threat cluster
                        threshold = distance_stats['q95']
                    else:  # Low-threat cluster
                        threshold = distance_stats['mean'] + 2 * distance_stats['std']
                    
                    self.cluster_stats[cluster_id] = {
                        'threat_ratio': threat_ratio,
                        'threshold': threshold,
                        'distance_stats': distance_stats,
                        'sample_count': len(cluster_points)
                    }
                else:
                    # Empty cluster
                    self.cluster_stats[cluster_id] = {
                        'threat_ratio': 0.0,
                        'threshold': 2.0,
                        'distance_stats': {'mean': 1.0, 'std': 0.5},
                        'sample_count': 0
                    }
            
            self.cluster_centers = self.model.cluster_centers_
            
            # Complete training tracking
            training_time = time.time() - training_start
            self.last_training_time = datetime.now()
            
            training_session = {
                'timestamp': self.last_training_time,
                'sample_count': len(training_data),
                'training_time_seconds': training_time,
                'threat_ratio': len([d for d in training_data if d.get('threat_indicators', [])]) / len(training_data)
            }
            self.training_history.append(training_session)
            
            self.is_trained = True
            self.model_version = f"1.{int(time.time()) % 10000}"
            
            self.logger.info(f"Clustering training complete. Found {self.model.n_clusters} clusters")
            for cid, stats in self.cluster_stats.items():
                self.logger.info(f"Cluster {cid}: {stats['sample_count']} samples, "
                               f"threat_ratio={stats['threat_ratio']:.2f}, "
                               f"threshold={stats['threshold']:.3f}")
                
        except Exception as e:
            self.logger.error(f"Training failed for {self.name}: {e}")
            self.error_count += 1
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with cluster-aware scoring"""
        if not self.is_trained:
            return 0.0, False
        
        prediction_start = time.time()
        
        try:
            features = self.extract_features_enhanced(log_data)
            features_scaled = self.scaler.transform(features)
            
            # Get cluster assignment and distances
            cluster_id = self.model.predict(features_scaled)[0]
            distances = self.model.transform(features_scaled)
            distance_to_cluster = distances[0, cluster_id]
            
            # Get cluster statistics
            cluster_info = self.cluster_stats.get(cluster_id, {})
            threshold = cluster_info.get('threshold', 2.0)
            threat_ratio = cluster_info.get('threat_ratio', 0.0)
            
            # Calculate anomaly score
            if threshold > 0:
                base_score = distance_to_cluster / threshold
            else:
                base_score = 1.0
            
            # Adjust score based on cluster threat level
            threat_multiplier = 1.0 + threat_ratio  # Higher for threat-prone clusters
            adjusted_score = base_score * threat_multiplier
            
            # Normalize to 0-1 range
            normalized_score = min(adjusted_score / 2.0, 1.0)
            normalized_score = self.normalize_score(normalized_score)
            
            # Determine if anomaly
            is_anomaly = distance_to_cluster > threshold
            
            # Update threshold manager
            self.threshold_manager.update_score_history([normalized_score])
            
            # Track prediction performance
            processing_time = (time.time() - prediction_start) * 1000  # ms
            self.processing_times.append(processing_time)
            
            self.prediction_count += 1
            self.last_prediction_time = datetime.now()
            
            # Calculate and track confidence
            confidence = min(abs(normalized_score - 0.5) * 2, 1.0)
            self.confidence_scores.append(confidence)
            
            return normalized_score, is_anomaly
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            self.error_count += 1
            return 0.0, False