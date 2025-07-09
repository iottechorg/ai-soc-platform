# enhanced_ml_models.py - FIXED VERSION with adaptive thresholds and proper calibration

import numpy as np
import pandas as pd
import random  # Add missing import
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
import logging
from typing import Dict, Any, List, Tuple
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
    """Enhanced base class with better feature engineering and calibration"""
    
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
        
        # Score normalization parameters
        self.score_min = 0.0
        self.score_max = 1.0
        self.score_history = deque(maxlen=500)
    
    def extract_features_enhanced(self, log_data: Dict[str, Any]) -> np.ndarray:
        """Enhanced feature extraction with better normalization"""
        features = []
        
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
            'brute_force': any('brute' in ind.lower() for ind in threat_indicators),
            'suspicious_source': any('suspicious' in ind.lower() for ind in threat_indicators),
            'malware': any('malware' in ind.lower() for ind in threat_indicators),
            'data_exfiltration': any('exfiltration' in ind.lower() for ind in threat_indicators)
        }
        features.extend([float(flag) for flag in threat_flags.values()])
        
        # Time-based features (enhanced)
        timestamp = log_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        features.append(timestamp.hour / 24.0)
        features.append(timestamp.weekday() / 7.0)
        features.append(np.sin(2 * np.pi * timestamp.hour / 24))  # Cyclic hour
        features.append(np.cos(2 * np.pi * timestamp.hour / 24))
        
        # Network features
        source_ip = log_data.get('source_ip', '0.0.0.0')
        ip_parts = source_ip.split('.')
        
        if len(ip_parts) == 4:
            # Private IP detection
            first_octet = int(ip_parts[0])
            is_private = first_octet in [10, 172, 192]
            features.append(float(is_private))
            
            # IP entropy (simplified)
            ip_entropy = np.mean([int(part) / 255.0 for part in ip_parts])
            features.append(ip_entropy)
        else:
            features.extend([0.0, 0.0])
        
        # Traffic volume features
        bytes_transferred = log_data.get('bytes_transferred', 0)
        features.append(min(np.log1p(bytes_transferred) / 20, 1.0))  # Log-normalized
        
        duration = log_data.get('duration_seconds', 0)
        features.append(min(duration / 300.0, 1.0))  # Normalized to 5 minutes
        
        return np.array(features).reshape(1, -1)
    
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

class EnhancedIsolationForest(BaseMLModel):
    """Enhanced Isolation Forest with better calibration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("isolation_forest", config)
        self.model = IsolationForest(
            contamination=config.get('contamination', 0.05),
            n_estimators=config.get('n_estimators', 200),  # Increased
            max_samples=config.get('max_samples', 256),
            random_state=42,
            n_jobs=-1  # Use all cores
        )
        self.baseline_scores = deque(maxlen=200)
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Enhanced training with better score calibration"""
        if not training_data:
            return
        
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
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
        
        self.is_trained = True
        self.logger.info(f"Training complete. Baseline score range: {np.min(baseline_scores):.3f} to {np.max(baseline_scores):.3f}")
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with better score normalization"""
        if not self.is_trained:
            return 0.0, False
        
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
            
            return normalized_score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            return 0.0, False

class EnhancedClusteringModel(BaseMLModel):
    """Enhanced clustering model with better anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("clustering", config)
        self.model = KMeans(
            n_clusters=config.get('n_clusters', 8),  # Increased clusters
            random_state=42,
            n_init=20,  # More initializations
            max_iter=500
        )
        self.cluster_stats = {}
        self.cluster_centers = None
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Enhanced training with better cluster analysis"""
        if not training_data:
            return
        
        self.logger.info(f"Training {self.name} with {len(training_data)} samples")
        
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
        self.is_trained = True
        
        self.logger.info(f"Clustering training complete. Found {self.model.n_clusters} clusters")
        for cid, stats in self.cluster_stats.items():
            self.logger.info(f"Cluster {cid}: {stats['sample_count']} samples, "
                           f"threat_ratio={stats['threat_ratio']:.2f}, "
                           f"threshold={stats['threshold']:.3f}")
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with cluster-aware scoring"""
        if not self.is_trained:
            return 0.0, False
        
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
            
            return normalized_score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            return 0.0, False

class MLModelManager:
    """Enhanced ML Model Manager with adaptive threshold management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.training_data = deque(maxlen=2000)  # Larger buffer
        self.global_threshold_manager = AdaptiveThresholdManager()
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.prediction_history = deque(maxlen=1000)
        self.alert_rate_history = deque(maxlen=100)
    
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
        
        # Keep simple models for comparison
        if model_configs.get('forbidden_ratio', {}).get('enabled', True):
            self.models['forbidden_ratio'] = EnhancedForbiddenRatioModel(
                model_configs.get('forbidden_ratio', {})
            )
        
        self.logger.info(f"Initialized {len(self.models)} enhanced ML models")
    
    def train_models(self, training_data: List[Dict[str, Any]] = None):
        """Enhanced model training with balanced data"""
        if training_data is None:
            training_data = list(self.training_data)
        
        if not training_data:
            self.logger.warning("No training data available")
            return
        
        # Analyze training data distribution
        threat_logs = [log for log in training_data if log.get('threat_indicators', [])]
        normal_logs = [log for log in training_data if not log.get('threat_indicators', [])]
        
        self.logger.info(f"Training data: {len(normal_logs)} normal, {len(threat_logs)} threats "
                        f"({len(threat_logs)/len(training_data)*100:.1f}% threats)")
        
        # Train each model
        for name, model in self.models.items():
            if model.enabled:
                try:
                    start_time = time.time()
                    model.train(training_data)
                    train_time = time.time() - start_time
                    self.logger.info(f"Trained {name} model in {train_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"Failed to train {name} model: {e}")
    
    def predict_all(self, log_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Enhanced prediction with score calibration"""
        predictions = {}
        scores = []
        
        for name, model in self.models.items():
            if model.enabled:
                try:
                    score, is_anomaly = model.predict(log_data)
                    predictions[name] = {
                        'score': score,
                        'is_anomaly': is_anomaly,
                        'timestamp': datetime.now()
                    }
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error in {name} prediction: {e}")
                    predictions[name] = {
                        'score': 0.0,
                        'is_anomaly': False,
                        'timestamp': datetime.now(),
                        'error': str(e)
                    }
        
        # Update global threshold manager
        if scores:
            self.global_threshold_manager.update_score_history(scores)
        
        # Track prediction history for performance monitoring
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': predictions,
            'entity_id': log_data.get('source_ip', 'unknown')
        })
        
        return predictions
    
    def get_adaptive_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        return self.global_threshold_manager.calculate_adaptive_thresholds()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with alert rate analysis"""
        recent_predictions = list(self.prediction_history)[-100:]
        
        if not recent_predictions:
            return {}
        
        # Calculate alert rates per model
        model_stats = {}
        for name in self.models.keys():
            model_predictions = [p['predictions'].get(name, {}) for p in recent_predictions]
            model_predictions = [p for p in model_predictions if p]
            
            if model_predictions:
                alert_rate = sum(1 for p in model_predictions if p.get('is_anomaly', False)) / len(model_predictions)
                avg_score = np.mean([p.get('score', 0) for p in model_predictions])
                
                model_stats[name] = {
                    'alert_rate': alert_rate,
                    'avg_score': avg_score,
                    'predictions_count': len(model_predictions)
                }
        
        return {
            'model_stats': model_stats,
            'adaptive_thresholds': self.get_adaptive_thresholds(),
            'total_predictions': len(recent_predictions),
            'global_alert_rate': sum(
                1 for p in recent_predictions 
                if any(pred.get('is_anomaly', False) for pred in p['predictions'].values())
            ) / len(recent_predictions) if recent_predictions else 0
        }
    
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
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models - REQUIRED BY ORCHESTRATOR"""
        status = {}
        for name, model in self.models.items():
            status[name] = {
                'enabled': model.enabled,
                'trained': model.is_trained,
                'performance_metrics': model.performance_metrics,
                'model_type': type(model).__name__
            }
        return status
    
    def add_training_data(self, log_data: Dict[str, Any]):
        """Add data to training buffer"""
        self.training_data.append(log_data)
        self.logger.debug(f"Added log data to training buffer (buffer size: {len(self.training_data)})")

class EnhancedForbiddenRatioModel(BaseMLModel):
    """Enhanced forbidden ratio model with adaptive thresholds"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("forbidden_ratio", config)
        self.adaptive_thresholds = {
            'high_port_ratio': 0.8,
            'threat_indicator_ratio': 0.3,
            'suspicious_protocol_ratio': 0.4
        }
        self.window_data = deque(maxlen=200)  # Larger window
        self.baseline_ratios = {}
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train with adaptive baseline calculation"""
        if not training_data:
            return
        
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
        
        self.is_trained = True
        self.logger.info(f"Trained {self.name}: baselines={self.baseline_ratios}, thresholds={self.adaptive_thresholds}")
    
    def predict(self, log_data: Dict[str, Any]) -> Tuple[float, bool]:
        """Enhanced prediction with multiple ratio checks"""
        if not self.is_trained:
            return 0.0, False
        
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
            
            return anomaly_score, is_anomaly
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            return 0.0, False