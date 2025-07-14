# enhanced_scoring_engine_fixed.py - FIXED alert generation with better thresholds and debugging

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import uuid
import statistics

class EnhancedAdaptiveThresholdManager:
    """Enhanced threshold manager with better calibration and debugging"""
    
    def __init__(self, target_alert_rate: float = 0.05, config: Dict[str, Any] = None):
        self.target_alert_rate = target_alert_rate
        
        # Load thresholds from configuration if available
        if config and 'adaptive_thresholds' in config and 'initial_thresholds' in config['adaptive_thresholds']:
            initial_thresholds = config['adaptive_thresholds']['initial_thresholds']
            self.current_thresholds = {
                'low': initial_thresholds.get('low', 0.60),
                'medium': initial_thresholds.get('medium', 0.70),
                'high': initial_thresholds.get('high', 0.80),
                'critical': initial_thresholds.get('critical', 0.90)
            }
        else:
            # Default thresholds if no config provided
            self.current_thresholds = {
                'low': 0.60,
                'medium': 0.70,
                'high': 0.80,
                'critical': 0.90
            }
        
        self.threshold_history = deque(maxlen=100)
        self.score_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=500)
        self.logger = logging.getLogger(__name__)
        
        # Log loaded thresholds
        self.logger.info(f"🎯 Loaded thresholds from config: {self.current_thresholds}")
        
        # Performance tracking
        self.calibration_stats = {
            'adjustments_made': 0,
            'last_adjustment': None,
            'score_distribution_history': deque(maxlen=20)
        }
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        return self.current_thresholds.copy()
    
    def update_score_history(self, scores: List[float]):
        """Update score history and track distribution"""
        if not scores:
            return
            
        self.score_history.extend(scores)
        
        # Track score distribution for calibration
        score_stats = {
            'timestamp': datetime.now(),
            'count': len(scores),
            'min': min(scores),
            'max': max(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'percentiles': {
                'p25': np.percentile(scores, 25),
                'p50': np.percentile(scores, 50),
                'p75': np.percentile(scores, 75),
                'p90': np.percentile(scores, 90),
                'p95': np.percentile(scores, 95)
            }
        }
        
        self.calibration_stats['score_distribution_history'].append(score_stats)
        
        # Log distribution periodically
        if len(self.calibration_stats['score_distribution_history']) % 5 == 0:
            self._log_score_distribution(score_stats)
    
    def _log_score_distribution(self, stats: Dict[str, Any]):
        """Log current score distribution for debugging"""
        self.logger.info(
            f"📊 Score Distribution: "
            f"mean={stats['mean']:.3f}, "
            f"median={stats['median']:.3f}, "
            f"std={stats['std']:.3f}"
        )
        self.logger.info(
            f"📈 Percentiles: "
            f"P25={stats['percentiles']['p25']:.3f}, "
            f"P75={stats['percentiles']['p75']:.3f}, "
            f"P90={stats['percentiles']['p90']:.3f}, "
            f"P95={stats['percentiles']['p95']:.3f}"
        )
        
        # Show how many would alert with current thresholds
        recent_scores = list(self.score_history)[-100:] if len(self.score_history) >= 100 else list(self.score_history)
        if recent_scores:
            alert_counts = self._count_alerts_by_threshold(recent_scores)
            total = len(recent_scores)
            self.logger.info(
                f"🚨 Alert Distribution (last {total} scores): "
                f"Low: {alert_counts['low']}/{total} ({alert_counts['low']/total*100:.1f}%), "
                f"Medium: {alert_counts['medium']}/{total} ({alert_counts['medium']/total*100:.1f}%), "
                f"High: {alert_counts['high']}/{total} ({alert_counts['high']/total*100:.1f}%), "
                f"Critical: {alert_counts['critical']}/{total} ({alert_counts['critical']/total*100:.1f}%)"
            )
    
    def _count_alerts_by_threshold(self, scores: List[float]) -> Dict[str, int]:
        """Count how many scores would generate alerts at each threshold"""
        counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for score in scores:
            if score >= self.current_thresholds['critical']:
                counts['critical'] += 1
            elif score >= self.current_thresholds['high']:
                counts['high'] += 1
            elif score >= self.current_thresholds['medium']:
                counts['medium'] += 1
            elif score >= self.current_thresholds['low']:
                counts['low'] += 1
        
        return counts
    
    def calculate_current_alert_rate(self, recent_scores: List[float]) -> float:
        """Calculate current alert rate from recent scores"""
        if not recent_scores:
            return 0.0
        
        # Count scores that would generate alerts (any threshold)
        alerts = sum(1 for score in recent_scores if score >= self.current_thresholds['low'])
        return alerts / len(recent_scores)
    
    def calibrate_thresholds(self) -> bool:
        """Enhanced threshold calibration with better logic"""
        if len(self.score_history) < 50:
            return False
        
        recent_scores = list(self.score_history)[-200:]  # Use more data
        current_alert_rate = self.calculate_current_alert_rate(recent_scores)
        
        # More conservative adjustment logic
        adjustment_needed = False
        adjustment_size = 0.02  # Smaller adjustments
        
        if current_alert_rate > self.target_alert_rate * 2.0:  # Too many alerts
            self._increase_thresholds(adjustment_size)
            adjustment_needed = True
            self.logger.info(f"🔧 Increased thresholds - alert rate: {current_alert_rate:.3f} > target: {self.target_alert_rate:.3f}")
            
        elif current_alert_rate < self.target_alert_rate * 0.3:  # Too few alerts
            self._decrease_thresholds(adjustment_size)
            adjustment_needed = True
            self.logger.info(f"🔧 Decreased thresholds - alert rate: {current_alert_rate:.3f} < target: {self.target_alert_rate:.3f}")
        
        if adjustment_needed:
            self.calibration_stats['adjustments_made'] += 1
            self.calibration_stats['last_adjustment'] = datetime.now()
            
            self.threshold_history.append({
                'timestamp': datetime.now(),
                'thresholds': self.current_thresholds.copy(),
                'alert_rate': current_alert_rate,
                'target_rate': self.target_alert_rate
            })
        
        return adjustment_needed
    
    def _increase_thresholds(self, increment: float):
        """Increase thresholds while maintaining order"""
        for key in self.current_thresholds:
            self.current_thresholds[key] = min(0.95, self.current_thresholds[key] + increment)
        self._ensure_threshold_order()
    
    def _decrease_thresholds(self, decrement: float):
        """Decrease thresholds while maintaining order"""
        for key in self.current_thresholds:
            self.current_thresholds[key] = max(0.05, self.current_thresholds[key] - decrement)
        self._ensure_threshold_order()
    
    def _ensure_threshold_order(self):
        """Ensure thresholds maintain proper order with minimum gaps"""
        min_gap = 0.05
        
        # Ensure ascending order
        if self.current_thresholds['medium'] < self.current_thresholds['low'] + min_gap:
            self.current_thresholds['medium'] = self.current_thresholds['low'] + min_gap
        
        if self.current_thresholds['high'] < self.current_thresholds['medium'] + min_gap:
            self.current_thresholds['high'] = self.current_thresholds['medium'] + min_gap
        
        if self.current_thresholds['critical'] < self.current_thresholds['high'] + min_gap:
            self.current_thresholds['critical'] = self.current_thresholds['high'] + min_gap
        
        # Cap at 0.95
        for key in self.current_thresholds:
            self.current_thresholds[key] = min(0.95, self.current_thresholds[key])

class FixedAdaptiveScoringEngine:
    """FIXED Adaptive Scoring Engine with improved alert generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enhanced model weights
        self.model_weights = config.get('model_weights', {
            'isolation_forest': 0.4,
            'clustering': 0.3,
            'forbidden_ratio': 0.3
        })
        
        # Enhanced threshold manager - PASS CONFIG TO LOAD THRESHOLDS
        self.threshold_manager = EnhancedAdaptiveThresholdManager(
            target_alert_rate=config.get('target_alert_rate', 0.05),
            config=config  # Pass full config to load thresholds
        )
        
        # Entity tracking with enhanced metadata
        self.entity_scores = defaultdict(lambda: {
            'current_score': 0.0,
            'history': deque(maxlen=200),
            'first_seen': datetime.now(),
            'last_updated': datetime.now(),
            'alert_count': 0,
            'contributing_models': set(),
            'score_trend': 'stable',
            'risk_level': 'low',
            'confidence': 0.0,
            'recent_alerts': deque(maxlen=10)
        })
        
        # Enhanced scoring methods
        self.aggregation_methods = {
            'adaptive_weighted': self._adaptive_weighted_average,
            'confidence_weighted': self._confidence_weighted_average,
            'ensemble_consensus': self._ensemble_consensus,
            'threat_focused': self._threat_focused_scoring
        }
        
        self.aggregation_method = config.get('aggregation_method', 'adaptive_weighted')
        
        # Performance tracking with debugging
        self.scoring_metrics = {
            'total_scored': 0,
            'alerts_generated': 0,
            'alerts_by_severity': defaultdict(int),
            'processing_time': deque(maxlen=1000),
            'score_distribution': deque(maxlen=2000),
            'alert_rate_history': deque(maxlen=100),
            'alert_generation_reasons': defaultdict(int)
        }
        
        # Enhanced auto-tagging
        self.tagging_rules = self._initialize_enhanced_tagging_rules()
        
        # Start background processes
        self.running = True
        self._start_calibration_thread()
        self._start_metrics_thread()
    
    def _start_calibration_thread(self):
        """Start enhanced calibration thread"""
        def calibration_loop():
            while self.running:
                try:
                    self._perform_enhanced_calibration()
                    time.sleep(60)  # Calibrate every minute
                except Exception as e:
                    self.logger.error(f"Calibration error: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=calibration_loop, daemon=True)
        thread.start()
        self.logger.info("Started enhanced calibration thread")
    
    def _start_metrics_thread(self):
        """Start metrics reporting thread"""
        def metrics_loop():
            while self.running:
                try:
                    self._report_detailed_metrics()
                    time.sleep(300)  # Report every 5 minutes
                except Exception as e:
                    self.logger.error(f"Metrics reporting error: {e}")
                    time.sleep(300)
        
        thread = threading.Thread(target=metrics_loop, daemon=True)
        thread.start()
        self.logger.info("Started metrics reporting thread")
    
    def _perform_enhanced_calibration(self):
        """Perform enhanced calibration with debugging"""
        try:
            # Update score history
            recent_scores = list(self.scoring_metrics['score_distribution'])[-100:]
            if recent_scores:
                self.threshold_manager.update_score_history(recent_scores)
                
                # Perform threshold calibration
                adjusted = self.threshold_manager.calibrate_thresholds()
                if adjusted:
                    new_thresholds = self.threshold_manager.get_current_thresholds()
                    self.logger.info(f"🔧 Adjusted thresholds: {new_thresholds}")
            
            # Update model weights
            self._update_model_weights()
            
            # Clean up old entities
            self._cleanup_old_entities()
            
        except Exception as e:
            self.logger.error(f"Enhanced calibration error: {e}")
    
    def _report_detailed_metrics(self):
        """Report detailed metrics for debugging"""
        try:
            total_scored = self.scoring_metrics['total_scored']
            alerts_generated = self.scoring_metrics['alerts_generated']
            
            if total_scored > 0:
                alert_rate = alerts_generated / total_scored
                
                self.logger.info(
                    f"📊 DETAILED METRICS (5min): "
                    f"Scored: {total_scored}, "
                    f"Alerts: {alerts_generated}, "
                    f"Rate: {alert_rate:.3f}"
                )
                
                # Alert breakdown by severity
                severity_breakdown = dict(self.scoring_metrics['alerts_by_severity'])
                if severity_breakdown:
                    self.logger.info(f"🚨 Alert Severity: {severity_breakdown}")
                
                # Alert generation reasons
                reasons = dict(self.scoring_metrics['alert_generation_reasons'])
                if reasons:
                    self.logger.info(f"💡 Alert Reasons: {reasons}")
                
                # Current thresholds
                thresholds = self.threshold_manager.get_current_thresholds()
                self.logger.info(f"🎯 Current Thresholds: {thresholds}")
                
                # Entity statistics
                active_entities = len([e for e in self.entity_scores.values() 
                                     if (datetime.now() - e['last_updated']).seconds < 3600])
                self.logger.info(f"🏢 Active Entities: {active_entities}")
        
        except Exception as e:
            self.logger.error(f"Metrics reporting error: {e}")
    
    def aggregate_scores(self, entity_id: str, model_predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced score aggregation with improved logic"""
        start_time = time.time()
        
        try:
            # Extract scores with LOWER threshold for inclusion
            model_scores = {}
            model_confidences = {}
            contributing_models = set()
            
            for model_name, prediction in model_predictions.items():
                score = prediction.get('score', 0.0)
                is_anomaly = prediction.get('is_anomaly', False)
                
                # LOWERED inclusion threshold from 0.1 to 0.05
                if score > 0.05 or is_anomaly:
                    model_scores[model_name] = score
                    
                    # Enhanced confidence calculation
                    confidence = score
                    if is_anomaly:
                        confidence = min(1.0, confidence + 0.15)  # Smaller boost
                    
                    model_confidences[model_name] = confidence
                    contributing_models.add(model_name)
            
            # If no models contributed, force at least one
            if not model_scores and model_predictions:
                best_model = max(model_predictions.items(), 
                               key=lambda x: x[1].get('score', 0.0))
                model_name, prediction = best_model
                score = max(prediction.get('score', 0.0), 0.1)  # Minimum score
                
                model_scores[model_name] = score
                model_confidences[model_name] = score
                contributing_models.add(model_name)
                
                self.logger.debug(f"Forced inclusion of {model_name} with score {score:.3f} for {entity_id}")
            
            # Aggregate using selected method
            aggregation_func = self.aggregation_methods.get(
                self.aggregation_method, self._adaptive_weighted_average
            )
            aggregated_score, overall_confidence = aggregation_func(model_scores, model_confidences)
            
            # Enhanced temporal smoothing
            entity_data = self.entity_scores[entity_id]
            if len(entity_data['history']) > 0:
                prev_score = entity_data['current_score']
                # Reduced smoothing for better responsiveness
                smoothing_factor = 0.2  # Reduced from 0.3
                aggregated_score = (1 - smoothing_factor) * aggregated_score + smoothing_factor * prev_score
            
            # Update entity tracking
            entity_data['current_score'] = aggregated_score
            entity_data['history'].append(aggregated_score)
            entity_data['last_updated'] = datetime.now()
            entity_data['contributing_models'].update(contributing_models)
            entity_data['confidence'] = overall_confidence
            
            # Calculate enhanced metrics
            entity_data['score_trend'] = self._calculate_score_trend(entity_data['history'])
            
            # Determine risk level using current thresholds
            adaptive_thresholds = self.threshold_manager.get_current_thresholds()
            risk_level = self._determine_risk_level(aggregated_score, adaptive_thresholds)
            entity_data['risk_level'] = risk_level
            
            # Enhanced auto-tagging
            tags, actions = self._apply_enhanced_tagging_rules(entity_data, model_predictions)
            
            # Create comprehensive result
            result = {
                'entity_id': entity_id,
                'aggregated_score': aggregated_score,
                'confidence': overall_confidence,
                'risk_level': risk_level,
                'severity': self._map_risk_to_severity(risk_level),
                'contributing_models': list(contributing_models),
                'model_scores': model_scores,
                'model_confidences': model_confidences,
                'tags': tags,
                'actions': actions,
                'timestamp': datetime.now(),
                'score_trend': entity_data['score_trend'],
                'entity_metadata': {
                    'first_seen': entity_data['first_seen'],
                    'last_updated': entity_data['last_updated'],
                    'alert_count': entity_data['alert_count'],
                    'score_history_length': len(entity_data['history']),
                    'avg_score_24h': self._calculate_avg_score_24h(entity_data)
                },
                'adaptive_thresholds': adaptive_thresholds,
                'debug_info': {
                    'original_model_predictions': model_predictions,
                    'included_models': list(contributing_models),
                    'aggregation_method': self.aggregation_method
                }
            }
            
            # Track metrics
            processing_time = time.time() - start_time
            self.scoring_metrics['processing_time'].append(processing_time)
            self.scoring_metrics['total_scored'] += 1
            self.scoring_metrics['score_distribution'].append(aggregated_score)
            
            # DEBUG: Log scoring details periodically
            if self.scoring_metrics['total_scored'] % 50 == 0:
                self.logger.debug(
                    f"🔍 Scoring Debug - Entity: {entity_id}, "
                    f"Score: {aggregated_score:.3f}, "
                    f"Risk: {risk_level}, "
                    f"Models: {list(contributing_models)}, "
                    f"Confidence: {overall_confidence:.3f}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced score aggregation for {entity_id}: {e}")
            return {
                'entity_id': entity_id,
                'aggregated_score': 0.0,
                'confidence': 0.0,
                'risk_level': 'low',
                'severity': 'info',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def should_generate_alert(self, scoring_result: Dict[str, Any]) -> bool:
        """FINE-TUNED alert generation logic - balanced approach"""
        score = scoring_result.get('aggregated_score', 0.0)
        confidence = scoring_result.get('confidence', 0.0)
        risk_level = scoring_result.get('risk_level', 'low')
        entity_id = scoring_result.get('entity_id', 'unknown')
        contributing_models = scoring_result.get('contributing_models', [])
        tags = scoring_result.get('tags', [])
        score_trend = scoring_result.get('score_trend', 'stable')
        
        alert_reasons = []
        should_alert = False
        
        # FINE-TUNED LOGIC - Balanced between too strict and too loose
        
        # Rule 1: Critical risk ALWAYS alerts
        if risk_level == 'critical':
            should_alert = True
            alert_reasons.append('critical_risk')
        
        # Rule 2: High risk with good confidence
        elif risk_level == 'high' and confidence > 0.7:  # Higher confidence required
            should_alert = True
            alert_reasons.append('high_risk_confident')
        
        # Rule 3: Medium risk with stricter conditions
        elif risk_level == 'medium':
            if confidence > 0.75:  # Higher confidence for medium
                should_alert = True
                alert_reasons.append('medium_risk_confident')
            elif len(contributing_models) >= 2 and confidence > 0.6:  # Multiple models + decent confidence
                should_alert = True
                alert_reasons.append('medium_risk_multi_model')
        
        # Rule 4: Low risk only with very high confidence
        elif risk_level == 'low' and confidence > 0.85 and len(contributing_models) >= 2:
            should_alert = True
            alert_reasons.append('low_risk_exceptional')
        
        # Rule 5: Multiple model strong consensus
        elif len(contributing_models) >= 3 and score > 0.6 and confidence > 0.7:
            should_alert = True
            alert_reasons.append('strong_multi_model_consensus')
        
        # Rule 6: Priority tags with score requirement
        critical_priority_tags = {'critical_threat', 'persistent_threat'}
        if any(tag in critical_priority_tags for tag in tags) and score > 0.65:
            should_alert = True
            alert_reasons.append('priority_tags')
        
        # Rule 7: Very high score override
        elif score > 0.85:  # Only very high scores
            should_alert = True
            alert_reasons.append('very_high_score_override')
        
        # DEBUG: Log decision making for debugging (reduced frequency)
        if should_alert or (self.scoring_metrics['total_scored'] % 100 == 0):
            self.logger.info(
                f"🔍 Alert Decision for {entity_id}: "
                f"score={score:.3f}, risk={risk_level}, confidence={confidence:.3f}, "
                f"models={len(contributing_models)}, alert={should_alert}, reasons={alert_reasons}"
            )
        
        if should_alert:
            # Track alert generation reasons
            for reason in alert_reasons:
                self.scoring_metrics['alert_generation_reasons'][reason] += 1
        
        return should_alert
    
    # Include other methods from original class...
    def _adaptive_weighted_average(self, model_scores: Dict[str, float], 
                                   model_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Adaptive weighted average based on model confidence"""
        if not model_scores:
            return 0.0, 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for model_name, score in model_scores.items():
            base_weight = self.model_weights.get(model_name, 0.25)
            confidence = model_confidences.get(model_name, 0.5)
            
            # Enhanced adaptive weight calculation
            adaptive_weight = base_weight * (0.3 + confidence)  # Increased minimum
            
            weighted_sum += score * adaptive_weight
            total_weight += adaptive_weight
            confidence_sum += confidence
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        avg_confidence = confidence_sum / len(model_scores) if model_scores else 0.0
        
        return final_score, avg_confidence
    
    def _confidence_weighted_average(self, model_scores: Dict[str, float], 
                                    model_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Confidence-weighted scoring"""
        if not model_scores:
            return 0.0, 0.0
        
        confidence_weights = []
        scores = []
        
        for model_name, score in model_scores.items():
            confidence = model_confidences.get(model_name, 0.5)
            confidence_weights.append(confidence)
            scores.append(score)
        
        if not scores:
            return 0.0, 0.0
        
        # Weight by confidence squared (emphasize high-confidence predictions)
        conf_weights = np.array(confidence_weights) ** 2
        conf_weights = conf_weights / np.sum(conf_weights) if np.sum(conf_weights) > 0 else np.ones_like(conf_weights)
        
        final_score = np.average(scores, weights=conf_weights)
        avg_confidence = np.mean(confidence_weights)
        
        return final_score, avg_confidence
    
    def _ensemble_consensus(self, model_scores: Dict[str, float], 
                           model_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Ensemble consensus with outlier detection"""
        if not model_scores:
            return 0.0, 0.0
        
        scores = list(model_scores.values())
        confidences = list(model_confidences.values())
        
        if len(scores) == 1:
            return scores[0], confidences[0]
        
        # Remove outlier scores
        if len(scores) >= 3:
            median_score = np.median(scores)
            mad = np.median(np.abs(np.array(scores) - median_score))
            
            filtered_scores = []
            filtered_confidences = []
            
            for score, conf in zip(scores, confidences):
                if mad == 0 or abs(score - median_score) <= 2 * mad:
                    filtered_scores.append(score)
                    filtered_confidences.append(conf)
            
            if filtered_scores:
                scores = filtered_scores
                confidences = filtered_confidences
        
        # Calculate consensus score
        final_score = np.mean(scores)
        avg_confidence = np.mean(confidences)
        
        # Boost confidence if models agree
        score_std = np.std(scores) if len(scores) > 1 else 0
        agreement_bonus = max(0, 1 - score_std) * 0.2
        avg_confidence = min(1.0, avg_confidence + agreement_bonus)
        
        return final_score, avg_confidence
    
    def _threat_focused_scoring(self, model_scores: Dict[str, float], 
                               model_confidences: Dict[str, float]) -> Tuple[float, float]:
        """Threat-focused scoring that emphasizes high scores"""
        if not model_scores:
            return 0.0, 0.0
        
        scores = list(model_scores.values())
        confidences = list(model_confidences.values())
        
        # Use max score with confidence weighting
        max_score = max(scores)
        max_idx = scores.index(max_score)
        max_confidence = confidences[max_idx]
        
        # If multiple models detect threats, boost the score
        threat_models = sum(1 for score in scores if score > 0.6)
        boost_factor = 1.0 + (threat_models - 1) * 0.1
        
        final_score = min(1.0, max_score * boost_factor)
        avg_confidence = np.mean(confidences)
        
        return final_score, avg_confidence
    
    def _determine_risk_level(self, score: float, thresholds: Dict[str, float]) -> str:
        """Determine risk level using adaptive thresholds"""
        if score >= thresholds.get('critical', 0.8):
            return 'critical'
        elif score >= thresholds.get('high', 0.7):
            return 'high'
        elif score >= thresholds.get('medium', 0.55):
            return 'medium'
        elif score >= thresholds.get('low', 0.4):
            return 'low'
        else:
            return 'minimal'
    
    def _map_risk_to_severity(self, risk_level: str) -> str:
        """Map risk level to alert severity"""
        mapping = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low',
            'minimal': 'info'
        }
        return mapping.get(risk_level, 'info')
    
    def _calculate_score_trend(self, history: deque) -> str:
        """Calculate score trend from history"""
        if len(history) < 5:
            return 'insufficient_data'
        
        recent_scores = list(history)[-5:]
        older_scores = list(history)[-10:-5] if len(history) >= 10 else list(history)[:-5]
        
        if not older_scores:
            return 'new_entity'
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        change_threshold = 0.1  # Reduced from 0.15 for better sensitivity
        
        if recent_avg > older_avg + change_threshold:
            return 'increasing'
        elif recent_avg < older_avg - change_threshold:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_avg_score_24h(self, entity_data: Dict[str, Any]) -> float:
        """Calculate average score for last period"""
        history = list(entity_data['history'])
        if len(history) >= 10:
            return np.mean(history[-10:])
        elif history:
            return np.mean(history)
        else:
            return 0.0
    
    def _initialize_enhanced_tagging_rules(self) -> List[Dict[str, Any]]:
        """Initialize MUCH MORE RESTRICTIVE auto-tagging rules"""
        return [
            {
                'name': 'critical_threat',
                'condition': lambda entity_data: (
                    entity_data['current_score'] > 0.9 and   # MUCH higher threshold
                    entity_data['confidence'] > 0.8
                ),
                'tags': ['critical_threat', 'immediate_response'],
                'actions': ['isolate_entity', 'emergency_alert', 'detailed_analysis']
            },
            {
                'name': 'persistent_high_risk',
                'condition': lambda entity_data: (
                    len(entity_data['history']) > 10 and     # MORE history required
                    np.mean(list(entity_data['history'])[-5:]) > 0.8 and  # HIGHER threshold
                    entity_data['score_trend'] == 'increasing'  # ONLY increasing, not stable
                ),
                'tags': ['persistent_threat', 'behavioral_anomaly'],
                'actions': ['escalate_alert', 'behavioral_analysis']
            },
            {
                'name': 'multi_model_consensus',
                'condition': lambda entity_data: (
                    len(entity_data['contributing_models']) >= 3 and  # At least 3 models
                    entity_data['confidence'] > 0.8 and              # HIGHER confidence
                    entity_data['current_score'] > 0.7               # HIGHER score
                ),
                'tags': ['multi_model_detection', 'validated_threat'],
                'actions': ['priority_investigation']
            },
            {
                'name': 'escalating_threat',
                'condition': lambda entity_data: (
                    entity_data['score_trend'] == 'increasing' and
                    entity_data['current_score'] > 0.8 and           # MUCH higher threshold
                    len(entity_data['history']) > 5 and
                    np.mean(list(entity_data['history'])[-3:]) > 0.75  # Recent high scores
                ),
                'tags': ['escalating_threat'],
                'actions': ['enhanced_monitoring']
            }
        ]
    
    def _apply_enhanced_tagging_rules(self, entity_data: Dict[str, Any], 
                                     model_predictions: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Apply enhanced auto-tagging rules"""
        tags = []
        actions = []
        
        for rule in self.tagging_rules:
            try:
                if rule['condition'](entity_data):
                    tags.extend(rule['tags'])
                    actions.extend(rule['actions'])
            except Exception as e:
                self.logger.error(f"Error applying tagging rule {rule['name']}: {e}")
        
        # Add model-specific tags
        for model_name, prediction in model_predictions.items():
            if prediction.get('is_anomaly', False):
                tags.append(f"{model_name}_detection")
        
        return list(set(tags)), list(set(actions))
    
    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        try:
            recent_entities = [(eid, data) for eid, data in self.entity_scores.items() 
                              if (datetime.now() - data['last_updated']).seconds < 3600]
            
            if len(recent_entities) > 10:
                model_performance = defaultdict(list)
                
                for entity_id, data in recent_entities:
                    if data['confidence'] > 0.5:
                        for model in data['contributing_models']:
                            model_performance[model].append(data['confidence'])
                
                # Update weights based on average confidence per model
                total_weight = 0
                new_weights = {}
                
                for model_name in self.model_weights.keys():
                    if model_name in model_performance:
                        avg_confidence = np.mean(model_performance[model_name])
                        new_weights[model_name] = max(0.1, avg_confidence)
                    else:
                        new_weights[model_name] = 0.2  # Default weight
                    total_weight += new_weights[model_name]
                
                # Normalize weights
                if total_weight > 0:
                    for model_name in new_weights:
                        new_weights[model_name] /= total_weight
                    
                    # Smooth transition
                    for model_name in self.model_weights:
                        if model_name in new_weights:
                            self.model_weights[model_name] = (
                                0.9 * self.model_weights[model_name] + 
                                0.1 * new_weights[model_name]
                            )
                
                self.logger.debug(f"Updated model weights: {self.model_weights}")
        
        except Exception as e:
            self.logger.error(f"Model weight update error: {e}")
    
    def _cleanup_old_entities(self, hours: int = 24):
        """Clean up old entity data to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        entities_to_remove = [
            entity_id for entity_id, data in self.entity_scores.items()
            if data['last_updated'] < cutoff_time
        ]
        
        for entity_id in entities_to_remove:
            del self.entity_scores[entity_id]
        
        if entities_to_remove:
            self.logger.debug(f"Cleaned up {len(entities_to_remove)} old entities")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        recent_scores = list(self.scoring_metrics['score_distribution'])[-100:]
        
        metrics = {
            'total_scored': self.scoring_metrics['total_scored'],
            'alerts_generated': self.scoring_metrics['alerts_generated'],
            'alert_rate': (self.scoring_metrics['alerts_generated'] / 
                          max(1, self.scoring_metrics['total_scored'])),
            'entities_tracked': len(self.entity_scores),
            'current_thresholds': self.threshold_manager.get_current_thresholds(),
            'model_weights': self.model_weights.copy(),
            'alerts_by_severity': dict(self.scoring_metrics['alerts_by_severity']),
            'alert_generation_reasons': dict(self.scoring_metrics['alert_generation_reasons'])
        }
        
        if recent_scores:
            metrics.update({
                'score_stats': {
                    'min': min(recent_scores),
                    'max': max(recent_scores),
                    'mean': np.mean(recent_scores),
                    'median': np.median(recent_scores),
                    'std': np.std(recent_scores)
                }
            })
        
        if self.scoring_metrics['processing_time']:
            metrics['avg_processing_time'] = np.mean(
                list(self.scoring_metrics['processing_time'])[-100:]
            )
        
        return metrics
    
    def stop(self):
        """Stop the scoring engine"""
        self.running = False
        self.logger.info("Enhanced scoring engine stopped")