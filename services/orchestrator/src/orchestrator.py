# services/orchestrator.py - Complete orchestrator with RLOrchestrator and fallback

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import numpy as np

# Try to import RL dependencies with graceful fallback
RL_AVAILABLE = False
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RL dependencies not available: {e}")
    print("📝 Install with: pip install gym stable-baselines3")
    print("🔄 Falling back to simple orchestrator")

class SOCEnvironment:
    """RL Environment for SOC model orchestration - Only created if RL is available"""
    
    def __init__(self, model_manager, config: Dict[str, Any]):
        if not RL_AVAILABLE:
            raise ImportError("RL dependencies not available")
            
        # Initialize gym environment
        import gym
        from gym import spaces
        
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State space: [system_load, alert_rate, precision, recall, f1_score] for each model
        n_models = len(model_manager.models)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_models * 5 + 3,), dtype=np.float32
        )
        
        # Action space: Use Discrete for better SB3 compatibility
        self.action_space = spaces.Discrete(2**n_models)
        
        # Environment state
        self.current_state = None
        self.performance_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=100)
        self.system_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'processing_latency': 0.0
        }
        
        # Reward calculation parameters
        self.reward_weights = {
            'detection_accuracy': 0.4,
            'false_positive_penalty': -0.3,
            'resource_efficiency': 0.2,
            'coverage_bonus': 0.1
        }
        
        self.model_names = list(model_manager.models.keys())
        self.n_models = len(self.model_names)
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_state = self._get_current_state()
        return self.current_state
    
    def _get_current_state(self) -> np.ndarray:
        """Get current system state"""
        state = []
        
        # Model-specific metrics
        model_status = self.model_manager.get_model_status()
        for model_name in self.model_names:
            status = model_status.get(model_name, {})
            metrics = status.get('performance_metrics', {})
            
            state.extend([
                float(status.get('enabled', False)),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1_score', 0.0),
                1.0 if status.get('trained', False) else 0.0
            ])
        
        # System-wide metrics
        state.extend([
            self.system_metrics['cpu_usage'],
            self.system_metrics['memory_usage'],
            self.system_metrics['processing_latency']
        ])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        # Convert discrete action to binary array
        binary_action = self._discrete_to_binary(action)
        
        # Apply action (enable/disable models)
        for i, enable in enumerate(binary_action):
            model_name = self.model_names[i]
            if enable:
                self.model_manager.enable_model(model_name)
            else:
                self.model_manager.disable_model(model_name)
        
        # Wait briefly to observe effects
        time.sleep(0.1)
        
        # Get new state
        next_state = self._get_current_state()
        
        # Calculate reward
        reward = self._calculate_reward(binary_action, next_state)
        
        # Update state
        self.current_state = next_state
        
        # Episode never ends in continuous learning
        done = False
        
        info = {
            'active_models': int(sum(binary_action)),
            'system_load': float(self.system_metrics['cpu_usage']),
            'reward_components': self._get_reward_components(binary_action, next_state),
            'binary_action': binary_action.tolist()
        }
        
        return next_state, float(reward), done, info
    
    def _calculate_reward(self, action, state) -> float:
        """Calculate reward based on system performance"""
        reward = 0.0
        
        # Ensure action is iterable
        if hasattr(action, '__iter__'):
            active_models = sum(action)
        else:
            action = self._discrete_to_binary(action)
            active_models = sum(action)
        
        # Detection accuracy reward
        if self.performance_history:
            recent_performance = list(self.performance_history)[-10:]
            avg_f1 = np.mean([p.get('f1_score', 0) for p in recent_performance])
            reward += self.reward_weights['detection_accuracy'] * avg_f1
        
        # False positive penalty
        if self.alert_history:
            recent_alerts = list(self.alert_history)[-10:]
            false_positive_rate = sum(1 for a in recent_alerts if not a.get('verified', True)) / len(recent_alerts)
            reward += self.reward_weights['false_positive_penalty'] * false_positive_rate
        
        # Resource efficiency reward
        max_models = len(self.model_names)
        efficiency = 1.0 - (self.system_metrics['cpu_usage'] * active_models / max_models)
        reward += self.reward_weights['resource_efficiency'] * efficiency
        
        # Coverage bonus (encourage having some models active)
        if active_models > 0:
            coverage = min(active_models / max_models, 1.0)
            reward += self.reward_weights['coverage_bonus'] * coverage
        
        return float(reward)
    
    def _get_reward_components(self, action, state) -> Dict[str, float]:
        """Get detailed reward components for logging"""
        components = {}
        
        if self.performance_history:
            recent_performance = list(self.performance_history)[-10:]
            components['avg_f1'] = np.mean([p.get('f1_score', 0) for p in recent_performance])
        
        components['active_models'] = sum(action)
        components['cpu_usage'] = self.system_metrics['cpu_usage']
        
        return components
    
    def _discrete_to_binary(self, action: int) -> np.ndarray:
        """Convert discrete action to binary array"""
        action = int(action)
        
        binary_action = []
        for i in range(self.n_models):
            binary_action.append((action >> i) & 1)
        
        return np.array(binary_action, dtype=bool)

    def update_system_metrics(self, metrics: Dict[str, float]):
        """Update system performance metrics"""
        self.system_metrics.update(metrics)
    
    def add_performance_data(self, performance: Dict[str, Any]):
        """Add performance data for reward calculation"""
        self.performance_history.append(performance)
    
    def add_alert_data(self, alert: Dict[str, Any]):
        """Add alert data for reward calculation"""
        self.alert_history.append(alert)

class RLOrchestrator:
    """Reinforcement Learning-based model orchestrator with fallback"""
    
    def __init__(self, model_manager, config: Dict[str, Any]):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize RL components if available
        self.rl_enabled = RL_AVAILABLE
        self.env = None
        self.vec_env = None
        self.agent = None
        
        if self.rl_enabled:
            try:
                self._initialize_rl_components()
                self.logger.info("✅ RL Orchestrator initialized with reinforcement learning")
            except Exception as e:
                self.logger.warning(f"⚠️ RL initialization failed: {e}")
                self.rl_enabled = False
                self.logger.info("🔄 Falling back to rule-based orchestrator")
        else:
            self.logger.info("📝 RL Orchestrator using rule-based fallback (RL dependencies unavailable)")
        
        # Control parameters
        self.running = False
        self.decision_interval = config.get('decision_interval', 30)
        self.learning_enabled = True
        self.exploration_rate = config.get('exploration_rate', 0.1)
        
        # Performance tracking
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
        # Simple orchestrator state (for fallback)
        self.simple_policy = {
            model_name: True for model_name in self.model_manager.models.keys()
        }
    
    def _initialize_rl_components(self):
        """Initialize RL components (only if dependencies available)"""
        if not RL_AVAILABLE:
            raise ImportError("RL dependencies not available")
        
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create RL environment
        self.env = SOCEnvironment(self.model_manager, self.config)
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize RL agent
        self.agent = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.get('learning_rate', 0.001),
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,  # Reduce verbosity
            tensorboard_log=None  # Disable tensorboard to avoid issues
        )
    
    def start(self):
        """Start the orchestrator"""
        self.running = True
        
        def orchestration_loop():
            while self.running:
                try:
                    if self.rl_enabled:
                        self._make_rl_decision()
                    else:
                        self._make_simple_decision()
                    
                    # Learn from recent experiences (only if RL enabled)
                    if self.rl_enabled and self.learning_enabled:
                        self._update_learning()
                    
                    time.sleep(self.decision_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in orchestration loop: {e}")
                    time.sleep(self.decision_interval)
        
        thread = threading.Thread(target=orchestration_loop)
        thread.daemon = True
        thread.start()
        
        orchestrator_type = "RL-based" if self.rl_enabled else "Rule-based"
        self.logger.info(f"Started {orchestrator_type} orchestrator")
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        self.logger.info("Stopped orchestrator")
    
    def _make_rl_decision(self):
        """Make RL-based decision about model configuration"""
        try:
            # Get current state
            current_state = self.env._get_current_state()
            
            # Decide whether to explore or exploit
            if np.random.random() < self.exploration_rate:
                action = self.env.action_space.sample()
                decision_type = "exploration"
            else:
                action, _ = self.agent.predict(current_state, deterministic=False)
                decision_type = "exploitation"
            
            # Ensure action is Python int
            action = int(action)
            
            # Apply action
            next_state, reward, done, info = self.env.step(action)
            
            # Get binary action from info
            binary_action = info.get('binary_action', [])
            
            # Log decision
            decision_log = {
                'timestamp': datetime.now(),
                'action': action,
                'binary_action': binary_action,
                'reward': float(reward),
                'decision_type': decision_type,
                'info': {k: v for k, v in info.items() if k != 'binary_action'},
                'state': current_state.tolist()
            }
            
            self.decision_history.append(decision_log)
            
            self.logger.info(
                f"RL Decision [{decision_type}]: "
                f"Action: {action}, Binary: {binary_action}, "
                f"Active: {sum(binary_action)}, Reward: {reward:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in RL decision: {e}")
            # Fallback to simple decision
            self._make_simple_decision()
    
    def _make_simple_decision(self):
        """Make simple rule-based decision (fallback)"""
        try:
            # Simple heuristic: enable all models by default
            # In practice, this could be more sophisticated
            model_status = self.model_manager.get_model_status()
            
            for model_name, status in model_status.items():
                if not status.get('enabled', False):
                    self.model_manager.enable_model(model_name)
                    self.simple_policy[model_name] = True
            
            decision_log = {
                'timestamp': datetime.now(),
                'decision_type': 'simple_rule',
                'policy': self.simple_policy.copy(),
                'active_models': sum(self.simple_policy.values())
            }
            
            self.decision_history.append(decision_log)
            
            active_count = sum(self.simple_policy.values())
            self.logger.info(f"Simple Decision: {active_count}/{len(self.simple_policy)} models active")
            
        except Exception as e:
            self.logger.error(f"Error in simple decision: {e}")
    
    def _update_learning(self):
        """Update RL agent with recent experiences"""
        if not self.rl_enabled:
            return
            
        try:
            # Train agent with recent experiences
            if len(self.decision_history) > 10:
                self.agent.learn(total_timesteps=1)
        except Exception as e:
            self.logger.error(f"Error in learning update: {e}")
    
    def update_system_metrics(self, metrics: Dict[str, float]):
        """Update system metrics for reward calculation"""
        if self.rl_enabled and self.env:
            self.env.update_system_metrics(metrics)
    
    def add_performance_feedback(self, performance: Dict[str, Any]):
        """Add performance feedback for learning"""
        if self.rl_enabled and self.env:
            self.env.add_performance_data(performance)
    
    def add_alert_feedback(self, alert: Dict[str, Any]):
        """Add alert feedback for learning"""
        if self.rl_enabled and self.env:
            self.env.add_alert_data(alert)
    
    def get_current_policy(self) -> Dict[str, bool]:
        """Get current model configuration policy"""
        try:
            if self.rl_enabled and self.env:
                # RL-based policy
                current_state = self.env._get_current_state()
                action, _ = self.agent.predict(current_state, deterministic=True)
                
                # Handle numpy scalar properly
                if hasattr(action, 'item'):
                    action = action.item()
                else:
                    action = int(action)
                
                # Convert discrete action to binary policy
                binary_action = self.env._discrete_to_binary(action)
                
                # Create policy dictionary
                policy = {}
                for i, model_name in enumerate(self.env.model_names):
                    if i < len(binary_action):
                        policy[model_name] = bool(binary_action[i])
                    else:
                        policy[model_name] = False
                
                return policy
            else:
                # Simple rule-based policy
                return self.simple_policy.copy()
                
        except Exception as e:
            self.logger.error(f"Error getting current policy: {e}")
            # Return safe default policy
            default_policy = {}
            for model_name in self.model_manager.models.keys():
                default_policy[model_name] = True
            return default_policy
    
    def force_model_configuration(self, config: Dict[str, bool]):
        """Force specific model configuration"""
        for model_name, enabled in config.items():
            if enabled:
                self.model_manager.enable_model(model_name)
            else:
                self.model_manager.disable_model(model_name)
        
        # Update simple policy
        self.simple_policy.update(config)
        
        self.logger.info(f"Forced model configuration: {config}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.decision_history:
            return {
                'orchestrator_type': 'RL' if self.rl_enabled else 'Rule-based',
                'total_decisions': 0,
                'recent_avg_reward': 0.0,
                'exploration_rate': self.exploration_rate,
                'learning_enabled': self.learning_enabled
            }
        
        recent_decisions = list(self.decision_history)[-50:]
        
        summary = {
            'orchestrator_type': 'RL' if self.rl_enabled else 'Rule-based',
            'total_decisions': len(self.decision_history),
            'exploration_rate': self.exploration_rate,
            'learning_enabled': self.learning_enabled and self.rl_enabled
        }
        
        if self.rl_enabled:
            # RL-specific metrics
            rl_decisions = [d for d in recent_decisions if 'reward' in d]
            if rl_decisions:
                summary.update({
                    'recent_avg_reward': np.mean([d['reward'] for d in rl_decisions]),
                    'avg_active_models': np.mean([sum(d.get('binary_action', [])) for d in rl_decisions])
                })
        else:
            # Simple orchestrator metrics
            summary.update({
                'recent_avg_reward': 0.0,
                'avg_active_models': sum(self.simple_policy.values())
            })
        
        return summary