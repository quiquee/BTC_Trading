# 🤖 Reinforcement Learning BTC Trading Agent

## Overview

This project implements an autonomous cryptocurrency trading system using deep reinforcement learning to develop intelligent trading strategies for the BTC/EUR market. The system employs Proximal Policy Optimization (PPO) to train an agent that learns optimal trading decisions based on technical indicators and market conditions.

> **Research Note**: This project represents ongoing research as part of my long-term exploration of machine learning applications in quantitative finance. It combines my interests in artificial intelligence, financial markets, and algorithmic trading.

## 🎯 Key Features

- **Reinforcement Learning Core**: PPO-based trading agent using Stable-Baselines3
- **Advanced Trading Environment**: Custom Gymnasium-compatible environment with realistic market friction
- **Technical Analysis Integration**: Comprehensive technical indicators using pandas-ta
- **Risk Management**: Configurable stop-loss and take-profit mechanisms
- **Performance Analytics**: Equity curve analysis and trade performance metrics
- **GPU Acceleration**: CUDA-optimized training for faster convergence
- **Model Persistence**: Checkpoint system for model versioning and recovery

## 🏗️ Architecture

### Core Components

```
├── train_agent.py       # Main training pipeline and model evaluation
├── test_agent.py        # Testing and backtesting utilities
├── trading_env.py       # Custom RL environment (ForexTradingEnv)
├── indicators.py        # Technical analysis and data preprocessing
├── data/               # Market data and datasets
│   ├── BTC_EUR_latest.csv
│   └── indicators-*.csv
├── checkpoints/        # Model checkpoints during training
└── tensorboard_log/    # Training metrics and monitoring
```

### Technical Stack

- **Reinforcement Learning**: Stable-Baselines3 (PPO algorithm)
- **Environment**: Custom Gymnasium environment with realistic trading mechanics
- **Technical Analysis**: pandas-ta for feature engineering
- **Neural Networks**: PyTorch-based MLP policy networks
- **Monitoring**: TensorBoard for training visualization
- **Data Processing**: pandas, numpy for efficient data handling

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r Requirements.txt
```

**Key Dependencies:**
- `stable-baselines3==2.5.0` - Reinforcement learning algorithms
- `gymnasium` / `gym` - RL environment framework
- `pandas-ta` - Technical analysis indicators
- `torch` - Deep learning backend
- `matplotlib` - Visualization
- `numpy`, `pandas` - Data processing

### Training a New Agent

```bash
python train_agent.py
```

This will:
1. Load and preprocess BTC/EUR market data
2. Split data into training (80%) and testing (20%) sets
3. Initialize PPO agent with MLP policy
4. Train for 1,000,000 timesteps with checkpointing
5. Evaluate multiple model checkpoints on out-of-sample data
6. Save the best-performing model

### Testing and Backtesting

```bash
python test_agent.py
```

Evaluates a trained model's performance with detailed trade analysis and visualization.

## 📊 Environment Design

### Observation Space
- **Technical Indicators**: Rolling window of market features (RSI, MACD, Bollinger Bands, etc.)
- **Position State**: Current position status, time in trade, unrealized P&L
- **Market Data**: OHLCV data with engineered features

### Action Space
- **0**: HOLD (no action)
- **1**: CLOSE (exit current position)
- **2+**: OPEN positions with various stop-loss/take-profit combinations

### Reward Function
- **Realized P&L**: Profit/loss from closed trades (primary signal)
- **Transaction Costs**: Spread, commission, slippage penalties
- **Risk-Adjusted**: Optional unrealized P&L shaping for position management

## ⚙️ Configuration

### Environment Parameters
```python
# Risk management
SL_OPTS = [500, 1000, 1500, 2500, 3000, 6000, 9000, 12000]  # Stop-loss options (pips)
TP_OPTS = [500, 1000, 1500, 2500, 3000, 6000, 9000, 12000]  # Take-profit options (pips)

# Market friction
spread_pips = 1.0           # Bid-ask spread
commission_pips = 0.0       # Commission per trade
max_slippage_pips = 0.2     # Maximum slippage
```

### Training Parameters
```python
total_timesteps = 1000000   # Training duration
checkpoint_freq = 50000     # Model saving frequency
window_size = 30           # Observation window length
```

## 📈 Performance Monitoring

### TensorBoard Integration
```bash
tensorboard --logdir ./tensorboard_log/
```

Monitor training progress with:
- Episode rewards and equity curves
- Loss functions and policy gradients
- Action distribution analysis
- Learning rate schedules

### Model Selection
The system automatically evaluates all checkpoints on out-of-sample data and selects the model with the highest test equity, preventing overfitting to training data.

## 🔬 Research Features

### Data Integrity
- **Time-based splits**: Prevents look-ahead bias in backtesting
- **Random episode starts**: Reduces memorization during training
- **Out-of-sample validation**: Rigorous model selection process

### Advanced RL Techniques
- **Experience replay**: Efficient learning from historical data
- **Policy regularization**: PPO's clipped objective prevents destructive updates
- **Advantage estimation**: GAE for reduced variance in policy gradients

### Market Realism
- **Transaction costs**: Realistic spread and commission modeling
- **Slippage simulation**: Market impact considerations
- **Position persistence**: Realistic position management mechanics

## 📁 Data Structure

The system expects OHLCV data in CSV format:
```csv
Gmt time,Open,High,Low,Close,Volume
2024-01-01 00:00:00,42500.0,42650.0,42400.0,42580.0,125.5
```

Technical indicators are automatically computed and cached for efficient reuse.

## 🎓 Research Context

This project serves as a practical exploration of several cutting-edge machine learning concepts:

- **Deep Reinforcement Learning**: Application of RL to sequential decision-making problems
- **Financial Time Series**: Handling non-stationary, noisy financial data
- **Feature Engineering**: Technical analysis as input representation for neural networks
- **Risk Management**: Integration of financial risk concepts into reward design
- **Model Validation**: Rigorous backtesting methodology for financial applications

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended for live trading or financial advice. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors.

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)

---

**Author**: Research project in machine learning applications for quantitative finance  
**Status**: Active development and experimentation  
**Last Updated**: December 2025
## Overview

This repository contains a reinforcement learning-based automated trading system for the BTC/EUR cryptocurrency pair. The project implements a sophisticated trading environment using **Proximal Policy Optimization (PPO)** from Stable-Baselines3 to develop intelligent trading strategies through deep reinforcement learning.

> **Note**: This project is part of ongoing research into machine learning applications in quantitative finance, reflecting a long-term interest in the intersection of artificial intelligence and financial markets.

## 🎯 Project Objectives

- Develop an autonomous trading agent capable of making profitable BTC/EUR trading decisions
- Research the effectiveness of reinforcement learning in cryptocurrency market prediction
- Implement a comprehensive trading environment with realistic market conditions
- Explore various technical indicators and their impact on trading performance
- Investigate the balance between exploration and exploitation in financial decision-making

## 🏗️ Architecture

### Core Components

1. **Trading Environment** (`trading_env.py`)
   - Custom Gymnasium-compatible environment for forex/crypto trading
   - Position-persistent trading with stop-loss and take-profit mechanisms
   - Realistic market friction modeling (spread, commission, slippage)
   - Dynamic reward shaping based on realized and unrealized P&L

2. **Technical Indicators** (`indicators.py`)
   - Comprehensive technical analysis using pandas-ta
   - Relative feature engineering for improved model generalization
   - Multiple timeframe indicator computation
   - News sentiment integration capabilities

3. **PPO Training Agent** (`train_agent.py`)
   - Stable-Baselines3 PPO implementation with GPU acceleration
   - Automated model checkpointing and evaluation
   - In-sample vs out-of-sample performance tracking
   - Advanced hyperparameter configuration

4. **Model Testing & Evaluation** (`test_agent.py`)
   - Comprehensive backtesting framework
   - Trade-by-trade analysis and reporting
   - Equity curve visualization and performance metrics
   - Statistical significance testing

## 📊 Features

### Trading Environment
- **Multi-action space**: Hold, Close, and various Open positions with configurable SL/TP levels
- **Realistic market simulation**: Incorporates spreads, commissions, and slippage
- **Position management**: Persistent positions until agent closes or SL/TP triggers
- **Risk management**: Built-in stop-loss and take-profit mechanisms
- **Random episode starts**: Reduces overfitting and improves generalization

### Technical Analysis
- **Momentum indicators**: RSI, MACD, Stochastic Oscillator
- **Trend indicators**: Moving averages, Bollinger Bands, ADX
- **Volatility measures**: ATR, Bollinger Band width
- **Volume analysis**: Volume-weighted indicators
- **Price action features**: Rate of change, relative strength

### Machine Learning Pipeline
- **PPO Algorithm**: State-of-the-art policy gradient method
- **Neural network policy**: Multi-layer perceptron for decision making
- **Experience replay**: Efficient learning from historical data
- **TensorBoard integration**: Real-time training monitoring
- **Model persistence**: Automated saving and loading of best performers

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r Requirements.txt
```

Key dependencies:
- `stable_baselines3==2.5.0` - Reinforcement learning algorithms
- `pandas_ta` - Technical analysis indicators
- `gymnasium` - RL environment framework
- `matplotlib` - Visualization
- `numpy`, `pandas` - Data manipulation

### Data Preparation

1. Place your BTC/EUR historical data in the `data/` directory
2. Ensure the CSV format includes: `Gmt time`, `Open`, `High`, `Low`, `Close`, `Volume`
3. The system automatically processes and creates technical indicators

### Training the Agent

```bash
python train_agent.py
```

This will:
- Load and preprocess market data
- Split data into training/testing sets (80/20)
- Train a PPO agent for 1,000,000 timesteps
- Save periodic checkpoints
- Evaluate all models and select the best performer
- Generate equity curve visualizations

### Testing and Evaluation

```bash
python test_agent.py
```

Provides comprehensive analysis including:
- Trade-by-trade breakdown
- Performance metrics and statistics
- Equity curve visualization
- Risk-adjusted returns analysis

## 📈 Performance Metrics

The system tracks multiple performance indicators:

- **Total Return**: Final portfolio value vs initial investment
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Typical holding periods
- **Profit Factor**: Ratio of gross profits to gross losses

## 🔬 Research Focus Areas

### Current Investigations

1. **Reward Function Design**: Optimizing the balance between immediate and delayed rewards
2. **Feature Engineering**: Impact of various technical indicators on model performance
3. **Market Regime Detection**: Adapting strategies to different market conditions
4. **Risk Management**: Dynamic position sizing and stop-loss optimization
5. **Multi-timeframe Analysis**: Incorporating multiple time horizons for decision making

### Future Research Directions

- **Multi-asset Portfolio Management**: Extending to multiple cryptocurrency pairs
- **Transformer-based Architectures**: Exploring attention mechanisms for market analysis
- **Adversarial Training**: Robust model development against market manipulation
- **Real-time Deployment**: Live trading system implementation
- **Market Microstructure**: Incorporating order book dynamics

## 📁 Project Structure

```
BTC_Trading/
├── README.md                    # Project documentation
├── Requirements.txt             # Python dependencies
├── train_agent.py              # PPO training pipeline
├── test_agent.py               # Model evaluation and testing
├── trading_env.py              # Custom trading environment
├── indicators.py               # Technical analysis features
├── data/                       # Market data and datasets
│   ├── BTC_EUR_latest.csv     # Main trading data
│   └── indicators-*.csv       # Preprocessed feature sets
├── checkpoints/                # Saved model checkpoints
├── tensorboard_log/           # Training metrics and logs
└── trade_history_output.csv   # Detailed trade analysis
```

## 🔧 Configuration

Key hyperparameters can be adjusted in `train_agent.py`:

- **Stop-loss/Take-profit levels**: Risk management parameters
- **Window size**: Historical data lookback period
- **Episode length**: Training episode duration
- **Reward weights**: Balance between different reward components
- **Market friction**: Spread, commission, and slippage modeling

## 📚 Technical Background

This project leverages cutting-edge reinforcement learning techniques applied to quantitative finance:

- **Proximal Policy Optimization**: Stable policy gradient method with clipped objective
- **Actor-Critic Architecture**: Separate networks for policy and value function estimation
- **Experience Replay**: Efficient learning from collected trading experiences
- **Curriculum Learning**: Progressive difficulty increase through random episode starts

## ⚠️ Risk Disclaimer

This project is for research and educational purposes only. Cryptocurrency trading involves substantial risk of loss and may not be suitable for all investors. Past performance does not guarantee future results. Always conduct your own research and consider your risk tolerance before making any trading decisions.

## 🤝 Contributing

This research project welcomes contributions in the following areas:

- Novel feature engineering approaches
- Alternative reward function designs
- Advanced neural network architectures
- Market regime detection algorithms
- Risk management improvements

## 📄 License

This project is available for educational and research purposes. Commercial use requires explicit permission.

---

**Disclaimer**: This is an active research project in machine learning applications for financial markets. The models and strategies contained herein are experimental and should not be used for actual trading without extensive additional testing and risk management.