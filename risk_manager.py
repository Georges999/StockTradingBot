"""
Risk management module for the StockTradingBot.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Class for managing trading risk, including position sizing, 
    stop-loss and take-profit settings.
    """
    def __init__(self, config=None):
        """
        Initialize the RiskManager with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Default risk parameters if config not provided
        if config is None:
            self.max_position_size = 0.05  # 5% of portfolio per position
            self.stop_loss_pct = 0.02      # 2% stop loss
            self.take_profit_pct = 0.04    # 4% take profit
            self.max_open_positions = 5    # Maximum number of open positions
            self.portfolio_stop_loss = 0.15  # Close all positions if portfolio drops by 15%
        else:
            self.max_position_size = config.MAX_POSITION_SIZE
            self.stop_loss_pct = config.STOP_LOSS_PCT
            self.take_profit_pct = config.TAKE_PROFIT_PCT
            self.max_open_positions = config.MAX_OPEN_POSITIONS
            self.portfolio_stop_loss = getattr(config, 'PORTFOLIO_STOP_LOSS', 0.15)
    
    def calculate_position_size(self, capital, current_price, risk_per_trade=None):
        """
        Calculate the number of shares to buy/sell based on risk parameters.
        
        Args:
            capital (float): Available capital
            current_price (float): Current price of the asset
            risk_per_trade (float, optional): Maximum risk per trade as percentage
                                             Defaults to max_position_size
                                             
        Returns:
            int: Number of shares to trade
        """
        if risk_per_trade is None:
            risk_per_trade = self.max_position_size
        
        # Calculate capital at risk
        capital_at_risk = capital * risk_per_trade
        
        # Calculate position size
        if current_price <= 0:
            logger.warning(f"Invalid price: {current_price}. Cannot calculate position size.")
            return 0
        
        position_size = int(capital_at_risk / current_price)
        
        return position_size
    
    def calculate_stop_loss(self, entry_price, position_type, atr=None, atr_multiplier=2.0):
        """
        Calculate stop loss price based on configured percentage or ATR.
        
        Args:
            entry_price (float): Entry price of the position
            position_type (str): 'long' or 'short'
            atr (float, optional): Average True Range value
            atr_multiplier (float): Multiplier for ATR-based stop loss
            
        Returns:
            float: Stop loss price
        """
        # Calculate percentage-based stop loss
        if position_type.lower() == 'long':
            stop_price = entry_price * (1 - self.stop_loss_pct)
        elif position_type.lower() == 'short':
            stop_price = entry_price * (1 + self.stop_loss_pct)
        else:
            logger.error(f"Invalid position type: {position_type}")
            return None
        
        # If ATR provided, use ATR-based stop loss
        if atr is not None and atr > 0:
            if position_type.lower() == 'long':
                atr_stop = entry_price - (atr * atr_multiplier)
                # Use tighter of the two stops
                stop_price = max(stop_price, atr_stop)
            else:  # short position
                atr_stop = entry_price + (atr * atr_multiplier)
                # Use tighter of the two stops
                stop_price = min(stop_price, atr_stop)
        
        return stop_price
    
    def calculate_take_profit(self, entry_price, position_type, risk_reward_ratio=None):
        """
        Calculate take profit price based on configured percentage or risk-reward ratio.
        
        Args:
            entry_price (float): Entry price of the position
            position_type (str): 'long' or 'short'
            risk_reward_ratio (float, optional): Risk-reward ratio for calculating take profit
            
        Returns:
            float: Take profit price
        """
        if risk_reward_ratio is None:
            # Use default take_profit_pct
            if position_type.lower() == 'long':
                take_profit = entry_price * (1 + self.take_profit_pct)
            else:  # short position
                take_profit = entry_price * (1 - self.take_profit_pct)
        else:
            # Calculate based on risk-reward ratio
            stop_loss = self.calculate_stop_loss(entry_price, position_type)
            risk = abs(entry_price - stop_loss)
            reward = risk * risk_reward_ratio
            
            if position_type.lower() == 'long':
                take_profit = entry_price + reward
            else:  # short position
                take_profit = entry_price - reward
        
        return take_profit
    
    def check_portfolio_risk(self, portfolio_value, initial_portfolio_value):
        """
        Check if portfolio risk threshold has been exceeded.
        
        Args:
            portfolio_value (float): Current portfolio value
            initial_portfolio_value (float): Initial portfolio value
            
        Returns:
            bool: True if portfolio risk threshold exceeded, False otherwise
        """
        # Calculate portfolio drawdown
        drawdown = (initial_portfolio_value - portfolio_value) / initial_portfolio_value
        
        # Check if drawdown exceeds portfolio stop loss threshold
        return drawdown >= self.portfolio_stop_loss
    
    def check_max_positions(self, current_positions_count):
        """
        Check if maximum number of open positions has been reached.
        
        Args:
            current_positions_count (int): Current number of open positions
            
        Returns:
            bool: True if can open more positions, False otherwise
        """
        return current_positions_count < self.max_open_positions
    
    def adjust_stop_loss(self, entry_price, current_price, current_stop, position_type, trailing_pct=0.5):
        """
        Adjust stop loss based on trailing stop technique.
        
        Args:
            entry_price (float): Entry price of the position
            current_price (float): Current market price
            current_stop (float): Current stop loss level
            position_type (str): 'long' or 'short'
            trailing_pct (float): Percentage for trailing stop
            
        Returns:
            float: New stop loss price
        """
        if position_type.lower() == 'long':
            # Calculate potential new stop loss
            profit = current_price - entry_price
            if profit > 0:
                potential_stop = current_price - (profit * trailing_pct)
                # Only adjust if new stop is higher than current stop
                if potential_stop > current_stop:
                    return potential_stop
        else:  # short position
            # Calculate potential new stop loss
            profit = entry_price - current_price
            if profit > 0:
                potential_stop = current_price + (profit * trailing_pct)
                # Only adjust if new stop is lower than current stop
                if potential_stop < current_stop:
                    return potential_stop
        
        # If no adjustment needed, return current stop
        return current_stop
    
    def risk_adjusted_returns(self, returns, risk_free_rate=0.0):
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            returns (pd.Series): Daily returns series
            risk_free_rate (float): Risk-free rate (annualized)
            
        Returns:
            dict: Dictionary of risk-adjusted return metrics
        """
        # Ensure returns is a pandas Series
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        # Calculate daily risk-free rate from annualized rate
        daily_rf = ((1 + risk_free_rate) ** (1/252)) - 1
        
        # Calculate excess returns
        excess_returns = returns - daily_rf
        
        # Calculate annualized return
        annualized_return = ((1 + returns.mean()) ** 252) - 1
        
        # Calculate annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_return = cumulative_returns.cummax()
        drawdown = (cumulative_returns - max_return) / max_return
        max_drawdown = abs(drawdown.min())
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
    
    def optimize_portfolio_weights(self, returns_dict, method='equal'):
        """
        Optimize portfolio weights based on historical returns.
        
        Args:
            returns_dict (dict): Dictionary of asset returns (asset_name: returns_series)
            method (str): Optimization method - 'equal', 'min_variance', 'max_sharpe'
            
        Returns:
            dict: Optimal weights for each asset
        """
        assets = list(returns_dict.keys())
        n_assets = len(assets)
        
        # Default to equal weighting
        weights = {asset: 1.0 / n_assets for asset in assets}
        
        if method == 'equal':
            return weights
        
        # Create a DataFrame of returns
        returns_df = pd.DataFrame(returns_dict)
        
        if method == 'min_variance':
            # Calculate covariance matrix
            cov_matrix = returns_df.cov()
            
            # Simple minimum variance optimization
            # This is a simplified approach - a real implementation would use optimization libraries
            # Inverse diagonal of covariance matrix as weights
            inv_var = 1 / np.diag(cov_matrix)
            total = np.sum(inv_var)
            weights = {asset: inv_var[i] / total for i, asset in enumerate(assets)}
            
        elif method == 'max_sharpe':
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Simple approach - weights proportional to Sharpe ratio
            # A real implementation would use optimization libraries
            asset_sharpe = expected_returns / np.sqrt(np.diag(cov_matrix))
            total = np.sum(asset_sharpe)
            weights = {asset: asset_sharpe[i] / total for i, asset in enumerate(assets) if asset_sharpe[i] > 0}
            
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            weights = {asset: weight / total_weight for asset, weight in weights.items()}
        
        return weights 