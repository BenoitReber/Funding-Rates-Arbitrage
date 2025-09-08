
import numpy as np
import time
from core.data.models.time_series import ThirdPartySerie

class OHLCVSimulator:
    """    
    Simulator for generating OHLCV (Open, High, Low, Close, Volume) price data
    using a Geometric Brownian Motion (GBM) model with intra-period steps.
    
    This simulator generates realistic price movements by simulating multiple
    price points within each candle period to determine high and low values,
    similar to how Ogata's thinning algorithm generates point processes.
    """
    def __init__(self, ts: ThirdPartySerie,
                 initial_price: float = 100.0,
                 base_volume: float = 1000.0,
                 drift: float = 0.0001,      # Per-period drift (0.01%)
                 volatility: float = 0.015,   # Per-period volatility (1.5%)
                 intra_steps: int = 5,        # Steps per candle for H/L calculation
                 initial_timestamp: int = None):
        """
        Initialize the OHLCV simulator with specified parameters.
        
        Parameters
        ----------
        ts : ThirdPartySerie
            Time series object to store the generated OHLCV data.
        initial_price : float, default=100.0
            Starting price for the simulation.
        base_volume : float, default=1000.0
            Base trading volume, which will be modulated by price movements.
        drift : float, default=0.0001
            Per-period price drift (0.01%), similar to baseline intensity in Hawkes.
        volatility : float, default=0.015
            Per-period price volatility (1.5%), controls the magnitude of random shocks.
        intra_steps : int, default=5
            Number of price steps to simulate within each candle period.
        initial_timestamp : int, optional
            Starting timestamp in milliseconds. If None, uses current time.
        """
        self.ts = ts
        self.current_close = initial_price
        self.base_volume = base_volume
        self.drift = drift
        self.volatility = volatility
        self.intra_steps = intra_steps

        # Initialize time
        self.current_time = initial_timestamp if initial_timestamp else int(time.time() * 1000)
        
        # Initialize first candle if empty
        if np.isnan(self.ts.timestamps).all():
            self._create_first_candle(initial_price)

    def _create_first_candle(self, price: float):
        """Initialize the first candle with flat values
        
        Parameters
        ----------
        price : float
            The price to use for all OHLC values in the first candle.
        """
        self.ts.timestamps[:] = self.current_time
        self.ts.data[:] = [price, price, price, price, self.base_volume]

    def next(self):
        """Generate next OHLCV candle and update the TimeSerie
        
        This method simulates the next time step in the price process, generating
        a new OHLCV candle by:
        1. Simulating intra-period price movements
        2. Calculating open, high, low, close values from the simulated path
        3. Generating realistic volume based on price movement
        4. Updating the time series with the new candle
        
        Returns
        -------
        list
            The newly generated OHLCV values [open, high, low, close, volume]
        """
        # Generate intra-period price path
        intra_prices = self._generate_intra_prices()
        
        # Calculate OHLCV values
        open_price = self.current_close
        close_price = intra_prices[-1]
        high = np.max(intra_prices)
        low = np.min(intra_prices)
        
        # Calculate volume based on price movement and randomness
        price_change = abs(close_price - open_price) / open_price
        volume = self.base_volume * (1 + price_change * 10) * np.random.lognormal(0, 0.2)
        
        # Update tracking
        self.current_close = close_price
        self.current_time += self.ts.duration
        
        # Update the series buffer
        self._update_series(
            timestamp=self.current_time,
            values=[open_price, high, low, close_price, volume]
        )
        
        return self.ts.data[-1]

    def _generate_intra_prices(self):
        """Generate intra-period prices using Geometric Brownian Motion (GBM)
        
        This method simulates multiple price points within a single candle period
        using a discretized GBM model. Similar to how Hawkes process simulation
        generates events based on intensity, this generates price movements based
        on drift and volatility parameters.
        
        Returns
        -------
        numpy.ndarray
            Array of simulated prices within the period, starting with current_close
        """
        prices = [self.current_close]
        dt = 1 / self.intra_steps  # Relative to candle period
        
        for _ in range(self.intra_steps - 1):
            drift = self.drift * dt
            shock = self.volatility * np.sqrt(dt) * np.random.normal()
            prices.append(prices[-1] * np.exp(drift + shock))
        
        return np.array(prices)

    def _update_series(self, timestamp: int, values: list):
        """Update the circular buffer with new OHLCV values
        
        This method manages the time series data structure, implementing a circular
        buffer pattern by rolling the arrays and updating the last position with new values.
        
        Parameters
        ----------
        timestamp : int
            The timestamp for the new candle in milliseconds
        values : list
            The OHLCV values [open, high, low, close, volume] to add
        """
        self.ts.data = np.roll(self.ts.data, -1, axis=0)
        self.ts.data[-1] = values
        
        self.ts.timestamps = np.roll(self.ts.timestamps, -1)
        self.ts.timestamps[-1] = timestamp
