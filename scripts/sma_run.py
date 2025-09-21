from src.backtest.engine import Backtester
from src.backtest.benchmark import compare_to_buy_and_hold
from src.lib.data import DataFeed, DataFeedConfig
from src.strategies.sma import SMA 
from src.execution.simbroker import SimBroker, SimBrokerConfig

feed = DataFeed(DataFeedConfig(
    symbols=["SPY"], 
    start="2000-01-01"))

strat = SMA(vol_target_ann=0.20, sma_window=200, vol_window=15, symbol="SPY")
broker = SimBroker(SimBrokerConfig(
    commission_per_order=0.50,  # dollars per order
    slippage_bps=15.0           # one-way bps added to price
))
bt = Backtester(
    feed, 
    strat, 
    broker=broker)

cmp = compare_to_buy_and_hold(
    bt, 
    symbol="SPY", 
    title="SMA vs Buy & Hold", 
    show=True,
    benchmark_commission=0.50,
    benchmark_slippage_bps=15.0,)

print("Strategy metrics:", cmp.strategy_metrics)
print("Benchmark metrics:", cmp.benchmark_metrics)