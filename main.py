import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import copy
import matplotlib.pyplot as plt
from typing import Any, Dict, List

def download_stock_data(tickers: list[str], start_date: str) -> pd.DataFrame:
    """
    Download historical stock data for a list of tickers from the specified start date to the current date.

    Parameters:
    - tickers (list[str]): A list of stock ticker symbols.
    - start_date (str): The start date for fetching historical data in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the historical stock data, organized by ticker.
    """
    data = yf.download(tickers, start=start_date, progress=False, group_by="ticker")
    return data


def get_daily_value_of_holdings(historical_data: pd.DataFrame, date: str, holding_stock: dict[str, int], last_known_value: float) -> float:
    """
    Calculate the total value of stock holdings on a specific date using historical data. If data for the specified date
    is not available, return the last known value.

    Parameters:
    - historical_data (pd.DataFrame): A DataFrame containing historical stock data, indexed by date and with columns including 'Close'.
    - date (str): The date for which to calculate the holdings' value, in 'YYYY-MM-DD' format.
    - holding_stock (dict[str, int]): A dictionary mapping stock symbols (str) to the number of shares held (int).
    - last_known_value (float): The fallback value to use if data for the specified date is unavailable.

    Returns:
    - float: The total value of the stock holdings on the specified date or the last known value if unavailable.
    """
    total_value = 0.0
    for symbol, quantity in holding_stock.items():
        try:
            if symbol in historical_data and date in historical_data[symbol].index:
                stock_data = historical_data[symbol].loc[date]
                closing_price = stock_data["Close"]
                total_value += closing_price * quantity
            else:
                total_value = last_known_value
                break  # Use last known value if data for this date is not available
        except Exception as e:
            print(f"Error processing data for {symbol} on {date}: {e}")
            total_value = last_known_value
            break  # Use last known value on error

    return total_value if total_value > 0 else last_known_value


def get_need_stock_lists(transactions: pd.DataFrame) -> list[str]:
    """
    Extract a list of unique stock ticker symbols from transaction records.

    Parameters:
    - transactions (pd.DataFrame): A DataFrame containing transaction records, with a 'Symbol' column for stock symbols.

    Returns:
    - list[str]: A list of unique stock ticker symbols present in the transactions.
    """
    tickers = transactions["Symbol"].unique().tolist()
    # Remove any whitespace and filter out empty strings
    tickers = [symbol.strip() for symbol in tickers if symbol.strip()]
    return tickers


def get_daily_cash_and_stocks(transactions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Calculate daily cash and stock holdings from transaction records.

    Parameters:
    - transactions (pd.DataFrame): DataFrame containing transaction records with columns
      'Symbol', 'Action', 'Amount', 'Quantity', and 'TradeDate'.

    Returns:
    - dict[str, dict[str, Any]]: A dictionary with dates as keys and dictionaries as values,
      where each nested dictionary includes 'current_value' and 'holding_stock' for the day.
    """
    original_value = 0
    current_value = 0
    holding_stock: dict[str, int] = {}
    daily_values: dict[str, dict[str, Any]] = {}

    for _, row in transactions.iterrows():
        symbol = row.Symbol.strip()
        if symbol:
            if row.Action == "BUY" or row.Action == "SELL":
                current_value += row.Amount
                holding_stock[symbol] = holding_stock.get(symbol, 0) + row.Quantity
            elif row.Action in ["Dividend", "Other"]:
                if row.Action == "Other":
                    holding_stock[symbol] = holding_stock.get(symbol, 0) + row.Quantity
            else:
                print(f"Format error in row: {row}")
                assert False, "Transaction format error"

        else:
            if row.Action != "Interest":
                original_value += row.Amount
            current_value += row.Amount

        daily_holdings = copy.deepcopy(holding_stock)
        daily_values[row.TradeDate] = {
            "current_value": current_value,
            "holding_stock": daily_holdings,
        }

    return daily_values


def get_daily_cash_and_chosen_index(transactions: pd.DataFrame, historical_data: pd.DataFrame, index_ticker: str) -> dict[str, dict[str, Any]]:
    """
    Calculate daily cash values and holdings of a chosen index from transaction records.

    Parameters:
    - transactions (pd.DataFrame): DataFrame of transactions.
    - historical_data (pd.DataFrame): DataFrame with historical data for the index.
    - index_ticker (str): Ticker symbol of the index.

    Returns:
    - dict[str, dict[str, Any]]: Daily cash values and stock holdings for the index.
    """
    current_value = 0
    holding_stock: dict[str, int] = {}
    daily_values: dict[str, dict[str, Any]] = {}

    for _, row in transactions.iterrows():
        if not row.Symbol.strip():
            if row.Action != "Interest" and "Funds Received" in row.Description:
                current_value += row.Amount
                stock_data = historical_data[index_ticker].loc[row.TradeDate]
                closing_price = stock_data["Close"]
                buy_amount = int(current_value // closing_price)
                holding_stock[index_ticker] = holding_stock.get(index_ticker, 0) + buy_amount
                current_value -= buy_amount * closing_price

        daily_holdings = copy.deepcopy(holding_stock)
        daily_values[row.TradeDate] = {
            "current_value": current_value,
            "holding_stock": daily_holdings,
        }

    return daily_values


def get_daily_cash_without_investment(transactions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Calculate daily cash values without investing in stocks or indices from transaction records.

    Parameters:
    - transactions (pd.DataFrame): DataFrame of transactions.

    Returns:
    - dict[str, dict[str, Any]]: Daily cash values without stock holdings.
    """
    current_value = 0
    daily_values: dict[str, dict[str, Any]] = {}

    for _, row in transactions.iterrows():
        # Adjust current value based on transaction amount without buying stocks
        if not row.Symbol.strip():
            # Consider all cash movements except 'Interest' as cash adjustments
            if row.Action != "Interest":
                current_value += row.Amount

        # Record daily cash value
        daily_values[row.TradeDate] = {
            "current_value": current_value,
            "holding_stock": {},  # No stocks held, empty dict for consistency
        }

    return daily_values




def get_portfolio_daily_values(daily_values: Dict[str, Dict[str, Dict[str, float]]], start_date: str, end_date: str, historical_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the portfolio's daily values over a specified date range based on transactions and historical stock data.

    Parameters:
    - daily_values (Dict[str, Dict[str, Dict[str, float]]]): A nested dictionary containing each date's holding information.
      Each key is a date string 'YYYY-MM-DD', mapping to a dictionary that contains 'holding_stock', which itself maps stock symbols
      to the quantity held for that day.
    - start_date (str): The start date of the period for calculating daily portfolio values.
    - end_date (str): The end date of the period.
    - historical_data (pd.DataFrame): The DataFrame containing historical stock data, indexed by date.

    Returns:
    - Dict[str, float]: A dictionary mapping each date within the specified range to the total value of the portfolio for that date.
    """
    portfolio_daily_values: Dict[str, float] = {}
    date_range = pd.date_range(start=start_date, end=end_date)
    last_known_value = 0.0
    last_holdings: Dict[str, int] = {}
    last_cash = 0
    for single_date in date_range:
        date_str = single_date.strftime("%Y-%m-%d")
        day_data = daily_values.get(date_str, {})

        current_holdings = day_data.get("holding_stock", last_holdings)

        last_known_value = get_daily_value_of_holdings(
            historical_data, date_str, current_holdings, last_known_value
        )
        portfolio_daily_values[date_str] = last_known_value + day_data.get("current_value", last_cash)

        # Update the last holdings for the next iteration
        last_holdings = current_holdings if current_holdings else last_holdings
        last_cash = day_data.get("current_value", last_cash)
    return portfolio_daily_values


def plot_portfolios(portfolios_daily_values: List[Dict[str, float]], sample_day: int, labels: List[str]) -> None:
    """
    Plot the portfolio values over time for multiple portfolios.

    Parameters:
    - portfolios_daily_values (List[Dict[str, float]]): A list where each item is a dictionary mapping dates (str) to portfolio values (float).
    - sample_day (int): Interval at which to sample the data points for plotting to reduce clutter.
    - labels (List[str]): Labels for each portfolio, to be used in the plot legend.

    Returns:
    - None: This function plots the portfolio values directly and does not return any value.
    """
    plt.figure(figsize=(10, 6))
    colors = ["b", "g", "r", "c", "m", "y", "k"]  # Define more colors if necessary.

    for i, portfolio_values in enumerate(portfolios_daily_values):
        dates = list(portfolio_values.keys())
        values = list(portfolio_values.values())

        # Sampling for less cluttered plots.
        sampled_dates = [date for j, date in enumerate(dates) if j % sample_day == 0]
        sampled_values = [value for j, value in enumerate(values) if j % sample_day == 0]

        # Use label if provided, otherwise default to "Portfolio {i+1}".
        label = labels[i] if i < len(labels) else f"Portfolio {i+1}"
        
        plt.plot(sampled_dates, sampled_values, marker=".", linestyle="-", color=colors[i % len(colors)], label=label)

    plt.title("Portfolio Values Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Compare portfolio performance against chosen indices.')
    parser.add_argument('file_path', type=str, help='CSV file path containing transaction data')
    parser.add_argument('indices', type=str, nargs='+', help='List of indices to compare (e.g., SPY QQQ)')

    # Parse arguments
    args = parser.parse_args()

    file_path = args.file_path
    compare_indices = args.indices

    transactions = pd.read_csv(file_path)
    # Ensure unique tickers and include market indices for comparison
    tickers = list(set(get_need_stock_lists(transactions) + compare_indices))

    # Determine the date range from transaction data
    start_date = transactions["TradeDate"].min()
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Download historical data for all tickers
    historical_data = download_stock_data(tickers, start_date)

    # Calculate daily values for the individual portfolio
    daily_values = get_daily_cash_and_stocks(transactions)
    portfolio_daily_values = get_portfolio_daily_values(daily_values, start_date, end_date, historical_data)

    # Initialize lists for storing portfolio values and labels
    portfolios_daily_values = [portfolio_daily_values]
    labels = ["Kevin's Portfolio"]

    # Calculate and store daily values for each index in compare_indices
    for index in compare_indices:
        daily_values_index = get_daily_cash_and_chosen_index(transactions, historical_data, index)
        portfolio_daily_values_index = get_portfolio_daily_values(daily_values_index, start_date, end_date, historical_data)
        portfolios_daily_values.append(portfolio_daily_values_index)
        labels.append(f"{index} Index")

    # Calculate daily values for the cash-only strategy
    daily_values_cash = get_daily_cash_without_investment(transactions)
    portfolio_daily_values_cash = get_portfolio_daily_values(daily_values_cash, start_date, end_date, historical_data)
    
    # Add cash-only strategy to the lists
    portfolios_daily_values.append(portfolio_daily_values_cash)
    labels.append("Cash Only")

    # Plot portfolio values over time
    plot_portfolios(
        portfolios_daily_values,
        sample_day=15,
        labels=labels
    )

    cash_only_final_value = list(portfolio_daily_values_cash.values())[-1]

    print("Final Value and Percentage Earnings Compared to Cash Only:")
    for label, portfolio_daily_values in zip(labels, portfolios_daily_values):
        # Extract the last value of the current portfolio
        portfolio_final_value = list(portfolio_daily_values.values())[-1]
        
        # Calculate percentage earning compared to cash only
        percentage_earning = ((portfolio_final_value - cash_only_final_value) / cash_only_final_value) * 100
        
        # Print both the final value of the portfolio and its percentage earning compared to cash only
        print(f"{label}: Final Value = {portfolio_final_value:.2f}, Percentage Earning = {percentage_earning:.2f}%")
