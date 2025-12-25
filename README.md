# Millennium Quantitative Research Playground

A backtesting framework for quantitative trading strategies with support for equity portfolios, performance metrics, and strategy evaluation.

📖 **For detailed documentation on the framework architecture, file structure, and creating custom strategies, see [GUIDE.md](GUIDE.md).**

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/lucas-309/millennium-data-quality-25-26
    cd millennium-data-quality
```

2. **Create a conda environment:**
    If you do not have conda installed, install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Make sure you cd into the root of this repository before running this command:
```sh
    conda env create -f environment.yml
```

3. **Activate the conda virtual environment:**

    Using conda (make sure to activate every time you create a new shell instance):
```sh
    conda activate data-quality
```

## Quick Start

### 1. Cache Historical Data (Required First Step)

Before running any backtests, you need to download and cache historical market data:
```sh
python backtester/cache_sp500_data.py
```

**Note:** A few tickers may fail to download (delisted companies) - this is expected and harmless.

### 2. Run the Main Script

Once data is cached, run the sample mean reversion strategy:
```sh
python backtester/main.py
```

This will backtest a simple mean reversion strategy on AAPL, MSFT, and GOOGL (2011-2024) and display performance metrics.

## Running Sample Research Notebooks

To run a `.ipynb` notebook (like `bab.ipynb`), simply select the `data-quality` kernel and run the cells. If the kernel is not visible, go to **Select Another Kernel > Python Environments**.

**[IMPORTANT]** When running on cached data, make sure date ranges in your notebook align with your order generator's expected range.

## Running Tests

To run the unit tests, use the following command. 

**Note:** Tests are not freshly maintained at the moment.
```sh
python -m unittest discover -s unit_tests
```