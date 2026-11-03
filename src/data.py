
import yfinance as yf
from pathlib import Path


def download_spy():
    """Download daily SPY data and save to CSV."""

    # Download historical prices
    spy = yf.download(
        "SPY",
        start="2010-01-01",
        end="2026-10-01",
        auto_adjust=True,
        progress=False,
        multi_level_index=False
    )

    # Create the data directory if necessary
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Save to CSV
    file_path = data_dir / "spy.csv"
    spy.to_csv(file_path)

    print(f"Saved {len(spy)} observations to {file_path}")
    print(spy.head())
    print(spy.shape)
    print(spy.isna().sum())
    return spy


if __name__ == "__main__":
    download_spy()
