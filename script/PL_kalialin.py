"""
Powerlink

Author: MK
"""

# =============================================================================
# Preparatory Code
# =============================================================================

# Imports ---------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates

# =============================================================================
# VARIABLES
# =============================================================================
sel_region="VIC1"
sel_interval="20:30"

# =============================================================================
# Data load
# =============================================================================

repo_root = os.path.dirname(os.path.dirname(__file__)) 

bid_path = os.path.join(repo_root, 'inputs', 'raw_bid_data_2025-06-26.csv')
bid_price = pd.read_csv(bid_path) 
bid_price = bid_price.rename(columns={'duid': 'DUID'})
bid_price['interval_datetime'] = pd.to_datetime(bid_price['interval_datetime'])
bid_price['time'] = bid_price['interval_datetime'].dt.strftime('%H:%M')

duid_path_SL = os.path.join(repo_root, 'inputs', 'NEM Registration and Exemption List.xlsx')
duid_SL = pd.read_excel(
    duid_path_SL,
    sheet_name='PU and Scheduled Loads',
    usecols=['DUID', 'Region', 'Fuel Source - Primary','Fuel Source - Descriptor'])

duid_path_DR = os.path.join(repo_root, 'inputs', 'NEM Registration and Exemption List.xlsx')
duid_DR = pd.read_excel(
    duid_path_DR,
    sheet_name='Wholesale Demand Response Units',
    usecols=['WDRU DUID', 'Region'])
duid_DR = duid_DR.rename(columns={'WDRU DUID': 'DUID'})
duid_DR['Fuel Source - Descriptor'] = 'DR'
duid_DR['Fuel Source - Primary'] = 'DR'


duid = pd.concat([duid_SL, duid_DR], ignore_index=True).drop_duplicates()

bid_price = pd.merge(bid_price, duid, on='DUID', how='left')
bid_price['Fuel'] = bid_price['Fuel Source - Primary'] + ' - ' + bid_price['Fuel Source - Descriptor']

missing_region_count = bid_price['Region'].isna().sum()
print("No of DUID with missing region:", missing_region_count)


# =============================================================================
# Function 
# =============================================================================

PRICE_BANDS = [f"PRICEBAND{i}" for i in range(1, 11)]
AVAIL_BANDS = [f"BANDAVAIL{i}" for i in range(1, 11)]


def unpivot_price_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide PRICEBAND columns into long format with extracted band numbers.
    """
    long_p_df = df.melt(
        id_vars=["DUID", "Fuel", "time", "rrp", 'Region', 'TOTALCLEARED'],
        value_vars=PRICE_BANDS,
        var_name="p_band",
        value_name="Price")

    # Extract band number as integer
    long_p_df["Band"] = long_p_df["p_band"].str.extract(r"(\d+)").astype(int)

    return long_p_df.drop(columns=["p_band"])


def unpivot_volume_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide BANDAVAIL columns into long format for joining with price bands.
    """
    long_q_df = df.melt(
        id_vars=["DUID", "time"],
        value_vars=AVAIL_BANDS,
        var_name="q_band",
        value_name="Volume")

    long_q_df["Band"] = long_q_df["q_band"].str.extract(r"(\d+)").astype(int)
    return long_q_df.drop(columns=["q_band"])


def merge_price_and_volume(price_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join price and volume by DUID, time and band index.
    """
    return price_df.merge(
        vol_df,
        on=["DUID", "time", "Band"],
        how="left")


def add_cumulative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative volume within each trading interval.
    """
    df = df.sort_values(["time", "Price"])
    df["CumulativeVolume"] = df.groupby("time")["Volume"].cumsum()
    return df


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_bid_data(df: pd.DataFrame) -> pd.DataFrame:
    # FILTER invalid cleared intervals
    df = df[(df["TOTALCLEARED"].notna()) & (df["TOTALCLEARED"] != 0)]

    # SELECT required columns
    df_subset = df[["DUID", "Fuel", "time", "rrp", "Region", "TOTALCLEARED"] + PRICE_BANDS + AVAIL_BANDS].copy()

    # UNPIVOT price and volume bands
    price = unpivot_price_bands(df_subset)
    quantity = unpivot_volume_bands(df_subset)

    # MERGE price with volume
    merged = merge_price_and_volume(price, quantity)
    summary = price.merge(quantity, on=["DUID", "time", "Band"], how="left")


    # FILTER out zero volumes
    merged = merged[merged["Volume"] != 0]

    # ADD cumulative volume
    merged = add_cumulative_volume(merged)

    # CLEAN sorting and indexing
    merged = merged.sort_values(["time", "Price"]).reset_index(drop=True)

    return merged


# =============================================================================
# SUPPLY CURVE PLOT FUNCTION
# =============================================================================

def plot_supply_curve(df: pd.DataFrame, region: str, interval: str):
    # --- Filter to requested region+interval ---
    df_sel = df[(df["Region"] == region) & (df["time"] == interval)].copy()

    if df_sel.empty:
        raise ValueError(
            f"No records found for Region={region} and time={interval}.\n"
            f"Available times include: {sorted(df['time'].unique())[:10]}...")

    rrp_value = df_sel["rrp"].iloc[0]

    total_cleared = df_sel.drop_duplicates(subset="DUID")["TOTALCLEARED"].sum()

    df_sel = df_sel.sort_values("Price")

    plt.figure(figsize=(10, 6))
    plt.step(
        df_sel["CumulativeVolume"],
        df_sel["Price"],
        where="post",
        linewidth=2,
        label="Supply Curve")  # <-- add label here

    # RRP horizontal line
    plt.axhline(
        y=rrp_value,
        linestyle="--",
        linewidth=2,
        color="red",
        label=f"RRP = ${rrp_value:,.2f}/MWh")

    # TOTALCLEARED vertical line
    plt.axvline(
        x=total_cleared,
        linestyle="--",
        linewidth=2,
        color="blue",
        label=f"Total cleard capacity = {total_cleared:.0f} MW")

    plt.title(f"Supply Curve — {region}, {interval}", fontsize=14)
    plt.xlabel("Cumulative Capacity (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.grid(True, alpha=0.3)
    plt.legend()  # <-- add legend
    plt.tight_layout()
    plt.show()

    return df_sel


# =============================================================================
# PRICE SETTER
# =============================================================================

def identify_price_setters(df: pd.DataFrame, region: str):
    df_region = df[df["Region"] == region].copy()
    
    # Compute distance from RRP
    df_region["PriceDiff"] = abs(df_region["Price"] - df_region["rrp"])

    # Find the bid(s) with minimal difference per interval
    df_price_setters = df_region.loc[df_region.groupby("time")["PriceDiff"].idxmin()].copy()

    # Optional: keep only relevant columns
    df_price_setters = df_price_setters[[
        "time", "DUID", "Fuel", "Price", "rrp", "Volume"
    ]].rename(columns={"Price":"BidPrice", "rrp":"ClearingPrice"})

    return df_price_setters


# =============================================================================
# REVENUE vs PRICE SETTING
# =============================================================================

def plot_revenue_vs_price_setting(df_supply_curve, df_price_setters, region="VIC1", interval_mins=5):
    """
    Scatter plot: Total Revenue vs # of Intervals as Price Setter, colored by Fuel type.
    """
    # --- Calculate Revenue ---
    df_rev = df_supply_curve[df_supply_curve["Region"]==region].copy()
    df_rev["IntervalHours"] = interval_mins / 60
    df_rev = df_rev.groupby(["DUID", "Fuel", "time"], as_index=False)[["TOTALCLEARED", "IntervalHours", "rrp"]].mean()

    df_rev["Revenue"] = df_rev["rrp"] * df_rev["TOTALCLEARED"] * df_rev["IntervalHours"]

    # Total revenue per DUID
    revenue_total = df_rev.groupby(["DUID", "Fuel"], as_index=False)["Revenue"].sum()

    # Count # intervals as price-setter
    ps_count = df_price_setters.groupby("DUID").size().rename("PriceSetterCount").reset_index()

    # Merge revenue and price-setting count
    summary = revenue_total.merge(ps_count, on="DUID", how="left")
    summary["PriceSetterCount"] = summary["PriceSetterCount"].fillna(0)

    # --- Scatter Plot ---
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=summary,
        x="PriceSetterCount",
        y="Revenue",
        hue="Fuel",
        s=100, alpha=0.8)
    
    plt.xlabel("Number of Intervals as Price Setter")
    plt.ylabel("Total Revenue ($)")
    plt.title(f"Revenue vs Price-Setting Frequency — {region}")

    # Format y-axis as $
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return summary


# =============================================================================
# EXECUTION
# =============================================================================

# Processed output
df_supply_curve = process_bid_data(bid_price)

# Supply cerve per interval - plot
df_plot = plot_supply_curve(df_supply_curve, region=sel_region, interval=sel_interval)

price_setters = identify_price_setters(df_supply_curve, region=sel_region)

# Dominant generator 
dominant_generators = price_setters.groupby("DUID").size().sort_values(ascending=False)
print(dominant_generators.head(10))

# Dominant generator type by fueal
dominant_by_fuel = price_setters.groupby("Fuel").size().sort_values(ascending=False)
print(dominant_by_fuel)

#Dominant fuel - plot
dominant_by_fuel.plot(kind="barh", figsize=(8,5), title="Price-setting frequency by Fuel Type")
plt.xlabel("Number of intervals")
plt.show()


# Revenue vs price setting
summary_df = plot_revenue_vs_price_setting(df_supply_curve, price_setters, region=sel_region)




# =============================================================================
# Wind SCENARIO
# =============================================================================
def counterfactual_supply_curve(df_supply_curve, region="VIC1", interval="20:30", extra_capacity_mw=100):
    """
    Counterfactual: Add hypothetical 100MW wind bid at $0.
    """

    # -----------------------------
    # Filter interval & region
    # -----------------------------
    df_sel = df_supply_curve[
        (df_supply_curve["Region"] == region) &
        (df_supply_curve["time"] == interval)
    ].copy()

    # -----------------------------
    # Extract actual values
    # -----------------------------
    rrp_value = df_sel["rrp"].iloc[0]

    # Energy-only cleared volume
    total_cleared_energy = df_sel.drop_duplicates("DUID")["TOTALCLEARED"].sum()

    # -----------------------------
    # Determine cumulative volume at actual RRP
    # -----------------------------
    df_sorted = df_sel.sort_values("Price").reset_index(drop=True)
    df_sorted["CumulativeVolume"] = df_sorted["Volume"].cumsum()

    # find CV where price crosses the actual RRP
    # (closest bid price to RRP)
    idx_rrp = (df_sorted["Price"] - rrp_value).abs().idxmin()
    CV_at_rrp = df_sorted.loc[idx_rrp, "CumulativeVolume"]

    # -----------------------------
    # FCAS SHADOW RESERVE (MW)
    # -----------------------------
    fcas_shadow = max(0, CV_at_rrp - total_cleared_energy)

    # -----------------------------
    # Create NEW supply curve with extra wind
    # -----------------------------
    new_row = pd.DataFrame({
        "DUID": ["WindHypothetical"],
        "Fuel": ["Wind - Hypothetical"],
        "Price": [0],
        "Volume": [extra_capacity_mw],
        "time": [interval],
        "Region": [region]})

    df_new = pd.concat([df_sel, new_row], ignore_index=True)
    df_new = df_new.sort_values("Price").reset_index(drop=True)
    df_new["CumulativeVolume"] = df_new["Volume"].cumsum()

    # -----------------------------
    # New clearing volume requirement
    # -----------------------------
    new_required_volume = total_cleared_energy + fcas_shadow

    # New marginal bid
    marginal_idx = df_new[df_new["CumulativeVolume"] >= new_required_volume].index[0]
    new_rrp = df_new.loc[marginal_idx, "Price"]

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10,6))

    # Original curve
    plt.step(df_sorted["CumulativeVolume"], df_sorted["Price"],
             where="post", linewidth=2, label="Original Supply Curve")

    # New curve
    plt.step(df_new["CumulativeVolume"], df_new["Price"],
             where="post", linewidth=2, label="With 100MW Wind")

    # Original & new RRPs
    plt.axhline(rrp_value, color='red', linestyle='--',
                label=f"Original RRP = ${rrp_value:.2f}")
    plt.axhline(new_rrp, color='green', linestyle='--',
                label=f"New RRP = ${new_rrp:.2f}")


    plt.xlabel("Cumulative Capacity (MW)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Counterfactual — {region} {interval}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



result = counterfactual_supply_curve(
    df_supply_curve,
    region=sel_region,
    interval=sel_interval,
    extra_capacity_mw=100)




# =============================================================================
# BESS
# =============================================================================
# Filter for VIC1 and 18:00–21:00
df_plot = bid_price[
    (bid_price['Region'] == sel_region) &
    (bid_price['interval_datetime'].dt.time >= pd.to_datetime('18:00').time()) &
    (bid_price['interval_datetime'].dt.time <= pd.to_datetime('21:00').time())
].copy()

df_avg = df_plot.groupby('interval_datetime', as_index=False).agg({
    'rrp': 'mean',
    'forecasted_rrp': 'mean'})

# Plot
plt.figure(figsize=(12,6))
plt.plot(df_avg['interval_datetime'], df_avg['rrp'], label='Actual RRP', linewidth=2)
plt.plot(df_avg['interval_datetime'], df_avg['forecasted_rrp'], label='Average Forecasted RRP', linewidth=2, linestyle='--')

plt.xlabel("Time")
plt.ylabel("Price ($/MWh)")
plt.title("Actual vs Average Forecasted RRP — (18:00–21:00)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.tight_layout()
plt.show()


df_avg['diff'] = df_avg['forecasted_rrp'] - df_avg['rrp']
#max_above = df_avg['diff'].max()    # Forecasted above RRP
max_below = df_avg['diff'].min()    # Forecasted below RRP (negative)

#print(max_above)
print(max_below)