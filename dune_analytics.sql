-- Dune Analytics Query for HybridOracle Analysis
-- This query extracts and analyzes DataUpdated events from the HybridOracle smart contract

WITH 

-- 1. Raw events extraction and value conversion
raw_events AS (
    SELECT
        block_time,
        block_number,
        tx_hash,
        -- Convert prices by dividing by appropriate scale factors
        (CAST(data->>'ethPrice' AS numeric)) / 100.0 as eth_price,
        (CAST(data->>'sp500Index' AS numeric)) / 100.0 as sp500_index,
        (CAST(data->>'hybridIndex' AS numeric)) / 1000.0 as hybrid_index
    FROM ethereum.events
    WHERE 
        address = '0xYourContractAddress'  -- Replace with actual HybridOracle contract address
        AND topics[0] = '0x' || encode(  -- Hash of DataUpdated event
            keccak256('DataUpdated(uint256,uint256,uint256,uint256)'), 'hex'
        )
),

-- 2. Daily statistics aggregation
daily_stats AS (
    SELECT
        DATE_TRUNC('day', block_time) as day,
        COUNT(*) as updates_count,
        AVG(eth_price) as avg_eth_price,
        AVG(sp500_index) as avg_sp500_index,
        AVG(hybrid_index) as avg_hybrid_index,
        MIN(hybrid_index) as min_hybrid_index,
        MAX(hybrid_index) as max_hybrid_index
    FROM raw_events
    GROUP BY DATE_TRUNC('day', block_time)
),

-- 3. Daily changes calculation
daily_changes AS (
    SELECT
        day,
        avg_hybrid_index,
        updates_count,
        -- Calculate percentage change
        ((avg_hybrid_index - LAG(avg_hybrid_index) OVER (ORDER BY day))
         / LAG(avg_hybrid_index) OVER (ORDER BY day)) * 100 as daily_change_pct
    FROM daily_stats
)

-- 4. Final query combining all metrics
SELECT
    -- Temporal information
    block_time as "Timestamp",
    -- Converted prices and indices
    eth_price as "ETH Price (USD)",
    sp500_index as "S&P 500 Index",
    hybrid_index as "Hybrid Index",
    -- Daily metrics
    updates_count as "Daily Updates",
    daily_change_pct as "Daily Change %",
    -- Daily statistics
    avg_hybrid_index as "Daily Avg Hybrid Index",
    min_hybrid_index as "Daily Min Hybrid Index",
    max_hybrid_index as "Daily Max Hybrid Index"
FROM raw_events
LEFT JOIN daily_stats ON DATE_TRUNC('day', block_time) = daily_stats.day
LEFT JOIN daily_changes ON DATE_TRUNC('day', block_time) = daily_changes.day
ORDER BY block_time DESC;

-- Visualization Instructions:
/*
1. Line Chart: Hybrid Index Evolution
   - X-axis: Timestamp
   - Y-axis: Hybrid Index
   - Add 7-day moving average

2. Bar Chart: Update Frequency
   - X-axis: Day
   - Y-axis: Daily Updates
   - Color by update frequency

3. Combined Chart:
   - Primary Y-axis: Hybrid Index (line)
   - Secondary Y-axis: Daily Updates (bars)
   - X-axis: Timestamp

4. Key Metrics Display:
   - Latest Hybrid Index
   - 24h Change
   - Total Updates
   - 7-day Moving Average
*/ 