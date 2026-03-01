import pandas as pd
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.models.stock_model import StockModel

def generate_signal(p_bullish, p_neutral, p_bearish, min_gap=0.10, min_prob=0.40):
    """
    User-proposed signal logic based on the 'Gap' between the top probability and the second.
    """
    probs = {
        'BULLISH': p_bullish / 100.0,
        'NEUTRAL': p_neutral / 100.0, 
        'BEARISH': p_bearish / 100.0
    }
    
    # Sort by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    top_class, top_prob = sorted_probs[0]
    second_class, second_prob = sorted_probs[1]
    
    gap = top_prob - second_prob
    gap_points = round(gap * 100, 1)
    
    # Signal only fires if top class is strong enough AND leads clearly
    if top_prob >= min_prob and gap >= min_gap:
        if top_class == 'BULLISH':
            return '🟢 BUY', gap_points
        elif top_class == 'BEARISH':
            return '🔴 SELL', gap_points
        else:
            return '🟡 HOLD', gap_points
    else:
        return '⚪ NO SIGNAL', gap_points

def get_confidence_bar(gap_points):
    """Generates a visual bar representing confidence (gap). Max 30 points for full bar."""
    filled = min(int(gap_points / 3), 10)
    return "█" * filled + "░" * (10 - filled)

def predict_latest(symbol, model):
    """
    Generate a directional prediction using the Gap-based logic and visual formatting.
    """
    report = []
    
    feature_file = f"data/features/{symbol}.parquet"
    if not os.path.exists(feature_file):
        msg = f"Error: No feature data found for {symbol}"
        print(msg)
        report.append(msg)
        return report
        
    df = pd.read_parquet(feature_file)
    latest_data = df.tail(1)
    latest_date = latest_data.index[0]
    
    # Get directional probabilities (they come back as 0-100 floats)
    result = model.predict_directional(latest_data)
    p_bull = result['p_bullish']
    p_bear = result['p_bearish']
    p_neutral = result['p_neutral']
    
    # NEW SIGNAL LOGIC
    signal_str, gap = generate_signal(p_bull, p_neutral, p_bear)
    bar = get_confidence_bar(gap)
    
    # TIER DETERMINATION
    tier = "NO SIGNAL"
    if gap > 20: tier = "HIGH CONVICTION"
    elif gap >= 10: tier = "MODERATE"
    
    # REGIME OVERRIDE
    latest_vix = latest_data['VIX_close'].values[0] if 'VIX_close' in latest_data.columns else 0
    regime_warning = ""
    if latest_vix > 30:
        regime_warning = f" ⚠️  HIGH VOLATILITY REGIME (VIX={latest_vix:.1f}) — PROCEED WITH CAUTION"

    # DASHBOARD FORMATTING (Compact & Visual)
    # Example: AAPL  🟢 BUY   ████████░░  Gap: +15pts  B:49% N:17% Be:34%
    line = f"{symbol:<6} {signal_str:<10} {bar}  Gap: {gap:>+5.1f}pts  B:{p_bull:>4.1f}% N:{p_neutral:>4.1f}% Be:{p_bear:>4.1f}%"
    
    if tier != "NO SIGNAL" and signal_str != '⚪ NO SIGNAL':
        line += f"  [{tier}]"
        
    if regime_warning:
        line += regime_warning

    print(line)
    report.append(line)
        
    return report

if __name__ == "__main__":
    model_dir = "models"
    
    import yaml
    config = {}
    try:
        with open("config/settings.yaml", "r") as f:
            config = yaml.safe_load(f)
    except:
        pass
    
    target_symbols = config.get("target_symbols", ["AAPL", "MSFT", "VRT", "AVGO", "SPY", "MELI"])
    
    # Load the single global model once
    model = StockModel()
    model.load(model_dir, "xgboost_global_stock_model")
    
    print(f"\n{'='*100}")
    print(f"{'SYMBOL':<6} {'SIGNAL':<10} {'CONFIDENCE':<10} {'METRICS':<30}")
    print(f"{'='*100}")
    
    all_reports = [f"# Latest Trading Signals ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})", ""]
    all_reports.append(f"{'SYMBOL':<6} {'SIGNAL':<10} {'CONFIDENCE':<10} {'METRICS'}")
    all_reports.append("-" * 100)

    for symbol in target_symbols:
        report_lines = predict_latest(symbol, model)
        all_reports.extend(report_lines)
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    output_path = "data/latest_predictions.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))
    
    print(f"{'='*100}")
    print(f"Results saved to {output_path}")

