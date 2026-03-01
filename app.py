import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import threading
import time
import concurrent.futures

# --- Hardcoded Config ---
STOCK_CODE = "600036.SH"
STOCK_CODE_AK = "600036" # For AKShare
BASE_SHARES = 4600
BASE_COST = 41.0
AVAILABLE_CASH = 300000.0
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(page_title="ZSYH Pyramiding Dashboard", layout="wide")

# --- Data Fetching Functions ---
@st.cache_data(ttl=300)
def get_spot_data():
    """Fetch real-time stock data"""
    try:
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
        def fetch_em():
            df = ak.stock_zh_a_spot_em()
            zsyh = df[df["代码"] == STOCK_CODE_AK]
            if zsyh.empty:
                raise ValueError("Cannot find stock data for 600036")
            zsyh = zsyh.iloc[0]
            
            price = float(zsyh["最新价"])
            pb = float(zsyh["市净率"])
            if pd.isna(pb) or pb == 0:
                pb = 1.0
            bps = price / pb 
            return price, pb, bps
            
        return fetch_em()
    except Exception as e:
        try:
            import requests
            url = f"http://hq.sinajs.cn/list=sh{STOCK_CODE_AK}"
            headers = {"Referer": "http://finance.sina.com.cn/"}
            r = requests.get(url, headers=headers, timeout=5)
            data = r.text.split('"')[1].split(',')
            if len(data) < 4:
                raise ValueError("Invalid Sina data")
                
            price = float(data[3])
            yest_close = float(data[2])
            
            try:
                df_pb = get_historical_pb()
                last_pb = float(df_pb.iloc[-1]['value'])
                bps = yest_close / last_pb if last_pb > 0 else 38.0
            except:
                bps = 39.0 # Approximated safe fallback BPS for 600036
                
            pb = price / bps if bps > 0 else 1.0
            return price, pb, bps
        except Exception as fallback_e:
            raise ValueError(f"Spot EM failed: {e}. Fallback Sina failed: {fallback_e}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _fetch_and_save_historical_pb():
    """Fetch 3-year historical PB and save to CSV"""
    df = ak.stock_zh_valuation_baidu(symbol=STOCK_CODE_AK, indicator="市净率", period="近十年")
    df['date'] = pd.to_datetime(df['date'])
    # Filter 3 years
    three_years_ago = datetime.now() - timedelta(days=3*365)
    df = df[df['date'] >= three_years_ago]
    df.to_csv(os.path.join(DATA_DIR, "historical_pb.csv"), index=False)

@st.cache_data(ttl=3600)
def get_historical_pb():
    """Get historical PB from local CSV. If missing, fetch synchronously. If stale, fetch in background."""
    file_path = os.path.join(DATA_DIR, "historical_pb.csv")
    
    if not os.path.exists(file_path):
        _fetch_and_save_historical_pb()
        
    else:
        file_time = os.path.getmtime(file_path)
        if (time.time() - file_time) > (12 * 3600): # 12 hours
            threading.Thread(target=_fetch_and_save_historical_pb, daemon=True).start()
            
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _fetch_and_save_macro_data():
    """Fetch M1/M2 data for 5 years and save to CSV"""
    df = ak.macro_china_money_supply()
    # Extract year and month, clean robustly
    df['Month'] = df['月份'].astype(str)
    def parse_chinese_date(ds):
        try:
            ds = ds.replace('月份', '').replace('月', '')
            year, month = ds.split('年')
            return pd.to_datetime(f"{year}-{month}-01")
        except:
            return pd.NaT
            
    df['date'] = df['Month'].apply(parse_chinese_date)
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    
    # Filter 5 years
    five_years_ago = datetime.now() - timedelta(days=5*365)
    df = df[df['date'] >= five_years_ago]
    
    # Dynamic column name discovery
    m2_yoy_col = [c for c in df.columns if 'M2' in c and '同比' in c][0]
    m1_yoy_col = [c for c in df.columns if 'M1' in c and '同比' in c][0]
    
    df['M1_YoY'] = pd.to_numeric(df[m1_yoy_col], errors='coerce')
    df['M2_YoY'] = pd.to_numeric(df[m2_yoy_col], errors='coerce')
    df['Scissors'] = df['M1_YoY'] - df['M2_YoY']
    
    df.to_csv(os.path.join(DATA_DIR, "macro_data.csv"), index=False)

@st.cache_data(ttl=3600)
def get_macro_data():
    """Get macro data from local CSV. If missing, fetch synchronously. If stale, fetch in background."""
    file_path = os.path.join(DATA_DIR, "macro_data.csv")
    
    if not os.path.exists(file_path):
        _fetch_and_save_macro_data()
        
    else:
        file_time = os.path.getmtime(file_path)
        if (time.time() - file_time) > (12 * 3600): # 12 hours
            threading.Thread(target=_fetch_and_save_macro_data, daemon=True).start()
            
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_pyramid(price, pb):
    """Calculate pyramiding logic"""
    tier = 0
    alloc_ratio = 0.0
    
    if pb > 0.90:
        tier = 0
        alloc_ratio = 0.0
    elif pb > 0.85:
        tier = 1
        alloc_ratio = 0.10
    elif pb > 0.75:
        tier = 2
        alloc_ratio = 0.20
    elif pb > 0.65:
        tier = 3
        alloc_ratio = 0.30
    else:
        tier = 4
        alloc_ratio = 1.00
        
    allocated_cash = AVAILABLE_CASH * alloc_ratio
    # Shares must be multiple of 100
    if price > 0:
        buy_shares = math.floor(allocated_cash / (price * 100)) * 100
    else:
        buy_shares = 0
        
    actual_cost = buy_shares * price
    rem_cash = AVAILABLE_CASH - actual_cost
    
    return tier, buy_shares, actual_cost, rem_cash


def main():
    st.title("💰 招商银行 (600036.SH) 个人家庭资产监控与加仓决策看板")
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_spot = executor.submit(get_spot_data)
            future_pb = executor.submit(get_historical_pb)
            future_macro = executor.submit(get_macro_data)
            
            price, pb, bps = future_spot.result()
            df_pb = future_pb.result()
            df_macro = future_macro.result()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    # 1. 顶部数据概览 (st.metric)
    mv_base = BASE_SHARES * price
    mv_total = mv_base + AVAILABLE_CASH
    profit_loss = (price - BASE_COST) * BASE_SHARES
    profit_loss_pct = (price / BASE_COST - 1) * 100 if BASE_COST > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("当前股价", f"¥ {price:.2f}")
    col2.metric("最新BPS (推算)", f"¥ {bps:.2f}")
    col3.metric("总持仓市值 (含现金)", f"¥ {mv_total:,.2f}")
    col4.metric("浮动盈亏", f"¥ {profit_loss:,.2f}", f"{profit_loss_pct:.2f}%")
    col5.metric("当前 PB (实盘)", f"{pb:.4f}")

    # 2. 决策指令区 (st.success / st.warning / st.error)
    st.markdown("---")
    st.subheader("⚡ 今日操作建议")
    tier, buy_shares, actual_cost, rem_cash = calculate_pyramid(price, pb)
    
    if tier == 0:
        st.info("🎯 **当前策略：观望期** | PB > 0.90\n\n**操作建议：【耐心持有，绝不加仓】**\n\n预计消耗资金：¥ 0.00 | 剩余可用资金：¥ {:,.2f}".format(AVAILABLE_CASH))
    else:
        # Build message
        msg_header = f"🔥 **当前处于 第 {tier} 档 加仓区间**"
        if tier == 1:
             msg_header += " | PB: (0.85, 0.90]"
        elif tier == 2:
             msg_header += " | PB: (0.75, 0.85]"
        elif tier == 3:
             msg_header += " | PB: (0.65, 0.75]"
        elif tier == 4:
             msg_header += " | PB: <= 0.65 (打光子弹)"
             
        msg_body = f"**建议买入股数：{buy_shares:,} 股**\n\n预计消耗资金：¥ {actual_cost:,.2f} | 剩余可用资金：¥ {rem_cash:,.2f}"
        
        if tier == 1:
            st.success(f"{msg_header}\n\n{msg_body}")
        elif tier == 2:
            st.warning(f"{msg_header}\n\n{msg_body}")
        else:
            st.error(f"{msg_header}\n\n{msg_body}")

    st.markdown("---")
    
    # 3. 估值走势图 (Plotly)
    st.subheader("📈 招商银行 近3年 PB 估值走势")
    try:
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Scatter(x=df_pb['date'], y=df_pb['value'], mode='lines', name='历史 PB', line=dict(color='blue')))
        
        # Add horizontal lines
        hline_config = [
            (0.90, '观望/一档边界 (0.90)', 'gray'),
            (0.85, '一档/二档边界 (0.85)', 'green'),
            (0.75, '二档/三档边界 (0.75)', 'orange'),
            (0.65, '三档/四档边界 (0.65)', 'red')
        ]
        for val, name, color in hline_config:
            fig_pb.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=name, annotation_position="top right")
            
        # Also mark current PB
        fig_pb.add_hline(y=pb, line_dash="solid", line_color="purple", annotation_text=f"当前 PB: {pb:.4f}", annotation_position="bottom right")
            
        fig_pb.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="日期", yaxis_title="PB 值")
        st.plotly_chart(fig_pb, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading historical PB data: {e}")

    # 4. 宏观观察哨 (Plotly)
    st.markdown("---")
    st.subheader("🔭 宏观观察哨：M1 与 M2 同比增速剪刀差 (近5年)")
    try:
        fig_macro = make_subplots(specs=[[{"secondary_y": False}]])
        
        # M1 and M2 Lines
        fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['M1_YoY'], mode='lines+markers', name='M1 同比(%)', line=dict(color='blue')))
        fig_macro.add_trace(go.Scatter(x=df_macro['date'], y=df_macro['M2_YoY'], mode='lines+markers', name='M2 同比(%)', line=dict(color='orange')))
        
        # Scissors Bar chart
        colors = ['red' if val > 0 else 'green' for val in df_macro['Scissors']]
        fig_macro.add_trace(go.Bar(x=df_macro['date'], y=df_macro['Scissors'], name='M1-M2 剪刀差', marker_color=colors, opacity=0.6))
        
        fig_macro.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_macro, use_container_width=True)
        
        st.info("注：宏观数据具滞后性，M1-M2 负剪刀差扩大代表存款定期化严重，息差承压。此图表仅作宏观周期判定，加仓唯一依据为 PB 估值，切勿因宏观数据回暖而盲目追高。")
    except Exception as e:
        st.error(f"Error loading macro data: {e}")

if __name__ == "__main__":
    main()
