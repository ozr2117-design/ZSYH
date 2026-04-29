import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import requests
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_fixed
import os
import threading
import time
import concurrent.futures
from streamlit_autorefresh import st_autorefresh
import pandas_ta as ta

# --- Hardcoded Config ---
STOCK_CODE = "600036.SH"
STOCK_CODE_AK = "600036" # For AKShare
BASE_SHARES = 4600
BASE_COST = 41.0
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Data Fetching Functions ---
@st.cache_data(ttl=30)
def get_spot_data():
    """Fetch real-time stock price via Sina Finance (fast, cloud-friendly)"""
    url = f"http://hq.sinajs.cn/list=sh{STOCK_CODE_AK}"
    headers = {"Referer": "http://finance.sina.com.cn/"}
    r = requests.get(url, headers=headers, timeout=8)
    data = r.text.split('"')[1].split(',')
    if len(data) < 4:
        raise ValueError(f"Invalid Sina response: {r.text[:100]}")
    price = float(data[3])
    if price <= 0:
        raise ValueError(f"Invalid price from Sina: {price}")
    return price

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _fetch_technical_data():
    start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    # Switch to Tencent Finance as requested to fix EastMoney connection issues
    df = ak.stock_zh_a_hist_tx(symbol=f"sh{STOCK_CODE_AK}", start_date=start_date, end_date=end_date, adjust="")
    if df.empty:
        raise ValueError("Empty dataframe from akshare")
    return df

@st.cache_data(ttl=3600)
def get_technical_indicators():
    df = _fetch_technical_data()
    df['close'] = pd.to_numeric(df['close'])
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    latest = df.iloc[-1]
    return float(latest['RSI_14']), float(latest['BBL_20_2.0'])


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

# ==========================================
# 标的估值基准手动配置文件 (需根据关键节点手动更新)
# ==========================================
stock_configs = {
    "600036.SH": {
        "stock_name": "招商银行",
        "anchor_bps": 44.90,         # 数据来源: 2026年一季报 (截至2026-03-31，归属于本行普通股股东的每股净资产)
        "deducted_dividend": 0,      # 一季报截止日后，1.013元分红已真实出表，物理对齐，清零
        "last_update": "2026-04-29"  # 维护时间戳，防止遗忘
    }
}

def get_manual_bps(stock_code: str):
    """
    根据手动配置的静态基准获取 BPS 基本信息
    """
    if stock_code not in stock_configs:
        return 39.0, "未知", 39.0, 0.0
        
    config = stock_configs[stock_code]
    raw_bps = config["anchor_bps"]
    div_amount = config["deducted_dividend"]
    real_bps = raw_bps - div_amount
    report_name = f"手动快照 {config['last_update']}"
    
    return real_bps, report_name, raw_bps, div_amount


def calculate_pyramid(price, pb, available_cash, t1_done, t2_done, t3_done, t4_done, rsi, bbl):
    """Calculate pyramiding logic based on 4 tiers, completion status, and technical indicators"""
    tier = 0
    alloc_ratio = 0.0
    buy_shares = 0
    actual_cost = 0.0
    rem_cash = available_cash
    warning_msg = None
    
    trigger = (rsi < 30) or (price <= bbl)
    
    # Base calculation amount is 300000
    base_calc_cash = 300000.0

    if pb > 0.90:
        tier = 0
        alloc_ratio = 0.0
    elif pb > 0.85:
        tier = 1
        if t1_done:
            warning_msg = "⚠️ **【按兵不动】您已完成第一档建仓！严禁重复买入，请锁死剩余资金，等待 PB 跌破 0.85！**"
        else:
            alloc_ratio = 0.15 # 45,000
    elif pb > 0.80:
        tier = 2
        if t2_done:
            warning_msg = "⚠️ **【按兵不动】您已完成核心区建仓！忍住波动，剩余子弹仅在 PB 跌破 0.80 时解锁！**"
        else:
            alloc_ratio = 0.20 # 60,000
    elif pb > 0.70:
        tier = 3
        if t3_done:
            warning_msg = "⚠️ **【按兵不动】您已买入历史深坑区！剩余末日子弹请锁死，仅在 PB 跌破 0.70 的极端绝境下使用！**"
        else:
            alloc_ratio = 0.30 # 90,000
    else:
        tier = 4
        if t4_done:
            warning_msg = "🛡️ **【子弹打光，安心躺平】您已买在历史绝对底部，卸载软件，安心吃股息。**"
        else:
            # 100% of remaining cash
            allocated_cash = available_cash
            if price > 0:
                buy_shares = math.floor(allocated_cash / (price * 100)) * 100
                actual_cost = buy_shares * price
                rem_cash = available_cash - actual_cost
            return tier, buy_shares, actual_cost, rem_cash, warning_msg, trigger

    if tier > 0 and warning_msg is None:
        target_allocation = base_calc_cash * alloc_ratio
        # Can't allocate more than what we actually have
        allocated_cash = min(target_allocation, available_cash)
        
        if price > 0:
            buy_shares = math.floor(allocated_cash / (price * 100)) * 100
            actual_cost = buy_shares * price
            rem_cash = available_cash - actual_cost
            
    return tier, buy_shares, actual_cost, rem_cash, warning_msg, trigger


def is_trading_time():
    """Check if current time is within A-share trading hours (09:15-11:30, 13:00-15:05) on a weekday"""
    tz_bj = timezone(timedelta(hours=8))
    now = datetime.now(tz_bj)
    if now.weekday() >= 5: # Saturday or Sunday
        return False
        
    current_time = now.time()
    morning_start = datetime.strptime("09:15", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("15:05", "%H:%M").time()
    
    if (morning_start <= current_time <= morning_end) or (afternoon_start <= current_time <= afternoon_end):
        return True
    return False

def render_valuation_maintenance_map():
    with st.sidebar.expander("🗺️ 招行 2026 估值维护节点地图", expanded=False):
        st.info("⚠️ 仅在以下关键节点触发时，需手动修改底层的 Anchor BPS 与 Deducted Dividend 参数。")
        
        # 核心维护节点表格
        maintenance_table = """
| 预计时间节点 | 触发事件 | Anchor BPS (净资产) 操作 | Deducted Dividend (已剥离现金) 操作 | 逻辑备忘 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **26年3月底** | 发布2025年报 | 更新为年报实际数据 | **保持 1.013 不变** | 1.013元是26年1月发的，年报截止日（25年底）这笔钱还在账上，需手动扣除。 | ✅ 已完成 |
| **26年4月底** | 发布26年一季报 | 更新为一季报普通股数据 (44.90) | **修改为 0** | 一季报截止日后，1.013元已真实出表，分母实现物理对齐。 | ✅ 已完成 (2026-04-29) |
| **26年6-7月** | 实施25年年度分红(除息日) | 保持一季报数据不变 | **修改为本次年度分红金额** | 股价已除息下跌，但财报未更新产生错位，必须手动累加扣除金额。 | ⏳ 待触发 |
        """
        st.markdown(maintenance_table)


def main():
    st.set_page_config(page_title="ZSYH Pyramiding Dashboard", layout="wide")
    
    # --- Sidebar UI ---
    st.sidebar.title("操作面板设置")
    available_cash_input = st.sidebar.number_input("当前剩余可用现金（元）", min_value=0.0, value=300000.0, step=1000.0)
    
    st.sidebar.markdown("### 建仓状态严格锁定")
    t1_done = st.sidebar.checkbox("已完成第一档加仓 (PB <= 0.90)", key="t1_done_cb")
    t2_done = st.sidebar.checkbox("已完成第二档加仓 (PB <= 0.85)", key="t2_done_cb")
    t3_done = st.sidebar.checkbox("已完成第三档加仓 (PB <= 0.80)", key="t3_done_cb")
    t4_done = st.sidebar.checkbox("已完成第四档加仓 (PB <= 0.70)", key="t4_done_cb")
    
    st.title("监控决策")
    
    if is_trading_time():
        st_autorefresh(interval=30000, key="data_refresh")
        st.caption("🔄 实盘数据已开启 30 秒自动刷新 (交易时段)")
    else:
        st.caption("⏸️ 非交易时段，自动刷新已暂停")
        
    render_valuation_maintenance_map()
    
    with st.status("📡 正在加载数据...", expanded=True) as status:
        try:
            st.write("🔗 获取实时股价 (新浪财经)...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_spot = executor.submit(get_spot_data)
                future_pb = executor.submit(get_historical_pb)
                future_macro = executor.submit(get_macro_data)
                future_tech = executor.submit(get_technical_indicators)

                st.write("📊 获取历史 PB 数据 (百度财经)...")
                price = future_spot.result(timeout=15)

                st.write("🔭 获取宏观 M1/M2 数据...")
                df_pb = future_pb.result(timeout=30)
                df_macro = future_macro.result(timeout=30)
                
                st.write("📈 获取技术超卖指标 (RSI/布林带)...")
                rsi, bbl = future_tech.result(timeout=30)

            adjusted_bps, report_name, raw_bps, div_amount = get_manual_bps(STOCK_CODE)
            bps = adjusted_bps
            pb = price / bps if bps > 0 else 1.0
            status.update(label="✅ 数据加载完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ 数据加载失败", state="error", expanded=True)
            st.error(f"数据获取失败: {e}")
            st.stop()

    # 1. 顶部数据概览 (st.metric)
    mv_base = BASE_SHARES * price
    mv_total = mv_base + available_cash_input
    profit_loss = (price - BASE_COST) * BASE_SHARES
    profit_loss_pct = (price / BASE_COST - 1) * 100 if BASE_COST > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1.2, 1.2, 1])
    col1.metric("当前股价", f"¥ {price:.2f}")
    col2.metric("最新BPS", f"¥ {bps:.2f}", help=f"数据来源: {report_name}")
    col3.metric("总市值 (含现金)", f"¥ {mv_total:,.0f}")
    col4.metric("浮动盈亏", f"¥ {profit_loss:,.0f}", f"{profit_loss_pct:.2f}%")
    col5.metric("当前 PB", f"{pb:.4f}")
    
    # 2. 技术指标概览
    st.markdown("---")
    tcol1, tcol2, tcol3 = st.columns([1, 1, 2])
    
    rsi_color = "🟢" if rsi < 30 else "🔴"
    bbl_color = "🟢" if price <= bbl else "🔴"
    
    tcol1.metric(f"{rsi_color} 当前 RSI(14)", f"{rsi:.2f}", "超卖区: <30", delta_color="off" if rsi >= 30 else "normal")
    tcol2.metric(f"{bbl_color} 布林带下轨", f"¥ {bbl:.2f}", "跌破触发" if price <= bbl else "未跌破下轨", delta_color="off" if price > bbl else "normal")
    tcol3.info("💡 战术进攻扳机：RSI < 30 或 当日收盘价跌破布林带下轨。只有当 PB 估值达标且战术扳机触发时，才构成强烈加仓共振。")

    # 3. 决策指令区 (st.success / st.warning / st.error)
    st.markdown("---")
    st.subheader("⚡ 今日操作建议")
    tier, buy_shares, actual_cost, rem_cash, warning_msg, trigger = calculate_pyramid(
        price, pb, available_cash_input, t1_done, t2_done, t3_done, t4_done, rsi, bbl
    )
    
    # Calculate next target price based on strictly new tiers
    next_pb_trigger = None
    if pb > 0.90:
        next_pb_trigger = 0.90
    elif pb > 0.85:
        next_pb_trigger = 0.85
    elif pb > 0.80:
        next_pb_trigger = 0.80
    elif pb > 0.70:
        next_pb_trigger = 0.70

    if next_pb_trigger is not None:
        target_price = next_pb_trigger * bps
        price_diff = price - target_price
        target_price_str = f"**📌 下一档触发目标价**：¥ {target_price:.2f}（距当前股价还差 ¥ {price_diff:.2f}）"
    else:
        target_price_str = "**📌 下一档触发目标价**：已到达底仓配置区间，无下一档目标价。"

    bps_explanation = f"(当前 BPS: {raw_bps:.2f} - 分红除息累计: {div_amount:.2f} = {bps:.2f})" if div_amount > 0 else f"(当前 BPS: {bps:.2f})"

    if tier == 0:
        st.info(f"☕ **估值偏高，持仓收息不动**\n\n🎯 **当前策略：观望期** | PB > 0.90\n\n预计消耗资金：¥ 0.00 | 剩余可用资金：¥ {available_cash_input:,.2f}\n\n{target_price_str}  {bps_explanation}")
    else:
        # Build message
        msg_header = f"🔥 **当前处于 第 {tier} 档 加仓区间**"
        if tier == 1:
             msg_header += " | PB: (0.85, 0.90]"
        elif tier == 2:
             msg_header += " | PB: (0.80, 0.85]"
        elif tier == 3:
             msg_header += " | PB: (0.70, 0.80]"
        elif tier == 4:
             msg_header += " | PB: <= 0.70 (打光子弹)"
             
        if warning_msg:
             st.error(f"{msg_header}\n\n{warning_msg}\n\n{target_price_str}  {bps_explanation}")
        else:
             if trigger:
                 st.success(f"🔥 强烈建议加仓：处于第 {tier} 档极寒区间且出现技术超卖共振\n\n{msg_header}\n\n**建议买入股数：{buy_shares:,} 股**\n\n预计消耗资金：¥ {actual_cost:,.2f} | 剩余可用资金：¥ {rem_cash:,.2f}\n\n{target_price_str}  {bps_explanation}")
             else:
                 st.warning(f"⏸️ 估值达标 (第 {tier} 档)，但未见超卖信号，建议持币观望等待技术底\n\n{msg_header}\n\n(原建议买入：{buy_shares:,} 股，当前因未触发超卖建议暂缓)\n\n预计消耗资金：¥ 0.00 | 剩余可用资金：¥ {available_cash_input:,.2f}\n\n{target_price_str}  {bps_explanation}")

    st.markdown("---")
    
    # 4. 估值走势图 (Plotly)
    st.subheader("📈 招商银行 近3年 PB 估值走势")
    try:
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Scatter(x=df_pb['date'], y=df_pb['value'], mode='lines', name='历史 PB', line=dict(color='blue')))
        
        # Add horizontal lines
        hline_config = [
            (0.90, '观望/一档边界 (0.90)', 'gray'),
            (0.85, '一档/二档边界 (0.85)', 'green'),
            (0.80, '二档/三档边界 (0.80)', 'orange'),
            (0.70, '三档/四档边界 (0.70)', 'red')
        ]
        for val, name, color in hline_config:
            fig_pb.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=name, annotation_position="top right")
            
        # Also mark current PB
        fig_pb.add_hline(y=pb, line_dash="solid", line_color="purple", annotation_text=f"当前 PB: {pb:.4f}", annotation_position="bottom right")
            
        fig_pb.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="日期", yaxis_title="PB 值")
        st.plotly_chart(fig_pb, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading historical PB data: {e}")

    # 5. 宏观观察哨 (Plotly)
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
