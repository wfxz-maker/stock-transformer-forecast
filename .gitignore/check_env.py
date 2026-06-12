# -*- coding: utf-8 -*-
import pandas as pd
# 这里需要保留你原来导入baostock的代码，比如：
import baostock as bs

# ========== 1. 初始化和参数设置 ==========
# 登录baostock
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 设置时间范围
start_date = "2021-01-01"
end_date = "2026-05-31"

# 获取沪深300成分股列表
rs = bs.query_hs300_stocks()
hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    hs300_stocks.append(rs.get_row_data())
hs300_df = pd.DataFrame(hs300_stocks, columns=rs.fields)
print(f"沪深300成分股总数：{len(hs300_df)}")

# ========== 2. 初始化数据保存 ==========
all_data = pd.DataFrame()
save_path = "hs300_stock_data_2021_2026.csv"

# 断点续跑：如果文件已存在，就接着上次的数据继续
try:
    all_data = pd.read_csv(save_path, encoding="utf-8-sig")
    print(f"检测到已有数据，当前已获取股票数量：{all_data['code'].nunique()}")
except FileNotFoundError:
    print("未找到已有数据，开始全新获取...")

# ========== 3. 循环获取数据 ==========
for code in hs300_df['code']:
    print(f"正在获取股票: {code}")
    try:
        # 获取K线数据
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )

        stock_data = []
        while (rs.error_code == '0') & rs.next():
            stock_data.append(rs.get_row_data())

        # 处理并保存数据
        if len(stock_data) > 0:
            df = pd.DataFrame(stock_data, columns=rs.fields)
            df['code'] = code
            all_data = pd.concat([all_data, df], ignore_index=True)
            
            # 边跑边存，防止卡住丢数据
            all_data.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ {code} 数据已保存，当前共 {len(all_data)} 条数据")
        else:
            print(f"⚠️ {code} 无数据，跳过")

    except Exception as e:
        print(f"❌ {code} 获取失败，错误信息：{e}，继续下一只")
        continue

# ========== 4. 数据清洗 ==========
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    all_data[col] = pd.to_numeric(all_data[col])

# 去除空值和异常值
all_data = all_data.dropna()
all_data = all_data[(all_data['close'] > 0) & (all_data['volume'] > 0)]

# 保存清洗后的数据
clean_path = "hs300_stock_data_clean_2021_2026.csv"
all_data.to_csv(clean_path, index=False, encoding="utf-8-sig")
print(f"🎉 数据清洗完成，已保存为：{clean_path}")

# 退出登录
bs.logout()
