import pandas as pd

# 資料預處理
data = pd.read_excel('全國訂單明細.xlsx')
df = pd.DataFrame(data)

# 顯示資料的摘要
print(df.head(10))  # 顯示前幾列
print(df.info())  # 顯示資料型別和缺失值
print(df.describe())  # 顯示摘要統計

# 找出訂單號相同但顧客資料不一樣的資料
# 使用 groupby 和 filter 找出訂單號相同但顧客姓名不同的資料
result = df.groupby('訂單號').filter(lambda x: x['顧客姓名'].nunique() > 1)
print(result.head())

# 刪除這些資料
df = df.drop(result.index)
print(df.describe())
print(df.tail(5))

#------------------------------------------------------------------------------
#探索性分析
# 視覺化探索
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
# 使用柱狀圖來展示
plt.figure(figsize=(12, 8))
sns.countplot(x='區域', hue='訂單等級', data=df, palette='viridis')
plt.title('各區域的訂單等級分布')
plt.xlabel('區域')
plt.ylabel('數量')
plt.legend(title='Order Level', loc='upper right')
plt.show()


# 根據產品類別計算總銷售額
sales_by_category = df.groupby('產品子類別')['銷售額'].sum().sort_values(ascending=False)

# 繪製長條圖
plt.figure(figsize=(10, 6))
sales_by_category.plot(kind='bar', color='skyblue')
plt.title('產品類別與總銷售額')
plt.xlabel('產品子類別')
plt.ylabel('總銷售額')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#顯示負號
matplotlib.rcParams['axes.unicode_minus']=False
# 根據產品子類別計算總利潤額
profit_by_subcategory = df.groupby('產品子類別')['利潤額'].sum().sort_values()

#分離正負值
positive_profits = profit_by_subcategory[profit_by_subcategory >= 0]
negative_profits = profit_by_subcategory[profit_by_subcategory < 0]

# 繪製帶負值的垂直長條圖
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

axes[0].bar(positive_profits.index, positive_profits, color='green')
axes[0].set_title('正利潤額')
axes[0].set_xlabel('產品子類別')
axes[0].set_ylabel('總利潤額')

axes[1].bar(negative_profits.index, negative_profits, color='red')
axes[1].set_title('負利潤額')
axes[1].set_xlabel('產品子類別')
axes[1].set_ylabel('總利潤額')

plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------

data['訂單日期'] = pd.to_datetime(data['訂單日期'])
data['year'] = data['訂單日期'].dt.to_period('Y')
order_count_per_month = data.groupby('year').size()
order_count_per_month = data.set_index('訂單日期').resample('Y')['顧客姓名'].nunique()

#顧客分析
# 依地區計算顧客數量
customer_by_region = df.groupby('區域')['顧客姓名'].nunique().sort_values()

# 繪製長條圖
customer_by_region.plot(kind='bar', figsize=(10, 6))
plt.title('各地區顧客人數')
plt.xlabel('地區')
plt.ylabel('顧客人數')
plt.xticks(rotation=45)
plt.show()

# 假設資料中的日期欄位是日期時間格式，如果不是，請先將其轉換為日期時間格式
df['訂單日期'] = pd.to_datetime(df['訂單日期'])

# 以顧客姓名分組，計算每個顧客的訂單次數
order_count_per_customer = df.groupby('顧客姓名').size()

# 將訂單次數為1的視為初次客，大於1的為回頭客
first_time_customers = order_count_per_customer[order_count_per_customer == 1].count()
repeat_customers = order_count_per_customer[order_count_per_customer > 1].count()

# 可視化初次客和回頭客的比例
labels = ['初次客', '回頭客']
sizes = [first_time_customers, repeat_customers]
colors = ['gold', 'lightblue']
explode = (0, 0)  # 讓圓餅圖平坦，不分離出來

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('初次客與回頭客比例')
plt.show()


#計算回頭客數量及百分比
#訂單匯總: 根據顧客編號匯總訂單資料，算出銷售總額、訂單日期
def groupby_mean(x):
    return x.mean()

def groupby_count(x):
    return x.count()

def purchase_duration(x):
    return (x.max() - x.min()).days #最後一次與第一次的間隔時間

def avg_frequency(x):
    return (x.max() - x.min()).days/x.count() #最後一次與第一次的間隔時間/次數

groupby_mean.__name__ = 'avg'
groupby_count.__name__ = 'count'
purchase_duration.__name__ = 'purchase_duration'
avg_frequency.__name__ = 'purchase_frequency'


#計算每一個顧客匯總資料
summary_df = data.groupby(['訂單號']).agg({
    '銷售額': [min, max, sum, groupby_mean, groupby_count],
    '訂單日期': [min, max, purchase_duration, avg_frequency]
})

# 顯示匯總資料
print(summary_df)
summary_df

#更改欄位名稱
summary_df.columns = ['_'.join(col).lower() for col in summary_df.columns]

#顯示匯總資料
#print(summary_df)



# 篩選至少購買兩次顧客-回頭客
summary_df= summary_df[summary_df['銷售額_count'] > 0]
#print(summary_df)

#繪圖: 顧客購買次數分布圖
#summary_df = summary_df[summary_df['銷售額_count'] > 0]

# 繪製顧客購買次數分布圖
summary_df.groupby('銷售額_count')['銷售額_avg'].count().plot(
    kind='bar',
    color='skyblue',
    figsize=(12, 7),
    grid=True
)

plt.title('購買次數長條圖')
plt.xlabel('購買次數')
plt.ylabel('總計')

plt.show()

#依年份計算回頭客數量以及百分比，並畫出現有顧客以及回頭客的數量以及百分比，當年有重複購買視為回頭客

 
monthly_returning_customers = data.groupby(['year', '顧客姓名']).size().groupby('year').apply(lambda x: (x > 1).sum())

# 計算每個月的總顧客數量
monthly_total_customers = data.groupby('year')['顧客姓名'].nunique()

# 計算回頭客百分比
returning_customers_percentage = (monthly_returning_customers / monthly_total_customers) * 100
print(returning_customers_percentage)
# 繪製合併的圖表
fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(
    [str(idx) for idx in monthly_total_customers.index.astype(str)],
    monthly_total_customers,
    marker='o',
    label='現有顧客',
    color='blue'
)
ax1.plot(
    [str(idx) for idx in monthly_returning_customers.index.astype(str)],
    monthly_returning_customers,
    marker='o',
    label='回頭客',
    color='green'
)

# 設定折線圖屬性
ax1.set_xlabel('年份')
ax1.set_ylabel('顧客數量')
ax1.legend(loc='upper left')

# 創建長條圖軸
ax2 = ax1.twinx()
ax2.bar(
    [str(idx) for idx in returning_customers_percentage.index.astype(str)],
    returning_customers_percentage,
    color='skyblue',
    alpha=0.7,
    label='returning_customers(%)'
)

# 設定長條圖屬性
ax2.set_ylabel('回頭客(%)', color='skyblue')
ax2.tick_params('y', colors='skyblue')
ax2.legend(loc='upper right')

# 設定圖表屬性
plt.title('現有顧客以及回頭客的數量(%)')
plt.xticks(rotation=45)

# 顯示圖表
plt.show()

# 依區域分組計算回頭客數量以及總顧客數量
returning_customers_per_region = data.groupby(['區域', '顧客姓名']).size().groupby('區域').apply(lambda x: (x > 1).sum())
total_customers_per_region = data.groupby('區域')['顧客姓名'].nunique()

# 計算回頭客比例
returning_customers_percentage_per_region = (returning_customers_per_region / total_customers_per_region) * 100

# 繪製圓餅圖
plt.figure(figsize=(10, 6))
plt.pie(returning_customers_percentage_per_region, labels=returning_customers_percentage_per_region.index, autopct='%1.1f%%')
plt.title('各區域回頭客比例')
plt.axis('equal')  # 使圓餅圖呈現為圓形
plt.show()

# 依區域分組計算初次客數量以及總顧客數量
first_time_customers_per_region = data.groupby(['區域', '顧客姓名']).size().groupby('區域').apply(lambda x: (x == 1).sum())
total_customers_per_region = data.groupby('區域')['顧客姓名'].nunique()

# 計算初次客比例
first_time_customers_percentage_per_region = (first_time_customers_per_region / total_customers_per_region) * 100

# 繪製圓餅圖
plt.figure(figsize=(10, 6))
plt.pie(first_time_customers_percentage_per_region, labels=first_time_customers_percentage_per_region.index, autopct='%1.1f%%')
plt.title('各區域初次客比例')
plt.axis('equal')  # 使圓餅圖呈現為圓形
plt.show()

# 計算每個顧客的總利潤額
total_profit_per_customer = data.groupby('顧客姓名')['利潤額'].sum()

# 使用 filter 函數來篩選初次客和回頭客的利潤額
first_time_customers_profit = total_profit_per_customer.groupby(data.groupby('顧客姓名').size().eq(1)).get_group(True)
repeat_customers_profit = total_profit_per_customer.groupby(data.groupby('顧客姓名').size().gt(1)).get_group(True)

# 計算初次客和回頭客的總利潤額
first_time_customers_total_profit = first_time_customers_profit.sum()
repeat_customers_total_profit = repeat_customers_profit.sum()

# 建立圓餅圖資料
profits = [first_time_customers_total_profit, repeat_customers_total_profit]
labels = ['初次客', '回頭客']

# 繪製圓餅圖
plt.figure(figsize=(8, 6))
plt.pie(profits, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('回頭客帶來的利潤額')
plt.axis('equal')  # 讓圓餅圖呈現為圓形
plt.show()