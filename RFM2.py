import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
# 資料預處理
data = pd.read_excel('全國訂單明細.xlsx')
df = pd.DataFrame(data)
# 顯示資料的摘要
#print(df.head())  # 顯示前幾列
#print(df.info())  # 顯示資料型別和缺失值
print(df.describe())  # 顯示摘要統計
# 找出訂單號相同但顧客資料不一樣的資料

# 使用 groupby 和 filter 找出訂單號相同但顧客姓名不同的資料
result = df.groupby('訂單號').filter(lambda x: x['顧客姓名'].nunique() > 1)
#print(result.head())
# 刪除這些資料
df = df.drop(result.index)
# 轉換 '訂單日期' 欄位為日期格式
df['訂單日期'] = pd.to_datetime(df['訂單日期'])

# 計算 Recency
snapshot_date = df['訂單日期'].max() + pd.Timedelta(days=1)
df['Recency'] = (snapshot_date - df['訂單日期']).dt.days

# 計算 Frequency 和 Monetary
rfm = df.groupby('顧客姓名').agg({
    'Recency': 'min',
    '訂單號': 'count',
    '銷售額': 'sum'
})
rfm.columns = ['最近消費天數', '消費頻率', '消費金額']
print(rfm)
# 選擇 K 值
k = 4

# 使用 K-均值聚類
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(rfm)



'''# 將購買金額（Monetary）的分布狀況可視化，以決定高級顧客閥值
plt.hist(rfm['Monetary'], bins=20)
plt.xlabel('Monetary')
plt.ylabel('Frequency')
plt.title('購買金額分布圖')
plt.show()'''

# 根據觀察選擇一個閥值，將購買金額高於該閥值的顧客標記為高級顧客，其餘為普通顧客
high_value_threshold = 3000  # 假設閥值為3000，您可以根據實際情況調整
rfm['Customer_Type'] = 'Regular'  # 預設所有顧客都是普通顧客
rfm.loc[rfm['消費金額'] >= high_value_threshold, 'Customer_Type'] = 'High Value'

# 可視化高級顧客和普通顧客的散點圖
plt.scatter(rfm[rfm['Customer_Type'] == 'Regular']['消費頻率'], rfm[rfm['Customer_Type'] == 'Regular']['消費金額'], label='普通顧客', color='blue', alpha=0.5)
plt.scatter(rfm[rfm['Customer_Type'] == 'High Value']['消費頻率'], rfm[rfm['Customer_Type'] == 'High Value']['消費金額'], label='高級顧客', color='red', alpha=0.5)
plt.xlabel('消費頻率')
plt.ylabel('消費金額')
plt.title('高級客戶與普通客戶散布圖')
plt.legend()
plt.show()
# 計算顧客的總銷售額並計算平均值
total_sales = data.groupby('顧客姓名')['銷售額'].sum()
average_sales = total_sales.mean()

# 將顧客分為高級顧客和普通顧客
data['Customer_Category'] = '普通顧客'
data.loc[data.groupby('顧客姓名')['銷售額'].transform('sum') > average_sales, 'Customer_Category'] = '高級顧客'

# 將 '銷售額' 和 '利潤額' 當作特徵
X = data[['銷售額', '利潤額']]

# 使用 K-means 聚類
k = 2  # 設置聚類數量為2
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# 獲取聚類中心和標籤
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 繪製散步圖
plt.figure(figsize=(8, 6))

plt.scatter(X['銷售額'], X['利潤額'], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=300)

plt.title('銷售額與利潤額的散佈圖')
plt.xlabel('銷售額')
plt.ylabel('利潤額')
plt.grid(True)
plt.show()
#-------------------------------------------------------------
#產品關聯度分析
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
basket = (data.groupby(['顧客姓名', '產品子類別'])['訂單數量']
          .sum().unstack().reset_index().fillna(0)
          .set_index('顧客姓名'))

# 轉換數據格式為0和1
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# 進行產品關聯度分析
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# 顯示關聯規則
print(rules.head(10))
#-----------------------------------------------------------------------
# 計算每個產品的購買次數
product_preference = data.groupby('產品子類別')['訂單數量'].sum().reset_index()

# 計算喜好度（這裡用購買次數當作喜好度的衡量標準，你也可以使用其他指標如銷售額等）
product_preference['喜好度'] = product_preference['訂單數量']

# 計算總購買次數（或其他指標）
total_purchases = product_preference['喜好度'].sum()

# 計算喜好度的百分比
product_preference['喜好度百分比'] = (product_preference['喜好度'] / total_purchases) * 100

# 根據喜好度百分比排序
product_preference = product_preference.sort_values(by='喜好度百分比', ascending=False)

# 顯示結果
#print(product_preference)
top_five_products = product_preference.head(5)
print(top_five_products)
#--------------------------------------------------------------------------------
'''# 找出銷售額和訂單數量最高的顧客
max_sales_customer = df.loc[df['銷售額'].idxmax()]
max_orders_customer = df.loc[df['訂單數量'].idxmax()]

# 顯示結果
print("銷售額最高的顧客：")
print(max_sales_customer)
print("\n訂單數量最高的顧客：")
print(max_orders_customer)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# 假設這是你的資料框架 df，確保它包含必要的資訊如 '顧客姓名' 和 '產品名稱'
# 在這個範例中，使用 LabelEncoder 將顧客姓名和產品名稱轉換成數字以進行計算
le_customer = LabelEncoder()
le_product = LabelEncoder()

df['顧客編碼'] = le_customer.fit_transform(df['顧客姓名'])
df['產品編碼'] = le_product.fit_transform(df['產品名稱'])

# 創建顧客-產品矩陣，這裡使用 pivot table
customer_product_matrix = df.pivot_table(index='顧客編碼', columns='產品編碼', values='訂單數量', fill_value=0)

# 計算餘弦相似度
similarities = cosine_similarity(customer_product_matrix)
print("顧客相似度矩陣:")
print(similarities)

# 進行推薦
def recommend_products(target_customer_name, num_recommendations=3):
    target_customer = df[df['顧客姓名'] == target_customer_name].iloc[0]
    target_customer_id = target_customer['顧客編碼']
    similar_customers = similarities[target_customer_id]
    similar_customer_ids = similar_customers.argsort()[::-1][1:]  # 排除自己，找出相似度最高的顧客
    
    products_purchased = set(df[df['顧客編碼'] == target_customer_id]['產品名稱'])
    recommendations = defaultdict(int)
    
    for customer_id in similar_customer_ids:
        similar_customer_products = df[df['顧客編碼'] == customer_id]['產品名稱']
        for product in similar_customer_products:
            if product not in products_purchased:
                recommendations[product] += 1  # 推薦產品計數
    
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    recommended_products = [product[0] for product in recommendations]  # 不再轉換為整數
    
    return recommended_products

# 替換 '張國華' 為目標顧客姓名
target_customer_name = '楊肇輝'
recommended_products = recommend_products(target_customer_name)

print(f"為顧客 {target_customer_name} 推薦的產品:")
for product in recommended_products:
    print(product)

# 根據運輸方式計算平均運輸成本和平均交貨速度
average_costs = df.groupby('運輸方式')['運輸成本'].mean().sort_values()


# 繪製運輸成本比較圖
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
average_costs.plot(kind='bar', color='skyblue')
plt.title('不同運輸方式的平均運輸成本')
plt.xlabel('運輸方式')
plt.ylabel('平均運輸成本')

plt.tight_layout()
plt.show()'''

# 假設資料中有顧客ID和銷售額兩列
top_customer = df.groupby('顧客姓名')['銷售額'].sum().idxmax()
print(top_customer)
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 您的資料集名為 df
# 假設有一個 df 包含客戶和其購買商品的特徵，並且某一列代表 '楊肇輝' 的資料
# 可以使用 cosine_similarity 計算楊肇輝與其他客戶的商品相似度

# 假設您有一個包含客戶商品特徵的 DataFrame，名為 df
# 假設 '楊肇輝' 的資料位於索引為 0 的行
'''customer_features = df.drop(columns=['顧客姓名'])  # 假設去除了顧客 ID 欄位以外的其他特徵

# 選擇楊肇輝的特徵資料
yang_zhaohui_features = customer_features.iloc[0].values.reshape(1, -1)  # 假設楊肇輝位於第一行
from sklearn.preprocessing import LabelEncoder

# 指定要進行 Label Encoding 的欄位名稱
columns_to_encode = ['訂單號', '顧客姓名', '訂單等級', '運輸方式', '區域', '省份', '城市', '產品類別', '產品子類別', '產品名稱', '產品包箱']

# 初始化 LabelEncoder
label_encoder = LabelEncoder()

# 對每個指定的欄位進行 Label Encoding
for col in columns_to_encode:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col].astype(str))

# 顯示轉換後的 DataFrame
print(df.head())
# 找出楊肇輝的索引
yang_zhaohui_index = df[df['顧客姓名'] == '楊肇輝'].index[0]

# 楊肇輝的商品特徵向量
yang_zhaohui_features = df.iloc[yang_zhaohui_index].values.reshape(1, -1)

# 所有顧客的商品特徵向量
customer_features = df.drop(columns=['顧客姓名_encoded'])  # 除了顧客姓名的特徵向量

# 計算楊肇輝與其他顧客的商品相似度
similarity_matrix = cosine_similarity(customer_features, yang_zhaohui_features)'''

