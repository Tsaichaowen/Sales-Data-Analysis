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

#--------------------------------------------------------------------------------------------------
#計算每一個顧客RFM資料
import datetime as dt

maximum = max(df.訂單日期)

maximum

theday=dt.datetime.strptime("2012-12-31","%Y-%m-%d")

summary_df = df.groupby('顧客姓名').agg({
    '銷售額': lambda x: x.sum(),
     '訂單日期': lambda x: (theday - pd.to_datetime(x.max(), errors='coerce')).days,
    '產品子類別': lambda x: x.nunique()
})



summary_df

#彙總後更改欄位名稱
summary_df.columns = ['monetary','recency','freqnency']
print(summary_df)
 
#去除recency後面多餘文字
summary_df["recency"]=summary_df["recency"].astype(str)
#summary_df["recency"]=summary_df["recency"].str.replace("days.*","")
summary_df["recency"]=summary_df["recency"].str.replace("days.*", "", regex=True)

summary_df

#RFM分類
summary_df["recency"]=summary_df["recency"].astype(int)
quantiles =summary_df.quantile(q=[0.2,0.4,0.6,0.8])

#定義Recency的分數，第一四分位以內的天數資料，給予4分，第三四分位數後的資料給予1分。
def R_Class(x,p,d):
    if x <= d[p][0.2]:
        return 5
    elif x <= d[p][0.4]:
        return 4
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]: 
        return 2
    else:
        return 1
   

#問題:def FM_Class(x,p,d)怎麼寫
#定義Frequency、Monetary的分數，第三四分位以後的天數資料，給予4分，第一四分位數內的資料給予1分
def FM_Class(x,p,d):
  
    if x <= d[p][0.2]:
        return 1
    elif x <= d[p][0.4]:
        return 2
    elif x <= d[p][0.6]: 
        return 3
    elif x <= d[p][0.8]: 
        return 4
    else:
        return 5
    
#RFM分類    
RFM_Segment = summary_df.copy()
RFM_Segment['R_Quartile'] = RFM_Segment['recency'].apply(R_Class, args=('recency',quantiles))
RFM_Segment['F_Quartile'] = RFM_Segment['freqnency'].apply(FM_Class, args=('freqnency',quantiles))
RFM_Segment['M_Quartile'] = RFM_Segment['monetary'].apply(FM_Class, args=('monetary',quantiles))

#最後將RFM三欄的分數變成字串，形成新的欄位，對於賣場最有價值的顧客其分類代號為”444″，最沒有價值的顧客其分類代號為”111″
RFM_Segment['RFMClass'] = RFM_Segment.R_Quartile.map(str) \
                            + RFM_Segment.F_Quartile.map(str) \
                            + RFM_Segment.M_Quartile.map(str)
RFM_means = summary_df[['recency', 'freqnency', 'monetary']].mean()
print("RFM平均數：")
print(RFM_means)                            
#問題: 顧客分類代號為 “555” 的黃金顧客佔全部顧客的比例                  
VIP_percentage=len(RFM_Segment[RFM_Segment['RFMClass']=='555'])/len(RFM_Segment)
print(f"黃金顧客佔全部顧客的比例是:{round(VIP_percentage,2)}%")


#----------------------------------------------------------------------------------------
#將FM自訂分成高低
#依中位數或平均數切frequency高低
summary_df['freqnency_label'] = summary_df['freqnency'].apply(
    lambda x: 'High' if x > summary_df['freqnency'].mean() else 'Low'
)

#另一種寫法
# cutlevel=[summary_df['freqnency'].min()-1,summary_df['freqnency'].mean(),summary_df['freqnency'].max()+1]
# summary_df['freqnency_label1']=pd.cut(summary_df['freqnency'],cutlevel,labels=('low','high'))


#問題: 依中位數或平均數切monetary高低    
summary_df['monetary'].describe()
summary_df['monetary']=summary_df['monetary'].astype(int)

summary_df['monetary_label'] = summary_df['monetary'].apply(
    lambda x: 'High' if x > summary_df['monetary'].mean() else 'Low'
)


#依FR高低值將顧客分群命名
import numpy as np

summary_df.info()
summary_df['recency'].describe()
summary_df['recency']=summary_df['recency'].astype(int)

summary_df['recency_label'] = summary_df['recency'].apply(
    lambda x: 'High' if x > summary_df['recency'].mean() else 'Low'
)
summary_df['freqnency'].describe()
summary_df['freqnency']=summary_df['freqnency'].astype(int)

summary_df['freqnency_label'] = summary_df['freqnency'].apply(
    lambda x: 'High' if x > summary_df['freqnency'].mean() else 'Low'
)
summary_df['customelabel'] = np.where( (summary_df['freqnency_label'] =='High') & (summary_df['recency_label']=='Low'), '常貴客',
                        np.where( (summary_df['freqnency_label'] =='High') & ( summary_df['recency_label']=='High'), '先前客',
                        np.where( (summary_df['freqnency_label'] =='Low') & ( summary_df['recency_label']=='Low'), '新顧客',
                                   '流失客'  )))

print(summary_df)
summary_df.to_excel("new_summary_df.xlsx")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
# 計算各個分類的數量
customer_labels_count = summary_df['customelabel'].value_counts()

# 繪製圓餅圖
plt.figure(figsize=(8, 6))
plt.pie(customer_labels_count, labels=customer_labels_count.index, autopct='%1.1f%%', startangle=140)
plt.title('顧客分類比例')
plt.axis('equal')
plt.show()

print(customer_labels_count)

# 按照 customelabel 分組並計算平均值
grouped_customers = summary_df.groupby('customelabel').agg({
    'monetary': 'mean',
    'freqnency': 'mean'
})

# 顯示結果
print(grouped_customers)

#-----------------------------------------------------------------------------


# 選擇常貴客群的資料
high_value_cluster = summary_df[summary_df['customelabel'] == '常貴客']

# 將常貴客群的顧客 ID 提取出來
high_value_clusters = high_value_cluster.index.tolist()

# 選擇常貴客群的交易資料
high_value_cluster_transactions = df[df['顧客姓名'].isin(high_value_clusters)]

# 計算每個商品的總銷售量
top_products = high_value_cluster_transactions.groupby('產品子類別')['訂單數量'].sum().reset_index()

# 找出前五項主力商品
top_5_products = top_products.sort_values(by='訂單數量', ascending=False).head(5)
last_5_products = top_products.sort_values(by='訂單數量', ascending=False).tail(5)
# 顯示結果
print("常貴客群前五項熱賣商品:")
print(top_5_products)
print("常貴客群銷售量差的前五項商品:")
print(last_5_products)
# 選擇流失客群的資料
high_value_cluster = summary_df[summary_df['customelabel'] == '流失客']

# 將流失客群的顧客 ID 提取出來
high_value_clusters = high_value_cluster.index.tolist()

# 選擇流失客群的交易資料
high_value_cluster_transactions = df[df['顧客姓名'].isin(high_value_clusters)]

# 計算每個商品的總銷售量
top_products = high_value_cluster_transactions.groupby('產品子類別')['訂單數量'].sum().reset_index()

# 找出前五項主力商品
top_5_products = top_products.sort_values(by='訂單數量', ascending=False).head(5)
last_5_products = top_products.sort_values(by='訂單數量', ascending=False).tail(5)
# 顯示結果
print("流失客群前五項熱賣商品:")
print(top_5_products)
print("流失客群銷售量差的前五項商品:")
print(last_5_products)
# 選擇先前客群的資料
high_value_cluster = summary_df[summary_df['customelabel'] == '先前客']

# 將先前客群的顧客 ID 提取出來
high_value_clusters = high_value_cluster.index.tolist()

# 選擇先前客群的交易資料
high_value_cluster_transactions = df[df['顧客姓名'].isin(high_value_clusters)]

# 計算每個商品的總銷售量
top_products = high_value_cluster_transactions.groupby('產品子類別')['訂單數量'].sum().reset_index()

# 找出前五項主力商品
top_5_products = top_products.sort_values(by='訂單數量', ascending=False).head(5)
last_5_products = top_products.sort_values(by='訂單數量', ascending=False).tail(5)
# 顯示結果
print("先前客群前五項熱賣商品:")
print(top_5_products)
print("先前客群銷售量差的前五項商品:")
print(last_5_products)
# 選擇新顧客群的資料
high_value_cluster = summary_df[summary_df['customelabel'] == '新顧客']

# 將常新顧群的顧客 ID 提取出來
high_value_clusters = high_value_cluster.index.tolist()

# 選擇新顧客群的交易資料
high_value_cluster_transactions = df[df['顧客姓名'].isin(high_value_clusters)]

# 計算每個商品的總銷售量
top_products = high_value_cluster_transactions.groupby('產品子類別')['訂單數量'].sum().reset_index()

# 找出前五項主力商品
top_5_products = top_products.sort_values(by='訂單數量', ascending=False).head(5)
last_5_products = top_products.sort_values(by='訂單數量', ascending=False).tail(5)
# 顯示結果
print("新顧客群前五項熱賣商品:")
print(top_5_products)
print("新顧客群銷售量差的前五項商品:")
print(last_5_products)
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# 獲取各個點的資料子集合
high_rec_high_freq = summary_df[(summary_df['recency_label'] == 'High') & (summary_df['freqnency_label'] == 'High')]
low_rec_high_freq = summary_df[(summary_df['recency_label'] == 'Low') & (summary_df['freqnency_label'] == 'High')]
high_rec_low_freq = summary_df[(summary_df['recency_label'] == 'High') & (summary_df['freqnency_label'] == 'Low')]
low_rec_low_freq = summary_df[(summary_df['recency_label'] == 'Low') & (summary_df['freqnency_label'] == 'Low')]

# 繪製散佈圖
ax = high_rec_high_freq.plot.scatter(x='freqnency', y='recency', color='red', label='先前客')
low_rec_high_freq.plot.scatter(ax=ax, x='freqnency', y='recency', color='blue', label='常貴客')
high_rec_low_freq.plot.scatter(ax=ax, x='freqnency', y='recency', color='orange', label='流失客')
low_rec_low_freq.plot.scatter(ax=ax, x='freqnency', y='recency', color='green', label='新顧客', grid=True)

# 圖例設定
plt.legend(loc='best')

# 設置座標軸標籤和標題
ax.set_ylabel('Recency')
ax.set_xlabel('Frequency')
ax.set_title('Segments by Recency and Frequency')

plt.show()
#------------------------------------------------------------------------------
#顧客分群繪圖泡泡圖(另一種圖形)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

summary_df_agg = summary_df.groupby(['recency_label','freqnency_label']).agg(
    {'recency':'mean',
    'freqnency':'mean',
    'monetary':'count', #人數
    'customelabel': max
    }
)

for i in range(len(summary_df_agg)):
    row = summary_df_agg.iloc[i]
    plt.scatter(y=row['recency'], x=row['freqnency'], s=row['monetary'])
    
    # 添加標籤
    plt.text(row['freqnency'], row['recency'], f"{row['customelabel']}: {row['monetary']}", fontsize=8, ha='right', va='bottom')

plt.title('泡泡圖')
plt.xlabel('frequency')
plt.ylabel('recency')
plt.ylim(0, 300)
plt.xlim(0, 7)
#-----------------------------------------------------------------------------
df = pd.read_excel('new_summary_df.xlsx').set_index("顧客姓名")

rfmdf=df[['monetary', 'freqnency', 'recency']]

rfmdf

scale=StandardScaler()

normalized_df=pd.DataFrame(scale.fit_transform(rfmdf))

# Customer Segmentation via K-Means Clustering
#先分四群看結果
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4).fit(normalized_df)

#各數值歸屬群組標籤以及中心值
kmeans.labels_

kmeans.cluster_centers_

#還原標準化後的值，找出各群中心點，可與上面標準化後的值比較

Customercenter_reverse= pd.DataFrame(scale.inverse_transform(kmeans.cluster_centers_))
Customercenter_reverse.columns=['monetary','freqnency','recency']
Customercenter_reverse

#找出各群標籤以及個數
four_cluster_df = df[['monetary', 'freqnency', 'recency']].copy()
four_cluster_df['Cluster'] = kmeans.labels_
four_cluster_df.groupby('Cluster')['monetary'].count()

#-------------------------------------------------------------------------------
#繪圖scatter-四群中心點與數量(x為frequency, y為monetary未標準化後的數值) 

plt.scatter(
   Customercenter_reverse.loc[:,'freqnency'], 
   Customercenter_reverse.loc[:,'monetary'],
   c= Customercenter_reverse.index,
   s= four_cluster_df.groupby('Cluster')['monetary'].count()*5,
   alpha=0.5
)

#加入各群中心點，以黑色顯示
# plt.scatter(
#    Customercenter_reverse.loc[:,'freqnency'], 
#    Customercenter_reverse.loc[:,'monetary'],
#    c='black',
#    #alpha=0.5 
# )


plt.xlabel("frequency")
plt.ylabel('monetary')
plt.title('scatter: frequency vs. monetary')
plt.xlim(-10, 70)
plt.ylim(-1000,7000)
plt.show()

#驗算上面中心點對不對
four_cluster_df_avg=four_cluster_df.groupby('Cluster').mean()
four_cluster_df_avg #平均數算法
Customercenter_reverse #StandardScaler還原算法
#------------------------------------------------------------------------------
#繪圖 monetary vs. freqnency Clusters
plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 0]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 0]['monetary'],
    c='blue'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 1]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 1]['monetary'],
    c='yellow'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 2]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 2]['monetary'],
    c='orange'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 3]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 3]['monetary'],
    c='green'
)


plt.title('monetary vs. freqnency Clusters--by myself')
plt.xlabel('freqnency')
plt.ylabel('monetary')


plt.grid()
plt.show()


#------------------------------------------------------------------------------
#問題:繪圖 recency vs. freqnency 
plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 0]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 0]['recency'],
    c='blue'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 1]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 1]['recency'],
    c='yellow'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 2]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 2]['recency'],
    c='orange'
)

plt.scatter(
    four_cluster_df[four_cluster_df['Cluster'] == 3]['freqnency'], 
    four_cluster_df[four_cluster_df['Cluster'] == 3]['recency'],
    c='green'
)


plt.title('monetary vs. freqnency Clusters--by myself')
plt.xlabel('freqnency')
plt.ylabel('recency')


plt.grid()
plt.show()


#------------------------------------------------------------------------------

#繪圖 recency vs. monetary Clusters'
plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['monetary'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 0]['recency'],
    c='blue'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['monetary'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 1]['recency'],
    c='red'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['monetary'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 2]['recency'],
    c='orange'
)

plt.scatter(
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['monetary'], 
    four_cluster_df.loc[four_cluster_df['Cluster'] == 3]['recency'],
    c='green'
)

#加入各群中心點，以黑色顯示
plt.scatter(
   Customercenter_reverse.iloc[:,1], 
   Customercenter_reverse.iloc[:,2],
   c='black',
   #alpha=0.5 
)


plt.title('recency vs. monetary Clusters--by myself')
plt.xlabel('monetary')
plt.ylabel('recency')

plt.grid()
plt.show()


#======================================================================================================================
#======================================================================================================================
##Kmeans 最適分群, 根據Silhouette值來決定
from sklearn.metrics import silhouette_score

for n in [2,3,4,5]:
    kmeans = KMeans(n_clusters=n).fit(
        normalized_df)
    silhouette_avg = silhouette_score(
        normalized_df, 
        kmeans.labels_
    )
    #print(silhouette_avg)  
    print(f'Silhouette Score for {n} Clusters: {silhouette_avg:0.4f}')

#根據silhouette值，找出最適合為3群

kmeans = KMeans(n_clusters=3).fit(normalized_df)

five_cluster_df = df[['monetary', 'freqnency', 'recency']].copy()
five_cluster_df['Cluster'] = kmeans.labels_

#各群中心點
five_cluster_center=kmeans.cluster_centers_ 

#還原標準化後的值，找出各群中心點，可與上面標準化後的值比較
five_cluster_center_reverse= pd.DataFrame(scale.inverse_transform(five_cluster_center))
five_cluster_center_reverse.columns=['monetary','freqnency','recency']

plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 0]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 0]['monetary'],
    c='blue', 
    label='Cluster 0'
)

plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 1]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 1]['monetary'],
    c='yellow', 
    label='Cluster 1'
)

plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 2]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 2]['monetary'],
    c='orange', 
    label='Cluster 2'
)
# 中心點
plt.scatter(
    five_cluster_center_reverse['freqnency'], 
    five_cluster_center_reverse['monetary'], 
    marker='*', 
    c='red', 
    s=300, 
    label='中心點'
)
plt.title('Monetary vs. Frequency Clusters')
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.legend()
plt.grid()
plt.show()
#------------------------------------------------------------------------------
plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 0]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 0]['recency'],
    c='blue', 
    label='Cluster 0'
)

plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 1]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 1]['recency'],
    c='yellow', 
    label='Cluster 1'
)

plt.scatter(
    five_cluster_df[five_cluster_df['Cluster'] == 2]['freqnency'], 
    five_cluster_df[five_cluster_df['Cluster'] == 2]['recency'],
    c='orange', 
    label='Cluster 2'
)
# 中心點
plt.scatter(
    five_cluster_center_reverse['freqnency'], 
    five_cluster_center_reverse['recency'], 
    marker='*', 
    c='red', 
    s=300, 
    label='中心點'
)
plt.title('recency vs. Frequency Clusters')
plt.xlabel('Frequency')
plt.ylabel('recency')
plt.legend()
plt.grid()
plt.show()
#------------------------------------------------------------------------------
  
'''#分群後可針對各群進行產品推薦
df_original = pd.read_excel('全國訂單明細.xlsx')
#針對分群的某一群整體推薦商品
#看哪一群最有價值，結果發現第1群，因此找出第1群top5產品名稱
five_cluster_center

 

#第3群的顧客編號
five_cluster_df.loc[five_cluster_df['Cluster'] == 3].index

#第3群名單
df_original[df_original['顧客姓名'].isin(five_cluster_df.loc[five_cluster_df['Cluster'] == 3].index)]


#第3群銷售商品前五名產品
top5=pd.DataFrame(
    df_original[
        df_original['顧客姓名'].isin(
            five_cluster_df.loc[five_cluster_df['Cluster'] == 3].index)
    ].groupby('訂單號')['產品類別'].count().sort_values(ascending=False).head()
)

print(top5)
#-----------------------------------------------------------------------
second_cluster=pd.DataFrame(
    df_original.loc[
        df_original['顧客姓名'].isin(
            five_cluster_df[five_cluster_df['Cluster'] ==3].index
        )
    ]
)
print(five_cluster_df['Cluster'] == 3)
#將每位顧客購買的每位產品轉置成一列
second_cluster_item_matrix =second_cluster.pivot_table(
     index='訂單號', 
     columns='顧客姓名', 
     values='訂單數量',
     aggfunc='sum'
 )

second_cluster_item_matrix


recom_list = [] #存放某顧客購買過的商品
second_cluster_cliendid=[] #存放某顧客編號

for i in second_cluster_item_matrix.columns:
    aa = second_cluster_item_matrix[i].rank().sort_values(ascending = False)
    aa.dropna(inplace=True)
    recom_list.append('、'.join((aa.index).astype(str)))
    second_cluster_cliendid.append(i)

second_cluster_cliendid
recom_list



#製作產品清單,將second_cluster_cliendid,recom_list申連起來，形成一個DataFrame, 並將結果存為seccustomer_recomm1.csv
second_cluster_cliendid=pd.DataFrame(second_cluster_cliendid)
recom_list=pd.DataFrame(recom_list)
second_cluster_recomm=pd.concat([second_cluster_cliendid,recom_list],axis=1)
second_cluster_recomm.columns=['顧客姓名','產品類別']


print(second_cluster_recomm.head(5))'''