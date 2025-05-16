import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 資料預處理
data = pd.read_excel('全國訂單明細.xlsx')
df = pd.DataFrame(data)
# 顯示資料的摘要
#print(df.head())  # 顯示前幾列
#print(df.info())  # 顯示資料型別和缺失值
#print(df.describe())  # 顯示摘要統計
# 找出訂單號相同但顧客資料不一樣的資料

# 使用 groupby 和 filter 找出訂單號相同但顧客姓名不同的資料
result = df.groupby('訂單號').filter(lambda x: x['顧客姓名'].nunique() > 1)
#print(result.head())
# 刪除這些資料
df = df.drop(result.index)
## 需要根據你的預測目標選擇特徵和目標變數
X = data[['訂單數量', '折扣點', '利潤額', '運輸成本']]  # 選擇特徵
y = data['銷售額']  # 預測目標

# 將資料分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化線性回歸模型
model = LinearRegression()

# 在訓練集上訓練模型
model.fit(X_train, y_train)

# 在測試集上做預測
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

# 評估模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#print(f"均方誤差 (MSE): {mse}")
print(f"決定係數 (R^2): {r2}")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')
#顯示負號
matplotlib.rcParams['axes.unicode_minus']=False
# 繪製散點圖展示實際值與預測值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.title('線性回歸模型的實際值 vs 預測值')
plt.xlabel('銷售額實際值')
plt.ylabel('銷售額預測值')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 繪製實際值與預測值相等的對角線
plt.show()