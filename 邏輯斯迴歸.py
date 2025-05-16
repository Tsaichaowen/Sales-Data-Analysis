import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 讀取資料

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
customer_data = df.drop(result.index)
'''# 假設 '訂單日期' 是日期欄位，將其轉換為 datetime 格式
customer_data['訂單日期'] = pd.to_datetime(customer_data['訂單日期'])

# 找出2012/12/31往前365天的日期範圍
end_date = pd.to_datetime('2012-12-31')
start_date = end_date - pd.Timedelta(days=365)

# 按照顧客姓名分組，找出在指定日期範圍內的購買日期
purchased_within_365_days = customer_data[customer_data['訂單日期'].between(start_date, end_date)]['顧客姓名'].unique()

# 判斷哪些顧客在該日期範圍內未購買產品，即為流失顧客
churned_customers = customer_data[~customer_data['顧客姓名'].isin(purchased_within_365_days)]['顧客姓名'].unique()
customer_data['churn_rate'] = customer_data['顧客姓名'].isin(churned_customers).astype(int)

# 使用get_dummies進行獨熱編碼
columns_to_encode = ['產品類別','產品子類別','區域'
    
]

encoded_df = pd.get_dummies(customer_data[columns_to_encode]) #直接把columns_to_encode的欄位進行one-hot encoding
encoded_df 

#把one-hot encoding後的欄位儲存
categorical_features=list(encoded_df.columns)

#把one-hot encoding後的資料與原先資料合併
df = pd.concat([customer_data, encoded_df], axis=1)
print(df.head())'''
#----------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 將銷售額分為高、中、低三個類別（示例中假設分界點為 300、1000）
bins = [0, 300, 1000, float('inf')]
labels = ['低', '中', '高']
df['銷售額分類'] = pd.cut(df['銷售額'], bins=bins, labels=labels)

# 假設特徵包含訂單數量、折扣點、利潤額、運輸成本等
X = df[['訂單數量', '折扣點', '利潤額', '運輸成本']]  # 選擇特徵
y = df['銷售額分類']  # 或者是銷售額本身作為目標變數

# 特徵編碼（One-Hot Encoding）
encoded_X = pd.get_dummies(X)

# 將目標變數進行編碼（如果是分類的銷售額分類）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 訓練集和測試集的分割
X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=42)

# 初始化邏輯斯迴歸模型
model = LogisticRegression()

# 在訓練集上訓練模型
model.fit(X_train, y_train)

# 在測試集上做預測
y_pred = model.predict(X_test)

# 模型評估
# 模型評估
class_report_dict = classification_report(y_test, y_pred, output_dict=True)
f1_scores = class_report_dict['weighted avg']['f1-score']
print(f"F1 分數：{f1_scores}")
accuracy = accuracy_score(y_test, y_pred)
print(f"準確率: {accuracy}")

class_report = classification_report(y_test, y_pred)
print("分類報告：")
print(class_report)

# 計算AUC
y_pred_proba = model.predict_proba(X_test)
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"AUC: {auc_score}")

# 繪製AUC曲線
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
