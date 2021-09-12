import pandas as pd
X = pd.DataFrame({'age': [25,32,47,51,62],
                    'heght':[182,176,174,168,181],
                    'weight':[75,71,78,72,86]})
Y = pd.Series([20,32,45,55,61], name='income')

# Lý tưởng nhất, tôi muốn tìm một thuật toán có thể tự quyết định xem bất kỳ đặc điểm nào đã cho của X 
# có mang một giá trị dự đoán nào đó về y hay không.

