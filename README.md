# Self-Organizing-Maps
- 在論文中, 作者利用自組特徵映射網路(SOM)將地面資料與地下水水位變化資料依時間與空間的關連性進
行聚類，並從聚類結果與其拓樸圖來探討關聯特性, 以及其他....
## 簡介
- 自組特徵映射網路(Self-Organizing Map, SOM)於 1982 年首先由Kohonen 提出，屬於前饋式、非監督式與競爭式神經網路，其演算法以特徵映射方式將任意維度的輸入向量映射至較低維度的特徵映射圖(拓樸層)
- 依據輸入向量與神經元之相似度彼此競
爭，距離最近者為優勝神經元可獲得調整連結權重的機會，SOM 的
神經元間具有鄰近關係的特性，故當優勝神經元被調整時，其鄰近神
經元也會進行連結權重的調整，最後拓樸層的神經元會依輸入向量的
「特徵」以有意義的「拓樸結構(topology structure)」表現在其權重值
上，也可稱為拓樸圖(topology)。
- 步驟簡述如下:
  1. Select random input
  2. Compute winner neuron
  3. Update neurons
  4. Repeat for all input data
  5. Classify input data
## 關於論文中的使用方法
- 將各年份之地下水位取月平均, 將資料丟入具有16個神經元的SOM去做訓練。
- 本計畫先利用 SOM 進行聚類分析後，將所有神經元(分類)不同分層觀測井分開，再以克利金法(Kriging Method)
推估各層月平均相對地下水水位之空間分布。
### 克利金法(Kriging Method)
- https://www.youtube.com/watch?v=J-IB4_QL7Oc
- https://www.supergeotek.com/tw/manuals/SSA/topic23.htm
