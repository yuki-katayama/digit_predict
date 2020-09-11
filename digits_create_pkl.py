from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

from sklearn import datasets
digits = datasets.load_digits()

for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.axis("off")
    plt.title(str(digits.target[i]))
    plt.imshow(digits.images[i], cmap="gray")

plt.show()

d0 = digits.images[0]
plt.imshow(d0, cmap="gray")
plt.show()

print(d0)

d = digits.images
d = d.reshape((-1, 64))
print(len(d[0]))
print(len(d))
print(d[0])
# １行64個 即ち画像一枚
# 合計1797枚ある


# データを読み込む
digits = datasets.load_digits()
x = digits.images
y = digits.target
x = x.reshape((-1, 64))  # 二次元配列を一次元配列に変換
# データを学習用とテスト用に分割する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# データを学習
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

# 予測して精度を確認する
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


# 学習済みデータを保存
joblib.dump(clf, 'digits.pkl')


def predict_digit(filename):
    # 学習済みデータを読み込む
    clf = joblib.load("digits.pkl")

    # 自分で用意した手書きの画像ファイルを読み込む
    my_img = cv2.imread(filename)

    # 画像データを学習済みデータに合わせる
    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    my_img = cv2.resize(my_img, (8, 8))

    # my_img = cv2.bitwise_not(my_img) ←これでも反転d
    my_img = 15 - my_img // 16  # 白黒反転する
    # 二次元を一次元に変換
    my_img = my_img.reshape((-1, 64))
    # データ予測する
    res = clf.predict(my_img)
    return res[0]


# 画像ファイルを指定して実行
n = predict_digit("my2.png")
print("my2.png = " + str(n))
n = predict_digit("my4.png")
print("my4.png = " + str(n))
n = predict_digit("my9.png")  # 上手に判定できない(モデルが悪い)
print("my9.png = " + str(n))
