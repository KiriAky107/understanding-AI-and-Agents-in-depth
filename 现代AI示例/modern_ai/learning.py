"""固定数据、训练/验证/测试分离的预测实验。"""
from time import perf_counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def linear_descent(x, y, steps=1200, rate=0.03):
    # 两个梯度共享同一份旧误差，避免更新顺序改变算法。
    x, y = np.asarray(x, float), np.asarray(y, float)
    w, b, history = 0.0, 0.0, []
    for _ in range(steps):
        error = w * x + b - y
        history.append(float(np.mean(error ** 2) / 2))
        w -= rate * np.mean(error * x)
        b -= rate * np.mean(error)
    return {'w': float(w), 'b': float(b), 'loss': float(np.mean((w*x+b-y)**2)/2), 'history': history}


def metrics(y, prediction):
    # labels 固定轴顺序，即使一批数据没有某个类也保持二维矩阵。
    return {'accuracy': float(np.mean(prediction == y)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'confusion': confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def experiment():
    # 原始规则、噪声和随机种子随输出保存。
    rng = np.random.default_rng(17)
    x = rng.normal(size=(240, 2))
    noise = rng.normal(0, 0.3, 240)
    y = (x[:, 0] + 0.6*x[:, 1] + noise > 0).astype(int)
    ids = np.arange(len(y))
    development, test = train_test_split(ids, test_size=0.2, random_state=17, stratify=y)
    train, validation = train_test_split(development, test_size=0.25, random_state=17,
                                        stratify=y[development])
    assert not (set(train) & set(test) or set(validation) & set(test))
    candidates = {'logistic': LogisticRegression(random_state=17),
                  'neighbors': KNeighborsClassifier(5),
                  'tree': DecisionTreeClassifier(max_depth=3, random_state=17)}
    fitted, results = {}, {}
    for name, model in candidates.items():
        # 缩放的 fit 只能看到训练样本。
        pipeline = make_pipeline(StandardScaler(), model)
        started = perf_counter()
        pipeline.fit(x[train], y[train])
        fit_ms = (perf_counter()-started)*1000
        fitted[name] = pipeline
        results[name] = {'validation': metrics(y[validation], pipeline.predict(x[validation])),
                         'fit_ms': fit_ms}
    # 先在验证集锁定阈值，再接触测试标签。
    probabilities = fitted['logistic'].predict_proba(x[validation])[:, 1]
    thresholds = {str(t): float(f1_score(y[validation], probabilities >= t))
                  for t in [0.4, 0.5, 0.6]}
    threshold = float(max(thresholds, key=thresholds.get))
    selected = max(results, key=lambda k: results[k]['validation']['f1'])
    for name, pipeline in fitted.items():
        started = perf_counter()
        prediction = pipeline.predict(x[test])
        results[name]['predict_ms'] = (perf_counter()-started)*1000
        results[name]['test'] = metrics(y[test], prediction)
    tuned = fitted['logistic'].predict_proba(x[test])[:, 1] >= threshold
    baseline = int(np.bincount(y[train]).argmax())
    # 错误列表来自验证侧，避免用测试错误继续调模型。
    errors = []
    for local, original in enumerate(validation):
        if int(probabilities[local] >= 0.5) != y[original]:
            errors.append({'id': int(original), 'x': x[original].tolist(),
                           'noise': float(noise[original]), 'label': int(y[original]),
                           'probability': float(probabilities[local])})
    simple_x = np.arange(1., 5.)
    descent = linear_descent(simple_x, 2*simple_x+1)
    exact = LinearRegression().fit(simple_x[:, None], 2*simple_x+1)
    clusters = KMeans(n_clusters=2, random_state=17, n_init=10).fit(x[train])
    return {'seed': 17, 'split': [len(train), len(validation), len(test)],
            'models': results, 'selected_by_validation': selected,
            'thresholds': thresholds, 'threshold': threshold,
            'tuned_logistic_test': metrics(y[test], tuned),
            'baseline_test': metrics(y[test], np.full(len(test), baseline)),
            'validation_errors': errors,
            'regression': {'descent': {k:v for k,v in descent.items() if k != 'history'},
                           'library_w': float(exact.coef_[0]), 'library_b': float(exact.intercept_)},
            'cluster_centers': clusters.cluster_centers_.tolist()}
