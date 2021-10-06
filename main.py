from sklearn.tree import RandomForestRegressor
from sklearn.metrics import mse_score
from module_a_dev import polynom_3
from useful_package/module_b_dev import hyperbola

x = np.random.uniform(0, 1)
y = polynom_3(x)

model = RandomForestRegressor()
model.fit(x, y)

prediction = model.predict(x)
print(mse_score(y, prediction))
