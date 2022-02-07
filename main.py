# importing a user-built import file
from res.imports import *

# fetching datasets
x = dc.Data().get_data().iloc[:, 1:-1].values
y = dc.Data().get_data().iloc[:, -1].values

# applying regression model Random Forest regression
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)

# printing the value of a employee with position lvl between 6 to 7
pr.Printer().print_to_console("self", regressor.predict([[6.5]]))

# visualizing Random Forest reg result
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.title('truth or bluff (Random forest)')
plt.xlabel('position lvl')
plt.ylabel('salary')
plt.show()
