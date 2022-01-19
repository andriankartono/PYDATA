import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

mse_list = []
mae_list = []
r2_list = []
alpha_list = []
gamma_list = []
y_train_list = []
y_pred_train_list = []
y_test_list = []
y_pred_test_list = []

df=pd.read_csv("nitride_compounds.csv")

x=df.iloc[:,2:-4]
y=df.iloc[:, -3]

train_sizes= np.arange(0.1, 1, 0.1)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8)

for train_size in train_sizes:
    x_train_frac, x_unused_frac, y_train_frac, y_unused_frac = train_test_split(x_train, y_train, shuffle=True, train_size=train_size)
    
    krr=KernelRidge(kernel='rbf')

    y_train_list.append(y_train_frac)
    y_test_list.append(y_test)

    krr_params = {'alpha': np.arange(0.01, 0.11, 0.01), 'gamma': np.arange(0, 0.3, 0.025)}

    hyperparam = GridSearchCV(krr, krr_params, scoring='neg_mean_squared_error')
    hyperparam.fit(x_train_frac, y_train_frac)

    y_pred_train= hyperparam.predict(x_train)
    y_pred_test= hyperparam.predict(x_test)

    y_pred_train_list.append(y_pred_train)
    y_pred_test_list.append(y_pred_test)

    alpha_list.append(hyperparam.best_params_.get('alpha'))
    gamma_list.append(hyperparam.best_params_.get('gamma'))

    mse_list.append(mean_squared_error(y_test, y_pred_test))
    mae_list.append(mean_absolute_error(y_test, y_pred_test))
    r2_list.append(r2_score(y_test, y_pred_test))

max_value= max(r2_list)
max_index= r2_list.index(max_value)

train_size = train_sizes[max_index]
alpha = alpha_list[max_index]
gamma = gamma_list[max_index]
r2 = r2_list[max_index]
mae = mae_list[max_index]
y_train_plot = y_train_list[max_index]
y_test_plot = y_test_list[max_index]
y_pred_train = y_pred_train_list[max_index]
y_pred_test = y_pred_test_list[max_index]

x=np.linspace(0,6,5000)
fig, axs= plt.subplots(1,2)

axs[0].set_title('learning curves')
axs[0].set_xlabel('fraction of training data used')

axs[0].plot(train_sizes, mse_list, color='blue')
axs[0].set_ylabel('MSE')

#invisible x axis and independent y axis
ax_twin=axs[0].twinx()
ax_twin.plot(train_sizes, r2_list, color='darkorange')
ax_twin.set_ylabel('$R^2$')

# Plot the training data
axs[1].scatter(y_train,y_pred_train, s=5)
#print(y_test_plot)
#print(y_pred_test)
# Plot the test data
axs[1].scatter(y_test_plot,y_pred_test, s=5)
axs[1].legend(['training data', 'test data'])
# Plot the y=x graph
axs[1].plot(x, x, 'k--')
axs[1].set_xlabel('Calculated gap')
axs[1].set_ylabel('Model gap')
axs[1].set_xlim(0, 6)
axs[1].set_ylim(0, 6)
axs[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
axs[1].set_yticks([0, 1, 2, 3, 4, 5, 6])
axs[1].set_title(f'Model $R^2$: {r2:.3f}, MAE: {mae:.3f}')

# Needed for a clear output
fig.tight_layout()

# Save figure
plt.savefig('Aufgabe7_3.png')
plt.show()