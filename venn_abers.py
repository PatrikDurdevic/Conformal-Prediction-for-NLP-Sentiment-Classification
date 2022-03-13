from sklearn.isotonic import IsotonicRegression
import numpy as np

def venn_abers(S_X, y, s_x_test):
	data = np.array(list(zip(*(S_X, y))))
	data_0 = np.append(data, [[s_x_test, 0]], axis=0)
	data_1 = np.append(data, [[s_x_test, 1]], axis=0)
	g_0 = IsotonicRegression()
	g_1 = IsotonicRegression()
	g_0.fit(data_0[:,0], data_0[:,1])
	g_1.fit(data_1[:,0], data_1[:,1])
	return g_0.predict([s_x_test]), g_1.predict([s_x_test])