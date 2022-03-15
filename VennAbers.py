from sklearn.isotonic import IsotonicRegression
import numpy as np

def venn_abers(data, s_x_test):
	#data = np.array(list(zip(*(S_X, y))))
	data_0 = np.append(data, [[s_x_test, 0]], axis=0)
	data_1 = np.append(data, [[s_x_test, 1]], axis=0)
	g_0 = IsotonicRegression()
	g_1 = IsotonicRegression()
	g_0.fit(data_0[:,0], data_0[:,1])
	g_1.fit(data_1[:,0], data_1[:,1])
	return g_0.predict([s_x_test]), g_1.predict([s_x_test])

'''
Faster implementation
'''
kPrime = None

def non_left_turn(a, b, c):   
	return np.cross((b-a), (c-b)) <= 0

def non_right_turn(a, b, c):
	return np.cross((b-a), (c-b)) >= 0

def slope(a, b):
	return (b[1]-a[1])/(b[0]-a[0])

def not_below(t, p1, p2):
	m = slope(p1, p2)
	b = (p2[0]*p1[1] - p1[0]*p2[1])/(p2[0]-p1[0])
	return t[1] >= t[0]*m+b

# Graham's scan to find GCM from CSD
def find_gcm_1(P):
	global kPrime

	S = []
	S.append(P[-1])
	S.append(P[0])
	for i in range(1, kPrime + 1):
		while len(S) > 1 and non_left_turn(S[-2], S[-1], P[i]):
			S.pop(-1)
		S.append(P[i])
	return S

def find_gcm_2(P):
	global kPrime

	S = []
	S.append(P[kPrime+1])
	S.append(P[kPrime])
	for i in range(kPrime-1, 0-1, -1):
		while len(S)>1 and non_right_turn(S[-2], S[-1], P[i]):
			S.pop(-1)
		S.append(P[i])
	return S

def find_f1_from_gcm(P, S):
	global kPrime

	Sp = S[::-1]
	F1 = np.zeros((kPrime+1,))
	for i in range(1, kPrime + 1):
		F1[i] = slope(Sp[-1], Sp[-2])
		P[i-1] = P[i-2] + P[i] - P[i-1]
		if not_below(P[i-1], Sp[-1], Sp[-2]):
			continue
		Sp.pop(-1)
		while len(Sp) > 1 and non_left_turn(P[i-1], Sp[-1], Sp[-2]):
			Sp.pop(-1)
		Sp.append(P[i-1])
	return F1

def find_f0_from_gcm(P,S):
	global kPrime
	
	Sp = S[::-1]
	F0 = np.zeros((kPrime+1,))
	for i in range(kPrime, 1-1, -1):
		F0[i] = slope(Sp[-1], Sp[-2])
		P[i] = P[i-1] + P[i+1] - P[i]
		if not_below(P[i], Sp[-1], Sp[-2]):
			continue
		Sp.pop(-1)
		while len(Sp) > 1 and non_right_turn(P[i], Sp[-1], Sp[-2]):
			Sp.pop(-1)
		Sp.append(P[i])
	return F0

def prep_data(calibration_points):
	global kPrime
	
	sorted_points = sorted(calibration_points)
	
	xs = np.fromiter((p[0] for p in sorted_points), float)
	ys = np.fromiter((p[1] for p in sorted_points), float)
	unique_points, inverse_points, points_counts = np.unique(xs, return_counts=True, return_inverse=True)
	a = np.zeros(unique_points.shape)
	np.add.at(a, inverse_points, ys)
	
	w = points_counts
	y_csd = np.cumsum(a) # Cumulative sum of labels up until the unique score
	x_csd = np.cumsum(w) # Cumulative sum of number of data points up until the unique score
	kPrime = len(x_csd) # Number of unique scores
	
	return y_csd, x_csd, unique_points

def computeF(x_csd, y_csd):
	global kPrime

	P = {0:np.array((0, 0))}
	P.update({i+1:np.array((k, v)) for i, (k, v) in enumerate(zip(x_csd, y_csd))})
	P[-1] = np.array((-1, -1))
	S = find_gcm_1(P)
	F1 = find_f1_from_gcm(P, S)
	
	P = {0:np.array((0,0))}
	P.update({i+1:np.array((k, v)) for i, (k,v) in enumerate(zip(x_csd, y_csd))})    
	P[kPrime+1] = P[kPrime] + np.array((1.0, 0.0))
	S = find_gcm_2(P)
	F0 = find_f0_from_gcm(P, S)
	
	return F0, F1


def getFVal(F0, F1, unique_points, test_objects):
	pos0 = np.searchsorted(unique_points, test_objects, side='left')
	pos1 = np.searchsorted(unique_points[:-1], test_objects, side='right') + 1
	return F0[pos0], F1[pos1]

def venn_abers_fast(calibration_points, test_objects):
	y_csd, x_csd, unique_points = prep_data(calibration_points)
	F0, F1 = computeF(x_csd, y_csd)
	p0, p1 = getFVal(F0, F1, unique_points, test_objects)
	
	return p0, p1