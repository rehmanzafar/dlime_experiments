import numpy as np

data_dict=[[2.0,3.43,4.37],[2.49,4.28,4.83],[2.58,4.36,4.48],[2.66,4.45,5.95],
[2.82,3.66,4.51],[3.03,4.37,5.07],[3.27,4.54,4.57],[3.41,3.94,5.35],
[3.53,4.32,5.41],[3.53,4.6,6.8],[3.61,4.25,5.21],[3.61,4.78,5.47],
[3.72,5.44,5.88],[3.87,4.96,4.52],[4.13,5.29,6.6],[4.25,5.97,5.48],
[4.61,4.9,5.11],[4.73,4.4,6.78],[4.97,4.25,5.0],[4.98,5.27,6.79],
[5.08,3.51,4.69],[5.15,3.58,4.2],[5.67,2.27,4.65],[5.67,3.81,5.75],
[5.94,2.34,4.12],[6.06,3.16,4.36],[6.09,3.19,4.02],[6.43,3.42,4.18],
[6.56,2.7,4.03],[6.79,3.46,4.81]]

y = [2,2,2,2,2,2,2,2,2,1,1,1,1,2,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3]

print(str(len(data_dict)))
print(str(len(y)))

c1 = [4, 5, 6]
c2 = [3, 4, 5]
c3 = [6, 3, 5]

predicted_labels = []


for x in data_dict:
    #x = [2.0,3.43,4.37]
    g_x_1 = np.sqrt((c1[0] - x[0])**2 + (c1[1] - x[1])**2  + (c1[2] - x[2])**2 )
    g_x_2 = np.sqrt((c2[0] - x[0])**2 + (c2[1] - x[1])**2  + (c2[2] - x[2])**2 )
    g_x_3 = np.sqrt((c3[0] - x[0])**2 + (c3[1] - x[1])**2  + (c3[2] - x[2])**2 )

    list_dist = [g_x_1, g_x_2, g_x_3]

    index_min = np.argmin(list_dist)

    if index_min == 0:
        predicted_labels.append(1)
    elif index_min == 1:
        predicted_labels.append(2)
    else:
        predicted_labels.append(3)

    print(str(g_x_1))
    print(str(g_x_2))
    print(str(g_x_3))
    print(str(index_min))

correct = set(y) & set(predicted_labels)
print(f"correct with EC {len(correct)}")

predicted_labels = []

for x in data_dict:
    #x = [2.0,3.43,4.37]
    h_x_1 = np.abs(c1[0] - x[0]) + np.abs(c1[1] - x[1])  + np.abs(c1[2] - x[2])
    h_x_2 = np.abs(c2[0] - x[0]) + np.abs(c2[1] - x[1])  + np.abs(c2[2] - x[2])
    h_x_3 = np.abs(c3[0] - x[0]) + np.abs(c3[1] - x[1])  + np.abs(c3[2] - x[2])

    list_dist = [h_x_1, h_x_2, h_x_3]

    index_min = np.argmin(list_dist)

    if index_min == 0:
        predicted_labels.append(1)
    elif index_min == 1:
        predicted_labels.append(2)
    else:
        predicted_labels.append(3)

    print(str(h_x_1))
    print(str(h_x_2))
    print(str(h_x_3))
    print(str(index_min))

correct = set(y) & set(predicted_labels)
print(f"correct with MA {len(correct)}")