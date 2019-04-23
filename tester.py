test_data = []
for i in range(10):
    test_frame = []
    for j in range(10):
        if j == i:
            test_frame.append(1)
        else:
            test_frame.append(0)
    test_frame.append(i)
    test_data.append(test_frame)

print(test_data)

