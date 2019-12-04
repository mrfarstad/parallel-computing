def prettify(li):
    return [float("{0:.2f}".format(i)) for i in li]


def calcSpeedup(new, old):
    res = []
    for i in range(0, len(old)):
        res.append(old[i] / new[i])
    return prettify(res)


serial = [1.666, 3 * 60 + 52.363, 7 * 60 + 42.617, 15 * 60 + 23.825]
print("Serial: ", prettify(serial))
print("")

basic = [1.574, 2.498, 3.420, 5.284]
print("Basic: ", prettify(basic))
print("Speedup vs serial: ", calcSpeedup(basic, serial))
print("")

shared = [1.558, 3.045, 4.452, 7.354]
print("Shared: ", prettify(shared))
print("Speedup vs serial: ", calcSpeedup(shared, serial))
print("Speedup vs basic: ", calcSpeedup(shared, basic))
print("")

cooperative = [1.558, 3.166, 4.661, 7.783]
print("Cooperative: ", prettify(cooperative))
print("Speedup vs serial: ", calcSpeedup(cooperative, serial))
print("Speedup vs basic: ", calcSpeedup(cooperative, basic))
print("Speedup vs shared: ", calcSpeedup(cooperative, shared))
