import time
timestamp = int(time.time())
start = 0
with open('watt.csv', 'r') as rf:
    start = len(rf.readlines())

time.sleep(10)

lines = None
with open('watt.csv', 'r') as rf:
    lines = rf.readlines()
usable_lines = lines[start:]
prev = None
ans = 0
for line in usable_lines:
    t, p = line.split(',')[:2]
    if prev is not None:
        ans += (int(t) - prev) / 1000 * float(p)
    prev = int(t)

print(f"Total energy: {ans}")