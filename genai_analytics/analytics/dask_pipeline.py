from dask import delayed, compute

@delayed
def step1(x): return x + 1

@delayed
def step2(x): return x * 2

def run(n=10):
    tasks = [step2(step1(i)) for i in range(n)]
    return compute(*tasks)

if __name__ == '__main__':
    print(run())
