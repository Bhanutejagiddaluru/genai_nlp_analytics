# Start Ray: ray start --head (or ray.init() in Python)
import ray

@ray.remote
class Scorer:
    def __init__(self):
        self.count = 0
    def score(self, x: float) -> float:
        self.count += 1
        return x * 1.234

def demo():
    ray.init(ignore_reinit_error=True)
    actor = Scorer.remote()
    futs = [actor.score.remote(i) for i in range(10)]
    print(ray.get(futs))

if __name__ == '__main__':
    demo()
