package stats

import "math/rand"

// Reservoir implements Algorithm R (Vitter, 1985): a uniformly-random sample
// of up to `cap` values drawn from a stream of arbitrary length, using O(cap)
// memory regardless of how many values are added. For the common case of a
// short local run (total additions <= cap), the reservoir holds every value
// added — this only becomes an approximation once a run's request count
// exceeds cap, which is exactly the long/high-QPS soak-test case a full
// in-memory slice would not bound.
type Reservoir struct {
	cap    int
	values []float64
	count  int
	rng    *rand.Rand
}

func NewReservoir(cap int) *Reservoir {
	return &Reservoir{cap: cap, rng: rand.New(rand.NewSource(1))}
}

func (r *Reservoir) Add(v float64) {
	r.count++
	if len(r.values) < r.cap {
		r.values = append(r.values, v)
		return
	}
	j := r.rng.Intn(r.count)
	if j < r.cap {
		r.values[j] = v
	}
}

// Values returns the current sample. Exact (contains every added value) as
// long as the total number of Add calls has not exceeded the reservoir's
// capacity; a uniform random sample of that capacity otherwise. The caller
// must not mutate the returned slice's order-sensitive assumptions before
// sorting it — Percentile sorts its own copy.
func (r *Reservoir) Values() []float64 {
	return r.values
}

// Count returns the total number of values ever added, which may exceed
// len(Values()) once the reservoir capacity has been exceeded.
func (r *Reservoir) Count() int {
	return r.count
}
