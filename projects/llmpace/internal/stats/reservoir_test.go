package stats

import "testing"

func TestReservoir_ExactBelowCapacity(t *testing.T) {
	r := NewReservoir(10)
	for i := 0; i < 5; i++ {
		r.Add(float64(i))
	}
	if got := len(r.Values()); got != 5 {
		t.Fatalf("expected 5 exact values, got %d", got)
	}
	if r.Count() != 5 {
		t.Fatalf("expected count 5, got %d", r.Count())
	}
}

func TestReservoir_BoundedAboveCapacity(t *testing.T) {
	const cap = 100
	r := NewReservoir(cap)
	const total = 100_000
	for i := 0; i < total; i++ {
		r.Add(float64(i))
	}
	if got := len(r.Values()); got != cap {
		t.Fatalf("expected reservoir bounded to %d, got %d", cap, got)
	}
	if r.Count() != total {
		t.Fatalf("expected count %d, got %d", total, r.Count())
	}
}

// TestReservoir_ApproximatesMedian is a sanity check, not a precise
// statistical test: a uniform sample of a uniform population should have a
// median in the right ballpark, not a wildly biased one (which would
// indicate a bug in Algorithm R's selection logic, e.g. always overwriting
// index 0).
func TestReservoir_ApproximatesMedian(t *testing.T) {
	r := NewReservoir(2000)
	const total = 200_000
	for i := 0; i < total; i++ {
		r.Add(float64(i))
	}
	median := Percentile(r.Values(), 50)
	wantApprox := float64(total) / 2
	tolerance := wantApprox * 0.1
	if median < wantApprox-tolerance || median > wantApprox+tolerance {
		t.Fatalf("reservoir median %.0f too far from expected ~%.0f (population 0..%d)", median, wantApprox, total-1)
	}
}
