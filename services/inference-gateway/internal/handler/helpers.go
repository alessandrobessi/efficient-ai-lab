package handler

import (
	"net/http"
	"strconv"

	"github.com/alessandrobessi/efficient-ai-lab/services/inference-gateway/internal/middleware"
)

func requestID(r *http.Request) string {
	return middleware.RequestIDFromContext(r.Context())
}

func itoa(n int) string {
	return strconv.Itoa(n)
}

func ftoa(f float64) string {
	return strconv.FormatFloat(f, 'g', -1, 64)
}
