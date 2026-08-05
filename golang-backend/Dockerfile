FROM golang:1.25-alpine AS builder

WORKDIR /src

# Bog'liqliklarni alohida keshlaymiz.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build \
        -trimpath -ldflags="-s -w" \
        -o /out/server ./cmd/server

# ---------------------------------------------------------------- runtime
FROM alpine:3.20

RUN apk add --no-cache ca-certificates wget \
    && adduser -D -u 10001 appuser

WORKDIR /app
COPY --from=builder /out/server /app/server
COPY --from=builder /src/web /app/web

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD wget -qO- http://127.0.0.1:8080/health >/dev/null || exit 1

ENTRYPOINT ["/app/server"]
