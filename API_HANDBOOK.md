# API Documentation

## Authentication

Team endpoints require an `X-API-Key` header. API keys are generated when an admin creates a team.

## Team Endpoints

### `GET /api/games/list`

Returns all games with their ID and start time.

**Response:**

```json
[
  { "id": 1, "start_time": "2025-06-01T12:00:00Z" },
  { "id": 2, "start_time": "2025-06-01T12:05:00Z" }
]
```

### `GET /api/games/{game_id}/key`

Returns the decryption key for a game. Only available after the game's `start_time`.

**Response:**

```json
{
  "decryption_key": "secret123"
}
```

**Errors:**

- `403` — Game has not started yet
- `404` — Game not found

### `PUT /api/games/{game_id}/submissions`

Submit or update values for line items during an active game. Accepts a JSON array of submission items. Last write wins (upsert behavior).

Omitted line items use the game defaults of `charge_price = 0` and
`acceptance_limit = 0`. They still participate in transactions; omitting a line
does not opt the team out of it.

Teams added during the tournament participate in games that start after their
creation time, but not in games already in progress.

**Request body:**

```json
[
  { "index": 1, "charge_price": 50.0, "acceptance_limit": 80.0 },
  { "index": 2, "charge_price": 120.0, "acceptance_limit": 150.0 }
]
```

**Response:**

```json
[
  {
    "game_id": 1,
    "team_id": 3,
    "line_item_index": 1,
    "charge_price": 50.0,
    "acceptance_limit": 80.0,
    "submitted_at": "2025-06-01T12:00:30Z"
  }
]
```

**Errors:**

- `401` — Missing or invalid API key
- `403` — Game has not started yet / Game has already ended
- `403` — Team was created after the game started
- `404` — Game not found
- `422` — A charge price or acceptance limit is negative or non-finite

Both monetary values must be finite and nonnegative.
