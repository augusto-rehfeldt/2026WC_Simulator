# 2026 World Cup Simulator

A comprehensive Python simulator for the 2026 FIFA World Cup that models matches, teams, and tournament progression using real FIFA rankings and player data from fifaratings.com.

## Features

- **Full Tournament Simulation**: Simulates the complete 2026 World Cup with 48 teams across group stage and knockout rounds
- **Real FIFA Data**: Uses official FIFA World Rankings and player OVA ratings from fifaratings.com
- **Combined Strength Model**: Merges player quality (60%) and FIFA ranking (40%) into an ELO-like scale (1200-1800)
- **Monte Carlo Simulation**: Run thousands of simulations to calculate title odds, finalist probabilities, and semifinalist percentages
- **Match Statistics**: Tracks goals, penalties, and team performance throughout the tournament
- **Caching System**: Local cache for FIFA rankings and player ratings to reduce API calls
- **Confederation Weighting**: Adjusts team strength based on confederation (UEFA, CONMEBOL, etc.)
- **Recent Form Factor**: Considers recent match results in team strength calculations

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- numpy, pandas, requests, tabulate, matplotlib

## Usage

### Single Tournament Simulation
```bash
python simulation.py
```

### Monte Carlo Simulation (Title Odds)
```bash
python simulation.py --mc --runs 5000
```

### Output
The simulator displays:
- Group stage results with match scores
- Knockout round fixtures and results
- Tournament statistics (average goals, highest-scoring match, top scorers)
- (With `--mc`) Win percentage, final appearance percentage, and semifinal appearance percentage for each team

### Data Refresh
Player ratings and FIFA rankings are cached in the `cache/` directory. To force a fresh data fetch, delete the cache files:
```bash
rm cache/*.json
```

## Project Structure

```
.
├── simulation.py # Main simulation engine
├── groups.json # 2026 World Cup group stage configuration
├── requirements.txt # Python dependencies
├── cache/ # Cached FIFA data (player ratings, rankings)
├── simulation.log # Simulation output log
└── README.md
```

## Tournament Structure

- **48 teams** divided into **12 groups** of 4 teams each
- **Group Stage**: Each team plays 3 matches (round-robin within group); top 2 from each group (24 teams) + 8 best third-place teams advance
- **Knockout Rounds**: Round of 32 → Round of 16 → Quarterfinals → Semifinals → Final + Third Place

## Algorithm

Match simulation uses a Poisson distribution model with:
- Team strength differential derived from combined FIFA ratings
- ELO-based win probability calculation
- Home advantage for host nations
- Draw probability with strength-gap adjustment

## License

MIT
