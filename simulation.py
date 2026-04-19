import random
import json
import os
import copy
import itertools
import logging
import hashlib
import re

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

# Constants
ELO_DIVISOR = 300
DRAW_PROBABILITY = 0.24
DEFAULT_TEAM_STRENGTH = 1500
BASE_GOALS = 1.35
FIFARATINGS_BASE_URL = "https://www.fifaratings.com"
FIFA_RANKING_URL = "https://inside.fifa.com/fifa-world-ranking/men"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
PLAYER_RATINGS_CACHE = CACHE_DIR / "player_ratings.json"
FIFA_RANKING_CACHE = CACHE_DIR / "fifa_ranking.json"
FORM_CACHE = CACHE_DIR / "team_form.json"
CACHE_VERSION = "v2"  # Bump to force full refresh

# Strength weighting: 60% player quality (OVA), 40% FIFA ranking (team performance)
PLAYER_WEIGHT = 0.6
RANKING_WEIGHT = 0.4
CONFEDERATION_STRENGTH = {
    "UEFA": 1.03,
    "CONMEBOL": 1.025,
    "CONCACAF": 1.0,
    "CAF": 0.995,
    "AFC": 0.985,
    "OFC": 0.97,
}

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

# Import dependencies
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'requests',
    'tabulate',
    'matplotlib',
]

def import_dependencies():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            logging.info(f"Installing {package}...")
            os.system(f"pip install {package}")

import_dependencies()

import numpy as np
import pandas as pd
import requests
from tabulate import tabulate
import matplotlib.pyplot as plt


def normalize_country_name(name: str) -> str:
    """Normalize country name for fifaratings.com URL generation."""
    mappings = {
        "Curaçao": "Curacao",
        "Ivory Coast": "Ivory-Coast",
        "South Korea": "South-Korea",
        "Bosnia and Herzegovina": "Bosnia",
        "DR Congo": "DR-Congo",
        "New Zealand": "New-Zealand",
        "Saudi Arabia": "Saudi-Arabia",
        "Cape Verde": "Cape-Verde",
    }
    return mappings.get(name, name.replace(" ", "-"))


def normalize_country_for_fifa(name: str) -> str:
    """Normalize country name for FIFA ranking lookup."""
    mappings = {
        "United States": "USA",
        "South Korea": "Korea Republic",
        "Ivory Coast": "Côte d'Ivoire",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "DR Congo": "Congo DR",
        "Cape Verde": "Cape Verde Islands",
        "Czech Republic": "Czechia",
        "Turkey": "Türkiye",
        "Curaçao": "Curaçao",
    }
    return mappings.get(name, name)


def compute_confederation_factor(confederation: Optional[str]) -> float:
    """Small regional strength prior derived from recent FIFA ranking distribution."""
    return CONFEDERATION_STRENGTH.get(confederation or "", 1.0)


def compute_recent_form_factor(recent_matches: Optional[List[Dict]]) -> float:
    """Return a bounded multiplicative factor based on a short recent-results window."""
    if not recent_matches:
        return 1.0

    weighted_points = 0.0
    weighted_goal_diff = 0.0
    total_weight = 0.0

    for idx, match in enumerate(recent_matches[:6]):
        weight = max(0.4, 1.0 - idx * 0.1)
        is_home = match.get("is_home", True)
        goals_for = match.get("home_score", 0) if is_home else match.get("away_score", 0)
        goals_against = match.get("away_score", 0) if is_home else match.get("home_score", 0)

        if goals_for > goals_against:
            points = 3
        elif goals_for == goals_against:
            points = 1
        else:
            points = 0

        weighted_points += points * weight
        weighted_goal_diff += (goals_for - goals_against) * weight
        total_weight += weight

    if total_weight == 0:
        return 1.0

    points_per_match = weighted_points / total_weight
    goal_diff_per_match = weighted_goal_diff / total_weight
    factor = 1.0 + (points_per_match - 1.2) * 0.035 + goal_diff_per_match * 0.015
    return max(0.94, min(1.06, factor))


def compute_adjusted_strength(
    base_strength: float,
    confederation: Optional[str] = None,
    recent_matches: Optional[List[Dict]] = None,
) -> float:
    """Apply bounded confederation and recent-form adjustments to base strength."""
    adjusted = base_strength
    adjusted *= compute_confederation_factor(confederation)
    adjusted *= compute_recent_form_factor(recent_matches)
    return max(1200.0, min(1800.0, adjusted))


def get_fifaratings_url(country: str) -> str:
    normalized = normalize_country_name(country)
    return f"{FIFARATINGS_BASE_URL}/country/{normalized}"


def fetch_player_ratings_from_fifaratings(country: str, session: requests.Session) -> Optional[float]:
    """
    Fetch team strength by averaging the top 11 players' OVA ratings from fifaratings.com.
    """
    url = get_fifaratings_url(country)
    try:
        response = session.get(url, timeout=15)
        if response.status_code != 200:
            return None

        html = response.text

        # The current page structure puts OVA in the 3rd column, not the 4th.
        # Extract only player-table rows: OVA cell immediately followed by TOTAL cell.
        matches = re.findall(
            r'<td><span class="[^"]*attribute-box[^"]*">\s*([0-9]+(?:\.[0-9]+)?)\s*</span></td>\s*'
            r'<td><span class="[^"]*total_attributes[^"]*">',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        player_ratings = [float(value) for value in matches if float(value) > 0]

        if not player_ratings:
            return None

        top_players = sorted(player_ratings, reverse=True)[:11]
        return sum(top_players) / len(top_players)

    except Exception:
        return None


def fetch_fifa_world_ranking(session: requests.Session) -> Dict[str, float]:
    """
    Fetch FIFA World Ranking points from FIFA's JSON API.
    Returns dict of {country_name: ranking_points}.
    """
    api_url = "https://api.fifa.com/api/v3/rankings?gender=1"
    try:
        response = session.get(api_url, timeout=30)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch FIFA ranking: HTTP {response.status_code}")
            return {}

        payload = response.json()
        ranking_data = {}

        for row in payload.get("Results", []):
            team_name_entries = row.get("TeamName") or []
            team_name = None
            if team_name_entries:
                team_name = team_name_entries[0].get("Description")
            points = row.get("DecimalTotalPoints") or row.get("TotalPoints")
            if team_name and isinstance(points, (int, float)) and points > 0:
                ranking_data[team_name] = float(points)

        logging.info(f"Fetched FIFA ranking for {len(ranking_data)} teams")
        return ranking_data

    except Exception as e:
        logging.warning(f"Error fetching FIFA ranking: {e}")
        return {}


def match_fifa_ranking_country(country: str, fifa_data: Dict[str, float]) -> Optional[float]:
    """Match a country name to FIFA ranking data with fuzzy matching."""
    # Direct match
    if country in fifa_data:
        return fifa_data[country]
    
    # Try normalized names
    normalized = normalize_country_for_fifa(country)
    if normalized in fifa_data:
        return fifa_data[normalized]
    
    # Fuzzy matching for common variations
    country_lower = country.lower()
    for fifa_country, points in fifa_data.items():
        if country_lower in fifa_country.lower() or fifa_country.lower() in country_lower:
            return points
    
    # Special cases
    special_mappings = {
        "United States": ["United States", "USA", "United States of America"],
        "South Korea": ["South Korea", "Korea Republic", "Korea"],
        "Bosnia and Herzegovina": ["Bosnia", "Bosnia and Herzegovina"],
        "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
        "DR Congo": ["DR Congo", "Congo DR"],
        "Czech Republic": ["Czech Republic", "Czechia"],
        "Cape Verde": ["Cape Verde", "Cape Verde Islands"],
        "Turkey": ["Turkey", "Türkiye"],
        "Curaçao": ["Curaçao", "Curacao"],
    }
    
    for key, variants in special_mappings.items():
        if country in variants:
            for variant in variants:
                if variant in fifa_data:
                    return fifa_data[variant]
    
    return None


def load_cached_player_ratings() -> Dict:
    if PLAYER_RATINGS_CACHE.exists():
        try:
            with open(PLAYER_RATINGS_CACHE, 'r') as f:
                data = json.load(f)
                if data.get("version") == CACHE_VERSION:
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return {"version": CACHE_VERSION, "teams": {}, "fetched_at": None}


def load_cached_fifa_ranking() -> Dict:
    if FIFA_RANKING_CACHE.exists():
        try:
            with open(FIFA_RANKING_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"teams": {}, "rows": [], "fetched_at": None}


def load_cached_form() -> Dict:
    if FORM_CACHE.exists():
        try:
            with open(FORM_CACHE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"teams": {}, "fetched_at": None}


def save_player_ratings_cache(ratings: Dict[str, float]):
    cache_data = {
        "version": CACHE_VERSION,
        "teams": ratings,
        "fetched_at": time.time()
    }
    with open(PLAYER_RATINGS_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    logging.info(f"Saved player ratings cache with {len(ratings)} teams")


def save_fifa_ranking_cache(ranking: Dict[str, float], rows: Optional[List[Dict]] = None):
    cache_data = {
        "teams": ranking,
        "rows": rows or [],
        "fetched_at": time.time()
    }
    with open(FIFA_RANKING_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)


def save_form_cache(team_form: Dict[str, Dict]):
    cache_data = {
        "teams": team_form,
        "fetched_at": time.time()
    }
    with open(FORM_CACHE, 'w') as f:
        json.dump(cache_data, f, indent=2)


def fetch_team_form_from_rankings(
    countries: List[str],
    fifa_ranking_rows: List[Dict],
) -> Dict[str, Dict]:
    """Build lightweight real-data metadata from the public FIFA ranking feed."""
    lookup = {}
    for row in fifa_ranking_rows:
        team_name_entries = row.get("TeamName") or []
        if not team_name_entries:
            continue
        team_name = team_name_entries[0].get("Description")
        if team_name:
            lookup[team_name] = row

    team_form = {}
    for country in countries:
        row = lookup.get(country) or lookup.get(normalize_country_for_fifa(country))
        if row:
            matches = row.get("Matches") or 0
            activity_factor = 1.0 + min(0.015, matches / 10000.0)
            team_form[country] = {
                "confederation": row.get("ConfederationName"),
                "recent_matches": [],
                "activity_factor": activity_factor,
            }
        else:
            team_form[country] = {
                "confederation": None,
                "recent_matches": [],
                "activity_factor": 1.0,
            }
    return team_form


def compute_combined_strength(
    player_rating: Optional[float], 
    fifa_points: Optional[float]
) -> float:
    """
    Combine player OVA ratings and FIFA World Ranking points into a single strength.
    Uses a shared ELO-like scale so the difference in ratings actually matters.
    """
    PLAYER_MIN, PLAYER_MAX = 60, 95
    FIFA_MIN, FIFA_MAX = 800, 1900

    if player_rating is None and fifa_points is None:
        return DEFAULT_TEAM_STRENGTH

    if player_rating is None:
        # If player ratings couldn't be fetched (e.g. rate limited), don't
        # inflate the team using purely fifa points. Historically, player_norm
        # runs lower than fifa_norm (e.g. 0.7 vs 0.95). So we infer a synthetic player norm.
        fifa_norm = max(0, min(1, (fifa_points - FIFA_MIN) / (FIFA_MAX - FIFA_MIN)))
        synthetic_player_norm = fifa_norm * 0.75 
        combined = PLAYER_WEIGHT * synthetic_player_norm + RANKING_WEIGHT * fifa_norm
        return 1200 + combined * 600

    if fifa_points is None:
        player_norm = max(0, min(1, (player_rating - PLAYER_MIN) / (PLAYER_MAX - PLAYER_MIN)))
        synthetic_fifa_norm = min(1.0, player_norm * 1.33)
        combined = PLAYER_WEIGHT * player_norm + RANKING_WEIGHT * synthetic_fifa_norm
        return 1200 + combined * 600

    player_norm = max(0, min(1, (player_rating - PLAYER_MIN) / (PLAYER_MAX - PLAYER_MIN)))
    fifa_norm = max(0, min(1, (fifa_points - FIFA_MIN) / (FIFA_MAX - FIFA_MIN)))
    combined = PLAYER_WEIGHT * player_norm + RANKING_WEIGHT * fifa_norm

    return 1200 + combined * 600


def compute_match_strength(team_ratings: Dict[str, float], team_name: str) -> float:
    """Return a usable strength for match simulation, even when data is missing."""
    rating = team_ratings.get(team_name)
    if rating is None:
        return DEFAULT_TEAM_STRENGTH
    # Clamp obvious bad values so missing/garbage data cannot dominate.
    if not isinstance(rating, (int, float)) or rating <= 0:
        return DEFAULT_TEAM_STRENGTH
    return float(rating)

def fetch_all_data(countries: List[str], use_cache: bool = True) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """
    Fetch player ratings, FIFA rankings, and lightweight team metadata,
    then compute adjusted strengths.
    """
    cached_player = load_cached_player_ratings()
    cached_fifa = load_cached_fifa_ranking()
    cached_form = load_cached_form()

    cached_player_teams = cached_player.get("teams", {})
    cached_fifa_teams = cached_fifa.get("teams", {})
    cached_fifa_rows = cached_fifa.get("rows", [])
    cached_form_teams = cached_form.get("teams", {})

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    if use_cache and cached_fifa_teams:
        fifa_ranking = cached_fifa_teams
        fifa_ranking_rows = cached_fifa_rows
        logging.info(f"Using cached FIFA ranking ({len(fifa_ranking)} teams)")
    else:
        logging.info("Fetching FIFA World Ranking from api.fifa.com...")
        api_url = "https://api.fifa.com/api/v3/rankings?gender=1"
        response = session.get(api_url, timeout=30)
        payload = response.json() if response.status_code == 200 else {}
        fifa_ranking_rows = payload.get("Results", [])
        fifa_ranking = {}
        for row in fifa_ranking_rows:
            team_name_entries = row.get("TeamName") or []
            team_name = team_name_entries[0].get("Description") if team_name_entries else None
            points = row.get("DecimalTotalPoints") or row.get("TotalPoints")
            if team_name and isinstance(points, (int, float)) and points > 0:
                fifa_ranking[team_name] = float(points)
        if fifa_ranking:
            save_fifa_ranking_cache(fifa_ranking, fifa_ranking_rows)
        else:
            fifa_ranking = cached_fifa_teams
            fifa_ranking_rows = cached_fifa_rows

    results = {}
    missing_player = []
    for country in countries:
        if use_cache and country in cached_player_teams:
            results[country] = cached_player_teams[country]
        else:
            missing_player.append(country)

    if not missing_player:
        logging.info("All player ratings loaded from cache")
    else:
        logging.info(f"Fetching player ratings for {len(missing_player)} countries...")
        for country in missing_player:
            rating = None
            for attempt in range(3):
                rating = fetch_player_ratings_from_fifaratings(country, session)
                if rating is not None:
                    break
                logging.warning(f"Failed to fetch {country}, retrying in {2 * (attempt + 1)}s...")
                time.sleep(2 * (attempt + 1))
            
            results[country] = rating if rating else None
            time.sleep(1.0)  # rate limit delay

    save_player_ratings_cache({k: v for k, v in results.items() if v is not None})

    if use_cache and cached_form_teams and all(country in cached_form_teams for country in countries):
        team_metadata = {country: cached_form_teams[country] for country in countries}
    else:
        team_metadata = fetch_team_form_from_rankings(countries, fifa_ranking_rows)
        save_form_cache(team_metadata)

    final_strength = {}
    for country in countries:
        player_rating = results.get(country)
        fifa_points = match_fifa_ranking_country(country, fifa_ranking)
        strength = compute_combined_strength(player_rating, fifa_points)
        metadata = team_metadata.get(country, {})
        strength = compute_adjusted_strength(
            strength,
            confederation=metadata.get("confederation"),
            recent_matches=metadata.get("recent_matches"),
        ) * metadata.get("activity_factor", 1.0)
        final_strength[country] = max(1200.0, min(1800.0, strength))

    return final_strength, team_metadata


@lru_cache(maxsize=128)
def win_expectancy(difference: float) -> float:
    """Calculate win probability using ELO rating system."""
    return 1 / (10 ** (-difference/ELO_DIVISOR) + 1)


def rank_group_teams(group: dict, team_ratings: Dict[str, float]) -> List[Tuple]:
    """Rank teams by results first, then goal stats, then strength as a final fallback."""
    standings = group.get("standings", {})
    goal_stats = group.get("goal_stats", {})
    ranked = []

    for team_name, points in standings.items():
        stats = goal_stats.get(team_name, {})
        goals_for = stats.get("goals_for", 0)
        goals_against = stats.get("goals_against", 0)
        goal_difference = goals_for - goals_against
        team_data = group.get(team_name, ())
        strength = compute_match_strength(team_ratings, team_name)
        ranked.append(
            (
                team_name,
                team_data,
                points,
                goal_difference,
                goals_for,
                goals_against,
                strength,
            )
        )

    return sorted(
        ranked,
        key=lambda x: (x[2], x[3], x[4], x[6], x[0]),
        reverse=True,
    )


def resolve_knockout_match(team_a: Tuple, team_b: Tuple, team_ratings: Dict[str, float]):
    """Resolve a knockout match, only using penalties if the score is tied."""
    regulation = sim(team_a, team_b, team_ratings, no_draw=False)
    score = regulation[1]

    if score[0] > score[1]:
        winner = team_a
        penalties = False
    elif score[1] > score[0]:
        winner = team_b
        penalties = False
    else:
        winner = sim(team_a, team_b, team_ratings, no_draw=True)[0]
        penalties = True

    return [winner, score, penalties]


def sim(team_a: Tuple, team_b: Tuple, team_ratings: Dict[str, float], no_draw: bool = False) -> List:
    """
    Simulate a football match using calibrated strength-derived expected goals.
    """
    team_a_name = team_a[0] if isinstance(team_a[0], str) else str(team_a[0])
    team_b_name = team_b[0] if isinstance(team_b[0], str) else str(team_b[0])

    strength_a = compute_match_strength(team_ratings, team_a_name)
    strength_b = compute_match_strength(team_ratings, team_b_name)

    difference = strength_a - strength_b
    win_prob_a = win_expectancy(difference)
    strength_gap = max(-1.0, min(1.0, difference / 220.0))

    lambda_a = max(0.2, BASE_GOALS * (1.0 + 0.38 * strength_gap))
    lambda_b = max(0.2, BASE_GOALS * (1.0 - 0.38 * strength_gap))

    goals_a = int(np.random.poisson(lambda_a))
    goals_b = int(np.random.poisson(lambda_b))

    if not no_draw and goals_a == goals_b:
        if random.random() < DRAW_PROBABILITY + max(0.0, 0.08 - abs(strength_gap) * 0.06):
            return [[team_a, team_b], (goals_a, goals_b)]

    if goals_a == goals_b:
        if random.random() < win_prob_a:
            goals_a += 1
        else:
            goals_b += 1

    if goals_a > goals_b:
        return [team_a, (goals_a, goals_b)]
    return [team_b, (goals_a, goals_b)]


def replace_many_one(text: str, chars_to_replace: list, replacement: str) -> str:
    for char in chars_to_replace:
        text = text.replace(char, replacement)
    return text


def return_removed(original_list: list, items_to_remove: list) -> list:
    for item in items_to_remove:
        original_list.remove(item)
    return original_list


def conf_in_group(conf: str, group: dict) -> bool:
    if len(group) == 0:
        return False
    in_group = [group[x][1] for x in group]
    if conf in in_group:
        if conf in ["AFC", "CAF", "UEFA"] and (in_group+[conf]).count(conf) <= 2:
            return False
        return True
    return False


def randomize_seeding(pots: dict, groups: dict) -> dict:
    while True:
        groups_copy = copy.deepcopy(groups)
        for i, pot in enumerate(pots):
            teams = pots[pot]
            for team in teams:
                choices = [x for x in groups_copy if len(
                    groups_copy[x].keys()) < i+1 and not conf_in_group(team[2], groups_copy[x])]
                if not len(choices) == 0:
                    group_name = random.choice(choices)
                    groups_copy[group_name][team[0]] = team[1:] + [{}]
        final_groups = groups_copy
        country_count = sum([len(final_groups[x]) for x in final_groups])
        if country_count == 48:
            break
    return final_groups


def seed(groups: dict, pots: dict):
    hosts = [x for x in pots["1"] if len(x) >= 3 and isinstance(x[1], (int, float)) and x[1] > 0][:3]
    for host in hosts:
        for pot_key in pots:
            if host in pots[pot_key]:
                pots[pot_key].remove(host)
                break
    
    for i, group in enumerate(groups):
        if i in [0, 4, 8] and hosts:
            choice = random.choice(hosts)
            hosts.remove(choice)
            groups[group][choice[0]] = choice[1:] + [{}]

    groups = randomize_seeding(pots, groups)
    return groups


def schedule_matches(groups: dict):
    for group in groups:
        countries = [k for k in groups[group]]
        groups[group]["matches"] = list(itertools.combinations(countries, r=2))
        groups[group]["standings"] = {}
    return groups


def group_stage(groups: dict, team_ratings: Dict[str, float]) -> dict:
    for group in groups:
        standings = {x: 0 for x in groups[group] if x[0].isupper()}
        goal_stats = {
            x: {"goals_for": 0, "goals_against": 0}
            for x in groups[group]
            if x[0].isupper()
        }
        for i, match in enumerate(groups[group]["matches"]):
            outcome = sim(
                (match[0], groups[group][match[0]]),
                (match[1], groups[group][match[1]]),
                team_ratings
            )
            groups[group]["matches"][i] = groups[group]["matches"][i] + ((outcome[1],))
            home_team, away_team = match[0], match[1]
            home_goals, away_goals = outcome[1]
            goal_stats[home_team]["goals_for"] += home_goals
            goal_stats[home_team]["goals_against"] += away_goals
            goal_stats[away_team]["goals_for"] += away_goals
            goal_stats[away_team]["goals_against"] += home_goals
            if isinstance(outcome[0][0], str):
                standings[outcome[0][0]] += 3
            else:
                for country in outcome[0]:
                    standings[country[0]] += 1
        groups[group]["standings"] = standings
        groups[group]["goal_stats"] = goal_stats
    return groups


def knockout_round(groups, team_ratings: Dict[str, float]):
    qual_dict = {"1": [], "2": [], "3": []}
    for group in groups:
        sorted_countries = rank_group_teams(groups[group], team_ratings)
        for i in range(3):
            qual_dict[str(i+1)].append(sorted_countries[i])

    qual_dict["3"] = sorted(
        qual_dict["3"],
        key=lambda x: (x[2], x[3], x[4], x[6], x[0]),
        reverse=True
    )[:8]

    all_countries = []
    for key in qual_dict:
        all_countries += qual_dict[key]

    knockout_dict = {x: [] for x in ["round of 32", "round of 16",
                                     "round of 8", "semifinals", "third place", "finals"]}

    for a, b in zip(all_countries[:16], reversed(all_countries[16:])):
        knockout_dict["round of 32"].append((resolve_knockout_match(a, b, team_ratings), (a, b)))

    for i in range(0, len(knockout_dict["round of 32"]), 2):
        a = knockout_dict["round of 32"][i][0][0]
        b = knockout_dict["round of 32"][i+1][0][0]
        knockout_dict["round of 16"].append((resolve_knockout_match(a, b, team_ratings), (a, b)))

    for i in range(0, len(knockout_dict["round of 16"]), 2):
        a = knockout_dict["round of 16"][i][0][0]
        b = knockout_dict["round of 16"][i+1][0][0]
        knockout_dict["round of 8"].append((resolve_knockout_match(a, b, team_ratings), (a, b)))

    for i in range(0, len(knockout_dict["round of 8"]), 2):
        a = knockout_dict["round of 8"][i][0][0]
        b = knockout_dict["round of 8"][i+1][0][0]
        knockout_dict["semifinals"].append((resolve_knockout_match(a, b, team_ratings), (a, b)))

    finalist_a = knockout_dict["semifinals"][0][0][0]
    finalist_b = knockout_dict["semifinals"][1][0][0]

    third_a = [x for x in knockout_dict["semifinals"]
               [0][1] if not x == finalist_a][0]
    third_b = [x for x in knockout_dict["semifinals"]
               [1][1] if not x == finalist_b][0]

    knockout_dict["third place"].append(
        (resolve_knockout_match(third_a, third_b, team_ratings), (third_a, third_b)))

    knockout_dict["finals"].append(
        (resolve_knockout_match(finalist_a, finalist_b, team_ratings), (finalist_a, finalist_b)))

    return knockout_dict


def display_fixture(groups, knockout_brackets):
    print("Fixture and Results\n")
    print("Group Stage:\n")

    total_goals = 0
    num_matches = 0
    most_goals_match = None
    most_goals = 0
    team_goals = defaultdict(int)

    for group in groups:
        print(f"Group {group}:\n")
        matches = []
        for match in groups[group]["matches"]:
            home_team, away_team, result = match
            matches.append([home_team, result[0], '-', result[1], away_team])

            total_goals += sum(result)
            num_matches += 1
            team_goals[home_team] += result[0]
            team_goals[away_team] += result[1]

            if sum(result) > most_goals:
                most_goals = sum(result)
                most_goals_match = f"{home_team} {result[0]} - {result[1]} {away_team}"

        print(tabulate(matches, headers=[
              'Home Team', 'Score', '', 'Score', 'Away Team']))
        print()

    print("Knockout Rounds:\n")

    knockout_round_names = ["round of 32", "round of 16",
                            "round of 8", "semifinals", "third place", "finals"]

    for round_name in knockout_round_names:
        print(f"{round_name.capitalize()}:\n")
        matches = []
        for match_data in knockout_brackets[round_name]:
            result = match_data[0][1]
            home_team, away_team = match_data[1]
            home_result, away_result = result
            if match_data[0][2]:
                winner = match_data[0][0][0] + " (P)"
            else:
                winner = match_data[0][0][0]
            matches.append([home_team[0], home_result, '-',
                           away_result, away_team[0], winner])

            total_goals += sum(result)
            num_matches += 1
            team_goals[home_team[0]] += home_result
            team_goals[away_team[0]] += away_result

            if sum(result) > most_goals:
                most_goals = sum(result)
                most_goals_match = f"{home_team[0]} {home_result} - {away_result} {away_team[0]}"

        print(tabulate(matches, headers=[
              'Home Team', 'Score', '', 'Score', 'Away Team', 'Winner (Penalties)']))
        print()

    avg_goals = total_goals / num_matches
    most_goals_team = [k for k, v in team_goals.items() if v == max(team_goals.values())]
    least_goals_team = [k for k, v in team_goals.items() if v == min(team_goals.values())]

    print("Statistics:")
    print(f"Total goal average: {avg_goals:.2f}")
    print(f"Match with most goals: {most_goals_match} ({most_goals} goals)")
    print(f"Team(s) with most goals: {', '.join(most_goals_team)} ({max(team_goals.values())} goals)")
    print(f"Team(s) with least goals: {', '.join(least_goals_team)} ({min(team_goals.values())} goals)")


def check_for_updates(countries: List[str]):
    """Background check for rating updates after fixture generation."""
    def background_check():
        logging.info("Checking for rating updates...")
        # Fresh fetch to detect changes
        new_data = fetch_all_data(countries, use_cache=False)
        return new_data
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(background_check)
        try:
            future.result(timeout=120)
            logging.info("Rating update check complete.")
        except Exception as e:
            logging.warning(f"Background check failed: {e}")


def run_monte_carlo(official_groups: dict, team_ratings: Dict[str, float], num_simulations: int = 2000):
    print(f"\nRunning Monte Carlo Simulation ({num_simulations} iterations)...")
    title_wins = defaultdict(int)
    finalist_counts = defaultdict(int)
    semifinalist_counts = defaultdict(int)
    
    # We reuse the schedule matches setup
    base_groups = {}
    for group_name, teams in official_groups.items():
        base_groups[group_name] = {}
        for team_name in teams:
            strength = team_ratings.get(team_name, DEFAULT_TEAM_STRENGTH)
            base_groups[group_name][team_name] = (strength, 1.0)
            
    base_groups = schedule_matches(base_groups)

    print("Simulating...")
    for i in range(num_simulations):
        sim_groups = copy.deepcopy(base_groups)
        sim_groups = group_stage(sim_groups, team_ratings)
        brackets = knockout_round(sim_groups, team_ratings)
        
        finals_match = brackets["finals"][0]
        winner_name = finals_match[0][0][0]
        finalists = [finals_match[1][0][0], finals_match[1][1][0]]
        
        semis_teams = []
        for match in brackets["semifinals"]:
            semis_teams.append(match[1][0][0])
            semis_teams.append(match[1][1][0])
            
        title_wins[winner_name] += 1
        for f in finalists:
            finalist_counts[f] += 1
        for sf in semis_teams:
            semifinalist_counts[sf] += 1
            
        if (i+1) % (max(1, num_simulations // 10)) == 0:
            print(f"  {i+1}/{num_simulations} complete...")

    print("\nSimulation Results - Title Winner Odds:")
    results = []
    # Print teams with at least 0.1% chance of making semis or better
    for team, wins in sorted(title_wins.items(), key=lambda x: x[1], reverse=True):
        odds = (wins / num_simulations) * 100
        finals_odds = (finalist_counts[team] / num_simulations) * 100
        semis_odds = (semifinalist_counts[team] / num_simulations) * 100
        if semis_odds >= 0.1 or odds > 0:
            results.append([team, f"{odds:.2f}%", f"{finals_odds:.2f}%", f"{semis_odds:.2f}%"])
        
    print(tabulate(results, headers=["Team", "Win %", "Final %", "Semi %"]))


def run_simulation(monte_carlo: bool = False, num_simulations: int = 2000):
    """Main simulation runner with combined FIFA ratings."""
    import datetime
    
    logging.info("="*60)
    logging.info("World Cup Simulation - FIFA Ratings + World Ranking")
    logging.info("="*60)
    
    # Load groups
    with open("groups.json", "r") as f:
        official_groups = json.load(f)
    
    # Collect all unique countries
    all_countries = []
    for teams in official_groups.values():
        all_countries.extend(teams)
    all_countries = list(set(all_countries))
    
    print(f"\nFetching data for {len(all_countries)} teams...")
    print(f"  - FIFA World Ranking from inside.fifa.com")
    print(f"  - Player ratings from fifaratings.com")
    print(f"  - Combined strength: normalized to an ELO-like 1200-1800 scale\n")

    # Fetch combined team strength data
    team_ratings, team_metadata = fetch_all_data(all_countries, use_cache=True)
    
    # Show top 10 teams
    top_teams = sorted(team_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 Teams by Combined Strength:")
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"  {i}. {team}: {rating:.1f}")
    print()
    
    if monte_carlo:
        run_monte_carlo(official_groups, team_ratings, num_simulations)
    else:
        # Prepare simulation groups
        simulation_groups = {}
        for group_name, teams in official_groups.items():
            simulation_groups[group_name] = {}
            for team_name in teams:
                strength = team_ratings.get(team_name, DEFAULT_TEAM_STRENGTH)
                simulation_groups[group_name][team_name] = (strength, 1.0)
        
        # Schedule and simulate
        simulation_groups = schedule_matches(simulation_groups)
        
        print("Simulating Group Stage...")
        simulation_groups = group_stage(simulation_groups, team_ratings)
        
        print("Simulating Knockout Rounds...")
        knockout_brackets = knockout_round(simulation_groups, team_ratings)
        
        # Display results
        display_fixture(simulation_groups, knockout_brackets)
    
    # Background update check
    print("\n[Background] Checking for rating updates...")
    check_for_updates(all_countries)
    
    # Cache info
    try:
        player_cache = load_cached_player_ratings()
        fifa_cache = load_cached_fifa_ranking()
        if player_cache.get("fetched_at"):
            fetched = datetime.datetime.fromtimestamp(player_cache["fetched_at"])
            print(f"\nData refreshed: {fetched.strftime('%Y-%m-%d %H:%M:%S')}")
    except:
        pass
    
    logging.info("Simulation complete!")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Simulator")
    parser.add_argument("--mc", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--runs", type=int, default=2000, help="Number of Monte Carlo simulations")
    args = parser.parse_args()
    
    run_simulation(monte_carlo=args.mc, num_simulations=args.runs)
