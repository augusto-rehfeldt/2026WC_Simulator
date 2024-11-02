import random
import json
import os
import copy
import itertools
import logging
from pathlib import Path
from typing import List, Tuple, Any
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import warnings
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

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

# Import dependencies with error handling
REQUIRED_PACKAGES = [
    'numpy',
    'pandas',
    'requests',
    'bs4',
    'tabulate',
    'matplotlib',
    'selenium'
]

def import_dependencies():
    """Import required packages, installing if necessary."""
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
from bs4 import BeautifulSoup
from tabulate import tabulate
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Add cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

@lru_cache(maxsize=128)
def win_expectancy(difference: float) -> float:
    """Calculate win probability using ELO rating system."""
    return 1 / (10 ** (-difference/400) + 1)

def sim(team_a: Tuple, team_b: Tuple, no_draw: bool = False) -> List:
    """
    Simulate match outcome with improved scoring model.
    Uses team FIFA ranking and player strength.
    """
    # Calculate strength multipliers with bounds checking
    if all(x[1][1] for x in [team_a, team_b]):
        strength_ratio = team_a[1][1] / team_b[1][1]
        ovr_multiplier_a = min(1.5, max(0.5, 1 + (strength_ratio - 1)/2))
        ovr_multiplier_b = min(1.5, max(0.5, 1 - (strength_ratio - 1)/2))
    else:
        ovr_multiplier_a = ovr_multiplier_b = 1

    # Calculate adjusted team strengths
    strength_a = team_a[1][0] * ovr_multiplier_a if team_a[1][0] else 1000
    strength_b = team_b[1][0] * ovr_multiplier_b if team_b[1][0] else 1000
    
    # Calculate win probabilities
    difference = strength_a - strength_b
    win_prob_a = win_expectancy(abs(difference)) if strength_a >= strength_b else 1 - win_expectancy(abs(difference))
    
    # Determine winner
    random_val = random.random()
    if no_draw:
        winner = team_a if random_val < win_prob_a else team_b
    else:
        draw_prob = 0.25  # Configurable draw probability
        if random_val < win_prob_a * (1 - draw_prob):
            winner = team_a
        elif random_val > 1 - ((1 - win_prob_a) * (1 - draw_prob)):
            winner = team_b
        else:
            winner = [team_a, team_b]

    # Generate realistic scores based on team strengths
    base_goals = 1.5
    # Ensure strength_factor is between 0 and 1
    strength_factor = max(0, min(1, abs(strength_a - strength_b) / max(strength_a, strength_b)))
    
    if isinstance(winner, list):  # Draw
        score = max(0, np.random.poisson(base_goals))
        return [winner, (score, score)]
    else:
        # Ensure non-negative lambda parameters
        winner_lambda = max(0.1, base_goals * (1 + strength_factor))
        loser_lambda = max(0.1, base_goals * (1 - strength_factor))
        
        winner_goals = np.random.poisson(winner_lambda)
        loser_goals = np.random.poisson(loser_lambda)
        
        if winner == team_a:
            return [winner, (winner_goals, loser_goals)]
        else:
            return [winner, (loser_goals, winner_goals)]

@lru_cache(maxsize=32)
def fetch_team_data(confederation: str, url: str) -> List[List]:
    """Fetch and cache team ranking data."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        rankings = []
        for table in soup.find_all("table", {"class": "wikitable"}):
            if "FIFA Men's Rankings" in table.text:
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        try:
                            team = cells[-2].text.strip()
                            rank = float(cells[-1].text.strip())
                            # Apply CONCACAF bonus
                            if team in ["United States", "Mexico", "Canada"]:
                                rank += 100
                            rankings.append([team, rank])
                        except (ValueError, IndexError):
                            continue
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    except Exception as e:
        logging.error(f"Error fetching {confederation} data: {e}")
        return []

def replace_many_one(string: str, lst: list, to_replace: str):
    for x in lst:
        string = string.replace(x, to_replace)
    return string


def return_removed(lst: list, items: list):
    for item in items:
        lst.remove(item)
    return lst


def get_teams(confederation: str, confederation_page: str, n: int, conf_dict: dict) -> dict:
    r = requests.get(confederation_page)
    soup = BeautifulSoup(r.content, "html.parser")
    ranking = []
    for table in soup.find_all("table", {"class": "wikitable"}):
        if len(table.find_all("caption")) > 0 and "FIFA Men's Rankings" in table.find_all("caption")[0].text:
            for tr in table.find("tbody").find_all("tr")[1:]:
                data = [replace_many_one(x.text, ["\n", "\xa0"], "").strip()
                        for x in tr.find_all("td")[-2:]]
                try:
                    data[-1] = float(data[-1])
                except ValueError:
                    pass
                else:
                    data[-1] = data[-1] + 100 if data[-2] in ["United States",
                                                              "Mexico", "Canada"] else data[-1]
                    ranking.append(data)
    # gives chances to underdogs against bigger opponents
    if confederation == "CONCACAF":
        conf_dict[confederation] = [x for x in ranking if x[0]
                                    in ["United States", "Canada", "Mexico"]]
        ranking = sorted([x for x in ranking if x[0] not in [
            "United States", "Canada", "Mexico"]], key=lambda x: x[1], reverse=True)
        ranking = random.sample(ranking[:n-1], n-3)
        conf_dict[confederation] = sorted(
            conf_dict[confederation]+ranking, key=lambda x: x[1], reverse=True)
    elif confederation == "OFC":
        conf_dict[confederation] = sorted(
            ranking[:n], key=lambda x: x[1], reverse=True)
    else:
        randomizer = random.sample(ranking[2:n+2], n-2)
        conf_dict[confederation] = sorted(
            ranking[:2]+randomizer, key=lambda x: x[1], reverse=True)

    return conf_dict


def final_qual_round(conf_dict: dict, n: int) -> dict:
    # runs a simulation of the playoffs between:
    # 2 teams from CONCACAF, and 1 from CONMEBOL, AFC, OFC AND CAF
    # to return 'n' qualifications (cannot be higher than 6)
    # 2 is default
    if not 0 < n < 6:
        n = 2
    qual_countries = conf_dict["CONCACAF"][-2:]
    for conf in ["CONMEBOL", "AFC", "OFC", "CAF"]:
        qual_countries.append(conf_dict[conf][-1])
    qual_countries = sorted(qual_countries, key=lambda x: x[1], reverse=True)
    lower_sorted = copy.deepcopy(qual_countries[2:])
    random.shuffle(lower_sorted)
    winners = [sim(qual_countries[0], sim(lower_sorted[0], lower_sorted[1], no_draw=True)[0], no_draw=True)[0], sim(
        qual_countries[1], sim(lower_sorted[2], lower_sorted[3], no_draw=True)[0], no_draw=True)[0]]
    for conf in conf_dict:
        if conf == "UEFA":
            continue
        else:
            if conf == "CONCACAF":
                resulting = [x for x in conf_dict[conf][-2:] if x in winners]
                conf_dict[conf] = conf_dict[conf][:-2] + resulting
            else:
                resulting = [x for x in conf_dict[conf][-1:] if x in winners]
                conf_dict[conf] = conf_dict[conf][:-1] + resulting
    return conf_dict


def create_pots(hosts: list, conf_dict: dict) -> dict:
    pots = {str(x): [] for x in range(1, 5)}
    pots["1"] = hosts
    all_countries = list(
        sorted([item for sublist in list(conf_dict.values()) for item in sublist], key=lambda x: x[1], reverse=True))

    all_countries = return_removed(
        all_countries, hosts)

    starting = 0
    for am, pot in zip([12-len(hosts), 12, 12, 12], pots):
        pots[pot] += all_countries[starting:starting+am]
        starting += am
    return pots


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
    hosts = [x for x in pots["1"][:3]]
    pots["1"] = return_removed(pots["1"], pots["1"][:3])
    for i, group in enumerate(groups):
        if i in [0, 4, 8]:
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


def best_formation(country_players: dict, formation: list):
    if len(list(country_players.keys())) == 0:
        return False

    players_copy = copy.deepcopy(country_players)

    common_substitutions = ['Striker', 'Left ',
                            ' Midfielder', 'Right ', "Goalkeeper"]

    chosen_formation = []

    used = []

    for position in formation+common_substitutions:
        for player in players_copy:
            if position in players_copy[player][0] and not player in used:
                chosen_formation.append([player] + players_copy[player])
                used.append(player)
                break

    overall_rating = sum([x[2] for x in chosen_formation]
                         )/len(chosen_formation)

    return (chosen_formation, overall_rating)


def group_stage(groups: dict) -> dict:
    # simulates the matches inside groups and the final positions
    for group in groups:
        standings = {x: 0 for x in groups[group] if x[0].isupper()}
        for i, match in enumerate(groups[group]["matches"]):
            outcome = sim((match[0], *groups[group][match[0]],),
                          (match[1], *groups[group][match[1]],))
            groups[group]["matches"][i] = groups[group]["matches"][i] + \
                ((outcome[1],))
            if isinstance(outcome[0][0], str):
                standings[outcome[0][0]] += 3
            else:
                for country in outcome[0]:
                    standings[country[0]] += 1
        groups[group]["standings"] = standings
    return groups


def knockout_round(groups):
    qual_dict = {"1": [], "2": [], "3": []}
    for group in groups:
        sorted_countries = [(x, *groups[group][x], groups[group]["standings"][x])
                            for x in groups[group] if x in groups[group]["standings"]]
        sorted_countries = sorted(
            sorted_countries, key=lambda x: x[2], reverse=True)
        for i in range(3):
            qual_dict[str(i+1)].append(sorted_countries[i])

    for i in range(2):
        qual_dict[str(i+1)] = sorted(qual_dict[str(i+1)],
                                     key=lambda x: x[2], reverse=True)
    qual_dict["3"] = sorted(
        qual_dict["3"], key=lambda x: x[2], reverse=True)[:8]

    all_countries = []
    for key in qual_dict:
        all_countries += qual_dict[key]

    knockout_dict = {x: [] for x in ["round of 32", "round of 16",
                                     "round of 8", "semifinals", "third place", "finals"]}

    for a, b in zip(all_countries[:16], reversed(all_countries[16:])):
        knockout_dict["round of 32"].append((sim(a, b, no_draw=True), (a, b)))

    for i in range(0, len(knockout_dict["round of 32"]), 2):
        a = knockout_dict["round of 32"][i][0][0]
        b = knockout_dict["round of 32"][i+1][0][0]
        knockout_dict["round of 16"].append((sim(a, b, no_draw=True), (a, b)))

    for i in range(0, len(knockout_dict["round of 16"]), 2):
        a = knockout_dict["round of 16"][i][0][0]
        b = knockout_dict["round of 16"][i+1][0][0]
        knockout_dict["round of 8"].append((sim(a, b, no_draw=True), (a, b)))

    for i in range(0, len(knockout_dict["round of 8"]), 2):
        a = knockout_dict["round of 8"][i][0][0]
        b = knockout_dict["round of 8"][i+1][0][0]
        knockout_dict["semifinals"].append((sim(a, b, no_draw=True), (a, b)))

    finalist_a = knockout_dict["semifinals"][0][0][0]
    finalist_b = knockout_dict["semifinals"][1][0][0]

    third_a = [x for x in knockout_dict["semifinals"]
               [0][1] if not x == finalist_a][0]
    third_b = [x for x in knockout_dict["semifinals"]
               [1][1] if not x == finalist_b][0]

    knockout_dict["third place"].append(
        (sim(third_a, third_b, no_draw=True), (third_a, third_b)))

    knockout_dict["finals"].append(
        (sim(finalist_a, finalist_b, no_draw=True), (finalist_a, finalist_b)))

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

            # Update statistics
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
            result = match_data[0][-1]
            home_team, away_team = match_data[1]
            home_result, away_result = result
            if home_result == away_result:
                winner = match_data[0][0][0] + " (P)"
            else:
                winner = match_data[0][0][0]
            matches.append([home_team[0], home_result, '-',
                           away_result, away_team[0], winner])

            # Update statistics
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

    # Print statistics
    avg_goals = total_goals / num_matches
    most_goals_team = [k for k, v in team_goals.items() if v == max(team_goals.values())]
    least_goals_team = [k for k, v in team_goals.items() if v == min(team_goals.values())]

    print("Statistics:")
    print(f"Total goal average: {avg_goals:.2f}")
    print(f"Match with most goals: {most_goals_match} ({most_goals} goals)")
    print(f"Team(s) with most goals: {', '.join(most_goals_team)} ({max(team_goals.values())} goals)")
    print(f"Team(s) with least goals: {', '.join(least_goals_team)} ({min(team_goals.values())} goals)")


def fetch_player_data(country):
    """Fetch player data with improved ChromeDriver management."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        # Use ChromeDriverManager to automatically handle driver versions
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        url = f"https://www.fifaratings.com/country/{country[0].replace(' ', '-').replace('Curaçao', 'Curacao')}"
        driver.get(url)
        
        # Add timeout for page load
        wait = WebDriverWait(driver, 20)
        table = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "table.table-striped.table-sm.table-hover.mb-0"))
        )
        
        rows = table.find_elements(By.TAG_NAME, "tr")
        player_data = {}
        
        for row in rows[1:]:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) > 1:
                    data = parse_player_row(cells)
                    if data:
                        player_data[data[0]] = data[1:]
                        logging.info(f"Fetched data for {data[0]} from {country[0]}")
            except Exception as e:
                logging.warning(f"Error parsing row for {country[0]}: {e}")
                continue
                
        if player_data:
            players[country[0]] = player_data
        else:
            logging.error(f"No player data found for {country[0]}")
            
    except Exception as e:
        logging.error(f"Error fetching data for {country[0]}: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass

def parse_player_row(cells):
    """Helper function to parse player data from table cells."""
    try:
        name = cells[1].find_element(By.CLASS_NAME, "entry-font.entry-font-narrow").text.strip()
        position = cells[1].find_elements(By.TAG_NAME, "a")[2].get_attribute("title").strip()
        rating = float(cells[4].find_element(By.TAG_NAME, "span").text.replace(",", "").strip())
        
        stats = []
        for i in [2, 3, 5, 6]:
            try:
                stat = int(cells[i].find_element(By.TAG_NAME, "span").text.strip())
            except (ValueError, IndexError):
                stat = 0
            stats.append(stat)
            
        return [name, position, rating] + stats
    except Exception:
        return None

if __name__ == "__main__":
    # Add webdriver-manager to required packages
    REQUIRED_PACKAGES.append('webdriver-manager')
    
    if not "confederations.json" in os.listdir():
        confederations = {x: []
                          for x in ["CONMEBOL", "CONCACAF", "UEFA", "CAF", "AFC", "OFC"]}

        for c, l, n in zip(["CONMEBOL", "CONCACAF", "UEFA", "CAF", "AFC", "OFC"], ["https://en.wikipedia.org/wiki/CONMEBOL", "https://en.wikipedia.org/wiki/CONCACAF", "https://en.wikipedia.org/wiki/UEFA", "https://en.wikipedia.org/wiki/Confederation_of_African_Football", "https://en.wikipedia.org/wiki/Asian_Football_Confederation", "https://en.wikipedia.org/wiki/Oceania_Football_Confederation"], [7, 8, 16, 10, 9, 2]):
            confederations = get_teams(c, l, n, confederations)

        with open("confederations.json", "w") as f:
            json.dump(confederations, f)
    else:
        with open("confederations.json", "r") as f:
            confederations = json.load(f)

    countries = []
    for conf in confederations:
        for key in confederations[conf]:
            countries.append(key)

    if not "players.json" in os.listdir():
        players = {x[0]: {} for x in countries}
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(fetch_player_data, countries)
        with open("players.json", "w") as f:
            json.dump(players, f)
    else:
        with open("players.json", "r") as f:
            players = json.load(f)

    # Retry fetching data for countries with missing data
    for country in countries:
        if not players[country[0]]:
            fetch_player_data(country)

    with open("players.json", "w") as f:
        json.dump(players, f)

    formations = {"4-4-2": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Defensive Midfielder', 'Left Wing', 'Right Wing', 'Center Forward', 'Striker'],
                  "4-3-3": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Midfielder', 'Center Midfielder', 'Left Wing', 'Center Forward', 'Right Wing'],
                  "4-2-3-1": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Defensive Midfielder', 'Center Attacking Midfielder', 'Center Midfielder', 'Center Midfielder', 'Center Forward'],
                  "3-5-2": ['Goalkeeper', 'Center Back', 'Center Back', 'Center Back', 'Left Midfielder', 'Center Midfielder', 'Right Midfielder', 'Left Back', 'Right Back', 'Center Forward', 'Striker'],
                  "5-4-1": ['Goalkeeper', 'Center Back', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Midfielder', 'Center Midfielder', 'Center Midfielder', 'Center Midfielder', 'Center Attacking Midfielder', 'Striker'],
                  "4-4-2b": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Defensive Midfielder', 'Right Wing Back', 'Left Wing Back', 'Center Forward', 'Striker'],
                  "4-3-3b": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Midfielder', 'Center Midfielder', 'Right Midfielder', 'Center Forward', 'Left Midfielder'],
                  "4-2-3-1b": ['Goalkeeper', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Defensive Midfielder', 'Center Defensive Midfielder', 'Left Wing', 'Center Midfielder', 'Right Wing', 'Center Forward'],
                  "3-5-2b": ['Goalkeeper', 'Center Back', 'Center Back', 'Center Back', 'Center Midfielder', 'Center Midfielder', 'Center Midfielder', 'Left Wing Back', 'Right Wing Back', 'Striker', 'Center Forward'],
                  "5-4-1b": ['Goalkeeper', 'Center Back', 'Center Back', 'Center Back', 'Left Back', 'Right Back', 'Center Midfielder', 'Center Midfielder', 'Center Midfielder', 'Right Midfielder', 'Left Midfielder', 'Center Forward']}

    if not "formations.json" in os.listdir():
        # best possible formation for each squad + overall rating
        formations_dict = {}

        for country in countries:
            results = [best_formation(players[country[0]], formations[formation])
                       for formation in formations]
            results = [x for x in results if x]
            if len(results) > 0:
                formations_dict[country[0]] = sorted(
                    results, key=lambda x: x[1])[-1]
            else:
                formations_dict[country[0]] = False

        with open("formations.json", "w") as f:
            json.dump(formations_dict, f)
    else:
        with open("formations.json", "r") as f:
            formations_dict = json.load(f)

    for conf in confederations:
        for country in confederations[conf]:
            country[1] = (country[1], formations_dict[country[0]]
                          [1] if formations_dict[country[0]] else False)

    final_results = {}

    for i in range(1):

        confederations_copy = copy.deepcopy(confederations)

        print(f"Iteration number {i}    ", end="\r")

        confederations_copy = final_qual_round(confederations_copy, 2)
        confederations_copy = {key: [x+[key]
                                     for x in confederations_copy[key]] for key in confederations_copy}

        hosts = [x for x in confederations_copy["CONCACAF"]
                 if x[0] in ["Mexico", "United States", "Canada"]]

        pots = create_pots(hosts, confederations_copy)

        groups = {x: {} for x in "ABCDEFGHIJKL"}

        groups = seed(groups, pots)

        groups = schedule_matches(groups)

        groups = group_stage(groups)

        knockout_brackets = knockout_round(groups)

        display_fixture(groups, knockout_brackets)

        first = knockout_brackets["finals"][0][0][0][0]
        second = [x for x in knockout_brackets["finals"]
                  [0][1] if not x[0] == first][0][0]
        third = knockout_brackets["third place"][0][0][0][0]
        fourth = [x for x in knockout_brackets["third place"]
                  [0][1] if not x[0] == third][0][0]

        for r, v in zip([first, second, third, fourth], [1, 0.75, 0.5, 0.25]):
            if not r in final_results:
                final_results[r] = v
            else:
                final_results[r] += v

    data = dict(sorted(final_results.items(),
                key=lambda x: x[1], reverse=True)[:20])

    df = pd.DataFrame(
        {"country": [x for x in data], "score": [x for x in data.values()]})

    plt.bar(df["country"], df["score"])
    plt.xticks(rotation=45)
    plt.show()
