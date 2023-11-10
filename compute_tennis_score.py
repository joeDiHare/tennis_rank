import re
import sys
import requests
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from bs4 import BeautifulSoup
from argparse import ArgumentParser


URL_BASE = "https://app.tennisrungs.com"


class ComputeElo:
	'''
	Class to implement the Elo score_elo. Formulas & parameters are taken from:
	https://en.wikipedia.org/wiki/Elo_score_elo_system
	'''
	def __init__(self, k=20, g=5):
		self.elo = {}
		self.usta_scores = {}
		self.elo_history = []
		self.players = {}
		self.k = k  # this should be >20 when we have small number of matches
		self.g = g  # dunno about this

	def add_players(self, names, score_elo=1500):
		for name in names:
			self.players[name] = Player(name=name, score_elo=score_elo)
			self.elo[name] = score_elo

	def expected_outcome(self, p1, p2, a=400.0, b=10.0):
		return 1 / (b**((p2 - p1) / a) + 1)

	def log_result(self, winner, loser, min_=100):
		exp_result = self.expected_outcome(self.elo[winner], self.elo[loser])
		kg = self.k * self.g
		self.elo[winner] = max(min_, self.elo[winner] + kg * (1 - exp_result))
		self.elo[loser]  = max(min_, self.elo[loser]  + kg * (exp_result - 1))
		self.elo_history.append(list(self.elo.values()))

		self.update_player_stats(winner, result='W')
		self.update_player_stats(loser,  result='L')

	def update_player_stats(self, name, result):
		self.players[name].score_elo = self.elo[name]
		self.players[name].results += result

	def rankings(self, score='elo'):
		scores = self.usta_scores if score=='usta' else self.elo
		return sorted(scores.items(), key=lambda item: -item[1])

	def top(self, N=None, verbose=False, score='elo'):
		rankings = self.rankings(score=score)[:N or -1]
		if verbose:
			for n, (a, b) in enumerate(rankings, 1):
				print(f"#{n} {b:.0f} {a}")
		return rankings

	def generate_history(self):
		df = pd.DataFrame(self.elo_history)
		df.columns = self.elo.keys()
		return df
	
	def compute_usta_scores(self, low_cutoff, hig_cutoff, pc=0.95, map_to=0.5, compress=True):

		score_usta = np.array(np.array(list(self.elo.values())), copy=True)
		score_usta /= score_usta.std()
		score_usta -= score_usta.mean()
		score_usta += low_cutoff + (hig_cutoff - low_cutoff) / 2.0

		cutoff_up = norm.ppf(pc, loc=score_usta.mean(), scale=score_usta.std())
		scale = cutoff_up - score_usta.mean()

		mu = score_usta.mean()
		score_usta -= mu
		score_usta /= scale
		score_usta += mu

		if compress:
			x_ = score_usta[score_usta>hig_cutoff]
			min_ = x_.min()
			x_ = np.log10(x_ / min_)
			x_ = (x_ / x_.max() * map_to) + min_
			score_usta[score_usta>hig_cutoff] = x_

			x_ = score_usta[score_usta<low_cutoff]
			max_ = x_.max()
			x_ = 10**(x_ / max_ - 1)  #np.log10(vals / v_m)
			x_ -= x_.min()
			x_ = (low_cutoff-map_to) + x_ / x_.max() * map_to
			score_usta[score_usta<low_cutoff] = x_
		
		for k, name in enumerate(self.elo.keys()):
			self.players[name].score_usta = score_usta[k]
			self.usta_scores[name] = score_usta[k]


class Player:
	'''Class for the Player.'''
	def __init__(self, name, score_elo):
		self.name = name
		self.score_elo = score_elo
		self.score_usta = None
		self.results = ''
		self.points = None

	@property
	def last_result(self):
		return self.results[-1]

def parse_table(table):
    '''Parse table from scraped html.'''
    data = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.strip())
            if cell.a is not None:
                row_data.append(cell.a['href'])
        data.append(row_data)
    return data[1:]

def calculate_result_stats(s, add_tb=False, option=1):
    '''Extract winning stats.'''
    for lbl in ['Forfeit', 'NoShow', 'CourtTimeExpired']:
        s = s.replace(lbl, '')
    rex = r'([0-9]{1,2}\-[0-9]{1,2})?\s?(\(([0-9]{1,2}\-[0-9]{1,2})\))?'
    num, den = 0, 0
    for g in s.split(','):
        res, _, tb = re.match(rex, g.strip()).groups()
        if res is not None:
            a, b = res.split('-')
            num += int(a)
            den += int(a) + int(b)
            # Ignore TB score, since the game has already been recorded
        else:
            # Super TB (no res)
            if not add_tb:
                continue
            if option == 1:
                # Option 1: Roughly equate 1 TB won to 2.5 games
                num += 2.5
                den += 2.5
            if option == 2:
                # Option 2: Roughly equate 5 points to 1 game won
                a, b = res.split('-')
                num += int(a) // 5
                den += int(a) // 5 + int(b) // 5
    return num/den if den>0 else 0

def cleanup_result_logs(df, df_player):
    df.columns = ['player_a_id', 'player_a',  'date', 'player_b', 'link', 'res', 'score']
    # Add players that dropped fromt the challenge:
    df['player_b_id'] = df['link'].apply(lambda x: x.split('teamId=')[-1])
    for _, row in df.iterrows():
        if row['player_b_id'] not in df_player['player_id'].values:
            logger.debug(f'Adding dropped-player "{row["player_b"]}" to list.')
            df_player.loc[len(df_player.index)] = [row['player_b'], row['link'], None, row['player_b_id']] 
    df['player_b'] = df['player_b_id'].apply(lambda x: df_player[df_player['player_id']==x].iloc[0]['name'])
    # Add useful fields:
    df['winner'] = df['res'].apply(lambda x: 'a' if x=='W' else 'b')
    df['loser'] = df['res'].apply(lambda x: 'b' if x=='W' else 'a')
    df['winner_name'] = df.apply(lambda x: x[f'player_{x["winner"]}'], axis=1)
    # Create winning stats, maybe useful in the future:
    df['ratio_win'] = df['score'].apply(lambda x: calculate_result_stats(x))
    df['ratio_win'] = df.apply(lambda x: x['ratio_win'] if x['res']=='W' else 1-x['ratio_win'], axis=1)
    df['ratio_win_tb'] = df['score'].apply(lambda x: calculate_result_stats(x, add_tb=True))
    df['ratio_win_tb'] = df.apply(lambda x: x['ratio_win_tb'] if x['res']=='W' else 1-x['ratio_win_tb'], axis=1)
    # Unique identifier for a match:
    df['match_uid'] = df.apply(lambda d: f"{d['date'].replace('/', '')}_{'_'.join(sorted(d[['player_a_id', 'player_b_id']]))}", axis=1)
    # Sort chronologically:
    df['date'] = pd.to_datetime(df['date'])
    df['ts'] = df['date'].apply(lambda x: pd.Timestamp(x).timestamp())
    df['week_nr'] = df['date'].dt.isocalendar().week
    df = df.sort_values('ts').reset_index(drop=True)
    df = df[['date', 'player_a', 'player_a_id', 'player_b', 'player_b_id', 'winner', 'loser', 'score',
            'ratio_win', 'ratio_win_tb', 'match_uid', 'winner_name', 'ts', 'week_nr']]
    logger.debug(f'{len(df)} total records (counting duplicates)')
    # De-duplicate logs:
    df = df[~df.duplicated(subset=['match_uid'], keep='first')].reset_index(drop=True)
    logger.debug(f'{len(df)} after de-duplication')
    return df


def crawl_players_page(args):
    url = f"{URL_BASE}/atc/tennis-ladders/{args.league_id}"
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    table = soup.find('table', {'id': 'ladderrankings'})
    data = parse_table(table)
    df_player = pd.DataFrame([(x[1], x[2], x[6]) for x in data])
    df_player.columns = ['name', 'link', 'points']
    df_player['player_id'] = df_player['link'].apply(lambda x: x.split('teamId=')[-1])
    return df_player


def crawl_results_pages(df_player):
    results = []
    for idx, row in df_player.iterrows():
        req = requests.get(f"{URL_BASE}{row['link']}")
        soup = BeautifulSoup(req.text, "html.parser")
        table = soup.find('table', {'id': 'challenges'})
        if table is not None:
            res_ = parse_table(table)
            res_ = [[row['player_id'], row['name']] + r for r in res_]
            results.append(res_)
    return pd.DataFrame([y for x in results for y in x])


def main(args):
    """ Main entry point of the app """
    logger.info(f"Crawling data for {args.league_id} and computing ELO score for all players.")
    
    # Crawl players
    logger.info("[1/4] Crawling list of players ...")
    df_player = crawl_players_page(args)

    # Create history by crawling all pages and logging results
    logger.info("[2/4] Creating log with all matches played ...")
    df_results = crawl_results_pages(df_player)
    df_results = cleanup_result_logs(df_results, df_player)
    logger.info(f'Found {len(df_player)} players with {len(df_results)} matched played.')

    # Compute ELO score
    logger.info("[3/4] Calculate ELO score for all players ...")
    CE = ComputeElo()
    CE.add_players(df_player['name'], score_elo=1500)
    # Iterate through all records and update elo's
    for idx, row in df_results.sort_values('ts').iterrows():
        CE.log_result(winner=row[f'player_{row["winner"]}'],
                      loser=row[f'player_{row["loser"]}'])
    
    # Compute USTA scores from elo's
    logger.info("[4/4] Estimate USTA score for all players ...")
    rex = r'([1-5]\.[0,5])\-([1-5]\.[0,5])'
    low_cutoff, hig_cutoff = map(float, re.findall(rex, args.league_id)[0])
    CE.compute_usta_scores(low_cutoff, hig_cutoff, pc=0.95, map_to=0.5, compress=True)

    # Save results
    if args.save:
        logger.info("Saving results to data/")
        safe_filename = "".join([x if x.isalnum() else "_" for x in args.league_id])
        logger.debug("Saving match results dataframe")
        df_results.to_csv(f'data/results_{safe_filename}.csv')
        logger.debug("Saving players dataframe")
        df_player.to_csv(f'data/players_{safe_filename}.csv')
        logger.debug("Saving score class")
        np.save(f'data/scores_{safe_filename}.npy', CE)

    # Print ranks and scores
    rankings = CE.top(verbose=False, score='elo')
    logger.info(f'Listing all {len(df_player)} players and their ELO/USTA scores:')
    logger.info(" #\tELO\tUSTA\tname")
    for n, (name, elo) in enumerate(rankings, 1):
        logger.info(f"({n})\t{elo:.0f}\t{CE.usta_scores[name]:.2f}\t{name}")
    logger.debug("All Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--league_id", default="singles-3.0-3.5/143889747",
                        help="Required positional argument")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, colorize=True,
               format='<green>{time:HH:mm:ss}</green> | <level>{level: <8} | {message}</level>',
               level="DEBUG" if args.verbose else "INFO")
    main(args)